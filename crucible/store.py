"""SQLite read/write via SQLAlchemy. All side effects are isolated here."""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import (
    Boolean,
    Column,
    Engine,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    Table,
    Text,
    create_engine,
    insert,
    select,
)
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_metadata = MetaData()

_scans = Table(
    "scans",
    _metadata,
    Column("id", Integer, primary_key=True),
    Column("run_ts", Text, nullable=False, unique=True),
    Column("universe_id", Text, nullable=False),
    Column("created_at", Text, nullable=False),
    Column("n_processed", Integer, nullable=False),
    Column("n_passed_filters", Integer, nullable=False),
)

# Columns that map directly from the processed DataFrame
_COMPANY_COLS: list[str] = [
    "sector", "sub_industry", "currency",
    "p_e", "p_fcf", "ev_ebitda",
    "data_years", "insufficient_data",
    "roic_proxy_avg", "fcf_latest", "fcf_positive_years",
    "net_debt_ebitda", "revenue_growth_positive_years",
    "gross_margin_latest", "gross_margin_avg", "gross_margin_trend_slope",
]

_companies = Table(
    "companies",
    _metadata,
    Column("id", Integer, primary_key=True),
    Column("scan_id", Integer, ForeignKey("scans.id"), nullable=False),
    Column("ticker", Text, nullable=False),
    Column("sector", Text),
    Column("sub_industry", Text),
    Column("currency", Text),
    Column("data_years", Integer, nullable=False),
    Column("insufficient_data", Boolean, nullable=False),
    Column("roic_proxy_avg", Float),
    Column("fcf_latest", Float),
    Column("fcf_positive_years", Float),
    Column("net_debt_ebitda", Float),
    Column("revenue_growth_positive_years", Float),
    Column("gross_margin_latest", Float),
    Column("gross_margin_avg", Float),
    Column("gross_margin_trend_slope", Float),
    Column("p_e", Float),
    Column("p_fcf", Float),
    Column("ev_ebitda", Float),
)

# Score columns added by scorer.py
_SCORE_COLS: list[str] = [
    "quality_score", "valuation_score", "fx_penalty", "composite_score",
]

_scores = Table(
    "scores",
    _metadata,
    Column("id", Integer, primary_key=True),
    Column("scan_id", Integer, ForeignKey("scans.id"), nullable=False),
    Column("ticker", Text, nullable=False),
    # Full-funnel flag: True only for tickers that passed all Layer 1 filters
    Column("passed_filters", Boolean, nullable=False),
    Column("quality_score", Float),
    Column("valuation_score", Float),
    Column("fx_penalty", Float),
    Column("composite_score", Float),
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_engine(db_path: str | Path) -> Engine:
    """Create a SQLAlchemy engine for the given SQLite path."""
    return create_engine(f"sqlite:///{db_path}")


def create_tables(engine: Engine) -> None:
    """Create all tables if they do not exist."""
    _metadata.create_all(engine)


def save_scan(
    engine: Engine,
    processed_df: pd.DataFrame,
    shortlist_df: pd.DataFrame,
    universe_id: str,
    run_ts: str,
) -> int:
    """Save a complete scan to the database; return the new scan_id.

    processed_df — all tickers output by clean() (index = ticker)
    shortlist_df — tickers that passed filters and were scored (index = ticker)

    Every ticker in processed_df gets a row in companies and scores.
    passed_filters=True only for tickers present in shortlist_df.
    Score columns are NULL for excluded tickers.
    """
    with Session(engine) as session:
        result = session.execute(
            insert(_scans).values(
                run_ts=run_ts,
                universe_id=universe_id,
                created_at=datetime.now(timezone.utc).isoformat(),
                n_processed=len(processed_df),
                n_passed_filters=len(shortlist_df),
            )
        )
        scan_id: int = result.inserted_primary_key[0]

        company_rows = _build_company_rows(processed_df, scan_id)
        if company_rows:
            session.execute(insert(_companies), company_rows)

        score_rows = _build_score_rows(processed_df, shortlist_df, scan_id)
        if score_rows:
            session.execute(insert(_scores), score_rows)

        session.commit()

    logger.info(
        "Saved scan_id=%d  run_ts=%s  processed=%d  shortlisted=%d",
        scan_id, run_ts, len(processed_df), len(shortlist_df),
    )
    return scan_id


def load_shortlist(engine: Engine, scan_id: int | None = None) -> pd.DataFrame:
    """Load the shortlist (passed_filters=True) with full fundamentals and scores.

    Defaults to the most recent scan. Returns an empty DataFrame if no scan exists.
    """
    scan_id = _resolve_scan_id(engine, scan_id)
    if scan_id is None:
        return pd.DataFrame()

    stmt = (
        _join_stmt()
        .where(_companies.c.scan_id == scan_id)
        .where(_scores.c.passed_filters == True)  # noqa: E712
        .order_by(_scores.c.composite_score.desc())
    )
    df = pd.read_sql(stmt, con=engine)
    return df.set_index("ticker") if not df.empty else df


def load_all_for_scan(engine: Engine, scan_id: int | None = None) -> pd.DataFrame:
    """Load ALL processed tickers for a scan, including excluded ones.

    Useful for debugging filter eliminations and Phase 2 backtest preparation.
    passed_filters column indicates whether each ticker reached the shortlist.
    """
    scan_id = _resolve_scan_id(engine, scan_id)
    if scan_id is None:
        return pd.DataFrame()

    stmt = (
        _join_stmt()
        .where(_companies.c.scan_id == scan_id)
        .order_by(_scores.c.composite_score.desc())
    )
    df = pd.read_sql(stmt, con=engine)
    return df.set_index("ticker") if not df.empty else df


def list_scans(engine: Engine) -> pd.DataFrame:
    """Return scan metadata for all runs, newest first."""
    return pd.read_sql(
        select(_scans).order_by(_scans.c.id.desc()), con=engine
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _join_stmt():
    """SELECT statement joining companies + scores on (scan_id, ticker)."""
    return select(
        _companies.c.ticker,
        _companies.c.sector,
        _companies.c.sub_industry,
        _companies.c.currency,
        _companies.c.data_years,
        _companies.c.insufficient_data,
        _companies.c.roic_proxy_avg,
        _companies.c.fcf_latest,
        _companies.c.fcf_positive_years,
        _companies.c.net_debt_ebitda,
        _companies.c.revenue_growth_positive_years,
        _companies.c.gross_margin_latest,
        _companies.c.gross_margin_avg,
        _companies.c.gross_margin_trend_slope,
        _companies.c.p_e,
        _companies.c.p_fcf,
        _companies.c.ev_ebitda,
        _scores.c.passed_filters,
        _scores.c.quality_score,
        _scores.c.valuation_score,
        _scores.c.fx_penalty,
        _scores.c.composite_score,
    ).select_from(
        _companies.join(
            _scores,
            (_companies.c.scan_id == _scores.c.scan_id)
            & (_companies.c.ticker == _scores.c.ticker),
        )
    )


def _resolve_scan_id(engine: Engine, scan_id: int | None) -> int | None:
    """Return scan_id as-is, or the latest scan_id, or None if no scans exist."""
    if scan_id is not None:
        return scan_id
    with engine.connect() as conn:
        row = conn.execute(
            select(_scans.c.id).order_by(_scans.c.id.desc()).limit(1)
        ).fetchone()
    return int(row[0]) if row else None


def _build_company_rows(df: pd.DataFrame, scan_id: int) -> list[dict]:
    rows = []
    for ticker, row in df.iterrows():
        rec: dict = {"scan_id": scan_id, "ticker": ticker}
        for col in _COMPANY_COLS:
            rec[col] = _to_sql(row.get(col))
        rows.append(rec)
    return rows


def _build_score_rows(
    processed_df: pd.DataFrame,
    shortlist_df: pd.DataFrame,
    scan_id: int,
) -> list[dict]:
    scored = set(shortlist_df.index)
    rows = []
    for ticker in processed_df.index:
        if ticker in scored:
            row = shortlist_df.loc[ticker]
            rows.append({
                "scan_id": scan_id,
                "ticker": ticker,
                "passed_filters": True,
                **{col: _to_sql(row.get(col)) for col in _SCORE_COLS},
            })
        else:
            rows.append({
                "scan_id": scan_id,
                "ticker": ticker,
                "passed_filters": False,
                **{col: None for col in _SCORE_COLS},
            })
    return rows


def _to_sql(value: object) -> object:
    """Convert NaN/NaT to None; convert numpy scalars to Python native types."""
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):  # numpy scalar → Python scalar
        return value.item()
    return value


# ---------------------------------------------------------------------------
# Prospective logging — monthly_picks table (sqlite3, wide schema)
# ---------------------------------------------------------------------------

_PICKS_FLOAT_COLS: list[str] = [
    "composite_score",
    "quality_score",
    "growth_quality_score",
    "momentum_score",
    "valuation_score",
    "value_score",
    "recovery_signal_score",
    "balance_sheet_score",
    "roic_proxy_avg",
    "fcf_positive_years",
    "fcf_positive_years_last5",
    "fcf_positive_last2yr",
    "fcf_trajectory",
    "net_debt_ebitda",
    "interest_coverage",
    "revenue_growth_yr1",
    "revenue_growth_yr2",
    "revenue_acceleration",
    "revenue_growth_positive_years",
    "gross_margin_latest",
    "gross_margin_yr1_change",
    "gross_margin_trend_slope",
    "momentum_raw",
    "momentum_3m",
    "p_fcf",
    "p_s",
    "ev_ebitda",
    "p_e",
    "share_buyback_signal",
    "p_fcf_vs_history",
]

_CREATE_MONTHLY_PICKS = """
CREATE TABLE IF NOT EXISTS monthly_picks (
    run_date   TEXT NOT NULL,
    track      INTEGER NOT NULL,
    ticker     TEXT NOT NULL,
    universe   TEXT NOT NULL,
    recommendation TEXT NOT NULL,
    sector     TEXT,
    composite_score           REAL,
    quality_score             REAL,
    growth_quality_score      REAL,
    momentum_score            REAL,
    valuation_score           REAL,
    value_score               REAL,
    recovery_signal_score     REAL,
    balance_sheet_score       REAL,
    roic_proxy_avg            REAL,
    fcf_positive_years        REAL,
    fcf_positive_years_last5  REAL,
    fcf_positive_last2yr      REAL,
    fcf_trajectory            REAL,
    net_debt_ebitda           REAL,
    interest_coverage         REAL,
    revenue_growth_yr1        REAL,
    revenue_growth_yr2        REAL,
    revenue_acceleration      REAL,
    revenue_growth_positive_years REAL,
    gross_margin_latest       REAL,
    gross_margin_yr1_change   REAL,
    gross_margin_trend_slope  REAL,
    momentum_raw              REAL,
    momentum_3m               REAL,
    p_fcf                     REAL,
    p_s                       REAL,
    ev_ebitda                 REAL,
    p_e                       REAL,
    share_buyback_signal      REAL,
    p_fcf_vs_history          REAL,
    PRIMARY KEY (run_date, track, ticker)
)
"""

_REC_MAP: dict[str, str] = {
    "REINFORCE":    "reinforce",
    "HOLD":         "hold",
    "REVIEW":       "review",
    "EXIT_SIGNAL":  "exit_signal",
    "DATA_MISSING": None,  # type: ignore[dict-item]  # skipped
}


def log_monthly_picks(
    db_path: str | Path,
    run_date: str,
    track: int,
    universe: str,
    result: pd.DataFrame,
    portfolio_recs: pd.DataFrame | None = None,
) -> None:
    """Upsert screener results and portfolio recommendations into monthly_picks.

    result         — screener shortlist (index=ticker); gets recommendation="new_pick"
    portfolio_recs — evaluate_portfolio() output (index=ticker); overrides new_pick
                     for any shared ticker; DATA_MISSING rows are skipped
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    rows: dict[str, dict] = {}

    def _base_row(ticker: str, rec: str, source: pd.Series | None) -> dict:
        row: dict = {
            "run_date":      run_date,
            "track":         track,
            "ticker":        ticker,
            "universe":      universe,
            "recommendation": rec,
            "sector":        None,
        }
        if source is not None:
            row["sector"] = _to_sql(source.get("sector"))
            for col in _PICKS_FLOAT_COLS:
                row[col] = _to_sql(source.get(col))
        else:
            for col in _PICKS_FLOAT_COLS:
                row[col] = None
        return row

    # Screener shortlist → new_pick
    for ticker, series in result.iterrows():
        rows[ticker] = _base_row(ticker, "new_pick", series)

    # Portfolio positions override (or add if not in screener result)
    if portfolio_recs is not None and not portfolio_recs.empty:
        for ticker, series in portfolio_recs.iterrows():
            raw_rec = str(series.get("recommendation", ""))
            mapped = _REC_MAP.get(raw_rec)
            if mapped is None:
                continue  # DATA_MISSING — skip
            # Use screener data for metrics if available, else no metrics
            source = result.loc[ticker] if ticker in result.index else None
            rows[ticker] = _base_row(ticker, mapped, source)

    if not rows:
        logger.info("log_monthly_picks: nothing to log for run_date=%s track=%d", run_date, track)
        return

    cols = ["run_date", "track", "ticker", "universe", "recommendation", "sector"] + _PICKS_FLOAT_COLS
    placeholders = ", ".join("?" for _ in cols)
    sql = f"INSERT OR REPLACE INTO monthly_picks ({', '.join(cols)}) VALUES ({placeholders})"

    with sqlite3.connect(db_path) as conn:
        conn.execute(_CREATE_MONTHLY_PICKS)
        conn.executemany(sql, [tuple(r[c] for c in cols) for r in rows.values()])
        conn.commit()

    logger.info(
        "log_monthly_picks: %d rows → %s  (run_date=%s track=%d)",
        len(rows), db_path, run_date, track,
    )
