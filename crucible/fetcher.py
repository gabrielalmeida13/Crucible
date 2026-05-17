"""Data extraction from Financial Modeling Prep (FMP) API with SQLite caching.

Rate limit: FMP free tier allows 250 requests/day. All HTTP responses are
cached in data/fmp_cache.db. Tickers already cached within their TTL are
never re-fetched, so a full SP500 run completes across multiple days without
double-counting against the quota.

NOTE: yfinance must never be used for backtesting (look-ahead bias).
FMP provides point-in-time financial statements required for Phase 2.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_FMP_BASE = "https://financialmodelingprep.com/api/v3"
_DAILY_LIMIT = 250

# Cache TTLs in seconds
_TTL_SP500 = 86_400          # 1 day   — constituent list can shift
_TTL_PROFILE = 7 * 86_400   # 7 days  — contains live P/E; staleness is acceptable
_TTL_STATEMENTS = 30 * 86_400  # 30 days — historical filings never change

# FMP field names → metric names expected by cleaner.py
_INCOME_MAP: dict[str, str] = {
    "revenue": "Total Revenue",
    "grossProfit": "Gross Profit",
    "netIncome": "Net Income",
    "ebitda": "EBITDA",
}
_BALANCE_MAP: dict[str, str] = {
    "totalAssets": "Total Assets",
    "totalCurrentLiabilities": "Current Liabilities",
    "totalDebt": "Total Debt",
    "cashAndCashEquivalents": "Cash And Cash Equivalents",
}
_CASHFLOW_MAP: dict[str, str] = {
    "operatingCashFlow": "Operating Cash Flow",
    "capitalExpenditure": "Capital Expenditure",
    "freeCashFlow": "Free Cash Flow",
}

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class _FMPCache:
    """SQLite-backed response cache + atomic daily request counter.

    Pass ``":memory:"`` as *db_path* to get an in-memory instance (tests only).
    """

    def __init__(self, db_path: str | Path, daily_limit: int = _DAILY_LIMIT) -> None:
        self._path = str(db_path)
        self._daily_limit = daily_limit
        self._mem_conn: sqlite3.Connection | None = None
        if self._path == ":memory:":
            self._mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
        else:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _db(self) -> Generator[sqlite3.Connection, None, None]:
        if self._mem_conn is not None:
            yield self._mem_conn
        else:
            conn = sqlite3.connect(self._path)
            try:
                yield conn
            finally:
                conn.close()

    def _init_db(self) -> None:
        with self._db() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS api_cache (
                    cache_key TEXT PRIMARY KEY,
                    data_json TEXT NOT NULL,
                    fetched_at INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS request_log (
                    date TEXT PRIMARY KEY,
                    count INTEGER NOT NULL DEFAULT 0
                );
                """
            )

    def get(self, cache_key: str, ttl: int) -> list | dict | None:
        """Return cached data if present and within TTL, else None."""
        with self._db() as conn:
            row = conn.execute(
                "SELECT data_json, fetched_at FROM api_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if not row:
            return None
        if time.time() - row[1] > ttl:
            return None
        return json.loads(row[0])

    def set(self, cache_key: str, data: list | dict) -> None:
        """Upsert a response into the cache."""
        with self._db() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO api_cache(cache_key, data_json, fetched_at) "
                "VALUES(?, ?, ?)",
                (cache_key, json.dumps(data), int(time.time())),
            )
            conn.commit()

    def requests_today(self) -> int:
        """Return the number of real HTTP requests made today (UTC)."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._db() as conn:
            row = conn.execute(
                "SELECT count FROM request_log WHERE date = ?", (today,)
            ).fetchone()
        return row[0] if row else 0

    def check_and_increment(self) -> bool:
        """Atomically check limit and increment counter.

        Returns True if the request is allowed, False if the daily limit is reached.
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._db() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO request_log(date, count) VALUES(?, 0)", (today,)
            )
            row = conn.execute(
                "SELECT count FROM request_log WHERE date = ?", (today,)
            ).fetchone()
            if row and row[0] >= self._daily_limit:
                return False
            conn.execute(
                "UPDATE request_log SET count = count + 1 WHERE date = ?", (today,)
            )
            conn.commit()
        return True


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------


def _fmp_get(
    session: requests.Session,
    path: str,
    api_key: str,
    params: dict | None = None,
) -> list | dict:
    """Execute one FMP API request. Raises on HTTP or application-level errors."""
    url = f"{_FMP_BASE}{path}"
    resp = session.get(url, params={"apikey": api_key, **(params or {})}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict) and "Error Message" in data:
        raise ValueError(f"FMP error for {path!r}: {data['Error Message']}")
    return data


def _cached_get(
    session: requests.Session,
    cache: _FMPCache,
    cache_key: str,
    ttl: int,
    path: str,
    api_key: str,
    params: dict | None = None,
) -> list | dict | None:
    """Serve from cache when valid; otherwise fetch, cache, and return.

    Returns None when the daily rate limit is already exhausted.
    """
    cached = cache.get(cache_key, ttl)
    if cached is not None:
        return cached
    if not cache.check_and_increment():
        logger.warning("Daily FMP limit reached (%d/day). Skipping %s.", _DAILY_LIMIT, cache_key)
        return None
    data = _fmp_get(session, path, api_key, params)
    cache.set(cache_key, data)
    logger.debug("FMP cached: %s", cache_key)
    return data


# ---------------------------------------------------------------------------
# Per-ticker data helpers
# ---------------------------------------------------------------------------


def _fetch_ticker_raw(
    ticker: str,
    api_key: str,
    cache: _FMPCache,
    session: requests.Session,
) -> dict[str, list]:
    """Fetch all four FMP endpoints for one ticker; cache each independently."""
    return {
        "profile": _cached_get(
            session, cache, f"profile:{ticker}", _TTL_PROFILE,
            f"/profile/{ticker}", api_key,
        ) or [],
        "income": _cached_get(
            session, cache, f"income:{ticker}", _TTL_STATEMENTS,
            f"/income-statement/{ticker}", api_key,
            {"period": "annual", "limit": 5},
        ) or [],
        "balance": _cached_get(
            session, cache, f"balance:{ticker}", _TTL_STATEMENTS,
            f"/balance-sheet-statement/{ticker}", api_key,
            {"period": "annual", "limit": 5},
        ) or [],
        "cashflow": _cached_get(
            session, cache, f"cashflow:{ticker}", _TTL_STATEMENTS,
            f"/cash-flow-statement/{ticker}", api_key,
            {"period": "annual", "limit": 5},
        ) or [],
    }


def _raw_to_info_row(ticker: str, raw: dict[str, list]) -> dict:
    """Derive the info_df row from raw FMP data for one ticker.

    P/E comes from the live profile. P/FCF and EV/EBITDA are computed from the
    most recent annual filing + market cap, matching the cleaner's expectations.
    """
    profile = raw["profile"]
    p = profile[0] if profile else {}
    latest_inc = raw["income"][0] if raw["income"] else {}
    latest_bal = raw["balance"][0] if raw["balance"] else {}
    latest_cf = raw["cashflow"][0] if raw["cashflow"] else {}

    row: dict = {
        "ticker": ticker,
        "sector": p.get("sector") or None,
        "sub_industry": p.get("industry") or None,
        "currency": p.get("currency") or None,
        "p_e": _to_float(p.get("pe")),
    }

    mkt_cap = _to_float(p.get("mktCap"))
    fcf = _to_float(latest_cf.get("freeCashFlow"))
    row["p_fcf"] = (mkt_cap / fcf) if (mkt_cap and fcf and fcf > 0) else None

    debt = _to_float(latest_bal.get("totalDebt")) or 0.0
    cash = _to_float(latest_bal.get("cashAndCashEquivalents")) or 0.0
    ebitda = _to_float(latest_inc.get("ebitda"))
    if mkt_cap is not None and ebitda and ebitda > 0:
        row["ev_ebitda"] = (mkt_cap + debt - cash) / ebitda
    else:
        row["ev_ebitda"] = None

    return row


def _raw_to_panel_rows(ticker: str, raw: dict[str, list]) -> list[dict]:
    """Convert raw FMP annual statements to long-panel rows for the processed DataFrame."""
    rows: list[dict] = []
    for stmt_key, field_map, stmt_name in (
        ("income", _INCOME_MAP, "income"),
        ("balance", _BALANCE_MAP, "balance"),
        ("cashflow", _CASHFLOW_MAP, "cashflow"),
    ):
        for annual in raw.get(stmt_key, []):
            date_str = annual.get("date")
            if not date_str:
                continue
            try:
                fiscal_year = pd.Timestamp(date_str, tz="UTC")
            except Exception:
                continue
            for fmp_field, our_name in field_map.items():
                val = _to_float(annual.get(fmp_field))
                if val is not None:
                    rows.append({
                        "ticker": ticker,
                        "fiscal_year": fiscal_year,
                        "statement": stmt_name,
                        "metric": our_name,
                        "value": val,
                    })
    return rows


def _make_empty_info_row(ticker: str) -> dict:
    return dict(
        ticker=ticker, sector=None, sub_industry=None, currency=None,
        p_e=None, p_fcf=None, ev_ebitda=None,
    )


# ---------------------------------------------------------------------------
# Public API — same external interface as the yfinance version
# ---------------------------------------------------------------------------


def fetch_sp500_tickers(
    api_key: str | None = None,
    *,
    _cache: _FMPCache | None = None,
    _session: requests.Session | None = None,
) -> list[str]:
    """Fetch the current S&P 500 constituent list from FMP."""
    api_key = api_key or os.environ.get("CRUCIBLE_FMP_API_KEY", "")
    if not api_key:
        raise ValueError("CRUCIBLE_FMP_API_KEY is not set")
    cache = _cache or _FMPCache(Path("data/fmp_cache.db"))
    session = _session or requests.Session()

    data = _cached_get(session, cache, "sp500_constituents", _TTL_SP500,
                       "/sp500_constituent", api_key)
    if not data:
        raise RuntimeError(
            "Could not fetch S&P 500 list — rate limit hit or API error. "
            "Try again tomorrow or check your API key."
        )
    tickers = [str(r["symbol"]).replace(".", "-") for r in data if r.get("symbol")]
    logger.info("Fetched %d S&P 500 tickers from FMP", len(tickers))
    return tickers


def fetch_info(
    tickers: list[str],
    api_key: str,
    cache: _FMPCache,
    session: requests.Session,
) -> pd.DataFrame:
    """Pull static snapshot + computed valuation multiples for each ticker."""
    records: list[dict] = []
    rate_limited_count = 0

    for i, ticker in enumerate(tickers, 1):
        if i % 50 == 0:
            logger.info(
                "fetch_info: %d/%d (%d FMP requests today)",
                i, len(tickers), cache.requests_today(),
            )
        try:
            raw = _fetch_ticker_raw(ticker, api_key, cache, session)
            if not any(raw.values()):
                rate_limited_count += 1
                records.append(_make_empty_info_row(ticker))
            else:
                records.append(_raw_to_info_row(ticker, raw))
        except Exception:
            logger.warning("fetch_info: skipping %s — error", ticker, exc_info=True)
            records.append(_make_empty_info_row(ticker))

    if rate_limited_count:
        logger.warning(
            "%d tickers had no data (daily rate limit). Re-run tomorrow to fill cache.",
            rate_limited_count,
        )

    df = pd.DataFrame(records).set_index("ticker")
    logger.info(
        "fetch_info: %d/%d tickers have sector data",
        df["sector"].notna().sum(), len(tickers),
    )
    return df


def fetch_financials(
    tickers: list[str],
    api_key: str,
    cache: _FMPCache,
    session: requests.Session,
) -> pd.DataFrame:
    """Pull annual income, balance, and cash flow data in long panel format.

    Columns: ticker, fiscal_year (UTC Timestamp), statement, metric, value.
    Metric names are normalized to match what cleaner.py expects.
    """
    all_rows: list[dict] = []

    for i, ticker in enumerate(tickers, 1):
        if i % 50 == 0:
            logger.info("fetch_financials: %d/%d", i, len(tickers))
        try:
            raw = _fetch_ticker_raw(ticker, api_key, cache, session)
            all_rows.extend(_raw_to_panel_rows(ticker, raw))
        except Exception:
            logger.warning("fetch_financials: skipping %s — error", ticker, exc_info=True)

    panel = pd.DataFrame(
        all_rows or [],
        columns=["ticker", "fiscal_year", "statement", "metric", "value"],
    )
    if not panel.empty:
        panel["fiscal_year"] = pd.to_datetime(panel["fiscal_year"], utc=True)

    logger.info(
        "fetch_financials: %d rows for %d tickers",
        len(panel), panel["ticker"].nunique() if not panel.empty else 0,
    )
    return panel


def save_raw(
    tickers: list[str],
    info_df: pd.DataFrame,
    panel_df: pd.DataFrame,
    raw_dir: Path,
    run_ts: str,
) -> tuple[Path, Path, Path]:
    """Write tickers, info, and panel Parquet files to raw_dir.

    Returns (tickers_path, info_path, panel_path).
    raw_dir is created if it does not exist.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)

    tickers_path = raw_dir / f"sp500_tickers_{run_ts}.parquet"
    info_path = raw_dir / f"sp500_info_{run_ts}.parquet"
    panel_path = raw_dir / f"sp500_panel_{run_ts}.parquet"

    pd.DataFrame({"ticker": tickers}).to_parquet(tickers_path, index=False)
    info_df.to_parquet(info_path)

    if panel_df.empty:
        panel_df = pd.DataFrame(
            columns=["ticker", "fiscal_year", "statement", "metric", "value"]
        )
    panel_df.to_parquet(panel_path, index=False)

    logger.info("Saved raw files — run_ts=%s  dir=%s", run_ts, raw_dir)
    return tickers_path, info_path, panel_path


def fetch_universe(
    universe_id: str,
    raw_dir: Path,
    tickers: list[str] | None = None,
) -> tuple[str, Path, Path, Path]:
    """Orchestrate a full data fetch for the given universe.

    Returns (run_ts, tickers_path, info_path, panel_path).
    Pass tickers to override the default universe list (useful for testing).
    """
    if universe_id != "SP500":
        raise NotImplementedError(f"Universe {universe_id!r} not yet supported")

    api_key = os.environ.get("CRUCIBLE_FMP_API_KEY", "")
    if not api_key:
        raise ValueError(
            "CRUCIBLE_FMP_API_KEY must be set in .env before running FMP scans"
        )

    cache_path = Path(os.environ.get("CRUCIBLE_FMP_CACHE_PATH", "data/fmp_cache.db"))
    cache = _FMPCache(cache_path)
    session = requests.Session()
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if tickers is None:
        tickers = fetch_sp500_tickers(api_key, _cache=cache, _session=session)

    logger.info(
        "FMP fetch starting — %d tickers, %d requests used today, %d remaining",
        len(tickers), cache.requests_today(), _DAILY_LIMIT - cache.requests_today(),
    )

    info_df = fetch_info(tickers, api_key, cache, session)
    panel_df = fetch_financials(tickers, api_key, cache, session)
    tickers_path, info_path, panel_path = save_raw(tickers, info_df, panel_df, raw_dir, run_ts)
    return run_ts, tickers_path, info_path, panel_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_float(value: object) -> float | None:
    """Convert a value to float, returning None on failure or non-finite result."""
    if value is None:
        return None
    try:
        f = float(value)  # type: ignore[arg-type]
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None
