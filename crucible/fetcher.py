"""Data extraction from yfinance (dev) and FMP (Phase 2+).

NOTE: yfinance does not provide point-in-time data. It must never be used
for backtesting. FMP migration is required before Phase 2.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def fetch_sp500_tickers() -> list[str]:
    """Pull the S&P 500 constituent list from Wikipedia."""
    tables = pd.read_html(_SP500_WIKI_URL)
    tickers: list[str] = (
        tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
    )
    logger.info("Fetched %d S&P 500 tickers from Wikipedia", len(tickers))
    return tickers


def fetch_info(tickers: list[str]) -> pd.DataFrame:
    """Pull static snapshot data (sector, currency, valuation multiples) for each ticker.

    Returns a DataFrame with one row per ticker, indexed by ticker.
    Missing fields are NaN — never filled with estimates.
    """
    records: list[dict] = []
    for i, ticker in enumerate(tickers, 1):
        if i % 50 == 0:
            logger.info("fetch_info: %d/%d", i, len(tickers))
        try:
            info = yf.Ticker(ticker).info
            records.append(
                {
                    "ticker": ticker,
                    "sector": info.get("sector") or None,
                    "sub_industry": info.get("industry") or None,
                    "currency": info.get("currency") or None,
                    "p_e": _to_float(info.get("trailingPE")),
                    "p_fcf": _to_float(info.get("priceToFreeCashflows")),
                    "ev_ebitda": _to_float(info.get("enterpriseToEbitda")),
                }
            )
        except Exception:
            logger.warning("fetch_info: skipping %s — API error", ticker, exc_info=True)
            records.append(
                {
                    "ticker": ticker,
                    "sector": None,
                    "sub_industry": None,
                    "currency": None,
                    "p_e": None,
                    "p_fcf": None,
                    "ev_ebitda": None,
                }
            )

    df = pd.DataFrame(records).set_index("ticker")
    logger.info(
        "fetch_info: %d/%d tickers have sector data",
        df["sector"].notna().sum(),
        len(tickers),
    )
    return df


def fetch_financials(tickers: list[str]) -> pd.DataFrame:
    """Pull annual income statement, balance sheet, and cash flow in long panel format.

    Columns: ticker, fiscal_year (UTC Timestamp), statement, metric, value.
    """
    rows: list[dict] = []

    for i, ticker in enumerate(tickers, 1):
        if i % 50 == 0:
            logger.info("fetch_financials: %d/%d", i, len(tickers))
        try:
            t = yf.Ticker(ticker)
            statements: dict[str, pd.DataFrame | None] = {
                "income": t.financials,
                "balance": t.balance_sheet,
                "cashflow": t.cashflow,
            }
            for stmt_name, stmt_df in statements.items():
                if stmt_df is None or stmt_df.empty:
                    logger.warning(
                        "fetch_financials: empty %s for %s", stmt_name, ticker
                    )
                    continue
                for fiscal_year_col in stmt_df.columns:
                    for metric in stmt_df.index:
                        rows.append(
                            {
                                "ticker": ticker,
                                "fiscal_year": fiscal_year_col,
                                "statement": stmt_name,
                                "metric": str(metric),
                                "value": _to_float(stmt_df.loc[metric, fiscal_year_col]),
                            }
                        )
        except Exception:
            logger.warning(
                "fetch_financials: skipping %s — API error", ticker, exc_info=True
            )

    panel = pd.DataFrame(
        rows if rows else [],
        columns=["ticker", "fiscal_year", "statement", "metric", "value"],
    )
    if not panel.empty:
        panel["fiscal_year"] = pd.to_datetime(panel["fiscal_year"], utc=True)

    logger.info(
        "fetch_financials: %d rows for %d tickers",
        len(panel),
        panel["ticker"].nunique() if not panel.empty else 0,
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

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if tickers is None:
        tickers = fetch_sp500_tickers()

    info_df = fetch_info(tickers)
    panel_df = fetch_financials(tickers)
    tickers_path, info_path, panel_path = save_raw(
        tickers, info_df, panel_df, raw_dir, run_ts
    )
    return run_ts, tickers_path, info_path, panel_path


def _to_float(value: object) -> float | None:
    """Convert a value to float, returning None on failure or non-finite result."""
    if value is None:
        return None
    try:
        f = float(value)  # type: ignore[arg-type]
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None
