"""Data extraction from SEC EDGAR (primary) and yfinance (prices + sector only).

Fundamental data comes from the local EDGAR bulk download (companyfacts.zip).
Run scripts/download_edgar_bulk.py once before using this module.

Point-in-time correctness: every filing record has a `filed` date (the exact date
the SEC received it).  _parse_edgar_json filters to `filed <= as_of_date` so that
a 10-K for FY2020 filed in Feb 2021 is never used in a Jan 2021 backtest scan.

yfinance is used exclusively for:
  - sector and sub_industry classification (fetch_info)
  - historical price data (used by run_backtest.py directly via yf.download)
It must never be used for fundamental metrics.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# XBRL taxonomy
# ---------------------------------------------------------------------------

# Maps our metric names to candidate XBRL concept names, ordered by preference.
# The first tag found with data for a given company wins; remaining candidates
# are skipped. Annual 10-K only — quarterly (fp != "FY") records are excluded
# in _parse_edgar_json.
_XBRL_TAXONOMY: dict[str, list[str]] = {
    "Total Revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "RevenuesNetOfInterestExpense",  # banks / financial services
    ],
    "Gross Profit": [
        "GrossProfit",
    ],
    "Net Income": [
        "NetIncomeLoss",
        "ProfitLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
    ],
    "Operating Income": [
        "OperatingIncomeLoss",
    ],
    "Depreciation Amortization": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
        "Depreciation",
    ],
    "Total Assets": [
        "Assets",
    ],
    "Current Liabilities": [
        "LiabilitiesCurrent",
    ],
    "Long Term Debt": [
        "LongTermDebtNoncurrent",
        "LongTermDebt",
        "LongTermNotesPayable",
    ],
    "Short Term Debt": [
        "DebtCurrent",
        "ShortTermBorrowings",
        "LongTermDebtCurrent",
        "NotesPayableCurrent",
    ],
    "Cash And Cash Equivalents": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsAndShortTermInvestments",
        "Cash",
    ],
    "Operating Cash Flow": [
        "NetCashProvidedByUsedInOperatingActivities",
    ],
    "Capital Expenditure": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "CapitalExpendituresIncurredButNotYetPaid",
    ],
    "Total Equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "Income Tax Expense": [
        "IncomeTaxExpenseBenefit",
    ],
    "Interest Expense": [
        "InterestExpense",
        "InterestAndDebtExpense",
    ],
}

# ---------------------------------------------------------------------------
# CIK mapping
# ---------------------------------------------------------------------------


def _load_cik_mapping(
    mapping_path: Path = Path("data/raw/edgar/cik_mapping.json"),
) -> dict[str, str]:
    """Load SEC company_tickers.json, return {TICKER: zero-padded-10-digit-CIK}.

    The SEC mapping file structure: {"0": {"cik_str": 320193, "ticker": "AAPL", ...}, ...}
    """
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"CIK mapping not found at {mapping_path}. "
            "Run scripts/download_edgar_bulk.py first."
        )
    raw: dict[str, dict[str, Any]] = json.loads(mapping_path.read_bytes())
    return {
        entry["ticker"].upper(): str(entry["cik_str"]).zfill(10)
        for entry in raw.values()
        if "ticker" in entry and "cik_str" in entry
    }


# ---------------------------------------------------------------------------
# Point-in-time EDGAR JSON parser
# ---------------------------------------------------------------------------


def _parse_edgar_json(
    cik: str,
    as_of_date: pd.Timestamp,
    edgar_dir: Path = Path("data/raw/edgar/companyfacts"),
) -> dict[str, list[dict[str, Any]]]:
    """Read a local EDGAR CIK JSON and return annual facts available as of as_of_date.

    Returns: {metric_name: [{"end": str, "val": float, "filed": str, "fy": int}, ...]},
    sorted newest fiscal year first. Only 10-K / 10-K/A annual (fp="FY") records
    with filed <= as_of_date are included. When multiple filings cover the same fiscal
    year (e.g. amendments), only the one with the latest filed date is kept.
    """
    padded_cik = str(cik).zfill(10)
    json_path = edgar_dir / f"CIK{padded_cik}.json"
    if not json_path.exists():
        logger.debug("No EDGAR JSON for CIK %s at %s", padded_cik, json_path)
        return {}

    raw: dict[str, Any] = json.loads(json_path.read_bytes())
    us_gaap: dict[str, Any] = raw.get("facts", {}).get("us-gaap", {})
    as_of_str = as_of_date.strftime("%Y-%m-%d")

    result: dict[str, list[dict[str, Any]]] = {}

    for metric_name, candidates in _XBRL_TAXONOMY.items():
        best_records: list[dict[str, Any]] = []

        for tag in candidates:
            concept = us_gaap.get(tag)
            if not concept:
                continue

            # Pick USD unit values only (skip shares, pure, USD/shares, etc.)
            units: dict[str, list[dict]] = concept.get("units", {})
            usd_records: list[dict] = []
            for unit_key, records in units.items():
                if unit_key.upper() == "USD":
                    usd_records = records
                    break
            if not usd_records:
                continue

            # Filter: annual 10-K/10-K/A, fiscal year (fp=FY), filed <= as_of_date
            filtered: list[dict[str, Any]] = []
            for rec in usd_records:
                if rec.get("form") not in {"10-K", "10-K/A"}:
                    continue
                if rec.get("fp") != "FY":
                    continue
                filed = rec.get("filed", "")
                if not filed or filed > as_of_str:
                    continue
                filtered.append({
                    "end": rec["end"],
                    "val": rec["val"],
                    "filed": filed,
                    "fy": rec.get("fy"),
                })

            if filtered:
                best_records = filtered
                break  # first candidate with data wins for this metric

        if not best_records:
            continue

        # De-duplicate by fiscal year — keep the record with the latest filed date
        by_fy: dict[Any, dict[str, Any]] = {}
        for rec in best_records:
            fy_key = rec.get("fy") or rec["end"][:4]
            existing = by_fy.get(fy_key)
            if existing is None or rec["filed"] > existing["filed"]:
                by_fy[fy_key] = rec

        result[metric_name] = sorted(by_fy.values(), key=lambda r: r["end"], reverse=True)

    return result


# ---------------------------------------------------------------------------
# Panel + info row builders
# ---------------------------------------------------------------------------


def _edgar_to_panel_rows(
    ticker: str,
    facts: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Convert parsed EDGAR facts to long-panel rows matching the cleaner.py schema.

    Derived metrics computed here (added alongside raw components):
    - EBITDA         = Operating Income + Depreciation Amortization
    - Free Cash Flow = Operating Cash Flow − |Capital Expenditure|
    - Total Debt     = Long Term Debt + Short Term Debt
    """
    rows: list[dict[str, Any]] = []

    for metric_name, records in facts.items():
        for rec in records:
            try:
                fiscal_year = pd.Timestamp(rec["end"], tz="UTC")
            except Exception:
                continue
            rows.append({
                "ticker": ticker,
                "fiscal_year": fiscal_year,
                "metric": metric_name,
                "value": float(rec["val"]),
            })

    # Build lookup for derived metric computation: {(metric, fy_date_str) → value}
    lookup: dict[tuple[str, str], float] = {}
    for row in rows:
        key = (row["metric"], row["fiscal_year"].strftime("%Y-%m-%d"))
        lookup[key] = row["value"]

    fy_dates: set[str] = {row["fiscal_year"].strftime("%Y-%m-%d") for row in rows}

    for fy_str in fy_dates:
        fy_ts = pd.Timestamp(fy_str, tz="UTC")

        op_inc = lookup.get(("Operating Income", fy_str))
        da = lookup.get(("Depreciation Amortization", fy_str))
        if op_inc is not None and da is not None:
            rows.append({
                "ticker": ticker,
                "fiscal_year": fy_ts,
                "metric": "EBITDA",
                "value": op_inc + da,
            })

        ocf = lookup.get(("Operating Cash Flow", fy_str))
        capex = lookup.get(("Capital Expenditure", fy_str))
        if ocf is not None and capex is not None:
            rows.append({
                "ticker": ticker,
                "fiscal_year": fy_ts,
                "metric": "Free Cash Flow",
                "value": ocf - abs(capex),
            })

        ltd = lookup.get(("Long Term Debt", fy_str))
        std = lookup.get(("Short Term Debt", fy_str))
        if ltd is not None or std is not None:
            rows.append({
                "ticker": ticker,
                "fiscal_year": fy_ts,
                "metric": "Total Debt",
                "value": (ltd or 0.0) + (std or 0.0),
            })

    return rows


def _edgar_to_info_row(
    ticker: str,
    facts: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Build an info row from EDGAR facts + yfinance sector/market-cap data.

    Sector and sub_industry come from yfinance fast_info (non-fundamental; acceptable).
    Valuation multiples use current market cap from yfinance + most recent EDGAR fundamentals.
    """
    row: dict[str, Any] = {
        "ticker": ticker,
        "sector": None,
        "sub_industry": None,
        "currency": "USD",
        "p_e": None,
        "p_fcf": None,
        "ev_ebitda": None,
    }

    mkt_cap: float | None = None
    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info
        row["sector"] = info.get("sector") or None
        row["sub_industry"] = info.get("industry") or None
        mkt_cap = _to_float(info.get("marketCap"))
    except Exception:
        logger.debug("yfinance info unavailable for %s", ticker)

    def _latest_val(metric: str) -> float | None:
        records = facts.get(metric)
        return _to_float(records[0]["val"]) if records else None

    # Compute derived values needed for multiples
    op_inc = _latest_val("Operating Income")
    da = _latest_val("Depreciation Amortization")
    ocf = _latest_val("Operating Cash Flow")
    capex = _latest_val("Capital Expenditure")
    ltd = _latest_val("Long Term Debt") or 0.0
    std = _latest_val("Short Term Debt") or 0.0
    cash = _latest_val("Cash And Cash Equivalents") or 0.0

    net_income = _latest_val("Net Income")
    ebitda = (op_inc + da) if (op_inc is not None and da is not None) else None
    fcf = (ocf - abs(capex)) if (ocf is not None and capex is not None) else None
    total_debt = ltd + std

    if mkt_cap and net_income and net_income > 0:
        row["p_e"] = mkt_cap / net_income
    if mkt_cap and fcf and fcf > 0:
        row["p_fcf"] = mkt_cap / fcf
    if mkt_cap and ebitda and ebitda > 0:
        row["ev_ebitda"] = (mkt_cap + total_debt - cash) / ebitda

    return row


def _make_empty_info_row(ticker: str) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "sector": None,
        "sub_industry": None,
        "currency": "USD",
        "p_e": None,
        "p_fcf": None,
        "ev_ebitda": None,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_sp500_tickers() -> list[str]:
    """Fetch the current S&P 500 constituent list from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url, storage_options={"User-Agent": "Mozilla/5.0 Crucible"})
    tickers = list(tables[0]["Symbol"].str.replace(".", "-", regex=False))
    logger.info("Fetched %d S&P 500 tickers from Wikipedia", len(tickers))
    return tickers


def fetch_financials(
    tickers: list[str],
    as_of_date: pd.Timestamp,
    edgar_dir: Path = Path("data/raw/edgar/companyfacts"),
    cik_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Pull annual fundamental data from local EDGAR JSON files in long-panel format.

    Columns: ticker, fiscal_year (UTC Timestamp), metric, value.
    Only filings with filed <= as_of_date are included (point-in-time).
    """
    if cik_map is None:
        cik_map = _load_cik_mapping()

    all_rows: list[dict[str, Any]] = []

    for i, ticker in enumerate(tickers, 1):
        if i % 100 == 0:
            logger.info("fetch_financials: %d / %d tickers processed", i, len(tickers))
        cik = cik_map.get(ticker.upper())
        if not cik:
            logger.debug("No CIK found for ticker %s — skipping", ticker)
            continue
        try:
            facts = _parse_edgar_json(cik, as_of_date, edgar_dir)
            all_rows.extend(_edgar_to_panel_rows(ticker, facts))
        except Exception:
            logger.warning("fetch_financials: error processing %s", ticker, exc_info=True)

    panel = pd.DataFrame(
        all_rows or [],
        columns=["ticker", "fiscal_year", "metric", "value"],
    )
    if not panel.empty:
        panel["fiscal_year"] = pd.to_datetime(panel["fiscal_year"], utc=True)

    logger.info(
        "fetch_financials: %d rows for %d tickers (as_of %s)",
        len(panel),
        panel["ticker"].nunique() if not panel.empty else 0,
        as_of_date.date(),
    )
    return panel


def fetch_info(
    tickers: list[str],
    as_of_date: pd.Timestamp,
    edgar_dir: Path = Path("data/raw/edgar/companyfacts"),
    cik_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build snapshot info (sector, valuation multiples) for each ticker.

    Fundamentals come from EDGAR (point-in-time as of as_of_date).
    Sector and market cap come from yfinance (current — acceptable for non-fundamental fields).
    """
    if cik_map is None:
        cik_map = _load_cik_mapping()

    records: list[dict[str, Any]] = []

    for i, ticker in enumerate(tickers, 1):
        if i % 50 == 0:
            logger.info("fetch_info: %d / %d", i, len(tickers))
        cik = cik_map.get(ticker.upper())
        if not cik:
            records.append(_make_empty_info_row(ticker))
            continue
        try:
            facts = _parse_edgar_json(cik, as_of_date, edgar_dir)
            records.append(_edgar_to_info_row(ticker, facts))
        except Exception:
            logger.warning("fetch_info: error for %s", ticker, exc_info=True)
            records.append(_make_empty_info_row(ticker))

    df = pd.DataFrame(records).set_index("ticker")
    logger.info(
        "fetch_info: %d / %d tickers have sector data",
        df["sector"].notna().sum(),
        len(tickers),
    )
    return df


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
        panel_df = pd.DataFrame(columns=["ticker", "fiscal_year", "metric", "value"])
    panel_df.to_parquet(panel_path, index=False)

    logger.info("Saved raw files — run_ts=%s  dir=%s", run_ts, raw_dir)
    return tickers_path, info_path, panel_path


def fetch_universe(
    universe_id: str,
    raw_dir: Path,
    tickers: list[str] | None = None,
    as_of_date: pd.Timestamp | None = None,
    edgar_dir: Path = Path("data/raw/edgar/companyfacts"),
) -> tuple[str, Path, Path, Path]:
    """Orchestrate a full data fetch for the given universe.

    Returns (run_ts, tickers_path, info_path, panel_path).
    Pass tickers to override the default universe list.
    Pass as_of_date for historical (backtest) scans; defaults to now (UTC).
    """
    if universe_id not in {"SP500", "RUSSELL1000", "RUSSELL3000"}:
        raise NotImplementedError(f"Universe {universe_id!r} not yet supported")

    import datetime

    if as_of_date is None:
        as_of_date = pd.Timestamp(datetime.datetime.now(timezone.utc))

    run_ts = datetime.datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if tickers is None:
        tickers = fetch_sp500_tickers()

    cik_map = _load_cik_mapping()

    logger.info(
        "EDGAR fetch starting — %d tickers, as_of=%s",
        len(tickers),
        as_of_date.date(),
    )

    info_df = fetch_info(tickers, as_of_date, edgar_dir, cik_map)
    panel_df = fetch_financials(tickers, as_of_date, edgar_dir, cik_map)
    tickers_path, info_path, panel_path = save_raw(
        tickers, info_df, panel_df, raw_dir, run_ts
    )
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
