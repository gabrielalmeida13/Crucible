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

import io
import json
import logging
import math
import os
from datetime import timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import requests
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


@lru_cache(maxsize=300)
@lru_cache(maxsize=600)
def _load_dei_shares_cached(
    padded_cik: str,
    edgar_dir_str: str,
) -> list[dict[str, Any]]:
    """Load EntityCommonStockSharesOutstanding records from the dei namespace. LRU-cached.

    Returns [{end, val, filed, form}, ...] with NO date filter applied.
    Includes all form types (10-K, 10-Q, etc.) — caller applies the as_of_date filter.
    """
    json_path = Path(edgar_dir_str) / f"CIK{padded_cik}.json"
    if not json_path.exists():
        return []
    raw: dict[str, Any] = json.loads(json_path.read_bytes())
    dei: dict[str, Any] = raw.get("facts", {}).get("dei", {})
    concept = dei.get("EntityCommonStockSharesOutstanding", {})
    recs: list[dict] = concept.get("units", {}).get("shares", [])
    result: list[dict[str, Any]] = []
    for rec in recs:
        filed = rec.get("filed", "")
        val = rec.get("val")
        if not filed or val is None:
            continue
        try:
            val_f = float(val)
        except (TypeError, ValueError):
            continue
        if val_f <= 0:
            continue
        result.append({
            "end": rec.get("end", ""),
            "val": val_f,
            "filed": filed,
            "form": rec.get("form", ""),
        })
    return result


def get_shares_outstanding(
    cik: str,
    as_of_date: pd.Timestamp,
    edgar_dir: Path = Path("data/raw/edgar/companyfacts"),
) -> float | None:
    """Return shares outstanding from EDGAR DEI filings as of as_of_date (point-in-time).

    Takes the most recently filed value on or before as_of_date.  When multiple
    records share the same filed date (e.g. multi-class share structures), sums
    them so that the total share count is correct.
    """
    padded_cik = str(cik).zfill(10)
    records = _load_dei_shares_cached(padded_cik, str(edgar_dir))
    as_of_str = as_of_date.strftime("%Y-%m-%d")
    eligible = [r for r in records if r["filed"] <= as_of_str]
    if not eligible:
        return None
    latest_filed = max(r["filed"] for r in eligible)
    same_date = [r for r in eligible if r["filed"] == latest_filed]
    return sum(r["val"] for r in same_date)


@lru_cache(maxsize=300)
def _load_cik_annual_facts(
    padded_cik: str,
    edgar_dir_str: str,
) -> dict[str, list[dict[str, Any]]]:
    """Load all 10-K/10-K/A annual facts for a CIK from disk. LRU-evicts at 300 entries.

    Returns {metric_name: [{end, val, filed, fy}, ...]} with NO date filter applied —
    _parse_edgar_json applies the as_of_date filter in memory on the cached result.

    The 300-entry cap keeps ≲50 MB of parsed dicts in memory (≈150 KB per company
    for 15 metrics × 15 years × ~200 bytes/record), well within the 2 GiB budget.

    Do NOT mutate the returned dicts — the cache returns the same object on every hit.
    """
    json_path = Path(edgar_dir_str) / f"CIK{padded_cik}.json"
    if not json_path.exists():
        logger.debug("No EDGAR JSON for CIK %s at %s", padded_cik, json_path)
        return {}

    raw: dict[str, Any] = json.loads(json_path.read_bytes())
    us_gaap: dict[str, Any] = raw.get("facts", {}).get("us-gaap", {})

    result: dict[str, list[dict[str, Any]]] = {}

    for metric_name, candidates in _XBRL_TAXONOMY.items():
        for tag in candidates:
            concept = us_gaap.get(tag)
            if not concept:
                continue

            units: dict[str, list[dict]] = concept.get("units", {})
            usd_records: list[dict] = []
            for unit_key, recs in units.items():
                if unit_key.upper() == "USD":
                    usd_records = recs
                    break
            if not usd_records:
                continue

            collected: list[dict[str, Any]] = []
            for rec in usd_records:
                if rec.get("form") not in {"10-K", "10-K/A"}:
                    continue
                if rec.get("fp") != "FY":
                    continue
                filed = rec.get("filed", "")
                end = rec.get("end", "")
                val = rec.get("val")
                if not filed or val is None:
                    continue
                try:
                    fy_key = rec.get("fy") or int(end[:4])
                except (ValueError, TypeError):
                    continue
                collected.append({"end": end, "val": val, "filed": filed, "fy": fy_key})

            if collected:
                result[metric_name] = collected
                break  # first candidate with data wins for this metric

    return result


def _parse_edgar_json(
    cik: str,
    as_of_date: pd.Timestamp,
    edgar_dir: Path = Path("data/raw/edgar/companyfacts"),
) -> dict[str, list[dict[str, Any]]]:
    """Return annual EDGAR facts available as of as_of_date (point-in-time).

    Returns: {metric_name: [{"end": str, "val": float, "filed": str, "fy": int}, ...]},
    sorted newest fiscal year first. Only 10-K / 10-K/A annual (fp="FY") records
    with filed <= as_of_date are included. When multiple filings cover the same fiscal
    year (e.g. amendments), only the one with the latest filed date is kept.

    Raw JSON loading is LRU-cached in _load_cik_annual_facts (maxsize=300).
    """
    padded_cik = str(cik).zfill(10)
    as_of_str = as_of_date.strftime("%Y-%m-%d")
    all_facts = _load_cik_annual_facts(padded_cik, str(edgar_dir))

    result: dict[str, list[dict[str, Any]]] = {}

    for metric_name, records in all_facts.items():
        # Point-in-time filter
        filtered = [r for r in records if r["filed"] <= as_of_str]
        if not filtered:
            continue

        # Dedup by fiscal year — keep latest filed within the cutoff
        by_fy: dict[Any, dict[str, Any]] = {}
        for rec in filtered:
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
    cik: str | None = None,
    as_of_date: pd.Timestamp | None = None,
    edgar_dir: Path = Path("data/raw/edgar/companyfacts"),
) -> dict[str, Any]:
    """Build an info row from EDGAR facts + yfinance sector/price data.

    Market cap = current price (yfinance) × shares outstanding (EDGAR point-in-time).
    Valuation multiples use 5-year average fundamentals from EDGAR for stability.
    Sector and sub_industry come from yfinance (non-fundamental; acceptable).
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

    price: float | None = None
    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info
        row["sector"] = info.get("sector") or None
        row["sub_industry"] = info.get("industry") or None
        price = _to_float(info.get("currentPrice") or info.get("regularMarketPrice"))
    except Exception:
        logger.debug("yfinance info unavailable for %s", ticker)

    # Market cap from EDGAR shares (point-in-time) × current price
    shares: float | None = None
    if cik and as_of_date:
        shares = get_shares_outstanding(cik, as_of_date, edgar_dir)
    mkt_cap = price * shares if (price is not None and shares is not None) else None

    def _latest_val(metric: str) -> float | None:
        records = facts.get(metric)
        return _to_float(records[0]["val"]) if records else None

    def _avg_5yr(metric: str, n: int = 5) -> float | None:
        """Average of the n most recent annual values for a raw EDGAR metric."""
        records = facts.get(metric, [])
        vals = [_to_float(r["val"]) for r in records[:n]]
        valid = [v for v in vals if v is not None]
        return sum(valid) / len(valid) if valid else None

    def _avg_fcf_5yr(n: int = 5) -> float | None:
        """5-year average FCF = avg(OCF - |CapEx|) aligned by fiscal year."""
        ocf_recs = facts.get("Operating Cash Flow", [])
        capex_recs = facts.get("Capital Expenditure", [])
        if not ocf_recs or not capex_recs:
            return None
        capex_by_year = {r["end"][:4]: _to_float(r["val"]) for r in capex_recs}
        vals: list[float] = []
        for r in ocf_recs:
            if len(vals) >= n:
                break
            ocf_v = _to_float(r["val"])
            capex_v = capex_by_year.get(r["end"][:4])
            if ocf_v is not None and capex_v is not None:
                vals.append(ocf_v - abs(capex_v))
        return sum(vals) / len(vals) if vals else None

    def _avg_ebitda_5yr(n: int = 5) -> float | None:
        """5-year average EBITDA = avg(Operating Income + D&A) aligned by fiscal year."""
        oi_recs = facts.get("Operating Income", [])
        da_recs = facts.get("Depreciation Amortization", [])
        if not oi_recs or not da_recs:
            return None
        da_by_year = {r["end"][:4]: _to_float(r["val"]) for r in da_recs}
        vals: list[float] = []
        for r in oi_recs:
            if len(vals) >= n:
                break
            oi_v = _to_float(r["val"])
            da_v = da_by_year.get(r["end"][:4])
            if oi_v is not None and da_v is not None:
                vals.append(oi_v + da_v)
        return sum(vals) / len(vals) if vals else None

    ltd = _latest_val("Long Term Debt") or 0.0
    std = _latest_val("Short Term Debt") or 0.0
    cash = _latest_val("Cash And Cash Equivalents") or 0.0
    total_debt = ltd + std

    fcf_5yr = _avg_fcf_5yr()
    ebitda_5yr = _avg_ebitda_5yr()
    ni_5yr_vals = [_to_float(r["val"]) for r in facts.get("Net Income", [])[:5]]
    ni_5yr_avg = (
        sum(v for v in ni_5yr_vals if v is not None) / len([v for v in ni_5yr_vals if v is not None])
        if any(v is not None for v in ni_5yr_vals) else None
    )

    _MAX_MULTIPLE = 200.0
    if mkt_cap and fcf_5yr and fcf_5yr > 0:
        row["p_fcf"] = min(mkt_cap / fcf_5yr, _MAX_MULTIPLE)
    if mkt_cap and ebitda_5yr and ebitda_5yr > 0:
        ev = mkt_cap + total_debt - cash
        if ev > 0:
            row["ev_ebitda"] = min(ev / ebitda_5yr, _MAX_MULTIPLE)
    if mkt_cap and ni_5yr_avg and ni_5yr_avg > 0:
        row["p_e"] = min(mkt_cap / ni_5yr_avg, _MAX_MULTIPLE)

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


_IWB_CSV_URL = (
    "https://www.ishares.com/us/products/239707/ishares-russell-1000-etf"
    "/1467271812596.ajax?fileType=csv&fileName=IWB_holdings&dataType=fund"
)
_IWB_CSV_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_RUSSELL1000_FALLBACK_PATH = Path("data/raw/russell1000_tickers.txt")
# The fallback file contains SP500 + SP400 (903 tickers), a practical approximation
# of the Russell 1000 built from Wikipedia when iShares CSV is bot-blocked.


def _parse_iwb_csv(content: bytes) -> list[str]:
    """Parse iShares IWB holdings CSV, returning equity tickers only.

    The CSV has several metadata rows before the actual header row.
    Filters to Asset Class == "Equity", strips empty and derivative tickers.
    """
    text = content.decode("utf-8", errors="replace")
    lines = text.splitlines()

    # Locate the header row (contains "Ticker")
    header_idx: int | None = None
    for i, line in enumerate(lines):
        if "Ticker" in line:
            header_idx = i
            break

    if header_idx is None:
        raise ValueError("Could not locate 'Ticker' header row in iShares CSV")

    data_text = "\n".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(data_text), on_bad_lines="skip")

    # Filter to equity rows only
    asset_col = next(
        (c for c in df.columns if c.strip().lower() in ("asset class", "assetclass")),
        None,
    )
    if asset_col:
        df = df[df[asset_col].astype(str).str.strip().str.lower() == "equity"]

    tickers = (
        df["Ticker"]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )
    # Drop empty strings and cash/derivative placeholders (contain "-")
    tickers = [t for t in tickers if t and "-" not in t]
    return sorted(set(tickers))


def fetch_russell1000_tickers() -> list[str]:
    """Return Russell 1000 constituents from the iShares IWB holdings CSV.

    Primary:  HTTP request to the iShares IWB CSV download endpoint.
    Fallback: data/raw/russell1000_tickers.txt (one ticker per line).

    Raises RuntimeError with download instructions if both sources are unavailable.
    """
    # Primary: iShares IWB CSV
    try:
        resp = requests.get(
            _IWB_CSV_URL,
            headers={"User-Agent": _IWB_CSV_UA},
            timeout=60,
        )
        resp.raise_for_status()
        # Guard against bot-protection pages returning HTML with a 200 status
        if resp.content[:20].lstrip().startswith(b"<"):
            raise ValueError("iShares returned HTML instead of CSV (bot protection active)")
        tickers = _parse_iwb_csv(resp.content)
        if tickers:
            logger.info("Russell 1000: %d tickers from iShares IWB CSV", len(tickers))
            return tickers
        logger.warning("iShares IWB CSV returned 0 equity tickers; trying local fallback")
    except Exception as exc:
        logger.warning("iShares IWB CSV fetch failed: %s", exc)

    # Fallback: local text file
    if _RUSSELL1000_FALLBACK_PATH.exists():
        tickers = [
            ln.strip()
            for ln in _RUSSELL1000_FALLBACK_PATH.read_text().splitlines()
            if ln.strip()
        ]
        if tickers:
            logger.info(
                "Russell 1000: %d tickers from local fallback %s",
                len(tickers),
                _RUSSELL1000_FALLBACK_PATH,
            )
            return tickers

    print(
        "\nCould not fetch Russell 1000 tickers automatically.\n"
        "To resolve this, download the holdings manually:\n"
        "  1. Go to https://www.ishares.com/us/products/239707/ishares-russell-1000-etf\n"
        "  2. Click 'Download Holdings' → CSV\n"
        "  3. Extract the Ticker column (equity rows only) and save one ticker per line to:\n"
        f"     {_RUSSELL1000_FALLBACK_PATH.resolve()}\n"
    )
    raise RuntimeError(
        "Russell 1000 tickers unavailable: iShares CSV unreachable and local fallback "
        f"not found at {_RUSSELL1000_FALLBACK_PATH}."
    )


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
            records.append(_edgar_to_info_row(ticker, facts, cik, as_of_date, edgar_dir))
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

    cik_map = _load_cik_mapping(raw_dir / "edgar" / "cik_mapping.json")

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
