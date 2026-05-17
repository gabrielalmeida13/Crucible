# EDGAR Data Acquisition Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace FMP with SEC EDGAR as the primary fundamental data source — CIK mapping download, `companyfacts.zip` bulk downloader, and point-in-time XBRL parser in `fetcher.py`.

**Architecture:** Local-first: `download_edgar_bulk.py` fetches `companyfacts.zip` once; `fetcher.py` reads local CIK JSON files and filters facts by `filed <= as_of_date`. `fetch_info` keeps yfinance for sector/price only. `save_raw` is unchanged.

**Tech Stack:** `edgartools`, `requests` (streaming download), `yfinance` (prices + sector), `pandas`

---

### Task 1: Update dependencies, env example, and ROADMAP status

**Files:**
- Modify: `pyproject.toml`
- Modify: `.env.example`
- Modify: `ROADMAP.md`

- [ ] **Step 1: Add `edgartools` to pyproject.toml**

In `pyproject.toml`, add `"edgartools>=2.0"` to the `dependencies` list and remove the `requests` line (edgartools brings it transitively — keep it explicit for clarity):

```toml
dependencies = [
    "pandas>=2.2",
    "numpy>=1.26",
    "yfinance>=0.2",
    "sqlalchemy>=2.0",
    "streamlit>=1.35",
    "pandera>=0.20",
    "python-dotenv>=1.0",
    "lxml>=6.1.0",
    "requests>=2.34.2",
    "edgartools>=2.0",
]
```

- [ ] **Step 2: Add EDGAR_USER_AGENT to .env.example**

```
CRUCIBLE_UNIVERSE=SP500            # SP500 | RUSSELL1000 | RUSSELL3000
CRUCIBLE_DB_PATH=data/crucible.db
CRUCIBLE_LOG_LEVEL=INFO
CRUCIBLE_ACCOUNT_CURRENCY=EUR
EDGAR_USER_AGENT=Crucible yourname@example.com  # Required — SEC blocks requests without it
EDGAR_DATA_DIR=data/raw/edgar                    # Optional — default shown
```

- [ ] **Step 3: Reset Phase 2.1 status in ROADMAP.md**

The ROADMAP currently shows Phase 2 as "complete" but the EDGAR implementation doesn't exist yet.
Change the header from `✓ done` to `← current` and mark the 2.1 checkboxes as unchecked.

Specifically, change:
```
## Phase 2 — Data migration to SEC EDGAR + Full backtest ✓
**Status: Complete**
```
to:
```
## Phase 2 — Data migration to SEC EDGAR + Full backtest ← current
**Status: In progress — 2.1 EDGAR migration underway**
```

And reset all `[x]` items under `### 2.1 — EDGAR migration` to `[ ]`.

Also update the overview diagram to show Phase 2 as in-progress:
```
  ✓ done    ✓ done    ← here
             Phase 2
```

- [ ] **Step 4: Install the new dependency**

Run: `uv sync`
Expected: edgartools appears in the lock; no errors.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .env.example ROADMAP.md uv.lock
git commit -m "Add edgartools dependency; reset Phase 2 EDGAR migration status"
```

---

### Task 2: Create scripts/download_edgar_bulk.py

**Files:**
- Create: `scripts/download_edgar_bulk.py`

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
"""One-time bulk download of SEC EDGAR companyfacts.zip and CIK mapping.

Run once before Phase 2 backtesting. The ZIP is ~1.5 GB compressed (~8 GB extracted).
Individual CIK JSON files land in data/raw/edgar/companyfacts/.

Usage:
    python scripts/download_edgar_bulk.py

Environment variables (set in .env):
    EDGAR_USER_AGENT — required; SEC blocks requests without a valid agent string.
                       Example: "Crucible yourname@example.com"
    EDGAR_DATA_DIR   — optional; defaults to data/raw/edgar
"""

from __future__ import annotations

import json
import logging
import os
import sys
import zipfile
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_CIK_MAPPING_URL = "https://www.sec.gov/files/company_tickers.json"
_COMPANYFACTS_URL = (
    "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"
)
_CHUNK_SIZE = 8 * 1024 * 1024   # 8 MB streaming chunks
_LOG_EVERY_BYTES = 100 * 1024 * 1024  # log progress every 100 MB


def _headers() -> dict[str, str]:
    """Build SEC-compliant request headers. Raises if EDGAR_USER_AGENT is missing."""
    agent = os.environ.get("EDGAR_USER_AGENT", "").strip()
    if not agent:
        raise ValueError(
            "EDGAR_USER_AGENT is not set in .env.\n"
            "The SEC requires a descriptive User-Agent string.\n"
            "Example: EDGAR_USER_AGENT=Crucible yourname@example.com"
        )
    return {"User-Agent": agent, "Accept-Encoding": "gzip, deflate"}


def download_cik_mapping(edgar_dir: Path) -> Path:
    """Download SEC company_tickers.json → edgar_dir/cik_mapping.json."""
    out = edgar_dir / "cik_mapping.json"
    if out.exists():
        log.info("CIK mapping already present at %s — skipping", out)
        return out

    log.info("Downloading CIK mapping from %s", _CIK_MAPPING_URL)
    resp = requests.get(_CIK_MAPPING_URL, headers=_headers(), timeout=30)
    resp.raise_for_status()
    out.write_bytes(resp.content)
    data = json.loads(resp.content)
    log.info("CIK mapping saved → %s (%d companies)", out, len(data))
    return out


def download_companyfacts_zip(edgar_dir: Path) -> Path:
    """Stream-download companyfacts.zip to edgar_dir/, logging every 100 MB."""
    zip_path = edgar_dir / "companyfacts.zip"
    if zip_path.exists():
        size_mb = zip_path.stat().st_size // (1024 * 1024)
        log.info(
            "companyfacts.zip already present (%d MB) at %s — skipping download",
            size_mb, zip_path,
        )
        return zip_path

    log.info("Streaming %s — this may take several minutes (~1.5 GB)", _COMPANYFACTS_URL)
    with requests.get(
        _COMPANYFACTS_URL, headers=_headers(), stream=True, timeout=600
    ) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        next_log = _LOG_EVERY_BYTES
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=_CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if downloaded >= next_log:
                        if total:
                            pct = f"{100 * downloaded / total:.0f}%"
                        else:
                            pct = f"{downloaded / 1e6:.0f} MB"
                        log.info("  Downloaded %s …", pct)
                        next_log += _LOG_EVERY_BYTES

    size_mb = zip_path.stat().st_size // (1024 * 1024)
    log.info("companyfacts.zip saved → %s (%d MB)", zip_path, size_mb)
    return zip_path


def extract_companyfacts(zip_path: Path, dest_dir: Path) -> None:
    """Extract all CIK*.json files from companyfacts.zip into dest_dir."""
    existing = list(dest_dir.glob("CIK*.json"))
    if existing:
        log.info(
            "companyfacts already extracted (%d files in %s) — skipping",
            len(existing), dest_dir,
        )
        return

    dest_dir.mkdir(parents=True, exist_ok=True)
    log.info("Extracting companyfacts.zip → %s", dest_dir)

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [n for n in zf.namelist() if n.endswith(".json")]
        total = len(names)
        log.info("  Found %d JSON files in archive", total)
        for i, name in enumerate(names, 1):
            zf.extract(name, dest_dir)
            if i % 10_000 == 0:
                log.info("  Extracted %d / %d (%.0f%%) …", i, total, 100 * i / total)

    final_count = len(list(dest_dir.glob("CIK*.json")))
    log.info("Extraction complete — %d files in %s", final_count, dest_dir)


def main() -> None:
    edgar_dir = Path(os.environ.get("EDGAR_DATA_DIR", ROOT / "data" / "raw" / "edgar"))
    edgar_dir.mkdir(parents=True, exist_ok=True)
    companyfacts_dir = edgar_dir / "companyfacts"

    download_cik_mapping(edgar_dir)
    zip_path = download_companyfacts_zip(edgar_dir)
    extract_companyfacts(zip_path, companyfacts_dir)

    log.info("EDGAR bulk download complete.")
    log.info("  CIK mapping  : %s", edgar_dir / "cik_mapping.json")
    log.info("  Facts dir    : %s", companyfacts_dir)
    log.info("Next step: run scripts/run_scan.py or scripts/run_backtest.py")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script is importable and the help text is correct**

Run: `python scripts/download_edgar_bulk.py --help 2>&1 || python scripts/download_edgar_bulk.py 2>&1 | head -5`

Expected: Should fail quickly with "EDGAR_USER_AGENT is not set" (proves the guard works without a real .env).

- [ ] **Step 3: Commit**

```bash
git add scripts/download_edgar_bulk.py
git commit -m "Add EDGAR bulk downloader: CIK mapping + companyfacts.zip"
```

---

### Task 3: Rewrite crucible/fetcher.py with EDGAR logic

**Files:**
- Modify: `crucible/fetcher.py` (full rewrite; keep `save_raw` and `_to_float` unchanged)

The new fetcher removes all FMP code and replaces it with:
1. An XBRL taxonomy map (our metric names → candidate XBRL tag list, ordered by preference)
2. `_load_cik_mapping` — loads `cik_mapping.json`, returns `{ticker: zero-padded-cik}`
3. `_parse_edgar_json` — reads a local CIK JSON, filters `form==10-K` facts by `filed <= as_of_date`
4. `_edgar_to_panel_rows` — converts parsed facts to long-panel format
5. `_edgar_to_info_row` — builds info row (sector via yfinance, currency USD, valuation from EDGAR + price)
6. Public functions: `fetch_sp500_tickers`, `fetch_financials`, `fetch_info`, `fetch_universe`, `save_raw`

**Key design notes:**
- EBITDA is not a direct XBRL concept. It is computed: Operating Income + D&A.
  `_edgar_to_panel_rows` adds raw `OperatingIncome` and `DepreciationAmortization` rows
  **in addition to** attempting to compute and output the final `EBITDA` metric.
- Free Cash Flow is computed: Operating Cash Flow − |Capital Expenditure|.
- Total Debt is computed: LongTermDebt (non-current) + ShortTermDebt (current portion).
- A single company may report the same concept under different XBRL tags across years.
  `_parse_edgar_json` tries each candidate tag in order and takes the first with data,
  then de-duplicates by fiscal year (keeps the record with the latest `filed` date per year).

- [ ] **Step 1: Write the new fetcher.py**

```python
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
# The first tag found with data for a given company/year wins.
# Annual 10-K only; quarterly concepts (fp != "FY") are excluded in _parse_edgar_json.
_XBRL_TAXONOMY: dict[str, list[str]] = {
    "Total Revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "RevenuesNetOfInterestExpense",  # banks
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
    "Invested Capital": [
        "InvestedCapital",  # rare direct XBRL; usually computed
    ],
}

# EDGAR CIK JSON facts that are reported in units other than USD
_NON_USD_UNITS: frozenset[str] = frozenset({"shares", "pure", "USD/shares"})

# ---------------------------------------------------------------------------
# CIK mapping
# ---------------------------------------------------------------------------


def _load_cik_mapping(
    mapping_path: Path = Path("data/raw/edgar/cik_mapping.json"),
) -> dict[str, str]:
    """Load SEC company_tickers.json, return {TICKER: zero-padded-10-digit-CIK}.

    The SEC mapping file has structure: {"0": {"cik_str": 320193, "ticker": "AAPL", ...}, ...}
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
    sorted newest fiscal year first. Only 10-K / 10-K/A annual (fp="FY") records with
    filed <= as_of_date are included. When multiple filings cover the same fiscal year
    (amendments), only the one with the latest filed date is kept.
    """
    padded_cik = cik.zfill(10)
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

            # Pick USD unit values (skip shares, pure, etc.)
            units: dict[str, list[dict]] = concept.get("units", {})
            usd_records: list[dict] = []
            for unit_key, records in units.items():
                if unit_key.upper() == "USD":
                    usd_records = records
                    break
            if not usd_records:
                continue

            # Filter: annual 10-K/10-K/A, filed <= as_of_date
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
                break  # first candidate with data wins

        if not best_records:
            continue

        # De-duplicate: one record per fiscal year — keep latest filed
        by_fy: dict[Any, dict[str, Any]] = {}
        for rec in best_records:
            fy = rec.get("fy") or rec["end"][:4]
            existing = by_fy.get(fy)
            if existing is None or rec["filed"] > existing["filed"]:
                by_fy[fy] = rec

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

    Computed metrics added here:
    - EBITDA  = Operating Income + Depreciation Amortization  (if both available)
    - Free Cash Flow = Operating Cash Flow - |Capital Expenditure|
    - Total Debt = Long Term Debt + Short Term Debt
    """
    rows: list[dict[str, Any]] = []

    # Raw concept → panel rows
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

    # Compute derived metrics per fiscal year
    # Build a quick lookup: {(metric, fy_str) → value}
    lookup: dict[tuple[str, str], float] = {}
    for row in rows:
        key = (row["metric"], row["fiscal_year"].strftime("%Y-%m-%d"))
        lookup[key] = row["value"]

    # Collect all fiscal years present
    fy_dates: set[str] = {row["fiscal_year"].strftime("%Y-%m-%d") for row in rows}

    for fy_str in fy_dates:
        fy_ts = pd.Timestamp(fy_str, tz="UTC")

        # EBITDA
        op_inc = lookup.get(("Operating Income", fy_str))
        da = lookup.get(("Depreciation Amortization", fy_str))
        if op_inc is not None and da is not None:
            rows.append({"ticker": ticker, "fiscal_year": fy_ts, "metric": "EBITDA", "value": op_inc + da})

        # Free Cash Flow
        ocf = lookup.get(("Operating Cash Flow", fy_str))
        capex = lookup.get(("Capital Expenditure", fy_str))
        if ocf is not None and capex is not None:
            rows.append({"ticker": ticker, "fiscal_year": fy_ts, "metric": "Free Cash Flow", "value": ocf - abs(capex)})

        # Total Debt
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
    """Build an info row from EDGAR facts + yfinance sector/price data.

    Sector and sub_industry come from yfinance (non-fundamental; acceptable).
    Valuation multiples (P/E, P/FCF, EV/EBITDA) are computed from:
      - market cap: yfinance fast_info.market_cap
      - fundamentals: most recent EDGAR record
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

    try:
        info = yf.Ticker(ticker).fast_info
        row["sector"] = getattr(info, "sector", None)
        row["sub_industry"] = getattr(info, "industry", None)
        mkt_cap = _to_float(getattr(info, "market_cap", None))
    except Exception:
        logger.debug("yfinance info unavailable for %s", ticker)
        mkt_cap = None

    def _latest(metric: str) -> float | None:
        records = facts.get(metric)
        return _to_float(records[0]["val"]) if records else None

    net_income = _latest("Net Income")
    fcf = _latest("Free Cash Flow")
    ebitda = _latest("EBITDA")
    debt = _latest("Total Debt") or 0.0
    cash = _to_float(facts.get("Cash And Cash Equivalents", [{}])[0].get("val") if facts.get("Cash And Cash Equivalents") else None) or 0.0

    if mkt_cap and net_income and net_income > 0:
        row["p_e"] = mkt_cap / net_income
    if mkt_cap and fcf and fcf > 0:
        row["p_fcf"] = mkt_cap / fcf
    if mkt_cap and ebitda and ebitda > 0:
        row["ev_ebitda"] = (mkt_cap + debt - cash) / ebitda

    return row


def _make_empty_info_row(ticker: str) -> dict[str, Any]:
    return {
        "ticker": ticker, "sector": None, "sub_industry": None, "currency": "USD",
        "p_e": None, "p_fcf": None, "ev_ebitda": None,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_sp500_tickers() -> list[str]:
    """Fetch the current S&P 500 constituent list from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        tables = pd.read_html(url, storage_options={"User-Agent": "Mozilla/5.0 Crucible"})
        tickers = list(tables[0]["Symbol"].str.replace(".", "-", regex=False))
        logger.info("Fetched %d S&P 500 tickers from Wikipedia", len(tickers))
        return tickers
    except Exception:
        logger.exception("Failed to fetch S&P 500 tickers from Wikipedia")
        raise


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

    Fundamentals come from EDGAR (point-in-time). Sector and market cap come
    from yfinance (current data — acceptable for non-fundamental fields).
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
        df["sector"].notna().sum(), len(tickers),
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
    Pass as_of_date for historical (backtest) scans; defaults to today (UTC).
    """
    if universe_id not in {"SP500", "RUSSELL1000", "RUSSELL3000"}:
        raise NotImplementedError(f"Universe {universe_id!r} not yet supported")

    if as_of_date is None:
        import datetime
        as_of_date = pd.Timestamp(datetime.datetime.now(timezone.utc))

    from datetime import datetime as _dt
    run_ts = _dt.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if tickers is None:
        tickers = fetch_sp500_tickers()

    cik_map = _load_cik_mapping()

    logger.info(
        "EDGAR fetch starting — %d tickers, as_of=%s",
        len(tickers), as_of_date.date(),
    )

    info_df = fetch_info(tickers, as_of_date, edgar_dir, cik_map)
    panel_df = fetch_financials(tickers, as_of_date, edgar_dir, cik_map)
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
```

- [ ] **Step 2: Run pytest and capture results**

Run: `cd /home/gabriel/Crucible && python -m pytest tests/test_fetcher.py -v 2>&1`

Expected: Most fetcher tests will FAIL because `_FMPCache`, `_raw_to_info_row`, `_raw_to_panel_rows`, `fetch_sp500_tickers` (old signature), `fetch_universe` (old signature) are all removed or changed. `test_save_raw_*` tests should still PASS since `save_raw` is unchanged.

- [ ] **Step 3: Run full test suite to see total breakage**

Run: `cd /home/gabriel/Crucible && python -m pytest tests/ -v 2>&1`

Expected: tests in `test_filters.py`, `test_scorer.py`, `test_validator.py`, `test_store.py`, `test_backtest.py` should be unaffected. Only `test_fetcher.py` tests that import removed FMP internals will fail.

- [ ] **Step 4: Commit**

```bash
git add crucible/fetcher.py
git commit -m "Rewrite fetcher.py: SEC EDGAR XBRL taxonomy + point-in-time parser"
```

---

## Self-Review Checklist

**Spec coverage:**
- ✓ Update docs (ROADMAP.md, .env.example, pyproject.toml) — Task 1
- ✓ CIK mapping download + save to `data/raw/edgar/cik_mapping.json` — Task 2
- ✓ `companyfacts.zip` streaming download with progress log — Task 2
- ✓ Extract CIK JSONs to `data/raw/edgar/companyfacts/` — Task 2
- ✓ `EDGAR_USER_AGENT` from env in headers — Task 2
- ✓ XBRL taxonomy dict at top of fetcher.py — Task 3
- ✓ `_parse_edgar_json(cik, as_of_date)` filtering on `filed <= as_of_date` — Task 3
- ✓ Run pytest and report breakage — Task 3 Step 2/3

**Type consistency:**
- `_parse_edgar_json` returns `dict[str, list[dict[str, Any]]]` — used correctly in `fetch_financials`, `fetch_info`
- `_load_cik_mapping` returns `dict[str, str]` — used correctly everywhere
- `save_raw` signature unchanged — same as before
- Panel columns: `ticker, fiscal_year, metric, value` — NOTE: old schema had a `statement` column (`income`/`balance`/`cashflow`). New schema drops it since EDGAR doesn't group that way naturally. This will break any downstream code relying on `statement`. **Acceptable per instructions** (do not fix downstream yet).

**Placeholder scan:** None found — all steps contain actual code.
