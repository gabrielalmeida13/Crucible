# EDGAR Wiring and Test Cleanup Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove all FMP references from `run_backtest.py`, wire it to EDGAR + yfinance, and rewrite `test_fetcher.py` with full coverage of the EDGAR parser — getting the test suite back to 100% green.

**Architecture:** `run_backtest.py` is a standalone script; it builds monthly snapshot DataFrames by calling `fetch_financials` per date (reads local EDGAR JSON files), then computes ROIC/FCF/leverage metrics in-script from the panel. `test_fetcher.py` tests the four new public/private surfaces using synthetic EDGAR JSON files written to `tmp_path` — no real HTTP calls, no real EDGAR data needed.

**Tech Stack:** `pytest`, `pandas`, `yfinance` (prices only), `json` (synthetic test fixtures)

---

## File map

| File | Action | Reason |
|------|--------|--------|
| `scripts/run_backtest.py` | Rewrite | Remove FMP imports/logic; use EDGAR + yfinance |
| `tests/test_fetcher.py` | Rewrite | FMP internals gone; test new EDGAR surfaces |

---

### Task 1: Rewrite `scripts/run_backtest.py`

**Files:**
- Modify: `scripts/run_backtest.py` (full rewrite)

The new script:
- Removes `_FMPCache`, `_cached_get`, API key, `requests.Session`, all `_fetch_*` FMP helpers
- Uses `_load_cik_mapping` + `fetch_financials` for fundamentals (local EDGAR files)
- Uses `yfinance.download` for prices
- Adds `_linear_slope` helper (pure computation, same formula as before)
- Adds `_pivot_panel(ticker, panel)` → `{metric: Series[fiscal_year]}`
- Adds `_compute_snapshot_row(ticker, pivoted)` → dict with the 14 columns `backtest.py` expects
- `_build_fundamentals_by_date` calls `fetch_financials` once per monthly date (point-in-time)
- `main()` checks for EDGAR data directory, loads CIK map, fetches prices, builds snapshots, runs backtest

Column mapping (FMP → EDGAR):

| Column | Old source | New computation |
|--------|-----------|-----------------|
| `roic_proxy_avg` | FMP `returnOnInvestedCapital` | avg(Net Income / (Total Equity + Total Debt)) per year |
| `fcf_positive_years` | FMP `freeCashFlowPerShare > 0` | count years with `Free Cash Flow > 0` |
| `fcf_latest` | FMP latest FCF/share | latest `Free Cash Flow` absolute value |
| `net_debt_ebitda` | FMP `netDebtToEBITDA` | (Total Debt − Cash) / EBITDA (latest year) |
| `revenue_growth_positive_years` | FMP `revenuePerShare` YoY | count YoY increases in `Total Revenue` |
| `gross_margin_*` | FMP `grossProfitMargin` | Gross Profit / Total Revenue per year |
| `data_years` | len(km_at) | len(Total Revenue records) |
| `sector`, `sub_industry` | FMP profile | `None` (not needed for Layer 1 filters or backtest) |
| `currency` | FMP profile | `"USD"` (all tickers are US stocks) |
| `p_e`, `p_fcf`, `ev_ebitda` | FMP period-end ratios | `None` (Layer 2 scoring — will revisit in 2.3) |

- [ ] **Step 1: Write the new `run_backtest.py`**

```python
#!/usr/bin/env python3
"""Mini backtest: 15 large-cap S&P 500 tickers, Jan 2024 → Mar 2025.

Data sources
------------
  SEC EDGAR companyfacts/{CIK}.json  — fundamentals (point-in-time via filed date)
  yfinance                           — price data only

Requirements
------------
  Run scripts/download_edgar_bulk.py once to populate data/raw/edgar/ before running.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from crucible.backtest import BacktestConfig, generate_report, run_backtest, run_sensitivity
from crucible.config import CrucibleConfig, FilterThresholds
from crucible.fetcher import _load_cik_mapping, _to_float, fetch_financials

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

TICKERS: list[str] = [
    "MSFT", "AAPL", "GOOGL", "META", "V",
    "JNJ",  "PG",   "JPM",   "UNH",  "HD",
    "LLY",  "AVGO", "COST",  "PEP",  "KO",
]

BACKTEST_START    = pd.Timestamp("2023-01-31", tz="UTC")
BACKTEST_END      = pd.Timestamp("2025-03-31", tz="UTC")
PRICE_FETCH_START = "2022-06-01"
PRICE_FETCH_END   = "2026-03-31"

TRAIN_MONTHS  = 12
TOP_N         = 10
REPORT_PATH   = ROOT / "data" / "backtest_report.md"
EDGAR_DIR     = ROOT / "data" / "raw" / "edgar" / "companyfacts"
CIK_MAP_PATH  = ROOT / "data" / "raw" / "edgar" / "cik_mapping.json"

# Relaxed thresholds because EDGAR data covers 2009+ but early backtest dates
# may have fewer than 5 years of available filings for some tickers.
_MINI_FILTERS = FilterThresholds(
    roic_min=0.15,
    fcf_positive_min_years=3,
    fcf_lookback_years=5,
    net_debt_ebitda_max=3.0,
    revenue_growth_positive_min_years=2,
    revenue_growth_lookback_years=5,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _linear_slope(values: list[float]) -> float | None:
    """OLS slope for a list of evenly-spaced values."""
    n = len(values)
    if n < 2:
        return None
    mean_x = (n - 1) / 2.0
    mean_y = sum(values) / n
    num = sum((i - mean_x) * (values[i] - mean_y) for i in range(n))
    den = sum((i - mean_x) ** 2 for i in range(n))
    return num / den if den else 0.0


def _pivot_panel(ticker: str, panel: pd.DataFrame) -> dict[str, pd.Series]:
    """Pivot the long panel for one ticker to {metric: Series[fiscal_year, sorted oldest→newest]}."""
    sub = panel[panel["ticker"] == ticker]
    if sub.empty:
        return {}
    pivoted: dict[str, pd.Series] = {}
    for metric, grp in sub.groupby("metric"):
        pivoted[metric] = grp.set_index("fiscal_year")["value"].sort_index()
    return pivoted


def _compute_snapshot_row(ticker: str, pivoted: dict[str, pd.Series]) -> dict:
    """Derive the 14-column processed row that apply_filters() and score() expect."""
    rev    = pivoted.get("Total Revenue",           pd.Series(dtype=float))
    gp     = pivoted.get("Gross Profit",            pd.Series(dtype=float))
    ni     = pivoted.get("Net Income",              pd.Series(dtype=float))
    fcf    = pivoted.get("Free Cash Flow",          pd.Series(dtype=float))
    td     = pivoted.get("Total Debt",              pd.Series(dtype=float))
    cash   = pivoted.get("Cash And Cash Equivalents", pd.Series(dtype=float))
    ebitda = pivoted.get("EBITDA",                  pd.Series(dtype=float))
    equity = pivoted.get("Total Equity",            pd.Series(dtype=float))

    data_years = len(rev)

    # ROIC proxy = Net Income / (Total Equity + Total Debt), averaged across years
    roic_vals: list[float] = []
    for fy in ni.index:
        ni_v  = _to_float(ni.get(fy))
        eq_v  = _to_float(equity.get(fy))
        td_v  = _to_float(td.get(fy))
        if ni_v is not None and eq_v is not None and td_v is not None:
            denom = eq_v + td_v
            if denom > 0:
                roic_vals.append(ni_v / denom)
    roic_avg = sum(roic_vals) / len(roic_vals) if roic_vals else None

    # FCF consistency
    fcf_vals   = [_to_float(v) for v in fcf.values]
    fcf_pos    = float(sum(1 for v in fcf_vals if v is not None and v > 0))
    fcf_latest = _to_float(fcf.iloc[-1]) if not fcf.empty else None

    # Net Debt / EBITDA (latest year only)
    td_last  = _to_float(td.iloc[-1])    if not td.empty    else None
    ca_last  = _to_float(cash.iloc[-1])  if not cash.empty  else None
    eb_last  = _to_float(ebitda.iloc[-1]) if not ebitda.empty else None
    nd_eb: float | None = None
    if td_last is not None and ca_last is not None and eb_last and eb_last > 0:
        nd_eb = (td_last - ca_last) / eb_last

    # Revenue growth: count YoY increases
    rev_vals = [_to_float(v) for v in rev.values]
    rev_vals = [v for v in rev_vals if v is not None]
    rev_growth = float(
        sum(1 for i in range(1, len(rev_vals)) if rev_vals[i] > rev_vals[i - 1])
    ) if len(rev_vals) >= 2 else None

    # Gross margins per year
    gm_vals: list[float] = []
    for fy in rev.index:
        r = _to_float(rev.get(fy))
        g = _to_float(gp.get(fy))
        if r and r > 0 and g is not None:
            gm_vals.append(g / r)

    return {
        "ticker":                        ticker,
        "sector":                        None,
        "sub_industry":                  None,
        "currency":                      "USD",
        "p_e":                           None,
        "p_fcf":                         None,
        "ev_ebitda":                     None,
        "data_years":                    data_years,
        "insufficient_data":             data_years < 3,
        "roic_proxy_avg":                roic_avg,
        "fcf_latest":                    fcf_latest,
        "fcf_positive_years":            fcf_pos if not fcf.empty else None,
        "net_debt_ebitda":               nd_eb,
        "revenue_growth_positive_years": rev_growth,
        "gross_margin_latest":           gm_vals[-1] if gm_vals else None,
        "gross_margin_avg":              sum(gm_vals) / len(gm_vals) if gm_vals else None,
        "gross_margin_trend_slope":      _linear_slope(gm_vals),
    }


def _build_fundamentals_by_date(
    tickers: list[str],
    monthly_dates: pd.DatetimeIndex,
    edgar_dir: Path,
    cik_map: dict[str, str],
) -> dict[pd.Timestamp, pd.DataFrame]:
    """Build a point-in-time snapshot DataFrame for each monthly date."""
    result: dict[pd.Timestamp, pd.DataFrame] = {}
    for date in monthly_dates:
        panel = fetch_financials(tickers, date, edgar_dir, cik_map)
        rows = [_compute_snapshot_row(t, _pivot_panel(t, panel)) for t in tickers]
        df = pd.DataFrame(rows).set_index("ticker")
        n_ok = int((~df["insufficient_data"]).sum())
        log.debug("%s  %d/%d tickers with sufficient data", date.date(), n_ok, len(tickers))
        result[date] = df
    return result


def _fetch_prices(ticker: str, label: str, start: str, end: str) -> pd.Series:
    """Monthly close prices from yfinance, resampled to month-end."""
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            log.warning("No price history for %s", ticker)
            return pd.Series(dtype=float, name=label)
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return close.resample("ME").last().rename(label)
    except Exception:
        log.warning("Price fetch failed for %s", ticker, exc_info=True)
        return pd.Series(dtype=float, name=label)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if not CIK_MAP_PATH.exists():
        log.error(
            "CIK mapping not found at %s. Run scripts/download_edgar_bulk.py first.",
            CIK_MAP_PATH,
        )
        sys.exit(1)
    if not EDGAR_DIR.exists():
        log.error(
            "EDGAR companyfacts directory not found at %s. "
            "Run scripts/download_edgar_bulk.py first.",
            EDGAR_DIR,
        )
        sys.exit(1)

    cik_map = _load_cik_mapping(CIK_MAP_PATH)

    log.info(
        "Mini Backtest | %d tickers | train %s – %s | test from %s",
        len(TICKERS),
        BACKTEST_START.date(),
        (BACKTEST_START + pd.DateOffset(months=TRAIN_MONTHS - 1)).date(),
        (BACKTEST_START + pd.DateOffset(months=TRAIN_MONTHS)).date(),
    )

    # ── Step 1: Fetch prices ─────────────────────────────────────────────────
    raw_price_series: list[pd.Series] = []
    for ticker in TICKERS:
        log.info("Fetching prices for %s …", ticker)
        s = _fetch_prices(ticker, ticker, PRICE_FETCH_START, PRICE_FETCH_END)
        if not s.empty:
            raw_price_series.append(s)

    log.info("Fetching SPY (benchmark) …")
    spy = _fetch_prices("SPY", "SP500", PRICE_FETCH_START, PRICE_FETCH_END)
    if not spy.empty:
        raw_price_series.append(spy)

    if not raw_price_series:
        log.error("No price data fetched.")
        sys.exit(1)

    prices = pd.concat(raw_price_series, axis=1)
    if prices.index.tz is None:
        prices.index = prices.index.tz_localize("UTC")
    log.info("Price matrix: %d month-end rows × %d series", len(prices), len(prices.columns))

    # ── Step 2: Build monthly snapshots from EDGAR ───────────────────────────
    monthly_dates = pd.date_range(BACKTEST_START, BACKTEST_END, freq="ME", tz="UTC")
    log.info("Building %d monthly snapshots from EDGAR …", len(monthly_dates))
    fund_by_date = _build_fundamentals_by_date(TICKERS, monthly_dates, EDGAR_DIR, cik_map)

    # Log first test month
    first_test_idx = TRAIN_MONTHS
    if first_test_idx < len(monthly_dates):
        snap = fund_by_date[monthly_dates[first_test_idx]]
        n_ok = int((~snap["insufficient_data"]).sum())
        log.info(
            "First test month (%s): %d/%d tickers with sufficient data",
            monthly_dates[first_test_idx].date(), n_ok, len(TICKERS),
        )

    # ── Step 3: Walk-forward backtest ────────────────────────────────────────
    config = CrucibleConfig(account_currency="USD", filters=_MINI_FILTERS)
    bt_cfg = BacktestConfig(
        train_months=TRAIN_MONTHS, top_n=TOP_N,
        holding_months=1, hit_rate_months=12,
        risk_free_annual=0.04, benchmark_col="SP500",
    )
    result = run_backtest(fund_by_date, prices, config, bt_cfg)
    log.info("Backtest complete — %d test months", len(result.monthly_results))

    if not result.monthly_results:
        log.error("No test results. Check EDGAR data coverage and price range.")
        sys.exit(1)

    # ── Step 4: Sensitivity analysis ─────────────────────────────────────────
    sensitivity = run_sensitivity(
        fund_by_date, prices, config, bt_cfg,
        roic_thresholds=(0.10, 0.12, 0.15, 0.18, 0.20),
    )

    # ── Step 5: Report ────────────────────────────────────────────────────────
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    generate_report(result, sensitivity, REPORT_PATH, config)
    log.info("Report saved → %s", REPORT_PATH)
    print("\n" + "═" * 70)
    print(REPORT_PATH.read_text())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script imports cleanly**

Run: `uv run python -c "import scripts.run_backtest" 2>&1 || uv run python -c "import importlib.util, sys; spec = importlib.util.spec_from_file_location('rb', 'scripts/run_backtest.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)" 2>&1 | head -10`

Expected: No ImportError.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_backtest.py
git commit -m "Rewrite run_backtest.py: remove FMP, wire EDGAR + yfinance prices"
```

---

### Task 2: Rewrite `tests/test_fetcher.py`

**Files:**
- Modify: `tests/test_fetcher.py` (full rewrite)

Tests are organised into five groups:
1. `_load_cik_mapping` — CIK JSON loading and ticker normalisation
2. `_parse_edgar_json` — taxonomy fallback, point-in-time filter, dedup, edge cases
3. `_edgar_to_panel_rows` — derived metrics (EBITDA, FCF, Total Debt)
4. `fetch_financials` — end-to-end panel using tmp_path synthetic files
5. `save_raw` — unchanged interface, same tests as before

All tests use synthetic JSON files written to `tmp_path` or inline dicts. No real HTTP calls.

- [ ] **Step 1: Write the new `tests/test_fetcher.py`**

```python
"""Unit tests for fetcher.py — no real HTTP calls, no EDGAR bulk data required."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from crucible.fetcher import (
    _edgar_to_panel_rows,
    _load_cik_mapping,
    _parse_edgar_json,
    fetch_financials,
    fetch_universe,
    save_raw,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_cik_json(cik: int, concepts: dict[str, list[dict[str, Any]]]) -> dict:
    """Build a minimal EDGAR CIK JSON structure for testing."""
    return {
        "cik": cik,
        "entityName": "Test Corp",
        "facts": {
            "us-gaap": {
                name: {"units": {"USD": records}}
                for name, records in concepts.items()
            }
        },
    }


def _rec(
    end: str,
    val: float,
    filed: str,
    fy: int,
    form: str = "10-K",
    fp: str = "FY",
) -> dict:
    """Build one EDGAR fact record."""
    return {"end": end, "val": val, "filed": filed, "fy": fy, "form": form, "fp": fp}


def _write_cik_json(
    tmp_path: Path, cik: int, concepts: dict[str, list[dict[str, Any]]]
) -> Path:
    padded = str(cik).zfill(10)
    p = tmp_path / f"CIK{padded}.json"
    p.write_text(json.dumps(_make_cik_json(cik, concepts)))
    return p


def _write_cik_mapping(tmp_path: Path, entries: list[dict]) -> Path:
    """Write a SEC-format company_tickers.json to tmp_path."""
    mapping = {str(i): e for i, e in enumerate(entries)}
    p = tmp_path / "cik_mapping.json"
    p.write_text(json.dumps(mapping))
    return p


# ---------------------------------------------------------------------------
# Fixtures for save_raw (interface unchanged)
# ---------------------------------------------------------------------------


@pytest.fixture()
def tickers() -> list[str]:
    return ["AAPL", "MSFT", "GOOGL"]


@pytest.fixture()
def info_df(tickers: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sector": ["Technology"] * 3,
            "sub_industry": ["Consumer Electronics", "Systems Software", "Internet"],
            "currency": ["USD"] * 3,
            "p_e": [25.0, 30.0, 22.0],
            "p_fcf": [20.0, 25.0, 18.0],
            "ev_ebitda": [15.0, 18.0, 12.0],
        },
        index=pd.Index(tickers, name="ticker"),
    )


@pytest.fixture()
def panel_df(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        for year in ["2021-12-31", "2022-12-31", "2023-12-31"]:
            for metric, value in [
                ("Total Revenue", 1e10),
                ("Gross Profit", 4e9),
                ("Net Income", 2e9),
                ("EBITDA", 3e9),
                ("Total Assets", 2e11),
                ("Current Liabilities", 5e10),
                ("Total Debt", 3e10),
                ("Cash And Cash Equivalents", 1e10),
                ("Operating Cash Flow", 2.5e9),
                ("Capital Expenditure", 5e8),
                ("Free Cash Flow", 2.0e9),
            ]:
                rows.append({
                    "ticker": ticker,
                    "fiscal_year": pd.Timestamp(year, tz="UTC"),
                    "metric": metric,
                    "value": float(value),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# _load_cik_mapping
# ---------------------------------------------------------------------------


def test_load_cik_mapping_returns_ticker_to_padded_cik(tmp_path: Path) -> None:
    """CIK integers must be zero-padded to 10 digits."""
    p = _write_cik_mapping(tmp_path, [{"cik_str": 320193, "ticker": "AAPL", "title": "Apple"}])
    result = _load_cik_mapping(p)
    assert result["AAPL"] == "0000320193"


def test_load_cik_mapping_uppercases_ticker(tmp_path: Path) -> None:
    """Tickers in the SEC file may be lowercase — must be normalised to uppercase."""
    p = _write_cik_mapping(tmp_path, [{"cik_str": 789019, "ticker": "msft", "title": "Microsoft"}])
    result = _load_cik_mapping(p)
    assert "MSFT" in result
    assert result["MSFT"] == "0000789019"


def test_load_cik_mapping_multiple_tickers(tmp_path: Path) -> None:
    """All entries in the mapping file must be loaded."""
    p = _write_cik_mapping(
        tmp_path,
        [
            {"cik_str": 320193, "ticker": "AAPL", "title": "Apple"},
            {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft"},
        ],
    )
    result = _load_cik_mapping(p)
    assert len(result) == 2
    assert "AAPL" in result
    assert "MSFT" in result


def test_load_cik_mapping_missing_file_raises(tmp_path: Path) -> None:
    """FileNotFoundError must be raised when the mapping file does not exist."""
    with pytest.raises(FileNotFoundError, match="CIK mapping not found"):
        _load_cik_mapping(tmp_path / "nonexistent.json")


# ---------------------------------------------------------------------------
# _parse_edgar_json — taxonomy fallback
# ---------------------------------------------------------------------------


def test_parse_edgar_json_finds_revenue_from_primary_tag(tmp_path: Path) -> None:
    """Total Revenue should be found under RevenueFromContractWithCustomerExcludingAssessedTax."""
    _write_cik_json(tmp_path, 320193, {
        "RevenueFromContractWithCustomerExcludingAssessedTax": [
            _rec("2022-09-24", 394328e6, "2022-10-28", 2022)
        ]
    })
    facts = _parse_edgar_json("0000320193", pd.Timestamp("2023-01-01", tz="UTC"), tmp_path)
    assert "Total Revenue" in facts
    assert facts["Total Revenue"][0]["val"] == 394328e6


def test_parse_edgar_json_falls_back_to_revenues_tag(tmp_path: Path) -> None:
    """If the primary tag is absent, the secondary tag (Revenues) must be used."""
    _write_cik_json(tmp_path, 320193, {
        "Revenues": [_rec("2022-09-24", 394328e6, "2022-10-28", 2022)]
    })
    facts = _parse_edgar_json("0000320193", pd.Timestamp("2023-01-01", tz="UTC"), tmp_path)
    assert "Total Revenue" in facts


def test_parse_edgar_json_primary_tag_wins_over_secondary(tmp_path: Path) -> None:
    """When both primary and secondary tags exist, the primary value is used."""
    _write_cik_json(tmp_path, 320193, {
        "RevenueFromContractWithCustomerExcludingAssessedTax": [
            _rec("2022-09-24", 100e9, "2022-10-28", 2022)
        ],
        "Revenues": [
            _rec("2022-09-24", 999e9, "2022-10-28", 2022)
        ],
    })
    facts = _parse_edgar_json("0000320193", pd.Timestamp("2023-01-01", tz="UTC"), tmp_path)
    assert facts["Total Revenue"][0]["val"] == pytest.approx(100e9)


# ---------------------------------------------------------------------------
# _parse_edgar_json — point-in-time integrity
# ---------------------------------------------------------------------------


def test_parse_edgar_json_excludes_filings_after_as_of(tmp_path: Path) -> None:
    """Facts with filed > as_of_date must be excluded."""
    _write_cik_json(tmp_path, 320193, {
        "Revenues": [
            _rec("2022-09-24", 394328e6, "2022-10-28", 2022),  # filed before
            _rec("2023-09-30", 383285e6, "2023-11-02", 2023),  # filed after
        ]
    })
    facts = _parse_edgar_json(
        "0000320193", pd.Timestamp("2023-01-01", tz="UTC"), tmp_path
    )
    assert len(facts.get("Total Revenue", [])) == 1
    assert facts["Total Revenue"][0]["fy"] == 2022


def test_parse_edgar_json_includes_filing_on_exact_as_of_date(tmp_path: Path) -> None:
    """A filing with filed == as_of_date must be included (boundary is inclusive)."""
    _write_cik_json(tmp_path, 320193, {
        "Revenues": [_rec("2022-09-24", 394328e6, "2022-10-28", 2022)]
    })
    facts = _parse_edgar_json(
        "0000320193", pd.Timestamp("2022-10-28", tz="UTC"), tmp_path
    )
    assert "Total Revenue" in facts
    assert len(facts["Total Revenue"]) == 1


def test_parse_edgar_json_all_future_filings_returns_empty(tmp_path: Path) -> None:
    """If all filings are in the future, the metric must not appear in results."""
    _write_cik_json(tmp_path, 320193, {
        "Revenues": [_rec("2023-09-30", 383285e6, "2023-11-02", 2023)]
    })
    facts = _parse_edgar_json(
        "0000320193", pd.Timestamp("2022-01-01", tz="UTC"), tmp_path
    )
    assert "Total Revenue" not in facts


# ---------------------------------------------------------------------------
# _parse_edgar_json — deduplication
# ---------------------------------------------------------------------------


def test_parse_edgar_json_dedup_keeps_latest_amendment(tmp_path: Path) -> None:
    """When 10-K and 10-K/A exist for the same fiscal year, keep the latest filed."""
    _write_cik_json(tmp_path, 320193, {
        "Revenues": [
            {"end": "2022-09-24", "val": 394328e6, "filed": "2022-10-28",
             "fy": 2022, "form": "10-K", "fp": "FY"},
            {"end": "2022-09-24", "val": 394500e6, "filed": "2022-12-01",
             "fy": 2022, "form": "10-K/A", "fp": "FY"},
        ]
    })
    facts = _parse_edgar_json(
        "0000320193", pd.Timestamp("2023-01-01", tz="UTC"), tmp_path
    )
    assert len(facts["Total Revenue"]) == 1
    assert facts["Total Revenue"][0]["val"] == pytest.approx(394500e6)


def test_parse_edgar_json_dedup_excludes_amendment_filed_after_as_of(tmp_path: Path) -> None:
    """If the 10-K/A was filed after as_of_date, only the original 10-K is used."""
    _write_cik_json(tmp_path, 320193, {
        "Revenues": [
            {"end": "2022-09-24", "val": 394328e6, "filed": "2022-10-28",
             "fy": 2022, "form": "10-K", "fp": "FY"},
            {"end": "2022-09-24", "val": 394500e6, "filed": "2023-03-15",
             "fy": 2022, "form": "10-K/A", "fp": "FY"},  # after as_of
        ]
    })
    facts = _parse_edgar_json(
        "0000320193", pd.Timestamp("2023-01-01", tz="UTC"), tmp_path
    )
    assert len(facts["Total Revenue"]) == 1
    assert facts["Total Revenue"][0]["val"] == pytest.approx(394328e6)  # original


# ---------------------------------------------------------------------------
# _parse_edgar_json — other edge cases
# ---------------------------------------------------------------------------


def test_parse_edgar_json_excludes_quarterly_forms(tmp_path: Path) -> None:
    """10-Q records must be excluded even if filed before as_of_date."""
    _write_cik_json(tmp_path, 320193, {
        "Revenues": [
            {"end": "2023-06-30", "val": 94760e6, "filed": "2023-08-01",
             "fy": 2023, "form": "10-Q", "fp": "Q3"},
            {"end": "2022-09-24", "val": 394328e6, "filed": "2022-10-28",
             "fy": 2022, "form": "10-K", "fp": "FY"},
        ]
    })
    facts = _parse_edgar_json(
        "0000320193", pd.Timestamp("2023-09-01", tz="UTC"), tmp_path
    )
    assert len(facts["Total Revenue"]) == 1
    assert facts["Total Revenue"][0]["form"] == "10-K"


def test_parse_edgar_json_missing_cik_file_returns_empty_dict(tmp_path: Path) -> None:
    """Missing CIK JSON must return empty dict without raising."""
    facts = _parse_edgar_json(
        "9999999999", pd.Timestamp("2023-01-01", tz="UTC"), tmp_path
    )
    assert facts == {}


def test_parse_edgar_json_results_sorted_newest_first(tmp_path: Path) -> None:
    """Multiple fiscal years must be sorted newest end date first."""
    _write_cik_json(tmp_path, 320193, {
        "Revenues": [
            _rec("2021-09-25", 365817e6, "2021-10-29", 2021),
            _rec("2022-09-24", 394328e6, "2022-10-28", 2022),
            _rec("2023-09-30", 383285e6, "2023-11-02", 2023),
        ]
    })
    facts = _parse_edgar_json(
        "0000320193", pd.Timestamp("2024-01-01", tz="UTC"), tmp_path
    )
    ends = [r["end"] for r in facts["Total Revenue"]]
    assert ends == sorted(ends, reverse=True)


# ---------------------------------------------------------------------------
# _edgar_to_panel_rows
# ---------------------------------------------------------------------------


def test_edgar_to_panel_rows_raw_metrics_present() -> None:
    """Every metric in the facts dict must appear as panel rows."""
    facts = {
        "Total Revenue": [
            {"end": "2022-09-24", "val": 394328e6, "filed": "2022-10-28", "fy": 2022}
        ],
        "Net Income": [
            {"end": "2022-09-24", "val": 99803e6, "filed": "2022-10-28", "fy": 2022}
        ],
    }
    rows = _edgar_to_panel_rows("AAPL", facts)
    metrics = {r["metric"] for r in rows}
    assert "Total Revenue" in metrics
    assert "Net Income" in metrics


def test_edgar_to_panel_rows_ebitda_computed_from_op_income_and_da() -> None:
    """EBITDA = Operating Income + Depreciation Amortization."""
    facts = {
        "Operating Income": [
            {"end": "2022-12-31", "val": 100e9, "filed": "2023-02-01", "fy": 2022}
        ],
        "Depreciation Amortization": [
            {"end": "2022-12-31", "val": 10e9, "filed": "2023-02-01", "fy": 2022}
        ],
    }
    rows = _edgar_to_panel_rows("TEST", facts)
    ebitda_rows = [r for r in rows if r["metric"] == "EBITDA"]
    assert len(ebitda_rows) == 1
    assert ebitda_rows[0]["value"] == pytest.approx(110e9)


def test_edgar_to_panel_rows_fcf_equals_ocf_minus_capex() -> None:
    """Free Cash Flow = Operating Cash Flow − |Capital Expenditure|."""
    facts = {
        "Operating Cash Flow": [
            {"end": "2022-12-31", "val": 90e9, "filed": "2023-02-01", "fy": 2022}
        ],
        "Capital Expenditure": [
            {"end": "2022-12-31", "val": 15e9, "filed": "2023-02-01", "fy": 2022}
        ],
    }
    rows = _edgar_to_panel_rows("TEST", facts)
    fcf_rows = [r for r in rows if r["metric"] == "Free Cash Flow"]
    assert len(fcf_rows) == 1
    assert fcf_rows[0]["value"] == pytest.approx(75e9)


def test_edgar_to_panel_rows_total_debt_sums_ltd_and_std() -> None:
    """Total Debt = Long Term Debt + Short Term Debt."""
    facts = {
        "Long Term Debt": [
            {"end": "2022-12-31", "val": 50e9, "filed": "2023-02-01", "fy": 2022}
        ],
        "Short Term Debt": [
            {"end": "2022-12-31", "val": 5e9, "filed": "2023-02-01", "fy": 2022}
        ],
    }
    rows = _edgar_to_panel_rows("TEST", facts)
    td_rows = [r for r in rows if r["metric"] == "Total Debt"]
    assert len(td_rows) == 1
    assert td_rows[0]["value"] == pytest.approx(55e9)


def test_edgar_to_panel_rows_total_debt_with_only_ltd() -> None:
    """Total Debt must still be computed when only Long Term Debt is present."""
    facts = {
        "Long Term Debt": [
            {"end": "2022-12-31", "val": 50e9, "filed": "2023-02-01", "fy": 2022}
        ],
    }
    rows = _edgar_to_panel_rows("TEST", facts)
    td_rows = [r for r in rows if r["metric"] == "Total Debt"]
    assert len(td_rows) == 1
    assert td_rows[0]["value"] == pytest.approx(50e9)


def test_edgar_to_panel_rows_no_ebitda_when_da_missing() -> None:
    """EBITDA must NOT appear if Depreciation Amortization is absent."""
    facts = {
        "Operating Income": [
            {"end": "2022-12-31", "val": 100e9, "filed": "2023-02-01", "fy": 2022}
        ],
    }
    rows = _edgar_to_panel_rows("TEST", facts)
    metrics = {r["metric"] for r in rows}
    assert "EBITDA" not in metrics


def test_edgar_to_panel_rows_fiscal_year_is_utc_timestamp() -> None:
    """fiscal_year column must be UTC-aware Timestamps."""
    facts = {
        "Total Revenue": [
            {"end": "2022-09-24", "val": 394328e6, "filed": "2022-10-28", "fy": 2022}
        ],
    }
    rows = _edgar_to_panel_rows("AAPL", facts)
    for row in rows:
        ts = row["fiscal_year"]
        assert isinstance(ts, pd.Timestamp)
        assert ts.tzinfo is not None


def test_edgar_to_panel_rows_empty_facts_returns_empty_list() -> None:
    """Empty facts dict must produce an empty list."""
    assert _edgar_to_panel_rows("EMPTY", {}) == []


# ---------------------------------------------------------------------------
# fetch_financials
# ---------------------------------------------------------------------------


def test_fetch_financials_returns_dataframe_with_correct_columns(tmp_path: Path) -> None:
    """fetch_financials must return a DataFrame with the four panel columns."""
    mapping_path = _write_cik_mapping(
        tmp_path, [{"cik_str": 320193, "ticker": "AAPL", "title": "Apple"}]
    )
    _write_cik_json(tmp_path, 320193, {
        "Revenues": [_rec("2022-09-24", 394328e6, "2022-10-28", 2022)]
    })
    cik_map = _load_cik_mapping(mapping_path)
    result = fetch_financials(
        ["AAPL"], pd.Timestamp("2023-01-01", tz="UTC"), tmp_path, cik_map
    )
    assert isinstance(result, pd.DataFrame)
    for col in ("ticker", "fiscal_year", "metric", "value"):
        assert col in result.columns, f"Missing column: {col}"


def test_fetch_financials_point_in_time_only_past_filings(tmp_path: Path) -> None:
    """Calling with an early as_of_date must exclude later filings."""
    mapping_path = _write_cik_mapping(
        tmp_path, [{"cik_str": 320193, "ticker": "AAPL", "title": "Apple"}]
    )
    _write_cik_json(tmp_path, 320193, {
        "Revenues": [
            _rec("2022-09-24", 394328e6, "2022-10-28", 2022),
            _rec("2023-09-30", 383285e6, "2023-11-02", 2023),
        ]
    })
    cik_map = _load_cik_mapping(mapping_path)

    early = fetch_financials(
        ["AAPL"], pd.Timestamp("2023-01-01", tz="UTC"), tmp_path, cik_map
    )
    late = fetch_financials(
        ["AAPL"], pd.Timestamp("2024-01-01", tz="UTC"), tmp_path, cik_map
    )

    early_rev = early[early["metric"] == "Total Revenue"]
    late_rev  = late[late["metric"] == "Total Revenue"]
    assert len(early_rev) == 1
    assert len(late_rev)  == 2


def test_fetch_financials_skips_ticker_with_no_cik(tmp_path: Path) -> None:
    """Tickers absent from the CIK mapping must produce no rows."""
    cik_map: dict[str, str] = {}  # empty mapping
    result = fetch_financials(
        ["UNKNOWN"], pd.Timestamp("2023-01-01", tz="UTC"), tmp_path, cik_map
    )
    assert result.empty


def test_fetch_financials_handles_multiple_tickers(tmp_path: Path) -> None:
    """Results for all tickers must be combined into one DataFrame."""
    mapping_path = _write_cik_mapping(
        tmp_path,
        [
            {"cik_str": 320193, "ticker": "AAPL", "title": "Apple"},
            {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft"},
        ],
    )
    for cik in (320193, 789019):
        _write_cik_json(tmp_path, cik, {
            "Revenues": [_rec("2022-12-31", 100e9, "2023-02-01", 2022)]
        })
    cik_map = _load_cik_mapping(mapping_path)
    result = fetch_financials(
        ["AAPL", "MSFT"], pd.Timestamp("2023-06-01", tz="UTC"), tmp_path, cik_map
    )
    assert set(result["ticker"].unique()) == {"AAPL", "MSFT"}


def test_fetch_financials_fiscal_year_is_utc(tmp_path: Path) -> None:
    """fiscal_year values in the panel must be UTC-aware Timestamps."""
    mapping_path = _write_cik_mapping(
        tmp_path, [{"cik_str": 320193, "ticker": "AAPL", "title": "Apple"}]
    )
    _write_cik_json(tmp_path, 320193, {
        "Revenues": [_rec("2022-09-24", 394328e6, "2022-10-28", 2022)]
    })
    cik_map = _load_cik_mapping(mapping_path)
    result = fetch_financials(
        ["AAPL"], pd.Timestamp("2023-01-01", tz="UTC"), tmp_path, cik_map
    )
    assert not result.empty
    assert result["fiscal_year"].dt.tz is not None


# ---------------------------------------------------------------------------
# save_raw — interface unchanged, same contract as before
# ---------------------------------------------------------------------------


def test_save_raw_creates_three_parquet_files(
    tmp_path: Path, tickers: list[str], info_df: pd.DataFrame, panel_df: pd.DataFrame
) -> None:
    """save_raw must create all three Parquet files."""
    paths = save_raw(tickers, info_df, panel_df, tmp_path, "20240101T000000Z")
    assert len(paths) == 3
    for p in paths:
        assert Path(p).exists(), f"Missing: {p}"


def test_save_raw_parquet_round_trip(
    tmp_path: Path, tickers: list[str], info_df: pd.DataFrame, panel_df: pd.DataFrame
) -> None:
    """Data written by save_raw must survive a Parquet round-trip."""
    run_ts = "20240101T000000Z"
    tickers_path, info_path, panel_path = save_raw(
        tickers, info_df, panel_df, tmp_path, run_ts
    )
    assert list(pd.read_parquet(tickers_path)["ticker"]) == tickers
    assert list(pd.read_parquet(info_path).index) == tickers
    assert len(pd.read_parquet(panel_path)) == len(panel_df)


def test_save_raw_empty_panel_writes_schema_only(
    tmp_path: Path, tickers: list[str], info_df: pd.DataFrame
) -> None:
    """save_raw must handle an empty panel without error and write a schema-only file."""
    empty = pd.DataFrame(columns=["ticker", "fiscal_year", "metric", "value"])
    _, _, panel_path = save_raw(tickers, info_df, empty, tmp_path, "20240101T000000Z")
    assert pd.read_parquet(panel_path).empty


def test_save_raw_creates_directory_if_missing(
    tmp_path: Path, tickers: list[str], info_df: pd.DataFrame, panel_df: pd.DataFrame
) -> None:
    """save_raw must create raw_dir if it does not exist."""
    nested = tmp_path / "a" / "b" / "raw"
    save_raw(tickers, info_df, panel_df, nested, "20240101T000000Z")
    assert nested.exists()


# ---------------------------------------------------------------------------
# fetch_universe — error-path tests only (no bulk EDGAR data needed)
# ---------------------------------------------------------------------------


def test_fetch_universe_unsupported_universe_raises() -> None:
    """fetch_universe must raise NotImplementedError for non-US universes."""
    with pytest.raises(NotImplementedError, match="EUROPE_LARGE"):
        fetch_universe("EUROPE_LARGE", Path("/tmp"))


def test_fetch_universe_russell3000_is_supported_symbol() -> None:
    """RUSSELL3000 is a supported universe ID — it must raise FileNotFoundError
    (CIK mapping absent), not NotImplementedError."""
    with pytest.raises(FileNotFoundError, match="CIK mapping not found"):
        fetch_universe(
            "RUSSELL3000",
            Path("/tmp"),
            tickers=["AAPL"],
            as_of_date=pd.Timestamp("2023-01-01", tz="UTC"),
        )
```

- [ ] **Step 2: Run the new test file in isolation**

Run: `uv run python -m pytest tests/test_fetcher.py -v 2>&1`

Expected: All tests pass. If any fail, fix them before continuing.

- [ ] **Step 3: Run the full test suite**

Run: `uv run python -m pytest tests/ -v 2>&1`

Expected: All tests pass (142 non-fetcher + new fetcher tests).

- [ ] **Step 4: Commit**

```bash
git add tests/test_fetcher.py
git commit -m "Rewrite test_fetcher.py: EDGAR taxonomy, point-in-time, dedup tests"
```

---

## Self-Review

**Spec coverage:**
- ✓ Remove FMP imports (`_FMPCache`, `_cached_get`, API key) from `run_backtest.py` — Task 1
- ✓ Wire backtest to EDGAR via `fetch_financials` + `_load_cik_mapping` — Task 1
- ✓ Use yfinance for price data in `run_backtest.py` — Task 1
- ✓ XBRL taxonomy mapping tests (primary and fallback tags) — Task 2
- ✓ Point-in-time integrity tests (`filed > as_of_date` excluded; boundary inclusive) — Task 2
- ✓ Deduplication tests (10-K/A wins; but not if its own filed is after as_of_date) — Task 2
- ✓ `save_raw` tests (unchanged interface) — Task 2

**Placeholder scan:** None — all steps have complete code.

**Type consistency:**
- `_load_cik_mapping(path) → dict[str, str]` — used consistently in tests and run_backtest.py
- `_parse_edgar_json(cik, as_of_date, edgar_dir) → dict[str, list[dict]]` — consistent
- `_edgar_to_panel_rows(ticker, facts) → list[dict]` — consistent
- `fetch_financials(tickers, as_of_date, edgar_dir, cik_map) → pd.DataFrame` — consistent
- `save_raw(tickers, info_df, panel_df, raw_dir, run_ts) → tuple[Path, Path, Path]` — consistent
- Panel columns: `ticker, fiscal_year, metric, value` — consistent throughout (old `statement` column removed)

**Note on `fetch_universe` test:** `test_fetch_universe_russell3000_is_supported_symbol` calls `fetch_universe("RUSSELL3000", ...)` without a `cik_mapping_path` override. The function uses the default path `data/raw/edgar/cik_mapping.json` which won't exist in CI/test environments. Since `_load_cik_mapping` is called before any network I/O, it raises `FileNotFoundError` immediately. This is the correct observed behavior for an environment without EDGAR data.
