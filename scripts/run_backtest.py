#!/usr/bin/env python3
"""Mini backtest: 15 large-cap S&P 500 tickers, Jan 2024 → Mar 2025.

Data sources
------------
  SEC EDGAR companyfacts/{CIK}.json  — fundamentals (point-in-time via filed date)
  yfinance                           — price data only

Point-in-time integrity
-----------------------
  EDGAR filing records are filtered to filed <= as_of_date.
  No filing-lag estimate is needed; the filed date is the exact date
  the SEC received the document.

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

# Relaxed thresholds: early backtest dates have fewer than 5 years of EDGAR filings.
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
# Snapshot helpers
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
    """Pivot the long panel for one ticker to {metric: Series[fiscal_year, oldest→newest]}."""
    sub = panel[panel["ticker"] == ticker]
    if sub.empty:
        return {}
    return {
        metric: grp.set_index("fiscal_year")["value"].sort_index()
        for metric, grp in sub.groupby("metric")
    }


def _compute_snapshot_row(ticker: str, pivoted: dict[str, pd.Series]) -> dict:
    """Derive the 14-column processed row that apply_filters() and score() expect."""
    rev    = pivoted.get("Total Revenue",             pd.Series(dtype=float))
    gp     = pivoted.get("Gross Profit",              pd.Series(dtype=float))
    ni     = pivoted.get("Net Income",                pd.Series(dtype=float))
    fcf    = pivoted.get("Free Cash Flow",            pd.Series(dtype=float))
    td     = pivoted.get("Total Debt",                pd.Series(dtype=float))
    cash   = pivoted.get("Cash And Cash Equivalents", pd.Series(dtype=float))
    ebitda = pivoted.get("EBITDA",                    pd.Series(dtype=float))
    equity = pivoted.get("Total Equity",              pd.Series(dtype=float))

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
    td_last = _to_float(td.iloc[-1])     if not td.empty     else None
    ca_last = _to_float(cash.iloc[-1])   if not cash.empty   else None
    eb_last = _to_float(ebitda.iloc[-1]) if not ebitda.empty else None
    nd_eb: float | None = None
    if td_last is not None and ca_last is not None and eb_last and eb_last > 0:
        nd_eb = (td_last - ca_last) / eb_last

    # Revenue growth: count YoY increases
    rev_vals = [_to_float(v) for v in rev.values if _to_float(v) is not None]
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
    """Build a point-in-time fundamentals snapshot for each monthly date."""
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
