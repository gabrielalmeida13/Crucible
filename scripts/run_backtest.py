#!/usr/bin/env python3
"""Mini backtest: 15 large-cap S&P 500 tickers, Jan 2024 → Mar 2025.

Data sources (FMP free tier)
-----------------------------
  /stable/profile              — sector, currency, industry
  /stable/key-metrics          — ROIC, net-debt/EBITDA, EV/EBITDA (pre-computed)
  /stable/ratios               — gross margin, P/FCF, P/E, revenue/FCF per share
  /stable/historical-price-eod/light — daily closing prices

Walk-forward design
-------------------
  Training   : 12 months Jan 2023 – Dec 2023 (calibration — no trades)
  Test window: Jan 2024 – Mar 2025 (15 monthly steps)
  Portfolio  : equal-weighted top-10 by composite_score
  Benchmark  : SPY (S&P 500 proxy)

Point-in-time integrity
-----------------------
  key-metrics and ratios rows are filtered by date + 90-day filing lag
  before being used at each monthly snapshot.  The 90-day lag is a
  conservative bound on SEC 10-K filing latency.

Filter threshold adjustments for 5-year data depth
----------------------------------------------------
  The FMP free tier returns at most 5 annual periods (2021–2025).
  At a Jan 2024 test date, only fiscal-year 2021–2023 statements are
  available (3 years).  Default thresholds assume 5-year history, so
  two are relaxed in the mini-backtest config:
    fcf_positive_min_years        : 4 → 3
    revenue_growth_positive_min_years : 3 → 2

API cost: ~62 FMP requests on first run (≤5 per ticker × 15 + SPY prices).
All responses are cached with 7-day (prices) or 30-day (fundamentals) TTLs.
REQUIRES: CRUCIBLE_FMP_API_KEY set in .env
"""

from __future__ import annotations

import logging
import math
import os
import sys
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from crucible.backtest import BacktestConfig, generate_report, run_backtest, run_sensitivity
from crucible.config import CrucibleConfig, FilterThresholds
from crucible.fetcher import _FMPCache, _cached_get, _to_float

# ---------------------------------------------------------------------------
# Mini backtest parameters
# ---------------------------------------------------------------------------

TICKERS: list[str] = [
    "MSFT", "AAPL", "GOOGL", "META", "V",
    "JNJ",  "PG",   "JPM",   "UNH",  "HD",
    "LLY",  "AVGO", "COST",  "PEP",  "KO",
]

BACKTEST_START    = pd.Timestamp("2023-01-31", tz="UTC")  # training window starts
BACKTEST_END      = pd.Timestamp("2025-03-31", tz="UTC")  # last test month
PRICE_FETCH_START = "2022-06-01"  # extra cushion before training window
PRICE_FETCH_END   = "2026-03-31"  # covers 12-month hit-rate lookforward from Mar 2025

TRAIN_MONTHS  = 12   # Jan 2023 – Dec 2023 (first test = Jan 2024)
TOP_N         = 10
REPORT_PATH   = ROOT / "data" / "backtest_report.md"
CACHE_PATH    = ROOT / "data" / "fmp_cache.db"

# Relaxed thresholds for 5-year data depth (see module docstring).
_MINI_FILTERS = FilterThresholds(
    roic_min=0.15,
    fcf_positive_min_years=3,             # max possible with 3 years at first test date
    fcf_lookback_years=5,
    net_debt_ebitda_max=3.0,
    revenue_growth_positive_min_years=2,  # 3 data points → only 2 YoY periods
    revenue_growth_lookback_years=5,
)

# 90-day conservative filing lag: use period-end date + 90 days as availability proxy
_FILING_LAG = pd.Timedelta(days=90)

_TTL_FUND  = 30 * 86_400   # 30 days — fundamental filings don't change
_TTL_PRICE =  7 * 86_400   # 7 days  — prices are updated daily

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data fetchers
# ---------------------------------------------------------------------------


def _fetch_profile(
    ticker: str, api_key: str, cache: _FMPCache, session: requests.Session
) -> dict:
    data = _cached_get(
        session, cache, f"bt_profile:{ticker}", _TTL_FUND,
        "/profile", api_key, {"symbol": ticker},
    )
    if isinstance(data, list) and data:
        return data[0]
    return {}


def _fetch_key_metrics(
    ticker: str, api_key: str, cache: _FMPCache, session: requests.Session
) -> list[dict]:
    data = _cached_get(
        session, cache, f"bt_km:{ticker}", _TTL_FUND,
        "/key-metrics", api_key,
        {"symbol": ticker, "period": "annual", "limit": 5},
    )
    return data if isinstance(data, list) else []


def _fetch_ratios(
    ticker: str, api_key: str, cache: _FMPCache, session: requests.Session
) -> list[dict]:
    data = _cached_get(
        session, cache, f"bt_ratios:{ticker}", _TTL_FUND,
        "/ratios", api_key,
        {"symbol": ticker, "period": "annual", "limit": 5},
    )
    return data if isinstance(data, list) else []


def _fetch_prices(
    label: str,
    fmp_ticker: str,
    start: str,
    end: str,
    api_key: str,
    cache: _FMPCache,
    session: requests.Session,
) -> pd.Series:
    """Fetch daily close prices resampled to month-end. Returns Series named `label`."""
    data = _cached_get(
        session, cache,
        f"bt_price:{label}:{start}:{end}",
        _TTL_PRICE,
        "/historical-price-eod/light",
        api_key,
        {"symbol": fmp_ticker, "from": start, "to": end},
    )
    rows = (data if isinstance(data, list) else []) or []
    if not rows:
        log.warning("No price history returned for %s (%s)", label, fmp_ticker)
        return pd.Series(dtype=float, name=label)
    df = pd.DataFrame(rows)[["date", "price"]]
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return (
        df.set_index("date")
        .sort_index()["price"]
        .resample("ME")
        .last()
        .rename(label)
    )


# ---------------------------------------------------------------------------
# Point-in-time snapshot construction
# ---------------------------------------------------------------------------


def _rows_available_at(rows: list[dict], as_of: pd.Timestamp) -> list[dict]:
    """Return only rows whose filing date <= as_of, sorted newest-period first.

    Tries the `filingDate` field first; falls back to `date` + 90-day lag.
    """
    available: list[dict] = []
    for r in rows:
        filing_raw = r.get("filingDate")
        try:
            if filing_raw:
                filing_dt = pd.Timestamp(filing_raw, tz="UTC")
            else:
                period_end = r.get("date")
                if not period_end:
                    continue
                filing_dt = pd.Timestamp(period_end, tz="UTC") + _FILING_LAG
        except Exception:
            continue
        if filing_dt <= as_of:
            available.append(r)
    available.sort(key=lambda r: r.get("date", ""), reverse=True)
    return available


def _linear_slope(values: list[float]) -> float | None:
    """Simple OLS slope for a list of evenly-spaced values."""
    n = len(values)
    if n < 2:
        return None
    x = list(range(n))
    mean_x = (n - 1) / 2.0
    mean_y = sum(values) / n
    num = sum((x[i] - mean_x) * (values[i] - mean_y) for i in range(n))
    den = sum((x[i] - mean_x) ** 2 for i in range(n))
    return num / den if den else 0.0


def _build_snapshot_row(
    ticker: str,
    profile: dict,
    km_rows: list[dict],
    rat_rows: list[dict],
    as_of: pd.Timestamp,
) -> dict:
    """Build one processed-fundamentals row using pre-computed FMP ratios.

    Only rows whose filing date <= as_of are used (point-in-time).
    Valuation multiples (P/FCF, EV/EBITDA) come from the period-end ratios —
    they are computed at the fiscal-year-end price, which is point-in-time
    relative to the filing date.
    """
    km_at  = _rows_available_at(km_rows,  as_of)
    rat_at = _rows_available_at(rat_rows, as_of)

    data_years = len(km_at)
    latest_km  = km_at[0]  if km_at  else {}
    latest_rat = rat_at[0] if rat_at else {}

    # ROIC proxy average across available years
    roic_vals = [_to_float(r.get("returnOnInvestedCapital")) for r in km_at]
    roic_vals = [v for v in roic_vals if v is not None]
    roic_avg  = sum(roic_vals) / len(roic_vals) if roic_vals else None

    # FCF positive years (out of available years)
    fcf_per_share = [_to_float(r.get("freeCashFlowPerShare")) for r in rat_at]
    fcf_positive  = float(sum(1 for v in fcf_per_share if v is not None and v > 0))
    fcf_latest_ps = _to_float(latest_rat.get("freeCashFlowPerShare"))

    # Net Debt / EBITDA — latest available
    nd_ebitda = _to_float(latest_km.get("netDebtToEBITDA"))

    # Revenue growth: count YoY positive changes in revenuePerShare
    rev_series = sorted(
        [(r.get("date", ""), _to_float(r.get("revenuePerShare"))) for r in rat_at],
        key=lambda t: t[0],
    )
    rev_vals = [v for _, v in rev_series if v is not None]
    rev_growth_positive = float(
        sum(1 for i in range(1, len(rev_vals)) if rev_vals[i] > rev_vals[i - 1])
    ) if len(rev_vals) >= 2 else None

    # Gross margin across available years (sorted oldest → newest for trend)
    gm_series = sorted(
        [(r.get("date", ""), _to_float(r.get("grossProfitMargin"))) for r in rat_at],
        key=lambda t: t[0],
    )
    gm_vals = [v for _, v in gm_series if v is not None]
    gm_latest = gm_vals[-1] if gm_vals else None
    gm_avg    = sum(gm_vals) / len(gm_vals) if gm_vals else None
    gm_slope  = _linear_slope(gm_vals)

    # Valuation multiples from pre-computed period-end ratios
    p_fcf    = _to_float(latest_rat.get("priceToFreeCashFlowRatio"))
    ev_ebitda = _to_float(latest_rat.get("enterpriseValueMultiple")) \
                or _to_float(latest_km.get("evToEBITDA"))
    p_e      = _to_float(latest_rat.get("priceToEarningsRatio"))

    return {
        "ticker":                        ticker,
        "sector":                        profile.get("sector") or None,
        "sub_industry":                  profile.get("industry") or None,
        "currency":                      profile.get("currency") or "USD",
        "p_e":                           p_e,
        "p_fcf":                         p_fcf,
        "ev_ebitda":                     ev_ebitda,
        "data_years":                    int(data_years),
        "insufficient_data":             data_years < 3,
        "roic_proxy_avg":                roic_avg,
        "fcf_latest":                    fcf_latest_ps,
        "fcf_positive_years":            fcf_positive if rat_at else None,
        "net_debt_ebitda":               nd_ebitda,
        "revenue_growth_positive_years": rev_growth_positive,
        "gross_margin_latest":           gm_latest,
        "gross_margin_avg":              gm_avg,
        "gross_margin_trend_slope":      gm_slope,
    }


def _build_fundamentals_by_date(
    tickers: list[str],
    profiles: dict[str, dict],
    key_metrics: dict[str, list[dict]],
    ratios: dict[str, list[dict]],
    monthly_dates: pd.DatetimeIndex,
) -> dict[pd.Timestamp, pd.DataFrame]:
    """Build a point-in-time fundamentals snapshot DataFrame for each monthly date."""
    result: dict[pd.Timestamp, pd.DataFrame] = {}
    for date in monthly_dates:
        rows = [
            _build_snapshot_row(
                ticker,
                profiles.get(ticker, {}),
                key_metrics.get(ticker, []),
                ratios.get(ticker, []),
                date,
            )
            for ticker in tickers
        ]
        df = pd.DataFrame(rows).set_index("ticker")
        n_ok = int((~df["insufficient_data"]).sum())
        log.debug("%s  %d/%d tickers with sufficient data", date.date(), n_ok, len(tickers))
        result[date] = df
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    api_key = os.getenv("CRUCIBLE_FMP_API_KEY", "")
    if not api_key:
        log.error("CRUCIBLE_FMP_API_KEY is not set in .env — aborting.")
        sys.exit(1)

    cache   = _FMPCache(CACHE_PATH)
    session = requests.Session()
    session.headers.update({"User-Agent": "Crucible-Backtest/1.0"})

    log.info(
        "Mini Backtest  |  %d tickers  |  train %s – %s  |  test from %s",
        len(TICKERS),
        BACKTEST_START.date(),
        (BACKTEST_START + pd.DateOffset(months=TRAIN_MONTHS - 1)).date(),
        (BACKTEST_START + pd.DateOffset(months=TRAIN_MONTHS)).date(),
    )
    log.info("FMP requests used today so far: %d / 250", cache.requests_today())

    # ── Step 1: Fetch fundamentals and prices ────────────────────────────────
    profiles: dict[str, dict]          = {}
    key_metrics: dict[str, list[dict]] = {}
    ratios: dict[str, list[dict]]      = {}
    raw_price_series: list[pd.Series]  = []

    for ticker in TICKERS:
        log.info("Fetching %s …", ticker)
        try:
            profiles[ticker]    = _fetch_profile(ticker, api_key, cache, session)
            key_metrics[ticker] = _fetch_key_metrics(ticker, api_key, cache, session)
            ratios[ticker]      = _fetch_ratios(ticker, api_key, cache, session)
            s = _fetch_prices(
                ticker, ticker,
                PRICE_FETCH_START, PRICE_FETCH_END,
                api_key, cache, session,
            )
            if not s.empty:
                raw_price_series.append(s)
        except Exception:
            log.warning("Data fetch failed for %s — excluding", ticker, exc_info=True)

    log.info("Fetching SPY (S&P 500 benchmark proxy) …")
    try:
        spy = _fetch_prices(
            "SP500", "SPY",
            PRICE_FETCH_START, PRICE_FETCH_END,
            api_key, cache, session,
        )
        if not spy.empty:
            raw_price_series.append(spy)
    except Exception:
        log.error("Failed to fetch SPY prices — benchmark will be missing", exc_info=True)

    log.info("FMP requests used today after fetch: %d / 250", cache.requests_today())

    if not raw_price_series:
        log.error("No price data fetched — check API key and network.")
        sys.exit(1)

    prices = pd.concat(raw_price_series, axis=1)
    if prices.index.tz is None:
        prices.index = prices.index.tz_localize("UTC")
    else:
        prices.index = prices.index.tz_convert("UTC")
    log.info("Price matrix: %d month-end rows × %d series", len(prices), len(prices.columns))

    # Log data coverage sanity check
    tickers_with_km = [t for t in TICKERS if key_metrics.get(t)]
    log.info(
        "Fundamental data: %d/%d tickers have key-metrics rows",
        len(tickers_with_km), len(TICKERS),
    )

    # ── Step 2: Build monthly point-in-time fundamentals snapshots ───────────
    monthly_dates = pd.date_range(BACKTEST_START, BACKTEST_END, freq="ME", tz="UTC")
    log.info("Building %d monthly snapshots …", len(monthly_dates))
    fund_by_date = _build_fundamentals_by_date(
        TICKERS, profiles, key_metrics, ratios, monthly_dates
    )

    # Log first test month data sufficiency
    first_test_idx = TRAIN_MONTHS
    if first_test_idx < len(monthly_dates):
        first_test_dt = monthly_dates[first_test_idx]
        snap = fund_by_date[first_test_dt]
        n_sufficient = int((~snap["insufficient_data"]).sum())
        log.info(
            "First test month (%s): %d/%d tickers with sufficient data",
            first_test_dt.date(), n_sufficient, len(TICKERS),
        )
        # Show who passes filters
        from crucible.filters import apply_filters
        try:
            passed = apply_filters(snap, _MINI_FILTERS)
            log.info(
                "Filter pass rate at first test month: %d/%d tickers",
                len(passed), n_sufficient,
            )
        except Exception:
            log.warning("Could not preview filter pass rate", exc_info=True)

    # ── Step 3: Walk-forward backtest ────────────────────────────────────────
    config = CrucibleConfig(
        account_currency="USD",
        filters=_MINI_FILTERS,
    )
    bt_cfg = BacktestConfig(
        train_months=TRAIN_MONTHS,
        top_n=TOP_N,
        holding_months=1,
        hit_rate_months=12,
        risk_free_annual=0.04,
        benchmark_col="SP500",
    )

    log.info(
        "Running walk-forward backtest (train=%d months, top_n=%d, holding=1 month) …",
        TRAIN_MONTHS, TOP_N,
    )
    result = run_backtest(fund_by_date, prices, config, bt_cfg)
    log.info("Backtest complete — %d test months", len(result.monthly_results))

    if not result.monthly_results:
        log.error(
            "No test results produced. Likely cause: all tickers failed filters or "
            "prices don't cover the test window far enough forward."
        )
        sys.exit(1)

    # ── Step 4: ROIC sensitivity analysis ────────────────────────────────────
    log.info("Running ROIC sensitivity analysis …")
    sensitivity = run_sensitivity(
        fund_by_date, prices, config, bt_cfg,
        roic_thresholds=(0.10, 0.12, 0.15, 0.18, 0.20),
    )

    # ── Step 5: Generate report ───────────────────────────────────────────────
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    generate_report(result, sensitivity, REPORT_PATH, config)
    log.info("Report saved → %s", REPORT_PATH)

    print("\n" + "═" * 70)
    print(REPORT_PATH.read_text())


if __name__ == "__main__":
    main()
