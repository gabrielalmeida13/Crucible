"""Feature engineering for the Phase 3a ML layer.

Extracts a 17-feature matrix and binary outperformance labels from
fund_by_date snapshots (same structure as run_backtest.py produces).

Label definition: 1 if ticker's 12m forward price return > S&P 500 12m
forward return at the same snapshot date, else 0. Rows where the forward
return cannot be computed (insufficient price history) are dropped.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_COLS: list[str] = [
    "roic_proxy_avg",
    "fcf_positive_years",
    "gross_margin_avg",
    "net_debt_ebitda",
    "revenue_growth_positive_years",
    "p_e",
    "p_fcf",
    "ev_ebitda",
    "momentum_raw",
    "interest_coverage",
    "cfo_to_ni",
    "capex_intensity",
    "operating_margin_trend",
    "revenue_acceleration",
    "share_buyback_signal",
    "insider_buy_ratio",
    "roic_direction",
]

_LABEL_HORIZON_MONTHS = 12
_BENCHMARK_COL = "SP500"


def add_roic_direction(
    fund_by_date: dict[pd.Timestamp, pd.DataFrame],
) -> None:
    """Add roic_direction column in-place: 1.0 if ROIC improved vs ~12m ago, else 0.0.

    NaN where either current or prior roic_proxy_avg is NaN, or no prior year exists.
    """
    sorted_dates = sorted(fund_by_date.keys())
    for i, date in enumerate(sorted_dates):
        df = fund_by_date[date]
        prior_date = _find_prior_year_date(sorted_dates, i)
        if prior_date is None:
            df["roic_direction"] = np.nan
            continue
        prior_df = fund_by_date[prior_date]
        current = pd.to_numeric(df["roic_proxy_avg"], errors="coerce")
        prior = pd.to_numeric(prior_df["roic_proxy_avg"].reindex(df.index), errors="coerce")
        direction = (current > prior).astype(float)
        direction[current.isna() | prior.isna()] = np.nan
        df["roic_direction"] = direction


def build_feature_matrix(
    fund_by_date: dict[pd.Timestamp, pd.DataFrame],
    prices: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    benchmark_col: str = _BENCHMARK_COL,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build (X, y) for snapshot dates in [start_date, end_date].

    X has MultiIndex (snapshot_date, ticker) and columns = FEATURE_COLS.
    y is binary: 1 if ticker 12m forward return > benchmark 12m forward return.

    Rows where the forward return cannot be computed are dropped from both X and y.
    Rows where the benchmark return is unavailable are also dropped.
    """
    add_roic_direction(fund_by_date)

    all_dates = sorted(fund_by_date.keys())
    window_dates = [d for d in all_dates if start_date <= d <= end_date]

    if not window_dates:
        return _empty_result()

    price_idx = prices.index
    x_parts: list[pd.DataFrame] = []
    y_parts: list[pd.Series] = []

    for date in window_dates:
        df = fund_by_date[date]

        pos = int(price_idx.searchsorted(date, side="right")) - 1
        exit_pos = pos + _LABEL_HORIZON_MONTHS
        if exit_pos >= len(price_idx) or pos < 0:
            continue
        if date not in price_idx:
            continue

        exit_date = price_idx[exit_pos]

        bench_ret = _safe_return(benchmark_col, date, exit_date, prices)
        if bench_ret is None:
            continue

        feat_rows: list[dict] = []
        label_rows: list[int] = []
        ticker_keys: list[tuple[pd.Timestamp, str]] = []

        for ticker in df.index:
            tkr_ret = _safe_return(ticker, date, exit_date, prices)
            if tkr_ret is None:
                continue

            row = df.loc[ticker]
            feat = {col: _scalar(row.get(col)) for col in FEATURE_COLS}
            feat_rows.append(feat)
            label_rows.append(1 if tkr_ret > bench_ret else 0)
            ticker_keys.append((date, ticker))

        if not feat_rows:
            continue

        idx = pd.MultiIndex.from_tuples(ticker_keys, names=["snapshot_date", "ticker"])
        x_parts.append(pd.DataFrame(feat_rows, index=idx))
        y_parts.append(pd.Series(label_rows, index=idx, name="label", dtype=int))

    if not x_parts:
        return _empty_result()

    X = pd.concat(x_parts).astype(float)
    y = pd.concat(y_parts)

    logger.info(
        "Feature matrix: %d rows, %d features, label balance %.1f%%",
        len(X), len(FEATURE_COLS), float(y.mean() * 100),
    )
    return X, y


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _empty_result() -> tuple[pd.DataFrame, pd.Series]:
    empty_idx = pd.MultiIndex.from_tuples([], names=["snapshot_date", "ticker"])
    X = pd.DataFrame(columns=FEATURE_COLS, index=empty_idx)
    y = pd.Series(dtype=int, name="label", index=empty_idx)
    return X, y


def _find_prior_year_date(
    sorted_dates: list[pd.Timestamp],
    current_idx: int,
    target_months: int = 12,
    tolerance_months: int = 3,
) -> pd.Timestamp | None:
    """Return the snapshot date closest to 12 months before sorted_dates[current_idx]."""
    if current_idx == 0:
        return None
    current = sorted_dates[current_idx]
    target = current - pd.DateOffset(months=target_months)
    best: pd.Timestamp | None = None
    best_delta = pd.Timedelta(days=tolerance_months * 31)
    for d in sorted_dates[:current_idx]:
        delta = abs(d - target)
        if delta < best_delta:
            best_delta = delta
            best = d
    return best


def _safe_return(
    ticker: str,
    t0: pd.Timestamp,
    t1: pd.Timestamp,
    prices: pd.DataFrame,
) -> float | None:
    """Simple price return; None if ticker missing or prices zero/NaN."""
    if ticker not in prices.columns:
        return None
    try:
        p0 = prices.at[t0, ticker]
        p1 = prices.at[t1, ticker]
    except KeyError:
        return None
    if pd.isna(p0) or pd.isna(p1) or float(p0) <= 0:
        return None
    return float(p1) / float(p0) - 1.0


def _scalar(val: object) -> float:
    """Coerce a scalar to float; NaN for missing."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    try:
        return float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return np.nan
