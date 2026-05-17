"""Cleaning, normalization, missing value detection, and Pandera validation."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from crucible.validator import PROCESSED_COLUMNS, ProcessedFundamentalsSchema

logger = logging.getLogger(__name__)

_MIN_DATA_YEARS = 3

# Outlier bounds — values outside these are data errors, replaced with NaN and logged
_ROIC_BOUNDS = (-10.0, 50.0)
_GROSS_MARGIN_BOUNDS = (-1.0, 1.0)
_NET_DEBT_EBITDA_BOUNDS = (-100.0, 1000.0)

# yfinance metric name candidates (first non-null match is used)
_REVENUE_METRICS = ("Total Revenue", "Operating Revenue")
_CASH_METRICS = (
    "Cash And Cash Equivalents",
    "Cash Cash Equivalents And Short Term Investments",
    "Cash And Short Term Investments",
)
_EBITDA_METRICS = ("EBITDA", "Normalized EBITDA")
_FCF_METRICS = ("Free Cash Flow",)


def load_raw(
    raw_dir: Path, run_ts: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three raw Parquet files for a given run timestamp.

    Returns (tickers_df, info_df, panel_df).
    """
    tickers_df = pd.read_parquet(raw_dir / f"sp500_tickers_{run_ts}.parquet")
    info_df = pd.read_parquet(raw_dir / f"sp500_info_{run_ts}.parquet")
    panel_df = pd.read_parquet(raw_dir / f"sp500_panel_{run_ts}.parquet")
    return tickers_df, info_df, panel_df


def clean(raw_dir: Path, run_ts: str, processed_dir: Path) -> pd.DataFrame:
    """Load raw files, derive metrics, validate schema, save to processed_dir.

    Raises pandera.errors.SchemaErrors if the processed DataFrame fails validation.
    Never passes dirty data downstream silently.
    """
    tickers_df, info_df, panel_df = load_raw(raw_dir, run_ts)
    logger.info(
        "Loaded raw — %d tickers, %d panel rows", len(tickers_df), len(panel_df)
    )

    processed = _merge_and_derive(info_df, panel_df)
    processed = _detect_outliers(processed)
    processed = _flag_insufficient_data(processed)
    processed = processed[PROCESSED_COLUMNS]

    n_insufficient = int(processed["insufficient_data"].sum())
    logger.info(
        "Derived metrics for %d tickers; %d flagged as insufficient data",
        len(processed),
        n_insufficient,
    )

    ProcessedFundamentalsSchema.validate(processed, lazy=True)

    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / f"sp500_{run_ts}.parquet"
    processed.to_parquet(out_path)
    logger.info("Saved processed file: %s", out_path)

    return processed


# ---------------------------------------------------------------------------
# Internal derivation helpers
# ---------------------------------------------------------------------------


def _merge_and_derive(info_df: pd.DataFrame, panel_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot the panel and compute all derived fundamental metrics."""
    if panel_df.empty:
        logger.warning("Panel is empty — all derived metrics will be NaN")
        result = info_df.copy()
        for col in _DERIVED_COLS:
            result[col] = np.nan
        result["data_years"] = 0
        return result

    income_panel = _pivot_statement(panel_df, "income")
    balance_panel = _pivot_statement(panel_df, "balance")
    cashflow_panel = _pivot_statement(panel_df, "cashflow")

    records: list[dict] = []
    for ticker in info_df.index:
        inc = _extract_ticker(income_panel, ticker)
        bal = _extract_ticker(balance_panel, ticker)
        cf = _extract_ticker(cashflow_panel, ticker)
        records.append(_derive_ticker_metrics(ticker, inc, bal, cf))

    derived_df = pd.DataFrame(records).set_index("ticker")
    return info_df.join(derived_df, how="left")


def _pivot_statement(panel_df: pd.DataFrame, statement: str) -> pd.DataFrame:
    """Pivot one statement type to MultiIndex(ticker, fiscal_year) × metric."""
    sub = panel_df[panel_df["statement"] == statement]
    if sub.empty:
        return pd.DataFrame()
    return sub.pivot_table(
        index=["ticker", "fiscal_year"],
        columns="metric",
        values="value",
        aggfunc="first",
    )


def _extract_ticker(panel: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Return a single-ticker slice (fiscal_year × metric) from a pivoted panel."""
    if panel.empty:
        return pd.DataFrame()
    try:
        return panel.xs(ticker, level="ticker")
    except KeyError:
        return pd.DataFrame()


def _derive_ticker_metrics(
    ticker: str,
    inc: pd.DataFrame,
    bal: pd.DataFrame,
    cf: pd.DataFrame,
) -> dict:
    """Compute all derived metrics for a single ticker."""
    rec: dict = {"ticker": ticker}

    years = inc.index.sort_values() if not inc.empty else pd.DatetimeIndex([])
    rec["data_years"] = len(years)

    if len(years) == 0:
        for col in _DERIVED_COLS:
            rec[col] = np.nan
        return rec

    # ROIC proxy = Net Income / (Total Assets − Current Liabilities)
    net_income = _col(inc, "Net Income")
    total_assets = _col(bal, "Total Assets")
    current_liab = _col(bal, "Current Liabilities")
    invested_capital = total_assets - current_liab
    roic = (net_income / invested_capital).replace([np.inf, -np.inf], np.nan)
    rec["roic_proxy_avg"] = _nanmean(roic)

    # FCF = Operating CF + CapEx (CapEx is negative in yfinance)
    direct_fcf = _col_first(cf, *_FCF_METRICS)
    op_cf = _col(cf, "Operating Cash Flow")
    capex = _col(cf, "Capital Expenditure")
    fcf = (
        direct_fcf
        if not direct_fcf.isna().all()
        else (op_cf + capex).replace([np.inf, -np.inf], np.nan)
    )
    latest_fcf = fcf.dropna()
    rec["fcf_latest"] = float(latest_fcf.iloc[-1]) if not latest_fcf.empty else np.nan
    rec["fcf_positive_years"] = float((fcf > 0).sum()) if not fcf.isna().all() else np.nan

    # Net Debt / EBITDA (most recent year)
    total_debt = _col(bal, "Total Debt")
    cash = _col_first(bal, *_CASH_METRICS)
    ebitda = _col_first(inc, *_EBITDA_METRICS)
    net_debt = total_debt - cash
    nd_ebitda = (net_debt / ebitda).replace([np.inf, -np.inf], np.nan).dropna()
    rec["net_debt_ebitda"] = float(nd_ebitda.iloc[-1]) if not nd_ebitda.empty else np.nan

    # Revenue growth: count of years with positive YoY growth
    revenue = _col_first(inc, *_REVENUE_METRICS)
    rev_growth = revenue.pct_change().replace([np.inf, -np.inf], np.nan)
    rec["revenue_growth_positive_years"] = (
        float((rev_growth > 0).sum()) if not rev_growth.isna().all() else np.nan
    )

    # Gross margin
    gross_profit = _col(inc, "Gross Profit")
    gm = (gross_profit / revenue).replace([np.inf, -np.inf], np.nan)
    gm_valid = gm.dropna()
    rec["gross_margin_latest"] = float(gm_valid.iloc[-1]) if not gm_valid.empty else np.nan
    rec["gross_margin_avg"] = _nanmean(gm)
    rec["gross_margin_trend_slope"] = (
        _linear_slope(gm) if gm_valid.shape[0] >= 2 else np.nan
    )

    return rec


_DERIVED_COLS = [
    "roic_proxy_avg",
    "fcf_latest",
    "fcf_positive_years",
    "net_debt_ebitda",
    "revenue_growth_positive_years",
    "gross_margin_latest",
    "gross_margin_avg",
    "gross_margin_trend_slope",
]


def _detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Replace statistically implausible values with NaN and log each replacement."""
    df = df.copy()

    def _clamp(col: str, lo: float, hi: float) -> None:
        if col not in df.columns:
            return
        mask = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
        n = int(mask.sum())
        if n:
            logger.warning(
                "Outlier: %d values in '%s' outside [%s, %s] → NaN", n, col, lo, hi
            )
            df.loc[mask, col] = np.nan

    _clamp("roic_proxy_avg", *_ROIC_BOUNDS)
    _clamp("gross_margin_latest", *_GROSS_MARGIN_BOUNDS)
    _clamp("gross_margin_avg", *_GROSS_MARGIN_BOUNDS)
    _clamp("net_debt_ebitda", *_NET_DEBT_EBITDA_BOUNDS)
    return df


def _flag_insufficient_data(df: pd.DataFrame) -> pd.DataFrame:
    """Set insufficient_data=True for tickers with fewer than _MIN_DATA_YEARS years."""
    df = df.copy()
    df["insufficient_data"] = df["data_years"].fillna(0).astype(int) < _MIN_DATA_YEARS
    return df


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Return a float Series for metric `name`, or empty Series if absent."""
    if df.empty or name not in df.columns:
        return pd.Series(dtype=float)
    return df[name].astype(float).sort_index()


def _col_first(df: pd.DataFrame, *names: str) -> pd.Series:
    """Return the first non-empty metric Series from `names`."""
    for name in names:
        s = _col(df, name)
        if not s.empty and not s.isna().all():
            return s
    return pd.Series(dtype=float)


def _nanmean(s: pd.Series) -> float:
    """Return mean of a Series ignoring NaN, or NaN if all values are NaN."""
    valid = s.dropna()
    return float(valid.mean()) if not valid.empty else np.nan


def _linear_slope(s: pd.Series) -> float:
    """Compute linear regression slope of a Series (positive = growing trend)."""
    s = s.dropna()
    if len(s) < 2:
        return np.nan
    x = np.arange(len(s), dtype=float)
    return float(np.polyfit(x, s.to_numpy(), 1)[0])
