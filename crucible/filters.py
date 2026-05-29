"""Layer 1 fundamental filters — hard rules applied before scoring.

Each filter is a pure function: takes a DataFrame, returns the subset that passes.
NaN on a filter metric is treated as a failure — missing data cannot confirm quality.
"""

from __future__ import annotations

import logging

import pandas as pd

from crucible.config import FilterThresholds

logger = logging.getLogger(__name__)


def apply_filters(df: pd.DataFrame, thresholds: FilterThresholds) -> pd.DataFrame:
    """Apply all Layer 1 filters in sequence; return only companies that pass all.

    Tickers flagged as insufficient_data are excluded before any filter runs.
    """
    usable = df[~df["insufficient_data"].astype(bool)].copy()
    logger.info(
        "apply_filters start: %d total, %d with sufficient data",
        len(df),
        len(usable),
    )

    pipeline = [
        ("roic",                 lambda d: filter_roic(d, thresholds.roic_min)),
        ("fcf_consistency",      lambda d: filter_fcf_consistency(d, thresholds.fcf_positive_min_years)),
        ("leverage",             lambda d: filter_leverage(d, thresholds.net_debt_ebitda_max)),
        ("revenue_growth",       lambda d: filter_revenue_growth(d, thresholds.revenue_growth_positive_min_years)),
        ("gross_margin_trend",   lambda d: filter_gross_margin_stability(d, thresholds.gross_margin_min_slope)),
    ]

    result = usable
    for name, fn in pipeline:
        before = len(result)
        result = fn(result)
        logger.info("  %-22s %3d → %3d", name, before, len(result))

    logger.info("apply_filters end: %d companies pass all filters", len(result))
    return result


def filter_roic(df: pd.DataFrame, threshold: float = 0.15) -> pd.DataFrame:
    """Keep tickers where average ROIC proxy exceeds threshold."""
    mask = df["roic_proxy_avg"].notna() & (df["roic_proxy_avg"] > threshold)
    return df[mask]


def filter_fcf_consistency(
    df: pd.DataFrame, min_positive_years: int = 4
) -> pd.DataFrame:
    """Keep tickers with FCF positive in at least min_positive_years."""
    mask = (
        df["fcf_positive_years"].notna()
        & (df["fcf_positive_years"] >= min_positive_years)
    )
    return df[mask]


def filter_leverage(df: pd.DataFrame, max_ratio: float = 3.0) -> pd.DataFrame:
    """Keep tickers where Net Debt / EBITDA is below max_ratio."""
    mask = df["net_debt_ebitda"].notna() & (df["net_debt_ebitda"] < max_ratio)
    return df[mask]


def filter_revenue_growth(
    df: pd.DataFrame, min_positive_years: int = 3
) -> pd.DataFrame:
    """Keep tickers with positive YoY revenue growth in at least min_positive_years."""
    mask = (
        df["revenue_growth_positive_years"].notna()
        & (df["revenue_growth_positive_years"] >= min_positive_years)
    )
    return df[mask]


def filter_gross_margin_stability(
    df: pd.DataFrame, min_slope: float = -0.005
) -> pd.DataFrame:
    """Keep tickers with gross margin trend slope ≥ min_slope."""
    mask = (
        df["gross_margin_trend_slope"].notna()
        & (df["gross_margin_trend_slope"] >= min_slope)
    )
    return df[mask]
