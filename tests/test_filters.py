"""Unit tests for filters.py — at least one positive and one negative case per filter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crucible.config import FilterThresholds
from crucible.filters import (
    apply_filters,
    filter_fcf_consistency,
    filter_gross_margin_stability,
    filter_leverage,
    filter_revenue_growth,
    filter_roic,
)

_DEFAULT = FilterThresholds()


def _row(**kwargs) -> dict:
    """Build a minimal processed-DataFrame row with all required columns."""
    defaults = dict(
        sector="Technology",
        sub_industry="Software",
        currency="USD",
        p_e=20.0,
        p_fcf=15.0,
        ev_ebitda=10.0,
        data_years=5,
        insufficient_data=False,
        roic_proxy_avg=0.20,
        fcf_latest=1e9,
        fcf_positive_years=4.0,
        net_debt_ebitda=1.0,
        revenue_growth_positive_years=4.0,
        gross_margin_latest=0.45,
        gross_margin_avg=0.44,
        gross_margin_trend_slope=0.01,
    )
    defaults.update(kwargs)
    return defaults


def _df(*rows: dict, tickers: list[str] | None = None) -> pd.DataFrame:
    """Build a small processed DataFrame from row dicts."""
    if tickers is None:
        tickers = [f"T{i}" for i in range(len(rows))]
    return pd.DataFrame(list(rows), index=pd.Index(tickers, name="ticker"))


# ---------------------------------------------------------------------------
# filter_roic
# ---------------------------------------------------------------------------


def test_roic_passes_above_threshold() -> None:
    """Ticker with ROIC above threshold must survive the filter."""
    df = _df(_row(roic_proxy_avg=0.20))
    assert len(filter_roic(df, threshold=0.15)) == 1


def test_roic_fails_below_threshold() -> None:
    """Ticker with ROIC below threshold must be removed."""
    df = _df(_row(roic_proxy_avg=0.10))
    assert filter_roic(df, threshold=0.15).empty


def test_roic_fails_exactly_at_threshold() -> None:
    """Threshold is strictly greater-than; equal value must fail."""
    df = _df(_row(roic_proxy_avg=0.15))
    assert filter_roic(df, threshold=0.15).empty


def test_roic_nan_fails() -> None:
    """NaN ROIC cannot confirm quality; ticker must be removed."""
    df = _df(_row(roic_proxy_avg=np.nan))
    assert filter_roic(df, threshold=0.15).empty


# ---------------------------------------------------------------------------
# filter_fcf_consistency
# ---------------------------------------------------------------------------


def test_fcf_consistency_passes() -> None:
    """Ticker with fcf_positive_years >= min must survive."""
    df = _df(_row(fcf_positive_years=4.0))
    assert len(filter_fcf_consistency(df, min_positive_years=4)) == 1


def test_fcf_consistency_fails_too_few_positive_years() -> None:
    """Ticker with fewer positive FCF years than minimum must be removed."""
    df = _df(_row(fcf_positive_years=3.0))
    assert filter_fcf_consistency(df, min_positive_years=4).empty


def test_fcf_consistency_nan_fails() -> None:
    """NaN FCF consistency cannot confirm quality; ticker must be removed."""
    df = _df(_row(fcf_positive_years=np.nan))
    assert filter_fcf_consistency(df, min_positive_years=4).empty


# ---------------------------------------------------------------------------
# filter_leverage
# ---------------------------------------------------------------------------


def test_leverage_passes_below_max() -> None:
    """Ticker with Net Debt/EBITDA below max must survive."""
    df = _df(_row(net_debt_ebitda=2.5))
    assert len(filter_leverage(df, max_ratio=3.0)) == 1


def test_leverage_fails_above_max() -> None:
    """Ticker with Net Debt/EBITDA above max must be removed."""
    df = _df(_row(net_debt_ebitda=3.5))
    assert filter_leverage(df, max_ratio=3.0).empty


def test_leverage_fails_exactly_at_max() -> None:
    """Threshold is strictly less-than; equal value must fail."""
    df = _df(_row(net_debt_ebitda=3.0))
    assert filter_leverage(df, max_ratio=3.0).empty


def test_leverage_nan_fails() -> None:
    """NaN leverage cannot be verified; ticker must be removed."""
    df = _df(_row(net_debt_ebitda=np.nan))
    assert filter_leverage(df, max_ratio=3.0).empty


# ---------------------------------------------------------------------------
# filter_revenue_growth
# ---------------------------------------------------------------------------


def test_revenue_growth_passes() -> None:
    """Ticker with enough positive-growth years must survive."""
    df = _df(_row(revenue_growth_positive_years=3.0))
    assert len(filter_revenue_growth(df, min_positive_years=3)) == 1


def test_revenue_growth_fails_too_few_years() -> None:
    """Ticker without enough positive-growth years must be removed."""
    df = _df(_row(revenue_growth_positive_years=2.0))
    assert filter_revenue_growth(df, min_positive_years=3).empty


def test_revenue_growth_nan_fails() -> None:
    """NaN growth data cannot confirm quality; ticker must be removed."""
    df = _df(_row(revenue_growth_positive_years=np.nan))
    assert filter_revenue_growth(df, min_positive_years=3).empty


# ---------------------------------------------------------------------------
# filter_gross_margin_stability
# ---------------------------------------------------------------------------


def test_gross_margin_stability_passes_positive_slope() -> None:
    """Ticker with a growing gross margin (slope > 0) must survive."""
    df = _df(_row(gross_margin_trend_slope=0.005))
    assert len(filter_gross_margin_stability(df)) == 1


def test_gross_margin_stability_passes_zero_slope() -> None:
    """Ticker with exactly flat gross margin (slope = 0) must survive."""
    df = _df(_row(gross_margin_trend_slope=0.0))
    assert len(filter_gross_margin_stability(df)) == 1


def test_gross_margin_stability_fails_negative_slope() -> None:
    """Ticker with declining gross margin must be removed."""
    df = _df(_row(gross_margin_trend_slope=-0.01))
    assert filter_gross_margin_stability(df).empty


def test_gross_margin_stability_nan_fails() -> None:
    """NaN trend slope cannot confirm stability; ticker must be removed."""
    df = _df(_row(gross_margin_trend_slope=np.nan))
    assert filter_gross_margin_stability(df).empty


# ---------------------------------------------------------------------------
# apply_filters (integration)
# ---------------------------------------------------------------------------


def test_apply_filters_perfect_company_passes() -> None:
    """A company meeting all default thresholds must pass all filters."""
    df = _df(_row())  # all defaults satisfy FilterThresholds defaults
    result = apply_filters(df, _DEFAULT)
    assert len(result) == 1


def test_apply_filters_excludes_insufficient_data() -> None:
    """Tickers flagged insufficient_data must be excluded before filters run."""
    good = _row()
    bad = _row(insufficient_data=True)
    df = _df(good, bad, tickers=["GOOD", "BAD"])
    result = apply_filters(df, _DEFAULT)
    assert "GOOD" in result.index
    assert "BAD" not in result.index


def test_apply_filters_one_failure_drops_ticker() -> None:
    """A ticker failing even one filter must be absent from the output."""
    passes = _row()
    fails_roic = _row(roic_proxy_avg=0.05)
    df = _df(passes, fails_roic, tickers=["PASS", "FAIL"])
    result = apply_filters(df, _DEFAULT)
    assert "PASS" in result.index
    assert "FAIL" not in result.index


def test_apply_filters_empty_input_returns_empty() -> None:
    """apply_filters on an empty DataFrame must return an empty DataFrame."""
    # iloc[0:0] preserves column names and dtypes; avoids object-dtype booleans
    empty = _df(_row()).iloc[0:0]
    result = apply_filters(empty, _DEFAULT)
    assert result.empty


def test_apply_filters_all_fail_returns_empty() -> None:
    """If no ticker passes all filters, the result must be empty."""
    row = _row(roic_proxy_avg=0.01, net_debt_ebitda=10.0)
    df = _df(row)
    assert apply_filters(df, _DEFAULT).empty
