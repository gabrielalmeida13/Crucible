# tests/test_track2.py
"""Tests for Track 2 filter functions and scorer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crucible.config import CrucibleConfig, Track2FilterThresholds, Track2ScoreWeights


def _row(**kwargs) -> dict:
    base = dict(
        sector="Technology",
        currency="USD",
        insufficient_data=False,
        # Track 2 filter fields
        revenue_growth_yr1=0.15,
        revenue_growth_yr2=0.12,
        revenue_acceleration=0.03,
        gross_margin_latest=0.40,
        gross_margin_yr1_change=0.02,
        fcf_positive_last2yr=2,
        net_debt_ebitda=2.0,
        momentum_raw=0.05,
        # Scorer fields
        operating_margin_trend=0.01,
        fcf_trajectory=5e6,
        momentum_3m=0.03,
        p_s=3.0,
        p_fcf=15.0,
        # Other columns scorer.py expects
        p_e=20.0,
        ev_ebitda=10.0,
        roic_proxy_avg=0.10,
        fcf_positive_years=3.0,
        gross_margin_avg=0.38,
        gross_margin_trend_slope=0.01,
        interest_coverage=5.0,
        cfo_to_ni=1.1,
        capex_intensity=0.05,
        share_buyback_signal=None,
        insider_buy_ratio=None,
        revenue_growth_positive_years=4.0,
    )
    base.update(kwargs)
    return base


def _df(*rows, tickers=None) -> pd.DataFrame:
    if tickers is None:
        tickers = [f"T{i}" for i in range(len(rows))]
    return pd.DataFrame(list(rows), index=pd.Index(tickers, name="ticker"))


# ---------------------------------------------------------------------------
# filter_revenue_growth_10pct
# ---------------------------------------------------------------------------

def test_filter_revenue_growth_10pct_passes():
    from crucible.tracks.track2_growth import filter_revenue_growth_10pct
    df = _df(_row(revenue_growth_yr1=0.15, revenue_growth_yr2=0.12))
    assert len(filter_revenue_growth_10pct(df)) == 1

def test_filter_revenue_growth_10pct_fails_yr1_too_low():
    from crucible.tracks.track2_growth import filter_revenue_growth_10pct
    df = _df(_row(revenue_growth_yr1=0.05, revenue_growth_yr2=0.15))
    assert len(filter_revenue_growth_10pct(df)) == 0

def test_filter_revenue_growth_10pct_fails_yr2_too_low():
    from crucible.tracks.track2_growth import filter_revenue_growth_10pct
    df = _df(_row(revenue_growth_yr1=0.15, revenue_growth_yr2=0.05))
    assert len(filter_revenue_growth_10pct(df)) == 0

def test_filter_revenue_growth_10pct_fails_nan():
    from crucible.tracks.track2_growth import filter_revenue_growth_10pct
    df = _df(_row(revenue_growth_yr1=None, revenue_growth_yr2=0.12))
    assert len(filter_revenue_growth_10pct(df)) == 0


# ---------------------------------------------------------------------------
# filter_revenue_acceleration
# ---------------------------------------------------------------------------

def test_filter_revenue_acceleration_passes():
    from crucible.tracks.track2_growth import filter_revenue_acceleration
    df = _df(_row(revenue_acceleration=0.05))
    assert len(filter_revenue_acceleration(df)) == 1

def test_filter_revenue_acceleration_fails_negative():
    from crucible.tracks.track2_growth import filter_revenue_acceleration
    df = _df(_row(revenue_acceleration=-0.01))
    assert len(filter_revenue_acceleration(df)) == 0

def test_filter_revenue_acceleration_fails_nan():
    from crucible.tracks.track2_growth import filter_revenue_acceleration
    df = _df(_row(revenue_acceleration=None))
    assert len(filter_revenue_acceleration(df)) == 0


# ---------------------------------------------------------------------------
# filter_gross_margin_growth
# ---------------------------------------------------------------------------

def test_filter_gross_margin_growth_passes_high_margin():
    from crucible.tracks.track2_growth import filter_gross_margin_growth
    df = _df(_row(gross_margin_latest=0.50, gross_margin_yr1_change=-0.05))
    assert len(filter_gross_margin_growth(df)) == 1

def test_filter_gross_margin_growth_passes_expanding():
    from crucible.tracks.track2_growth import filter_gross_margin_growth
    df = _df(_row(gross_margin_latest=0.25, gross_margin_yr1_change=0.02))
    assert len(filter_gross_margin_growth(df)) == 1

def test_filter_gross_margin_growth_fails_low_and_contracting():
    from crucible.tracks.track2_growth import filter_gross_margin_growth
    df = _df(_row(gross_margin_latest=0.25, gross_margin_yr1_change=-0.01))
    assert len(filter_gross_margin_growth(df)) == 0

def test_filter_gross_margin_growth_fails_both_nan():
    from crucible.tracks.track2_growth import filter_gross_margin_growth
    df = _df(_row(gross_margin_latest=None, gross_margin_yr1_change=None))
    assert len(filter_gross_margin_growth(df)) == 0


# ---------------------------------------------------------------------------
# filter_fcf_positive_last2yr
# ---------------------------------------------------------------------------

def test_filter_fcf_positive_last2yr_passes_two():
    from crucible.tracks.track2_growth import filter_fcf_positive_last2yr
    df = _df(_row(fcf_positive_last2yr=2))
    assert len(filter_fcf_positive_last2yr(df)) == 1

def test_filter_fcf_positive_last2yr_passes_one():
    from crucible.tracks.track2_growth import filter_fcf_positive_last2yr
    df = _df(_row(fcf_positive_last2yr=1))
    assert len(filter_fcf_positive_last2yr(df)) == 1

def test_filter_fcf_positive_last2yr_fails_zero():
    from crucible.tracks.track2_growth import filter_fcf_positive_last2yr
    df = _df(_row(fcf_positive_last2yr=0))
    assert len(filter_fcf_positive_last2yr(df)) == 0

def test_filter_fcf_positive_last2yr_fails_nan():
    from crucible.tracks.track2_growth import filter_fcf_positive_last2yr
    df = _df(_row(fcf_positive_last2yr=None))
    assert len(filter_fcf_positive_last2yr(df)) == 0


# ---------------------------------------------------------------------------
# filter_leverage (Track 2 version — 5x threshold)
# ---------------------------------------------------------------------------

def test_filter_leverage_passes_below_5():
    from crucible.tracks.track2_growth import filter_leverage
    df = _df(_row(net_debt_ebitda=4.5))
    assert len(filter_leverage(df)) == 1

def test_filter_leverage_fails_above_5():
    from crucible.tracks.track2_growth import filter_leverage
    df = _df(_row(net_debt_ebitda=5.5))
    assert len(filter_leverage(df)) == 0

def test_filter_leverage_fails_nan():
    from crucible.tracks.track2_growth import filter_leverage
    df = _df(_row(net_debt_ebitda=None))
    assert len(filter_leverage(df)) == 0


# ---------------------------------------------------------------------------
# apply_filters pipeline
# ---------------------------------------------------------------------------

def test_apply_filters_all_pass():
    from crucible.tracks.track2_growth import apply_filters
    df = _df(_row(), _row(), tickers=["T0", "T1"])
    result = apply_filters(df, Track2FilterThresholds())
    assert len(result) == 2

def test_apply_filters_removes_insufficient_data():
    from crucible.tracks.track2_growth import apply_filters
    df = _df(_row(insufficient_data=True), _row(), tickers=["T0", "T1"])
    result = apply_filters(df, Track2FilterThresholds())
    assert "T0" not in result.index
    assert len(result) == 1

def test_apply_filters_removes_failing_company():
    from crucible.tracks.track2_growth import apply_filters
    df = _df(
        _row(revenue_growth_yr1=0.05),  # fails growth filter
        _row(),
        tickers=["T0", "T1"],
    )
    result = apply_filters(df, Track2FilterThresholds())
    assert "T0" not in result.index


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

def test_score_produces_composite_score():
    from crucible.tracks.track2_growth import score
    df = _df(_row(), _row(), _row(), tickers=["T0", "T1", "T2"])
    config = CrucibleConfig(account_currency="USD")
    weights = Track2ScoreWeights()
    result = score(df, config, weights)
    assert "composite_score" in result.columns
    assert result["composite_score"].notna().all()

def test_score_returns_sorted_descending():
    from crucible.tracks.track2_growth import score
    df = _df(
        _row(revenue_acceleration=0.01, momentum_raw=0.01, momentum_3m=0.01, p_s=20.0),
        _row(revenue_acceleration=0.20, momentum_raw=0.30, momentum_3m=0.25, p_s=2.0),
        tickers=["WEAK", "STRONG"],
    )
    config = CrucibleConfig(account_currency="USD")
    result = score(df, config, Track2ScoreWeights())
    assert result.index[0] == "STRONG"

def test_score_p_s_fallback_to_p_fcf():
    from crucible.tracks.track2_growth import score
    df = _df(
        _row(p_s=3.0, p_fcf=15.0),
        _row(p_s=None, p_fcf=12.0),
        tickers=["T0", "T1"],
    )
    config = CrucibleConfig(account_currency="USD")
    result = score(df, config, Track2ScoreWeights())
    assert "composite_score" in result.columns
    assert result.loc["T0", "composite_score"] is not None
    assert result.loc["T1", "composite_score"] is not None

def test_run_filters_then_scores():
    from crucible.tracks.track2_growth import run
    df = _df(
        _row(revenue_growth_yr1=0.03),  # will be filtered out
        _row(),
        tickers=["FILTERED", "KEEPER"],
    )
    config = CrucibleConfig(account_currency="USD")
    result = run(df, config, Track2ScoreWeights())
    assert "FILTERED" not in result.index
    assert "composite_score" in result.columns
