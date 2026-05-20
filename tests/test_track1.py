"""Tests for Track 1 — verifies it delegates to existing filters.py and scorer.py."""
from __future__ import annotations

import pandas as pd
import pytest

from crucible.config import CrucibleConfig


def _make_df(n: int = 5) -> pd.DataFrame:
    """Minimal passing DataFrame for Track 1 filters."""
    rows = []
    for i in range(n):
        rows.append({
            "sector": "Technology",
            "currency": "USD",
            "insufficient_data": False,
            "roic_proxy_avg": 0.20,
            "fcf_positive_years": 5.0,
            "net_debt_ebitda": 1.0,
            "revenue_growth_positive_years": 4.0,
            "gross_margin_trend_slope": 0.01,
            "gross_margin_latest": 0.45,
            "gross_margin_avg": 0.44,
            "p_e": 20.0,
            "p_fcf": 15.0,
            "ev_ebitda": 10.0,
            "fcf_latest": 1e9,
            "momentum_raw": 0.10,
            "interest_coverage": 5.0,
            "cfo_to_ni": 1.1,
            "capex_intensity": 0.05,
            "operating_margin_trend": 0.005,
            "revenue_acceleration": 0.02,
            "share_buyback_signal": None,
            "insider_buy_ratio": None,
        })
    tickers = [f"T{i}" for i in range(n)]
    return pd.DataFrame(rows, index=pd.Index(tickers, name="ticker"))


def test_track1_apply_filters_returns_dataframe():
    from crucible.tracks.track1_quality import apply_filters
    from crucible.config import FilterThresholds
    df = _make_df(5)
    result = apply_filters(df, FilterThresholds())
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 5  # all rows pass with good data


def test_track1_apply_filters_removes_failing_roic():
    from crucible.tracks.track1_quality import apply_filters
    from crucible.config import FilterThresholds
    df = _make_df(3)
    df.loc["T0", "roic_proxy_avg"] = 0.05  # below 15% threshold
    result = apply_filters(df, FilterThresholds())
    assert "T0" not in result.index
    assert len(result) == 2


def test_track1_score_returns_composite_score():
    from crucible.tracks.track1_quality import score
    df = _make_df(3)
    config = CrucibleConfig(account_currency="USD")
    result = score(df, config)
    assert "composite_score" in result.columns
    assert result["composite_score"].notna().all()


def test_track1_run_filters_then_scores():
    from crucible.tracks.track1_quality import run
    df = _make_df(5)
    df.loc["T0", "roic_proxy_avg"] = 0.05  # will be filtered out
    config = CrucibleConfig(account_currency="USD")
    result = run(df, config)
    assert "T0" not in result.index
    assert "composite_score" in result.columns
    assert result.iloc[0]["composite_score"] >= result.iloc[-1]["composite_score"]
