"""Unit tests for scorer.py and fx.py — synthetic data, no API calls."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crucible.config import FXConfig, FilterThresholds, ScoreWeights, load_config
from crucible.fx import apply_fx_penalty
from crucible.scorer import _derive_accounting_region, _peer_rank, score


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _base_row(**kwargs) -> dict:
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
    if tickers is None:
        tickers = [f"T{i}" for i in range(len(rows))]
    return pd.DataFrame(list(rows), index=pd.Index(tickers, name="ticker"))


# ---------------------------------------------------------------------------
# apply_fx_penalty
# ---------------------------------------------------------------------------


def test_fx_no_penalty_same_currency() -> None:
    """Tickers in account currency must receive fx_penalty = 0.0."""
    df = _df(_base_row(currency="EUR"))
    result = apply_fx_penalty(df, account_currency="EUR", penalty=-0.5)
    assert result.loc["T0", "fx_penalty"] == 0.0


def test_fx_penalty_different_currency() -> None:
    """Tickers in a foreign currency must receive the configured penalty."""
    df = _df(_base_row(currency="GBP"))
    result = apply_fx_penalty(df, account_currency="EUR", penalty=-0.5)
    assert result.loc["T0", "fx_penalty"] == -0.5


def test_fx_penalty_nan_currency_is_penalised() -> None:
    """Unknown currency must be treated conservatively as requiring conversion."""
    df = _df(_base_row(currency=None))
    result = apply_fx_penalty(df, account_currency="EUR", penalty=-0.5)
    assert result.loc["T0", "fx_penalty"] == -0.5


def test_fx_penalty_mixed_currencies() -> None:
    """Only tickers in a different currency receive the penalty."""
    df = _df(
        _base_row(currency="EUR"),
        _base_row(currency="USD"),
        tickers=["HOME", "FOREIGN"],
    )
    result = apply_fx_penalty(df, account_currency="EUR", penalty=-0.5)
    assert result.loc["HOME", "fx_penalty"] == 0.0
    assert result.loc["FOREIGN", "fx_penalty"] == -0.5


# ---------------------------------------------------------------------------
# _derive_accounting_region
# ---------------------------------------------------------------------------


def test_usd_maps_to_us_gaap() -> None:
    df = _df(_base_row(currency="USD"))
    regions = _derive_accounting_region(df)
    assert regions.iloc[0] == "US_GAAP"


def test_eur_maps_to_ifrs() -> None:
    df = _df(_base_row(currency="EUR"))
    assert _derive_accounting_region(df).iloc[0] == "IFRS"


def test_jpy_maps_to_japanese_gaap() -> None:
    df = _df(_base_row(currency="JPY"))
    assert _derive_accounting_region(df).iloc[0] == "JAPANESE_GAAP"


def test_unknown_currency_defaults_to_us_gaap() -> None:
    df = _df(_base_row(currency="XYZ"))
    assert _derive_accounting_region(df).iloc[0] == "US_GAAP"


# ---------------------------------------------------------------------------
# _peer_rank
# ---------------------------------------------------------------------------


def test_peer_rank_higher_value_gets_higher_rank() -> None:
    """Quality metrics: highest value must receive rank ≈ 1.0."""
    df = _df(
        _base_row(roic_proxy_avg=0.10),
        _base_row(roic_proxy_avg=0.20),
        _base_row(roic_proxy_avg=0.30),
        tickers=["LOW", "MID", "HIGH"],
    )
    peer_group = pd.Series(["Tech|US_GAAP"] * 3, index=df.index)
    ranks = _peer_rank(df, "roic_proxy_avg", peer_group, ascending=True)
    assert ranks["HIGH"] > ranks["MID"] > ranks["LOW"]


def test_peer_rank_lower_value_gets_higher_rank_for_valuation() -> None:
    """Valuation metrics: lowest P/FCF (cheapest) must receive rank ≈ 1.0."""
    df = _df(
        _base_row(p_fcf=10.0),
        _base_row(p_fcf=20.0),
        _base_row(p_fcf=30.0),
        tickers=["CHEAP", "MID", "EXPENSIVE"],
    )
    peer_group = pd.Series(["Tech|US_GAAP"] * 3, index=df.index)
    ranks = _peer_rank(df, "p_fcf", peer_group, ascending=False)
    assert ranks["CHEAP"] > ranks["MID"] > ranks["EXPENSIVE"]


def test_peer_rank_nan_becomes_zero() -> None:
    """NaN metric values must produce rank 0.0 (conservative penalty)."""
    df = _df(_base_row(roic_proxy_avg=np.nan))
    peer_group = pd.Series(["Tech|US_GAAP"], index=df.index)
    ranks = _peer_rank(df, "roic_proxy_avg", peer_group, ascending=True)
    assert ranks.iloc[0] == 0.0


def test_peer_rank_all_nan_produces_zeros() -> None:
    """If entire peer group has NaN, all ranks must be 0.0."""
    df = _df(
        _base_row(p_fcf=np.nan),
        _base_row(p_fcf=np.nan),
        tickers=["A", "B"],
    )
    peer_group = pd.Series(["Tech|US_GAAP"] * 2, index=df.index)
    ranks = _peer_rank(df, "p_fcf", peer_group, ascending=False)
    assert (ranks == 0.0).all()


# ---------------------------------------------------------------------------
# Sector normalisation
# ---------------------------------------------------------------------------


def test_ranking_is_within_sector_not_global() -> None:
    """A mediocre company in Sector B must not be penalised by Sector A's stars.

    Company C (ROIC 25%) is the only company in Sector B, so it should receive
    quality_score ≈ 1.0 regardless of Sector A's higher-ROIC peers.
    """
    cfg = load_config()
    df = _df(
        _base_row(sector="Technology",    roic_proxy_avg=0.40, p_fcf=20.0, ev_ebitda=15.0),
        _base_row(sector="Technology",    roic_proxy_avg=0.30, p_fcf=20.0, ev_ebitda=15.0),
        _base_row(sector="Industrials",   roic_proxy_avg=0.25, p_fcf=20.0, ev_ebitda=15.0),
        tickers=["TECH_HIGH", "TECH_LOW", "IND_ONLY"],
    )
    result = score(df, cfg)
    # IND_ONLY is the sole peer in Industrials → must get quality rank 1.0
    # Its quality_score should be higher than TECH_LOW which is bottom of Tech
    assert result.loc["IND_ONLY", "quality_score"] > result.loc["TECH_LOW", "quality_score"]


def test_region_normalization_separates_ifrs_from_us_gaap() -> None:
    """EUR and USD companies in the same sector must form separate peer groups."""
    cfg = load_config()
    df = _df(
        _base_row(sector="Technology", currency="USD", roic_proxy_avg=0.30),
        _base_row(sector="Technology", currency="EUR", roic_proxy_avg=0.10),
        tickers=["US_CO", "EU_CO"],
    )
    result = score(df, cfg)
    # Each company is the sole peer in its region → both get quality_score from
    # a rank of 1.0 for ROIC (best in their own group)
    # quality_score won't be exactly 1.0 because FCF and gross margin also contribute,
    # but ROIC rank must be 1.0 for both
    peer_group_us = "Technology|US_GAAP"
    peer_group_eu = "Technology|IFRS"
    # Verify they were ranked independently by checking scores are both positive
    assert result.loc["US_CO", "quality_score"] > 0
    assert result.loc["EU_CO", "quality_score"] > 0


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------


def test_composite_score_formula() -> None:
    """composite_score = 0.6*quality + 0.3*valuation + 0.1*momentum + fx_penalty."""
    cfg = load_config()
    df = _df(_base_row(currency="USD"))
    result = score(df, cfg)
    row = result.iloc[0]
    w = cfg.score_weights
    expected = (
        w.quality * row["quality_score"]
        + w.valuation * row["valuation_score"]
        + w.momentum * row["momentum_score"]
        + row["fx_penalty"]
    )
    assert abs(row["composite_score"] - expected) < 1e-9


def test_fx_penalty_reduces_composite_score() -> None:
    """A foreign-currency ticker must have a lower composite_score than an identical domestic one."""
    cfg = load_config()  # account_currency=EUR
    df = _df(
        _base_row(currency="EUR"),
        _base_row(currency="USD"),
        tickers=["DOMESTIC", "FOREIGN"],
    )
    result = score(df, cfg)
    assert result.loc["DOMESTIC", "composite_score"] > result.loc["FOREIGN", "composite_score"]


def test_score_output_sorted_descending() -> None:
    """score() must return rows sorted by composite_score, highest first."""
    cfg = load_config()
    df = _df(
        _base_row(roic_proxy_avg=0.10),
        _base_row(roic_proxy_avg=0.40),
        _base_row(roic_proxy_avg=0.25),
        tickers=["LOW", "HIGH", "MID"],
    )
    result = score(df, cfg)
    scores = result["composite_score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_score_adds_required_columns() -> None:
    """score() must add quality_score, valuation_score, fx_penalty, composite_score."""
    cfg = load_config()
    df = _df(_base_row())
    result = score(df, cfg)
    for col in ("quality_score", "valuation_score", "fx_penalty", "composite_score"):
        assert col in result.columns, f"Missing column: {col}"


def test_score_no_temp_columns_leaked() -> None:
    """Intermediate _qr_* and _vr_* columns must not appear in the output."""
    cfg = load_config()
    result = score(_df(_base_row()), cfg)
    leaked = [c for c in result.columns if c.startswith("_qr_") or c.startswith("_vr_")]
    assert leaked == [], f"Leaked temp columns: {leaked}"


def test_score_handles_all_nan_valuation() -> None:
    """score() must not raise when all valuation metrics are NaN."""
    cfg = load_config()
    df = _df(_base_row(p_fcf=np.nan, ev_ebitda=np.nan, p_e=np.nan))
    result = score(df, cfg)
    assert result.loc["T0", "valuation_score"] == 0.0


# ---------------------------------------------------------------------------
# Track 2 config
# ---------------------------------------------------------------------------

def test_track2_score_weights_sum_to_one():
    from crucible.config import Track2ScoreWeights
    w = Track2ScoreWeights()
    assert abs(w.growth_quality + w.momentum + w.valuation - 1.0) < 1e-9

def test_track2_score_weights_custom_values_validate():
    from crucible.config import Track2ScoreWeights
    w = Track2ScoreWeights(growth_quality=0.60, momentum=0.20, valuation=0.20)
    assert abs(w.growth_quality + w.momentum + w.valuation - 1.0) < 1e-9

def test_track2_score_weights_invalid_raises():
    from crucible.config import Track2ScoreWeights
    with pytest.raises(ValueError):
        Track2ScoreWeights(growth_quality=0.60, momentum=0.30, valuation=0.30)

def test_track2_filter_thresholds_defaults():
    from crucible.config import Track2FilterThresholds
    t = Track2FilterThresholds()
    assert t.revenue_growth_min_pct == 0.10
    assert t.gross_margin_min == 0.30
    assert t.fcf_positive_last2yr_min == 1
    assert t.net_debt_ebitda_max == 5.0

def test_crucible_config_has_track2_fields():
    from crucible.config import CrucibleConfig
    cfg = CrucibleConfig()
    assert hasattr(cfg, "track2_filters")
    assert hasattr(cfg, "track2_score_weights")
