# crucible/tracks/track2_growth.py
"""Track 2 — Growth Inflection.

Filter stack targets companies with accelerating revenue, expanding margins,
and positive price momentum.  ROIC > 15% is NOT required — a lower-ROIC
company accelerating toward quality may be early-stage compounder material.

Scorer weights: growth_quality=50%, momentum=30%, valuation=20%.
"""
from __future__ import annotations

import logging

import pandas as pd

from crucible.config import CrucibleConfig, Track2FilterThresholds, Track2ScoreWeights
from crucible.fx import apply_fx_penalty
from crucible.scorer import _derive_accounting_region, _peer_rank

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layer 1 — Filter functions
# ---------------------------------------------------------------------------


def filter_revenue_growth_10pct(
    df: pd.DataFrame,
    min_pct: float = 0.10,
) -> pd.DataFrame:
    """Keep tickers where both revenue_growth_yr1 and revenue_growth_yr2 > min_pct."""
    mask = (
        df["revenue_growth_yr1"].notna()
        & (df["revenue_growth_yr1"] > min_pct)
        & df["revenue_growth_yr2"].notna()
        & (df["revenue_growth_yr2"] > min_pct)
    )
    return df[mask]


def filter_revenue_growth_qyoy(
    df: pd.DataFrame,
    min_qyoy_pct: float = 0.06,
    annual_fallback_pct: float = 0.08,
) -> pd.DataFrame:
    """Keep tickers where revenue_growth_q1yoy > min_qyoy_pct.

    More responsive than the annual filter: uses the most recent quarter's
    revenue vs the same quarter one year ago, catching slowdowns and
    re-accelerations that annual figures would average away.

    Fallback: when revenue_growth_q1yoy is NaN or absent from the snapshot
    (pre-Phase-5 caches), falls back to revenue_growth_yr1 > annual_fallback_pct.
    This ensures backward compatibility with cached snapshots built before
    quarterly EDGAR data was added.
    """
    if "revenue_growth_q1yoy" not in df.columns:
        # Column absent entirely (old cache) — use annual filter exclusively
        return df[
            df["revenue_growth_yr1"].notna()
            & (df["revenue_growth_yr1"] > annual_fallback_pct)
        ]
    q_mask = df["revenue_growth_q1yoy"].notna() & (df["revenue_growth_q1yoy"] > min_qyoy_pct)
    ann_fallback = (
        df["revenue_growth_q1yoy"].isna()
        & df["revenue_growth_yr1"].notna()
        & (df["revenue_growth_yr1"] > annual_fallback_pct)
    )
    return df[q_mask | ann_fallback]


def filter_revenue_acceleration(df: pd.DataFrame) -> pd.DataFrame:
    """Keep tickers where revenue_acceleration > 0 (YoY growth rate is increasing)."""
    mask = df["revenue_acceleration"].notna() & (df["revenue_acceleration"] > 0)
    return df[mask]


def filter_gross_margin_growth(
    df: pd.DataFrame,
    min_margin: float = 0.30,
) -> pd.DataFrame:
    """Keep tickers where gross_margin_latest >= min_margin OR gross_margin_yr1_change > 0."""
    high_margin = df["gross_margin_latest"].notna() & (df["gross_margin_latest"] >= min_margin)
    expanding   = df["gross_margin_yr1_change"].notna() & (df["gross_margin_yr1_change"] > 0)
    return df[high_margin | expanding]


def filter_fcf_positive_last2yr(
    df: pd.DataFrame,
    min_years: int = 1,
) -> pd.DataFrame:
    """Keep tickers with FCF positive in at least min_years of the last 2 years."""
    mask = (
        df["fcf_positive_last2yr"].notna()
        & (df["fcf_positive_last2yr"] >= min_years)
    )
    return df[mask]


def filter_leverage_soft(
    df: pd.DataFrame,
    soft_max: float = 8.0,
) -> pd.DataFrame:
    """Pass if net_debt_ebitda < soft_max OR fcf_trajectory > 0 (debt-reducing via FCF growth)."""
    low_debt  = df["net_debt_ebitda"].notna() & (df["net_debt_ebitda"] < soft_max)
    deleveraging = df["fcf_trajectory"].notna() & (df["fcf_trajectory"] > 0)
    return df[low_debt | deleveraging]


def apply_filters(
    df: pd.DataFrame,
    thresholds: Track2FilterThresholds,
) -> pd.DataFrame:
    """Apply all Track 2 Layer 1 filters in sequence.

    Revenue growth filter uses quarterly YoY (revenue_growth_q1yoy) when
    available, falling back to annual (revenue_growth_yr1) for snapshots
    built before Phase 5 quarterly data was added.

    Note: momentum filter (momentum > 0) is NOT applied here — it is enforced
    in run_monthly.py after prices are attached.
    """
    usable = df[~df["insufficient_data"].astype(bool)].copy()
    logger.info(
        "track2 apply_filters start: %d total, %d with sufficient data",
        len(df), len(usable),
    )

    pipeline = [
        ("rev_growth_qyoy",   lambda d: filter_revenue_growth_qyoy(
                                  d,
                                  min_qyoy_pct=thresholds.revenue_growth_qyoy_min_pct,
                                  annual_fallback_pct=thresholds.revenue_growth_min_pct,
                              )),
        ("rev_acceleration",  lambda d: filter_revenue_acceleration(d)),
        ("gross_margin",      lambda d: filter_gross_margin_growth(d, thresholds.gross_margin_min)),
        ("fcf_last2yr",       lambda d: filter_fcf_positive_last2yr(d, thresholds.fcf_positive_last2yr_min)),
        ("leverage_soft",     lambda d: filter_leverage_soft(d, thresholds.net_debt_ebitda_soft_max)),
    ]

    result = usable
    for name, fn in pipeline:
        before = len(result)
        result = fn(result)
        logger.info("  %-22s %3d → %3d", name, before, len(result))

    logger.info("track2 apply_filters end: %d companies pass all filters", len(result))
    return result


# ---------------------------------------------------------------------------
# Layer 2 — Scorer
# ---------------------------------------------------------------------------


def score(
    df: pd.DataFrame,
    config: CrucibleConfig,
    weights: Track2ScoreWeights,
) -> pd.DataFrame:
    """Compute growth_quality, momentum, valuation, and composite scores.

    growth_quality (50% of composite):
        revenue_acceleration      10%  — annual: accelerating top-line growth
        revenue_accel_quarterly   10%  — quarterly: QoQ acceleration (Phase 5, more responsive)
        operating_margin_trend    15%  — expanding margins
        fcf_trajectory            15%  — improving free cash flow
        deferred_revenue_growth   10%  — rising order backlog (book-to-bill proxy)
        eps_surprise_last_q       10%  — earnings beat strength (Phase 4.7)
        asset_growth_penalty     -10%  — penalises over-investment (Fama-French CMA)
    momentum (30%): average rank of momentum_raw (12-1m) and momentum_3m.
    valuation (20%): p_s rank vs sector median; falls back to p_fcf per-ticker.

    Note on revenue_accel_quarterly: falls back gracefully to fill_nan=0.0 when the
    quarterly snapshot feature is absent (pre-Phase-5 caches).
    """
    df = df.copy()
    accounting_region = _derive_accounting_region(df)
    peer_group = df["sector"].fillna("Unknown") + "|" + accounting_region

    # --- growth quality (7 sub-components, weighted) ---
    gq_rev   = _peer_rank(df, "revenue_acceleration",    peer_group, ascending=True, fill_nan=0.0)
    # Quarterly acceleration (Phase 5): more sensitive, fills 0 when unavailable.
    gq_rev_q = _peer_rank(df, "revenue_accel_quarterly", peer_group, ascending=True, fill_nan=0.0)
    gq_om    = _peer_rank(df, "operating_margin_trend",  peer_group, ascending=True, fill_nan=0.0)
    gq_fcf   = _peer_rank(df, "fcf_trajectory",          peer_group, ascending=True, fill_nan=0.0)
    gq_def   = _peer_rank(df, "deferred_revenue_growth", peer_group, ascending=True, fill_nan=0.0)
    gq_eps   = _peer_rank(df, "eps_surprise_last_q",     peer_group, ascending=True, fill_nan=0.5)

    # Asset growth penalty: ascending=False → lower growth = higher quality rank (0→1).
    # Companies with >30% asset growth are hard-capped at the worst quality rank (0.0),
    # making their penalty term = 1 − 0.0 = 1.0 (maximum deduction from score).
    gq_asset_rank = _peer_rank(df, "asset_growth_yoy", peer_group, ascending=False, fill_nan=0.5)
    if "asset_growth_yoy" in df.columns:
        overinvest = df["asset_growth_yoy"].notna() & (df["asset_growth_yoy"] > 0.30)
        gq_asset_rank = gq_asset_rank.where(~overinvest, 0.0)
    gq_asset_penalty = 1.0 - gq_asset_rank  # rank 0.0 → max penalty 1.0; rank 1.0 → no penalty

    df["growth_quality_score"] = (
        0.10 * gq_rev
        + 0.10 * gq_rev_q          # Phase 5: quarterly acceleration
        + 0.15 * gq_om
        + 0.15 * gq_fcf
        + 0.10 * gq_def
        + 0.10 * gq_eps
        - 0.10 * gq_asset_penalty
    )

    # --- momentum (average of 12-1m and 3m ranks) ---
    mom_raw = _peer_rank(df, "momentum_raw", peer_group, ascending=True, fill_nan=0.5)
    mom_3m  = _peer_rank(df, "momentum_3m",  peer_group, ascending=True, fill_nan=0.5)
    df["momentum_score"] = (mom_raw + mom_3m) / 2.0

    # --- valuation: p_s primary, p_fcf fallback per-ticker ---
    vr_p_s   = _peer_rank(df, "p_s",   peer_group, ascending=False, fill_nan=float("nan"))
    vr_p_fcf = _peer_rank(df, "p_fcf", peer_group, ascending=False, fill_nan=0.0)
    df["valuation_score"] = vr_p_s.where(vr_p_s.notna(), vr_p_fcf)

    # FX penalty (same as Track 1)
    df = apply_fx_penalty(df, config.account_currency, config.fx.conversion_penalty)

    df["composite_score"] = (
        weights.growth_quality * df["growth_quality_score"]
        + weights.momentum     * df["momentum_score"]
        + weights.valuation    * df["valuation_score"]
        + df["fx_penalty"]
    )

    logger.info(
        "track2 scored %d companies; top=%.3f  bottom=%.3f",
        len(df),
        df["composite_score"].max() if len(df) else float("nan"),
        df["composite_score"].min() if len(df) else float("nan"),
    )
    return df.sort_values("composite_score", ascending=False)


def run(
    df: pd.DataFrame,
    config: CrucibleConfig,
    weights: Track2ScoreWeights | None = None,
) -> pd.DataFrame:
    """Apply Track 2 filters then scorer; return sorted by composite_score."""
    if weights is None:
        weights = config.track2_score_weights
    filtered = apply_filters(df, config.track2_filters)
    return score(filtered, config, weights)
