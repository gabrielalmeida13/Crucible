# crucible/tracks/track3_value.py
"""Track 3 — Value Recovery (contrarian).

Filter stack targets statistically cheap companies that show at least one
concrete recovery signal.  Philosophy: cheap AND turning around — not
"cheap and getting cheaper".

Scorer weights: value=50%, recovery_signal=30%, balance_sheet=20%.

p_fcf_vs_history requires attach_p_fcf_history() to have been called on the
snapshot dict before the walk-forward loop.  In the diagnostic (no prices),
that filter degrades to a no-op automatically.
"""
from __future__ import annotations

import logging

import pandas as pd

from crucible.config import CrucibleConfig, Track3FilterThresholds, Track3ScoreWeights
from crucible.fx import apply_fx_penalty
from crucible.scorer import _derive_accounting_region, _peer_rank

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layer 1 — Filter functions
# ---------------------------------------------------------------------------


def filter_roic_proxy(
    df: pd.DataFrame,
    min_roic: float = 0.08,
) -> pd.DataFrame:
    """Keep tickers where roic_proxy_avg > min_roic."""
    mask = df["roic_proxy_avg"].notna() & (df["roic_proxy_avg"] > min_roic)
    return df[mask]


def filter_p_fcf_vs_history(
    df: pd.DataFrame,
    min_score: float = 1.0,
) -> pd.DataFrame:
    """Keep tickers statistically cheap vs own history.

    Pass condition: p_fcf_vs_history >= min_score, i.e. current P/FCF sits
    more than min_score standard deviations below the ticker's own 5-year mean.

    If the column is absent (diagnostic mode, no prices) or contains no valid
    values (warm-up period), the filter is skipped entirely so the pool is
    not wiped out prematurely.  Once at least one ticker has a valid history
    score, all tickers without a score are EXCLUDED (we cannot verify cheapness).
    """
    if "p_fcf_vs_history" not in df.columns:
        return df
    has_history = df["p_fcf_vs_history"].notna()
    if not has_history.any():
        return df   # warm-up: no history yet — skip rather than eliminate all
    # For tickers WITH history: require >= min_score.
    # For tickers WITHOUT history: exclude (cannot verify they are cheap).
    return df[has_history & (df["p_fcf_vs_history"] >= min_score)]


def filter_fcf_positive_last5(
    df: pd.DataFrame,
    min_years: int = 2,
) -> pd.DataFrame:
    """Keep tickers with FCF positive in >= min_years of the last 5 fiscal years."""
    mask = (
        df["fcf_positive_years_last5"].notna()
        & (df["fcf_positive_years_last5"] >= min_years)
    )
    return df[mask]


def filter_recovery_signal(
    df: pd.DataFrame,
    buyback_min: float = 0.03,
    gm_recovery_change_min: float = 0.02,
) -> pd.DataFrame:
    """Keep tickers with at least one recovery signal present.

    Signal A — share_buyback_signal > buyback_min
        Management buying back > 3% of shares: capital-allocation confidence.
    Signal B — revenue_growth_yr1 > 0 AND revenue_growth_yr2 < 0
        Revenue turned positive after a negative year: inflection.
    Signal C — gross_margin_yr1_change > gm_recovery_change_min AND gross_margin_trend_slope < 0
        Recent margin improvement (> 2pp) despite a longer declining trend.
    """
    signal_a = df["share_buyback_signal"].notna() & (df["share_buyback_signal"] > buyback_min)

    signal_b = (
        df["revenue_growth_yr1"].notna()
        & (df["revenue_growth_yr1"] > 0)
        & df["revenue_growth_yr2"].notna()
        & (df["revenue_growth_yr2"] < 0)
    )

    signal_c = (
        df["gross_margin_yr1_change"].notna()
        & (df["gross_margin_yr1_change"] > gm_recovery_change_min)
        & df["gross_margin_trend_slope"].notna()
        & (df["gross_margin_trend_slope"] < 0)
    )

    return df[signal_a | signal_b | signal_c]


def apply_filters(
    df: pd.DataFrame,
    thresholds: Track3FilterThresholds,
) -> pd.DataFrame:
    """Apply all Track 3 Layer 1 filters in sequence.

    p_fcf_vs_history filter requires attach_p_fcf_history() to have been called.
    """
    usable = df[~df["insufficient_data"].astype(bool)].copy()
    logger.info(
        "track3 apply_filters start: %d total, %d with sufficient data",
        len(df), len(usable),
    )

    pipeline = [
        ("roic_proxy",        lambda d: filter_roic_proxy(d, thresholds.roic_proxy_min)),
        ("p_fcf_vs_history",  lambda d: filter_p_fcf_vs_history(d, thresholds.p_fcf_vs_history_min)),
        ("fcf_positive_5yr",  lambda d: filter_fcf_positive_last5(d, thresholds.fcf_positive_min_years)),
        ("recovery_signal",   lambda d: filter_recovery_signal(
            d,
            buyback_min=thresholds.buyback_signal_min,
            gm_recovery_change_min=thresholds.gm_recovery_change_min,
        )),
    ]

    result = usable
    for name, fn in pipeline:
        before = len(result)
        result = fn(result)
        logger.info("  %-22s %3d → %3d", name, before, len(result))

    logger.info("track3 apply_filters end: %d companies pass all filters", len(result))
    return result


# ---------------------------------------------------------------------------
# Layer 2 — Scorer
# ---------------------------------------------------------------------------


def score(
    df: pd.DataFrame,
    config: CrucibleConfig,
    weights: Track3ScoreWeights,
) -> pd.DataFrame:
    """Compute value, recovery_signal, balance_sheet, and composite scores.

    value (50%):
        equal-weight rank of p_fcf and ev_ebitda (ascending=False: lower = cheaper = better).
    recovery_signal (30%):
        equal-weight rank of share_buyback_signal (ascending=True),
        gross_margin_yr1_change (ascending=True),
        and revenue_growth_yr1 rank — non-zero only where revenue_growth_yr2 < 0.
    balance_sheet (20%):
        equal-weight rank of net_debt_ebitda (ascending=False) and
        interest_coverage (ascending=True).
    """
    df = df.copy()
    accounting_region = _derive_accounting_region(df)
    peer_group = df["sector"].fillna("Unknown") + "|" + accounting_region

    # --- value score: lower multiples = cheaper = higher rank ---
    vr_pfcf     = _peer_rank(df, "p_fcf",     peer_group, ascending=False, fill_nan=0.0)
    vr_evebitda = _peer_rank(df, "ev_ebitda", peer_group, ascending=False, fill_nan=0.0)
    df["value_score"] = (vr_pfcf + vr_evebitda) / 2.0

    # --- recovery signal score ---
    rs_buyback = _peer_rank(df, "share_buyback_signal",   peer_group, ascending=True, fill_nan=0.0)
    rs_margin  = _peer_rank(df, "gross_margin_yr1_change", peer_group, ascending=True, fill_nan=0.0)

    # Revenue inflection sub-score: rank revenue_growth_yr1 across ALL tickers
    # but zero out any ticker where revenue_growth_yr2 was NOT negative (not an inflection).
    rs_rev_all    = _peer_rank(df, "revenue_growth_yr1", peer_group, ascending=True, fill_nan=0.0)
    inflection_on = df["revenue_growth_yr2"].notna() & (df["revenue_growth_yr2"] < 0)
    rs_inflection = rs_rev_all.where(inflection_on, other=0.0)

    df["recovery_signal_score"] = (rs_buyback + rs_margin + rs_inflection) / 3.0

    # --- balance sheet safety score ---
    bs_nd = _peer_rank(df, "net_debt_ebitda",   peer_group, ascending=False, fill_nan=0.0)
    bs_ic = _peer_rank(df, "interest_coverage", peer_group, ascending=True,  fill_nan=0.5)
    df["balance_sheet_score"] = (bs_nd + bs_ic) / 2.0

    df = apply_fx_penalty(df, config.account_currency, config.fx.conversion_penalty)

    df["composite_score"] = (
        weights.value           * df["value_score"]
        + weights.recovery_signal * df["recovery_signal_score"]
        + weights.balance_sheet   * df["balance_sheet_score"]
        + df["fx_penalty"]
    )

    logger.info(
        "track3 scored %d companies; top=%.3f  bottom=%.3f",
        len(df),
        df["composite_score"].max() if len(df) else float("nan"),
        df["composite_score"].min() if len(df) else float("nan"),
    )
    return df.sort_values("composite_score", ascending=False)


def run(
    df: pd.DataFrame,
    config: CrucibleConfig,
    weights: Track3ScoreWeights | None = None,
) -> pd.DataFrame:
    """Apply Track 3 filters then scorer; return sorted by composite_score."""
    if weights is None:
        weights = config.track3_score_weights
    filtered = apply_filters(df, config.track3_filters)
    return score(filtered, config, weights)
