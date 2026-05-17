"""Layer 2 composite quality + valuation scorer.

All metrics are scored as percentile ranks within the same GICS sector AND
accounting region peer group.  Absolute ROIC values are never compared across
sectors or regions — only relative position within the peer group matters.

Valuation phase (Phase 1):
  P/FCF and EV/EBITDA are scored peer-relative (lower multiple = higher rank).
  Historical self-comparison requires FMP point-in-time data; deferred to Phase 2.
"""

from __future__ import annotations

import logging

import pandas as pd

from crucible.config import CrucibleConfig
from crucible.fx import apply_fx_penalty

logger = logging.getLogger(__name__)

# Quality metrics: higher value → better company
_QUALITY_METRICS: list[str] = ["roic_proxy_avg", "fcf_positive_years", "gross_margin_avg"]

# Valuation metrics: lower value → cheaper stock → ascending=False gives cheapest rank 1.0
_VALUATION_METRICS: list[str] = ["p_fcf", "ev_ebitda"]

_CURRENCY_REGION: dict[str, str] = {
    "USD": "US_GAAP",
    "CAD": "US_GAAP",
    "GBP": "IFRS",
    "EUR": "IFRS",
    "CHF": "IFRS",
    "SEK": "IFRS",
    "NOK": "IFRS",
    "DKK": "IFRS",
    "JPY": "JAPANESE_GAAP",
}


def score(df: pd.DataFrame, config: CrucibleConfig) -> pd.DataFrame:
    """Compute quality, valuation, FX and composite scores; sort descending.

    Output columns added: quality_score, valuation_score, fx_penalty, composite_score.
    Comparisons are strictly within GICS sector + accounting region peer groups.
    """
    df = df.copy()

    accounting_region = _derive_accounting_region(df)
    peer_group = df["sector"].fillna("Unknown") + "|" + accounting_region

    # --- quality percentile ranks (higher metric → higher rank) ---
    q_rank_cols: list[str] = []
    for col in _QUALITY_METRICS:
        rc = f"_qr_{col}"
        df[rc] = _peer_rank(df, col, peer_group, ascending=True)
        q_rank_cols.append(rc)

    # --- valuation percentile ranks (lower metric → higher rank) ---
    v_rank_cols: list[str] = []
    for col in _VALUATION_METRICS:
        rc = f"_vr_{col}"
        df[rc] = _peer_rank(df, col, peer_group, ascending=False)
        v_rank_cols.append(rc)

    df["quality_score"] = df[q_rank_cols].mean(axis=1)
    df["valuation_score"] = df[v_rank_cols].mean(axis=1)

    df = apply_fx_penalty(df, config.account_currency, config.fx.conversion_penalty)

    w = config.score_weights
    df["composite_score"] = (
        w.quality * df["quality_score"]
        + w.valuation * df["valuation_score"]
        + df["fx_penalty"]
    )

    drop_cols = [c for c in df.columns if c.startswith("_qr_") or c.startswith("_vr_")]
    df = df.drop(columns=drop_cols)

    logger.info(
        "Scored %d companies; top composite=%.3f  bottom=%.3f",
        len(df),
        df["composite_score"].max() if len(df) else float("nan"),
        df["composite_score"].min() if len(df) else float("nan"),
    )
    return df.sort_values("composite_score", ascending=False)


def _peer_rank(
    df: pd.DataFrame,
    col: str,
    peer_group: pd.Series,
    ascending: bool,
) -> pd.Series:
    """Percentile rank of col within each peer group; NaN values → 0.0."""
    if col not in df.columns or df[col].isna().all():
        return pd.Series(0.0, index=df.index)
    ranks = df.groupby(peer_group)[col].rank(
        pct=True, ascending=ascending, na_option="keep"
    )
    return ranks.fillna(0.0)


def _derive_accounting_region(df: pd.DataFrame) -> pd.Series:
    """Map each ticker's currency to its accounting standard region."""
    return df["currency"].fillna("USD").map(_CURRENCY_REGION).fillna("US_GAAP")
