"""Layer 2 composite quality + valuation scorer.

All metrics are scored as percentile ranks within the same GICS sector AND
accounting region peer group.  Absolute ROIC values are never compared across
sectors or regions — only relative position within the peer group matters.

Valuation (Phase 2.5a):
  P/FCF, EV/EBITDA, P/E are scored peer-relative (lower multiple = higher rank).
  Denominators use 5-year average fundamentals (Shiller-style normalisation)
  so that cyclical peaks in a single year do not distort the multiple.
  Market cap is computed from EDGAR shares × price; yfinance is not used for
  fundamentals in the backtest path.
"""

from __future__ import annotations

import logging

import pandas as pd

from crucible.config import CrucibleConfig
from crucible.fx import apply_fx_penalty

logger = logging.getLogger(__name__)

# Quality metrics: higher value → better company (all ranked ascending=True)
_QUALITY_METRICS: list[str] = ["roic_proxy_avg", "fcf_positive_years", "gross_margin_avg"]

# Sub-weights within the quality block (must sum to 1.0).
# capex_intensity is ranked ascending=False (lower = better = asset-light).
# interest_coverage, cfo_to_ni, op_margin_trend, rev_acceleration: ascending=True.
_QUALITY_WEIGHTS: dict[str, float] = {
    "roic_proxy_avg":         0.35,
    "fcf_positive_years":     0.30,
    "gross_margin_avg":       0.10,
    "interest_coverage":      0.08,
    "cfo_to_ni":              0.07,
    "capex_intensity":        0.05,
    "operating_margin_trend": 0.03,
    "revenue_acceleration":   0.02,
}

# Valuation metrics: lower value → cheaper stock → ascending=False gives cheapest rank 1.0
_VALUATION_METRICS: list[str] = ["p_fcf", "ev_ebitda", "p_e"]

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
    """Compute quality, valuation, momentum, FX and composite scores; sort descending.

    Output columns added: quality_score, valuation_score, momentum_score,
    fx_penalty, composite_score.
    Comparisons are strictly within GICS sector + accounting region peer groups.

    momentum_raw must be pre-computed by the caller (12-1 month price return).
    Tickers missing momentum_raw receive a neutral rank of 0.5.
    """
    df = df.copy()

    accounting_region = _derive_accounting_region(df)
    peer_group = df["sector"].fillna("Unknown") + "|" + accounting_region

    # --- quality percentile ranks ---
    # Base three: ascending=True (higher ROIC/FCF/margin = higher rank)
    for col in _QUALITY_METRICS:
        df[f"_qr_{col}"] = _peer_rank(df, col, peer_group, ascending=True)

    # Additional quality signals with correct direction
    df["_qr_interest_coverage"]      = _peer_rank(df, "interest_coverage",      peer_group, ascending=True)
    df["_qr_cfo_to_ni"]              = _peer_rank(df, "cfo_to_ni",              peer_group, ascending=True)
    df["_qr_capex_intensity"]        = _peer_rank(df, "capex_intensity",        peer_group, ascending=False)  # lower = better
    df["_qr_operating_margin_trend"] = _peer_rank(df, "operating_margin_trend", peer_group, ascending=True)
    df["_qr_revenue_acceleration"]   = _peer_rank(df, "revenue_acceleration",   peer_group, ascending=True)

    # --- valuation percentile ranks (lower metric → higher rank) ---
    v_rank_cols: list[str] = []
    for col in _VALUATION_METRICS:
        rc = f"_vr_{col}"
        df[rc] = _peer_rank(df, col, peer_group, ascending=False)
        v_rank_cols.append(rc)

    # --- momentum percentile rank (higher return → higher rank; NaN → neutral 0.5) ---
    df["_mr_momentum"] = _peer_rank(
        df, "momentum_raw", peer_group, ascending=True, fill_nan=0.5
    )

    # --- ML score percentile rank (higher P(outperform) → higher rank; NaN → neutral 0.5) ---
    # Only active when ml_score column is present AND config weight > 0.
    w = config.score_weights
    if "ml_score" in df.columns and w.ml_score > 0:
        df["_mr_ml_score"] = _peer_rank(
            df, "ml_score", peer_group, ascending=True, fill_nan=0.5
        )
    else:
        df["_mr_ml_score"] = pd.Series(0.5, index=df.index)

    df["quality_score"] = (
        _QUALITY_WEIGHTS["roic_proxy_avg"]         * df["_qr_roic_proxy_avg"]
        + _QUALITY_WEIGHTS["fcf_positive_years"]   * df["_qr_fcf_positive_years"]
        + _QUALITY_WEIGHTS["gross_margin_avg"]      * df["_qr_gross_margin_avg"]
        + _QUALITY_WEIGHTS["interest_coverage"]     * df["_qr_interest_coverage"]
        + _QUALITY_WEIGHTS["cfo_to_ni"]             * df["_qr_cfo_to_ni"]
        + _QUALITY_WEIGHTS["capex_intensity"]       * df["_qr_capex_intensity"]
        + _QUALITY_WEIGHTS["operating_margin_trend"] * df["_qr_operating_margin_trend"]
        + _QUALITY_WEIGHTS["revenue_acceleration"]  * df["_qr_revenue_acceleration"]
    )
    df["valuation_score"] = df[v_rank_cols].mean(axis=1)
    df["momentum_score"] = df["_mr_momentum"]
    df["ml_score_rank"] = df["_mr_ml_score"]

    df = apply_fx_penalty(df, config.account_currency, config.fx.conversion_penalty)

    # Binary signals: +0.02 if positive, 0.0 if NaN or ≤ 0. Not sector-ranked.
    buyback_adj = _binary_signal(df, "share_buyback_signal")
    insider_adj = _binary_signal(df, "insider_buy_ratio")

    df["composite_score"] = (
        w.quality * df["quality_score"]
        + w.valuation * df["valuation_score"]
        + w.momentum * df["momentum_score"]
        + w.ml_score * df["ml_score_rank"]
        + df["fx_penalty"]
        + buyback_adj
        + insider_adj
    )

    drop_cols = [
        c for c in df.columns
        if c.startswith("_qr_") or c.startswith("_vr_") or c.startswith("_mr_")
        or c == "ml_score_rank"
    ]
    df = df.drop(columns=drop_cols)

    logger.info(
        "Scored %d companies; top composite=%.3f  bottom=%.3f",
        len(df),
        df["composite_score"].max() if len(df) else float("nan"),
        df["composite_score"].min() if len(df) else float("nan"),
    )
    return df.sort_values("composite_score", ascending=False)


def _binary_signal(df: pd.DataFrame, col: str, bonus: float = 0.02) -> pd.Series:
    """Return bonus where col > 0, else 0.0. Returns all-zeros if col absent."""
    if col not in df.columns:
        return pd.Series(0.0, index=df.index)
    return df[col].apply(lambda v: bonus if pd.notna(v) and v > 0 else 0.0)


def _peer_rank(
    df: pd.DataFrame,
    col: str,
    peer_group: pd.Series,
    ascending: bool,
    fill_nan: float = 0.0,
) -> pd.Series:
    """Percentile rank of col within each peer group; NaN values → fill_nan.

    fill_nan=0.0  for quality/valuation (missing data → worst rank, penalises gaps)
    fill_nan=0.5  for momentum (missing history → neutral, no signal)
    """
    if col not in df.columns or df[col].isna().all():
        return pd.Series(fill_nan, index=df.index)
    ranks = df.groupby(peer_group)[col].rank(
        pct=True, ascending=ascending, na_option="keep"
    )
    return ranks.fillna(fill_nan)


def _derive_accounting_region(df: pd.DataFrame) -> pd.Series:
    """Map each ticker's currency to its accounting standard region."""
    return df["currency"].fillna("USD").map(_CURRENCY_REGION).fillna("US_GAAP")
