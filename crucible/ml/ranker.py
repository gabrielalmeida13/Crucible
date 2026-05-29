"""Phase 5.0 — LightGBM LambdaMART ranker for Track 2 monthly shortlist.

EXPERIMENTAL. Not integrated into production scorer or run_monthly.py.

The problem
-----------
Given the 9–22 companies that already passed Track 2 filters in a given
month, rank them so that top picks have the best 3-month forward return.

This differs from Phase 3a (which classified all 500 companies as
outperform/underperform — a hard, noisy problem). Here we rank within a
pre-filtered shortlist: a narrower problem with more signal per observation.

LambdaMART objective
--------------------
LightGBM's 'lambdarank' objective optimises NDCG directly. Each monthly
shortlist is one "query group". Relevance labels are 0–4 quintile buckets
based on within-group 3-month forward return rank.

Feature sets
------------
FEATURES (13): all Track 2 scorer outputs + raw signals + Phase 4.7 signals.
  Includes composite_score and derived scorer components. Risk: the model
  may learn to replicate the scorer rather than discover new signal.

ECONOMIC_FEATURES (4): momentum_3m, revenue_growth_yr1, gross_margin_latest,
  revenue_acceleration. Raw inputs with independent economic justification —
  no scorer-derived quantities. Avoids circular learning.

Walk-forward options
--------------------
walk_forward_validate()       — single train/val split (2013–2021 / 2022–2024)
expanding_window_validate()   — 10-fold expanding window (min 24m train, 12m val)

Deployment gate (December 2026)
-------------------------------
Do NOT integrate into production until hit-rate improvement ≥ 3pp vs
score-based ranking is confirmed on the prospective held-out
(June 2026 → December 2026 — truly clean, zero-iteration data).

Public API
----------
build_training_dataset()          — (X, y_labels, group_sizes) from snapshots
train_ranker()                    — LGBMRanker with any feature set
train_ranker_economic_features()  — LGBMRanker using only ECONOMIC_FEATURES
rank_shortlist()                  — adds ml_score to a scored shortlist
blend_rankings()                  — average score-rank + ml-rank positions
evaluate_ranker()                 — NDCG@5 and hit-rate metrics
walk_forward_validate()           — single-split pipeline
expanding_window_validate()       — 10-fold expanding-window CV
save_ranker() / load_ranker()
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    from lightgbm import LGBMRanker
    _HAS_LGBM = True
except ImportError:  # pragma: no cover
    _HAS_LGBM = False
    lgb = None          # type: ignore[assignment]
    LGBMRanker = None   # type: ignore[assignment,misc]

from crucible.backtest import _advance, _single_return
from crucible.config import CrucibleConfig
from crucible.tracks import track2_growth

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURES: list[str] = [
    "composite_score",
    "growth_quality_score",
    "momentum_score",
    "valuation_score",
    "momentum_raw",
    "momentum_3m",
    "revenue_growth_yr1",
    "revenue_acceleration",
    "gross_margin_latest",
    "fcf_trajectory",
    "asset_growth_yoy",         # Phase 4.7 — NaN in pre-2026 cache
    "deferred_revenue_growth",  # Phase 4.7 — NaN in pre-2026 cache
    "eps_surprise_last_q",      # Phase 4.7 — NaN in pre-2026 cache
]

# Four raw features with independent economic justification — no scorer-derived
# quantities, so the model cannot simply learn to replicate composite_score.
#
#  momentum_3m         — short-term price momentum: post-earnings drift, trend
#  revenue_growth_yr1  — most recent year revenue growth: direct growth signal
#  gross_margin_latest — profitability quality: structural moat proxy
#  revenue_acceleration— growth rate is rising: captures inflection point
ECONOMIC_FEATURES: list[str] = [
    "momentum_3m",
    "revenue_growth_yr1",
    "gross_margin_latest",
    "revenue_acceleration",
]

TRAIN_START_DEFAULT = pd.Timestamp("2013-01-31", tz="UTC")
TRAIN_END_DEFAULT   = pd.Timestamp("2021-12-31", tz="UTC")
VAL_START_DEFAULT   = pd.Timestamp("2022-01-31", tz="UTC")
VAL_END_DEFAULT     = pd.Timestamp("2024-12-31", tz="UTC")

FORWARD_MONTHS   = 3  # label = 3-month forward return
NDCG_K           = 5  # NDCG@5
N_RELEVANCE_BINS = 5  # quintile labels 0–4

# Conservative hyperparameters: groups of 9–22 companies are tiny, so
# we keep depth shallow and regularisation high to avoid overfitting.
_LGBM_PARAMS: dict = {
    "objective":         "lambdarank",
    "n_estimators":      300,
    "learning_rate":     0.03,
    "max_depth":         4,
    "num_leaves":        15,      # 2^4 − 1, consistent with max_depth
    "min_child_samples": 1,       # default (20) is too large for our group sizes
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.1,
    "reg_lambda":        0.1,
    "random_state":      42,
    "n_jobs":            -1,
    "verbose":           -1,
}


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class RankerArtifact:
    """Trained LambdaMART model and all metadata needed for inference."""

    model: object                        # LGBMRanker instance
    feature_names: list[str]             # all 13 attempted features
    available_features: list[str]        # non-constant features actually used
    imputation_values: dict[str, float]  # medians from training data only
    train_start: str
    train_end: str
    n_training_groups: int               # months with ≥2 priced picks
    n_training_samples: int              # total company-months


@dataclass
class MonthRecord:
    """Per-month validation outcome."""

    date: str
    n_picks: int
    score_ndcg: float
    ml_ndcg: float
    score_top1: str
    ml_top1: str
    score_top1_ret_pct: float   # % (already ×100)
    ml_top1_ret_pct: float
    score_top3_avg_ret_pct: float
    ml_top3_avg_ret_pct: float


@dataclass
class ValidationResult:
    """Aggregated validation metrics from evaluate_ranker()."""

    months: list[MonthRecord] = field(default_factory=list)
    n_months: int              = 0

    ml_ndcg_mean: float    = float("nan")
    ml_ndcg_median: float  = float("nan")
    score_ndcg_mean: float = float("nan")
    score_ndcg_median: float = float("nan")
    ndcg_improvement: float  = float("nan")

    ml_hit_rate_1: float    = float("nan")   # top-1 positive return rate
    score_hit_rate_1: float = float("nan")
    ml_hit_rate_3: float    = float("nan")   # ≥1 of top-3 positive
    score_hit_rate_3: float = float("nan")

    ml_avg_return_1: float    = float("nan")  # avg 3m return of top-1 pick
    score_avg_return_1: float = float("nan")
    ml_avg_return_3: float    = float("nan")  # avg 3m return of top-3 picks
    score_avg_return_3: float = float("nan")

    feature_importances: pd.Series = field(
        default_factory=lambda: pd.Series(dtype=float)
    )


@dataclass
class FoldResult:
    """Per-fold outcome from expanding_window_validate().

    Each metric is the mean across validation months within that fold
    (not a per-month record). This mirrors ValidationResult but is
    lighter — it stores one number per metric per fold.
    """

    fold_idx: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    n_train_groups: int   # months with ≥2 priced picks used for training
    n_val_months: int     # evaluation months in this fold's validation window

    # Fold-mean metrics for score-based, ML, and blended rankings
    score_ndcg:    float
    ml_ndcg:       float
    blend_ndcg:    float

    score_hr1:     float   # top-1 positive-3m-return rate
    ml_hr1:        float
    blend_hr1:     float

    score_avg_ret1: float  # avg 3m return of top-1 pick
    ml_avg_ret1:    float
    blend_avg_ret1: float


@dataclass
class ExpandingWindowResult:
    """Aggregated results from expanding-window cross-validation.

    Each fold trains on all data up to its cutoff and validates on the
    next val_months.  Cross-fold means and standard deviations capture
    the stability of each approach across different market regimes.
    """

    folds: list[FoldResult]
    n_folds: int
    feature_names: list[str]
    blend_weight: float

    # Cross-fold mean ± std for NDCG@5
    score_ndcg_mean: float = float("nan")
    score_ndcg_std:  float = float("nan")
    ml_ndcg_mean:    float = float("nan")
    ml_ndcg_std:     float = float("nan")
    blend_ndcg_mean: float = float("nan")
    blend_ndcg_std:  float = float("nan")

    # Cross-fold mean ± std for top-1 hit rate
    score_hr1_mean:  float = float("nan")
    score_hr1_std:   float = float("nan")
    ml_hr1_mean:     float = float("nan")
    ml_hr1_std:      float = float("nan")
    blend_hr1_mean:  float = float("nan")
    blend_hr1_std:   float = float("nan")

    # Improvement vs score baseline (mean ± std across folds)
    ml_hr1_delta_mean:    float = float("nan")
    ml_hr1_delta_std:     float = float("nan")
    blend_hr1_delta_mean: float = float("nan")
    blend_hr1_delta_std:  float = float("nan")

    # Gate: mean improvement across folds ≥ 3pp
    beats_3pp_ml:    bool = False
    beats_3pp_blend: bool = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_shortlist(
    snapshot: pd.DataFrame,
    config: CrucibleConfig,
) -> pd.DataFrame:
    """Apply Track 2 filters, positive-momentum gate, and scorer.

    Returns an empty DataFrame when nothing passes.
    """
    try:
        filtered = track2_growth.apply_filters(snapshot, config.track2_filters)
    except Exception:
        log.debug("track2 filter error at snapshot", exc_info=True)
        return pd.DataFrame()

    if "momentum_raw" not in filtered.columns or filtered.empty:
        return pd.DataFrame()

    mom_mask = filtered["momentum_raw"].notna() & (filtered["momentum_raw"] > 0)
    filtered = filtered[mom_mask]
    if filtered.empty:
        return pd.DataFrame()

    try:
        scored = track2_growth.score(filtered, config, config.track2_score_weights)
    except Exception:
        log.debug("track2 scorer error", exc_info=True)
        return pd.DataFrame()

    return scored


def _forward_returns(
    tickers: list[str],
    t0: pd.Timestamp,
    prices: pd.DataFrame,
    price_idx: pd.DatetimeIndex,
    forward_months: int,
) -> dict[str, float]:
    """Return {ticker: forward_return} for tickers with available prices."""
    t1 = _advance(t0, price_idx, forward_months)
    if t1 is None:
        return {}
    return {
        tkr: r
        for tkr in tickers
        for r in (_single_return(tkr, t0, t1, prices),)
        if r is not None
    }


def _assign_relevance_labels(
    returns: dict[str, float],
    n_bins: int = N_RELEVANCE_BINS,
) -> dict[str, int]:
    """Assign integer relevance labels 0…n_bins-1 by within-group return rank.

    Label 0 = lowest return, n_bins-1 = highest.
    Uses linear interpolation of rank into [0, n_bins-1].
    Groups of size 1 receive the middle label.
    """
    n = len(returns)
    if n == 0:
        return {}
    if n == 1:
        return {next(iter(returns)): n_bins // 2}

    tickers_asc = sorted(returns, key=lambda t: returns[t])
    return {
        tkr: round(rank / (n - 1) * (n_bins - 1))
        for rank, tkr in enumerate(tickers_asc)
    }


def _extract_features(
    shortlist: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    """Extract feature columns from a scored shortlist, filling absent ones with NaN."""
    X = pd.DataFrame(index=shortlist.index)
    for feat in feature_names:
        X[feat] = shortlist[feat].astype(float) if feat in shortlist.columns else float("nan")
    return X


def _compute_medians(X: pd.DataFrame) -> dict[str, float]:
    """Per-column median from training data; entirely-NaN columns get 0.0."""
    result: dict[str, float] = {}
    for col in X.columns:
        med = X[col].median()
        result[col] = float(med) if pd.notna(med) else 0.0
    return result


def _impute(X: pd.DataFrame, medians: dict[str, float]) -> pd.DataFrame:
    """Fill NaN with pre-computed training medians."""
    X = X.copy()
    for col in X.columns:
        if X[col].isna().any():
            fill = medians.get(col, 0.0)
            X[col] = X[col].fillna(float(fill) if pd.notna(fill) else 0.0)
    return X


def _ndcg_at_k(
    ranked_returns: list[float],
    all_returns: list[float],
    k: int,
) -> float:
    """NDCG@k where relevance = 3m return normalised to [0, 1] within the group.

    ranked_returns — returns in the order produced by the ranking being evaluated
    all_returns    — all returns in the group (used to build ideal ranking)
    """
    n = len(all_returns)
    if n == 0:
        return float("nan")
    k_eff = min(k, n)

    r_min = min(all_returns)
    r_max = max(all_returns)
    if r_max == r_min:
        return 1.0  # all equally relevant; any ranking is perfect

    def _rel(r: float) -> float:
        return (r - r_min) / (r_max - r_min)

    dcg = sum(
        _rel(ranked_returns[i]) / np.log2(i + 2)
        for i in range(min(k_eff, len(ranked_returns)))
    )
    ideal = sorted(all_returns, reverse=True)[:k_eff]
    idcg = sum(_rel(ideal[i]) / np.log2(i + 2) for i in range(len(ideal)))

    return dcg / idcg if idcg > 0 else 1.0


def _blend_rank_tickers(
    score_ranked: list[str],
    ml_ranked: list[str],
    weight: float,
) -> list[str]:
    """Return tickers sorted by weighted average of 1-indexed rank positions.

    weight applies to the ML ranking (0 = pure score, 1 = pure ML, 0.5 = equal).
    Tickers absent from one list are placed last in that ordering.
    """
    all_tickers = list(dict.fromkeys(score_ranked + ml_ranked))
    n = len(all_tickers)
    score_pos = {t: i + 1 for i, t in enumerate(score_ranked)}
    ml_pos    = {t: i + 1 for i, t in enumerate(ml_ranked)}
    blend: dict[str, float] = {
        t: (1 - weight) * score_pos.get(t, n) + weight * ml_pos.get(t, n)
        for t in all_tickers
    }
    return sorted(all_tickers, key=lambda t: blend[t])


def _eval_month(
    shortlist_ok: pd.DataFrame,
    fwd: dict[str, float],
    artifact: RankerArtifact,
    blend_weight: float,
    k: int,
) -> dict:
    """Evaluate score / ML / blend rankings for one month; return metrics dict."""
    tickers_ok = [t for t in shortlist_ok.index if t in fwd]
    all_rets = [fwd[t] for t in tickers_ok]

    score_ranked = tickers_ok                                   # composite_score order
    ml_sl        = rank_shortlist(shortlist_ok, artifact)
    ml_ranked    = [t for t in ml_sl.index if t in fwd]
    blend_ranked = _blend_rank_tickers(score_ranked, ml_ranked, blend_weight)

    def _ndcg(ranked: list[str]) -> float:
        rets_in_order = [fwd[t] for t in ranked if t in fwd]
        return _ndcg_at_k(rets_in_order, all_rets, k)

    def _hr1(ranked: list[str]) -> float | None:
        t = ranked[0] if ranked else None
        r = fwd.get(t, float("nan")) if t else float("nan")
        return None if np.isnan(r) else float(r > 0)

    def _ret1(ranked: list[str]) -> float | None:
        t = ranked[0] if ranked else None
        r = fwd.get(t, float("nan")) if t else float("nan")
        return None if np.isnan(r) else r

    return {
        "score_ndcg":    _ndcg(score_ranked),
        "ml_ndcg":       _ndcg(ml_ranked),
        "blend_ndcg":    _ndcg(blend_ranked),
        "score_hr1":     _hr1(score_ranked),
        "ml_hr1":        _hr1(ml_ranked),
        "blend_hr1":     _hr1(blend_ranked),
        "score_ret1":    _ret1(score_ranked),
        "ml_ret1":       _ret1(ml_ranked),
        "blend_ret1":    _ret1(blend_ranked),
    }


# ---------------------------------------------------------------------------
# Public API — data preparation
# ---------------------------------------------------------------------------


def build_training_dataset(
    fund_by_date: dict[pd.Timestamp, pd.DataFrame],
    prices: pd.DataFrame,
    config: CrucibleConfig,
    start: pd.Timestamp,
    end: pd.Timestamp,
    forward_months: int = FORWARD_MONTHS,
    feature_names: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[int]]:
    """Build (X, y_labels, group_sizes) for LambdaMART training.

    For each month in [start, end] where Track 2 produces a shortlist:
      1. Run filters + scorer to get scored shortlist.
      2. Compute 3-month forward returns; drop companies with no price.
      3. Assign integer relevance labels 0–4 by within-group return rank.

    Months with < 2 priced companies are skipped (can't form a ranking).

    Returns
    -------
    X           : DataFrame (n_samples × n_features) — raw, not imputed
    y_labels    : Series of int labels (0 = worst, 4 = best within group)
    group_sizes : list[int] parallel to groups in X, required by LGBMRanker
    """
    if feature_names is None:
        feature_names = FEATURES

    price_idx = prices.index
    dates = sorted(d for d in fund_by_date if start <= d <= end)

    all_X: list[pd.DataFrame] = []
    all_y: list[int] = []
    group_sizes: list[int] = []
    n_skipped_no_fwd = 0

    for date in dates:
        shortlist = _get_shortlist(fund_by_date[date], config)
        if shortlist.empty:
            continue

        tickers = shortlist.index.tolist()
        fwd = _forward_returns(tickers, date, prices, price_idx, forward_months)
        if len(fwd) < 2:
            n_skipped_no_fwd += 1
            continue

        tickers_ok = [t for t in tickers if t in fwd]
        shortlist_ok = shortlist.loc[tickers_ok]

        labels = _assign_relevance_labels(fwd, N_RELEVANCE_BINS)
        X_month = _extract_features(shortlist_ok, feature_names)
        y_month = [labels[t] for t in tickers_ok]

        all_X.append(X_month)
        all_y.extend(y_month)
        group_sizes.append(len(tickers_ok))

    if not all_X:
        log.warning("build_training_dataset: no usable months in [%s, %s]", start, end)
        return pd.DataFrame(columns=feature_names), pd.Series(dtype=int), []

    X = pd.concat(all_X, axis=0)
    y = pd.Series(all_y, index=X.index, name="relevance", dtype=int)

    log.info(
        "build_training_dataset: %d months used, %d samples, %d skipped (no fwd price)  [%s → %s]",
        len(group_sizes), len(X), n_skipped_no_fwd,
        start.strftime("%Y-%m"), end.strftime("%Y-%m"),
    )
    return X, y, group_sizes


# ---------------------------------------------------------------------------
# Public API — training
# ---------------------------------------------------------------------------


def train_ranker(
    X: pd.DataFrame,
    y: pd.Series,
    groups: list[int],
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
) -> RankerArtifact:
    """Train a LGBMRanker (LambdaMART) and return a RankerArtifact.

    Uses the last 20% of groups as an early-stopping validation set.
    Imputation medians are computed strictly from X (no leakage).

    Raises ImportError if lightgbm is not installed.
    """
    if not _HAS_LGBM:
        raise ImportError(
            "lightgbm is required for the Phase 5.0 ranker. "
            "Install: uv add lightgbm"
        )

    imputation_values = _compute_medians(X)
    X_imp = _impute(X, imputation_values)

    # Drop features that are constant after imputation — LightGBM ignores
    # them anyway but being explicit makes the artifact cleaner.
    available_features = [
        col for col in X_imp.columns if X_imp[col].nunique() > 1
    ]
    if not available_features:
        raise ValueError(
            "All features are constant after imputation. "
            "Check snapshot coverage and feature extraction."
        )

    n_val_groups  = max(1, len(groups) // 5)
    n_train_groups = len(groups) - n_val_groups
    split_idx = sum(groups[:n_train_groups])

    X_tr = X_imp.iloc[:split_idx][available_features]
    y_tr = y.iloc[:split_idx]
    X_vl = X_imp.iloc[split_idx:][available_features]
    y_vl = y.iloc[split_idx:]
    grp_tr = groups[:n_train_groups]
    grp_vl = groups[n_train_groups:]

    model = LGBMRanker(**_LGBM_PARAMS)

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=-1),  # suppress per-iteration output
    ]

    model.fit(
        X_tr, y_tr,
        group=grp_tr,
        eval_set=[(X_vl, y_vl)],
        eval_group=[grp_vl],
        callbacks=callbacks,
    )

    best_iter = getattr(model, "best_iteration_", None) or _LGBM_PARAMS["n_estimators"]
    log.info(
        "LGBMRanker trained: best_iter=%d  train_groups=%d  val_groups=%d  features=%d",
        best_iter, n_train_groups, n_val_groups, len(available_features),
    )

    return RankerArtifact(
        model=model,
        feature_names=list(X.columns),
        available_features=available_features,
        imputation_values=imputation_values,
        train_start=train_start.strftime("%Y-%m-%d"),
        train_end=train_end.strftime("%Y-%m-%d"),
        n_training_groups=len(groups),
        n_training_samples=len(X),
    )


# ---------------------------------------------------------------------------
# Public API — inference
# ---------------------------------------------------------------------------


def rank_shortlist(
    shortlist: pd.DataFrame,
    artifact: RankerArtifact,
) -> pd.DataFrame:
    """Add ml_score to a scored Track 2 shortlist, sorted ml_score descending.

    The shortlist must be the output of track2_growth.score() so that
    composite_score and scorer sub-components are present.

    This DOES NOT modify composite_score or any production column.
    ml_score is added purely for experimental comparison.
    """
    if shortlist.empty:
        return shortlist.copy()

    X = _extract_features(shortlist, artifact.feature_names)
    X_imp = _impute(X, artifact.imputation_values)
    X_pred = X_imp[artifact.available_features]

    try:
        scores = artifact.model.predict(X_pred)
    except Exception:
        log.warning("rank_shortlist: model.predict failed", exc_info=True)
        result = shortlist.copy()
        result["ml_score"] = float("nan")
        return result

    result = shortlist.copy()
    result["ml_score"] = scores
    return result.sort_values("ml_score", ascending=False)


# ---------------------------------------------------------------------------
# Public API — evaluation
# ---------------------------------------------------------------------------


def evaluate_ranker(
    artifact: RankerArtifact,
    fund_by_date: dict[pd.Timestamp, pd.DataFrame],
    prices: pd.DataFrame,
    config: CrucibleConfig,
    start: pd.Timestamp,
    end: pd.Timestamp,
    forward_months: int = FORWARD_MONTHS,
    k: int = NDCG_K,
) -> ValidationResult:
    """Evaluate ranker on a date range and return ValidationResult.

    For each month with a priced Track 2 shortlist, computes:
      - NDCG@k for the ML ranking and score-based ranking
      - Top-1 and top-3 hit rates and average 3m returns
    """
    price_idx = prices.index
    dates = sorted(d for d in fund_by_date if start <= d <= end)

    month_records: list[MonthRecord] = []
    ml_ndcgs:     list[float] = []
    score_ndcgs:  list[float] = []
    ml_hr1:   list[bool]  = []
    score_hr1: list[bool] = []
    ml_hr3:   list[bool]  = []
    score_hr3: list[bool] = []
    ml_ret1:   list[float] = []
    score_ret1: list[float] = []
    ml_ret3:   list[float] = []
    score_ret3: list[float] = []

    for date in dates:
        shortlist = _get_shortlist(fund_by_date[date], config)
        if shortlist.empty:
            continue

        tickers = shortlist.index.tolist()
        fwd = _forward_returns(tickers, date, prices, price_idx, forward_months)
        if len(fwd) < 2:
            continue

        tickers_ok = [t for t in tickers if t in fwd]
        shortlist_ok = shortlist.loc[tickers_ok]
        all_rets = [fwd[t] for t in tickers_ok]

        # Score-based ordering: shortlist is already sorted by composite_score desc
        score_ranked = tickers_ok

        # ML ordering
        ml_shortlist = rank_shortlist(shortlist_ok, artifact)
        ml_ranked = [t for t in ml_shortlist.index if t in fwd]

        # NDCG@k
        score_ret_ordered = [fwd[t] for t in score_ranked if t in fwd]
        ml_ret_ordered    = [fwd[t] for t in ml_ranked    if t in fwd]

        s_ndcg = _ndcg_at_k(score_ret_ordered, all_rets, k)
        m_ndcg = _ndcg_at_k(ml_ret_ordered,    all_rets, k)

        if not np.isnan(s_ndcg):
            score_ndcgs.append(s_ndcg)
        if not np.isnan(m_ndcg):
            ml_ndcgs.append(m_ndcg)

        # Top-1 metrics
        s1 = score_ranked[0] if score_ranked else None
        m1 = ml_ranked[0]    if ml_ranked    else None
        s1_ret = fwd.get(s1, float("nan")) if s1 else float("nan")
        m1_ret = fwd.get(m1, float("nan")) if m1 else float("nan")

        if not np.isnan(s1_ret):
            score_hr1.append(s1_ret > 0)
            score_ret1.append(s1_ret)
        if not np.isnan(m1_ret):
            ml_hr1.append(m1_ret > 0)
            ml_ret1.append(m1_ret)

        # Top-3 metrics
        s3_rets = [fwd[t] for t in score_ranked[:3] if t in fwd]
        m3_rets = [fwd[t] for t in ml_ranked[:3]    if t in fwd]
        if s3_rets:
            score_hr3.append(any(r > 0 for r in s3_rets))
            score_ret3.append(float(np.mean(s3_rets)))
        if m3_rets:
            ml_hr3.append(any(r > 0 for r in m3_rets))
            ml_ret3.append(float(np.mean(m3_rets)))

        month_records.append(MonthRecord(
            date=date.strftime("%Y-%m"),
            n_picks=len(tickers_ok),
            score_ndcg=round(s_ndcg, 4) if not np.isnan(s_ndcg) else float("nan"),
            ml_ndcg=round(m_ndcg, 4)    if not np.isnan(m_ndcg) else float("nan"),
            score_top1=s1 or "—",
            ml_top1=m1 or "—",
            score_top1_ret_pct=round(s1_ret * 100, 2) if not np.isnan(s1_ret) else float("nan"),
            ml_top1_ret_pct=round(m1_ret * 100, 2)    if not np.isnan(m1_ret) else float("nan"),
            score_top3_avg_ret_pct=round(float(np.mean(s3_rets)) * 100, 2) if s3_rets else float("nan"),
            ml_top3_avg_ret_pct=round(float(np.mean(m3_rets)) * 100, 2)    if m3_rets else float("nan"),
        ))

    def _mean(lst: list) -> float:
        return float(np.nanmean(lst)) if lst else float("nan")

    def _hr(lst: list[bool]) -> float:
        return float(np.mean(lst)) if lst else float("nan")

    # Feature importances from trained model
    fi = pd.Series(
        artifact.model.feature_importances_,
        index=artifact.available_features,
    ).sort_values(ascending=False)

    return ValidationResult(
        months=month_records,
        n_months=len(month_records),
        ml_ndcg_mean=_mean(ml_ndcgs),
        ml_ndcg_median=float(np.median(ml_ndcgs)) if ml_ndcgs else float("nan"),
        score_ndcg_mean=_mean(score_ndcgs),
        score_ndcg_median=float(np.median(score_ndcgs)) if score_ndcgs else float("nan"),
        ndcg_improvement=_mean(ml_ndcgs) - _mean(score_ndcgs),
        ml_hit_rate_1=_hr(ml_hr1),
        score_hit_rate_1=_hr(score_hr1),
        ml_hit_rate_3=_hr(ml_hr3),
        score_hit_rate_3=_hr(score_hr3),
        ml_avg_return_1=_mean(ml_ret1),
        score_avg_return_1=_mean(score_ret1),
        ml_avg_return_3=_mean(ml_ret3),
        score_avg_return_3=_mean(score_ret3),
        feature_importances=fi,
    )


# ---------------------------------------------------------------------------
# Public API — pipeline
# ---------------------------------------------------------------------------


def walk_forward_validate(
    fund_by_date: dict[pd.Timestamp, pd.DataFrame],
    prices: pd.DataFrame,
    config: CrucibleConfig,
    train_start: pd.Timestamp = TRAIN_START_DEFAULT,
    train_end: pd.Timestamp   = TRAIN_END_DEFAULT,
    val_start: pd.Timestamp   = VAL_START_DEFAULT,
    val_end: pd.Timestamp     = VAL_END_DEFAULT,
    forward_months: int       = FORWARD_MONTHS,
    k: int                    = NDCG_K,
) -> tuple[RankerArtifact, ValidationResult]:
    """Full train-then-validate pipeline.

    Returns (artifact, validation_result).

    The artifact is suitable for:
      - December 2026: prospective held-out evaluation on June–December 2026 data
      - Comparison in the dashboard (experimental, not production)
    """
    if not _HAS_LGBM:
        raise ImportError(
            "lightgbm is required. Install: uv add lightgbm"
        )

    log.info(
        "Phase 5.0: building training dataset [%s → %s] forward=%dm ...",
        train_start.strftime("%Y-%m"), train_end.strftime("%Y-%m"), forward_months,
    )
    X_train, y_train, groups_train = build_training_dataset(
        fund_by_date, prices, config,
        start=train_start, end=train_end,
        forward_months=forward_months,
    )

    if X_train.empty:
        raise ValueError(
            "No training data produced. Check snapshot coverage and "
            "Track 2 filter thresholds for the training window."
        )

    log.info(
        "Phase 5.0: training LGBMRanker on %d samples (%d groups) ...",
        len(X_train), len(groups_train),
    )
    artifact = train_ranker(X_train, y_train, groups_train, train_start, train_end)

    log.info(
        "Phase 5.0: evaluating on validation set [%s → %s] ...",
        val_start.strftime("%Y-%m"), val_end.strftime("%Y-%m"),
    )
    result = evaluate_ranker(
        artifact, fund_by_date, prices, config,
        start=val_start, end=val_end,
        forward_months=forward_months, k=k,
    )

    log.info(
        "Phase 5.0 validation: %d months | ML NDCG@%d=%.4f (baseline=%.4f Δ=%+.4f) | "
        "HR@1 ML=%.1f%% baseline=%.1f%% (Δ=%+.1f pp)",
        result.n_months, k,
        result.ml_ndcg_mean, result.score_ndcg_mean, result.ndcg_improvement,
        result.ml_hit_rate_1 * 100,
        result.score_hit_rate_1 * 100,
        (result.ml_hit_rate_1 - result.score_hit_rate_1) * 100,
    )
    return artifact, result


# ---------------------------------------------------------------------------
# Public API — economic-features trainer
# ---------------------------------------------------------------------------


def train_ranker_economic_features(
    fund_by_date: dict[pd.Timestamp, pd.DataFrame],
    prices: pd.DataFrame,
    config: CrucibleConfig,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    forward_months: int = FORWARD_MONTHS,
) -> RankerArtifact:
    """Train a LGBMRanker using only ECONOMIC_FEATURES.

    Uses four raw signals with independent economic justification:
      momentum_3m, revenue_growth_yr1, gross_margin_latest, revenue_acceleration

    These are absent from composite_score's construction, so the model
    cannot learn a trivial mapping from scorer outputs back to itself.
    This is the cleanest test of whether the fundamentals alone carry
    ranking signal beyond the hand-tuned scorer weights.

    See also: train_ranker() for the full 13-feature variant.
    """
    if not _HAS_LGBM:
        raise ImportError("lightgbm is required. Install: uv add lightgbm")

    log.info(
        "Building economic-features training dataset [%s → %s] ...",
        train_start.strftime("%Y-%m"), train_end.strftime("%Y-%m"),
    )
    X, y, groups = build_training_dataset(
        fund_by_date, prices, config,
        start=train_start, end=train_end,
        forward_months=forward_months,
        feature_names=ECONOMIC_FEATURES,
    )
    if X.empty:
        raise ValueError(
            "No training data for economic features. "
            "Check snapshot coverage and Track 2 filter thresholds."
        )
    log.info(
        "Economic-features training: %d samples (%d groups), features=%s",
        len(X), len(groups), ECONOMIC_FEATURES,
    )
    return train_ranker(X, y, groups, train_start, train_end)


# ---------------------------------------------------------------------------
# Public API — rank blending
# ---------------------------------------------------------------------------


def blend_rankings(
    score_df: pd.DataFrame,
    ml_df: pd.DataFrame,
    weight: float = 0.5,
) -> pd.DataFrame:
    """Combine score-based and ML rankings by averaging rank positions.

    Parameters
    ----------
    score_df : shortlist sorted by composite_score descending (standard scorer output)
    ml_df    : same tickers sorted by ml_score descending (output of rank_shortlist)
    weight   : fraction of the final rank from the ML ordering
               0.0 = pure score, 1.0 = pure ML, 0.5 = equal blend (default)

    Returns
    -------
    DataFrame with all columns from score_df, plus ml_score (from ml_df),
    score_rank, ml_rank, and blend_rank, sorted by blend_rank ascending
    (lower blend_rank = better combined position).

    Ties in blend_rank are broken by score_rank (i.e., score-based order wins ties).
    """
    if score_df.empty:
        return score_df.copy()

    tickers = list(score_df.index)
    n = len(tickers)

    score_pos: dict[str, int] = {t: i + 1 for i, t in enumerate(tickers)}
    ml_pos:    dict[str, int] = {
        t: i + 1 for i, t in enumerate(ml_df.index) if t in score_pos
    }

    result = score_df.copy()

    # Carry ml_score across if present
    if "ml_score" in ml_df.columns:
        result["ml_score"] = ml_df.reindex(result.index)["ml_score"]

    result["score_rank"] = result.index.map(lambda t: score_pos.get(t, n))
    result["ml_rank"]    = result.index.map(lambda t: ml_pos.get(t, n))
    result["blend_rank"] = (
        (1 - weight) * result["score_rank"] + weight * result["ml_rank"]
    )

    return result.sort_values(["blend_rank", "score_rank"], ascending=True)


# ---------------------------------------------------------------------------
# Public API — expanding-window cross-validation
# ---------------------------------------------------------------------------


def expanding_window_validate(
    fund_by_date: dict[pd.Timestamp, pd.DataFrame],
    prices: pd.DataFrame,
    config: CrucibleConfig,
    n_folds: int = 10,
    min_train_months: int = 24,
    val_months: int = 12,
    forward_months: int = FORWARD_MONTHS,
    k: int = NDCG_K,
    feature_names: list[str] | None = None,
    blend_weight: float = 0.5,
) -> ExpandingWindowResult:
    """10-fold expanding-window cross-validation of the LambdaMART ranker.

    Fold layout (default: 10 folds, 24-month min-train, 12-month val)
    -----------------------------------------------------------------
    Fold 0 : train [dates 0..23]   → val [dates 24..35]
    Fold 1 : train [dates 0..35]   → val [dates 36..47]
    ...
    Fold 9 : train [dates 0..131]  → val [dates 132..143]

    For 144 available dates (2013-01 → 2024-12) this covers the full range
    with training expanding by val_months each fold.

    A fresh model is trained on each fold's training window (no data leakage
    across folds). The last forward_months of each training window may have
    incomplete labels; build_training_dataset handles this gracefully.

    Each fold reports mean metrics across its validation months:
      - NDCG@k for score-based, ML, and 50/50 blend rankings
      - Top-1 hit rate (positive 3m return) for all three

    Returns ExpandingWindowResult with per-fold details and cross-fold
    mean ± std, plus a boolean flag for whether mean improvement ≥ 3pp.

    Parameters
    ----------
    feature_names : which features to use (defaults to FEATURES).
                    Pass ECONOMIC_FEATURES to test the no-circular variant.
    blend_weight  : ML weight in the blend (0.5 = equal score + ML).
    """
    if not _HAS_LGBM:
        raise ImportError("lightgbm is required. Install: uv add lightgbm")

    if feature_names is None:
        feature_names = FEATURES

    dates     = sorted(fund_by_date.keys())
    price_idx = prices.index
    n_dates   = len(dates)

    # Validate that we have enough months for the requested folds
    required = min_train_months + n_folds * val_months
    if n_dates < required:
        raise ValueError(
            f"Not enough snapshot dates for {n_folds} folds: "
            f"need {required}, have {n_dates}. "
            f"Reduce n_folds or min_train_months."
        )

    fold_results: list[FoldResult] = []

    for fold_idx in range(n_folds):
        train_end_idx = min_train_months + fold_idx * val_months - 1
        val_start_idx = train_end_idx + 1
        val_end_idx   = val_start_idx + val_months - 1

        train_start_ts = dates[0]
        train_end_ts   = dates[train_end_idx]
        val_start_ts   = dates[val_start_idx]
        val_end_ts     = dates[val_end_idx]

        log.info(
            "Fold %d/%d  train [%s → %s]  val [%s → %s]",
            fold_idx + 1, n_folds,
            train_start_ts.strftime("%Y-%m"), train_end_ts.strftime("%Y-%m"),
            val_start_ts.strftime("%Y-%m"),   val_end_ts.strftime("%Y-%m"),
        )

        # ── Train ────────────────────────────────────────────────────────────
        X, y, groups = build_training_dataset(
            fund_by_date, prices, config,
            start=train_start_ts, end=train_end_ts,
            forward_months=forward_months,
            feature_names=feature_names,
        )
        if X.empty or len(groups) < 4:
            log.warning("Fold %d: insufficient training data — skipping", fold_idx)
            continue

        artifact = train_ranker(X, y, groups, train_start_ts, train_end_ts)

        # ── Evaluate ─────────────────────────────────────────────────────────
        val_dates = [d for d in dates[val_start_idx : val_end_idx + 1]]

        col_lists: dict[str, list[float]] = {
            k: [] for k in (
                "score_ndcg", "ml_ndcg", "blend_ndcg",
                "score_hr1",  "ml_hr1",  "blend_hr1",
                "score_ret1", "ml_ret1", "blend_ret1",
            )
        }
        n_val_used = 0

        for date in val_dates:
            shortlist = _get_shortlist(fund_by_date[date], config)
            if shortlist.empty:
                continue
            tickers = shortlist.index.tolist()
            fwd = _forward_returns(tickers, date, prices, price_idx, forward_months)
            if len(fwd) < 2:
                continue

            tickers_ok   = [t for t in tickers if t in fwd]
            shortlist_ok = shortlist.loc[tickers_ok]

            m = _eval_month(shortlist_ok, fwd, artifact, blend_weight, k)
            n_val_used += 1
            for key in col_lists:
                v = m[key]
                if v is not None and not np.isnan(v):
                    col_lists[key].append(v)

        def _fold_mean(key: str) -> float:
            lst = col_lists[key]
            return float(np.mean(lst)) if lst else float("nan")

        fold_results.append(FoldResult(
            fold_idx=fold_idx,
            train_start=train_start_ts.strftime("%Y-%m"),
            train_end=train_end_ts.strftime("%Y-%m"),
            val_start=val_start_ts.strftime("%Y-%m"),
            val_end=val_end_ts.strftime("%Y-%m"),
            n_train_groups=len(groups),
            n_val_months=n_val_used,
            score_ndcg=_fold_mean("score_ndcg"),
            ml_ndcg=_fold_mean("ml_ndcg"),
            blend_ndcg=_fold_mean("blend_ndcg"),
            score_hr1=_fold_mean("score_hr1"),
            ml_hr1=_fold_mean("ml_hr1"),
            blend_hr1=_fold_mean("blend_hr1"),
            score_avg_ret1=_fold_mean("score_ret1"),
            ml_avg_ret1=_fold_mean("ml_ret1"),
            blend_avg_ret1=_fold_mean("blend_ret1"),
        ))

    if not fold_results:
        raise ValueError("No folds produced results — check snapshot and price coverage.")

    # ── Aggregate across folds ───────────────────────────────────────────────
    def _agg(attr: str) -> tuple[float, float]:
        vals = [getattr(f, attr) for f in fold_results if not np.isnan(getattr(f, attr))]
        if not vals:
            return float("nan"), float("nan")
        return float(np.mean(vals)), float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)

    ml_hr1_deltas    = [f.ml_hr1    - f.score_hr1 for f in fold_results
                        if not (np.isnan(f.ml_hr1) or np.isnan(f.score_hr1))]
    blend_hr1_deltas = [f.blend_hr1 - f.score_hr1 for f in fold_results
                        if not (np.isnan(f.blend_hr1) or np.isnan(f.score_hr1))]

    def _delta_agg(deltas: list[float]) -> tuple[float, float]:
        if not deltas:
            return float("nan"), float("nan")
        return float(np.mean(deltas)), float(np.std(deltas, ddof=1) if len(deltas) > 1 else 0.0)

    ml_dm,    ml_ds    = _delta_agg(ml_hr1_deltas)
    blend_dm, blend_ds = _delta_agg(blend_hr1_deltas)

    sn_m, sn_s = _agg("score_ndcg");  mn_m, mn_s = _agg("ml_ndcg");   bn_m, bn_s = _agg("blend_ndcg")
    sh_m, sh_s = _agg("score_hr1");   mh_m, mh_s = _agg("ml_hr1");    bh_m, bh_s = _agg("blend_hr1")

    result = ExpandingWindowResult(
        folds=fold_results,
        n_folds=len(fold_results),
        feature_names=feature_names,
        blend_weight=blend_weight,
        score_ndcg_mean=sn_m,  score_ndcg_std=sn_s,
        ml_ndcg_mean=mn_m,     ml_ndcg_std=mn_s,
        blend_ndcg_mean=bn_m,  blend_ndcg_std=bn_s,
        score_hr1_mean=sh_m,   score_hr1_std=sh_s,
        ml_hr1_mean=mh_m,      ml_hr1_std=mh_s,
        blend_hr1_mean=bh_m,   blend_hr1_std=bh_s,
        ml_hr1_delta_mean=ml_dm,       ml_hr1_delta_std=ml_ds,
        blend_hr1_delta_mean=blend_dm, blend_hr1_delta_std=blend_ds,
        beats_3pp_ml=ml_dm >= 0.03    if not np.isnan(ml_dm)    else False,
        beats_3pp_blend=blend_dm >= 0.03 if not np.isnan(blend_dm) else False,
    )

    log.info(
        "Expanding-window CV (%d folds) | "
        "ML HR@1: %.1f%%±%.1f (Δ%+.1f pp) | "
        "Blend HR@1: %.1f%%±%.1f (Δ%+.1f pp) | "
        "Score HR@1: %.1f%%",
        result.n_folds,
        result.ml_hr1_mean * 100,    result.ml_hr1_std * 100,
        result.ml_hr1_delta_mean * 100,
        result.blend_hr1_mean * 100, result.blend_hr1_std * 100,
        result.blend_hr1_delta_mean * 100,
        result.score_hr1_mean * 100,
    )
    return result


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_ranker(artifact: RankerArtifact, path: Path) -> None:
    """Pickle the RankerArtifact to path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(artifact, fh)
    log.info("Ranker saved: %s", path)


def load_ranker(path: Path) -> RankerArtifact:
    """Load a pickled RankerArtifact from path."""
    with open(path, "rb") as fh:
        return pickle.load(fh)  # noqa: S301
