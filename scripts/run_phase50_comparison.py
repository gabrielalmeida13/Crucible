#!/usr/bin/env python3
"""Phase 5.0 — Expanding-window comparison of all ranking combinations.

Tests four combinations against the pure score-based baseline using
10-fold expanding-window cross-validation (min 24m train, 12m val/fold).

Combinations
------------
  A. Full ML (13 features incl. composite_score and scorer components)
  B. Economic ML (4 features: momentum_3m, revenue_growth_yr1,
                  gross_margin_latest, revenue_acceleration)
  C. Blend 50/50 — score rank + full-ML rank averaged
  D. Blend 50/50 — score rank + economic-ML rank averaged

Each combination reports: mean NDCG@5, mean top-1 hit rate, and
improvement vs score baseline (mean ± std across 10 folds).

Gate: which combination(s) achieve ≥ 3pp hit-rate improvement at top-1?

Output
------
  data/results/phase50_ranker_validation.md  — updated with comparison table
  stdout                                     — summary table

Usage
-----
  python scripts/run_phase50_comparison.py
  python scripts/run_phase50_comparison.py --folds 5   # faster dev run
"""
from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from crucible.config import CrucibleConfig
from crucible.fetcher import fetch_sp500_tickers
from crucible.ml.ranker import (
    ECONOMIC_FEATURES,
    FEATURES,
    FORWARD_MONTHS,
    NDCG_K,
    ExpandingWindowResult,
    expanding_window_validate,
)
from crucible.snapshot import _CACHE_DIR, attach_momentum

SP500_CACHE  = _CACHE_DIR / "snapshots_SP500_201301_202412.pkl"
RESULTS_DIR  = ROOT / "data" / "results"
REPORT_PATH  = RESULTS_DIR / "phase50_ranker_validation.md"

PRICE_FETCH_START = "2012-01-01"
PRICE_FETCH_END   = "2025-12-31"
PRICE_WORKERS     = 20

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Price fetching (identical to run_phase50_ranker.py)
# ---------------------------------------------------------------------------


def _fetch_one(ticker: str, start: str, end: str) -> tuple[str, pd.Series]:
    label = "SP500" if ticker == "SPY" else ticker
    try:
        raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if raw.empty:
            return label, pd.Series(dtype=float, name=label)
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return label, close.resample("ME").last().rename(label)
    except Exception:
        log.warning("Price fetch failed for %s", ticker, exc_info=True)
        return label, pd.Series(dtype=float, name=label)


def _fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    all_t = list(tickers) + ["SPY"]
    series_map: dict[str, pd.Series] = {}
    total, done = len(all_t), 0
    with ThreadPoolExecutor(max_workers=PRICE_WORKERS) as pool:
        futures = {pool.submit(_fetch_one, t, start, end): t for t in all_t}
        for fut in as_completed(futures):
            label, s = fut.result()
            done += 1
            if done % 100 == 0 or done == total:
                log.info("Prices: %d / %d", done, total)
            if not s.empty:
                series_map[label] = s
    if not series_map:
        return pd.DataFrame()
    prices = pd.concat(series_map.values(), axis=1)
    if prices.index.tz is None:
        prices.index = prices.index.tz_localize("UTC")
    return prices


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _fmt(v: float, pct: bool = False, signed: bool = False) -> str:
    if np.isnan(v):
        return "—"
    if pct:
        s = f"{v * 100:.1f}%"
        return f"+{s}" if signed and v > 0 else s
    return f"{v:.4f}"


def _fmt_pp(v: float) -> str:
    if np.isnan(v):
        return "—"
    sign = "+" if v > 0 else ""
    return f"{sign}{v * 100:.1f} pp"


def _gate(r: ExpandingWindowResult) -> str:
    """Return gate status for ML and blend variants."""
    lines = []
    if r.beats_3pp_ml:
        lines.append(f"ML ✅ ({_fmt_pp(r.ml_hr1_delta_mean)})")
    else:
        lines.append(f"ML ❌ ({_fmt_pp(r.ml_hr1_delta_mean)})")
    if r.beats_3pp_blend:
        lines.append(f"Blend ✅ ({_fmt_pp(r.blend_hr1_delta_mean)})")
    else:
        lines.append(f"Blend ❌ ({_fmt_pp(r.blend_hr1_delta_mean)})")
    return "  ".join(lines)


def _write_report(
    full_result:  ExpandingWindowResult,
    econ_result:  ExpandingWindowResult,
    n_folds: int,
) -> None:
    now = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC")

    def _row(label: str, r: ExpandingWindowResult, variant: str) -> str:
        if variant == "score":
            ndcg  = f"{r.score_ndcg_mean:.4f} ±{r.score_ndcg_std:.4f}"
            hr1   = f"{r.score_hr1_mean * 100:.1f}% ±{r.score_hr1_std * 100:.1f}"
            delta = "baseline"
            gate  = "—"
        elif variant == "ml":
            ndcg  = f"{r.ml_ndcg_mean:.4f} ±{r.ml_ndcg_std:.4f}"
            hr1   = f"{r.ml_hr1_mean * 100:.1f}% ±{r.ml_hr1_std * 100:.1f}"
            delta = _fmt_pp(r.ml_hr1_delta_mean)
            gate  = "✅" if r.beats_3pp_ml else "❌"
        else:  # blend
            ndcg  = f"{r.blend_ndcg_mean:.4f} ±{r.blend_ndcg_std:.4f}"
            hr1   = f"{r.blend_hr1_mean * 100:.1f}% ±{r.blend_hr1_std * 100:.1f}"
            delta = _fmt_pp(r.blend_hr1_delta_mean)
            gate  = "✅" if r.beats_3pp_blend else "❌"
        return f"| {label} | {ndcg} | {hr1} | {delta} | {gate} |"

    # Determine winners
    winner_lines: list[str] = []
    for label, r, v in [
        ("Full ML", full_result, "ml"),
        ("Full + Blend 50/50", full_result, "blend"),
        ("Economic ML", econ_result, "ml"),
        ("Economic + Blend 50/50", econ_result, "blend"),
    ]:
        beats = (r.beats_3pp_ml if v == "ml" else r.beats_3pp_blend)
        if beats:
            delta = r.ml_hr1_delta_mean if v == "ml" else r.blend_hr1_delta_mean
            winner_lines.append(f"- **{label}** clears ≥ 3pp gate ({_fmt_pp(delta)} improvement)")
    if not winner_lines:
        winner_lines = ["- **No combination** clears the ≥ 3pp gate on the historical expanding-window CV."]

    per_fold_lines = [
        "",
        "### Per-Fold Detail (Full ML features)",
        "",
        "| Fold | Train window | Val window | Groups | Score HR@1 | ML HR@1 | Blend HR@1 | ML Δ |",
        "|------|-------------|-----------|--------|-----------|---------|-----------|------|",
    ]
    for f in full_result.folds:
        per_fold_lines.append(
            f"| {f.fold_idx} | {f.train_start}→{f.train_end} | "
            f"{f.val_start}→{f.val_end} | {f.n_train_groups} | "
            f"{f.score_hr1 * 100:.1f}% | {f.ml_hr1 * 100:.1f}% | "
            f"{f.blend_hr1 * 100:.1f}% | {_fmt_pp(f.ml_hr1 - f.score_hr1)} |"
        )

    econ_fold_lines = [
        "",
        "### Per-Fold Detail (Economic features)",
        "",
        "| Fold | Train window | Val window | Groups | Score HR@1 | ML HR@1 | Blend HR@1 | ML Δ |",
        "|------|-------------|-----------|--------|-----------|---------|-----------|------|",
    ]
    for f in econ_result.folds:
        econ_fold_lines.append(
            f"| {f.fold_idx} | {f.train_start}→{f.train_end} | "
            f"{f.val_start}→{f.val_end} | {f.n_train_groups} | "
            f"{f.score_hr1 * 100:.1f}% | {f.ml_hr1 * 100:.1f}% | "
            f"{f.blend_hr1 * 100:.1f}% | {_fmt_pp(f.ml_hr1 - f.score_hr1)} |"
        )

    lines: list[str] = [
        "# Phase 5.0 — LambdaMART Ranker: Expanding-Window Comparison",
        "",
        f"**Generated:** {now}  ",
        f"**Validation method:** {n_folds}-fold expanding window "
        f"(min 24 months training, 12 months validation each fold)  ",
        f"**Forward return window:** {FORWARD_MONTHS} months  ",
        f"**Ranking metric:** NDCG@{NDCG_K}, top-1 hit rate (positive 3m return)  ",
        f"**Blend weight:** 50% score + 50% ML rank positions  ",
        "",
        "---",
        "",
        "## Summary",
        "",
        "Four combinations vs the pure score-based baseline:",
        "",
        "| Combination | NDCG@5 mean ± std | HR@1 mean ± std | Δ vs baseline | 3pp gate |",
        "|-------------|------------------|----------------|---------------|---------|",
        _row("Score baseline", full_result, "score"),
        _row("A. Full ML (13 features)", full_result, "ml"),
        _row("B. Full + Blend 50/50", full_result, "blend"),
        _row("C. Economic ML (4 features)", econ_result, "ml"),
        _row("D. Economic + Blend 50/50", econ_result, "blend"),
        "",
        f"*{n_folds} folds × 12 validation months each. "
        "Mean ± std across folds (each fold's value is itself a mean across its validation months).*",
        "",
        "---",
        "",
        "## Gate Result",
        "",
        "**Historical gate (≥ 3pp HR@1 improvement vs score baseline):**",
        "",
        *winner_lines,
        "",
        "> **Important:** the historical gate is indicative only. The deployment gate",
        "> requires ≥ 3pp improvement on the **prospective held-out**",
        "> (June 2026 → December 2026 — zero-iteration data collected after model training).",
        "> Re-evaluate in December 2026.",
        "",
        "---",
        "",
        "## Feature Sets",
        "",
        "**Full features (13):** " + ", ".join(f"`{f}`" for f in FEATURES),
        "",
        "**Economic features (4) — no scorer-derived quantities:**  ",
        "`momentum_3m`, `revenue_growth_yr1`, `gross_margin_latest`, `revenue_acceleration`",
        "",
        "The economic features avoid the circular-learning risk: composite_score and",
        "scorer sub-components are derived from the same fundamentals the model would",
        "learn from, so including them may teach the model to replicate the scorer",
        "rather than discover new signal. Economic features are a cleaner test.",
        "",
        "---",
        "",
        "## Context & Caveats",
        "",
        "- Group sizes are small (9–22 per month): NDCG and HR estimates are noisy.",
        f"- Expanding window uses the same snapshot data that informed Track 2 scorer",
        "  design → results are **not** independent of researcher choices.",
        "- Phase 4.7 features (asset_growth_yoy, deferred_revenue_growth,",
        "  eps_surprise_last_q) are absent from the pre-2026 cache and imputed to 0.",
        "- The score baseline is itself well-calibrated: beating it consistently",
        "  across 10 folds is a high bar.",
        "",
        "---",
        "",
        *per_fold_lines,
        *econ_fold_lines,
    ]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Report saved: %s", REPORT_PATH)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5.0 expanding-window comparison")
    parser.add_argument("--folds", type=int, default=10,
                        help="Number of CV folds (default 10; use 5 for a faster run)")
    args = parser.parse_args()
    n_folds = args.folds

    # 1. Load snapshot cache
    if not SP500_CACHE.exists():
        log.error(
            "SP500 snapshot cache not found: %s\n"
            "Run scripts/run_backtest_track2_sp500.py first.",
            SP500_CACHE,
        )
        sys.exit(1)

    log.info("Loading SP500 snapshots from %s ...", SP500_CACHE)
    fund_by_date = joblib.load(SP500_CACHE)
    log.info("Loaded %d snapshot dates", len(fund_by_date))

    # 2. Prices
    config  = CrucibleConfig(account_currency="USD")
    tickers = fetch_sp500_tickers()
    log.info("Fetching prices for %d tickers ...", len(tickers))
    prices = _fetch_prices(tickers, PRICE_FETCH_START, PRICE_FETCH_END)
    if prices.empty:
        log.error("No prices downloaded — aborting")
        sys.exit(1)

    # 3. Momentum
    log.info("Attaching momentum ...")
    attach_momentum(fund_by_date, prices)

    # 4. Full-feature expanding-window CV (combinations A + C)
    log.info("=== Combination A: Full ML (13 features), %d folds ===", n_folds)
    full_result = expanding_window_validate(
        fund_by_date, prices, config,
        n_folds=n_folds,
        feature_names=FEATURES,
        blend_weight=0.5,
    )

    # 5. Economic-feature expanding-window CV (combinations B + D)
    log.info("=== Combination C: Economic ML (4 features), %d folds ===", n_folds)
    econ_result = expanding_window_validate(
        fund_by_date, prices, config,
        n_folds=n_folds,
        feature_names=ECONOMIC_FEATURES,
        blend_weight=0.5,
    )

    # 6. Report
    _write_report(full_result, econ_result, n_folds)

    # 7. Print summary
    W = 68
    print("\n" + "═" * W)
    print(f"  Phase 5.0 — {n_folds}-fold Expanding-Window Comparison")
    print("═" * W)
    print(f"  {'Combination':<32} {'HR@1':>7}  {'Δ vs score':>10}  {'3pp gate':>9}")
    print(f"  {'-'*32} {'-------':>7}  {'----------':>10}  {'---------':>9}")

    def _pct(v: float) -> str:
        return f"{v * 100:.1f}%" if not np.isnan(v) else "  —"

    def _pp(v: float) -> str:
        if np.isnan(v):
            return "  —"
        return f"{v * 100:+.1f} pp"

    score_hr = full_result.score_hr1_mean
    print(f"  {'Score baseline':<32} {_pct(score_hr):>7}  {'—':>10}  {'—':>9}")
    print(f"  {'A. Full ML (13 feat.)':<32} {_pct(full_result.ml_hr1_mean):>7}  "
          f"{_pp(full_result.ml_hr1_delta_mean):>10}  "
          f"{'✅' if full_result.beats_3pp_ml else '❌':>9}")
    print(f"  {'B. Full + Blend 50/50':<32} {_pct(full_result.blend_hr1_mean):>7}  "
          f"{_pp(full_result.blend_hr1_delta_mean):>10}  "
          f"{'✅' if full_result.beats_3pp_blend else '❌':>9}")
    print(f"  {'C. Economic ML (4 feat.)':<32} {_pct(econ_result.ml_hr1_mean):>7}  "
          f"{_pp(econ_result.ml_hr1_delta_mean):>10}  "
          f"{'✅' if econ_result.beats_3pp_ml else '❌':>9}")
    print(f"  {'D. Economic + Blend 50/50':<32} {_pct(econ_result.blend_hr1_mean):>7}  "
          f"{_pp(econ_result.blend_hr1_delta_mean):>10}  "
          f"{'✅' if econ_result.beats_3pp_blend else '❌':>9}")
    print("═" * W)

    winners = []
    if full_result.beats_3pp_ml:    winners.append("A")
    if full_result.beats_3pp_blend: winners.append("B")
    if econ_result.beats_3pp_ml:    winners.append("C")
    if econ_result.beats_3pp_blend: winners.append("D")

    if winners:
        print(f"\n  Combinations that clear ≥ 3pp historical gate: {', '.join(winners)}")
    else:
        print("\n  No combination clears the ≥ 3pp historical gate.")
    print(f"\n  Report: {REPORT_PATH}")
    print()


if __name__ == "__main__":
    main()
