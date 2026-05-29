#!/usr/bin/env python3
"""Phase 5.0 — LambdaMART ranker walk-forward validation.

Trains a LightGBM LambdaMART ranker on Track 2 monthly shortlists from
2013-01 → 2021-12 (training) and validates on 2022-01 → 2024-12 (held-out).

The key metric is NDCG@5 and hit-rate improvement vs the score-based ranking.
Deployment gate (December 2026): hit rate ≥ score-based + 3pp on prospective
held-out (June 2026 → December 2026). This script evaluates historical only.

Inputs (required):
  data/cache/snapshots_SP500_201301_202412.pkl   — pre-built snapshot cache
  yfinance prices (downloaded on first run, ~15–30 min for SP500)

Outputs:
  data/results/phase50_ranker_validation.md   — full validation report
  data/models/phase50_ranker.pkl              — trained artifact for Dec held-out

Usage:
  python scripts/run_phase50_ranker.py
  python scripts/run_phase50_ranker.py --no-cache     # force snapshot rebuild
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
    FEATURES,
    FORWARD_MONTHS,
    NDCG_K,
    TRAIN_END_DEFAULT,
    TRAIN_START_DEFAULT,
    VAL_END_DEFAULT,
    VAL_START_DEFAULT,
    MonthRecord,
    ValidationResult,
    save_ranker,
    walk_forward_validate,
)
from crucible.snapshot import _CACHE_DIR, attach_momentum

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SP500_CACHE  = _CACHE_DIR / "snapshots_SP500_201301_202412.pkl"
RESULTS_DIR  = ROOT / "data" / "results"
MODELS_DIR   = ROOT / "data" / "models"
REPORT_PATH  = RESULTS_DIR / "phase50_ranker_validation.md"
MODEL_PATH   = MODELS_DIR  / "phase50_ranker.pkl"

PRICE_FETCH_START = "2012-01-01"
PRICE_FETCH_END   = "2025-12-31"   # covers 3m forward from 2024-12-31
PRICE_WORKERS     = 20
BENCHMARK_COL     = "SP500"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Price fetching
# ---------------------------------------------------------------------------


def _fetch_one(ticker: str, start: str, end: str) -> tuple[str, pd.Series]:
    label = BENCHMARK_COL if ticker == "SPY" else ticker
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
    log.info("Price matrix: %d rows × %d cols", len(prices), len(prices.columns))
    return prices


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _fmt_pct(v: float, decimals: int = 2) -> str:
    if np.isnan(v):
        return "—"
    return f"{v * 100:.{decimals}f}%"


def _fmt_pp(v: float) -> str:
    if np.isnan(v):
        return "—"
    return f"{v * 100:+.2f} pp"


def _fmt_f(v: float, decimals: int = 4) -> str:
    if np.isnan(v):
        return "—"
    return f"{v:.{decimals}f}"


def _generate_report(result: ValidationResult, output_path: Path) -> None:
    """Write the full validation Markdown report."""
    now = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC")
    hr1_delta = result.ml_hit_rate_1 - result.score_hit_rate_1
    hr3_delta = result.ml_hit_rate_3 - result.score_hit_rate_3
    ndcg_delta = result.ndcg_improvement

    # Gate check: does historical validation clear the 3pp threshold?
    # (This is indicative only — the real gate is on June–December 2026 prospective)
    gate_hr1 = hr1_delta * 100 >= 3.0
    gate_note = (
        "✅ CLEARS historical gate (≥ 3pp improvement at top-1)"
        if gate_hr1
        else "❌ DOES NOT clear historical gate (< 3pp at top-1) — "
             "await December 2026 prospective held-out before deciding"
    )

    lines: list[str] = [
        "# Phase 5.0 — LambdaMART Ranker Validation Report",
        "",
        f"**Generated:** {now}  ",
        f"**Model:** LightGBM LambdaMART (`lambdarank` objective)  ",
        f"**Training period:** {TRAIN_START_DEFAULT.strftime('%Y-%m-%d')} → "
        f"{TRAIN_END_DEFAULT.strftime('%Y-%m-%d')}  ",
        f"**Validation period:** {VAL_START_DEFAULT.strftime('%Y-%m-%d')} → "
        f"{VAL_END_DEFAULT.strftime('%Y-%m-%d')}  ",
        f"**Forward return window:** {FORWARD_MONTHS} months  ",
        f"**Evaluation metric:** NDCG@{NDCG_K}  ",
        "",
        "---",
        "",
        "## Context",
        "",
        "This experiment tests whether a LambdaMART model can improve the "
        "ranking of companies within the Track 2 monthly shortlist "
        "(9–22 companies that already passed all fundamental filters). "
        "It is **experimental only** and is NOT integrated into the production "
        "screener. The deployment decision will be made in December 2026 "
        "after evaluating on the prospective held-out "
        "(June 2026 → December 2026 — zero-iteration data).",
        "",
        "Key difference from Phase 3a (which failed): Phase 3a classified all "
        "500 SP500 companies as outperform/underperform — a hard, noisy problem. "
        "Phase 5.0 ranks within a pre-filtered shortlist of 9–22 companies — "
        "a narrower problem with higher signal density.",
        "",
        "**Note on Phase 4.7 features:** `asset_growth_yoy`, "
        "`deferred_revenue_growth`, and `eps_surprise_last_q` are absent from "
        "the pre-2026 snapshot cache and appear as NaN (imputed to 0.0). "
        "These features effectively do not contribute to the historical model. "
        "They will be present in the prospective held-out (June 2026+).",
        "",
        "---",
        "",
        "## Training Summary",
        "",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Training months with picks | "
        f"{result.n_months} (validation) — see training log for train count |",
        f"| Features attempted | {len(FEATURES)} |",
        f"| Features available (non-constant) | "
        f"{len(result.feature_importances)} |",
        f"| Validation months evaluated | {result.n_months} |",
        "",
        "---",
        "",
        "## Validation Results",
        "",
        "### NDCG@5",
        "",
        f"NDCG@{NDCG_K} measures how well the top-{NDCG_K} ranking matches the "
        "ideal ranking by actual 3-month return. 1.0 = perfect; random ≈ 0.5–0.7 "
        "for groups of this size.",
        "",
        f"| Approach | Mean NDCG@{NDCG_K} | Median NDCG@{NDCG_K} |",
        f"|----------|-------------------|--------------------|",
        f"| ML Ranker (LambdaMART) | {_fmt_f(result.ml_ndcg_mean)} | "
        f"{_fmt_f(result.ml_ndcg_median)} |",
        f"| Score-based baseline | {_fmt_f(result.score_ndcg_mean)} | "
        f"{_fmt_f(result.score_ndcg_median)} |",
        f"| **Improvement** | **{_fmt_f(ndcg_delta, 4)}** | — |",
        "",
        "### Hit Rate (positive 3-month return)",
        "",
        "| Approach | Top-1 hit rate | Top-3 hit rate |",
        "|----------|---------------|----------------|",
        f"| ML Ranker | {_fmt_pct(result.ml_hit_rate_1)} | "
        f"{_fmt_pct(result.ml_hit_rate_3)} |",
        f"| Score-based baseline | {_fmt_pct(result.score_hit_rate_1)} | "
        f"{_fmt_pct(result.score_hit_rate_3)} |",
        f"| **Improvement** | **{_fmt_pp(hr1_delta)}** | "
        f"**{_fmt_pp(hr3_delta)}** |",
        "",
        "### Average 3-Month Return (top picks)",
        "",
        "| Approach | Top-1 avg return | Top-3 avg return |",
        "|----------|-----------------|-----------------|",
        f"| ML Ranker | {_fmt_pct(result.ml_avg_return_1)} | "
        f"{_fmt_pct(result.ml_avg_return_3)} |",
        f"| Score-based | {_fmt_pct(result.score_avg_return_1)} | "
        f"{_fmt_pct(result.score_avg_return_3)} |",
        "",
        "---",
        "",
        "## Feature Importances",
        "",
        "LightGBM feature importance (total gain — higher = more useful for ranking).",
        "",
        "| Feature | Importance |",
        "|---------|-----------|",
    ]

    for feat, imp in result.feature_importances.items():
        lines.append(f"| {feat} | {imp:.1f} |")

    lines += [
        "",
        "---",
        "",
        "## Month-by-Month Validation Detail",
        "",
        "| Date | Picks | Score NDCG | ML NDCG | Score #1 | ML #1 | "
        "Score #1 ret | ML #1 ret | Score top-3 | ML top-3 |",
        "|------|-------|-----------|---------|----------|-------|"
        "------------|----------|------------|---------|",
    ]

    def _r(v: float) -> str:
        return f"{v:+.1f}%" if not np.isnan(v) else "—"

    for m in result.months:
        lines.append(
            f"| {m.date} | {m.n_picks} | {_fmt_f(m.score_ndcg, 3)} | "
            f"{_fmt_f(m.ml_ndcg, 3)} | {m.score_top1} | {m.ml_top1} | "
            f"{_r(m.score_top1_ret_pct)} | {_r(m.ml_top1_ret_pct)} | "
            f"{_r(m.score_top3_avg_ret_pct)} | {_r(m.ml_top3_avg_ret_pct)} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Interpretation",
        "",
        f"The LambdaMART model {'improved' if ndcg_delta > 0 else 'did not improve'} "
        f"NDCG@{NDCG_K} by {abs(ndcg_delta):.4f} vs the score-based baseline on the "
        f"2022–2024 validation set.",
        "",
        f"Hit rate at top-1 changed by **{hr1_delta * 100:+.2f} pp** "
        f"({'improvement' if hr1_delta > 0 else 'regression'}).",
        "",
        "The historical validation has important caveats:",
        "- The training set (2013–2021) overlaps with the period used to design",
        "  and calibrate the Track 2 scorer. The scorer's composite_score feature",
        "  already encodes a hand-tuned optimisation over this window. The ML",
        "  model is competing against a baseline that had access to the same data.",
        "- Phase 4.7 features are absent from the pre-2026 cache — the model cannot",
        "  leverage deferred revenue growth, EPS surprise, or asset growth penalty",
        "  in this historical evaluation.",
        "- Small group sizes (9–22 per month) make NDCG estimates noisy.",
        "  36 validation months is a limited sample.",
        "",
        "---",
        "",
        "## Deployment Decision",
        "",
        f"**Historical gate:** {gate_note}",
        "",
        "**Hard gate (December 2026):** The model will be considered for production",
        "deployment ONLY IF it achieves ≥ 3pp hit-rate improvement vs score-based",
        "ranking on the prospective held-out (June 2026 → December 2026).",
        "That data is truly clean — it was collected AFTER this model was trained",
        "and the Track 2 scorer was frozen.",
        "",
        "The artifact saved to `data/models/phase50_ranker.pkl` is ready for the",
        "December 2026 prospective evaluation.",
        "",
        "> **Current status:** EXPERIMENTAL — do not use in production until",
        "> December 2026 held-out confirms the 3pp gate is cleared.",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Report saved: %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5.0 LambdaMART ranker validation")
    parser.add_argument(
        "--no-cache", dest="no_cache", action="store_true",
        help="Force snapshot rebuild (slow — requires EDGAR bulk data)",
    )
    args = parser.parse_args()

    # 1. Load snapshot cache
    if not SP500_CACHE.exists():
        log.error(
            "SP500 snapshot cache not found: %s\n"
            "Run scripts/run_backtest_track2_sp500.py first to build it.",
            SP500_CACHE,
        )
        sys.exit(1)

    log.info("Loading SP500 snapshots from %s ...", SP500_CACHE)
    fund_by_date: dict[pd.Timestamp, pd.DataFrame] = joblib.load(SP500_CACHE)
    log.info("Loaded %d snapshot dates", len(fund_by_date))

    # 2. Fetch prices for all SP500 tickers + SPY
    config = CrucibleConfig(account_currency="USD")
    log.info("Fetching SP500 tickers ...")
    tickers = fetch_sp500_tickers()
    log.info("%d tickers in SP500 universe", len(tickers))

    log.info(
        "Fetching prices %s → %s for %d tickers (+ SPY) ...",
        PRICE_FETCH_START, PRICE_FETCH_END, len(tickers),
    )
    prices = _fetch_prices(tickers, PRICE_FETCH_START, PRICE_FETCH_END)
    if prices.empty:
        log.error("No prices downloaded — aborting")
        sys.exit(1)

    # 3. Attach momentum to all snapshots
    log.info("Attaching momentum ...")
    attach_momentum(fund_by_date, prices)

    # 4. Run walk-forward validation
    log.info(
        "Running walk-forward: train [%s → %s] / val [%s → %s] ...",
        TRAIN_START_DEFAULT.strftime("%Y-%m"),
        TRAIN_END_DEFAULT.strftime("%Y-%m"),
        VAL_START_DEFAULT.strftime("%Y-%m"),
        VAL_END_DEFAULT.strftime("%Y-%m"),
    )
    artifact, result = walk_forward_validate(
        fund_by_date=fund_by_date,
        prices=prices,
        config=config,
        train_start=TRAIN_START_DEFAULT,
        train_end=TRAIN_END_DEFAULT,
        val_start=VAL_START_DEFAULT,
        val_end=VAL_END_DEFAULT,
        forward_months=FORWARD_MONTHS,
        k=NDCG_K,
    )

    # 5. Save artifact and report
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_ranker(artifact, MODEL_PATH)
    _generate_report(result, REPORT_PATH)

    # 6. Print summary
    hr1_delta = result.ml_hit_rate_1 - result.score_hit_rate_1
    print("\n" + "═" * 62)
    print("  Phase 5.0 — LambdaMART Ranker  (2022–2024 validation)")
    print("═" * 62)
    print(f"  Validation months:          {result.n_months}")
    print(f"  Features used:              {len(result.feature_importances)}")
    print()
    print(f"  {'Metric':<30} {'ML':>8}  {'Score':>8}  {'Δ':>8}")
    print(f"  {'-'*30} {'--------':>8}  {'--------':>8}  {'--------':>8}")

    def _p(v: float) -> str:
        return f"{v * 100:.1f}%" if not np.isnan(v) else "   —"

    def _f(v: float) -> str:
        return f"{v:.4f}" if not np.isnan(v) else "  —"

    def _d(v: float) -> str:
        if np.isnan(v):
            return "   —"
        return f"{v * 100:+.1f} pp"

    print(f"  {'NDCG@5 (mean)':<30} {_f(result.ml_ndcg_mean):>8}  "
          f"{_f(result.score_ndcg_mean):>8}  {result.ndcg_improvement:>+.4f}")
    print(f"  {'Hit rate @1 (3m positive)':<30} {_p(result.ml_hit_rate_1):>8}  "
          f"{_p(result.score_hit_rate_1):>8}  {_d(hr1_delta):>8}")
    print(f"  {'Hit rate @3 (≥1 positive)':<30} {_p(result.ml_hit_rate_3):>8}  "
          f"{_p(result.score_hit_rate_3):>8}  "
          f"{_d(result.ml_hit_rate_3 - result.score_hit_rate_3):>8}")
    print(f"  {'Avg 3m return @1':<30} {_p(result.ml_avg_return_1):>8}  "
          f"{_p(result.score_avg_return_1):>8}  "
          f"{_d(result.ml_avg_return_1 - result.score_avg_return_1):>8}")
    print()
    print("  Top features by importance:")
    for feat, imp in result.feature_importances.head(5).items():
        print(f"    {feat:<32} {imp:.1f}")
    print()
    gate = "✅ CLEARS" if hr1_delta * 100 >= 3.0 else "❌ DOES NOT CLEAR"
    print(f"  Historical 3pp gate: {gate}")
    print(f"  December 2026 gate: PENDING (prospective held-out not yet available)")
    print("═" * 62)
    print(f"\n  Report: {REPORT_PATH}")
    print(f"  Artifact: {MODEL_PATH}")
    print()


if __name__ == "__main__":
    main()
