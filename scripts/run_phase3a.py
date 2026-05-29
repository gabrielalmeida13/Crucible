"""Phase 3a ML training and validation entry point.

Usage:
    python scripts/run_phase3a.py

Train window : 2010-01-01 – 2021-12-31
Validation   : 2022-01-01 – 2023-12-31
Held-out     : 2024-01-01 – present  (NOT evaluated here)

Outputs:
    data/models/phase3a_model.pkl       — serialised ModelArtifact
    data/results/phase3a_validation.md  — accuracy / confusion matrix / feature importance
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.feature_selection import mutual_info_classif

from crucible.ml.features import FEATURE_COLS, build_feature_matrix
from crucible.ml.model import (
    _impute,
    evaluate,
    feature_importances,
    save_model,
    train_phase3a,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

EDGAR_DIR    = PROJECT_ROOT / "data" / "raw" / "edgar" / "companyfacts"
CIK_MAP_PATH = PROJECT_ROOT / "data" / "raw" / "edgar" / "cik_mapping.json"
MODEL_PATH   = PROJECT_ROOT / "data" / "models" / "phase3a_model.pkl"
REPORT_PATH  = PROJECT_ROOT / "data" / "results" / "phase3a_validation.md"

TRAIN_START = pd.Timestamp("2010-01-01")
TRAIN_END   = pd.Timestamp("2021-12-31")
VAL_START   = pd.Timestamp("2022-01-01")
VAL_END     = pd.Timestamp("2023-12-31")
# 2024+ is held-out — never evaluate here

_SNAPSHOT_START   = pd.Timestamp("2010-01-31", tz="UTC")
_SNAPSHOT_END     = pd.Timestamp("2023-12-31", tz="UTC")
_PRICE_START      = "2009-01-01"
_PRICE_END        = "2024-12-31"  # covers 12m forward label from Dec 2023
_SNAPSHOT_WORKERS = 4
_PRICE_WORKERS    = 20


def main() -> None:
    # Import loading primitives from run_backtest — same EDGAR/price infrastructure
    from scripts.run_backtest import (  # type: ignore[import]
        _build_fundamentals_parallel,
        _fetch_prices_parallel,
    )
    from crucible.fetcher import _load_cik_mapping, fetch_sp500_tickers
    from crucible.backtest import attach_momentum

    if not CIK_MAP_PATH.exists():
        logger.error(
            "CIK mapping not found at %s. Run scripts/download_edgar_bulk.py first.",
            CIK_MAP_PATH,
        )
        sys.exit(1)
    if not EDGAR_DIR.exists():
        logger.error(
            "EDGAR companyfacts not found at %s. Run scripts/download_edgar_bulk.py first.",
            EDGAR_DIR,
        )
        sys.exit(1)

    cik_map = _load_cik_mapping(CIK_MAP_PATH)
    monthly_dates = pd.date_range(_SNAPSHOT_START, _SNAPSHOT_END, freq="ME", tz="UTC")

    logger.info("Fetching S&P 500 tickers...")
    tickers = fetch_sp500_tickers()
    logger.info("Universe: %d tickers", len(tickers))

    logger.info(
        "Fetching prices (%s – %s, %d workers)...",
        _PRICE_START, _PRICE_END, _PRICE_WORKERS,
    )
    prices = _fetch_prices_parallel(tickers, _PRICE_START, _PRICE_END)
    if prices.empty:
        logger.error("No price data — aborting.")
        sys.exit(1)

    logger.info(
        "Building %d monthly snapshots (%d workers)...",
        len(monthly_dates), _SNAPSHOT_WORKERS,
    )
    fund_by_date = _build_fundamentals_parallel(tickers, monthly_dates, EDGAR_DIR, cik_map, prices)
    attach_momentum(fund_by_date, prices)
    logger.info("Snapshots complete: %d dates", len(fund_by_date))

    # Pre-compute roic_direction once for all snapshots — avoids redundant work
    # when build_feature_matrix is called twice (train + val windows).
    from crucible.ml.features import add_roic_direction
    add_roic_direction(fund_by_date)

    logger.info("Building training feature matrix (2010–2021)...")
    X_train, y_train = build_feature_matrix(
        fund_by_date, prices,
        start_date=TRAIN_START,
        end_date=TRAIN_END,
    )
    logger.info("Train: %d rows, label balance %.1f%%", len(X_train), y_train.mean() * 100)

    logger.info("Building validation feature matrix (2022–2023)...")
    X_val, y_val = build_feature_matrix(
        fund_by_date, prices,
        start_date=VAL_START,
        end_date=VAL_END,
    )
    logger.info("Val: %d rows, label balance %.1f%%", len(X_val), y_val.mean() * 100)

    if len(X_train) == 0:
        logger.error("No training data — check EDGAR bulk download.")
        sys.exit(1)

    if len(X_val) == 0:
        logger.error("No validation data — check price history for 2022–2023.")
        sys.exit(1)

    # Inject a pure-noise column as an objective importance floor.
    # Train and val use independent seeds so neither leaks signal.
    rng_tr = np.random.default_rng(0)
    rng_vl = np.random.default_rng(1)
    X_train = X_train.copy()
    X_val   = X_val.copy()
    X_train["random_baseline"] = rng_tr.standard_normal(len(X_train))
    X_val["random_baseline"]   = rng_vl.standard_normal(len(X_val))
    logger.info("random_baseline column added to train and val sets")

    logger.info("Training all models (LR, RF, XGBoost)...")
    artifact = train_phase3a(X_train, y_train, X_val, y_val, train_end_date=TRAIN_END)
    logger.info("Best model: %s  val_accuracy=%.3f", artifact.model_type, artifact.val_accuracy)

    val_metrics = evaluate(artifact, X_val, y_val)
    imp = feature_importances(artifact)  # all features, sorted descending

    X_tr_imp = _impute(X_train, artifact.imputation_values)
    mi_scores = mutual_info_classif(X_tr_imp, y_train, random_state=42)
    mi_series = pd.Series(mi_scores, index=X_train.columns).sort_values(ascending=False)
    logger.info("MI computed for %d features (including random_baseline)", len(mi_series))

    save_model(artifact, MODEL_PATH)
    _write_report(artifact, val_metrics, imp, mi_series, X_train, y_train, X_val, y_val)
    logger.info("Report saved: %s", REPORT_PATH)


def _write_report(
    artifact,
    val_metrics: dict,
    imp: pd.Series,
    mi_series: pd.Series,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> None:
    acc = val_metrics["accuracy"]
    cm = val_metrics["confusion_matrix"]
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    cm_md = (
        "| | Predicted 0 | Predicted 1 |\n"
        "|---|---|---|\n"
        f"| **Actual 0** | {tn} | {fp} |\n"
        f"| **Actual 1** | {fn} | {tp} |"
    )

    rand_imp = imp.get("random_baseline", 0.0)
    rand_mi  = mi_series.get("random_baseline", 0.0)

    def _imp_row(rank: int, name: str, val: float) -> str:
        if name == "random_baseline":
            return f"| {rank} | **random_baseline** | **{val:.4f}** | ← threshold |"
        note = "candidate for removal" if val < rand_imp else ""
        return f"| {rank} | {name} | {val:.4f} | {note} |"

    def _mi_row(rank: int, name: str, val: float) -> str:
        if name == "random_baseline":
            return f"| {rank} | **random_baseline** | **{val:.4f}** | ← threshold |"
        note = "candidate for removal" if val < rand_mi else ""
        return f"| {rank} | {name} | {val:.4f} | {note} |"

    imp_rows = "\n".join(
        _imp_row(i + 1, name, val) for i, (name, val) in enumerate(imp.items())
    )
    mi_rows = "\n".join(
        _mi_row(i + 1, name, val) for i, (name, val) in enumerate(mi_series.items())
    )

    all_accs = artifact.all_val_accuracies
    model_rows = "\n".join(
        "| {} | {:.3f} ({:.1%}) | {} | {} |".format(
            name,
            a,
            a,
            "ABOVE" if a >= 0.55 else "below",
            "**selected**" if name == artifact.model_type else "",
        )
        for name, a in all_accs.items()
    )

    lines = [
        "# Phase 3a ML Validation Report",
        "",
        f"**Generated:** {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "---",
        "",
        "## Training configuration",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Train window | {TRAIN_START.date()} – {TRAIN_END.date()} |",
        f"| Validation window | {VAL_START.date()} – {VAL_END.date()} |",
        "| Held-out | 2024-01-01 – present (not evaluated) |",
        f"| Features | {len(FEATURE_COLS)} real + 1 random_baseline |",
        f"| Model selected | {artifact.model_type} |",
        f"| Train rows | {len(X_train)} |",
        f"| Train label balance | {y_train.mean():.1%} outperform |",
        f"| Val rows | {len(X_val)} |",
        f"| Val label balance | {y_val.mean():.1%} outperform |",
        "",
        "---",
        "",
        "## Model comparison (2022–2023 validation)",
        "",
        "All three models are always trained. The highest validation accuracy wins.",
        "Threshold (55.0%) is informational only.",
        "",
        "| Model | Accuracy | vs threshold | |",
        "|-------|----------|--------------|---|",
        model_rows,
        "",
        "---",
        "",
        "## Validation accuracy (selected model)",
        "",
        f"**Accuracy on 2022–2023 validation set: {acc:.3f} ({acc:.1%})**",
        "",
        f"Selected model: `{artifact.model_type}`",
        f"Threshold (informational): 55.0% — Result: {'ABOVE' if acc >= 0.55 else 'BELOW'}",
        "",
        "---",
        "",
        "## Confusion matrix (2022–2023 validation)",
        "",
        cm_md,
        "",
        "---",
        "",
        "## Feature importances (model-derived)",
        "",
        f"Model type: `{artifact.model_type}` — importances are absolute coefficients for LR, "
        "mean decrease in impurity for RF/XGBoost. "
        "`random_baseline` is pure N(0,1) noise; real features below its importance are flagged.",
        "",
        "| Rank | Feature | Importance | Notes |",
        "|------|---------|------------|-------|",
        imp_rows,
        "",
        "---",
        "",
        "## Feature ranking by Mutual Information (model-independent)",
        "",
        "MI is computed on the imputed training set (2010–2021) against the binary outperform label. "
        "`random_baseline` is pure N(0,1) noise; real features below its MI score are flagged.",
        "",
        "| Rank | Feature | MI Score | Notes |",
        "|------|---------|----------|-------|",
        mi_rows,
        "",
        "---",
        "",
        "## Notes",
        "",
        "- `insider_buy_ratio` is NaN for all training/validation rows (ENABLE_INSIDER_FORM4=False).",
        "  Its imputed value is 0.0 from training medians. It will carry real weight only after",
        "  live monthly runs compute it for the shortlist.",
        "- Imputation medians are computed from the training window only — no leakage.",
        "- The held-out 2024+ window has NOT been evaluated. It is the final performance gate.",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
