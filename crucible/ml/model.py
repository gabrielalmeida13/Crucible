"""ML model training and inference for Phase 3a.

Model progression: Logistic Regression → Random Forest → XGBoost.
Advances to the next model only if the previous one scores below MIN_ACCURACY
on the validation set. All imputation medians are computed from training data
only — no leakage into validation.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

MIN_ACCURACY: float = 0.55

try:
    from xgboost import XGBClassifier as _XGBClassifier
    _HAS_XGBOOST = True
except ImportError:  # pragma: no cover
    _HAS_XGBOOST = False
    _XGBClassifier = None  # type: ignore[assignment,misc]


@dataclass
class ModelArtifact:
    """Serialisable container for a trained model and its metadata."""

    model: object
    model_type: str  # "logistic_regression" | "random_forest" | "xgboost"
    scaler: StandardScaler | None
    imputation_values: dict[str, float]
    feature_names: list[str]
    train_end_date: pd.Timestamp
    val_accuracy: float


def train_phase3a(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    train_end_date: pd.Timestamp,
) -> ModelArtifact:
    """Train with LR → RF → XGBoost escalation; return first model reaching MIN_ACCURACY.

    Imputation medians are computed from X_train only.
    StandardScaler is applied only for Logistic Regression.
    """
    feature_names = list(X_train.columns)
    imputation_values = _compute_medians(X_train)

    X_tr = _impute(X_train, imputation_values)
    X_vl = _impute(X_val, imputation_values)

    # --- Logistic Regression ---
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_vl_s = scaler.transform(X_vl)

    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr.fit(X_tr_s, y_train)
    lr_acc = float(accuracy_score(y_val, lr.predict(X_vl_s)))
    logger.info("LR validation accuracy: %.3f", lr_acc)

    if lr_acc >= MIN_ACCURACY:
        return ModelArtifact(lr, "logistic_regression", scaler, imputation_values,
                             feature_names, train_end_date, lr_acc)

    # --- Random Forest ---
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1,
                                class_weight="balanced")
    rf.fit(X_tr, y_train)
    rf_acc = float(accuracy_score(y_val, rf.predict(X_vl)))
    logger.info("RF validation accuracy: %.3f", rf_acc)

    if rf_acc >= MIN_ACCURACY:
        return ModelArtifact(rf, "random_forest", None, imputation_values,
                             feature_names, train_end_date, rf_acc)

    # --- XGBoost ---
    if _HAS_XGBOOST:
        xgb = _XGBClassifier(n_estimators=300, random_state=42, eval_metric="logloss",
                              verbosity=0)
        xgb.fit(X_tr, y_train)
        xgb_acc = float(accuracy_score(y_val, xgb.predict(X_vl)))
        logger.info("XGBoost validation accuracy: %.3f", xgb_acc)

        best = max(
            [(lr_acc, "logistic_regression", lr, scaler),
             (rf_acc, "random_forest", rf, None),
             (xgb_acc, "xgboost", xgb, None)],
            key=lambda t: t[0],
        )
        return ModelArtifact(best[2], best[1], best[3], imputation_values,
                             feature_names, train_end_date, best[0])

    # No XGBoost — return best of LR / RF
    if rf_acc >= lr_acc:
        return ModelArtifact(rf, "random_forest", None, imputation_values,
                             feature_names, train_end_date, rf_acc)
    return ModelArtifact(lr, "logistic_regression", scaler, imputation_values,
                         feature_names, train_end_date, lr_acc)


def evaluate(
    artifact: ModelArtifact,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, object]:
    """Return accuracy + confusion matrix for a fitted artifact on (X, y).

    Uses the artifact's imputation_values and scaler — no re-fitting.
    """
    X_imp = _impute(X[artifact.feature_names], artifact.imputation_values)
    if artifact.scaler is not None:
        X_imp = pd.DataFrame(
            artifact.scaler.transform(X_imp),
            columns=artifact.feature_names,
        )
    preds = artifact.model.predict(X_imp)  # type: ignore[union-attr]
    acc = float(accuracy_score(y, preds))
    cm = confusion_matrix(y, preds).tolist()
    return {"accuracy": acc, "confusion_matrix": cm}


def feature_importances(artifact: ModelArtifact) -> pd.Series:
    """Return feature importances as a Series (descending); abs coef for LR."""
    model = artifact.model
    names = artifact.feature_names

    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_  # type: ignore[union-attr]
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_[0])  # type: ignore[union-attr]
    else:
        imp = np.zeros(len(names))

    return pd.Series(imp, index=names).sort_values(ascending=False)


def save_model(artifact: ModelArtifact, path: Path) -> None:
    """Pickle the ModelArtifact to path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(artifact, fh)
    logger.info("Model saved: %s (type=%s, val_acc=%.3f)", path, artifact.model_type,
                artifact.val_accuracy)


def load_model(path: Path) -> ModelArtifact:
    """Load a pickled ModelArtifact from path."""
    with open(path, "rb") as fh:
        return pickle.load(fh)  # noqa: S301


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _compute_medians(X: pd.DataFrame) -> dict[str, float]:
    """Compute per-column medians from training data (used for imputation).

    Entirely-NaN columns (e.g. insider_buy_ratio when disabled) get 0.0 so
    that fillna downstream always receives a finite value.
    """
    result: dict[str, float] = {}
    for col in X.columns:
        med = X[col].median()
        result[col] = float(med) if pd.notna(med) else 0.0
    return result


def _impute(X: pd.DataFrame, medians: dict[str, float]) -> pd.DataFrame:
    """Fill NaN with pre-computed training medians; unknown or NaN medians get 0.0."""
    X = X.copy()
    for col in X.columns:
        if X[col].isna().any():
            fill_val = medians.get(col, 0.0)
            if pd.isna(fill_val):
                fill_val = 0.0
            X[col] = X[col].fillna(fill_val)
    return X
