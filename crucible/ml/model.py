"""ML model training and inference for Phase 3a.

All three models (Logistic Regression, Random Forest, XGBoost) are always trained.
The best-performing model on the validation set is returned. MIN_ACCURACY is a
documentation threshold — models above/below it are logged but it does not stop
training. All imputation medians are computed from training data only — no leakage.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

MIN_ACCURACY: float = 0.53

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
    all_val_accuracies: dict[str, float] = field(default_factory=dict)  # accuracy for every model trained


def train_phase3a(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    train_end_date: pd.Timestamp,
) -> ModelArtifact:
    """Train all three models and return the one with the highest validation accuracy.

    Imputation medians are computed from X_train only.
    StandardScaler is applied only for Logistic Regression.
    MIN_ACCURACY is logged as a reference threshold but does not stop training.
    """
    feature_names = list(X_train.columns)
    imputation_values = _compute_medians(X_train)

    X_tr = _impute(X_train, imputation_values)
    X_vl = _impute(X_val, imputation_values)

    # --- Logistic Regression ---
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_vl_s = scaler.transform(X_vl)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_C, best_cv_score = 1.0, -1.0
    for C in [0.01, 0.1, 1.0]:
        candidate = LogisticRegression(C=C, max_iter=1000, random_state=42,
                                       class_weight="balanced")
        cv_score = float(cross_val_score(candidate, X_tr_s, y_train,
                                         cv=cv, scoring="accuracy").mean())
        logger.info("LR C=%.2f  cv_accuracy=%.3f", C, cv_score)
        if cv_score > best_cv_score:
            best_cv_score, best_C = cv_score, C

    logger.info("LR best C=%.2f  (cv_accuracy=%.3f)", best_C, best_cv_score)
    lr = LogisticRegression(C=best_C, max_iter=1000, random_state=42,
                            class_weight="balanced")
    lr.fit(X_tr_s, y_train)
    lr_acc = float(accuracy_score(y_val, lr.predict(X_vl_s)))
    logger.info("LR  val_accuracy=%.3f  (%s threshold %.2f)",
                lr_acc, "ABOVE" if lr_acc >= MIN_ACCURACY else "BELOW", MIN_ACCURACY)

    # --- Random Forest ---
    rf = RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=50,
                                random_state=42, n_jobs=-1, class_weight="balanced")
    rf.fit(X_tr, y_train)
    rf_acc = float(accuracy_score(y_val, rf.predict(X_vl)))
    logger.info("RF  val_accuracy=%.3f  (%s threshold %.2f)",
                rf_acc, "ABOVE" if rf_acc >= MIN_ACCURACY else "BELOW", MIN_ACCURACY)

    # --- XGBoost ---
    all_accuracies: dict[str, float] = {
        "logistic_regression": lr_acc,
        "random_forest": rf_acc,
    }
    candidates: list[tuple[float, str, object, StandardScaler | None]] = [
        (lr_acc, "logistic_regression", lr, scaler),
        (rf_acc, "random_forest", rf, None),
    ]

    if _HAS_XGBOOST:
        _neg = int((y_train == 0).sum())
        _pos = int((y_train == 1).sum())
        xgb = _XGBClassifier(n_estimators=300, random_state=42, eval_metric="logloss",
                              verbosity=0, scale_pos_weight=_neg / _pos)
        xgb.fit(X_tr, y_train)
        xgb_acc = float(accuracy_score(y_val, xgb.predict(X_vl)))
        logger.info("XGB val_accuracy=%.3f  (%s threshold %.2f)",
                    xgb_acc, "ABOVE" if xgb_acc >= MIN_ACCURACY else "BELOW", MIN_ACCURACY)
        all_accuracies["xgboost"] = xgb_acc
        candidates.append((xgb_acc, "xgboost", xgb, None))

    best_acc, best_type, best_model, best_scaler = max(candidates, key=lambda t: t[0])
    logger.info("Selected: %s  val_accuracy=%.3f", best_type, best_acc)

    return ModelArtifact(
        model=best_model,
        model_type=best_type,
        scaler=best_scaler,
        imputation_values=imputation_values,
        feature_names=feature_names,
        train_end_date=train_end_date,
        val_accuracy=best_acc,
        all_val_accuracies=all_accuracies,
    )


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
