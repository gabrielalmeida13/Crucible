"""Tests for crucible/ml/model.py."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from crucible.ml.model import (
    ModelArtifact,
    _compute_medians,
    _impute,
    evaluate,
    feature_importances,
    load_model,
    save_model,
    train_phase3a,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _xy(n_train: int = 200, n_val: int = 60, n_features: int = 5) -> tuple:
    X_all, y_all = make_classification(
        n_samples=n_train + n_val,
        n_features=n_features,
        n_informative=3,
        random_state=42,
    )
    cols = [f"f{i}" for i in range(n_features)]
    X_tr = pd.DataFrame(X_all[:n_train], columns=cols)
    y_tr = pd.Series(y_all[:n_train], name="label")
    X_vl = pd.DataFrame(X_all[n_train:], columns=cols)
    y_vl = pd.Series(y_all[n_train:], name="label")
    return X_tr, y_tr, X_vl, y_vl


# ---------------------------------------------------------------------------
# Tests: imputation helpers
# ---------------------------------------------------------------------------


def test_compute_medians_no_nan() -> None:
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    meds = _compute_medians(X)
    assert meds == {"a": 2.0, "b": 5.0}


def test_impute_fills_nan() -> None:
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    result = _impute(X, {"a": 2.0})
    assert not result["a"].isna().any()
    assert result["a"].iloc[1] == 2.0


def test_impute_does_not_modify_original() -> None:
    X = pd.DataFrame({"a": [1.0, np.nan]})
    _impute(X, {"a": 99.0})
    assert X["a"].isna().any()


def test_impute_unknown_column_gets_zero() -> None:
    X = pd.DataFrame({"a": [np.nan]})
    result = _impute(X, {})
    assert result["a"].iloc[0] == 0.0


# ---------------------------------------------------------------------------
# Tests: train_phase3a
# ---------------------------------------------------------------------------


def test_train_returns_model_artifact() -> None:
    X_tr, y_tr, X_vl, y_vl = _xy()
    artifact = train_phase3a(X_tr, y_tr, X_vl, y_vl, pd.Timestamp("2021-12-31"))
    assert isinstance(artifact, ModelArtifact)


def test_artifact_has_correct_feature_names() -> None:
    X_tr, y_tr, X_vl, y_vl = _xy(n_features=5)
    artifact = train_phase3a(X_tr, y_tr, X_vl, y_vl, pd.Timestamp("2021-12-31"))
    assert artifact.feature_names == [f"f{i}" for i in range(5)]


def test_artifact_model_type_is_valid() -> None:
    X_tr, y_tr, X_vl, y_vl = _xy()
    artifact = train_phase3a(X_tr, y_tr, X_vl, y_vl, pd.Timestamp("2021-12-31"))
    assert artifact.model_type in {"logistic_regression", "random_forest", "xgboost"}


def test_artifact_val_accuracy_between_0_and_1() -> None:
    X_tr, y_tr, X_vl, y_vl = _xy()
    artifact = train_phase3a(X_tr, y_tr, X_vl, y_vl, pd.Timestamp("2021-12-31"))
    assert 0.0 <= artifact.val_accuracy <= 1.0


def test_lr_used_when_data_is_linearly_separable() -> None:
    """On cleanly separable data LR achieves >= MIN_ACCURACY and is selected."""
    rng = np.random.default_rng(0)
    n = 300
    X_tr = pd.DataFrame({"x": rng.normal(size=n)})
    y_tr = pd.Series((X_tr["x"] > 0).astype(int))
    X_vl = pd.DataFrame({"x": rng.normal(size=100)})
    y_vl = pd.Series((X_vl["x"] > 0).astype(int))
    artifact = train_phase3a(X_tr, y_tr, X_vl, y_vl, pd.Timestamp("2021-12-31"))
    assert artifact.model_type == "logistic_regression"
    assert artifact.scaler is not None


def test_imputation_values_from_training_only() -> None:
    """Imputation medians are computed on training data, not validation."""
    X_tr, y_tr, X_vl, y_vl = _xy()
    X_tr_nan = X_tr.copy()
    X_tr_nan.loc[0, "f0"] = np.nan
    artifact = train_phase3a(X_tr_nan, y_tr, X_vl, y_vl, pd.Timestamp("2021-12-31"))
    assert "f0" in artifact.imputation_values
    expected_median = float(X_tr_nan["f0"].median())
    assert abs(artifact.imputation_values["f0"] - expected_median) < 1e-9


# ---------------------------------------------------------------------------
# Tests: evaluate
# ---------------------------------------------------------------------------


def test_evaluate_returns_accuracy_and_cm() -> None:
    X_tr, y_tr, X_vl, y_vl = _xy()
    artifact = train_phase3a(X_tr, y_tr, X_vl, y_vl, pd.Timestamp("2021-12-31"))
    result = evaluate(artifact, X_vl, y_vl)
    assert "accuracy" in result
    assert "confusion_matrix" in result
    assert 0.0 <= result["accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# Tests: feature_importances
# ---------------------------------------------------------------------------


def test_feature_importances_length_matches_features() -> None:
    X_tr, y_tr, X_vl, y_vl = _xy(n_features=5)
    artifact = train_phase3a(X_tr, y_tr, X_vl, y_vl, pd.Timestamp("2021-12-31"))
    imp = feature_importances(artifact)
    assert len(imp) == 5


def test_feature_importances_sorted_descending() -> None:
    X_tr, y_tr, X_vl, y_vl = _xy()
    artifact = train_phase3a(X_tr, y_tr, X_vl, y_vl, pd.Timestamp("2021-12-31"))
    imp = feature_importances(artifact)
    assert list(imp) == sorted(imp, reverse=True)


# ---------------------------------------------------------------------------
# Tests: save/load
# ---------------------------------------------------------------------------


def test_save_load_roundtrip() -> None:
    X_tr, y_tr, X_vl, y_vl = _xy()
    artifact = train_phase3a(X_tr, y_tr, X_vl, y_vl, pd.Timestamp("2021-12-31"))

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.pkl"
        save_model(artifact, path)
        loaded = load_model(path)

    assert loaded.model_type == artifact.model_type
    assert loaded.feature_names == artifact.feature_names
    assert abs(loaded.val_accuracy - artifact.val_accuracy) < 1e-9
