"""Tests for model training and evaluation — src/models/trainer.py and src/evaluation/metrics.py."""
import pandas as pd
import numpy as np
import pytest

from src.evaluation.metrics import asymmetric_rmse, compute_metrics


def _make_training_data(n_rows=100, n_features=5, n_targets=7):
    np.random.seed(42)
    X = np.random.randn(n_rows, n_features)
    # Multi-output targets (7 horizons) — models use MultiOutputRegressor
    y = np.column_stack([
        X[:, 0] * 2 + X[:, 1] * 0.5 + np.random.randn(n_rows) * 0.1
        for _ in range(n_targets)
    ])
    return X, y


class TestTemporalSplit:
    def test_split_temporal_imports(self):
        from src.models.trainer import split_temporal
        assert callable(split_temporal)

    def test_split_temporal_works(self):
        from src.models.trainer import split_temporal
        n = 100
        df = pd.DataFrame({
            "state": ["CA"] * n,
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "f1": np.random.randn(n),
            "f2": np.random.randn(n),
            "y_t_plus_1": np.random.randn(n),
        })
        config = {"data": {"test_size_days": 20, "validation_size_days": 14}}
        feature_cols = ["f1", "f2"]
        target_cols = ["y_t_plus_1"]
        X_train, X_test, y_train, y_test = split_temporal(df, feature_cols, target_cols, config)
        # split_temporal creates train/val/test splits internally
        # It returns the portions used for model training and final test
        assert len(X_train) > 0
        assert len(X_test) > 0


class TestAsymmetricRMSE:
    def test_equal_predictions(self):
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([100.0, 200.0, 300.0])
        result = asymmetric_rmse(y_true, y_pred, underprediction_penalty=3.0)
        assert result < 1e-6

    def test_overprediction_lower_penalty(self):
        y_true = np.array([100.0])
        over_pred = np.array([110.0])
        under_pred = np.array([90.0])
        over_rmse = asymmetric_rmse(y_true, over_pred, underprediction_penalty=3.0)
        under_rmse = asymmetric_rmse(y_true, under_pred, underprediction_penalty=3.0)
        assert under_rmse > over_rmse

    def test_underprediction_weighted(self):
        y_true = np.array([100.0])
        y_pred = np.array([90.0])
        result = asymmetric_rmse(y_true, y_pred, underprediction_penalty=3.0)
        assert result > 10

    def test_symmetric_when_no_underprediction(self):
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 220.0])
        rmse_1x = asymmetric_rmse(y_true, y_pred, underprediction_penalty=1.0)
        rmse_3x = asymmetric_rmse(y_true, y_pred, underprediction_penalty=3.0)
        assert abs(rmse_1x - rmse_3x) < 1e-6


class TestComputeMetrics:
    def test_returns_expected_keys(self):
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([105.0, 195.0, 310.0])
        metrics = compute_metrics(y_true, y_pred, underprediction_penalty=3.0)
        assert "asymmetric_rmse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "underprediction_rate" in metrics

    def test_perfect_predictions(self):
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([100.0, 200.0])
        metrics = compute_metrics(y_true, y_pred, underprediction_penalty=3.0)
        assert metrics["rmse"] < 1e-6
        assert metrics["mae"] < 1e-6


class TestModelTraining:
    def test_train_ridge_multioutput(self):
        from src.models.trainer import train_ridge
        X, y = _make_training_data(n_rows=100, n_features=5, n_targets=7)
        X_train, y_train = X[:80], y[:80]
        X_val, y_val = X[80:], y[80:]
        model, preds = train_ridge(X_train, y_train, X_val, y_val)
        assert model is not None
        assert preds is not None

    def test_train_xgboost_multioutput(self):
        from src.models.trainer import train_xgboost
        X, y = _make_training_data(n_rows=100, n_features=5, n_targets=7)
        X_train, y_train = X[:80], y[:80]
        X_val, y_val = X[80:], y[80:]
        model, preds = train_xgboost(X_train, y_train, X_val, y_val)
        assert model is not None
        assert preds is not None
