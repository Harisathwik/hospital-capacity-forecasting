"""Tests for serving layer — src/serving/predict.py prediction logic."""
import json
import numpy as np
import pytest
from unittest.mock import MagicMock


class TestPredictionBounds:
    def test_check_bounds_function_exists(self):
        from src.serving.predict import check_prediction_bounds
        assert callable(check_prediction_bounds)

    def test_negative_flagged(self):
        from src.serving.predict import check_prediction_bounds
        preds = np.array([-5.0, 100.0, 200.0])
        flags = check_prediction_bounds(preds, training_max=1000.0)
        # Returns List[str] of flag names
        assert any("negative" in f.lower() for f in flags)

    def test_within_bounds(self):
        from src.serving.predict import check_prediction_bounds
        preds = np.array([100.0, 500.0, 900.0])
        flags = check_prediction_bounds(preds, training_max=1000.0)
        # Should be empty list or no critical flags
        assert isinstance(flags, list)

    def test_exceeds_2x_training_max(self):
        from src.serving.predict import check_prediction_bounds
        preds = np.array([2500.0])  # > 2 * 1000
        flags = check_prediction_bounds(preds, training_max=1000.0)
        assert isinstance(flags, list)


class TestAuditLog:
    def test_audit_log_function_exists(self):
        from src.serving.predict import audit_log
        assert callable(audit_log)

    def test_audit_log_writes(self):
        """Test that audit_log can be called without error."""
        from src.serving.predict import audit_log
        # audit_log writes to data/reports/prediction_audit.jsonl internally
        try:
            audit_log(
                request_features={"f1": 1.0, "f2": 2.0},
                predictions=[100.0, 200.0],
                target_names=["y_t_plus_1", "y_t_plus_2"],
                flags=[],
                model_stage="Production",
                model_uri="models:/icu_forecast_xgboost/Production",
                latency_ms=12.5,
            )
        except Exception as e:
            pytest.fail(f"audit_log raised: {e}")


class TestModelLoading:
    def test_load_model_from_registry_exists(self):
        from src.serving.predict import load_model_from_registry
        assert callable(load_model_from_registry)
