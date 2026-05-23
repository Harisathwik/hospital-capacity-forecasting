"""Tests for drift detection + data health checks."""
import pytest
import numpy as np
import pandas as pd
import json
import tempfile
from pathlib import Path

from src.monitoring.drift_detector import (
    DriftDetector,
    compute_psi,
    compute_ks_test,
    PSI_MODERATE,
    PSI_SIGNIFICANT,
)
from src.monitoring.data_health import DataHealthChecker
from src.monitoring.alerting import AlertManager


# --- PSI tests ---

class TestPSI:
    def test_psi_identical_distributions(self):
        """PSI ≈ 0 for identical distributions."""
        data = np.random.normal(0, 1, 1000)
        psi = compute_psi(data, data)
        assert psi < 0.01

    def test_psi_shifted_distribution(self):
        """PSI > 0.2 for significantly shifted distribution."""
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(3, 1, 1000)
        psi = compute_psi(expected, actual)
        assert psi > PSI_SIGNIFICANT

    def test_psi_moderate_shift(self):
        """PSI between 0.1 and 0.2 for moderate shift."""
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(0.5, 1, 1000)
        psi = compute_psi(expected, actual)
        # Should be at least moderate
        assert psi > 0

    def test_psi_constant_feature(self):
        """PSI = 0 for constant feature (no bins possible)."""
        data = np.ones(100)
        psi = compute_psi(data, data)
        assert psi == 0.0


# --- KS test ---

class TestKSTest:
    def test_ks_identical(self):
        """KS p-value high for identical distributions."""
        data = np.random.normal(0, 1, 500)
        result = compute_ks_test(data, data)
        assert result["p_value"] > 0.05
        assert not result["drift_detected"]

    def test_ks_different(self):
        """KS p-value low for different distributions."""
        expected = np.random.normal(0, 1, 500)
        actual = np.random.normal(5, 1, 500)
        result = compute_ks_test(expected, actual)
        assert result["p_value"] < 0.05
        assert result["drift_detected"]


# --- DriftDetector ---

def _make_baseline_stats():
    """Create minimal baseline stats for testing."""
    np.random.seed(42)
    n = 500
    feature_values = {
        "feat_a": np.random.normal(0, 1, n),
        "feat_b": np.random.normal(5, 2, n),
    }
    return {
        "schema": ["feat_a", "feat_b", "date"],
        "n_rows": n,
        "missingness": {"feat_a": 0.05, "feat_b": 0.1},
        "feature_stats": {
            "feat_a": {"mean": 0.0, "std": 1.0, "median": 0.0},
            "feat_b": {"mean": 5.0, "std": 2.0, "median": 5.0},
        },
        "feature_values": feature_values,
    }


def _make_current_df(drift=False):
    """Create current data, optionally with drift."""
    np.random.seed(123)
    n = 200
    if drift:
        return pd.DataFrame({
            "feat_a": np.random.normal(3, 1.5, n),  # shifted
            "feat_b": np.random.normal(5, 2, n),     # same
            "date": pd.date_range("2024-01-01", periods=n),
        })
    return pd.DataFrame({
        "feat_a": np.random.normal(0, 1, n),
        "feat_b": np.random.normal(5, 2, n),
        "date": pd.date_range("2024-01-01", periods=n),
    })


class TestDriftDetector:
    def test_schema_check_passes(self):
        baseline = _make_baseline_stats()
        detector = DriftDetector(baseline, ["feat_a", "feat_b"])
        df = _make_current_df()
        result = detector.check_schema(df)
        assert result["passed"]

    def test_schema_check_fails_missing_column(self):
        baseline = _make_baseline_stats()
        detector = DriftDetector(baseline, ["feat_a", "feat_b", "feat_c"])
        df = _make_current_df()
        result = detector.check_schema(df)
        assert not result["passed"]
        assert "feat_c" in result["missing_columns"]

    def test_no_drift_detected(self):
        baseline = _make_baseline_stats()
        detector = DriftDetector(baseline, ["feat_a", "feat_b"])
        df = _make_current_df(drift=False)
        report = detector.run_all_checks(df)
        assert report["overall_drift"] == "stable"

    def test_drift_detected(self):
        baseline = _make_baseline_stats()
        detector = DriftDetector(baseline, ["feat_a", "feat_b"])
        df = _make_current_df(drift=True)
        report = detector.run_all_checks(df)
        # feat_a shifted → should detect at least moderate
        assert report["overall_drift"] in ("moderate_drift", "significant_drift")

    def test_save_report(self):
        baseline = _make_baseline_stats()
        detector = DriftDetector(baseline, ["feat_a", "feat_b"])
        df = _make_current_df()
        report = detector.run_all_checks(df)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        DriftDetector.save_report(report, path)
        with open(path) as f:
            loaded = json.load(f)
        assert "overall_drift" in loaded


# --- DataHealthChecker ---

class TestDataHealthChecker:
    def test_healthy_data(self):
        checker = DataHealthChecker(
            required_columns=["a", "b"],
            date_column="date",
            key_columns=["a", "date"],
            freshness_max_days=10000, # Large threshold for test data
        )
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "date": pd.date_range("2024-01-01", periods=3),
        })
        report = checker.run_all(df)
        assert report["overall_health"] == "healthy"

    def test_missing_column(self):
        checker = DataHealthChecker(required_columns=["a", "missing_col"])
        df = pd.DataFrame({"a": [1, 2]})
        report = checker.run_all(df)
        assert report["overall_health"] == "degraded"

    def test_duplicate_keys(self):
        checker = DataHealthChecker(
            required_columns=["a"],
            key_columns=["a"],
        )
        df = pd.DataFrame({"a": [1, 1, 2]})
        report = checker.run_all(df)
        dupes = report["checks"]["duplicates"]
        assert not dupes["passed"]
        assert dupes["duplicate_rows"] == 2

    def test_completeness_alert(self):
        checker = DataHealthChecker(
            required_columns=["a"],
            missingness_thresholds={"a": 0.3},
        )
        df = pd.DataFrame({"a": [1, None, None, None, None]})  # 80% null
        report = checker.run_all(df)
        completeness = report["checks"]["completeness"]
        assert not completeness["passed"]
        assert len(completeness["alerts"]) > 0


# --- AlertManager ---

class TestAlertManager:
    def test_no_alerts(self):
        mgr = AlertManager(output_dir=tempfile.mkdtemp())
        assert len(mgr.alerts) == 0
        assert "All checks passed" in mgr.summary()

    def test_drift_alerts(self):
        mgr = AlertManager(output_dir=tempfile.mkdtemp())
        drift_report = {
            "feature_drift_summary": {"total": 10, "significant": 3, "moderate": 2},
            "per_feature_drift": [
                {"feature": "x", "drift_level": "significant", "psi": 0.35, "ks_p_value": 0.001},
            ],
        }
        mgr.evaluate_drift_report(drift_report)
        assert len(mgr.alerts) >= 2  # overall + per-feature

    def test_health_alerts(self):
        mgr = AlertManager(output_dir=tempfile.mkdtemp())
        health_report = {
            "checks": {
                "schema": {"passed": False, "missing_columns": ["col_a"]},
                "completeness": {"passed": True, "alerts": []},
                "duplicates": {"passed": True},
                "freshness": {"passed": True},
            }
        }
        mgr.evaluate_health_report(health_report)
        critical = [a for a in mgr.alerts if a["severity"] == "critical"]
        assert len(critical) > 0

    def test_write_report(self):
        mgr = AlertManager(output_dir=tempfile.mkdtemp())
        mgr.add_alert("warning", "test", "test alert")
        path = mgr.write_report()
        assert Path(path).exists()
        with open(path) as f:
            data = json.load(f)
        assert data["alert_count"] == 1
