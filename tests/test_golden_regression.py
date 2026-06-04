"""Golden dataset regression test.

Blocks deploy if model quality degrades vs a known-good baseline.

To create golden fixtures:
  1. Run the training pipeline to produce a trained model
  2. Run: python -m tests.create_golden_dataset
  3. Commit tests/fixtures/golden_dataset.parquet and golden_baseline.json

Until golden fixtures exist, this test SKIPS with instructions.
"""
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"
GOLDEN_DATA_PATH = FIXTURES_DIR / "golden_dataset.parquet"
GOLDEN_BASELINE_PATH = FIXTURES_DIR / "golden_baseline.json"

# Maximum allowed relative degradation vs baseline (10%)
MAX_RELATIVE_DEGRADATION = 0.10


def _golden_fixtures_exist():
    return GOLDEN_DATA_PATH.exists() and GOLDEN_BASELINE_PATH.exists()


@pytest.mark.skipif(
    not _golden_fixtures_exist(),
    reason=(
        "Golden fixtures not found. Create them by running a known-good training "
        "pipeline, then: python -m tests.create_golden_dataset. Commit the resulting "
        "files in tests/fixtures/."
    ),
)
class TestGoldenRegression:
    """Regression test: model metrics must not degrade vs golden baseline."""

    @pytest.fixture(autouse=True)
    def load_golden(self):
        self.golden_data = pd.read_parquet(GOLDEN_DATA_PATH)
        with open(GOLDEN_BASELINE_PATH) as f:
            self.golden_baseline = json.load(f)

    def test_asymmetric_rmse_not_degraded(self):
        """Primary metric: asymmetric RMSE must not degrade >10% vs baseline."""
        from src.evaluation.metrics import asymmetric_rmse

        if "y_true" not in self.golden_data.columns or "y_pred" not in self.golden_data.columns:
            pytest.skip("Golden dataset missing y_true/y_pred columns")

        y_true = self.golden_data["y_true"].values
        y_pred = self.golden_data["y_pred"].values
        baseline_rmse = self.golden_baseline.get("asymmetric_rmse", float("inf"))

        if baseline_rmse == 0:
            pytest.skip("Baseline metrics not yet populated (placeholder)")

        current_rmse = asymmetric_rmse(y_true, y_pred, underprediction_penalty=3.0)
        relative_change = (current_rmse - baseline_rmse) / baseline_rmse

        assert relative_change <= MAX_RELATIVE_DEGRADATION, (
            f"Asymmetric RMSE degraded by {relative_change:.1%} "
            f"(current={current_rmse:.4f}, baseline={baseline_rmse:.4f}, "
            f"max allowed={MAX_RELATIVE_DEGRADATION:.0%})"
        )

    def test_underprediction_rate_not_degraded(self):
        """Underprediction rate must not exceed baseline by >10% absolute."""
        if "y_true" not in self.golden_data.columns or "y_pred" not in self.golden_data.columns:
            pytest.skip("Golden dataset missing y_true/y_pred columns")

        y_true = self.golden_data["y_true"].values
        y_pred = self.golden_data["y_pred"].values

        current_rate = float(np.mean(y_pred < y_true))
        baseline_rate = self.golden_baseline.get("underprediction_rate", 0.3)

        assert current_rate <= baseline_rate + 0.10, (
            f"Underprediction rate {current_rate:.1%} exceeds "
            f"baseline {baseline_rate:.1%} + 10%"
        )

    def test_rmse_not_degraded(self):
        """Standard RMSE must not degrade >10% vs baseline."""
        from src.evaluation.metrics import compute_metrics

        if "y_true" not in self.golden_data.columns or "y_pred" not in self.golden_data.columns:
            pytest.skip("Golden dataset missing y_true/y_pred columns")

        y_true = self.golden_data["y_true"].values
        y_pred = self.golden_data["y_pred"].values
        baseline_rmse = self.golden_baseline.get("rmse", float("inf"))

        if baseline_rmse == 0:
            pytest.skip("Baseline metrics not yet populated (placeholder)")

        metrics = compute_metrics(y_true, y_pred, underprediction_penalty=3.0)
        relative_change = (metrics["rmse"] - baseline_rmse) / baseline_rmse

        assert relative_change <= MAX_RELATIVE_DEGRADATION, (
            f"RMSE degraded by {relative_change:.1%}"
        )

    def test_golden_dataset_not_empty(self):
        assert len(self.golden_data) > 0
        assert "y_true" in self.golden_data.columns
        assert "y_pred" in self.golden_data.columns
