"""Drift detection — PSI, KS test, feature distribution comparison vs training baseline."""
import numpy as np
import pandas as pd
import json
import logging
from typing import Dict, List, Any, Optional
from scipy import stats
from pathlib import Path

logger = logging.getLogger(__name__)

# PSI thresholds: <0.1 no drift, 0.1-0.2 moderate, >0.2 significant
PSI_MODERATE = 0.1
PSI_SIGNIFICANT = 0.2

# KS test p-value threshold
KS_ALPHA = 0.05

# Missingness spike threshold (relative change)
MISSINGNESS_SPIKE = 1.5  # 50% increase triggers alert


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Compute Population Stability Index between two distributions.
    PSI < 0.1 = no drift, 0.1-0.2 = moderate, > 0.2 = significant.
    """
    # Create bins from expected distribution
    breakpoints = np.nanpercentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)  # remove duplicates

    if len(breakpoints) < 3:
        return 0.0  # constant feature, no drift computable

    # Bin both distributions
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    # Add small epsilon to avoid log(0)
    eps = 1e-6
    expected_pct = (expected_counts + eps) / (len(expected) + eps * len(expected_counts))
    actual_pct = (actual_counts + eps) / (len(actual) + eps * len(actual_counts))

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def compute_ks_test(expected: np.ndarray, actual: np.ndarray) -> Dict[str, float]:
    """Kolmogorov-Smirnov test for distribution shift."""
    stat, p_value = stats.ks_2samp(expected, actual)
    return {"statistic": float(stat), "p_value": float(p_value), "drift_detected": p_value < KS_ALPHA}


def compare_statistics(expected: Dict[str, float], actual: Dict[str, float]) -> Dict[str, Any]:
    """Compare basic stats (mean, std, median) between baseline and current."""
    comparison = {}
    for key in expected:
        if key in actual:
            e, a = expected[key], actual[key]
            if e != 0:
                rel_change = (a - e) / abs(e)
            else:
                rel_change = 0.0 if a == 0 else float("inf")
            comparison[key] = {
                "baseline": e,
                "current": a,
                "relative_change": rel_change,
            }
    return comparison


class DriftDetector:
    def __init__(self, baseline_stats: Dict[str, Any], feature_columns: List[str]):
        """
        Initialize with training baseline statistics.

        baseline_stats: dict with keys:
            - "feature_values": {col: np.ndarray of training values}
            - "feature_stats": {col: {"mean": ..., "std": ..., "median": ..., "p5": ..., "p95": ...}}
            - "missingness": {col: float} (fraction missing per feature)
            - "schema": list of expected columns
            - "n_rows": int
        """
        self.baseline_stats = baseline_stats
        self.feature_columns = feature_columns

    def check_schema(self, current_df: pd.DataFrame) -> Dict[str, Any]:
        """Check if current data has all expected columns."""
        # Priority: Use the passed feature_columns if they differ from baseline schema
        expected = set(self.feature_columns) if self.feature_columns else set(self.baseline_stats.get("schema", []))
        actual = set(current_df.columns)
        missing = expected - actual
        extra = actual - expected
        
        passed = len(missing) == 0
        return {
            "check": "schema",
            "passed": passed,
            "missing_columns": sorted(list(missing)),
            "extra_columns": sorted(list(extra)),
        }

    def check_missingness(self, current_df: pd.DataFrame) -> Dict[str, Any]:
        """Check if missingness rates spiked vs baseline."""
        baseline_miss = self.baseline_stats.get("missingness", {})
        alerts = []
        warnings = []

        for col in self.feature_columns:
            if col not in current_df.columns:
                continue
            current_miss = current_df[col].isna().mean()
            base_miss = baseline_miss.get(col, 0.0)

            if base_miss > 0 and current_miss > base_miss * MISSINGNESS_SPIKE:
                alerts.append({
                    "feature": col,
                    "baseline_missingness": base_miss,
                    "current_missingness": current_miss,
                    "ratio": current_miss / base_miss,
                })
            elif current_miss > base_miss + 0.1:  # absolute 10% jump
                warnings.append({
                    "feature": col,
                    "baseline_missingness": base_miss,
                    "current_missingness": current_miss,
                })

        return {
            "check": "missingness",
            "passed": len(alerts) == 0,
            "alerts": alerts,
            "warnings": warnings,
        }

    def check_drift_per_feature(self, current_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Compute PSI + KS for each feature column."""
        results = []
        baseline_values = self.baseline_stats.get("feature_values", {})

        for col in self.feature_columns:
            if col not in current_df.columns or col not in baseline_values:
                continue

            expected = baseline_values[col]
            actual = current_df[col].dropna().values

            if len(actual) == 0:
                results.append({"feature": col, "error": "no non-null values in current data"})
                continue

            psi = compute_psi(expected, actual)
            ks = compute_ks_test(expected, actual)

            # Determine drift level
            if psi >= PSI_SIGNIFICANT or ks["drift_detected"]:
                drift_level = "significant"
            elif psi >= PSI_MODERATE:
                drift_level = "moderate"
            else:
                drift_level = "none"

            results.append({
                "feature": col,
                "psi": round(psi, 4),
                "ks_statistic": round(ks["statistic"], 4),
                "ks_p_value": round(ks["p_value"], 4),
                "drift_level": drift_level,
            })

        return results

    def check_statistics(self, current_df: pd.DataFrame) -> Dict[str, Any]:
        """Compare mean/std/median vs baseline."""
        baseline_stats = self.baseline_stats.get("feature_stats", {})
        deviations = {}

        for col in self.feature_columns:
            if col not in current_df.columns or col not in baseline_stats:
                continue
            actual_stats = {
                "mean": float(current_df[col].mean()),
                "std": float(current_df[col].std()),
                "median": float(current_df[col].median()),
            }
            deviations[col] = compare_statistics(baseline_stats[col], actual_stats)

        return {
            "check": "statistics",
            "comparisons": deviations,
        }

    def run_all_checks(self, current_df: pd.DataFrame) -> Dict[str, Any]:
        """Run full drift detection suite."""
        schema_result = self.check_schema(current_df)
        missingness_result = self.check_missingness(current_df)
        drift_results = self.check_drift_per_feature(current_df)
        stats_result = self.check_statistics(current_df)

        # Overall drift summary
        drift_levels = [r.get("drift_level", "none") for r in drift_results if isinstance(r, dict) and "error" not in r]
        significant_count = sum(1 for r in drift_results if isinstance(r, dict) and r.get("drift_level") == "significant")
        moderate_count = sum(1 for r in drift_results if isinstance(r, dict) and r.get("drift_level") == "moderate")
        total_features = len(drift_results)

        if significant_count > total_features * 0.3:
            overall = "significant_drift"
        elif significant_count > 0 or moderate_count > total_features * 0.3:
            overall = "moderate_drift"
        else:
            overall = "stable"

        return {
            "overall_drift": overall,
            "feature_drift_summary": {
                "total": total_features,
                "significant": significant_count,
                "moderate": moderate_count,
            },
            "schema_check": schema_result,
            "missingness_check": missingness_result,
            "statistics_check": stats_result,
            "per_feature_drift": drift_results,
        }

    @staticmethod
    def save_report(report: Dict[str, Any], output_path: str):
        """Save drift report as JSON."""
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(p, "w") as f:
            json.dump(report, f, indent=2, default=convert)
        logger.info(f"Drift report saved to {output_path}")
