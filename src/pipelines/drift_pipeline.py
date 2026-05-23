"""Drift detection pipeline — run health checks + drift detection, write reports."""
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.monitoring.drift_detector import DriftDetector
from src.monitoring.data_health import DataHealthChecker
from src.monitoring.alerting import AlertManager

logger = logging.getLogger(__name__)


def load_baseline_stats(path: str) -> Dict[str, Any]:
    """Load baseline statistics from training (saved as JSON)."""
    with open(path) as f:
        stats = json.load(f)
    # Convert lists back to numpy arrays for feature_values
    if "feature_values" in stats:
        for col, vals in stats["feature_values"].items():
            stats["feature_values"][col] = np.array(vals)
    return stats


def save_baseline_stats(df: pd.DataFrame, feature_columns: List[str], output_path: str):
    """Compute and save baseline statistics from training data."""
    stats = {
        "schema": feature_columns,
        "n_rows": len(df),
        "missingness": {},
        "feature_stats": {},
        "feature_values": {},
    }

    for col in feature_columns:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        stats["missingness"][col] = float(df[col].isna().mean())
        stats["feature_stats"][col] = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "median": float(series.median()),
            "p5": float(series.quantile(0.05)),
            "p95": float(series.quantile(0.95)),
        }
        stats["feature_values"][col] = series.values.tolist()

    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Baseline stats saved to {output_path}")


def run_drift_pipeline(
    current_data_path: str,
    baseline_stats_path: str,
    feature_columns: List[str],
    date_column: str = "date",
    key_columns: List[str] = None,
    output_dir: str = "data/reports",
) -> Dict[str, Any]:
    """
    Full drift detection pipeline:
    1. Load current data + baseline stats
    2. Run data health checks
    3. Run drift detection
    4. Evaluate alerts
    5. Write report
    """
    logger.info(f"Loading current data from {current_data_path}")
    current_df = pd.read_csv(current_data_path)

    logger.info(f"Loading baseline stats from {baseline_stats_path}")
    baseline_stats = load_baseline_stats(baseline_stats_path)

    # Step 1: Data health
    logger.info("Running data health checks...")
    health_checker = DataHealthChecker(
        required_columns=feature_columns,
        date_column=date_column,
        key_columns=key_columns or ["state", "date"],
        missingness_thresholds={col: 0.5 for col in feature_columns},  # 50% nulls = alert
    )
    health_report = health_checker.run_all(current_df)

    # Step 2: Drift detection
    logger.info("Running drift detection...")
    drift_detector = DriftDetector(baseline_stats=baseline_stats, feature_columns=feature_columns)
    drift_report = drift_detector.run_all_checks(current_df)

    # Step 3: Alerts
    alert_manager = AlertManager(output_dir=output_dir)
    alert_manager.evaluate_health_report(health_report)
    alert_manager.evaluate_drift_report(drift_report)

    # Step 4: Write report
    report_path = alert_manager.write_report(
        drift_report=drift_report,
        health_report=health_report,
    )

    print(alert_manager.summary())
    logger.info(f"Report written to {report_path}")

    return {
        "report_path": report_path,
        "health_report": health_report,
        "drift_report": drift_report,
        "alerts": alert_manager.alerts,
    }


if __name__ == "__main__":
    import sys

    # Default paths for local run
    current_data = sys.argv[1] if len(sys.argv) > 1 else "data/raw/hhs_hospital_capacity_g62h-syeh.csv"
    baseline = sys.argv[2] if len(sys.argv) > 2 else "data/processed/baseline_stats.json"

    # Load feature columns from baseline
    with open(baseline) as f:
        bs = json.load(f)
    features = bs.get("schema", [])

    result = run_drift_pipeline(
        current_data_path=current_data,
        baseline_stats_path=baseline,
        feature_columns=features,
        date_column="date",
        key_columns=["state", "date"],
    )
    print(f"\nReport: {result['report_path']}")
