"""Evidently-based drift detection and auto-retraining trigger."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime
import json
import logging

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.test_suite import TestSuite
from evidently.tests import *

logger = logging.getLogger(__name__)

# Paths
DATA_PATH = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
DRIFT_LOG_PATH = Path("monitoring/drift_log.json")
REFERENCE_STATS_PATH = Path("monitoring/reference_stats.csv")


def prepare_reference_data() -> pd.DataFrame:
    """Load and prepare reference (training) data."""
    from src.data.loader import load_data, validate_data
    from src.features.engineer import engineer_features

    df = load_data()
    df = validate_data(df)
    df = engineer_features(df)
    return df


def generate_reference_stats() -> pd.DataFrame:
    """Generate and save reference statistics from training data."""
    df = prepare_reference_data()
    REFERENCE_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(REFERENCE_STATS_PATH, index=False)
    logger.info(f"Reference stats saved: {df.shape}")
    return df


def detect_drift(
    current_data: pd.DataFrame,
    reference_data: Optional[pd.DataFrame] = None,
    threshold: float = 0.05,
) -> Dict:
    """
    Run Evidently drift detection comparing current data vs reference.

    Returns drift report summary and whether drift was detected.
    """
    if reference_data is None:
        if REFERENCE_STATS_PATH.exists():
            reference_data = pd.read_csv(REFERENCE_STATS_PATH)
        else:
            reference_data = prepare_reference_data()

    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges",
                        "AvgMonthlyCharge", "ChargeDiff", "NumServices", "IsNewCustomer"]

    column_mapping = ColumnMapping()
    column_mapping.numerical_features = numeric_features

    # Build report
    report = Report(metrics=[
        DataDriftPreset(),
    ])

    report.run(
        reference_data=reference_data.drop(columns=["Churn"], errors="ignore"),
        current_data=current_data.drop(columns=["Churn"], errors="ignore"),
        column_mapping=column_mapping,
    )

    # Extract results
    report_dict = report.as_dict()
    drift_results = {
        "timestamp": datetime.now().isoformat(),
        "dataset_drift": False,
        "n_drifted_features": 0,
        "drifted_features": [],
        "threshold": threshold,
    }

    for metric in report_dict.get("metrics", []):
        if metric.get("metric") == "DataDriftTable":
            result = metric.get("result", {})
            drift_results["dataset_drift"] = result.get("dataset_drift", False)
            drift_results["n_drifted_features"] = len(
                result.get("drift_by_columns", {})
            )
            for col_name, col_result in result.get("drift_by_columns", {}).items():
                if col_result.get("drift_detected"):
                    drift_results["drifted_features"].append({
                        "feature": col_name,
                        "drift_score": col_result.get("drift_score", 0),
                        "method": col_result.get("stattest_name", "unknown"),
                    })

    # Log drift results
    _log_drift(drift_results)

    logger.info(
        f"Drift detection: dataset_drift={drift_results['dataset_drift']}, "
        f"n_drifted={drift_results['n_drifted_features']}"
    )

    return drift_results


def _log_drift(drift_results: Dict):
    """Append drift results to log."""
    DRIFT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_entries = []
    if DRIFT_LOG_PATH.exists():
        with open(DRIFT_LOG_PATH) as f:
            log_entries = json.load(f)
    log_entries.append(drift_results)
    with open(DRIFT_LOG_PATH, "w") as f:
        json.dump(log_entries, f, indent=2, default=str)


def should_retrain(drift_results: Dict, min_drifted_features: int = 3) -> bool:
    """Determine if drift is severe enough to trigger retraining."""
    return (
        drift_results["dataset_drift"]
        and drift_results["n_drifted_features"] >= min_drifted_features
    )


def simulate_drift(reference_data: pd.DataFrame, n_samples: int = 500) -> pd.DataFrame:
    """Simulate data drift by shifting feature distributions (for demo purposes)."""
    drifted = reference_data.head(n_samples).copy()

    # Shift monthly charges up (price increase scenario)
    drifted["MonthlyCharges"] = drifted["MonthlyCharges"] * 1.3 + np.random.normal(5, 2, n_samples)

    # Shift tenure down (newer customers)
    drifted["tenure"] = (drifted["tenure"] * 0.5).clip(lower=0).astype(int)

    # Shift contract types (more month-to-month)
    contract_probs = [0.6, 0.25, 0.15]  # Month-to-month, One year, Two year
    drifted["Contract"] = np.random.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n_samples,
        p=contract_probs,
    )

    logger.info(f"Simulated drift on {n_samples} samples")
    return drifted
