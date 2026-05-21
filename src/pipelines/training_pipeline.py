"""ZenML pipeline: data -> validate -> features -> naive baseline evaluation."""

import json

import numpy as np
import pandas as pd
import mlflow
from zenml import pipeline, step
from zenml.client import Client

from src.data.loader import load_from_cache, pull_from_socrata
from src.data.validator import validate
from src.features.engineer import build_features, get_feature_columns
from src.evaluation.metrics import compute_metrics
from src.core.config import load_config


@step
def ingest_data() -> pd.DataFrame:
    """Load cached data or pull from Socrata."""
    config = load_config()
    raw_path = config["data"]["raw_dir"] + "/hhs_hospital_capacity_g62h-syeh.csv"
    from pathlib import Path
    if Path(raw_path).exists():
        df = load_from_cache(config)
    else:
        df, _ = pull_from_socrata(config)
    mlflow.log_param("data_rows", len(df))
    mlflow.log_param("data_date_min", str(df["date"].min().date()))
    mlflow.log_param("data_date_max", str(df["date"].max().date()))
    return df


@step
def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate schema/types/ranges/missingness."""
    report = validate(df)
    print(str(report))

    mlflow.log_param("validation_passed", str(report.passed))
    mlflow.log_param("validation_errors", len(report.errors))
    mlflow.log_param("validation_warnings", len(report.warnings))

    for i, w in enumerate(report.warnings):
        mlflow.log_param(f"warning_{i}", w)

    if not report.passed:
        error_msg = "; ".join(report.errors)
        raise ValueError(f"Data validation failed: {error_msg}")

    return df


@step
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build feature frame with lags, rolling, calendar, and targets."""
    config = load_config()
    horizon = config["forecast"]["horizon_days"]

    feature_df, _ = build_features(df, horizon=horizon)

    feature_cols = get_feature_columns(feature_df)

    mlflow.log_param("feature_rows", len(feature_df))
    mlflow.log_param("feature_cols_count", len(feature_cols))
    mlflow.log_param("horizon", horizon)

    for h in range(1, horizon + 1):
        col = f"y_target_{h}"
        mlflow.log_param(f"target_{h}_mean", float(feature_df[col].mean()))
        mlflow.log_param(f"target_{h}_std", float(feature_df[col].std()))

    return feature_df


@step
def evaluate_baseline(feature_df: pd.DataFrame) -> str:
    """Evaluate naive persistence baseline. Returns JSON string of metrics."""
    config = load_config()
    horizon = config["forecast"]["horizon_days"]
    under_penalty = config["evaluation"]["underprediction_penalty"]

    target_cols = [f"y_target_{h}" for h in range(1, horizon + 1)]
    y_true = feature_df[target_cols].values

    # Naive: predict t+h = observed t+1 (persistence)
    naive_preds = np.column_stack([y_true[:, 0]] * horizon)

    naive_metrics = {}
    for h in range(horizon):
        m = compute_metrics(y_true[:, h], naive_preds[:, h], under_penalty)
        naive_metrics[f"horizon_{h+1}"] = m

    for h_name, metrics in naive_metrics.items():
        for metric_name, value in metrics.items():
            mlflow.log_metric(f"naive_{h_name}_{metric_name}", value)

    avg_asym_rmse = float(np.mean([m["asymmetric_rmse"] for m in naive_metrics.values()]))
    avg_mae = float(np.mean([m["mae"] for m in naive_metrics.values()]))
    avg_under_rate = float(np.mean([m["underprediction_rate"] for m in naive_metrics.values()]))

    mlflow.log_metric("naive_avg_asymmetric_rmse", avg_asym_rmse)
    mlflow.log_metric("naive_avg_mae", avg_mae)
    mlflow.log_metric("naive_avg_underprediction_rate", avg_under_rate)

    print("=" * 60)
    print("NAIVE BASELINE RESULTS (persistence forecast)")
    print("=" * 60)
    for h_name, metrics in naive_metrics.items():
        print(f"\n{h_name}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
    print(f"\n--- Averages across {horizon} horizons ---")
    print(f"  asymmetric_rmse: {avg_asym_rmse:.4f}")
    print(f"  mae: {avg_mae:.4f}")
    print(f"  underprediction_rate: {avg_under_rate:.4f}")

    return json.dumps({
        "naive_metrics": naive_metrics,
        "avg_asymmetric_rmse": avg_asym_rmse,
        "avg_mae": avg_mae,
        "avg_underprediction_rate": avg_under_rate,
    })


@pipeline(name="hospital_forecast_baseline")
def training_pipeline():
    """End-to-end pipeline: ingest → validate → features → baseline eval."""
    df = ingest_data()
    df = validate_data(df)
    feature_df = engineer_features(df)
    results_json = evaluate_baseline(feature_df)
    return results_json


if __name__ == "__main__":
    client = Client()
    mlflow.set_experiment("hospital_forecast")
    results_json = training_pipeline()
    print("\nPipeline completed successfully!")
