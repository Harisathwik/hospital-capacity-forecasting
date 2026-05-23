from zenml import pipeline, step
from typing import Tuple, List, Dict, Any
import pandas as pd

# Import the steps we created
from src.steps.data_pull_step import data_pull_step
from src.steps.eda_step import eda_step
from src.steps.feature_engineering_step import feature_engineering_step
from src.steps.target_construction_step import target_construction_step
from src.steps.split_step import split_step
from src.steps.training_step import trainer_step


@pipeline
def icu_forecast_training_pipeline(
    # We can make some parameters configurable if needed
    dataset_id: str = "g62h-syeh",
    select_cols: str = "",  # empty string means no $select
    where_clause: str = "date>='2020-01-01T00:00:00.000'",
    order_clause: str = "date ASC",
    limit: int = 1000,  # Keep small for testing
    horizons: List[int] = [1, 2, 3, 4, 5, 6, 7],
    test_size_days: int = 28,
    ridge_alpha: float = 1.0,
    xgboost_n_estimators: int = 100,
    xgboost_max_depth: int = 5,
    target_col: str = "inpatient_beds_used",  # Changed from staffed_adult_icu_bed_occupancy
):
    """
    Training pipeline for ICU occupancy forecasting.
    Steps:
      1. Pull data (with validation)
      2. EDA (for reporting)
      3. Feature engineering
      4. Target construction (multi-horizon)
      5. Temporal split
      6. Train Ridge regression baseline
      7. Train XGBoost baseline
    """
    # Step 1: Pull data
    df, manifest = data_pull_step(
        dataset_id=dataset_id,
        select_cols=select_cols,
        where_clause=where_clause,
        order_clause=order_clause,
        limit=limit,
    )

    # Step 2: EDA (we don't need to pass the manifest forward, but we can keep it for logging)
    eda_report = eda_step(df=df)

    # Step 3: Feature engineering
    df_with_features, feature_cols = feature_engineering_step(
        df=df,
        target_col=target_col,
        # We'll use the default parameters from the step
    )

    # Step 4: Target construction
    df_with_targets, target_cols = target_construction_step(
        df=df_with_features,  # Now we pass the df with features
        horizons=horizons,
        target_col=target_col,
    )

    # Step 5: Split
    train_df, test_df = split_step(
        df=df_with_targets,
        test_size_days=test_size_days,
    )

    # Step 6: Train Ridge model
    ridge_model, ridge_preprocessing, ridge_feature_names, ridge_target_names, ridge_metrics = trainer_step(
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        target_cols=target_cols,
        model_type="ridge",
        model_params={"alpha": ridge_alpha},
    )

    # Step 7: Train XGBoost model
    xgb_model, xgb_preprocessing, xgb_feature_names, xgb_target_names, xgb_metrics = trainer_step(
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        target_cols=target_cols,
        model_type="xgboost",
        model_params={"n_estimators": xgboost_n_estimators, "max_depth": xgboost_max_depth},
    )

    # We can return the models and metrics if needed, but the pipeline will store them as artifacts.
    return {
        "ridge_model": ridge_model,
        "xgb_model": xgb_model,
        "ridge_metrics": ridge_metrics,
        "xgb_metrics": xgb_metrics,
        "manifest": manifest,
        "eda_report": eda_report,
    }


if __name__ == "__main__":
    # Run the pipeline
    run = icu_forecast_training_pipeline()
    print("Pipeline run finished.")
    # Optionally, we can print some metrics
    print(f"Ridge metrics: {run.outputs['ridge_metrics']}")
    print(f"XGBoost metrics: {run.outputs['xgb_metrics']}")