"""Create golden dataset and register model in MLflow.

Run: cd D:\\Sathwik\\Ayush\\MLOps-Github && PYTHONPATH=. .venv\\Scripts\\python.exe scripts/create_golden_and_register.py
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# === Part 1: Create golden dataset from latest training run ===
print("=" * 60)
print("Step 1: Creating golden dataset from latest training run")
print("=" * 60)

from zenml.client import Client
from src.evaluation.metrics import compute_metrics

client = Client()

# Get latest completed training run
runs = client.list_pipeline_runs(
    sort_by="desc:end_time",
    size=5,
)
training_run = None
for r in runs:
    if "icu_forecast_training_pipeline" in r.name and r.status == "completed":
        training_run = r
        break

if training_run is None:
    print("ERROR: No completed training pipeline run found. Run the pipeline first.")
    exit(1)

print(f"Using run: {training_run.name} (id={training_run.id})")


def load_artifact(suffix):
    avs = [av for av in training_run.artifact_versions if av.name.endswith(suffix)]
    if not avs:
        raise ValueError(f"No artifact with suffix '{suffix}'")
    return client.get_artifact_version(avs[0].id).load()


# Load split data and model artifacts
try:
    split_test = load_artifact("split_step::output_1")  # (X_test, y_test)
    model = load_artifact("trainer_step_2::output_0")    # XGBoost model
    preprocessing = load_artifact("trainer_step_2::output_1")
    feature_names = load_artifact("trainer_step_2::output_2")
    target_names = load_artifact("trainer_step_2::output_3")
    metrics_dict = load_artifact("trainer_step_2::output_4")

    print(f"Model loaded: {type(model)}")
    print(f"Features: {len(feature_names)}")
    print(f"Targets: {target_names}")
    print(f"Metrics: {metrics_dict}")
    print(f"Split test type: {type(split_test)}")

    # Generate predictions on test data
    if isinstance(split_test, tuple) and len(split_test) == 2:
        X_test, y_test = split_test
    elif isinstance(split_test, dict):
        X_test = split_test.get("X_test", split_test.get("X"))
        y_test = split_test.get("y_test", split_test.get("y"))
    else:
        print(f"Split data format: {type(split_test)}")
        # Try to extract test portion
        if hasattr(split_test, 'shape'):
            print(f"Shape: {split_test.shape}")
        X_test = None
        y_test = None

    if X_test is not None and y_test is not None:
        # Apply preprocessing and predict
        X_processed = preprocessing.transform(X_test) if preprocessing else X_test
        y_pred = model.predict(X_processed)
        if y_pred.ndim == 2:
            # For multi-output, take first horizon for golden
            y_true_flat = y_test.ravel() if y_test.ndim > 1 else y_test
            y_pred_flat = y_pred.ravel()
        else:
            y_true_flat = y_test
            y_pred_flat = y_pred

        # Save golden dataset
        fixtures_dir = Path("tests/fixtures")
        fixtures_dir.mkdir(parents=True, exist_ok=True)

        golden_df = pd.DataFrame({
            "y_true": y_true_flat,
            "y_pred": y_pred_flat,
        })
        golden_df.to_parquet(fixtures_dir / "golden_dataset.parquet")
        print(f"Golden dataset saved: {len(golden_df)} rows")

        # Compute and save baseline metrics
        baseline_metrics = compute_metrics(y_true_flat, y_pred_flat, underprediction_penalty=3.0)
        baseline_metrics["created_at"] = datetime.utcnow().isoformat()
        baseline_metrics["source_run_id"] = str(training_run.id)
        baseline_metrics["n_samples"] = len(y_true_flat)

        with open(fixtures_dir / "golden_baseline.json", "w") as f:
            json.dump(baseline_metrics, f, indent=2)
        print(f"Baseline metrics saved: {baseline_metrics}")

except Exception as e:
    print(f"Error loading artifacts: {e}")
    print("Skipping golden dataset creation.")

# === Part 2: Register model in MLflow ===
print()
print("=" * 60)
print("Step 2: Registering model in MLflow")
print("=" * 60)

try:
    import mlflow
    from mlflow.models import infer_signature

    # Get MLflow tracking URI from ZenML stack
    # The experiment tracker tracks runs
    experiment_name = "icu_forecast_experiment"

    # Find the MLflow run corresponding to the ZenML pipeline run
    mlflow_runs = mlflow.search_runs(
        filter_string=f"tags.zenml_pipeline_run_id = '{training_run.id}'",
        output_format="pandas",
    )

    if len(mlflow_runs) == 0:
        # Try finding by pipeline name
        print("No MLflow run found with ZenML tag. Searching by experiment...")
        mlflow_runs = mlflow.search_runs(output_format="pandas")

    print(f"Found {len(mlflow_runs)} MLflow runs")

    # Load the model from the ZenML artifact and register it
    model_name = "icu_forecast_xgboost"

    # We already have the model loaded from above
    # Register it in MLflow model registry
    try:
        # Try logging and registering via MLflow
        with mlflow.start_run(run_name="model_registration") as run:
            # Log the model using mlflow.sklearn (works for sklearn-compatible models)
            import mlflow.sklearn

            # Create a sample input for signature
            if X_test is not None and preprocessing is not None:
                sample_input = pd.DataFrame(
                    preprocessing.transform(X_test[:3]),
                    columns=feature_names if isinstance(feature_names, list) else None,
                )
                signature = infer_signature(sample_input, model.predict(preprocessing.transform(X_test[:3])))
            else:
                signature = None

            mlflow.sklearn.log_model(
                model,
                "model",
                signature=signature,
                registered_model_name=model_name,
            )

            # Log metrics
            if isinstance(metrics_dict, dict):
                mlflow.log_metrics({k: float(v) for k, v in metrics_dict.items() if isinstance(v, (int, float, np.number))})

            print(f"Model registered as '{model_name}' in MLflow")
            print(f"MLflow run ID: {run.info.run_id}")

    except Exception as e2:
        print(f"MLflow registration error: {e2}")
        print("Trying alternative registration...")

        # Alternative: Use MLflow client directly
        from mlflow.tracking import MlflowClient
        mlflow_client = MlflowClient()

        # Check if model already exists
        try:
            mlflow_client.get_registered_model(model_name)
            print(f"Model '{model_name}' already registered.")
        except mlflow.exceptions.RestException:
            mlflow_client.create_registered_model(model_name)
            print(f"Created registered model '{model_name}'")

except ImportError:
    print("MLflow not available. Skipping model registration.")
except Exception as e:
    print(f"Error during MLflow registration: {e}")

print()
print("=" * 60)
print("DONE")
print("=" * 60)
print()
print("Next steps:")
print("1. Promote model to Production: mlflow models promote --model icu_forecast_xgboost --stage Production")
print("2. Verify serving: curl http://localhost:8000/health")
print("3. Run golden regression: PYTHONPATH=. pytest tests/test_golden_regression.py -v")
