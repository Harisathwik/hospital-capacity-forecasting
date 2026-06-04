"""Create golden dataset from the latest training run."""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from zenml.client import Client
from src.evaluation.metrics import compute_metrics

client = Client()
run = client.get_pipeline_run("4dd228f7-2d39-40dd-a8b5-1e5dc8d9d7b8")

def load(suffix):
    avs = [av for av in run.artifact_versions if av.name.endswith(suffix)]
    return client.get_artifact_version(avs[0].id).load()

model = load("trainer_step_2::output_0")
preprocessing = load("trainer_step_2::output_1")
feature_names = load("trainer_step_2::output_2")
metrics_dict = load("trainer_step_2::output_4")

split_test_df = load("split_step::output_1")
print(f"Test DF shape: {split_test_df.shape}")
print(f"Columns (first 10): {list(split_test_df.columns[:10])}")

target_cols = [c for c in split_test_df.columns if "_t_plus_" in c]
print(f"Target columns: {target_cols}")

feat_cols = feature_names if isinstance(feature_names, list) else list(feature_names)
available_feats = [c for c in feat_cols if c in split_test_df.columns]
print(f"Available features: {len(available_feats)}/{len(feat_cols)}")

X_test = split_test_df[available_feats]
y_test = split_test_df[target_cols].values

X_processed = preprocessing.transform(X_test)
y_pred = model.predict(X_processed)

y_true_flat = y_test.ravel()
y_pred_flat = y_pred.ravel()

fixtures_dir = Path("tests/fixtures")
golden_df = pd.DataFrame({"y_true": y_true_flat, "y_pred": y_pred_flat})
golden_df.to_parquet(fixtures_dir / "golden_dataset.parquet")
print(f"Golden dataset saved: {len(golden_df)} rows")

baseline = compute_metrics(y_true_flat, y_pred_flat, underprediction_penalty=3.0)
baseline["rmse"] = float(np.sqrt(np.mean((y_true_flat - y_pred_flat) ** 2)))
baseline["created_at"] = datetime.utcnow().isoformat()
baseline["n_samples"] = len(y_true_flat)

with open(fixtures_dir / "golden_baseline.json", "w") as f:
    json.dump(baseline, f, indent=2)
print(f"Baseline metrics: {json.dumps(baseline, indent=2)}")
