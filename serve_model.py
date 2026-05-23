import os
import json
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from zenml.client import Client

# Initialize ZenML client
client = Client()

# Get the pipeline run (we assume the latest run of the training pipeline)
# For simplicity, we hardcode the run ID from the previous run.
# In a production setting, we would fetch the latest successful run.
run_id = "7ecc5275-4b07-4ffb-bfd7-2ceebd87eccc"
run = client.get_pipeline_run(run_id)

# Helper function to load an artifact by name suffix
def load_artifact_by_name_suffix(suffix: str):
    # Find artifact version with the given suffix in the name
    matching_avs = [av for av in run.artifact_versions if av.name.endswith(suffix)]
    if not matching_avs:
        raise ValueError(f"No artifact version found with suffix '{suffix}'")
    # We assume there is only one matching artifact version for the suffix
    av = matching_avs[0]
    print(f"Loading artifact: {av.name} (ID: {av.id})")
    artifact_version = client.get_artifact_version(av.id)
    return artifact_version.load()

# Load the XGBoost model and preprocessing
print("Loading XGBoost model...")
try:
    model = load_artifact_by_name_suffix("trainer_step_2::output_0")
    print(f"Model loaded: {type(model)}")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

print("Loading preprocessing pipeline...")
try:
    preprocessing = load_artifact_by_name_suffix("trainer_step_2::output_1")
    print(f"Preprocessing loaded: {type(preprocessing)}")
except Exception as e:
    print(f"Error loading preprocessing: {e}")
    raise

# Load feature names
print("Loading feature names...")
try:
    feature_names = load_artifact_by_name_suffix("trainer_step_2::output_2")
    print(f"Loaded {len(feature_names)} feature names")
except Exception as e:
    print(f"Error loading feature names: {e}")
    raise

# Load target names
print("Loading target names...")
try:
    target_names = load_artifact_by_name_suffix("trainer_step_2::output_3")
    print(f"Target names: {target_names}")
except Exception as e:
    print(f"Error loading target names: {e}")
    raise

# Create FastAPI app
app = FastAPI(title="ICU Forecast XGBoost Model Server")

# Define the input data schema
# We expect a dictionary of feature names to values for a single instance
# Alternatively, we can accept a list of instances for batch prediction
class PredictRequest(BaseModel):
    features: Dict[str, Any]

class PredictResponse(BaseModel):
    predictions: List[float]
    target_names: List[str]

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        # Convert the input dictionary to a DataFrame with one row
        # Ensure the columns are in the same order as during training
        input_df = pd.DataFrame([request.features])
        # Reorder columns to match the training feature order
        input_df = input_df[feature_names]
        
        # Apply preprocessing
        processed = preprocessing.transform(input_df)
        
        # Make predictions
        preds = model.predict(processed)
        
        # If the model outputs a 2D array (e.g., for multi-output), we flatten
        if preds.ndim == 2:
            preds = preds.ravel()
        
        return PredictResponse(
            predictions=preds.tolist(),
            target_names=target_names
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)