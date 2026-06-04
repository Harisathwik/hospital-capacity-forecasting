"""Prediction logic — extracted as a standalone module.

Handles model loading from MLflow model registry (Production with Staging
fallback), prediction execution, bounds checking, and audit logging.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Audit logging ──────────────────────────────────────────────────────────

AUDIT_LOG_PATH = Path("data/reports/prediction_audit.jsonl")


def audit_log(
    request_features: Dict[str, Any],
    predictions: List[float],
    target_names: List[str],
    flags: List[str],
    model_stage: str,
    model_uri: str,
    latency_ms: float,
) -> None:
    """Append a prediction audit record to the JSONL file (append-only)."""
    try:
        AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass  # directory may already exist
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_stage": model_stage,
        "model_uri": model_uri,
        "request_features": request_features,
        "predictions": predictions,
        "target_names": target_names,
        "flags": flags,
        "latency_ms": round(latency_ms, 3),
    }
    with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


# ── Model loading from MLflow registry ────────────────────────────────────

def load_model_from_registry(
    model_name: str = "icu_forecast_xgboost",
    preferred_stage: str = "Production",
    fallback_stage: str = "Staging",
) -> Tuple[Any, str, str]:
    """
    Load an MLflow pyfunc model from the model registry.

    Tries *preferred_stage* first, then *fallback_stage*.
    Returns (model, actual_stage, model_uri).

    Raises RuntimeError if neither stage has a deployable model.
    """
    for stage in (preferred_stage, fallback_stage):
        model_uri = f"models:/{model_name}/{stage}"
        try:
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info(
                "Loaded model '%s' at stage '%s' via %s",
                model_name, stage, model_uri,
            )
            return model, stage, model_uri
        except mlflow.exceptions.MlflowException as exc:
            logger.warning(
                "Model '%s' not found at stage '%s': %s",
                model_name, stage, exc,
            )
            continue
        except Exception as exc:
            logger.warning(
                "Failed to load model '%s' at stage '%s': %s",
                model_name, stage, exc,
            )
            continue

    raise RuntimeError(
        f"Could not load model '{model_name}' from either "
        f"'{preferred_stage}' or '{fallback_stage}' stage in the MLflow registry. "
        f"Ensure a model version has been registered and promoted to one of these stages."
    )


# ── Bounds checking ───────────────────────────────────────────────────────

# Default training max for staffed_adult_icu_bed_occupancy.
# Overridden from config (serving.training_max_icu) or model metadata if available.
DEFAULT_TRAINING_MAX_ICU = 5000.0


def check_prediction_bounds(
    predictions: np.ndarray,
    training_max: float,
) -> List[str]:
    """
    Validate ICU occupancy prediction bounds.

    Rules:
      - ICU occupancy **cannot be negative**.
      - Flag if any prediction exceeds 2× the training-set maximum
        (suggests out-of-distribution input).

    Returns a list of flag strings (empty if all predictions are within bounds).
    """
    flags: List[str] = []
    if np.any(predictions < 0):
        flags.append("negative_prediction")
    if training_max > 0 and np.any(predictions > 2 * training_max):
        flags.append("exceeds_2x_training_max")
    return flags


# ── Core prediction function ──────────────────────────────────────────────

def predict(
    model: Any,
    features: Dict[str, Any],
    feature_names: Optional[List[str]] = None,
    training_max: float = DEFAULT_TRAINING_MAX_ICU,
) -> Tuple[List[float], List[str]]:
    """
    Run a single-row prediction with bounds checking.

    The *model* is expected to be an MLflow pyfunc model whose
    ``predict()`` accepts a pandas DataFrame and returns predictions.

    If *feature_names* is provided the input columns are reordered to
    match the training order; missing columns raise ``ValueError``.

    Returns ``(predictions_list, bounds_flags)``.  Negative predictions
    are clamped to 0 (ICU occupancy cannot be negative).
    """
    # Build input DataFrame from the feature dictionary
    input_df = pd.DataFrame([features])

    # Reorder columns to match training feature order if provided
    if feature_names is not None:
        missing = set(feature_names) - set(input_df.columns)
        if missing:
            raise ValueError(f"Missing required features: {sorted(missing)}")
        input_df = input_df[feature_names]

    # Predict — mlflow pyfunc model expects a DataFrame
    preds = model.predict(input_df)

    # Normalise to a 1-D float numpy array
    if isinstance(preds, pd.DataFrame):
        preds = preds.values
    if isinstance(preds, pd.Series):
        preds = preds.values
    preds = np.asarray(preds, dtype=float).ravel()

    # Bounds checking (before clamping, so we flag the raw model output)
    flags = check_prediction_bounds(preds, training_max)

    # Clamp negatives to 0 — ICU occupancy cannot be negative
    preds = np.clip(preds, 0.0, None)

    return preds.tolist(), flags
