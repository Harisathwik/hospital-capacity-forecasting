"""Inference pipeline — batch scoring and serving orchestration."""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


def run_inference_pipeline(
    model,
    preprocessing,
    feature_names: list,
    input_data: pd.DataFrame,
    training_max: float = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run batch inference on input data.

    Args:
        model: Fitted model (sklearn/xgboost compatible)
        preprocessing: Fitted preprocessing pipeline
        feature_names: Expected feature column order
        input_data: DataFrame with feature columns
        training_max: Max training target value for bounds checking

    Returns:
        Tuple of (predictions, metadata) where metadata includes bounds flags.
    """
    from src.serving.predict import check_prediction_bounds

    # Reorder columns to match training
    input_aligned = input_data[feature_names]

    # Apply preprocessing
    processed = preprocessing.transform(input_aligned)

    # Predict
    raw_preds = model.predict(processed)
    if raw_preds.ndim == 2:
        raw_preds = raw_preds.ravel()

    # Bounds checking
    flags = check_prediction_bounds(raw_preds, training_max=training_max)

    # Clamp negatives to 0 (ICU occupancy cannot be negative)
    predictions = np.maximum(raw_preds, 0.0)

    metadata = {
        "n_predictions": len(predictions),
        "bounds_flags": flags,
        "any_clamped": bool(flags.get("negative_prediction", False)),
    }

    if flags.get("negative_prediction"):
        n_neg = int((raw_preds < 0).sum())
        logger.warning(f"Clamped {n_neg} negative predictions to 0")

    return predictions, metadata
