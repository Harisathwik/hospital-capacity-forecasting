"""Feature selection — identify most important features and reduce dimensionality."""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional


def select_by_importance(
    model,
    feature_names: List[str],
    top_k: int = None,
    threshold: float = 0.01,
) -> List[str]:
    """Select features by model importance scores.

    Args:
        model: Fitted model with feature_importances_ attribute (e.g., XGBoost)
        feature_names: List of feature column names
        top_k: Keep only top K features (optional, overrides threshold)
        threshold: Minimum importance score to keep (default 0.01)

    Returns:
        List of selected feature names.
    """
    if not hasattr(model, "feature_importances_"):
        return feature_names  # can't select without importances

    importances = model.feature_importances_
    paired = list(zip(feature_names, importances))
    paired.sort(key=lambda x: x[1], reverse=True)

    if top_k is not None:
        return [name for name, _ in paired[:top_k]]

    return [name for name, imp in paired if imp >= threshold]


def detect_leakage_risk(
    feature_names: List[str],
    target_col: str = "staffed_adult_icu_bed_occupancy",
) -> List[str]:
    """Flag features that might leak target information.

    Heuristic: any feature containing the target column name without a lag/shift indicator.
    """
    risky = []
    safe_indicators = ["lag", "rolling", "shift", "prev", "t_minus"]
    for name in feature_names:
        if target_col in name:
            is_safe = any(ind in name.lower() for ind in safe_indicators)
            if not is_safe and name != target_col:
                risky.append(name)
    return risky
