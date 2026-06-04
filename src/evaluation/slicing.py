"""Slice-level evaluation — evaluate model performance by subgroups.

Provides per-slice metric computation for fair evaluation across:
  - States (geographic)
  - Day of week (temporal)
  - Season (quarterly)
  - High vs low occupancy (value-based)
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from src.evaluation.metrics import compute_metrics


def slice_by_column(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    slice_values: pd.Series,
    underprediction_penalty: float = 3.0,
) -> Dict[str, Dict[str, float]]:
    """Compute metrics per unique value in slice_values.

    Args:
        y_true: True values
        y_pred: Predicted values
        slice_values: Series with same length, to group by
        underprediction_penalty: Asymmetric loss weight

    Returns:
        Dict mapping each unique slice value to its metrics dict.
    """
    results = {}
    for val in slice_values.unique():
        mask = slice_values == val
        if mask.sum() < 5:  # skip tiny slices
            continue
        metrics = compute_metrics(
            y_true[mask], y_pred[mask], underprediction_penalty=underprediction_penalty
        )
        metrics["n_samples"] = int(mask.sum())
        results[str(val)] = metrics
    return results


def slice_by_quantile(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    reference: np.ndarray,
    n_bins: int = 4,
    underprediction_penalty: float = 3.0,
) -> Dict[str, Dict[str, float]]:
    """Compute metrics per quantile bin of reference values.

    Useful for evaluating performance across high/low occupancy ranges.
    """
    results = {}
    bin_edges = np.nanpercentile(reference, np.linspace(0, 100, n_bins + 1))
    bin_edges = np.unique(bin_edges)

    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (reference >= lo) & (reference < hi)
        if mask.sum() < 5:
            continue
        metrics = compute_metrics(
            y_true[mask], y_pred[mask], underprediction_penalty=underprediction_penalty
        )
        metrics["n_samples"] = int(mask.sum())
        metrics["range"] = f"[{lo:.1f}, {hi:.1f})"
        results[f"bin_{i}"] = metrics

    return results
