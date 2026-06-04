"""Model evaluator — compare models and select best for promotion."""
import numpy as np
from typing import Dict, Any, List, Optional
from src.evaluation.metrics import compute_metrics


class ModelEvaluator:
    """Compare candidate models and select the best one for promotion."""

    def __init__(self, primary_metric: str = "asymmetric_rmse", lower_is_better: bool = True):
        self.primary_metric = primary_metric
        self.lower_is_better = lower_is_better

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "candidate",
        underprediction_penalty: float = 3.0,
    ) -> Dict[str, Any]:
        """Compute all metrics for a single model."""
        metrics = compute_metrics(y_true, y_pred, underprediction_penalty)
        metrics["model_name"] = model_name
        return metrics

    def compare(
        self,
        y_true: np.ndarray,
        candidates: Dict[str, np.ndarray],
        underprediction_penalty: float = 3.0,
    ) -> Dict[str, Any]:
        """Compare multiple candidate models and return ranked results.

        Args:
            y_true: Ground truth
            candidates: Dict of {model_name: predictions}

        Returns:
            Dict with 'ranked' list, 'best' name, and 'best_metrics'.
        """
        results = []
        for name, preds in candidates.items():
            metrics = self.evaluate(y_true, preds, name, underprediction_penalty)
            results.append(metrics)

        results.sort(
            key=lambda m: m[self.primary_metric],
            reverse=not self.lower_is_better,
        )

        return {
            "ranked": results,
            "best": results[0]["model_name"],
            "best_metrics": results[0],
        }
