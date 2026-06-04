"""DEPRECATED: Use src.models.trainer and src.evaluation.metrics instead.

This module is kept only for backward compatibility and will be removed in v0.2.0.
All logic has been moved to the canonical modules:
  - temporal_split → src.models.trainer.split_temporal
  - asymmetric_rmse → src.evaluation.metrics.asymmetric_rmse
  - compute_metrics → src.evaluation.metrics.compute_metrics
"""
import warnings

warnings.warn(
    "src.core.train is deprecated. Use src.models.trainer and src.evaluation.metrics instead.",
    DeprecationWarning,
    stacklevel=2,
)

from src.evaluation.metrics import asymmetric_rmse, compute_metrics  # noqa: F401
from src.models.trainer import split_temporal as temporal_split  # noqa: F401
