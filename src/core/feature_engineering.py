"""DEPRECATED: Use src.features.engineer instead.

This module is kept only for backward compatibility and will be removed in v0.2.0.
All logic has been moved to the canonical module:
  - build_features → src.features.engineer.build_features
"""
import warnings

warnings.warn(
    "src.core.feature_engineering is deprecated. Use src.features.engineer instead.",
    DeprecationWarning,
    stacklevel=2,
)

from src.features.engineer import build_features  # noqa: F401
