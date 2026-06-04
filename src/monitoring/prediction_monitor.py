"""Prediction monitoring — track output distribution health, not just input drift.

Checks:
  - prediction_drift   : PSI between current and baseline prediction distributions
  - dead_predictions   : >50% of predictions identical (degenerate / stalled model)
  - prediction_freshness: is the model still producing timely predictions?
  - negative_predictions: any negative ICU occupancy forecasts?

Pure Python (numpy, scipy) — no ZenML / MLflow.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import numpy as np

from src.monitoring.drift_detector import compute_psi

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults (can be overridden via config)
# ---------------------------------------------------------------------------

DEFAULT_PSI_THRESHOLD = 0.2          # significant drift
DEFAULT_DEAD_PREDICTION_FRACTION = 0.5  # >50% identical → dead
DEFAULT_STALE_MINUTES = 60


class PredictionMonitor:
    """Monitor the health of model *predictions* (outputs), not just inputs.

    Parameters
    ----------
    baseline_predictions : np.ndarray
        1-D array of baseline (e.g. training / last-known-good) predictions.
    config : dict
        Project config. Looks for ``monitoring.prediction_monitoring`` sub-dict
        for threshold overrides.  Falls back to ``monitoring.psi_threshold`` for
        the PSI drift threshold if the sub-dict is absent.
    """

    def __init__(self, baseline_predictions: np.ndarray, config: Dict[str, Any]):
        self.baseline_predictions = np.asarray(baseline_predictions, dtype=float)
        self.config = config

        # Resolve thresholds from config
        pm_cfg = config.get("monitoring", {}).get("prediction_monitoring", {})
        mon_cfg = config.get("monitoring", {})

        self.psi_threshold: float = pm_cfg.get(
            "psi_threshold", mon_cfg.get("psi_threshold", DEFAULT_PSI_THRESHOLD)
        )
        self.dead_fraction: float = pm_cfg.get(
            "dead_prediction_fraction", DEFAULT_DEAD_PREDICTION_FRACTION
        )
        self.max_stale_minutes: int = int(
            pm_cfg.get("max_stale_minutes", DEFAULT_STALE_MINUTES)
        )

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def check_prediction_drift(
        self, current_predictions: np.ndarray
    ) -> Dict[str, Any]:
        """PSI between the current prediction distribution and the baseline.

        Returns dict with keys: check, passed, psi, threshold, drift_level.
        """
        current = np.asarray(current_predictions, dtype=float)

        psi = compute_psi(self.baseline_predictions, current)

        if psi >= self.psi_threshold:
            drift_level = "significant"
        elif psi >= self.psi_threshold / 2:
            drift_level = "moderate"
        else:
            drift_level = "none"

        passed = psi < self.psi_threshold

        result = {
            "check": "prediction_drift",
            "passed": passed,
            "psi": round(psi, 6),
            "threshold": self.psi_threshold,
            "drift_level": drift_level,
        }
        logger.info(
            "Prediction drift: PSI=%.4f (threshold=%.4f) → %s",
            psi, self.psi_threshold, drift_level,
        )
        return result

    def check_dead_predictions(
        self, predictions: np.ndarray
    ) -> Dict[str, Any]:
        """Detect if a large fraction of predictions are identical.

        If > ``dead_fraction`` (default 50%) of predictions share the same
        value, the model may be stuck / outputting a constant.
        """
        preds = np.asarray(predictions, dtype=float)
        if len(preds) == 0:
            return {
                "check": "dead_predictions",
                "passed": False,
                "reason": "empty_predictions",
                "identical_fraction": 1.0,
            }

        # Find the most common value
        unique, counts = np.unique(preds, return_counts=True)
        max_count = int(counts.max())
        identical_fraction = max_count / len(preds)

        passed = identical_fraction < self.dead_fraction

        result = {
            "check": "dead_predictions",
            "passed": passed,
            "identical_fraction": round(identical_fraction, 6),
            "threshold": self.dead_fraction,
            "most_common_value": float(unique[counts.argmax()]),
            "most_common_count": max_count,
            "total_predictions": len(preds),
            "unique_values": len(unique),
        }
        if not passed:
            logger.warning(
                "Dead prediction signal: %.1f%% of predictions are identical (threshold %.1f%%)",
                identical_fraction * 100, self.dead_fraction * 100,
            )
        return result

    def check_prediction_freshness(
        self,
        last_prediction_time: datetime,
        max_stale_minutes: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Check that predictions are being produced within the freshness SLA.

        Parameters
        ----------
        last_prediction_time : datetime
            Timestamp of the most recent prediction.
        max_stale_minutes : int, optional
            Override for the max allowed staleness in minutes.
        """
        if max_stale_minutes is None:
            max_stale_minutes = self.max_stale_minutes

        now = datetime.utcnow()
        stale_minutes = (now - last_prediction_time).total_seconds() / 60.0
        passed = stale_minutes <= max_stale_minutes

        result = {
            "check": "prediction_freshness",
            "passed": passed,
            "last_prediction_time": last_prediction_time.isoformat(),
            "current_time": now.isoformat(),
            "stale_minutes": round(stale_minutes, 2),
            "max_stale_minutes": max_stale_minutes,
        }
        if not passed:
            logger.warning(
                "Predictions are stale: %.1f min since last prediction (max %d min)",
                stale_minutes, max_stale_minutes,
            )
        return result

    def check_negative_predictions(
        self, predictions: np.ndarray
    ) -> Dict[str, Any]:
        """Flag any negative ICU occupancy predictions (physically impossible)."""
        preds = np.asarray(predictions, dtype=float)
        n_negative = int(np.sum(preds < 0))
        passed = n_negative == 0

        result = {
            "check": "negative_predictions",
            "passed": passed,
            "n_negative": n_negative,
            "total_predictions": len(preds),
            "min_prediction": float(preds.min()) if len(preds) > 0 else None,
        }
        if not passed:
            fraction = n_negative / len(preds) if len(preds) > 0 else 0.0
            logger.warning(
                "Negative predictions detected: %d / %d (%.2f%%)",
                n_negative, len(preds), fraction * 100,
            )
        return result

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    def run_all(
        self,
        current_predictions: np.ndarray,
        last_prediction_time: datetime,
    ) -> Dict[str, Any]:
        """Run every prediction health check and return an aggregate report.

        Parameters
        ----------
        current_predictions : np.ndarray
            The current batch of model predictions.
        last_prediction_time : datetime
            Timestamp of the most recent prediction for freshness check.

        Returns
        -------
        dict with keys:
            overall_status  – "healthy" | "degraded"
            timestamp       – ISO-8601 evaluation time
            checks          – dict of individual check results
            failed_checks   – list of check names that failed
        """
        checks: Dict[str, Dict[str, Any]] = {}

        checks["prediction_drift"] = self.check_prediction_drift(current_predictions)
        checks["dead_predictions"] = self.check_dead_predictions(current_predictions)
        checks["prediction_freshness"] = self.check_prediction_freshness(last_prediction_time)
        checks["negative_predictions"] = self.check_negative_predictions(current_predictions)

        failed = [name for name, res in checks.items() if not res.get("passed", False)]
        overall = "healthy" if len(failed) == 0 else "degraded"

        report = {
            "overall_status": overall,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": checks,
            "failed_checks": failed,
        }

        if failed:
            logger.warning(
                "Prediction monitor: DEGRADED — %d check(s) failed: %s",
                len(failed), ", ".join(failed),
            )
        else:
            logger.info("Prediction monitor: all checks passed (healthy)")

        return report
