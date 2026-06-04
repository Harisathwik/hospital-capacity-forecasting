"""Promotion gates — enforce quality criteria before model promotion.

Four gates (all must pass for promotion):
  1. beats_naive_baseline       – model must outperform naive persistence baseline
  2. data_validation_pass       – data validation report must show overall PASS
  3. reproducible_run           – re-run with same config/data must reproduce metrics within tolerance
  4. underprediction_rate_below_threshold – underprediction rate must stay below configured max

Pure Python — no ZenML / MLflow imports.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    """Result of a single promotion gate evaluation."""
    gate_name: str
    passed: bool
    required: bool
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"


@dataclass
class PromotionReport:
    """Aggregated promotion report across all gates."""
    gates: List[GateResult] = field(default_factory=list)
    overall_passed: bool = False

    @property
    def failed_gates(self) -> List[GateResult]:
        return [g for g in self.gates if not g.passed]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_passed": self.overall_passed,
            "overall_status": "PROMOTED" if self.overall_passed else "BLOCKED",
            "gates": [
                {
                    "gate": g.gate_name,
                    "status": g.status,
                    "required": g.required,
                    "details": g.details,
                }
                for g in self.gates
            ],
            "failed_gates": [g.gate_name for g in self.failed_gates],
        }


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class PromotionBlockedError(Exception):
    """Raised when one or more promotion gates fail, blocking model promotion."""

    def __init__(self, report: PromotionReport):
        self.report = report
        failed = report.failed_gates
        names = ", ".join(g.gate_name for g in failed)
        super().__init__(
            f"Promotion BLOCKED — {len(failed)} gate(s) failed: {names}"
        )


# ---------------------------------------------------------------------------
# Individual gate implementations
# ---------------------------------------------------------------------------

def gate_beats_naive_baseline(
    model_asymmetric_rmse: float,
    naive_asymmetric_rmse: float,
) -> GateResult:
    """Gate 1: model must beat (lower) the naive baseline asymmetric RMSE."""
    passed = model_asymmetric_rmse < naive_asymmetric_rmse
    return GateResult(
        gate_name="beats_naive_baseline",
        passed=passed,
        required=True,
        details={
            "model_asymmetric_rmse": round(model_asymmetric_rmse, 6),
            "naive_asymmetric_rmse": round(naive_asymmetric_rmse, 6),
            "improvement_pct": round(
                (naive_asymmetric_rmse - model_asymmetric_rmse) / naive_asymmetric_rmse * 100, 4
            ) if naive_asymmetric_rmse != 0 else 0.0,
        },
    )


def gate_data_validation_pass(
    validation_report: Dict[str, Any],
) -> GateResult:
    """Gate 2: overall_status in the validation report must be 'PASS'."""
    overall_status = validation_report.get("overall_status", "UNKNOWN")
    passed = overall_status == "PASS"
    return GateResult(
        gate_name="data_validation_pass",
        passed=passed,
        required=True,
        details={
            "overall_status": overall_status,
        },
    )


def gate_reproducible_run(
    current_metrics: Dict[str, float],
    previous_metrics: Dict[str, float],
    tolerance: float = 0.01,
) -> GateResult:
    """Gate 3: current run metrics must match previous run within relative tolerance.

    Default tolerance = 0.01 (1% relative difference).
    A metric passes if |current - previous| / |previous| <= tolerance.
    """
    # Compare on common keys only
    common_keys = sorted(set(current_metrics.keys()) & set(previous_metrics.keys()))
    deviations: Dict[str, Dict[str, float]] = {}
    all_within = True

    for key in common_keys:
        prev = previous_metrics[key]
        curr = current_metrics[key]
        if prev == 0:
            rel_diff = 0.0 if curr == 0 else float("inf")
        else:
            rel_diff = abs(curr - prev) / abs(prev)
        within = rel_diff <= tolerance
        deviations[key] = {
            "previous": round(prev, 8),
            "current": round(curr, 8),
            "relative_diff": round(rel_diff, 6),
            "within_tolerance": within,
        }
        if not within:
            all_within = False

    return GateResult(
        gate_name="reproducible_run",
        passed=all_within,
        required=True,
        details={
            "tolerance": tolerance,
            "metrics_compared": len(deviations),
            "deviations": deviations,
        },
    )


def gate_underprediction_rate_below_threshold(
    underprediction_rate: float,
    max_underprediction_rate: float = 0.3,
) -> GateResult:
    """Gate 4: underprediction rate must be ≤ max allowed threshold."""
    passed = underprediction_rate <= max_underprediction_rate
    return GateResult(
        gate_name="underprediction_rate_below_threshold",
        passed=passed,
        required=True,
        details={
            "underprediction_rate": round(underprediction_rate, 6),
            "max_underprediction_rate": max_underprediction_rate,
            "margin": round(max_underprediction_rate - underprediction_rate, 6),
        },
    )


# ---------------------------------------------------------------------------
# Promotion gate runner
# ---------------------------------------------------------------------------

class PromotionGateRunner:
    """Run all configured promotion gates and decide promote vs. block.

    Parameters
    ----------
    config : dict
        Full project config (loaded from configs/config.yaml).
        Expected keys:
          governance.promotion_gates.require_beats_naive_baseline  (bool)
          governance.promotion_gates.require_data_validation_pass  (bool)
          governance.promotion_gates.require_reproducible_run      (bool)
          governance.promotion_gates.require_underprediction_rate_below_threshold (bool)
          governance.thresholds.max_underprediction_rate  (float | None -> defaults to 0.3)
    """

    # Default threshold when config value is null / missing
    DEFAULT_MAX_UNDERPREDICTION_RATE = 0.3
    # Default reproducibility tolerance (1% relative diff)
    DEFAULT_REPRODUCIBILITY_TOLERANCE = 0.01

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._gates_cfg = config.get("governance", {}).get("promotion_gates", {})
        self._thresholds = config.get("governance", {}).get("thresholds", {})

    # -- public API ---------------------------------------------------------

    def run(
        self,
        model_asymmetric_rmse: float,
        naive_asymmetric_rmse: float,
        validation_report: Dict[str, Any],
        current_metrics: Dict[str, float],
        previous_metrics: Dict[str, float],
        underprediction_rate: float,
    ) -> PromotionReport:
        """Evaluate all four promotion gates and return a report.

        Parameters
        ----------
        model_asymmetric_rmse : float
            The candidate model's asymmetric RMSE on the evaluation set.
        naive_asymmetric_rmse : float
            The naive baseline's asymmetric RMSE on the same set.
        validation_report : dict
            Data validation report dict with at least ``overall_status`` key.
        current_metrics : dict
            Metrics from the current model run  (e.g. {"asymmetric_rmse": 12.3, ...}).
        previous_metrics : dict
            Metrics from the last recorded / reference run (same keys).
        underprediction_rate : float
            Fraction of predictions that underpredicted (y_pred < y_true).
        """
        gate_results: List[GateResult] = []

        # Gate 1 — beats naive baseline
        if self._gates_cfg.get("require_beats_naive_baseline", True):
            r = gate_beats_naive_baseline(model_asymmetric_rmse, naive_asymmetric_rmse)
            gate_results.append(r)
            logger.info("Gate [%s]: %s | details=%s", r.gate_name, r.status, r.details)

        # Gate 2 — data validation pass
        if self._gates_cfg.get("require_data_validation_pass", True):
            r = gate_data_validation_pass(validation_report)
            gate_results.append(r)
            logger.info("Gate [%s]: %s | details=%s", r.gate_name, r.status, r.details)

        # Gate 3 — reproducible run
        if self._gates_cfg.get("require_reproducible_run", True):
            tolerance = self._thresholds.get("reproducibility_tolerance", self.DEFAULT_REPRODUCIBILITY_TOLERANCE)
            r = gate_reproducible_run(current_metrics, previous_metrics, tolerance=tolerance)
            gate_results.append(r)
            logger.info("Gate [%s]: %s | details=%s", r.gate_name, r.status, r.details)

        # Gate 4 — underprediction rate below threshold
        if self._gates_cfg.get("require_underprediction_rate_below_threshold", True):
            max_upr = self._thresholds.get("max_underprediction_rate", None)
            if max_upr is None:
                max_upr = self.DEFAULT_MAX_UNDERPREDICTION_RATE
            r = gate_underprediction_rate_below_threshold(underprediction_rate, max_upr)
            gate_results.append(r)
            logger.info("Gate [%s]: %s | details=%s", r.gate_name, r.status, r.details)

        overall = all(g.passed for g in gate_results)
        report = PromotionReport(gates=gate_results, overall_passed=overall)

        # Log the final decision
        if overall:
            logger.info("Promotion decision: PROMOTED — all gates passed")
        else:
            failed_names = [g.gate_name for g in report.failed_gates]
            logger.warning(
                "Promotion decision: BLOCKED — %d gate(s) failed: %s",
                len(failed_names), ", ".join(failed_names),
            )

        return report

    def promote_or_block(self, report: PromotionReport) -> None:
        """Raise ``PromotionBlockedError`` if any gate failed.

        Call after ``run()`` to enforce the gate decision.
        """
        if not report.overall_passed:
            raise PromotionBlockedError(report)
        logger.info("Model PROMOTED successfully")
