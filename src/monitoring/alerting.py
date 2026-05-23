"""Alerting — threshold-based alerts from drift + health checks (report-only MVP)."""
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class AlertManager:
    """Collects alerts from monitoring checks and writes report artifacts."""

    def __init__(self, output_dir: str = "data/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.alerts: List[Dict[str, Any]] = []

    def add_alert(self, severity: str, source: str, message: str, details: Dict = None):
        """severity: info | warning | critical"""
        alert = {
            "timestamp": datetime.utcnow().isoformat(),
            "severity": severity,
            "source": source,
            "message": message,
            "details": details or {},
        }
        self.alerts.append(alert)
        log_fn = {"info": logger.info, "warning": logger.warning, "critical": logger.error}.get(severity, logger.info)
        log_fn(f"[{source}] {message}")

    def evaluate_drift_report(self, drift_report: Dict[str, Any]):
        """Generate alerts from drift detection results."""
        summary = drift_report.get("feature_drift_summary", {})
        significant = summary.get("significant", 0)
        moderate = summary.get("moderate", 0)
        total = summary.get("total", 0)

        if significant > 0:
            self.add_alert(
                "critical",
                "drift",
                f"{significant}/{total} features show significant drift (PSI > 0.2 or KS p < 0.05)",
                {"significant_count": significant, "total_features": total},
            )
        elif moderate > 0:
            self.add_alert(
                "warning",
                "drift",
                f"{moderate}/{total} features show moderate drift (PSI 0.1-0.2)",
                {"moderate_count": moderate, "total_features": total},
            )
        else:
            self.add_alert("info", "drift", "All features stable — no significant drift detected")

        # Per-feature alerts for significant drift
        for feat in drift_report.get("per_feature_drift", []):
            if feat.get("drift_level") == "significant":
                self.add_alert(
                    "warning",
                    "drift",
                    f"Feature '{feat['feature']}' drifted: PSI={feat['psi']}, KS_p={feat['ks_p_value']}",
                    feat,
                )

    def evaluate_health_report(self, health_report: Dict[str, Any]):
        """Generate alerts from data health checks."""
        checks = health_report.get("checks", {})

        schema = checks.get("schema", {})
        if not schema.get("passed", True):
            missing = schema.get("missing_columns", [])
            self.add_alert("critical", "health", f"Schema check failed: missing columns {missing}")

        completeness = checks.get("completeness", {})
        if not completeness.get("passed", True):
            alerts = completeness.get("alerts", [])
            for a in alerts:
                self.add_alert(
                    "warning",
                    "health",
                    f"Column '{a['column']}' null rate {a['null_rate']:.1%} exceeds threshold {a['threshold']:.1%}",
                    a,
                )

        duplicates = checks.get("duplicates", {})
        if not duplicates.get("passed", True):
            self.add_alert(
                "warning",
                "health",
                f"{duplicates.get('duplicate_rows', 0)} duplicate rows on key {duplicates.get('key_columns', [])}",
            )

        freshness = checks.get("freshness", {})
        if not freshness.get("passed", True):
            days = freshness.get("days_behind", "unknown")
            self.add_alert(
                "warning",
                "health",
                f"Data stale — latest record is {days} days behind (threshold: {freshness.get('threshold_days', '?')})",
                freshness,
            )

    def write_report(self, drift_report: Dict = None, health_report: Dict = None) -> str:
        """Write combined monitoring report to JSON file."""
        # Avoid circular references by copying data into simple dicts
        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "alert_count": len(self.alerts),
            "alerts": [a.copy() for a in self.alerts],
        }

        if drift_report:
            # Deep copy just the necessary parts to avoid circular refs in nested objects
            report["drift_report"] = json.loads(json.dumps(drift_report, default=str))
        if health_report:
            report["health_report"] = json.loads(json.dumps(health_report, default=str))

        filename = f"monitoring_report_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.json"
        output_path = self.output_dir / filename

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Monitoring report saved to {output_path}")
        return str(output_path)

    def summary(self) -> str:
        """Human-readable alert summary."""
        if not self.alerts:
            return "All checks passed. No alerts."

        lines = [f"Monitoring Alerts ({len(self.alerts)} total):"]
        for a in self.alerts:
            emoji = {"critical": "🔴", "warning": "🟡", "info": "🟢"}.get(a["severity"], "⚪")
            lines.append(f"  {emoji} [{a['severity'].upper()}] {a['source']}: {a['message']}")
        return "\n".join(lines)
