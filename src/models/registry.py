"""Model registry — promote models through governance gates."""
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Simple file-based model registry tracking.

    For production, use MLflow Model Registry instead.
    This provides a lightweight alternative for local dev.
    """

    def __init__(self, registry_dir: str = "data/model_registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    def register(
        self,
        model_name: str,
        stage: str,
        run_id: str,
        metrics: Dict[str, float],
        artifact_path: str,
    ) -> Dict[str, Any]:
        """Register a model version."""
        import json
        from datetime import datetime

        entry = {
            "model_name": model_name,
            "stage": stage,
            "run_id": run_id,
            "metrics": metrics,
            "artifact_path": artifact_path,
            "registered_at": datetime.utcnow().isoformat(),
        }

        registry_file = self.registry_dir / f"{model_name}_{stage}.json"
        with open(registry_file, "w") as f:
            json.dump(entry, f, indent=2)

        logger.info(f"Registered {model_name} at {stage} stage (run_id={run_id})")
        return entry

    def get_latest(self, model_name: str, stage: str = "Production") -> Optional[Dict[str, Any]]:
        """Get the latest registered model at a given stage."""
        import json

        registry_file = self.registry_dir / f"{model_name}_{stage}.json"
        if not registry_file.exists():
            return None
        with open(registry_file) as f:
            return json.load(f)
