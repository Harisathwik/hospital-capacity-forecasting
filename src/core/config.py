"""Core config loader + shared utilities."""


from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    """Load the YAML config into a flat-ish dict."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
