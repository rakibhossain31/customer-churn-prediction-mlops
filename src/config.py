"""Configuration helpers for the Telco churn project."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Load YAML configuration and return an empty dict when the file is missing."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return data


def get_nested(config: Dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Read a dotted key from a nested dictionary."""
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current
