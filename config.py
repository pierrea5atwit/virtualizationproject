from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


ConfigDict = Dict[str, Any]


def load_config(path: str | Path) -> ConfigDict:
    """Load and validate a YAML experiment configuration file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    required_top_level = ["environment", "experiment", "workload", "concurrency", "monitoring", "logging"]
    missing = [key for key in required_top_level if key not in data]
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")

    return data
