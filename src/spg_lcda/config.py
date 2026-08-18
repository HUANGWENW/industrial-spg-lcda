from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """Load an experiment configuration from YAML."""
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Experiment configuration must be a YAML mapping")
    return config

