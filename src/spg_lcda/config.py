from pathlib import Path

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path, _seen: set[Path] | None = None) -> dict:
    """Load YAML and resolve an optional repository-relative ``inherits`` path."""
    config_path = Path(path).resolve()
    seen = set() if _seen is None else _seen
    if config_path in seen:
        raise ValueError(f"Circular configuration inheritance: {config_path}")
    seen.add(config_path)

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Experiment configuration must be a YAML mapping")

    parent = config.pop("inherits", None)
    if parent is None:
        return config
    parent_config = load_config(parent, seen)
    return _deep_merge(parent_config, config)
