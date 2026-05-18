# Copyright (c) 2026 Ting Liang. All rights reserved.
"""Load PESMaker YAML and TOML configuration files."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from pesmaker.config.schema import PESMakerConfig


def load_config(path: str | Path) -> PESMakerConfig:
    """Load and validate a PESMaker configuration file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    data = _load_mapping(config_path)
    return PESMakerConfig.from_mapping(data)


def _load_mapping(path: Path) -> dict[str, Any]:
    """Read a YAML or TOML file into a raw mapping."""
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            message = "YAML config files require PyYAML. Install pesmaker with PyYAML."
            raise RuntimeError(message) from exc
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    elif suffix == ".toml":
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    else:
        raise ValueError(f"unsupported config suffix: {path.suffix}")

    if not isinstance(data, dict):
        raise ValueError(f"config must contain a mapping at top level: {path}")
    return data

