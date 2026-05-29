# Copyright 2026 Ting Liang and PESMaker development team
# This file is part of PESMaker.
#
# PESMaker is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PESMaker is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PESMaker. If not, see <https://www.gnu.org/licenses/>.
"""Load PESMaker YAML and TOML configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pesmaker.config.schema import PESMakerConfig

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    tomllib = None


def load_config(path: str | Path) -> PESMakerConfig:
    """Load and validate a PESMaker configuration file.

    Args:
        path: YAML or TOML configuration file path.

    Returns:
        A validated `PESMakerConfig` object ready for workflow planning or
        execution.

    Raises:
        FileNotFoundError: If `path` does not exist.
        ValueError: If the file suffix or top-level structure is invalid.
        RuntimeError: If a YAML file is requested but PyYAML is unavailable.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    data = _load_mapping(config_path)
    return PESMakerConfig.from_mapping(data)


def _load_mapping(path: Path) -> dict[str, Any]:
    """Read a YAML or TOML file into a raw mapping.

    Args:
        path: Existing input file with `.yaml`, `.yml`, or `.toml` suffix.

    Returns:
        Raw dictionary loaded from the file.

    Raises:
        ValueError: If the suffix is unsupported or the top-level document is
            not a mapping.
        RuntimeError: If PyYAML is required but not installed.
    """
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            message = "YAML config files require PyYAML. Install pesmaker with PyYAML."
            raise RuntimeError(message) from exc
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.load(handle, Loader=_UniqueKeyLoader)
    elif suffix == ".toml":
        if tomllib is None:
            message = (
                "TOML config files require Python 3.11+ or the optional 'tomli' "
                "package. Use YAML config files to avoid this extra dependency."
            )
            raise RuntimeError(message)
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    else:
        raise ValueError(f"unsupported config suffix: {path.suffix}")

    if not isinstance(data, dict):
        raise ValueError(f"config must contain a mapping at top level: {path}")
    return data


def _construct_unique_mapping(loader, node, deep=False):
    """Construct a YAML mapping while rejecting duplicate keys."""
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            if key == "supercell":
                raise ValueError(
                    "duplicate YAML key: supercell. Use generation.tasks for "
                    "multiple independent supercells."
                )
            raise ValueError(f"duplicate YAML key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


try:
    import yaml

    class _UniqueKeyLoader(yaml.SafeLoader):
        """PyYAML loader that rejects duplicate mapping keys."""

    _UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        _construct_unique_mapping,
    )
except ImportError:  # pragma: no cover - handled when YAML loading is requested
    _UniqueKeyLoader = None
