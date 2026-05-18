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
"""Configuration schema and validation for PESMaker input files."""

from __future__ import annotations

from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StructureInput:
    """A user-provided structure file used as a generation seed."""

    path: Path

    @classmethod
    def from_value(cls, data: Any) -> "StructureInput":
        """Build a structure input from either a path string or a mapping."""
        if isinstance(data, (str, Path)):
            return cls(path=Path(data))
        return cls.from_mapping(_require_mapping(data, "structures entry"))

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "StructureInput":
        """Build a structure input from the explicit `{path: ...}` form."""
        path = data.get("path")
        if not path:
            raise ValueError("each structure entry requires 'path'")
        return cls(path=Path(str(path)))


@dataclass(frozen=True)
class GenerationConfig:
    """Options for supercell construction and structure perturbation."""

    supercell: tuple[int, int, int] = (1, 1, 1)
    output_dir: Path | None = None
    perturb: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "GenerationConfig":
        """Parse the `generation` section of a PESMaker config."""
        data = data or {}
        supercell = data.get("supercell", [1, 1, 1])
        if len(supercell) != 3:
            raise ValueError("generation.supercell must contain three integers")
        return cls(
            supercell=tuple(int(value) for value in supercell),
            output_dir=Path(str(data["output_dir"]))
            if data.get("output_dir")
            else None,
            perturb=dict(data.get("perturb", {})),
        )


@dataclass(frozen=True)
class EngineConfig:
    """Generic external engine configuration with free-form options."""

    engine: str
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        data: dict[str, Any] | None,
        *,
        default_engine: str,
        alias_engine_key: str | None = None,
    ) -> "EngineConfig":
        """Parse a workflow engine section such as `labeling` or `training`."""
        data = data or {}
        engine_value = data.get("engine", default_engine)
        if alias_engine_key and "engine" not in data:
            engine_value = data.get(alias_engine_key, engine_value)
        engine = str(engine_value)
        excluded_keys = {"engine"}
        if alias_engine_key:
            excluded_keys.add(alias_engine_key)
        options = {
            key: value for key, value in data.items() if key not in excluded_keys
        }
        return cls(engine=engine, options=options)


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset export format and train/validation/test split settings."""

    format: str = "extxyz"
    split: tuple[float, float, float] = (0.8, 0.1, 0.1)

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "DatasetConfig":
        """Parse and validate the `dataset` section."""
        data = data or {}
        split = data.get("split", [0.8, 0.1, 0.1])
        if len(split) != 3:
            raise ValueError(
                "dataset.split must contain train, validation, test ratios"
            )
        split_tuple = tuple(float(value) for value in split)
        if abs(sum(split_tuple) - 1.0) > 1e-8:
            raise ValueError("dataset.split ratios must sum to 1.0")
        return cls(format=str(data.get("format", "extxyz")), split=split_tuple)


@dataclass(frozen=True)
class PESMakerConfig:
    """Top-level validated PESMaker configuration."""

    project: str
    structures: tuple[StructureInput, ...]
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    sampling: EngineConfig = field(
        default_factory=lambda: EngineConfig(engine="none", options={})
    )
    labeling: EngineConfig = field(
        default_factory=lambda: EngineConfig(engine="vasp", options={})
    )
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: EngineConfig = field(
        default_factory=lambda: EngineConfig(engine="nep", options={})
    )

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "PESMakerConfig":
        """Create a validated config object from a raw mapping."""
        project = data.get("project")
        if not project:
            raise ValueError("config requires 'project'")

        structures = _parse_structures(data.get("structures"))

        return cls(
            project=str(project),
            structures=structures,
            generation=GenerationConfig.from_mapping(
                _optional_mapping(data.get("generation"), "generation")
            ),
            sampling=EngineConfig.from_mapping(
                _optional_mapping(data.get("sampling"), "sampling"),
                default_engine="none",
            ),
            labeling=EngineConfig.from_mapping(
                _optional_mapping(data.get("labeling"), "labeling"),
                default_engine="vasp",
            ),
            dataset=DatasetConfig.from_mapping(
                _optional_mapping(data.get("dataset"), "dataset")
            ),
            training=EngineConfig.from_mapping(
                _optional_mapping(data.get("training"), "training"),
                default_engine="nep",
                alias_engine_key="model",
            ),
        )


def _optional_mapping(value: Any, name: str) -> dict[str, Any] | None:
    """Return `None` for missing sections and validate provided mappings."""
    if value is None:
        return None
    return _require_mapping(value, name)


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    """Validate that a config section is a mapping."""
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _parse_structures(value: Any) -> tuple[StructureInput, ...]:
    """Parse structure inputs from a simple list or an `include` glob map."""
    if isinstance(value, list):
        if not value:
            raise ValueError("config requires at least one structure")
        return tuple(StructureInput.from_value(entry) for entry in value)

    if isinstance(value, dict):
        include = value.get("include")
        if not isinstance(include, list) or not include:
            raise ValueError("structures.include must be a non-empty list")
        paths: list[Path] = []
        for pattern in include:
            matches = [Path(match) for match in sorted(glob(str(pattern)))]
            if not matches:
                raise ValueError(
                    f"structures include pattern matched no files: {pattern}"
                )
            paths.extend(matches)
        return tuple(StructureInput(path=path) for path in paths)

    raise ValueError("config requires 'structures' as a non-empty list or include map")
