from __future__ import annotations

from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StructureInput:
    path: Path
    role: str = "initial"

    @classmethod
    def from_value(cls, data: Any) -> "StructureInput":
        if isinstance(data, (str, Path)):
            return cls(path=Path(data))
        return cls.from_mapping(_require_mapping(data, "structures entry"))

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "StructureInput":
        path = data.get("path")
        if not path:
            raise ValueError("each structure entry requires 'path'")
        return cls(path=Path(str(path)), role=str(data.get("role", "initial")))


@dataclass(frozen=True)
class GenerationConfig:
    supercell: tuple[int, int, int] = (1, 1, 1)
    output_dir: Path | None = None
    perturb: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "GenerationConfig":
        data = data or {}
        supercell = data.get("supercell", [1, 1, 1])
        if len(supercell) != 3:
            raise ValueError("generation.supercell must contain three integers")
        return cls(
            supercell=tuple(int(value) for value in supercell),
            output_dir=Path(str(data["output_dir"])) if data.get("output_dir") else None,
            perturb=dict(data.get("perturb", {})),
        )


@dataclass(frozen=True)
class EngineConfig:
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
        data = data or {}
        engine_value = data.get("engine", default_engine)
        if alias_engine_key and "engine" not in data:
            engine_value = data.get(alias_engine_key, engine_value)
        engine = str(engine_value)
        excluded_keys = {"engine"}
        if alias_engine_key:
            excluded_keys.add(alias_engine_key)
        options = {key: value for key, value in data.items() if key not in excluded_keys}
        return cls(engine=engine, options=options)


@dataclass(frozen=True)
class DatasetConfig:
    format: str = "extxyz"
    split: tuple[float, float, float] = (0.8, 0.1, 0.1)

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "DatasetConfig":
        data = data or {}
        split = data.get("split", [0.8, 0.1, 0.1])
        if len(split) != 3:
            raise ValueError("dataset.split must contain train, validation, test ratios")
        split_tuple = tuple(float(value) for value in split)
        if abs(sum(split_tuple) - 1.0) > 1e-8:
            raise ValueError("dataset.split ratios must sum to 1.0")
        return cls(format=str(data.get("format", "extxyz")), split=split_tuple)


@dataclass(frozen=True)
class PESMakerConfig:
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
    if value is None:
        return None
    return _require_mapping(value, name)


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _parse_structures(value: Any) -> tuple[StructureInput, ...]:
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
                raise ValueError(f"structures include pattern matched no files: {pattern}")
            paths.extend(matches)
        return tuple(StructureInput(path=path) for path in paths)

    raise ValueError("config requires 'structures' as a non-empty list or include map")

