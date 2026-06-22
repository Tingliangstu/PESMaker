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
    """A user-provided structure file used as a generation seed.

    Attributes:
        path: Path to an ASE-readable structure file such as CIF, POSCAR, or
            extxyz.
    """

    path: Path

    @classmethod
    def from_value(cls, data: Any) -> "StructureInput":
        """Build a structure input from either a path string or a mapping.

        Args:
            data: Either a path-like value, or a mapping containing `path`.

        Returns:
            Parsed structure input.

        Raises:
            ValueError: If `data` is a mapping without a `path` key or is not a
                supported structure input form.
        """
        if isinstance(data, (str, Path)):
            return cls(path=Path(data))
        return cls.from_mapping(_require_mapping(data, "structures entry"))

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "StructureInput":
        """Build a structure input from the explicit `{path: ...}` form.

        Args:
            data: Mapping with a required `path` key.

        Returns:
            Parsed structure input.

        Raises:
            ValueError: If `path` is missing or empty.
        """
        path = data.get("path")
        if not path:
            raise ValueError("each structure entry requires 'path'")
        return cls(path=Path(str(path)))


@dataclass(frozen=True)
class GenerationTask:
    """One independent structure generation recipe.

    Attributes:
        name: Filesystem-safe task name used for output grouping.
        supercell: Three positive integer expansion factors.
        surface: Optional surface/vacuum settings.
        defects: Optional defect settings applied after surface generation.
        perturb: Optional perturbation settings applied to final variants.
    """

    name: str
    supercell: tuple[int, int, int] = (1, 1, 1)
    surface: dict[str, Any] = field(default_factory=dict)
    defects: dict[str, Any] = field(default_factory=dict)
    perturb: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationConfig:
    """Options for supercell construction and structure perturbation.

    Attributes:
        supercell: Three positive integer expansion factors.
        output_dir: Optional output directory. If omitted, PESMaker writes to
            `runs/<project>/generated`.
        surface: Optional surface/vacuum settings for slab-like 2D systems.
        defects: Optional vacancy and line-defect generation settings. These
            can be written directly under `generation.defects` or nested under
            `generation.surface.defects`.
        perturb: Free-form perturbation options consumed by
            `PerturbationSettings`. These can be written directly under
            `generation.perturb` or nested under `generation.surface.perturb`.
            PESMaker writes expanded pristine structures by default. Random
            perturbations are enabled only when `pert_num` is greater than
            zero. Set `include_pristine: true` inside `perturb` to also write
            pristine defect variants when random perturbations are enabled.
        tasks: Independent generation tasks. New configs should use this when
            multiple supercells or nested operation chains are needed.
    """

    supercell: tuple[int, int, int] = (1, 1, 1)
    output_dir: Path | None = None
    surface: dict[str, Any] = field(default_factory=dict)
    defects: dict[str, Any] = field(default_factory=dict)
    perturb: dict[str, Any] = field(default_factory=dict)
    tasks: tuple[GenerationTask, ...] = field(
        default_factory=lambda: (GenerationTask(name="default"),)
    )

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "GenerationConfig":
        """Parse the `generation` section of a PESMaker config.

        Args:
            data: Optional raw mapping from the config file.

        Returns:
            Parsed generation configuration.

        Raises:
            ValueError: If `supercell` does not contain exactly three values.
        """
        data = data or {}
        tasks = _parse_generation_tasks(data)
        first_task = tasks[0]

        return cls(
            supercell=first_task.supercell,
            output_dir=Path(str(data["output_dir"]))
            if data.get("output_dir")
            else None,
            surface=first_task.surface,
            defects=first_task.defects,
            perturb=first_task.perturb,
            tasks=tasks,
        )


@dataclass(frozen=True)
class EngineConfig:
    """Generic external engine configuration with free-form options.

    Attributes:
        engine: Selected engine name, such as `vasp`, `none`, `nep`, or `mace`.
        options: Remaining key-value options for the selected engine.
    """

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
        """Parse a workflow engine section such as `labeling` or `training`.

        Args:
            data: Optional raw engine mapping.
            default_engine: Engine name used when the section is omitted.
            alias_engine_key: Optional alternative key that can name the
                engine. This is used for concise inputs such as
                `training.model`.

        Returns:
            Parsed engine configuration with engine-specific options separated
            from the engine selector.
        """
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
    """Dataset export format and train/validation/test split settings.

    Attributes:
        format: Dataset export format, currently planned around `extxyz`.
        split: Train, validation, and test fractions.
    """

    format: str = "extxyz"
    split: tuple[float, float, float] = (0.8, 0.1, 0.1)

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "DatasetConfig":
        """Parse and validate the `dataset` section.

        Args:
            data: Optional raw dataset mapping.

        Returns:
            Parsed dataset configuration.

        Raises:
            ValueError: If `split` does not contain three values or the values
                do not sum to 1.0.
        """
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


WORKFLOW_MODES = {"auto", "direct-scf", "sampling-training"}


@dataclass(frozen=True)
class WorkflowConfig:
    """Optional advanced override for `pesmaker next`.

    Attributes:
        mode: Smart-next compatibility mode. `auto` lets PESMaker infer the
            flow from configured sections and artifacts. `direct-scf` skips
            sampling and training even if those sections are present.
            `sampling-training` is accepted for older configs.
    """

    mode: str = "auto"

    @classmethod
    def from_value(cls, data: Any) -> "WorkflowConfig":
        """Parse the optional top-level `workflow` config value."""
        if data is None:
            return cls()
        if isinstance(data, str):
            return cls(mode=_normalize_workflow_mode(data))
        mapping = _require_mapping(data, "workflow")
        return cls(mode=_normalize_workflow_mode(str(mapping.get("mode", "auto"))))


@dataclass(frozen=True)
class PESMakerConfig:
    """Top-level validated PESMaker configuration.

    Attributes:
        project: Project name used in reports and default output paths.
        structures: Structure files used by `generate`. Later stages can omit
            this and read existing manifests or generated structure folders.
        generation: Supercell and perturbation settings.
        sampling: Optional sampling engine configuration.
        labeling: DFT labeling engine configuration.
        collecting: Labeled dataset collection configuration.
        dataset: Dataset export configuration.
        training: Potential training engine configuration.
        jobs: Cluster submission and machine-specific template options.
        workflow: Optional advanced smart-next override.
        sampling_configured: Whether the user wrote a `sampling` section.
        labeling_configured: Whether the user wrote a `labeling` section.
        collecting_configured: Whether the user wrote a `collecting` section.
        training_configured: Whether the user wrote a `training` section.
    """

    project: str
    structures: tuple[StructureInput, ...] = ()
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    sampling: EngineConfig = field(
        default_factory=lambda: EngineConfig(engine="none", options={})
    )
    labeling: EngineConfig = field(
        default_factory=lambda: EngineConfig(engine="vasp", options={})
    )
    collecting: EngineConfig = field(
        default_factory=lambda: EngineConfig(engine="vasp", options={})
    )
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: EngineConfig = field(
        default_factory=lambda: EngineConfig(engine="nep", options={})
    )
    jobs: EngineConfig = field(
        default_factory=lambda: EngineConfig(engine="local", options={})
    )
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    sampling_configured: bool = False
    labeling_configured: bool = False
    collecting_configured: bool = False
    training_configured: bool = False

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "PESMakerConfig":
        """Create a validated config object from a raw mapping.

        Args:
            data: Raw top-level configuration mapping.

        Returns:
            Validated PESMaker configuration.

        Raises:
            ValueError: If required top-level fields are missing or malformed.
        """
        project = data.get("project")
        if not project:
            raise ValueError("config requires 'project'")

        structures = _parse_structures(data.get("structures"), required=False)
        sampling = _optional_alias_mapping(
            data,
            "sampling",
            aliases=("MD_sampling", "md_sampling"),
        )
        labeling = _optional_mapping(data.get("labeling"), "labeling")
        collecting = _optional_mapping(data.get("collecting"), "collecting")
        training = _optional_mapping(data.get("training"), "training")

        return cls(
            project=str(project),
            structures=structures,
            generation=GenerationConfig.from_mapping(
                _optional_mapping(data.get("generation"), "generation")
            ),
            sampling=EngineConfig.from_mapping(
                sampling,
                default_engine="none",
            ),
            labeling=EngineConfig.from_mapping(
                labeling,
                default_engine="vasp",
            ),
            collecting=EngineConfig.from_mapping(
                collecting,
                default_engine="vasp",
            ),
            dataset=DatasetConfig.from_mapping(
                _optional_mapping(data.get("dataset"), "dataset")
            ),
            training=EngineConfig.from_mapping(
                training,
                default_engine="nep",
                alias_engine_key="model",
            ),
            jobs=EngineConfig.from_mapping(
                _optional_mapping(data.get("jobs"), "jobs"),
                default_engine="local",
                alias_engine_key="machine",
            ),
            workflow=WorkflowConfig.from_value(data.get("workflow")),
            sampling_configured=sampling is not None,
            labeling_configured=labeling is not None,
            collecting_configured=collecting is not None,
            training_configured=training is not None,
        )


def _optional_mapping(value: Any, name: str) -> dict[str, Any] | None:
    """Return `None` for missing sections and validate provided mappings.

    Args:
        value: Raw value from a config section.
        name: Section name used in error messages.

    Returns:
        `None` for omitted sections, otherwise the original mapping.

    Raises:
        ValueError: If `value` is present but is not a mapping.
    """
    if value is None:
        return None
    return _require_mapping(value, name)


def _optional_alias_mapping(
    data: dict[str, Any],
    canonical: str,
    *,
    aliases: tuple[str, ...],
) -> dict[str, Any] | None:
    """Return a config section while accepting a small set of aliases."""
    keys = (canonical, *aliases)
    present = [key for key in keys if data.get(key) is not None]
    if not present:
        return None
    if len(present) > 1:
        names = ", ".join(keys)
        raise ValueError(f"use only one of these config sections: {names}")
    key = present[0]
    return _optional_mapping(data.get(key), key)


def _nested_generation_options(
    generation: dict[str, Any],
    surface: dict[str, Any],
    section: str,
    inline_keys: set[str],
) -> dict[str, Any]:
    """Collect generation options from top-level and surface-nested forms."""
    options = dict(generation.get(section, {}) or {})
    nested = surface.get(section)
    if nested:
        options.update(_require_mapping(nested, f"generation.surface.{section}"))
    inline = {key: surface[key] for key in inline_keys if key in surface}
    if inline:
        options.update(inline)
    return options


def _parse_generation_tasks(data: dict[str, Any]) -> tuple[GenerationTask, ...]:
    """Parse generation recipes from the legacy single form or `tasks` list."""
    raw_tasks = data.get("tasks")
    if raw_tasks is None:
        return (_parse_generation_task(data, default_name="default"),)
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("generation.tasks must be a non-empty list")

    tasks = []
    for index, task_data in enumerate(raw_tasks):
        task = _require_mapping(task_data, "generation.tasks entry")
        inherited = {key: value for key, value in data.items() if key != "tasks"}
        inherited.update(task)
        tasks.append(
            _parse_generation_task(
                inherited,
                default_name=f"task_{index + 1:02d}",
            )
        )
    return tuple(tasks)


def _parse_generation_task(
    data: dict[str, Any],
    *,
    default_name: str,
) -> GenerationTask:
    supercell = data.get("supercell", [1, 1, 1])
    if len(supercell) != 3:
        raise ValueError("generation.supercell must contain three integers")
    surface = dict(data.get("surface", {}))
    defects = _nested_generation_options(
        data,
        surface,
        "defects",
        {
            "include_pristine",
            "mode",
            "seed",
            "single_vacancies",
            "double_vacancies",
            "line_defects",
        },
    )
    defects_perturb = defects.pop("perturb", None)
    perturb = _nested_generation_options(
        data,
        surface,
        "perturb",
        {
            "pert_num",
            "cell_pert_fraction",
            "atom_pert_distance",
            "atom_pert_style",
            "atom_pert_prob",
            "seed",
            "format",
        },
    )
    if defects_perturb:
        perturb.update(_require_mapping(defects_perturb, "generation.defects.perturb"))
    name = str(data.get("name", default_name))
    return GenerationTask(
        name=_safe_generation_task_name(name),
        supercell=tuple(int(value) for value in supercell),
        surface=surface,
        defects=defects,
        perturb=perturb,
    )


def _safe_generation_task_name(name: str) -> str:
    safe = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_" for char in name
    )
    safe = safe.strip("_")
    return safe or "task"


def _normalize_workflow_mode(value: str) -> str:
    mode = value.strip().lower().replace("_", "-")
    if mode not in WORKFLOW_MODES:
        allowed = ", ".join(sorted(WORKFLOW_MODES))
        raise ValueError(f"workflow.mode must be one of: {allowed}")
    return mode


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    """Validate that a config section is a mapping.

    Args:
        value: Raw value to validate.
        name: Human-readable name used in error messages.

    Returns:
        The original mapping.

    Raises:
        ValueError: If `value` is not a dictionary.
    """
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _parse_structures(
    value: Any,
    *,
    required: bool = True,
) -> tuple[StructureInput, ...]:
    """Parse structure inputs from a simple list or an `include` glob map.

    Args:
        value: Either a non-empty list of paths/mappings or a mapping with
            `include` patterns.

    Returns:
        Tuple of parsed structure inputs, sorted by glob expansion order for
        `include` patterns.

    Raises:
        ValueError: If required structures are not provided, an include pattern
            is empty, or an include pattern matches no files.
    """
    if value is None:
        if required:
            raise ValueError(
                "config requires 'structures' as a non-empty list or include map"
            )
        return ()

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
