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
"""Structure generation workflow for supercells and perturbations."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from pesmaker.config.schema import GenerationTask, PESMakerConfig
from pesmaker.structures import (
    PerturbationSettings,
    apply_surface_settings,
    generate_defect_variants,
    load_structure,
    make_supercell,
    perturb_structures,
    write_structure,
)


@dataclass(frozen=True)
class GeneratedStructure:
    """Metadata for one generated structure file.

    Attributes:
        source: Original input structure path.
        path: Generated structure file path.
        task: Generation task name.
        supercell: Three integer expansion factors used for this structure.
        variant: Structural variant name, such as `pristine` or a defect name.
        variant_description: Human-readable variant description.
        generation_type: Concise origin tag used as the output filename prefix.
        index: Zero-based index within the source structure's output folder.
        atom_count: Number of atoms in the generated structure.
    """

    source: Path
    path: Path
    index: int
    atom_count: int
    task: str = "default"
    supercell: tuple[int, int, int] = (1, 1, 1)
    variant: str = "pristine"
    variant_description: str = "pristine"
    generation_type: str = "perturb"


@dataclass(frozen=True)
class GenerateResult:
    """Summary returned after structure generation completes.

    Attributes:
        output_dir: Root directory containing generated structures.
        structures: Metadata for every generated structure.
        summary_path: Human-readable generation summary path.
    """

    output_dir: Path
    structures: tuple[GeneratedStructure, ...]
    summary_path: Path


def generate_structures(config: PESMakerConfig) -> GenerateResult:
    """Generate perturbed structures from every configured input structure.

    Args:
        config: Validated PESMaker configuration.

    Returns:
        Generation result containing the output directory and generated-file
        metadata.

    Raises:
        RuntimeError: If ASE is unavailable for structure IO.
        ValueError: If output format or perturbation settings are invalid.
        OSError: If output folders or files cannot be written.
    """
    if not config.structures:
        raise ValueError(
            "generate requires 'structures' as a non-empty list or include map"
        )

    output_dir = (
        config.generation.output_dir or Path("runs") / config.project / "generated"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    generated: list[GeneratedStructure] = []
    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for task in config.generation.tasks:
            generated.extend(
                _generate_task_structures(
                    config,
                    task,
                    output_dir,
                    manifest,
                    use_task_dir=len(config.generation.tasks) > 1,
                )
            )

    result = GenerateResult(
        output_dir=output_dir,
        structures=tuple(generated),
        summary_path=output_dir / "generation_summary.txt",
    )
    result.summary_path.write_text(format_generate_summary(result), encoding="utf-8")
    return result


def _generate_task_structures(
    config: PESMakerConfig,
    task: GenerationTask,
    output_dir: Path,
    manifest,
    *,
    use_task_dir: bool,
) -> list[GeneratedStructure]:
    settings = PerturbationSettings.from_mapping(task.perturb)
    output_format = str(task.perturb.get("format", "vasp")).lower()
    ase_format, suffix = _resolve_output_format(output_format)

    generated: list[GeneratedStructure] = []
    task_output_dir = output_dir / task.name if use_task_dir else output_dir
    structure_dirs = _structure_output_dirs(config, task_output_dir)
    for input_index, structure in enumerate(config.structures):
        atoms = load_structure(structure.path)
        supercell_atoms = make_supercell(atoms, task.supercell)
        base_atoms = apply_surface_settings(supercell_atoms, task.surface)
        variants = generate_defect_variants(base_atoms, task.defects)
        structure_dir = structure_dirs[input_index]
        structure_dir.mkdir(parents=True, exist_ok=True)
        use_variant_dirs = bool(task.defects) or use_task_dir
        for variant in variants:
            variant_dir = (
                structure_dir / variant.name if use_variant_dirs else structure_dir
            )
            variant_dir.mkdir(parents=True, exist_ok=True)
            generation_type = _generation_type(task, variant.name)
            for structure_index, perturbed in enumerate(
                perturb_structures(variant.atoms, settings)
            ):
                output_path = (
                    variant_dir / f"{generation_type}_{structure_index:06d}.{suffix}"
                )
                write_structure(perturbed, output_path, fmt=ase_format)
                item = GeneratedStructure(
                    source=structure.path,
                    path=output_path,
                    task=task.name,
                    supercell=task.supercell,
                    variant=variant.name,
                    variant_description=variant.description,
                    generation_type=generation_type,
                    index=structure_index,
                    atom_count=len(perturbed),
                )
                generated.append(item)
                manifest.write(json.dumps(_manifest_record(item)) + "\n")
    return generated


def _structure_output_dirs(
    config: PESMakerConfig,
    output_dir: Path,
) -> tuple[Path, ...]:
    """Build one unique output directory for each input structure.

    Args:
        config: Validated PESMaker configuration.
        output_dir: Root generation output directory.

    Returns:
        Tuple of output directories aligned with `config.structures`. Duplicate
        structure stems receive numeric suffixes such as `_2`.
    """
    seen: dict[str, int] = {}
    paths: list[Path] = []
    for structure in config.structures:
        stem = structure.path.stem
        seen[stem] = seen.get(stem, 0) + 1
        folder = stem if seen[stem] == 1 else f"{stem}_{seen[stem]}"
        paths.append(output_dir / folder)
    return tuple(paths)


def _generation_type(task: GenerationTask, variant: str) -> str:
    """Return the concise origin tag used as the generated filename prefix."""
    if variant != "pristine":
        return "defect"
    if _has_surface_generation(task.surface):
        return "surface"
    return "perturb"


def _has_surface_generation(surface: dict) -> bool:
    """Return whether current surface settings actually change the structure."""
    return "vacuum" in surface


def _resolve_output_format(name: str) -> tuple[str, str]:
    """Map a user-facing output format to ASE format and file suffix.

    Args:
        name: User-facing format name from `generation.perturb.format`.

    Returns:
        Pair of ASE writer format and file suffix.

    Raises:
        ValueError: If `name` is not a supported output format.
    """
    if name in {"vasp", "poscar"}:
        return "vasp", "vasp"
    if name in {"extxyz", "xyz"}:
        return "extxyz", "xyz"
    raise ValueError(f"unsupported generation output format: {name}")


def _manifest_record(item: GeneratedStructure) -> dict[str, str | int | list[int]]:
    """Serialize generated-structure metadata for manifest.jsonl.

    Args:
        item: Generated-structure metadata object.

    Returns:
        JSON-serializable dictionary written as one manifest line.
    """
    return {
        "index": item.index,
        "task": item.task,
        "source": str(item.source),
        "path": str(item.path),
        "supercell": list(item.supercell),
        "variant": item.variant,
        "variant_description": item.variant_description,
        "generation_type": item.generation_type,
        "atom_count": item.atom_count,
    }


def format_generate_summary(result: GenerateResult) -> str:
    """Render a grouped, human-readable generation summary."""
    task_counts: dict[str, int] = defaultdict(int)
    task_supercells: dict[str, tuple[int, int, int]] = {}
    source_counts: dict[tuple[str, Path], int] = defaultdict(int)
    source_type_counts: dict[tuple[str, Path, str], int] = defaultdict(int)
    variant_counts: dict[tuple[str, Path, str, str, Path], int] = defaultdict(int)
    for structure in result.structures:
        task_counts[structure.task] += 1
        task_supercells[structure.task] = structure.supercell
        source_counts[(structure.task, structure.source)] += 1
        source_type_counts[
            (structure.task, structure.source, structure.generation_type)
        ] += 1
        variant_counts[
            (
                structure.task,
                structure.source,
                structure.generation_type,
                structure.variant,
                structure.path.parent,
            )
        ] += 1

    lines = [
        "Perturbation generation complete.",
        f"Generated structures : {len(result.structures)}",
        f"Output directory     : {result.output_dir}",
        f"Manifest             : {result.output_dir / 'manifest.jsonl'}",
        f"Summary              : {result.summary_path}",
        "Generation tasks:",
    ]
    for task, task_count in task_counts.items():
        supercell = task_supercells[task]
        lines.append(f"  - {task}: {task_count} structure(s), supercell={supercell}")
        task_sources = [
            (source, count)
            for (source_task, source), count in source_counts.items()
            if source_task == task
        ]
        for source, source_count in task_sources:
            type_summary = _generation_type_summary(
                source_type_counts,
                task=task,
                source=source,
            )
            lines.append(f"      {source}: {source_count} {type_summary} structure(s)")
            task_variants = [
                (generation_type, variant, folder, count)
                for (
                    variant_task,
                    variant_source,
                    generation_type,
                    variant,
                    folder,
                ), count in (
                    variant_counts.items()
                )
                if variant_task == task and variant_source == source
            ]
            for generation_type, variant, folder, variant_count in task_variants[:8]:
                label = _summary_variant_label(generation_type, variant)
                lines.append(f"        - {label} -> {folder} ({variant_count})")
            omitted = len(task_variants) - 8
            if omitted > 0:
                lines.append(f"        - ... {omitted} more variant folder(s)")
    return "\n".join(lines) + "\n"


def _generation_type_summary(
    counts: dict[tuple[str, Path, str], int],
    *,
    task: str,
    source: Path,
) -> str:
    type_counts = [
        (generation_type, count)
        for (count_task, count_source, generation_type), count in counts.items()
        if count_task == task and count_source == source
    ]
    if len(type_counts) == 1:
        return type_counts[0][0]
    return ", ".join(
        f"{generation_type}={count}" for generation_type, count in type_counts
    )


def _summary_variant_label(generation_type: str, variant: str) -> str:
    if variant == "pristine":
        return generation_type
    return f"{generation_type}:{variant}"
