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

"""Collect VASP OUTCAR files into labeled extxyz train/test datasets."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from glob import glob
from pathlib import Path
import random
import re
from typing import Any

import numpy as np

from pesmaker.config.schema import PESMakerConfig
from pesmaker.jobs.submit import (
    VASP_COMPLETION_MARKER,
    VASP_SCF_NOT_CONVERGED_MARKER,
)
from pesmaker.results import StageResult


@dataclass(frozen=True)
class LabeledFrame:
    """One labeled training frame parsed from a VASP OUTCAR."""

    outcar: Path
    natoms: int
    atom_symbols: list[str]
    cell: np.ndarray
    energy: float
    positions_forces: np.ndarray
    virial: np.ndarray | None
    virial_offset: int | None
    config_type: str
    weight: float


def collect_labeled_dataset(config: PESMakerConfig) -> StageResult:
    """Collect completed VASP SCF calculations into `train.xyz`."""
    options = config.collecting.options
    output_path = Path(str(options.get("dataset_path", "train.xyz")))
    output_dir = output_path.parent
    outputs = _matched_outcars(options)
    if not outputs:
        raise ValueError(
            "no VASP outputs matched pattern(s): "
            f"{', '.join(_outcar_patterns(options))}"
        )

    frames: list[LabeledFrame] = []
    collection_records: list[tuple[Path, int]] = []
    incomplete_records: list[Path] = []
    nonconverged_records: list[Path] = []
    unreadable_records: list[Path] = []
    warnings: list[str] = []
    check_scf_convergence = _check_scf_convergence(config)
    check_vasp_completion = _check_vasp_completion(config)
    for output in outputs:
        if check_scf_convergence and _outcar_is_nonconverged(output):
            nonconverged_records.append(output)
            warnings.append(f"Skipped nonconverged VASP OUTCAR: {output}")
            continue
        if check_vasp_completion and not _outcar_is_complete(output):
            incomplete_records.append(output)
            warnings.append(f"Skipped incomplete VASP OUTCAR: {output}")
            continue
        try:
            frame = _read_vasp_labeled_frame(output, options)
        except ValueError as exc:
            unreadable_records.append(output)
            warnings.append(f"Skipped unreadable VASP OUTCAR: {output} ({exc})")
            continue
        frames.append(frame)
        collection_records.append((output, 1))
    if not frames:
        raise ValueError("no converged VASP frames were available for collection")

    train_frames, test_frames = _split_test_frames(frames, options)
    _write_labeled_xyz(output_path, train_frames, options)
    files = [output_path]
    test_path = _test_dataset_path(options)
    if test_frames:
        _write_labeled_xyz(test_path, test_frames, options)
        files.append(test_path)

    summary_path = Path(
        str(
            options.get(
                "summary_path",
                output_path.with_name(f"{output_path.stem}_collection_summary.txt"),
            )
        )
    )
    _write_collection_summary(
        summary_path,
        collected_records=collection_records,
        incomplete_records=incomplete_records,
        nonconverged_records=nonconverged_records,
        unreadable_records=unreadable_records,
        matched_count=len(outputs),
    )
    files.append(summary_path)
    return StageResult(
        output_dir,
        tuple(files),
        _collection_message(
            train_path=output_path,
            train_count=len(train_frames),
            test_path=test_path,
            test_count=len(test_frames),
            summary_path=summary_path,
            source_counts=_source_counts(collection_records),
            matched_count=len(outputs),
            collected_count=len(collection_records),
            incomplete_count=len(incomplete_records),
            nonconverged_count=len(nonconverged_records),
            unreadable_count=len(unreadable_records),
            virial_offset_counts=_virial_offset_counts(frames),
        ),
        warnings=tuple(warnings),
    )


def _matched_outcars(options: dict[str, Any]) -> list[Path]:
    """Return sorted unique OUTCAR paths from one or more glob patterns."""
    outputs: list[Path] = []
    seen = set()
    for pattern in _outcar_patterns(options):
        for match in sorted(glob(pattern, recursive=True)):
            path = Path(match)
            if path.name != "OUTCAR":
                continue
            key = str(path)
            if key not in seen:
                outputs.append(path)
                seen.add(key)
    return outputs


def _outcar_patterns(options: dict[str, Any]) -> list[str]:
    """Read optional OUTCAR globs, defaulting to every OUTCAR below cwd."""
    if "outcar_patterns" in options:
        patterns = options["outcar_patterns"]
        if not isinstance(patterns, list) or not patterns:
            raise ValueError("labeling.outcar_patterns must be a non-empty list")
        return [str(pattern) for pattern in patterns]
    return [str(options.get("outcar_pattern", "**/OUTCAR"))]


def _read_vasp_labeled_frame(path: Path, options: dict[str, Any]) -> LabeledFrame:
    """Parse one VASP OUTCAR with vasp2nep-style labeled-data logic."""
    lines = path.read_text(errors="replace").splitlines()
    atom_names, atom_nums = _parse_system_info(lines)
    atom_symbols = [
        symbol for symbol, count in zip(atom_names, atom_nums) for _ in range(count)
    ]
    natoms = sum(atom_nums)
    cell = _parse_cell(lines)
    positions_forces = _parse_positions_forces(lines, natoms)
    energy = _parse_energy(lines)
    virial = None
    virial_offset = None
    if _include_virial(options):
        virial, virial_offset = _parse_virial(lines)
    return LabeledFrame(
        outcar=path,
        natoms=natoms,
        atom_symbols=atom_symbols,
        cell=cell,
        energy=energy,
        positions_forces=positions_forces,
        virial=virial,
        virial_offset=virial_offset,
        config_type=_config_type(path, options),
        weight=_weight_value(options),
    )


def _parse_system_info(lines: list[str]) -> tuple[list[str], list[int]]:
    atom_names = []
    atom_nums = None
    for line in lines:
        if "TITEL" in line:
            parts = line.split()
            if len(parts) >= 4:
                symbol = parts[3].split("_", 1)[0]
                atom_names.append(symbol)
        if "ions per type" in line:
            atom_nums = [int(value) for value in line.split()[4:]]
    if atom_nums is None:
        raise ValueError("could not find 'ions per type'")
    atom_names = atom_names[: len(atom_nums)]
    if len(atom_names) != len(atom_nums):
        raise ValueError("could not match POTCAR TITEL entries to atom counts")
    return atom_names, atom_nums


def _parse_cell(lines: list[str]) -> np.ndarray:
    token = "VOLUME and BASIS-vectors are now"
    for index, line in enumerate(lines):
        if token in line:
            return _parse_cell_rows(lines, index + 5)
    token = "direct lattice vectors"
    for index, line in enumerate(lines):
        if token in line:
            return _parse_cell_rows(lines, index + 1)
    raise ValueError("could not find cell vectors")


def _parse_cell_rows(lines: list[str], start: int) -> np.ndarray:
    cell = []
    for offset in range(3):
        parts = _float_tokens(lines[start + offset])
        if len(parts) < 3:
            raise ValueError("could not parse cell vector")
        cell.append(parts[:3])
    return np.array(cell)


def _parse_positions_forces(lines: list[str], natoms: int) -> np.ndarray:
    token = "TOTAL-FORCE (eV/Angst)"
    positions_forces = None
    for index, line in enumerate(lines):
        if token not in line:
            continue
        values = []
        for offset in range(natoms):
            parts = _float_tokens(lines[index + 2 + offset])
            if len(parts) < 6:
                raise ValueError("could not parse position/force line")
            values.append(parts[:6])
        positions_forces = np.array(values)
    if positions_forces is None:
        raise ValueError("could not find TOTAL-FORCE block")
    return positions_forces


def _parse_energy(lines: list[str]) -> float:
    energy = None
    for line in lines:
        if "free  energy   TOTEN" in line:
            parts = line.split()
            if len(parts) >= 5:
                energy = float(parts[4])
    if energy is None:
        raise ValueError("could not find TOTEN energy")
    return energy


def _parse_virial(lines: list[str]) -> tuple[np.ndarray, int]:
    token = "FORCE on cell =-STRESS"
    for index, line in enumerate(lines):
        if token not in line:
            continue
        offset, values = _find_virial_values(lines, index)
        return _vasp_virial_to_matrix(values), offset
    raise ValueError("could not find FORCE on cell =-STRESS virial block")


def _find_virial_values(lines: list[str], start: int) -> tuple[int, np.ndarray]:
    """Find VASP's six virial values after a FORCE-on-cell block."""
    candidates: list[tuple[int, np.ndarray]] = []
    for offset in range(1, 25):
        line_index = start + offset
        if line_index >= len(lines):
            break
        parts = _float_tokens(lines[line_index])
        if len(parts) >= 6:
            candidates.append((offset, np.array(parts[:6])))

    for preferred_offset in (13, 14):
        for offset, values in candidates:
            if offset == preferred_offset:
                return offset, values
    if candidates:
        return candidates[0]
    raise ValueError("could not parse VASP virial line")


def _vasp_virial_to_matrix(values: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [values[0], values[3], values[5]],
            [values[3], values[1], values[4]],
            [values[5], values[4], values[2]],
        ]
    )


def _float_tokens(line: str) -> list[float]:
    text = line.replace("-", " -")
    values = []
    for token in text.split():
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


def _write_labeled_xyz(
    path: Path,
    frames: list[LabeledFrame],
    options: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    include_virial = _include_virial(options)
    include_config_type = _include_config_type(options)
    include_weight = _include_weight(options)
    with path.open("w", encoding="utf-8") as handle:
        for frame in frames:
            handle.write(f"{frame.natoms}\n")
            comment = _labeled_comment_line(
                frame,
                include_virial=include_virial,
                include_config_type=include_config_type,
                include_weight=include_weight,
            )
            handle.write(f"{comment}\n")
            for symbol, row in zip(frame.atom_symbols, frame.positions_forces):
                handle.write(
                    f"{symbol} "
                    f"{row[0]:15.8f} {row[1]:15.8f} {row[2]:15.8f} "
                    f"{row[3]:15.8f} {row[4]:15.8f} {row[5]:15.8f}\n"
                )


def _labeled_comment_line(
    frame: LabeledFrame,
    *,
    include_virial: bool,
    include_config_type: bool,
    include_weight: bool,
) -> str:
    lattice = " ".join(f"{value:10.9f}" for value in frame.cell.reshape(-1))
    parts = [
        f'Lattice="{lattice}"',
        f"Energy={frame.energy:10.9f}",
        "Properties=species:S:1:pos:R:3:force:R:3",
    ]
    if include_virial:
        if frame.virial is None:
            raise ValueError(f"missing virial for frame: {frame.outcar}")
        virial = " ".join(f"{value:10.8f}" for value in frame.virial.reshape(-1))
        parts.append(f'Virial="{virial}"')
    parts.append('pbc="T T T"')
    if include_config_type:
        parts.append(f"Config_type={frame.config_type}")
    if include_weight:
        parts.append(f"weight={frame.weight}")
    return " ".join(parts)


def _check_scf_convergence(config: PESMakerConfig) -> bool:
    """Return whether nonconverged VASP OUTCAR files should be skipped."""
    value = config.collecting.options.get(
        "check_scf_convergence",
        config.jobs.options.get("check_scf_convergence", True),
    )
    if not isinstance(value, bool):
        raise ValueError("collecting.check_scf_convergence must be true or false")
    return value


def _check_vasp_completion(config: PESMakerConfig) -> bool:
    """Return whether VASP normal termination is required for collection."""
    value = config.collecting.options.get("check_vasp_completion", True)
    if not isinstance(value, bool):
        raise ValueError("collecting.check_vasp_completion must be true or false")
    return value


def _include_virial(options: dict[str, Any]) -> bool:
    value = options.get("include_virial", True)
    if not isinstance(value, bool):
        raise ValueError("collecting.include_virial must be true or false")
    return value


def _include_config_type(options: dict[str, Any]) -> bool:
    value = options.get("config_type", True)
    if not isinstance(value, bool):
        raise ValueError("collecting.config_type must be true or false")
    return value


def _include_weight(options: dict[str, Any]) -> bool:
    value = options.get("include_weight", False)
    if not isinstance(value, bool):
        raise ValueError("collecting.include_weight must be true or false")
    return value


def _weight_value(options: dict[str, Any]) -> float:
    return float(options.get("weight_value", 1.0))


def _outcar_is_nonconverged(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return any(VASP_SCF_NOT_CONVERGED_MARKER in line for line in handle)
    except OSError:
        return True


def _outcar_is_complete(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return any(VASP_COMPLETION_MARKER in line for line in handle)
    except OSError:
        return False


def _config_type(path: Path, options: dict[str, Any]) -> str:
    root = _collection_root(options)
    try:
        relative_parts = path.parent.relative_to(root).parts
    except ValueError:
        relative_parts = path.parent.parts
    semantic_parts = [
        part for part in relative_parts if _is_config_type_part(part)
    ]
    if not semantic_parts:
        semantic_parts = [path.parent.name]
    return _safe_config_type("_".join(semantic_parts))


def _is_config_type_part(part: str) -> bool:
    name = part.strip()
    lowered = name.lower()
    if not name:
        return False
    if "run_vasp_scf" in lowered:
        return False
    if lowered in {"labeling", "vasp", "scf"}:
        return False
    if re.fullmatch(r"(calc|job|task|frame|structure|selected)[_-]?\d+", lowered):
        return False
    return True


def _safe_config_type(value: str) -> str:
    safe = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value
    )
    safe = safe.strip("_")
    return safe or "unknown"


def _collection_root(options: dict[str, Any]) -> Path:
    return Path(str(options.get("collection_root", "."))).resolve()


def _split_test_frames(
    frames: list[LabeledFrame],
    options: dict[str, Any],
) -> tuple[list[LabeledFrame], list[LabeledFrame]]:
    test_count = _test_data_frames(options)
    if test_count == 0:
        return frames, []
    if test_count >= len(frames):
        raise ValueError(
            "collecting.test_data_frames must be smaller than the collected frame count"
        )
    seed = int(options.get("test_seed", 1000))
    test_indices = set(random.Random(seed).sample(range(len(frames)), test_count))
    train_frames = [
        frame for index, frame in enumerate(frames) if index not in test_indices
    ]
    test_frames = [frame for index, frame in enumerate(frames) if index in test_indices]
    return train_frames, test_frames


def _test_data_frames(options: dict[str, Any]) -> int:
    value = options.get("test_data_frames", 0)
    if not isinstance(value, int):
        raise ValueError("collecting.test_data_frames must be an integer")
    if value < 0:
        raise ValueError(
            "collecting.test_data_frames must be greater than or equal to 0"
        )
    return value


def _test_dataset_path(options: dict[str, Any]) -> Path:
    return Path(
        str(options.get("test_path", options.get("test_dataset_path", "test.xyz")))
    )


def _collection_message(
    *,
    train_path: Path,
    train_count: int,
    test_path: Path,
    test_count: int,
    summary_path: Path,
    source_counts: dict[str, dict[str, int]],
    matched_count: int,
    collected_count: int,
    incomplete_count: int,
    nonconverged_count: int,
    unreadable_count: int,
    virial_offset_counts: dict[int, int],
) -> str:
    total_count = train_count + test_count
    lines = [
        "Labeled dataset collection complete.",
        "",
        "Totals:",
        f"  OUTCAR matched       : {matched_count}",
        f"  OUTCAR collected     : {collected_count}",
        f"  Structures written   : {total_count}",
        f"  Incomplete skipped   : {incomplete_count}",
        f"  Nonconverged skipped : {nonconverged_count}",
        f"  Unreadable skipped   : {unreadable_count}",
        "",
        "Datasets:",
        f"  Train : {train_path} ({train_count} structures)",
    ]
    if test_count:
        lines.append(f"  Test  : {test_path} ({test_count} structures)")
    else:
        lines.append("  Test  : not written (test_data_frames = 0)")
    lines.extend(
        [
            "",
            f"Summary file : {summary_path}",
            "",
            "Sources:",
            *_format_source_overview(source_counts),
        ]
    )
    if virial_offset_counts:
        lines.append("")
        lines.append(_vdw_summary_line(virial_offset_counts))
    return "\n".join(lines)


def _virial_offset_counts(frames: list[LabeledFrame]) -> dict[int, int]:
    return dict(
        Counter(
            frame.virial_offset
            for frame in frames
            if frame.virial_offset is not None
        )
    )


def _vdw_summary_line(counts: dict[int, int]) -> str:
    """Summarize whether OUTCAR virial blocks look VDW/MBD-adjusted."""
    total = sum(counts.values())
    standard = counts.get(13, 0)
    vdw = counts.get(14, 0)
    unknown = total - standard - vdw
    if total == 0:
        return "Van der Waals correction : not checked (no virial blocks parsed)"
    if vdw == total:
        return (
            "Van der Waals correction : detected "
            f"in {vdw}/{total} collected calculation(s)"
        )
    if standard == total:
        return (
            "Van der Waals correction : not detected "
            f"in {standard}/{total} collected calculation(s)"
        )
    if unknown:
        return (
            "Van der Waals correction : uncertain "
            f"({vdw}/{total} detected, {standard}/{total} not detected, "
            f"{unknown}/{total} unclear; check INCAR consistency)"
        )
    return (
        "Van der Waals correction : mixed "
        f"({vdw}/{total} detected, {standard}/{total} not detected; "
        "check INCAR consistency)"
    )


def _write_collection_summary(
    path: Path,
    *,
    collected_records: list[tuple[Path, int]],
    incomplete_records: list[Path],
    nonconverged_records: list[Path],
    unreadable_records: list[Path],
    matched_count: int,
) -> None:
    """Write a human-readable collection report."""
    collected_grouped = _group_collected_records(collected_records)
    incomplete_grouped = _group_path_records(incomplete_records)
    nonconverged_grouped = _group_path_records(nonconverged_records)
    unreadable_grouped = _group_path_records(unreadable_records)
    collected_count = sum(counts["outcars"] for counts in collected_grouped.values())
    frame_count = sum(counts["frames"] for counts in collected_grouped.values())

    lines = [
        "PESMaker collection summary",
        "",
        "Totals",
        f"  OUTCAR files matched          : {matched_count}",
        f"  OUTCAR files collected        : {collected_count}",
        f"  Structures written            : {frame_count}",
        f"  Incomplete OUTCAR skipped     : {len(incomplete_records)}",
        f"  Nonconverged OUTCAR skipped   : {len(nonconverged_records)}",
        f"  Unreadable OUTCAR skipped     : {len(unreadable_records)}",
        "",
        "Collected structures by source",
    ]
    lines.extend(_format_grouped_counts(collected_grouped, include_frames=True))
    if incomplete_records:
        lines.extend(
            [
                "",
                "Incomplete OUTCAR by source",
                *_format_grouped_counts(incomplete_grouped, include_frames=False),
            ]
        )
    if nonconverged_records:
        lines.extend(
            [
                "",
                "Nonconverged OUTCAR by source",
                *_format_grouped_counts(nonconverged_grouped, include_frames=False),
            ]
        )
    if unreadable_records:
        lines.extend(
            [
                "",
                "Unreadable OUTCAR by source",
                *_format_grouped_counts(unreadable_grouped, include_frames=False),
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _group_collected_records(
    records: list[tuple[Path, int]],
) -> dict[str, dict[str, int]]:
    grouped: dict[str, dict[str, int]] = {}
    for outcar, frame_count in records:
        key = _source_label(outcar)
        item = grouped.setdefault(key, {"outcars": 0, "frames": 0})
        item["outcars"] += 1
        item["frames"] += frame_count
    return grouped


def _group_path_records(paths: list[Path]) -> dict[str, dict[str, int]]:
    return _group_collected_records([(path, 0) for path in paths])


def _format_grouped_counts(
    grouped: dict[str, dict[str, int]],
    *,
    include_frames: bool,
) -> list[str]:
    if not grouped:
        return ["  none"]
    name_width = max(len("source"), *(len(name) for name in grouped))
    lines = [f"  {'source'.ljust(name_width)}  OUTCARs  structures"]
    lines.append(f"  {'-' * name_width}  ------  ----------")
    for source, counts in sorted(grouped.items()):
        frame_text = str(counts["frames"]) if include_frames else "-"
        lines.append(
            f"  {source.ljust(name_width)}  {counts['outcars']:>6}  {frame_text:>10}"
        )
    return lines


def _format_source_overview(
    grouped: dict[str, dict[str, int]],
    *,
    limit: int = 20,
) -> list[str]:
    if not grouped:
        return ["  none"]
    sorted_items = sorted(
        grouped.items(),
        key=lambda item: (-item[1]["frames"], item[0]),
    )
    show_all = len(grouped) <= limit
    shown = sorted_items if show_all else sorted_items[:limit]
    name_width = max(len("source"), *(len(name) for name, _ in shown))
    lines = [f"  Source groups : {len(grouped)}"]
    if not show_all:
        lines.append(f"  Showing top {limit} groups by structure count.")
    lines.append(f"  {'source'.ljust(name_width)}  structures")
    lines.append(f"  {'-' * name_width}  ----------")
    for source, counts in shown:
        lines.append(f"  {source.ljust(name_width)}  {counts['frames']:>10}")
    total_structures = sum(counts["frames"] for counts in grouped.values())
    if not show_all:
        remaining = len(grouped) - limit
        lines.append(f"  ... {remaining} more group(s); see summary file.")
    lines.append(f"  Total structures in sources : {total_structures}")
    return lines


def _source_counts(records: list[tuple[Path, int]]) -> dict[str, dict[str, int]]:
    return _group_collected_records(records)


def _source_label(path: Path) -> str:
    sub_yaml_dir = _nearest_sub_yaml_dir(path)
    if sub_yaml_dir is not None:
        return _relative_label(sub_yaml_dir)
    return _fallback_source_label(path)


def _nearest_sub_yaml_dir(path: Path) -> Path | None:
    for directory in (path.parent, *path.parents):
        if (directory / "sub.yaml").exists():
            return directory
    return None


def _fallback_source_label(path: Path) -> str:
    try:
        parts = path.parent.relative_to(_collection_root({})).parts
    except ValueError:
        parts = path.parent.parts
    if not parts:
        return "."
    calculation_index = _first_calculation_dir_index(parts)
    if calculation_index is not None and calculation_index > 0:
        return "/".join(parts[:calculation_index])
    semantic_parts = [part for part in parts if _is_source_group_part(part)]
    if semantic_parts:
        return "/".join(semantic_parts)
    return "/".join(parts)


def _relative_label(path: Path) -> str:
    try:
        return path.relative_to(_collection_root({})).as_posix()
    except ValueError:
        return path.as_posix()


def _first_calculation_dir_index(parts: tuple[str, ...]) -> int | None:
    for index, part in enumerate(parts):
        if _is_calculation_dir(part):
            return index
    return None


def _is_calculation_dir(part: str) -> bool:
    lower = part.lower()
    return (
        lower == "run_vasp_scf"
        or lower.endswith("_run_vasp_scf")
        or lower.startswith("calc_")
    )


def _is_source_group_part(part: str) -> bool:
    return _is_config_type_part(part) and not _is_calculation_dir(part)
