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

"""Common manifest, input-discovery, and stage path helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pesmaker.config.schema import PESMakerConfig

STRUCTURE_INPUT_SUFFIXES = {".cif", ".extxyz", ".poscar", ".traj", ".vasp", ".xyz"}

STRUCTURE_INPUT_NAMES = {"CONTCAR", "POSCAR", "XDATCAR"}

MULTI_FRAME_INPUT_SUFFIXES = {".extxyz", ".traj", ".xyz"}

MULTI_FRAME_INPUT_NAMES = {"XDATCAR"}


def _section_output_dir(
    config: PESMakerConfig,
    options: dict[str, Any],
    leaf: str,
) -> Path:
    value = options.get("output_dir")
    return Path(str(value)) if value else Path("runs") / config.project / leaf


def _load_input_records(
    config: PESMakerConfig,
    options: dict[str, Any],
) -> list[dict[str, Any]]:
    manifest = options.get("input_manifest")
    if manifest:
        manifest_path = Path(str(manifest))
        return _mark_input_records(
            _read_manifest(manifest_path),
            input_dir=manifest_path.parent,
            input_mode="input_manifest",
        )
    input_dir = options.get("input_dir")
    if input_dir:
        return _load_input_dir_records(Path(str(input_dir)), input_mode="input_dir")
    generation_dir = _generated_structures_dir(config)
    return _load_input_dir_records(generation_dir, input_mode="generated_dir")


def _load_input_dir_records(
    input_dir: Path, *, input_mode: str
) -> list[dict[str, Any]]:
    if not input_dir.exists():
        raise ValueError(f"input structure directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise ValueError(f"input structure path must be a directory: {input_dir}")

    manifest_path = input_dir / "manifest.jsonl"
    if manifest_path.exists():
        return _mark_input_records(
            _read_manifest(manifest_path),
            input_dir=input_dir,
            input_mode=f"{input_mode}_manifest",
        )

    paths = _discover_input_structure_files(input_dir)
    if not paths:
        raise ValueError(f"no structure files found in {input_dir}")
    records = []
    for path in paths:
        records.extend(
            _scanned_structure_records(
                path,
                input_dir=input_dir,
                input_mode=f"{input_mode}_scan",
            )
        )
    return records


def _mark_input_records(
    records: list[dict[str, Any]],
    *,
    input_dir: Path,
    input_mode: str,
) -> list[dict[str, Any]]:
    marked = []
    for source_index, record in enumerate(records):
        path = Path(str(record["path"]))
        if not path.is_absolute() and not path.exists():
            candidate = input_dir / path
            if candidate.exists():
                path = candidate
        marked_record = {
            **record,
            "path": str(path),
            "input_dir": str(input_dir),
            "input_mode": input_mode,
            "source_record_index": record.get("index", source_index),
        }
        try:
            marked_record["input_relative_path"] = path.relative_to(
                input_dir
            ).as_posix()
        except ValueError:
            pass
        marked.append(marked_record)
    return marked


def _discover_input_structure_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and _is_input_structure_file(path)
    )


def _is_input_structure_file(path: Path) -> bool:
    if path.name.upper() in STRUCTURE_INPUT_NAMES:
        return True
    return path.suffix.lower() in STRUCTURE_INPUT_SUFFIXES


def _scanned_structure_records(
    path: Path,
    *,
    input_dir: Path,
    input_mode: str,
) -> list[dict[str, Any]]:
    record = {
        "path": str(path),
        "input_dir": str(input_dir),
        "input_mode": input_mode,
        "input_relative_path": path.relative_to(input_dir).as_posix(),
    }
    frame_count = _scanned_structure_frame_count(path)
    if frame_count <= 1:
        return [record]
    return [
        {
            **record,
            "frame_index": frame_index,
            "source_frame": frame_index,
        }
        for frame_index in range(frame_count)
    ]


def _scanned_structure_frame_count(path: Path) -> int:
    if not _is_multi_frame_input_file(path):
        return 1
    try:
        from ase.io import read
    except ImportError as exc:
        raise RuntimeError("Reading multi-frame structure files requires ASE") from exc

    frames = read(path, index=":")
    if isinstance(frames, list):
        return len(frames)
    return 1


def _is_multi_frame_input_file(path: Path) -> bool:
    if path.name.upper() in MULTI_FRAME_INPUT_NAMES:
        return True
    return path.suffix.lower() in MULTI_FRAME_INPUT_SUFFIXES


def _generated_structures_dir(config: PESMakerConfig) -> Path:
    if config.generation.output_dir:
        return config.generation.output_dir
    local_generated = Path("generated")
    if local_generated.exists():
        return local_generated
    return Path("runs") / config.project / "generated"


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if "path" in record:
                record = {"path": str(record["path"]), **record}
            records.append(record)
    return records


def _read_optional_file(value: Any, *, default: str) -> str:
    if value:
        return Path(str(value)).read_text(encoding="utf-8")
    return default
