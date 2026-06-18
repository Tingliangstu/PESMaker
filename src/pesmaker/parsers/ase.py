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
"""ASE-backed structure and trajectory parsing helpers."""

from __future__ import annotations

from glob import glob
from pathlib import Path
from typing import Any


def read_frames(pattern: str | Path) -> list[Any]:
    """Read all ASE frames matching a path or glob pattern."""
    groups = read_frame_groups(pattern)
    return [frame for _, frames in groups for frame in frames]


def read_frame_groups(pattern: str | Path) -> list[tuple[Path, list[Any]]]:
    """Read ASE frames grouped by each matched trajectory file."""
    try:
        from ase.io import read
    except ImportError as exc:
        raise RuntimeError("Reading trajectory frames requires ASE") from exc

    pattern_text = str(pattern)
    paths = [Path(path) for path in sorted(glob(pattern_text, recursive=True))]
    if not paths and Path(pattern_text).exists():
        paths = [Path(pattern_text)]

    groups = []
    for path in paths:
        items = read(path, index=":")
        if not isinstance(items, list):
            items = [items]
        groups.append((path, items))
    if not any(frames for _, frames in groups):
        raise ValueError(f"no frames matched pattern: {pattern_text}")
    return groups


def write_extxyz_many(path: str | Path, frames) -> None:
    """Write multiple frames as an extxyz dataset."""
    try:
        from ase.io import write
    except ImportError as exc:
        raise RuntimeError("Writing extxyz requires ASE") from exc

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write(output_path, frames, format="extxyz")
