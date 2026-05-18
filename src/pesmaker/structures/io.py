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
"""ASE-backed structure input and output helpers."""

from __future__ import annotations

from pathlib import Path


def load_structure(path: str | Path):
    """Read one atomistic structure from any ASE-readable file."""
    try:
        from ase.io import read
    except ImportError as exc:
        message = "Structure IO requires ASE. Install pesmaker with: pip install -e .[atomistic]"
        raise RuntimeError(message) from exc

    return read(Path(path))


def write_structure(atoms, path: str | Path, *, fmt: str | None = None) -> None:
    """Write one atomistic structure and create parent directories if needed."""
    try:
        from ase.io import write
    except ImportError as exc:
        message = "Structure IO requires ASE. Install pesmaker with: pip install -e .[atomistic]"
        raise RuntimeError(message) from exc

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write(output_path, atoms, format=fmt)
