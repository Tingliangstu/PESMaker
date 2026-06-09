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


def load_structure(path: str | Path, *, index: int | str | None = None):
    """Read one atomistic structure from any ASE-readable file.

    Args:
        path: File path accepted by ASE, such as CIF, POSCAR, VASP, or extxyz.
        index: Optional ASE frame index for multi-frame files.

    Returns:
        An ASE `Atoms` object.

    Raises:
        RuntimeError: If ASE is not installed.
        FileNotFoundError: If ASE cannot find the requested path.
    """
    try:
        from ase.io import read
    except ImportError as exc:
        message = "Structure IO requires ASE. Install pesmaker with: pip install -e .[atomistic]"
        raise RuntimeError(message) from exc

    if index is None:
        return read(Path(path))
    return read(Path(path), index=index)


def write_structure(atoms, path: str | Path, *, fmt: str | None = None) -> None:
    """Write one atomistic structure and create parent directories if needed.

    Args:
        atoms: ASE `Atoms` object to write.
        path: Destination file path.
        fmt: Optional ASE writer format, such as `vasp` or `extxyz`.

    Raises:
        RuntimeError: If ASE is not installed.
        OSError: If the destination cannot be created or written.
    """
    try:
        from ase.io import write
    except ImportError as exc:
        message = "Structure IO requires ASE. Install pesmaker with: pip install -e .[atomistic]"
        raise RuntimeError(message) from exc

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write(output_path, _prepare_atoms_for_write(atoms, output_path, fmt), format=fmt)


def _prepare_atoms_for_write(atoms, path: Path, fmt: str | None):
    """Return an output-ready Atoms object for format-specific conventions."""
    if _is_vasp_output(path, fmt):
        return _group_atoms_by_first_symbol(atoms)
    return atoms


def _is_vasp_output(path: Path, fmt: str | None) -> bool:
    if fmt is not None:
        return fmt.lower() in {"vasp", "poscar"}
    return path.suffix.lower() in {".vasp", ".poscar"}


def _group_atoms_by_first_symbol(atoms):
    """Group atoms by first-seen element order for compact VASP POSCAR output."""
    symbols = atoms.get_chemical_symbols()
    groups: dict[str, list[int]] = {}
    order: list[str] = []
    for index, symbol in enumerate(symbols):
        if symbol not in groups:
            groups[symbol] = []
            order.append(symbol)
        groups[symbol].append(index)

    indices = [index for symbol in order for index in groups[symbol]]
    if indices == list(range(len(symbols))):
        return atoms
    return atoms[indices]
