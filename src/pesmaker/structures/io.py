# Copyright (c) 2026 Ting Liang. All rights reserved.
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
