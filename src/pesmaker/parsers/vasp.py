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
"""VASP output parsing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pesmaker.parsers.ase import read_frames


def read_outcar_frames(path: str | Path) -> list[Any]:
    """Read frames from a VASP OUTCAR through ASE."""
    return read_frames(path)
