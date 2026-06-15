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

"""Common result objects returned by PESMaker stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SubmissionSummary:
    """Structured counts for one scheduler submission pass."""

    total_jobs: int
    completed_jobs: int
    pending_jobs: int


@dataclass(frozen=True)
class StageResult:
    """Summary for a prepared or collected workflow stage."""

    output_dir: Path
    files: tuple[Path, ...]
    message: str
    warnings: tuple[str, ...] = field(default_factory=tuple)
    submission: SubmissionSummary | None = None
