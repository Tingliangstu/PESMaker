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
"""Plot command registry used by the CLI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pesmaker.plot.nep import plot_nep_training
from pesmaker.plot.result import PlotResult

PlotHandler = Callable[[Path, Path], PlotResult]


@dataclass(frozen=True)
class PlotCommand:
    """One supported plot command."""

    name: str
    description: str
    handler: PlotHandler


def available_plot_commands() -> tuple[str, ...]:
    """Return stable plot command names for argument parsing."""
    return tuple(PLOT_COMMANDS)


def run_plot_command(name: str, *, source_dir: Path, output_dir: Path) -> PlotResult:
    """Run a named plot command."""
    command = PLOT_COMMANDS.get(name)
    if command is None:
        choices = ", ".join(available_plot_commands())
        raise ValueError(f"unknown plot type: {name}. Available plot types: {choices}")
    return command.handler(source_dir, output_dir)


def _run_nep_training_plot(source_dir: Path, output_dir: Path) -> PlotResult:
    return plot_nep_training(source_dir, output_dir=output_dir)


PLOT_COMMANDS: dict[str, PlotCommand] = {
    "train": PlotCommand(
        name="train",
        description="Plot GPUMD/NEP training outputs.",
        handler=_run_nep_training_plot,
    ),
}
