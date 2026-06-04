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

"""Collect labeled structures into extxyz datasets."""

from __future__ import annotations

from glob import glob
from pathlib import Path

from pesmaker.artifacts import _section_output_dir
from pesmaker.config.schema import PESMakerConfig
from pesmaker.results import StageResult
from pesmaker.samplers.selection import _read_trajectory_frames, _write_extxyz_many


def collect_labeled_dataset(config: PESMakerConfig) -> StageResult:
    """Collect completed VASP SCF calculations into `train.xyz`."""
    output_dir = _section_output_dir(config, config.dataset.__dict__, "dataset")
    output_dir.mkdir(parents=True, exist_ok=True)
    default_pattern = (
        _section_output_dir(config, config.labeling.options, "labeling")
        / "**"
        / "OUTCAR"
    )
    pattern = str(config.labeling.options.get("outcar_pattern", default_pattern))
    output_path = Path(
        str(config.labeling.options.get("dataset_path", output_dir / "train.xyz"))
    )
    outputs = [Path(path) for path in sorted(glob(pattern, recursive=True))]
    if not outputs:
        raise ValueError(f"no VASP outputs matched pattern: {pattern}")

    frames = []
    for output in outputs:
        frames.extend(_read_trajectory_frames(str(output)))
    _write_extxyz_many(output_path, frames)
    return StageResult(
        output_dir,
        (output_path,),
        f"Collected {len(frames)} labeled frame(s) into {output_path}",
    )
