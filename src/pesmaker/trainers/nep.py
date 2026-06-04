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

"""Potential-training setup helpers for NEP and compatible trainers."""

from __future__ import annotations

import shutil
from pathlib import Path

from pesmaker.artifacts import _read_optional_file, _section_output_dir
from pesmaker.config.schema import PESMakerConfig
from pesmaker.jobs.scripts import _write_submit_script
from pesmaker.results import StageResult

DEFAULT_NEP_IN = """type 1 Te
version 4
prediction 0
potential nep.txt
"""


def setup_training(config: PESMakerConfig) -> StageResult:
    """Prepare potential training inputs and a submission script."""
    output_dir = _section_output_dir(config, config.training.options, "training")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(str(config.training.options.get("dataset", "train.xyz")))
    target_dataset = output_dir / dataset_path.name
    if dataset_path.exists():
        shutil.copy2(dataset_path, target_dataset)

    if config.training.engine.lower() == "nep":
        input_name = "nep.in"
        default_input = DEFAULT_NEP_IN
        command = str(config.training.options.get("command", "nep"))
    else:
        input_name = "train.in"
        default_input = "# Add trainer-specific options here.\n"
        command = str(config.training.options.get("command", config.training.engine))
    input_text = _read_optional_file(
        config.training.options.get("input"),
        default=default_input,
    )
    input_path = output_dir / input_name
    input_path.write_text(input_text, encoding="utf-8")
    submit_path = _write_submit_script(
        config,
        output_dir,
        stage="training",
        command=command,
    )
    return StageResult(
        output_dir,
        tuple(
            path for path in (target_dataset, input_path, submit_path) if path.exists()
        ),
        f"Prepared training folder for {config.training.engine}",
    )
