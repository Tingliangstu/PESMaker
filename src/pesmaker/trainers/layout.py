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
"""Training-stage directory layout helpers."""

from __future__ import annotations

from pathlib import Path

from pesmaker.artifacts import _section_output_dir
from pesmaker.config.schema import PESMakerConfig


def training_root(config: PESMakerConfig) -> Path:
    """Return the configured training output root."""
    return _section_output_dir(config, config.training.options, "training")


def training_manifest_path(config: PESMakerConfig) -> Path:
    """Return the manifest that points submit logic at the active training job."""
    return training_root(config) / "training_manifest.jsonl"


def training_submit_path(config: PESMakerConfig) -> Path:
    """Return the active training submit script path."""
    return training_workdir(config) / "submit.sh"


def training_workdir(config: PESMakerConfig) -> Path:
    """Return the directory for the active training job."""
    if not training_two_step_enabled(config):
        return training_root(config)
    if training_step1_complete(config):
        return training_step2_dir(config)
    return training_step1_dir(config)


def training_step1_dir(config: PESMakerConfig) -> Path:
    """Return the first-step training directory."""
    return training_root(config) / "step1"


def training_step2_dir(config: PESMakerConfig) -> Path:
    """Return the second-step training directory."""
    return training_root(config) / "step2"


def training_step1_complete(config: PESMakerConfig) -> bool:
    """Return whether first-step NEP training has produced a model file."""
    return (training_step1_dir(config) / "nep.txt").is_file()


def training_step2_submit_path(config: PESMakerConfig) -> Path:
    """Return the second-step submit script path."""
    return training_step2_dir(config) / "submit.sh"


def training_two_step_enabled(config: PESMakerConfig) -> bool:
    """Return whether the training config requests a two-step workflow."""
    options = config.training.options
    value = options.get("two_step", options.get("two-step", options.get("two_stage")))
    return bool(value)


def training_dry_run_key(config: PESMakerConfig) -> str:
    """Return the state key for the active training dry-run gate."""
    if not training_two_step_enabled(config):
        return "training"
    return "training_step2" if training_step1_complete(config) else "training_step1"
