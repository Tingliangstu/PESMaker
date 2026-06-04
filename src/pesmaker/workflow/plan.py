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
"""Workflow-mode and artifact helpers for `pesmaker next`."""

from __future__ import annotations

from glob import glob
from pathlib import Path

from pesmaker.artifacts import _generated_structures_dir, _section_output_dir
from pesmaker.config.schema import PESMakerConfig


def resolve_workflow_mode(config: PESMakerConfig) -> str:
    """Resolve `workflow: auto` into a concrete smart-next path."""
    mode = config.workflow.mode
    if mode != "auto":
        return mode
    selection = config.sampling.options.get("selection")
    if config.sampling.engine.lower() != "none" and isinstance(selection, dict):
        return "sampling-training"
    return "direct-scf"


def generated_manifest_path(config: PESMakerConfig) -> Path:
    return _generated_structures_dir(config) / "manifest.jsonl"


def sampling_manifest_path(config: PESMakerConfig) -> Path:
    return _section_output_dir(config, config.sampling.options, "sampling") / (
        "sampling_manifest.jsonl"
    )


def selected_manifest_path(config: PESMakerConfig) -> Path:
    selection = config.sampling.options.get("selection", {})
    output_dir = Path("selected")
    if isinstance(selection, dict):
        output_dir = Path(str(selection.get("output_dir", output_dir)))
    return output_dir / "manifest.jsonl"


def labeling_manifest_path(config: PESMakerConfig) -> Path:
    return _section_output_dir(config, config.labeling.options, "labeling") / (
        "labeling_manifest.jsonl"
    )


def training_submit_path(config: PESMakerConfig) -> Path:
    return (
        _section_output_dir(config, config.training.options, "training") / "submit.sh"
    )


def dataset_path(config: PESMakerConfig) -> Path:
    output_dir = _section_output_dir(config, config.dataset.__dict__, "dataset")
    return Path(
        str(config.labeling.options.get("dataset_path", output_dir / "train.xyz"))
    )


def outcar_pattern(config: PESMakerConfig) -> str:
    default_pattern = (
        _section_output_dir(config, config.labeling.options, "labeling")
        / "**"
        / "OUTCAR"
    )
    return str(config.labeling.options.get("outcar_pattern", default_pattern))


def matched_outcars(config: PESMakerConfig) -> list[Path]:
    pattern = outcar_pattern(config)
    paths = [Path(path) for path in sorted(glob(pattern, recursive=True))]
    if not paths and Path(pattern).exists():
        paths = [Path(pattern)]
    return paths


def sampling_trajectory_pattern(config: PESMakerConfig) -> str:
    selection = config.sampling.options.get("selection", {})
    if isinstance(selection, dict):
        return str(selection.get("trajectory_pattern", "sampling/**/movie.xyz"))
    return "sampling/**/movie.xyz"


def matched_sampling_trajectories(config: PESMakerConfig) -> list[Path]:
    pattern = sampling_trajectory_pattern(config)
    paths = [Path(path) for path in sorted(glob(pattern, recursive=True))]
    if not paths and Path(pattern).exists():
        paths = [Path(pattern)]
    return paths


def submit_command_text(config_path: Path, stage: str) -> str:
    command = f"pesmaker submit {config_path}"
    if stage != "scf":
        command = f"{command} --stage {stage}"
    return command
