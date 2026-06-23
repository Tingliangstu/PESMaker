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

import json
import shutil
from pathlib import Path
from typing import Any

from pesmaker.artifacts import _read_optional_file
from pesmaker.config.schema import PESMakerConfig
from pesmaker.jobs.scripts import _write_submit_script
from pesmaker.results import StageResult
from pesmaker.trainers.layout import (
    training_manifest_path,
    training_root,
    training_step1_complete,
    training_step1_dir,
    training_two_step_enabled,
    training_workdir,
)

NEP_DEFAULT_PARAMETERS: dict[str, str] = {
    "version": "4",
    "cutoff": "8 4",
    "n_max": "6 6",
    "basis_size": "6 6",
    "l_max": "4 1",
    "neuron": "30",
    "lambda_e": "1.0",
    "lambda_f": "1.0",
    "lambda_v": "0.1",
    "batch": "1000",
    "population": "50",
    "generation": "100000",
}

NEP_DEFAULT_COMMENTS: dict[str, str] = {
    "type": "mandatory; inferred from train.xyz when available",
    "version": "NEP4",
    "cutoff": "choose reasonably for your chemistry",
    "n_max": "default",
    "basis_size": "default",
    "l_max": "default",
    "neuron": "default",
    "lambda_e": "default",
    "lambda_f": "default",
    "lambda_v": "default",
    "batch": "default",
    "population": "default",
    "generation": "default",
}

NEP_SINGLE_STEP_WEIGHTS = {"lambda_e": "1.0", "lambda_f": "1.0", "lambda_v": "0.1"}
NEP_TWO_STEP_WEIGHTS = {
    "step1": {"lambda_e": "0.2", "lambda_f": "2", "lambda_v": "0.1"},
    "step2": {"lambda_e": "2", "lambda_f": "1", "lambda_v": "1"},
}


def setup_training(config: PESMakerConfig) -> StageResult:
    """Prepare potential training inputs and a submission script."""
    output_dir = _prepare_training_workdir(config)
    dataset_path = Path(str(config.training.options.get("dataset", "train.xyz")))
    target_dataset = output_dir / dataset_path.name
    if dataset_path.exists():
        shutil.copy2(dataset_path, target_dataset)

    if config.training.engine.lower() == "nep":
        input_name = "nep.in"
        default_input = _default_nep_input(config, dataset_path)
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
    manifest_path = _write_training_manifest(config, output_dir, submit_path)
    return StageResult(
        output_dir,
        tuple(
            path
            for path in (target_dataset, input_path, submit_path, manifest_path)
            if path.exists()
        ),
        _training_message(config, output_dir),
    )


def _prepare_training_workdir(config: PESMakerConfig) -> Path:
    workdir = training_workdir(config)
    if (
        training_two_step_enabled(config)
        and training_step1_complete(config)
        and not workdir.exists()
    ):
        shutil.copytree(training_step1_dir(config), workdir)
        return workdir
    workdir.mkdir(parents=True, exist_ok=True)
    return workdir


def _write_training_manifest(
    config: PESMakerConfig,
    workdir: Path,
    submit_path: Path,
) -> Path:
    output_root = training_root(config)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = training_manifest_path(config)
    record = {
        "workdir": str(workdir),
        "submit_script": str(submit_path),
    }
    if workdir.parent == output_root:
        record["training_step"] = workdir.name
    manifest_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return manifest_path


def _training_message(config: PESMakerConfig, output_dir: Path) -> str:
    engine = config.training.engine
    if training_two_step_enabled(config):
        return f"Prepared {output_dir.name} training folder for {engine}"
    return f"Prepared training folder for {engine}"


def _default_nep_input(config: PESMakerConfig, dataset_path: Path) -> str:
    parameters = dict(NEP_DEFAULT_PARAMETERS)
    parameters.update(_nep_parameter_overrides(config.training.options))
    if training_two_step_enabled(config):
        step = "step2" if training_step1_complete(config) else "step1"
        parameters.update(NEP_TWO_STEP_WEIGHTS[step])
    else:
        parameters.update(NEP_SINGLE_STEP_WEIGHTS)

    elements = _nep_elements(config.training.options, dataset_path)
    lines = [
        "# Example NEP input generated by PESMaker.",
        "# Edit the values after checking GPUMD's nep.in documentation.",
        _format_nep_line("type", f"{len(elements)} {' '.join(elements)}"),
    ]
    for key in NEP_DEFAULT_PARAMETERS:
        if key in parameters:
            lines.append(_format_nep_line(key, parameters[key]))
    return "\n".join(lines) + "\n"


def _format_nep_line(key: str, value: object) -> str:
    text = _parameter_text(value)
    comment = NEP_DEFAULT_COMMENTS.get(key)
    if comment:
        return f"{key:<14}{text:<10} # {comment}"
    return f"{key:<14}{text}"


def _parameter_text(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return " ".join(str(item) for item in value)
    return str(value)


def _nep_parameter_overrides(options: dict[str, Any]) -> dict[str, str]:
    nested = options.get(
        "nep",
        options.get("parameters", options.get("nep_options", {})),
    )
    overrides: dict[str, Any] = {}
    if isinstance(nested, dict):
        overrides.update(nested)
    for key in NEP_DEFAULT_PARAMETERS:
        if key in options:
            overrides[key] = options[key]
    return {key: _parameter_text(value) for key, value in overrides.items()}


def _nep_elements(options: dict[str, Any], dataset_path: Path) -> list[str]:
    for key in ("elements", "types"):
        value = options.get(key)
        if isinstance(value, str):
            elements = value.split()
            if elements:
                return elements
        if isinstance(value, (list, tuple)):
            elements = [str(item) for item in value]
            if elements:
                return elements
    if dataset_path.exists():
        elements = _elements_from_extxyz(dataset_path)
        if elements:
            return elements
    return ["Te"]


def _elements_from_extxyz(path: Path) -> list[str]:
    elements: list[str] = []
    try:
        with path.open(encoding="utf-8") as handle:
            while True:
                first = handle.readline()
                if not first:
                    break
                first = first.strip()
                if not first:
                    continue
                try:
                    atom_count = int(first)
                except ValueError:
                    break
                handle.readline()
                for _ in range(atom_count):
                    atom_line = handle.readline()
                    if not atom_line:
                        break
                    symbol = atom_line.split()[0]
                    if symbol not in elements:
                        elements.append(symbol)
    except OSError:
        return []
    return elements
