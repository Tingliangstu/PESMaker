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
"""Smart-next workflow execution."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from pesmaker.config.schema import EngineConfig, PESMakerConfig
from pesmaker.dataset.extxyz import collect_labeled_dataset
from pesmaker.generators.structures import format_generate_summary, generate_structures
from pesmaker.jobs.submit import submit_jobs
from pesmaker.labelers.vasp import setup_labeling
from pesmaker.results import StageResult
from pesmaker.samplers.gpumd import setup_sampling
from pesmaker.samplers.selection import select_sampling_frames
from pesmaker.trainers.nep import setup_training
from pesmaker.workflow.plan import (
    dataset_path,
    generated_manifest_path,
    labeling_manifest_path,
    matched_outcars,
    matched_sampling_trajectories,
    outcar_pattern,
    resolve_workflow_mode,
    sampling_manifest_path,
    sampling_trajectory_pattern,
    selected_manifest_path,
    submit_command_text,
    training_submit_path,
)
from pesmaker.workflow.state import (
    dry_run_recorded,
    load_next_state,
    next_state_path,
    record_dry_run,
)


@dataclass(frozen=True)
class NextEvent:
    """One action performed or one boundary reached by `pesmaker next`."""

    kind: str
    message: str
    result: StageResult | None = None
    command: str | None = None
    log_path: Path | None = None


@dataclass(frozen=True)
class NextResult:
    """Summary returned by `run_next`."""

    mode: str
    status: str
    events: tuple[NextEvent, ...] = field(default_factory=tuple)
    state_path: Path | None = None


def run_next(config: PESMakerConfig, config_path: Path) -> NextResult:
    """Run local stages until the next submit or external-result boundary."""
    mode = resolve_workflow_mode(config)
    state = load_next_state(config)
    events: list[NextEvent] = []

    while True:
        if not generated_manifest_path(config).exists():
            result = generate_structures(config)
            events.append(
                NextEvent(
                    kind="run",
                    message=format_generate_summary(
                        result, include_details=False
                    ).strip(),
                )
            )
            continue

        if mode == "sampling-training":
            if not sampling_manifest_path(config).exists():
                result = setup_sampling(config)
                events.append(
                    NextEvent(kind="run", message=result.message, result=result)
                )
                continue

            if not dry_run_recorded(state, "sampling"):
                event = _preview_submit(config, config_path, state, stage="sampling")
                events.append(event)
                return _result(config, mode, "submit-preview", events)

            if not matched_sampling_trajectories(config):
                command = submit_command_text(config_path, "sampling")
                events.append(
                    NextEvent(
                        kind="wait",
                        message=(
                            "Waiting for sampling trajectories matching "
                            f"{sampling_trajectory_pattern(config)}."
                        ),
                        command=command,
                    )
                )
                return _result(config, mode, "waiting", events)

            if not selected_manifest_path(config).exists():
                result = select_sampling_frames(config)
                events.append(
                    NextEvent(kind="run", message=result.message, result=result)
                )
                continue

        if not labeling_manifest_path(config).exists():
            labeling_config = _labeling_config(config, mode)
            result = setup_labeling(labeling_config)
            events.append(NextEvent(kind="run", message=result.message, result=result))
            continue

        if not dry_run_recorded(state, "scf"):
            event = _preview_submit(config, config_path, state, stage="scf")
            events.append(event)
            return _result(config, mode, "submit-preview", events)

        if not matched_outcars(config):
            command = submit_command_text(config_path, "scf")
            events.append(
                NextEvent(
                    kind="wait",
                    message=f"Waiting for SCF outputs matching {outcar_pattern(config)}.",
                    command=command,
                )
            )
            return _result(config, mode, "waiting", events)

        if not dataset_path(config).exists():
            result = collect_labeled_dataset(config)
            events.append(NextEvent(kind="run", message=result.message, result=result))
            if mode == "direct-scf":
                return _result(config, mode, "complete", events)
            continue

        if mode == "direct-scf":
            events.append(
                NextEvent(
                    kind="complete",
                    message=f"Direct SCF workflow is complete: {dataset_path(config)}",
                )
            )
            return _result(config, mode, "complete", events)

        if not training_submit_path(config).exists():
            result = setup_training(config)
            events.append(NextEvent(kind="run", message=result.message, result=result))
            continue

        if not dry_run_recorded(state, "training"):
            event = _preview_submit(config, config_path, state, stage="training")
            events.append(event)
            return _result(config, mode, "submit-preview", events)

        events.append(
            NextEvent(
                kind="complete",
                message="Sampling, labeling, collection, and training setup are complete.",
            )
        )
        return _result(config, mode, "complete", events)


def _preview_submit(
    config: PESMakerConfig,
    config_path: Path,
    state: dict,
    *,
    stage: str,
) -> NextEvent:
    result = submit_jobs(config, stage=stage, dry_run=True)
    command = submit_command_text(config_path, stage)
    log_path = result.files[0] if result.files else result.output_dir
    record_dry_run(config, state, stage=stage, command=command, log_path=log_path)
    return NextEvent(
        kind="submit-preview",
        message=result.message,
        result=result,
        command=command,
        log_path=log_path,
    )


def _labeling_config(config: PESMakerConfig, mode: str) -> PESMakerConfig:
    """Default sampling-training SCF input to the selected-frame manifest."""
    if mode != "sampling-training":
        return config
    if config.labeling.options.get("input_manifest") or config.labeling.options.get(
        "input_dir"
    ):
        return config
    manifest = selected_manifest_path(config)
    if not manifest.exists():
        return config
    options = {**config.labeling.options, "input_manifest": str(manifest)}
    return replace(
        config,
        labeling=EngineConfig(engine=config.labeling.engine, options=options),
    )


def _result(
    config: PESMakerConfig,
    mode: str,
    status: str,
    events: list[NextEvent],
) -> NextResult:
    return NextResult(
        mode=mode,
        status=status,
        events=tuple(events),
        state_path=next_state_path(config),
    )
