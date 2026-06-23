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
"""Artifact-driven smart workflow execution."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from pesmaker.config.schema import EngineConfig, PESMakerConfig
from pesmaker.dataset.extxyz import collect_labeled_dataset
from pesmaker.generators.structures import format_generate_summary, generate_structures
from pesmaker.jobs.submit import submit_jobs
from pesmaker.labelers.vasp import setup_labeling
from pesmaker.results import StageResult
from pesmaker.samplers import setup_sampling
from pesmaker.samplers.selection import select_sampling_frames
from pesmaker.trainers.nep import setup_training
from pesmaker.workflow.plan import (
    dataset_path,
    generated_manifest_path,
    labeling_manifest_path,
    matched_outcars,
    matched_sampling_trajectories,
    outcar_pattern,
    sampling_manifest_path,
    sampling_trajectory_pattern,
    selected_manifest_path,
    submit_command_text,
    training_dry_run_key,
    training_step1_complete,
    training_step2_submit_path,
    training_submit_path,
    training_two_step_enabled,
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
    stage: str | None = None
    log_path: Path | None = None
    template_path: Path | None = None
    template_created: bool = False


@dataclass(frozen=True)
class NextResult:
    """Summary returned by `run_next` or `inspect_next`."""

    flow: str
    status: str
    events: tuple[NextEvent, ...] = field(default_factory=tuple)
    state_path: Path | None = None


@dataclass(frozen=True)
class NextStep:
    """Internal description of the next required workflow step."""

    action: str
    kind: str
    message: str
    stage: str | None = None
    command: str | None = None
    log_path: Path | None = None
    template_path: Path | None = None
    stage_key: str | None = None


def run_next(config: PESMakerConfig, config_path: Path) -> NextResult:
    """Run local stages until the next submit or external-result boundary."""
    state = load_next_state(config)
    events: list[NextEvent] = []

    while True:
        step = determine_next_step(config, config_path, state)

        if step.action == "generate":
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

        if step.action == "setup_sampling":
            result = setup_sampling(config)
            events.append(NextEvent(kind="run", message=result.message, result=result))
            continue

        if step.action == "select":
            result = select_sampling_frames(config)
            events.append(NextEvent(kind="run", message=result.message, result=result))
            continue

        if step.action == "setup_labeling":
            result = setup_labeling(_labeling_config(config))
            events.append(NextEvent(kind="run", message=result.message, result=result))
            continue

        if step.action == "collect":
            result = collect_labeled_dataset(config)
            events.append(
                NextEvent(
                    kind="run",
                    message=result.message,
                    result=result,
                    stage=step.stage,
                )
            )
            continue

        if step.action == "setup_training":
            result = setup_training(config)
            events.append(NextEvent(kind="run", message=result.message, result=result))
            continue

        if step.action == "write_next_config" and step.template_path is not None:
            created = _write_next_config_template(config, step.template_path)
            events.append(
                NextEvent(
                    kind=step.kind,
                    message=step.message,
                    command=step.command,
                    stage=step.stage,
                    template_path=step.template_path,
                    template_created=created,
                )
            )
            return _result(config, step.kind, events)

        if step.action == "preview_submit" and step.stage:
            event = _preview_submit(
                config,
                config_path,
                state,
                stage=step.stage,
                dry_run_key=step.stage_key,
            )
            events.append(event)
            return _result(config, step.kind, events)

        events.append(
            NextEvent(
                kind=step.kind,
                message=step.message,
                command=step.command,
                stage=step.stage,
                log_path=step.log_path,
            )
        )
        return _result(config, step.kind, events)


def inspect_next(config: PESMakerConfig, config_path: Path) -> NextResult:
    """Inspect the next workflow step without writing files."""
    state = load_next_state(config)
    step = determine_next_step(config, config_path, state)
    event = NextEvent(
        kind=f"next-action:{step.kind}",
        message=step.message,
        command=step.command,
        stage=step.stage,
        log_path=step.log_path,
        template_path=step.template_path,
    )
    return _result(config, "status", [event])


def determine_next_step(
    config: PESMakerConfig,
    config_path: Path,
    state: dict,
) -> NextStep:
    """Return the next action from config sections and artifact state."""
    if _should_generate(config):
        return NextStep(
            action="generate",
            kind="run",
            message="Generate structures from the configured inputs.",
        )

    if _needs_next_config(config):
        template_path = _next_config_template_path(config_path)
        return NextStep(
            action="write_next_config",
            kind="config-needed",
            message="More settings are needed before SCF setup.",
            command=f"pesmaker next {template_path}",
            template_path=template_path,
        )

    if _sampling_enabled(config):
        if _selection_enabled(config) and not sampling_manifest_path(config).exists():
            if matched_sampling_trajectories(config):
                if not selected_manifest_path(config).exists():
                    return NextStep(
                        action="select",
                        kind="run",
                        message="Select frames from an existing trajectory.",
                    )
            elif not _sampling_setup_inputs_available(config):
                return NextStep(
                    action="wait",
                    kind="waiting",
                    stage="selection",
                    message=(
                        "Waiting for trajectory files matching "
                        f"{sampling_trajectory_pattern(config)}."
                    ),
                )
        if not sampling_manifest_path(config).exists():
            return NextStep(
                action="setup_sampling",
                kind="run",
                message="Prepare MD-sampling folders.",
            )
        if not dry_run_recorded(state, "sampling"):
            return NextStep(
                action="preview_submit",
                kind="submit-preview",
                stage="sampling",
                command=submit_command_text(config_path, "sampling"),
                message="Preview MD-sampling job submission.",
            )
        if _selection_enabled(config):
            if not matched_sampling_trajectories(config):
                return NextStep(
                    action="wait",
                    kind="waiting",
                    command=submit_command_text(config_path, "sampling"),
                    message=(
                        "Waiting for MD-sampling trajectories matching "
                        f"{sampling_trajectory_pattern(config)}."
                    ),
                )
            if not selected_manifest_path(config).exists():
                return NextStep(
                    action="select",
                    kind="run",
                    message="Select representative frames from sampling trajectories.",
                )
        elif not _labeling_has_explicit_input(config):
            return NextStep(
                action="complete",
                kind="complete",
                message=(
                    "Sampling setup is ready. Add sampling.selection or "
                    "labeling.input_manifest/input_dir if later stages should run."
                ),
            )

    if _selection_only_enabled(config):
        if not matched_sampling_trajectories(config):
            return NextStep(
                action="wait",
                kind="waiting",
                stage="selection",
                message=(
                    "Waiting for trajectory files matching "
                    f"{sampling_trajectory_pattern(config)}."
                ),
            )
        if not selected_manifest_path(config).exists():
            return NextStep(
                action="select",
                kind="run",
                message="Select frames from an existing trajectory.",
            )

    if _labeling_enabled(config):
        if _is_migrated_scf_submission(config):
            return NextStep(
                action="manual_scf_submit",
                kind="scf-retry",
                stage="scf",
                command=submit_command_text(config_path, "scf"),
                log_path=_scf_submission_log_path(config),
                message=(
                    "Existing VASP calculation folders look like a migrated "
                    "or retry submission."
                ),
            )
        if not labeling_manifest_path(config).exists():
            return NextStep(
                action="setup_labeling",
                kind="run",
                message="Prepare VASP SCF labeling folders.",
            )
        if not dry_run_recorded(state, "scf"):
            return NextStep(
                action="preview_submit",
                kind="submit-preview",
                stage="scf",
                command=submit_command_text(config_path, "scf"),
                message="Preview SCF job submission.",
            )
        if not matched_outcars(config):
            return NextStep(
                action="wait",
                kind="waiting",
                command=submit_command_text(config_path, "scf"),
                message=f"Waiting for SCF outputs matching {outcar_pattern(config)}.",
            )
        if not dataset_path(config).exists():
            return NextStep(
                action="collect",
                kind="run",
                stage="collect",
                message="Collect finished SCF outputs into the training dataset.",
            )

    if _collecting_enabled(config) and not dataset_path(config).exists():
        return NextStep(
            action="collect",
            kind="run",
            stage="collect",
            message="Collect finished labeled outputs into the training dataset.",
        )

    if _training_enabled(config):
        if not training_submit_path(config).exists():
            return NextStep(
                action="setup_training",
                kind="run",
                message="Prepare model-training inputs.",
            )
        training_key = training_dry_run_key(config)
        if not dry_run_recorded(state, training_key):
            return NextStep(
                action="preview_submit",
                kind="submit-preview",
                stage="training",
                stage_key=training_key,
                command=submit_command_text(config_path, "training"),
                message="Preview training job submission.",
            )
        if (
            training_two_step_enabled(config)
            and not training_step1_complete(config)
        ):
            return NextStep(
                action="wait",
                kind="waiting",
                stage="training",
                command=submit_command_text(config_path, "training"),
                message=(
                    "Waiting for first-step NEP output "
                    f"{training_submit_path(config).parent / 'nep.txt'}."
                ),
            )
        if (
            training_two_step_enabled(config)
            and training_step1_complete(config)
            and not training_step2_submit_path(config).exists()
        ):
            return NextStep(
                action="setup_training",
                kind="run",
                message="Prepare second-step NEP training inputs.",
            )

    if _collecting_enabled(config) and dataset_path(config).exists():
        return NextStep(
            action="training_config_needed",
            kind="training-config-needed",
            stage="training",
            message="The dataset is ready; configure model training next.",
        )

    return NextStep(
        action="complete",
        kind="complete",
        message="No further PESMaker action is required for the current artifacts.",
    )


def inferred_flow(config: PESMakerConfig) -> str:
    """Return a concise human-readable workflow inferred from config sections."""
    if _is_migrated_scf_submission(config):
        return "SCF-retry submission"
    stages = []
    if config.structures:
        stages.append("generate")
    if _existing_trajectory_selection(config):
        stages.append(_selection_flow_label(config))
    elif _sampling_enabled(config):
        stages.append("MD-sampling")
        if _selection_enabled(config):
            stages.append(_selection_flow_label(config))
    elif _selection_enabled(config):
        stages.append(_selection_flow_label(config))
    if _labeling_enabled(config):
        stages.extend(["SCF-labeling", "dataset-collect"])
    elif _collecting_enabled(config):
        stages.append("dataset-collect")
    if _training_enabled(config):
        stages.append("train")
    if (
        (config.structures or selected_manifest_path(config).exists())
        and not _labeling_enabled(config)
        and not _training_enabled(config)
        and (not _sampling_enabled(config) or _selection_enabled(config))
    ):
        stages.append("SCF-config-needed")
    return " -> ".join(stages) if stages else "inspect existing artifacts"


def _preview_submit(
    config: PESMakerConfig,
    config_path: Path,
    state: dict,
    *,
    stage: str,
    dry_run_key: str | None = None,
) -> NextEvent:
    result = submit_jobs(config, stage=stage, dry_run=True)
    command = submit_command_text(config_path, stage)
    log_path = result.files[0] if result.files else result.output_dir
    record_dry_run(
        config,
        state,
        stage=dry_run_key or stage,
        command=command,
        log_path=log_path,
    )
    return NextEvent(
        kind="submit-preview",
        message=result.message,
        result=result,
        command=command,
        stage=stage,
        log_path=log_path,
    )


def _should_generate(config: PESMakerConfig) -> bool:
    return bool(config.structures) and not generated_manifest_path(config).exists()


def _needs_next_config(config: PESMakerConfig) -> bool:
    generation_only = (
        bool(config.structures)
        and generated_manifest_path(config).exists()
        and not _sampling_enabled(config)
        and not _labeling_enabled(config)
        and not _training_enabled(config)
    )
    selected_without_labeling = (
        _selection_enabled(config)
        and selected_manifest_path(config).exists()
        and not _labeling_enabled(config)
        and not _training_enabled(config)
    )
    return generation_only or selected_without_labeling


def _sampling_enabled(config: PESMakerConfig) -> bool:
    if config.workflow.mode == "direct-scf":
        return False
    if not config.sampling_configured:
        return False
    return config.sampling.engine.strip().lower() not in {"", "none"}


def _sampling_setup_inputs_available(config: PESMakerConfig) -> bool:
    options = config.sampling.options
    if config.structures:
        return True
    if options.get("input_manifest") or options.get("input_dir"):
        return True
    generated_manifest = generated_manifest_path(config)
    return generated_manifest.exists() or generated_manifest.parent.exists()


def _selection_enabled(config: PESMakerConfig) -> bool:
    return isinstance(config.sampling.options.get("selection"), dict)


def _selection_only_enabled(config: PESMakerConfig) -> bool:
    return (
        config.workflow.mode != "direct-scf"
        and _selection_enabled(config)
        and not _sampling_enabled(config)
    )


def _existing_trajectory_selection(config: PESMakerConfig) -> bool:
    return (
        config.workflow.mode != "direct-scf"
        and _selection_enabled(config)
        and _sampling_enabled(config)
        and not sampling_manifest_path(config).exists()
        and not _sampling_setup_inputs_available(config)
    )


def _selection_flow_label(config: PESMakerConfig) -> str:
    selection = config.sampling.options.get("selection", {})
    if not isinstance(selection, dict):
        return "select"
    raw_method = selection.get(
        "method",
        selection.get("strategy", selection.get("mode")),
    )
    if raw_method is None and any(
        key in selection for key in ("interval", "stride", "step", "frame_interval")
    ):
        raw_method = "interval"
    method = str(raw_method or "fps").lower().replace("-", "_")
    if method in {"interval", "stride", "uniform", "even", "every_n"}:
        return "interval-select"
    return "FPS-select"


def _labeling_enabled(config: PESMakerConfig) -> bool:
    if not config.labeling_configured:
        return False
    return config.labeling.engine.strip().lower() not in {"", "none"}


def _collecting_enabled(config: PESMakerConfig) -> bool:
    if not config.collecting_configured:
        return False
    return config.collecting.engine.strip().lower() not in {"", "none"}


def _is_migrated_scf_submission(config: PESMakerConfig) -> bool:
    """Detect prepared VASP folders that should be submitted, not set up."""
    if config.structures or _sampling_enabled(config):
        return False
    if _labeling_has_explicit_input(config):
        return False
    output_dir = labeling_manifest_path(config).parent
    if not output_dir.is_dir():
        return False
    return any(output_dir.rglob("POSCAR"))


def _scf_submission_log_path(config: PESMakerConfig) -> Path:
    return labeling_manifest_path(config).parent / "scf_submitted_jobs.txt"


def _training_enabled(config: PESMakerConfig) -> bool:
    if config.workflow.mode == "direct-scf":
        return False
    if not config.training_configured:
        return False
    return (
        config.training.engine.strip().lower() not in {"", "none"}
        and bool(config.training.options)
    )


def _next_config_template_path(config_path: Path) -> Path:
    suffix = config_path.suffix or ".yaml"
    return config_path.with_name(f"{config_path.stem}.next{suffix}")


def _write_next_config_template(config: PESMakerConfig, path: Path) -> bool:
    if path.exists():
        return False
    input_line = _next_config_input_line(config)
    text = f"""project: {config.project}

labeling:
  engine: vasp
  output_dir: run_vasp_scf
  {input_line}
  incar: /path/to/INCAR
  potcar_library: /path/to/VASP/potentials
  command: /path/to/vasp_std

jobs:
  submit_command: sbatch
  cores_cpu: 36
  skip_completed: true
  check_scf_convergence: true
  sub_file: /path/to/sub.sh
"""
    path.write_text(text, encoding="utf-8")
    return True


def _next_config_input_line(config: PESMakerConfig) -> str:
    manifest = selected_manifest_path(config)
    if manifest.exists():
        return f"input_manifest: {manifest.as_posix()}"
    return f"input_dir: {generated_manifest_path(config).parent.as_posix()}"


def _labeling_has_explicit_input(config: PESMakerConfig) -> bool:
    return bool(
        config.labeling.options.get("input_manifest")
        or config.labeling.options.get("input_dir")
    )


def _labeling_config(config: PESMakerConfig) -> PESMakerConfig:
    """Prefer selected frames for SCF setup when selection has run."""
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
    status: str,
    events: list[NextEvent],
) -> NextResult:
    return NextResult(
        flow=inferred_flow(config),
        status=status,
        events=tuple(events),
        state_path=next_state_path(config),
    )
