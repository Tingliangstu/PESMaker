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
"""User-facing output formatting for `pesmaker next` and `pesmaker status`."""

from __future__ import annotations

from pathlib import Path

from pesmaker.workflow.next import NextEvent, NextResult


def print_next_concise(result: NextResult, *, config_path: Path) -> None:
    """Print the normal short `pesmaker next` output."""
    run_events = [event for event in result.events if event.kind == "run"]
    boundary_events = [event for event in result.events if event.kind != "run"]
    current_event = boundary_events[-1] if boundary_events else None

    print("Next flow")
    print(f"Flow             : {result.flow}")
    print(f"Current          : {_current_next_label(current_event, result.status)}")
    print()

    if run_events:
        print("Work done:")
        for event in run_events:
            _print_run_event(event)
        print()

    if not boundary_events:
        print("Next:")
        print(f"  1. Run: pesmaker next {config_path}")
        print()
        return

    for event in boundary_events:
        if event.kind == "submit-preview":
            _print_submit_preview(
                event,
                config_path=config_path,
                show_output_dir=not _run_events_include_output_dir(
                    run_events, event
                ),
            )
            continue
        if event.kind == "config-needed":
            _print_config_needed(event, config_path=config_path)
            continue
        if event.kind == "training-config-needed":
            _print_training_config_needed(event, config_path=config_path)
            continue
        if event.kind == "scf-retry":
            _print_scf_retry(event, config_path=config_path)
            continue
        if event.kind in {"wait", "waiting"}:
            _print_waiting(event, config_path=config_path)
            continue
        if event.kind == "complete":
            print("Complete:")
            print("  - No local PESMaker task needs to run now.")
            print()
            continue
        print("Next:")
        print(f"  1. Run: pesmaker next {config_path}")
        print()


def print_next_status(result: NextResult, *, config_path: Path) -> None:
    """Print read-only detailed status output."""
    print_next_diagnostics(result, config_path=config_path)


def print_next_starting(result: NextResult) -> None:
    """Print an immediate progress message before a long local `next` stage."""
    event = result.events[0] if result.events else None
    if _next_action_kind(event) != "run":
        return
    if _event_stage(event) != "collect":
        return
    print("Starting collection:")
    print("  - PESMaker is scanning OUTCAR files and parsing VASP results.")
    print("  - For many calculations this can take several minutes. Please wait.")
    print(flush=True)


def print_next_verbose(
    preflight: NextResult,
    result: NextResult,
    *,
    config_path: Path,
) -> None:
    """Print verbose `next` diagnostics followed by execution details."""
    print_next_preflight(preflight, config_path=config_path)
    print_next_diagnostics(result, config_path=config_path)


def print_next_diagnostics(result: NextResult, *, config_path: Path) -> None:
    """Print detailed flow/status diagnostics for `status` and `next --verbose`."""
    print("Smart next")
    print(f"Inferred flow    : {result.flow}")
    print(f"Status           : {result.status}")
    if result.state_path is not None:
        print(f"State            : {result.state_path}")
    print()

    run_events = [event for event in result.events if event.kind == "run"]
    boundary_events = [event for event in result.events if event.kind != "run"]

    if run_events:
        print("Work done this run:")
        for event in run_events:
            print(f"  - {event.message}")
            if event.result is not None:
                print(f"    Output: {event.result.output_dir}")
                _print_result_warnings(event.result, prefix="    ")
        print()

    if not boundary_events:
        print("What you should do next:")
        print(f"  - Run again to continue: pesmaker next {config_path}")
        print()
        return

    if all(event.kind.startswith("next-action") for event in boundary_events):
        for event in boundary_events:
            print(f"Next action      : {event.message}")
            print()
            print("What you should do next:")
            if _next_action_kind(event) == "scf-retry":
                _print_scf_retry_steps(event, config_path=config_path)
            elif _next_action_kind(event) == "training-config-needed":
                _print_training_config_steps(config_path=config_path)
            else:
                print(f"  - Run: pesmaker next {config_path}")
            if _next_action_kind(event) == "config-needed" and event.template_path:
                print(f"  - Edit the generated template: {event.template_path}")
            elif event.command and _next_action_kind(event) != "scf-retry":
                print(f"  - {submit_action_label(_event_stage(event))}: {event.command}")
            print()
        return

    print("Stopped because:")
    for event in result.events:
        if event.kind == "run":
            continue
        if event.kind == "submit-preview":
            if event.log_path is None:
                print("  - A submission preview is the next required step.")
            else:
                print("Submission preview complete.")
            stage = _event_stage(event)
            print(f"Stage            : {stage}")
            if event.log_path is not None:
                print(f"Dry-run log      : {event.log_path}")
            print()
            print("What you should do next:")
            if event.log_path is not None:
                print(f"  1. Review the dry-run log: {event.log_path}")
            if event.command:
                print(f"  2. {submit_action_label(stage)}: {event.command}")
            print(f"  3. After those jobs finish, run: pesmaker next {config_path}")
            print(f"{submit_action_label(stage):<25}: {event.command}")
            continue
        if event.kind == "config-needed":
            template_path = _template_path(event, config_path)
            print(event.message)
            if event.template_created:
                print(f"Template written : {template_path}")
            else:
                print(f"Template exists  : {template_path}")
            print()
            print("What you should do next:")
            print(
                f"  1. Edit {template_path} and set INCAR, POTCAR, VASP, "
                "and submit script paths."
            )
            print(f"  2. Check it: pesmaker validate {template_path}")
            print(f"  3. Continue: pesmaker next {template_path}")
            continue
        if event.kind == "training-config-needed":
            print(f"  - {event.message}")
            print()
            print("What you should do next:")
            _print_training_config_steps(config_path=config_path)
            continue
        if event.kind == "scf-retry":
            print(f"  - {event.message}")
            print()
            print("What you should do next:")
            _print_scf_retry_steps(event, config_path=config_path)
            continue
        if event.kind in {"wait", "waiting"}:
            print("  - PESMaker is waiting for external job outputs.")
            print(f"Waiting for       : {event.message}")
            print()
            print("What you should do next:")
            stage = _event_stage(event)
            if event.command:
                print(
                    f"  1. If the jobs were not submitted yet, run: "
                    f"{event.command}"
                )
                print("  2. Wait for the scheduler jobs to finish.")
                print(f"  3. Run again: pesmaker next {config_path}")
                print(f"{submit_action_label(stage):<25}: {event.command}")
            else:
                print("  1. Wait until the required files exist.")
                print(f"  2. Run again: pesmaker next {config_path}")
            continue
        if event.kind == "complete":
            print(f"Complete          : {event.message}")
            print()
            print("What you should do next:")
            print("  - Inspect the generated outputs and archive this run if needed.")
        else:
            print(f"Next action      : {event.message}")
            print()
            print("What you should do next:")
            print(f"  - Run: pesmaker next {config_path}")
        if event.result is not None:
            print(f"Output directory : {event.result.output_dir}")
        print()


def print_next_preflight(result: NextResult, *, config_path: Path) -> None:
    """Print verbose preflight details before `next` writes files."""
    event = result.events[0] if result.events else None
    step_kind = _next_action_kind(event)
    print("Plan before execution")
    print(f"Inferred flow    : {result.flow}")
    print()

    if event is None:
        print("PESMaker did not find a next action.")
        print()
        return

    if step_kind == "run":
        print(f"Start with       : {event.message}")
        print(
            "Then             : continue through any later local PESMaker "
            "stages whose inputs are ready"
        )
        print(
            "Stop rule        : stop before real scheduler submission, or "
            "when external outputs are missing"
        )
        print("Submit behavior  : dry-run only; PESMaker will print the submit command")
    elif step_kind == "submit-preview":
        print(f"Start with       : {event.message}")
        print("Stop rule        : stop before real scheduler submission")
        if event.command:
            print(f"Submit command   : {event.command}")
    elif step_kind == "waiting":
        print("No local stage will run now.")
        print(f"Waiting for      : {event.message}")
        if event.command:
            print(f"Submit command   : {event.command}")
    elif step_kind == "config-needed":
        print("Start with       : write a follow-up VASP SCF config template")
        print("Stop rule        : wait for the user to edit the follow-up YAML")
        if event.template_path is not None:
            print(f"Template         : {event.template_path}")
    elif step_kind == "training-config-needed":
        print("No local stage will run now.")
        print("Reason           : the dataset exists but training is not configured")
        print("Stop rule        : wait for the user to add a training section")
    elif step_kind == "scf-retry":
        print("No SCF setup will run.")
        print(f"Detected         : {event.message}")
        print("Stop rule        : wait for explicit retry submission")
        if event.command:
            print(f"Preview command  : {event.command} --dry-run")
    elif step_kind == "complete":
        print("No PESMaker task needs to run now.")
        print(f"Reason           : {event.message}")
    else:
        print(f"Start with       : {event.message}")
        print("Stop rule        : stop before real scheduler submission")
    print()


def submit_action_label(stage: str) -> str:
    """Return the user-facing label for a submit stage."""
    if stage == "sampling":
        return "Submit sampling jobs"
    if stage == "training":
        return "Submit training jobs"
    return "Submit SCF jobs"


def _print_run_event(event: NextEvent) -> None:
    print(f"  - {event.message}")
    if event.result is not None:
        if not _hide_output_dir(event):
            print(f"Output directory : {event.result.output_dir}")
        _print_result_warnings(event.result)


def _hide_output_dir(event: NextEvent) -> bool:
    return _event_stage(event) == "collect" and event.result.output_dir == Path(".")


def _run_events_include_output_dir(
    run_events: list[NextEvent],
    boundary_event: NextEvent,
) -> bool:
    if boundary_event.result is None:
        return False
    return any(
        event.result is not None
        and event.result.output_dir == boundary_event.result.output_dir
        for event in run_events
    )


def _print_result_warnings(result, *, prefix: str = "") -> None:
    for warning in result.warnings[:5]:
        print(f"{prefix}Warning        : {warning}")
    omitted = len(result.warnings) - 5
    if omitted > 0:
        print(f"{prefix}Warning        : ... {omitted} more warning(s)")


def _print_submit_preview(
    event: NextEvent,
    *,
    config_path: Path,
    show_output_dir: bool = True,
) -> None:
    stage = _event_stage(event)
    if show_output_dir and event.result is not None:
        print(f"Output directory : {event.result.output_dir}")
    if event.log_path is not None:
        print(f"Dry-run log      : {event.log_path}")
    print()
    print("Next:")
    if event.log_path is not None:
        print(f"  1. Review: {event.log_path}")
    if event.command:
        print(f"  2. {submit_action_label(stage)}: {event.command}")
    print(f"  3. After jobs finish: pesmaker next {config_path}")
    print()


def _print_config_needed(event: NextEvent, *, config_path: Path) -> None:
    template_path = _template_path(event, config_path)
    if event.template_created:
        print(f"Template written : {template_path}")
    else:
        print(f"Template exists  : {template_path}")
    print()
    print("Next:")
    print(
        f"  1. Edit {template_path} and set INCAR, POTCAR, VASP, "
        "and submit script paths."
    )
    print(f"  2. Run: pesmaker validate {template_path}")
    print(f"  3. Run: pesmaker next {template_path}")
    print()


def _print_training_config_needed(event: NextEvent, *, config_path: Path) -> None:
    print("Next:")
    print(f"  - {event.message}")
    _print_training_config_steps(config_path=config_path)
    print()


def _print_training_config_steps(*, config_path: Path) -> None:
    print("  1. Add a `training` section to the YAML.")
    print(f"  2. Run: pesmaker validate {config_path}")
    print(f"  3. Run: pesmaker next {config_path}")


def _print_waiting(event: NextEvent, *, config_path: Path) -> None:
    stage = _event_stage(event)
    print("Waiting:")
    print(f"  - {_waiting_message(stage)}")
    print()
    print("Next:")
    if event.command:
        print(f"  1. If not submitted yet: {event.command}")
        print(f"  2. After jobs finish: pesmaker next {config_path}")
    else:
        print("  1. Wait until the required files exist.")
        print(f"  2. Run: pesmaker next {config_path}")
    print()


def _print_scf_retry(event: NextEvent, *, config_path: Path) -> None:
    print("SCF retry submission:")
    print(f"  - {event.message}")
    print("  - PESMaker will not run scf-setup for these existing folders.")
    print()
    print("Next:")
    _print_scf_retry_steps(event, config_path=config_path)
    print()


def _print_scf_retry_steps(event: NextEvent, *, config_path: Path) -> None:
    command = event.command or f"pesmaker submit {config_path}"
    log_path = event.log_path or Path("labeling") / "scf_submitted_jobs.txt"
    print(f"  1. Preview and refresh retry scripts: {command} --dry-run")
    print(f"  2. Review: cat {log_path}")
    print(f"  3. Submit retry jobs: {command}")


def _waiting_message(stage: str) -> str:
    if stage == "selection":
        return "Trajectory files are not ready."
    if stage == "sampling":
        return "Sampling trajectory files are not ready."
    if stage == "training":
        return "Training job output is not ready."
    return "SCF OUTCAR files are not ready."


def _current_next_label(event: NextEvent | None, status: str) -> str:
    if event is None:
        return status
    step_kind = _next_action_kind(event)
    if step_kind == "run":
        return event.message
    if step_kind == "config-needed":
        return "waiting for SCF settings"
    if step_kind == "training-config-needed":
        return "waiting for training settings"
    if step_kind == "scf-retry":
        return "SCF retry submission"
    if step_kind == "submit-preview":
        return f"{_stage_display(_event_stage(event))} submission preview"
    if step_kind in {"wait", "waiting"}:
        return f"waiting for {_stage_display(_event_stage(event))} outputs"
    if step_kind == "complete":
        return "complete"
    return event.message


def _stage_display(stage: str) -> str:
    if stage == "selection":
        return "selection"
    if stage == "sampling":
        return "MD-sampling"
    if stage == "training":
        return "training"
    return "SCF"


def _template_path(event: NextEvent, config_path: Path) -> Path:
    if event.template_path is not None:
        return event.template_path
    suffix = config_path.suffix or ".yaml"
    return config_path.with_name(f"{config_path.stem}.next{suffix}")


def _next_action_kind(event) -> str:
    if event is None:
        return ""
    prefix = "next-action:"
    if isinstance(event.kind, str) and event.kind.startswith(prefix):
        return event.kind[len(prefix) :]
    return str(event.kind)


def _event_stage(event: NextEvent) -> str:
    if event.stage:
        return event.stage
    if event.command and "--stage sampling" in event.command:
        return "sampling"
    if event.command and "--stage training" in event.command:
        return "training"
    return "scf"
