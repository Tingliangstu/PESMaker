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

"""Submit prepared stage job scripts through the configured scheduler."""

from __future__ import annotations

from dataclasses import dataclass
import shlex
import subprocess
from pathlib import Path

from pesmaker.artifacts import _read_manifest, _section_output_dir
from pesmaker.config.schema import PESMakerConfig
from pesmaker.jobs.resources import _job_resources
from pesmaker.jobs.scripts import SUBMIT_RESOURCE_KEYS, _write_submit_script
from pesmaker.results import StageResult, SubmissionSummary
from pesmaker.structures import load_structure

VASP_COMPLETION_MARKER = (
    b"General timing and accounting informations for this job"
)
VASP_SCF_NOT_CONVERGED_MARKER = (
    b"The electronic self-consistency was not achieved in"
)


@dataclass(frozen=True)
class VaspOutcarState:
    """Completion and electronic-convergence state read from one OUTCAR."""

    exists: bool
    terminated: bool
    scf_converged: bool


@dataclass(frozen=True)
class PreparedJob:
    """One prepared scheduler job and its existing script, when known."""

    workdir: Path
    submit_script: Path | None = None


def submit_jobs(
    config: PESMakerConfig,
    *,
    stage: str = "scf",
    dry_run: bool = False,
) -> StageResult:
    """Submit prepared stage jobs with the configured scheduler command."""
    jobs = _stage_prepared_jobs(config, stage)
    if not jobs:
        raise ValueError(f"no prepared job folders found for stage: {stage}")

    submit_command = str(config.jobs.options.get("submit_command", "sbatch"))
    output_dir = _stage_output_dir(config, stage)
    output_dir.mkdir(parents=True, exist_ok=True)
    submitted_log = output_dir / f"{stage}_submitted_jobs.txt"
    lines: list[str] = []
    submitted_count = 0
    skipped_count = 0
    skip_completed = _skip_completed_jobs(config, stage)
    check_scf_convergence = _check_scf_convergence(config, stage)
    refresh_scripts = _refresh_vasp_submit_scripts(config, stage, skip_completed)
    for job in jobs:
        workdir = job.workdir
        if skip_completed:
            outcar_state = _vasp_outcar_state(workdir)
            if _vasp_job_is_complete(
                outcar_state,
                check_scf_convergence=check_scf_convergence,
            ):
                lines.append(f"SKIPPED completed VASP job: {workdir}")
                skipped_count += 1
                continue
            retry_reason = _vasp_retry_reason(outcar_state)
            if retry_reason:
                lines.append(f"RETRY {retry_reason}: {workdir}")

        if refresh_scripts:
            script = _refresh_vasp_submit_script(config, workdir)
            lines.append(f"REFRESHED submit script: {script}")
        else:
            script = _existing_submit_script(job)
            if script is None:
                raise ValueError(f"no submit script found in job folder: {workdir}")

        display = _submit_display(submit_command, script)
        if dry_run:
            lines.append(f"DRY-RUN {display}")
            submitted_count += 1
            continue
        message = _run_submit_command(submit_command, script)
        lines.append(f"{script.parent}: {message}")
        submitted_count += 1
    submitted_log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    action = "Would submit" if dry_run else "Submitted"
    message = f"{action} {submitted_count} {stage} job(s)"
    if skipped_count:
        message += f"; skipped {skipped_count} completed VASP job(s)"
    return StageResult(
        output_dir,
        (submitted_log,),
        message,
        submission=SubmissionSummary(
            total_jobs=len(jobs),
            completed_jobs=skipped_count,
            pending_jobs=submitted_count,
        ),
    )


def _skip_completed_jobs(config: PESMakerConfig, stage: str) -> bool:
    """Return whether completed VASP SCF folders should be skipped."""
    if stage != "scf" or config.labeling.engine.strip().lower() != "vasp":
        return False
    value = config.jobs.options.get("skip_completed", True)
    if not isinstance(value, bool):
        raise ValueError("jobs.skip_completed must be true or false")
    return value


def _check_scf_convergence(config: PESMakerConfig, stage: str) -> bool:
    """Return whether electronic SCF convergence is required before skipping."""
    if stage != "scf" or config.labeling.engine.strip().lower() != "vasp":
        return False
    value = config.jobs.options.get("check_scf_convergence", True)
    if not isinstance(value, bool):
        raise ValueError("jobs.check_scf_convergence must be true or false")
    return value


def _refresh_vasp_submit_scripts(
    config: PESMakerConfig,
    stage: str,
    skip_completed: bool,
) -> bool:
    """Refresh retry scripts when completed VASP filtering is enabled."""
    return (
        skip_completed
        and stage == "scf"
        and config.labeling.engine.strip().lower() == "vasp"
        and _has_submit_refresh_options(config)
    )


def _has_submit_refresh_options(config: PESMakerConfig) -> bool:
    return (
        "command" in config.labeling.options
        or any(
            key in config.jobs.options
            for key in (
                "sub_file",
                "sbatch_template",
                "sbatch_templates",
                *SUBMIT_RESOURCE_KEYS,
            )
        )
    )


def _vasp_outcar_state(workdir: Path) -> VaspOutcarState:
    """Read normal-termination and electronic-convergence markers."""
    outcar = workdir / "OUTCAR"
    if not outcar.is_file():
        return VaspOutcarState(
            exists=False,
            terminated=False,
            scf_converged=False,
        )
    terminated = False
    scf_converged = True
    try:
        with outcar.open("rb") as handle:
            for line in handle:
                if VASP_COMPLETION_MARKER in line:
                    terminated = True
                if VASP_SCF_NOT_CONVERGED_MARKER in line:
                    scf_converged = False
    except OSError:
        return VaspOutcarState(
            exists=True,
            terminated=False,
            scf_converged=False,
        )
    return VaspOutcarState(
        exists=True,
        terminated=terminated,
        scf_converged=scf_converged,
    )


def _vasp_job_is_complete(
    state: VaspOutcarState,
    *,
    check_scf_convergence: bool,
) -> bool:
    """Return whether a VASP result is safe to skip during submission."""
    return state.terminated and (
        state.scf_converged or not check_scf_convergence
    )


def _vasp_retry_reason(state: VaspOutcarState) -> str | None:
    if not state.exists:
        return None
    if not state.scf_converged:
        return "electronic SCF not converged"
    if not state.terminated:
        return "incomplete VASP output"
    return None


def _refresh_vasp_submit_script(
    config: PESMakerConfig,
    workdir: Path,
) -> Path:
    """Render a migrated or retry VASP job with the current machine settings."""
    atom_count = None
    poscar = workdir / "POSCAR"
    if poscar.is_file():
        try:
            atom_count = len(load_structure(poscar))
        except Exception:
            atom_count = None
    return _write_submit_script(
        config,
        workdir,
        stage="labeling",
        command=str(config.labeling.options.get("command", "vasp_std")),
        resources=_job_resources(config, atom_count=atom_count),
    )


def _submit_display(submit_command: str, script: Path) -> str:
    if _is_nohup_submit(submit_command):
        return f"(cd {script.parent} && nohup bash {script.name} > out 2>&1 &)"
    command = [*shlex.split(submit_command), script.name]
    return f"(cd {script.parent} && {' '.join(command)})"


def _run_submit_command(submit_command: str, script: Path) -> str:
    if _is_nohup_submit(submit_command):
        log_path = script.parent / "out"
        with log_path.open("ab") as output:
            process = subprocess.Popen(
                ["nohup", "bash", script.name],
                cwd=script.parent,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        return f"started PID {process.pid}; log: {log_path.name}"

    command = [*shlex.split(submit_command), script.name]
    result = subprocess.run(
        command,
        cwd=script.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or result.stderr.strip()


def _is_nohup_submit(submit_command: str) -> bool:
    return shlex.split(submit_command) == ["nohup"]


def _stage_prepared_jobs(
    config: PESMakerConfig,
    stage: str,
) -> list[PreparedJob]:
    manifest_name = _stage_manifest_name(stage)
    output_dir = _stage_output_dir(config, stage)
    manifest_path = output_dir / manifest_name
    jobs: list[PreparedJob] = []
    if manifest_path.exists():
        for record in _read_manifest(manifest_path):
            submit_script = record.get("submit_script")
            if submit_script:
                script = Path(str(submit_script))
                if script.parent.is_dir():
                    jobs.append(
                        PreparedJob(
                            workdir=script.parent,
                            submit_script=script if script.is_file() else None,
                        )
                    )
                    continue
            workdir = record.get("workdir")
            if workdir:
                path = Path(str(workdir))
                if path.is_dir():
                    jobs.append(
                        PreparedJob(
                            workdir=path,
                            submit_script=_submit_sh_if_present(path),
                        )
                    )

    if stage == "training" and manifest_path.exists():
        return _unique_jobs(jobs)

    if stage == "scf" and config.labeling.engine.strip().lower() == "vasp":
        jobs.extend(
            PreparedJob(
                workdir=path.parent,
                submit_script=_submit_sh_if_present(path.parent),
            )
            for path in output_dir.rglob("POSCAR")
        )
    jobs.extend(
        PreparedJob(workdir=path.parent, submit_script=path)
        for path in output_dir.rglob("submit.sh")
    )
    return _unique_jobs(jobs)


def _unique_jobs(jobs: list[PreparedJob]) -> list[PreparedJob]:
    unique: list[PreparedJob] = []
    seen: set[Path] = set()
    for job in jobs:
        resolved = job.workdir.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(job)
    return unique


def _submit_sh_if_present(workdir: Path) -> Path | None:
    script = workdir / "submit.sh"
    return script if script.is_file() else None


def _existing_submit_script(job: PreparedJob) -> Path | None:
    if job.submit_script is not None and job.submit_script.is_file():
        return job.submit_script
    return _submit_sh_if_present(job.workdir)


def _stage_output_dir(config: PESMakerConfig, stage: str) -> Path:
    if stage == "sampling":
        return _section_output_dir(config, config.sampling.options, "sampling")
    if stage == "scf":
        return _section_output_dir(config, config.labeling.options, "labeling")
    if stage == "training":
        return _section_output_dir(config, config.training.options, "training")
    raise ValueError("stage must be one of: sampling, scf, training")


def _stage_manifest_name(stage: str) -> str:
    if stage == "scf":
        return "labeling_manifest.jsonl"
    return f"{stage}_manifest.jsonl"
