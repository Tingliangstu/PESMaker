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

"""Submission script rendering helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pesmaker.config.schema import PESMakerConfig
from pesmaker.jobs.resources import JobResources, _job_resources

SUBMIT_RESOURCE_KEYS = {
    "nodes",
    "cores_cpu",
    "gpus",
    "gpus_gpu",
    "vasp_kpar",
    "vasp_ncore",
}


def _write_submit_script(
    config: PESMakerConfig,
    workdir: Path,
    *,
    stage: str,
    command: str,
    resources: JobResources | None = None,
) -> Path:
    template_path = _job_template_path(config, stage)
    job_name = workdir.name
    resources = resources or _job_resources(config)
    engine = _stage_engine(config, stage)
    mpi_ranks = _requested_mpi_ranks(config, resources)
    if template_path:
        ntasks = resources.nodes * resources.cores_cpu
        use_resource_command = _should_use_resource_command(config, stage, engine)
        run_command = (
            _default_run_command(
                command,
                stage=stage,
                engine=engine,
                resources=resources,
                mpi_ranks=mpi_ranks,
            )
            if use_resource_command
            else command
        )
        text = _format_submit_template(
            template_path.read_text(encoding="utf-8"),
            {
                "command": run_command,
                "job_name": job_name,
                "workdir": str(workdir),
                "nodes": resources.nodes,
                "ntasks": ntasks,
                "cores_cpu": resources.cores_cpu,
                "ntasks_per_node": resources.cores_cpu,
                "gpus": resources.gpus,
                "vasp_kpar": resources.vasp_kpar,
                "vasp_ncore": resources.vasp_ncore,
            },
        )
    else:
        text = _default_submit_script(
            command=command,
            job_name=job_name,
            stage=stage,
            engine=engine,
            resources=resources,
            mpi_ranks=mpi_ranks,
        )
    path = _submit_script_path(workdir, template_path, stage=stage, engine=engine)
    path.write_text(text, encoding="utf-8")
    compatibility_path = workdir / "submit.sh"
    if path != compatibility_path:
        compatibility_path.write_text(text, encoding="utf-8")
    return path


def _job_template_path(config: PESMakerConfig, stage: str) -> Path | None:
    sub_file = config.jobs.options.get("sub_file")
    if isinstance(sub_file, dict):
        stage_template = _stage_template_value(sub_file, stage)
        if stage_template:
            return Path(str(stage_template))
    if sub_file and not isinstance(sub_file, dict):
        return Path(str(sub_file))

    templates = config.jobs.options.get("sbatch_templates", {})
    if isinstance(templates, dict):
        stage_template = _stage_template_value(templates, stage)
        if stage_template:
            return Path(str(stage_template))
    template = config.jobs.options.get("sbatch_template")
    return Path(str(template)) if template else None


def _preserve_user_submit_template(stage: str, engine: str) -> bool:
    """Return true when PESMaker should keep sampling templates literal."""
    normalized_engine = engine.lower().replace("_", "-")
    return stage == "sampling" and normalized_engine in {
        "gpumd",
        "mace",
        "lammps-mace",
    }


def _should_use_resource_command(
    config: PESMakerConfig,
    stage: str,
    engine: str,
) -> bool:
    if _preserve_user_submit_template(stage, engine):
        return False
    return any(key in config.jobs.options for key in SUBMIT_RESOURCE_KEYS)


def _requested_mpi_ranks(
    config: PESMakerConfig,
    resources: JobResources,
) -> int | None:
    options = config.jobs.options
    if resources.gpus and any(key in options for key in ("gpus", "gpus_gpu")):
        return resources.gpus
    if any(key in options for key in ("cores_cpu", "nodes")):
        return resources.nodes * resources.cores_cpu
    return None


def _submit_script_path(
    workdir: Path,
    template_path: Path | None,
    *,
    stage: str,
    engine: str,
) -> Path:
    if template_path is not None and _preserve_user_submit_template(stage, engine):
        return workdir / template_path.name
    return workdir / "submit.sh"


def _stage_template_value(templates: dict[str, Any], stage: str) -> Any:
    if templates.get(stage):
        return templates[stage]
    if stage == "labeling" and templates.get("scf"):
        return templates["scf"]
    return None


def _stage_engine(config: PESMakerConfig, stage: str) -> str:
    if stage == "sampling":
        return config.sampling.engine
    if stage == "labeling":
        return config.labeling.engine
    if stage == "training":
        return config.training.engine
    return stage


def _format_submit_template(text: str, values: dict[str, object]) -> str:
    for key, value in values.items():
        text = text.replace(f"{{{key}}}", str(value))
    return text


def _default_submit_script(
    *,
    command: str,
    job_name: str,
    stage: str,
    engine: str,
    resources: JobResources,
    mpi_ranks: int | None = None,
) -> str:
    if _preserve_user_submit_template(stage, engine):
        return _default_sampling_submit_script(command)

    ntasks = resources.nodes * resources.cores_cpu
    lines = [
        "#!/bin/bash -l",
        f"#SBATCH --job-name={job_name}",
        "#SBATCH --output=out.%j",
        "#SBATCH --error=err.%j",
        f"#SBATCH --nodes={resources.nodes}",
        f"#SBATCH --ntasks={ntasks}",
        "#SBATCH --cpus-per-task=1",
    ]
    if resources.gpus:
        lines.append(f"#SBATCH --gres=gpu:{resources.gpus}")
    lines.extend(
        [
            "",
            "set -euo pipefail",
            "",
            "export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}",
            "ulimit -s unlimited",
            "",
            'echo "--------------------------------"',
            'echo "Job started at $(date)"',
            'echo "Running on node(s): ${SLURM_NODELIST:-unknown}"',
            'echo "Using total tasks: ${SLURM_NTASKS:-unknown}"',
            'echo "Working directory: $(pwd)"',
            'echo "--------------------------------"',
            "",
            _default_run_command(
                command,
                stage=stage,
                engine=engine,
                resources=resources,
                mpi_ranks=mpi_ranks,
            ),
            "",
            'echo "Simulation finished at $(date)"',
            "",
        ]
    )
    return "\n".join(lines)


def _default_sampling_submit_script(command: str) -> str:
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        command,
        "",
    ]
    return "\n".join(lines)


def _default_run_command(
    command: str,
    *,
    stage: str,
    engine: str,
    resources: JobResources,
    mpi_ranks: int | None = None,
) -> str:
    if stage == "labeling" and engine.lower() == "vasp":
        if mpi_ranks is not None:
            return _with_mpi_ranks(command, mpi_ranks)
        if _has_mpi_launcher(command):
            return command
        return f"mpirun {command}"
    return command


def _has_mpi_launcher(command: str) -> bool:
    command_start = command.strip().lower()
    return any(
        command_start == launcher or command_start.startswith(f"{launcher} ")
        for launcher in ("mpirun", "mpiexec", "srun")
    )


def _with_mpi_ranks(command: str, mpi_ranks: int) -> str:
    prefix = command[: len(command) - len(command.lstrip())]
    stripped = command.strip()
    launcher, separator, rest = stripped.partition(" ")
    launcher_lower = launcher.lower()
    if launcher_lower in {"mpirun", "mpiexec"}:
        if _mpi_command_has_rank_count(rest):
            return command
        suffix = f"{separator}{rest}" if rest else ""
        return f"{prefix}{launcher} -np {mpi_ranks}{suffix}"
    if launcher_lower == "srun":
        return command
    return f"mpirun -np {mpi_ranks} {command}"


def _mpi_command_has_rank_count(command: str) -> bool:
    rank_options = {"-np", "-n", "--np", "--ntasks"}
    return any(
        token in rank_options
        or token.startswith(("-np=", "-n=", "--np=", "--ntasks="))
        for token in command.split()
    )
