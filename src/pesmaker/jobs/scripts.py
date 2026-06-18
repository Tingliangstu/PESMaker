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

"""Submission script rendering and normalization helpers."""

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
    if template_path:
        ntasks = resources.nodes * resources.cores_cpu
        run_command = _default_run_command(
            command,
            stage=stage,
            engine=engine,
            resources=resources,
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
        if _should_normalize_submit_template(config, stage, engine):
            text = _normalize_submit_template(
                text,
                command=command,
                job_name=job_name,
                workdir=workdir,
                stage=stage,
                engine=engine,
                resources=resources,
            )
        elif not text.endswith("\n"):
            text += "\n"
    else:
        text = _default_submit_script(
            command=command,
            job_name=job_name,
            stage=stage,
            engine=engine,
            resources=resources,
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
    """Return true when PESMaker should not rewrite scheduler resources."""
    normalized_engine = engine.lower().replace("_", "-")
    return stage == "sampling" and normalized_engine in {
        "gpumd",
        "mace",
        "lammps-mace",
    }


def _should_normalize_submit_template(
    config: PESMakerConfig,
    stage: str,
    engine: str,
) -> bool:
    if _preserve_user_submit_template(stage, engine):
        return False
    return any(key in config.jobs.options for key in SUBMIT_RESOURCE_KEYS)


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


def _normalize_submit_template(
    text: str,
    *,
    command: str,
    job_name: str,
    workdir: Path,
    stage: str,
    engine: str,
    resources: JobResources,
) -> str:
    ntasks = resources.nodes * resources.cores_cpu
    lines: list[str] = []
    for line in text.splitlines():
        if _is_generated_workdir_cd(line, workdir):
            continue
        updated = _set_sbatch_directive(line, "--job-name", job_name)
        if updated is None:
            updated = _set_sbatch_directive(line, "--ntasks", str(ntasks))
        if updated is None:
            updated = _set_sbatch_directive(
                line,
                "--ntasks-per-node",
                str(resources.cores_cpu),
            )
        if updated is None:
            updated = _replace_vasp_run_command(
                line,
                command=command,
                stage=stage,
                engine=engine,
                resources=resources,
            )
        lines.append(updated if updated is not None else line)
    return "\n".join(lines) + "\n"


def _set_sbatch_directive(line: str, option: str, value: str) -> str | None:
    prefix = line[: len(line) - len(line.lstrip())]
    stripped = line.lstrip()
    if not stripped.startswith("#SBATCH"):
        return None
    rest = stripped[len("#SBATCH") :].lstrip()
    if rest.startswith(f"{option}="):
        suffix = _directive_suffix(rest[len(option) + 1 :])
        return f"{prefix}#SBATCH {option}={value}{suffix}"
    if rest == option or rest.startswith(f"{option} "):
        suffix = _directive_suffix(rest[len(option) :].lstrip())
        return f"{prefix}#SBATCH {option}={value}{suffix}"
    return None


def _directive_suffix(value_text: str) -> str:
    comment_index = value_text.find(" #")
    if comment_index >= 0:
        return value_text[comment_index:]
    return ""


def _is_generated_workdir_cd(line: str, workdir: Path) -> bool:
    stripped = line.strip()
    workdir_text = str(workdir)
    return stripped in {
        f'cd "{workdir_text}"',
        f"cd '{workdir_text}'",
        f"cd {workdir_text}",
    }


def _replace_vasp_run_command(
    line: str,
    *,
    command: str,
    stage: str,
    engine: str,
    resources: JobResources,
) -> str | None:
    if stage != "labeling" or engine.lower() != "vasp":
        return None
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    lower = stripped.lower()
    if lower.startswith(("echo ", "export ", "module ", "source ", "ulimit ", "set ")):
        return None
    if "vasp" not in lower:
        return None
    prefix = line[: len(line) - len(line.lstrip())]
    return prefix + _default_run_command(
        command,
        stage=stage,
        engine=engine,
        resources=resources,
    )


def _default_submit_script(
    *,
    command: str,
    job_name: str,
    stage: str,
    engine: str,
    resources: JobResources,
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
                command, stage=stage, engine=engine, resources=resources
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
) -> str:
    if stage == "labeling" and engine.lower() == "vasp":
        if _has_mpi_launcher(command):
            return command
        if resources.gpus:
            return f"mpirun -np {resources.gpus} {command}"
        return f"mpirun {command}"
    return command


def _has_mpi_launcher(command: str) -> bool:
    command_start = command.strip().lower()
    return any(
        command_start == launcher or command_start.startswith(f"{launcher} ")
        for launcher in ("mpirun", "mpiexec", "srun")
    )
