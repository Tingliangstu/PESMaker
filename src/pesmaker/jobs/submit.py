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

import shlex
import subprocess
from pathlib import Path

from pesmaker.artifacts import _read_manifest, _section_output_dir
from pesmaker.config.schema import PESMakerConfig
from pesmaker.results import StageResult


def submit_jobs(
    config: PESMakerConfig,
    *,
    stage: str = "scf",
    dry_run: bool = False,
) -> StageResult:
    """Submit prepared stage jobs with the configured scheduler command."""
    submit_scripts = _stage_submit_scripts(config, stage)
    if not submit_scripts:
        raise ValueError(f"no submit.sh scripts found for stage: {stage}")

    submit_command = str(config.jobs.options.get("submit_command", "sbatch"))
    output_dir = _stage_output_dir(config, stage)
    output_dir.mkdir(parents=True, exist_ok=True)
    submitted_log = output_dir / f"{stage}_submitted_jobs.txt"
    lines: list[str] = []
    for script in submit_scripts:
        command = [*shlex.split(submit_command), script.name]
        display = f"(cd {script.parent} && {' '.join(command)})"
        if dry_run:
            lines.append(f"DRY-RUN {display}")
            continue
        result = subprocess.run(
            command,
            cwd=script.parent,
            check=True,
            capture_output=True,
            text=True,
        )
        message = result.stdout.strip() or result.stderr.strip()
        lines.append(f"{script.parent}: {message}")
    submitted_log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    action = "Would submit" if dry_run else "Submitted"
    return StageResult(
        output_dir,
        (submitted_log,),
        f"{action} {len(submit_scripts)} {stage} job(s)",
    )


def _stage_submit_scripts(config: PESMakerConfig, stage: str) -> list[Path]:
    manifest_name = _stage_manifest_name(stage)
    output_dir = _stage_output_dir(config, stage)
    manifest_path = output_dir / manifest_name
    if manifest_path.exists():
        scripts = []
        for record in _read_manifest(manifest_path):
            workdir = record.get("workdir")
            if workdir:
                script = Path(str(workdir)) / "submit.sh"
                if script.exists():
                    scripts.append(script)
        if scripts:
            return scripts
    return sorted(output_dir.rglob("submit.sh"))


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
