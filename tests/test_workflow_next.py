# Copyright 2026 Ting Liang and PESMaker development team
# This file is part of PESMaker.
#
# PESMaker is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# PESMaker is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PESMaker. If not, see <https://www.gnu.org/licenses/>.
"""Tests for the smart-next workflow entry point."""

from __future__ import annotations

import json

from pesmaker.cli import main
from pesmaker.results import StageResult
from pesmaker.workflow.stages import submit_jobs


def test_next_direct_scf_generates_labels_and_previews_submit(
    tmp_path,
    monkeypatch,
    capsys,
):
    """`next` should advance direct SCF workflows to the SCF dry-run gate."""
    structure = _write_structure(tmp_path)
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        f"""project: direct_next
workflow: direct-scf
structures:
  - {structure.as_posix()}
generation:
  output_dir: generated
labeling:
  output_dir: labeling
jobs:
  submit_command: sbatch
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    assert main(["next", str(config_path)]) == 0
    output = capsys.readouterr().out

    assert (tmp_path / "generated" / "manifest.jsonl").exists()
    assert (tmp_path / "labeling" / "labeling_manifest.jsonl").exists()
    assert (tmp_path / "labeling" / "scf_submitted_jobs.txt").exists()
    state_path = tmp_path / ".pesmaker" / "direct_next" / "next_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert "scf" in state["dry_runs"]
    assert "Workflow mode    : direct-scf" in output
    assert "Submission preview complete." in output
    assert f"Submit jobs      : pesmaker submit {config_path}" in output

    assert main(["next", str(config_path)]) == 0
    output = capsys.readouterr().out
    assert "Waiting for SCF outputs matching" in output
    assert "Submission preview complete." not in output


def test_next_sampling_training_previews_sampling_then_waits(
    tmp_path,
    monkeypatch,
    capsys,
):
    """Sampling workflows should stop at MD submission before selection."""
    structure = _write_structure(tmp_path)
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        f"""project: sampling_next
workflow: sampling-training
structures:
  - {structure.as_posix()}
generation:
  output_dir: generated
sampling:
  engine: gpumd
  output_dir: sampling
  temperatures: [300]
  selection:
    trajectory_pattern: sampling/**/movie.xyz
    output_dir: selected
    descriptor: simple
labeling:
  output_dir: labeling
jobs:
  submit_command: sbatch
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    assert main(["next", str(config_path)]) == 0
    output = capsys.readouterr().out

    assert (tmp_path / "generated" / "manifest.jsonl").exists()
    assert (tmp_path / "sampling" / "sampling_manifest.jsonl").exists()
    assert (tmp_path / "sampling" / "sampling_submitted_jobs.txt").exists()
    assert not (tmp_path / "selected" / "manifest.jsonl").exists()
    state_path = tmp_path / ".pesmaker" / "sampling_next" / "next_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert "sampling" in state["dry_runs"]
    assert "Workflow mode    : sampling-training" in output
    assert "Submit jobs      : pesmaker submit" in output
    assert "--stage sampling" in output

    assert main(["next", str(config_path)]) == 0
    output = capsys.readouterr().out
    assert "Waiting for sampling trajectories matching sampling/**/movie.xyz" in output
    assert "--stage sampling" in output

    from ase import Atoms
    from ase.io import write

    movie = tmp_path / "sampling" / "md_000000_temp_300K" / "movie.xyz"
    write(
        movie,
        [
            Atoms("Te", positions=[(0.0, 0.0, 0.0)], cell=[3, 3, 20], pbc=True),
            Atoms("Te", positions=[(1.0, 0.0, 0.0)], cell=[3, 3, 20], pbc=True),
        ],
        format="extxyz",
    )

    assert main(["next", str(config_path)]) == 0
    output = capsys.readouterr().out
    assert (tmp_path / "selected" / "manifest.jsonl").exists()
    assert (tmp_path / "labeling" / "labeling_manifest.jsonl").exists()
    records = [
        json.loads(line)
        for line in (tmp_path / "labeling" / "labeling_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert all("selected" in record["source"] for record in records)
    assert "Submit jobs      : pesmaker submit" in output


def test_old_workflow_stage_imports_still_work():
    """Legacy workflow imports should remain available after the split."""
    assert callable(submit_jobs)
    assert StageResult.__name__ == "StageResult"


def _write_structure(tmp_path):
    from ase import Atoms
    from ase.io import write

    structure = tmp_path / "structure.xyz"
    write(
        structure,
        Atoms("Te", positions=[(0.0, 0.0, 0.0)], cell=[3, 3, 20], pbc=True),
        format="extxyz",
    )
    return structure
