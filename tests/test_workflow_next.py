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
    assert "Potential Energy Surface Maker" in output
    assert "Next flow" in output
    assert "Flow             : generate -> SCF-labeling -> dataset-collect" in output
    assert "Current          : SCF submission preview" in output
    assert "Work done:" in output
    assert "Structure generation complete." in output
    assert "Prepared 1 SCF job(s)" in output
    assert "Plan before execution" not in output
    assert "Inferred flow" not in output
    assert "Submit SCF jobs" in output
    assert f"pesmaker submit {config_path}" in output
    assert "pesmaker generate" not in output

    assert main(["next", str(config_path), "--verbose"]) == 0
    output = capsys.readouterr().out
    assert "Plan before execution" in output
    assert "Inferred flow    : generate -> SCF-labeling -> dataset-collect" in output
    assert "Smart next" in output
    assert "Status           : waiting" in output
    assert "State            :" in output
    assert "No local stage will run now." in output
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
    assert "Next flow" in output
    assert (
        "Flow             : generate -> MD-sampling -> FPS-select -> "
        "SCF-labeling -> dataset-collect"
    ) in output
    assert "Current          : MD-sampling submission preview" in output
    assert "Work done:" in output
    assert "Plan before execution" not in output
    assert "Inferred flow" not in output
    assert "Submit GPUMD sampling jobs" in output
    assert "pesmaker submit" in output
    assert "--stage sampling" in output

    assert main(["next", str(config_path)]) == 0
    output = capsys.readouterr().out
    assert "Current          : waiting for MD-sampling outputs" in output
    assert "Waiting:" in output
    assert "GPUMD movie.xyz files are not ready." in output
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
    assert "Submit SCF jobs" in output
    assert "pesmaker submit" in output


def test_next_sampling_selection_without_labeling_writes_scf_template(
    tmp_path,
    monkeypatch,
    capsys,
):
    """Selected MD frames should lead to a follow-up SCF config template."""
    structure = _write_structure(tmp_path)
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        f"""project: selected_needs_scf
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
    min_distance: 0.2
jobs:
  submit_command: sbatch
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    assert main(["next", str(config_path)]) == 0
    capsys.readouterr()

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
    followup = tmp_path / "run.next.yaml"
    followup_text = followup.read_text(encoding="utf-8")

    assert (tmp_path / "selected" / "manifest.jsonl").exists()
    assert followup.exists()
    assert not (tmp_path / "labeling" / "labeling_manifest.jsonl").exists()
    assert (
        "Flow             : generate -> MD-sampling -> FPS-select -> "
        "SCF-config-needed"
    ) in output
    assert "Current          : waiting for SCF settings" in output
    assert "Selected 2 of 2 MD frame(s)" in output
    assert f"Template written : {followup}" in output
    assert "input_manifest: selected/manifest.jsonl" in followup_text
    assert "input_dir: generated" not in followup_text
    assert f"Run: pesmaker validate {followup}" in output
    assert f"Run: pesmaker next {followup}" in output


def test_next_generation_only_writes_followup_scf_template(
    tmp_path,
    monkeypatch,
    capsys,
):
    """Generation-only configs should stop and ask for follow-up SCF settings."""
    structure = _write_structure(tmp_path)
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        f"""project: generation_only
structures:
  - {structure.as_posix()}
generation:
  output_dir: generated
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    assert main(["next", str(config_path)]) == 0
    output = capsys.readouterr().out
    followup = tmp_path / "run.next.yaml"
    followup_text = followup.read_text(encoding="utf-8")

    assert (tmp_path / "generated" / "manifest.jsonl").exists()
    assert followup.exists()
    assert not (tmp_path / "labeling" / "labeling_manifest.jsonl").exists()
    assert not (tmp_path / ".pesmaker").exists()
    assert "Next flow" in output
    assert "Flow             : generate -> SCF-config-needed" in output
    assert "Current          : waiting for SCF settings" in output
    assert "Work done:" in output
    assert "Plan before execution" not in output
    assert "Inferred flow" not in output
    assert f"Template written : {followup}" in output
    assert f"Run: pesmaker validate {followup}" in output
    assert f"Run: pesmaker next {followup}" in output
    assert "dataset_path" not in followup_text
    assert "input_dir: generated" in followup_text
    assert "output_dir: run_vasp_scf" in followup_text
    assert "vasp_kpar: 3" in followup_text
    assert "vasp_ncore: 6" in followup_text

    followup.write_text("custom: keep\n", encoding="utf-8")
    assert main(["next", str(config_path)]) == 0
    output = capsys.readouterr().out

    assert followup.read_text(encoding="utf-8") == "custom: keep\n"
    assert f"Template exists  : {followup}" in output

    assert main(["status", str(config_path)]) == 0
    output = capsys.readouterr().out
    assert f"Edit the generated template: {followup}" in output
    assert "Submit SCF jobs" not in output


def test_status_reports_next_action_without_writing_files(tmp_path, monkeypatch, capsys):
    """`status` should inspect the inferred flow without running it."""
    structure = _write_structure(tmp_path)
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        f"""project: status_next
structures:
  - {structure.as_posix()}
generation:
  output_dir: generated
labeling:
  output_dir: labeling
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    assert main(["status", str(config_path)]) == 0
    output = capsys.readouterr().out

    assert "Inferred flow    : generate -> SCF-labeling -> dataset-collect" in output
    assert "Next action      : Generate structures" in output
    assert "Plan before execution" not in output
    assert not (tmp_path / "generated").exists()
    assert not (tmp_path / ".pesmaker").exists()


def test_next_continues_from_finished_scf_to_training_preview(
    tmp_path,
    monkeypatch,
    capsys,
):
    """When external results already exist, `next` should keep advancing."""
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        """project: smooth_next
labeling:
  output_dir: labeling
  dataset_path: train.xyz
training:
  model: nep
  output_dir: training
  dataset: train.xyz
  command: nep
jobs:
  submit_command: sbatch
  sub_file:
    training: templates/sbatch/nep.sh
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    labeling_dir = tmp_path / "labeling"
    labeling_dir.mkdir()
    (labeling_dir / "labeling_manifest.jsonl").write_text("{}\n", encoding="utf-8")
    (labeling_dir / "calc").mkdir()
    (labeling_dir / "calc" / "OUTCAR").write_text("done\n", encoding="utf-8")
    scf_log = labeling_dir / "scf_submitted_jobs.txt"
    scf_log.write_text("sbatch submit.sh\n", encoding="utf-8")
    state_dir = tmp_path / ".pesmaker" / "smooth_next"
    state_dir.mkdir(parents=True)
    (state_dir / "next_state.json").write_text(
        json.dumps({"project": "smooth_next", "dry_runs": {"scf": {"log": str(scf_log)}}}),
        encoding="utf-8",
    )

    def fake_collect(config):
        dataset = tmp_path / "train.xyz"
        dataset.write_text("dataset\n", encoding="utf-8")
        return StageResult(
            output_dir=tmp_path,
            files=(dataset,),
            message="Collected finished SCF outputs.",
        )

    def fake_training(config):
        training_dir = tmp_path / "training"
        training_dir.mkdir()
        submit = training_dir / "submit.sh"
        submit.write_text("#!/bin/sh\n", encoding="utf-8")
        return StageResult(
            output_dir=training_dir,
            files=(submit,),
            message="Prepared training inputs.",
        )

    def fake_submit(config, *, stage="scf", dry_run=False):
        assert stage == "training"
        assert dry_run is True
        log = tmp_path / "training" / "training_submitted_jobs.txt"
        log.write_text("sbatch submit.sh\n", encoding="utf-8")
        return StageResult(
            output_dir=tmp_path / "training",
            files=(log,),
            message="Dry run: 1 job(s)",
        )

    monkeypatch.setattr("pesmaker.workflow.next.collect_labeled_dataset", fake_collect)
    monkeypatch.setattr("pesmaker.workflow.next.setup_training", fake_training)
    monkeypatch.setattr("pesmaker.workflow.next.submit_jobs", fake_submit)

    assert main(["next", str(config_path)]) == 0
    output = capsys.readouterr().out

    assert (tmp_path / "train.xyz").exists()
    assert (tmp_path / "training" / "submit.sh").exists()
    assert "Collected finished SCF outputs." in output
    assert "Prepared training inputs." in output
    assert "Current          : training submission preview" in output
    assert "Plan before execution" not in output
    assert "Stage            : training" not in output
    assert "Submit training jobs" in output
    assert f"pesmaker submit {config_path} --stage training" in output


def test_next_reports_complete_without_running_when_no_task_exists(
    tmp_path,
    monkeypatch,
    capsys,
):
    """A config with no enabled stages should complete without writing state."""
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        """project: complete_next
labeling:
  engine: none
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    assert main(["next", str(config_path)]) == 0
    output = capsys.readouterr().out

    assert "Next flow" in output
    assert "Current          : complete" in output
    assert "Plan before execution" not in output
    assert "Complete:" in output
    assert "No local PESMaker task needs to run now." in output
    assert not (tmp_path / ".pesmaker").exists()


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
