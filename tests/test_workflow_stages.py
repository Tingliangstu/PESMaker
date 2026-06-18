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
"""Tests for MD, labeling, selection, and training setup stages."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
import warnings

import numpy as np

from pesmaker.cli import _print_submit_result, main
from pesmaker.config.io import load_config
from pesmaker.workflow.stages import (
    RECOMMENDED_GW_POTCARS,
    RECOMMENDED_PBE_POTCARS,
    StageResult,
    _potcar_directory_name,
    _vasp_parallel_factors,
    submit_jobs,
)


def test_sampling_labeling_and_training_setup_write_stage_files(tmp_path):
    """Setup commands should prepare independent stage directories."""
    from ase import Atoms
    from ase.io import write

    structure_path = tmp_path / "structure.xyz"
    write(
        structure_path,
        Atoms("Te", positions=[(0.0, 0.0, 0.0)], cell=[3, 3, 20], pbc=True),
        format="extxyz",
    )
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    (generated_dir / "manifest.jsonl").write_text(
        json.dumps({"path": str(structure_path)}) + "\n",
        encoding="utf-8",
    )
    run_in = tmp_path / "run.in"
    run_in.write_text("potential nep.txt\nrun 10\n", encoding="utf-8")
    incar = tmp_path / "INCAR"
    incar.write_text("NSW = 0\n", encoding="utf-8")
    train_xyz = tmp_path / "train.xyz"
    train_xyz.write_text("", encoding="utf-8")
    config_path = tmp_path / "pesmaker.yaml"
    config_path.write_text(
        f"""project: stage_test
structures:
  - {structure_path.as_posix()}
generation:
  output_dir: {generated_dir.as_posix()}
sampling:
  engine: gpumd
  output_dir: {(tmp_path / 'sampling').as_posix()}
  run_in: {run_in.as_posix()}
  command: /opt/gpumd/gpumd
labeling:
  engine: vasp
  output_dir: {(tmp_path / 'labeling').as_posix()}
  incar: {incar.as_posix()}
  workdir_naming: indexed
training:
  model: nep
  output_dir: {(tmp_path / 'training').as_posix()}
  dataset: {train_xyz.as_posix()}
""",
        encoding="utf-8",
    )

    assert main(["sample-setup", str(config_path)]) == 0
    assert main(["scf-setup", str(config_path)]) == 0
    assert main(["train-setup", str(config_path)]) == 0

    assert (tmp_path / "sampling" / "structure_temp_300K" / "run.in").exists()
    assert (tmp_path / "sampling" / "structure_temp_300K" / "submit.sh").exists()
    assert (tmp_path / "labeling" / "calc_000000" / "POSCAR").exists()
    incar_text = (tmp_path / "labeling" / "calc_000000" / "INCAR").read_text(
        encoding="utf-8"
    )
    assert "NSW = 0\n" in incar_text
    assert "KPAR = 1" in incar_text
    assert "NCORE = 1" in incar_text
    assert (tmp_path / "training" / "nep.in").exists()


def test_scf_setup_prints_clear_next_steps(tmp_path, capsys):
    """SCF setup output should show useful next commands instead of file counts."""
    generated_dir = tmp_path / "generated"
    source_dir = generated_dir / "mp-105_Te"
    source_dir.mkdir(parents=True)
    source_path = source_dir / "perturb_000000.vasp"
    source_path.write_text(
        "Te\n1.0\n1 0 0\n0 1 0\n0 0 1\nTe\n1\nDirect\n0 0 0\n",
        encoding="utf-8",
    )
    (generated_dir / "manifest.jsonl").write_text(
        json.dumps({"path": str(source_path)}) + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "sub.yaml"
    config_path.write_text(
        f"""project: scf_print_test
generation:
  output_dir: {generated_dir.as_posix()}
labeling:
  output_dir: {(tmp_path / 'run_vasp_scf').as_posix()}
  command: /opt/vasp/vasp_std
jobs:
  cores_cpu: 36
""",
        encoding="utf-8",
    )

    assert main(["scf-setup", str(config_path)]) == 0
    output = capsys.readouterr().out

    assert "SCF setup complete." in output
    assert "Jobs prepared    : 1" in output
    assert f"Output directory : {tmp_path / 'run_vasp_scf'}" in output
    assert f"Manifest         : {tmp_path / 'run_vasp_scf' / 'labeling_manifest.jsonl'}" in output
    assert "Next steps:" in output
    assert "Inspect one job folder" in output
    assert f"pesmaker submit {config_path} --dry-run" in output
    assert f"pesmaker submit {config_path}" in output
    assert "--stage scf" not in output
    assert "Files written" not in output


def test_labeling_setup_can_preserve_generated_vasp_source_tree(tmp_path):
    """VASP SCF setup can keep generated path identity for batch jobs."""
    generated_dir = tmp_path / "generated"
    source_dir = generated_dir / "mp-105_Te"
    source_dir.mkdir(parents=True)
    source_path = source_dir / "structure_000000.vasp"
    source_text = "Te\n1.0\n1 0 0\n0 1 0\n0 0 1\nTe\n1\nDirect\n0 0 0\n"
    source_path.write_text(source_text, encoding="utf-8")
    (generated_dir / "manifest.jsonl").write_text(
        json.dumps({"path": str(source_path)}) + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "pesmaker.yaml"
    config_path.write_text(
        f"""project: source_tree_test
structures:
  - POSCAR
generation:
  output_dir: {generated_dir.as_posix()}
labeling:
  output_dir: {(tmp_path / 'labeling').as_posix()}
jobs:
  sbatch_templates:
    labeling: {(tmp_path / 'vasp.sh').as_posix()}
""",
        encoding="utf-8",
    )
    (tmp_path / "vasp.sh").write_text(
        "#!/bin/bash\ncd \"{workdir}\"\n{command}\n",
        encoding="utf-8",
    )

    assert main(["scf-setup", str(config_path)]) == 0

    workdir = tmp_path / "labeling" / "mp-105_Te" / "structure_000000"
    poscar_lines = (workdir / "POSCAR").read_text(encoding="utf-8").splitlines()
    assert poscar_lines[5].split() == ["Te"]
    assert poscar_lines[6].split() == ["1"]
    assert (workdir / "structure_000000.vasp-bak").read_text(
        encoding="utf-8"
    ) == source_text
    manifest_record = json.loads(
        (tmp_path / "labeling" / "labeling_manifest.jsonl").read_text(
            encoding="utf-8"
        )
    )
    assert manifest_record["workdir"] == str(workdir)


def test_labeling_setup_uses_local_generated_without_structures(tmp_path, monkeypatch):
    """SCF setup can submit structures from a prior generate-only run."""
    monkeypatch.chdir(tmp_path)
    generated_dir = tmp_path / "generated"
    source_dir = generated_dir / "mp-105_Te"
    source_dir.mkdir(parents=True)
    source_path = source_dir / "structure_000000.vasp"
    source_text = "Te\n1.0\n1 0 0\n0 1 0\n0 0 1\nTe\n1\nDirect\n0 0 0\n"
    source_path.write_text(source_text, encoding="utf-8")
    (generated_dir / "manifest.jsonl").write_text(
        json.dumps({"path": source_path.relative_to(tmp_path).as_posix()}) + "\n",
        encoding="utf-8",
    )
    sub_file = tmp_path / "templates" / "sbatch" / "vasp_cpu_36.sh"
    sub_file.parent.mkdir(parents=True)
    sub_file.write_text(
        "#!/bin/bash\n#SBATCH --job-name={job_name}\ncd \"{workdir}\"\n{command}\n",
        encoding="utf-8",
    )
    incar = tmp_path / "templates" / "vasp" / "INCAR"
    incar.parent.mkdir(parents=True)
    incar.write_text("NSW = 0\n", encoding="utf-8")
    config_path = tmp_path / "sub.yaml"
    config_path.write_text(
        """project: Te_bulk_mp
labeling:
  engine: vasp
  output_dir: labeling
  incar: templates/vasp/INCAR
  command: /opt/vasp/vasp_std

jobs:
  submit_command: sbatch
  sub_file: templates/sbatch/vasp_cpu_36.sh
  cores_cpu: 36
""",
        encoding="utf-8",
    )

    assert main(["scf-setup", str(config_path)]) == 0

    workdir = tmp_path / "labeling" / "mp-105_Te" / "structure_000000"
    poscar_lines = (workdir / "POSCAR").read_text(encoding="utf-8").splitlines()
    assert poscar_lines[5].split() == ["Te"]
    assert poscar_lines[6].split() == ["1"]
    incar_text = (workdir / "INCAR").read_text(encoding="utf-8")
    submit_text = (workdir / "submit.sh").read_text(encoding="utf-8")
    assert "NSW = 0\n" in incar_text
    assert "KPAR = 2" in incar_text
    assert "NCORE = 3" in incar_text
    assert f'cd "{workdir}"' not in submit_text
    assert "#SBATCH --job-name=structure_000000" in submit_text
    assert "mpirun -np 36 /opt/vasp/vasp_std" in submit_text


def test_labeling_setup_normalizes_interleaved_vasp_source(tmp_path, capsys):
    """SCF setup should not copy an interleaved VASP species block as-is."""
    generated_dir = tmp_path / "generated"
    source_dir = generated_dir / "Te2Pd"
    source_dir.mkdir(parents=True)
    source_path = source_dir / "perturb_000000.vasp"
    source_text = """Te Pd Te Pd
1.0
10 0 0
0 10 0
0 0 10
Te Pd Te Pd
2 1 2 1
Cartesian
0 0 0
1 0 0
2 0 0
3 0 0
4 0 0
5 0 0
"""
    source_path.write_text(source_text, encoding="utf-8")
    (generated_dir / "manifest.jsonl").write_text(
        json.dumps({"path": str(source_path)}) + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "pesmaker.yaml"
    config_path.write_text(
        f"""project: normalize_interleaved_poscar
generation:
  output_dir: {generated_dir.as_posix()}
labeling:
  output_dir: {(tmp_path / 'labeling').as_posix()}
jobs:
  cores_cpu: 36
""",
        encoding="utf-8",
    )

    assert main(["scf-setup", str(config_path)]) == 0
    output = capsys.readouterr().out

    poscar = tmp_path / "labeling" / "Te2Pd" / "perturb_000000" / "POSCAR"
    lines = poscar.read_text(encoding="utf-8").splitlines()
    assert lines[5].split() == ["Te", "Pd"]
    assert lines[6].split() == ["4", "2"]
    assert "Warnings:" in output
    assert "Normalized non-compact VASP species block" in output
    assert "Inspect POSCAR before submission" in output
    assert (
        tmp_path
        / "labeling"
        / "Te2Pd"
        / "perturb_000000"
        / "perturb_000000.vasp-bak"
    ).read_text(encoding="utf-8") == source_text


def test_labeling_setup_warns_for_large_scf_jobs(tmp_path, capsys):
    """Large SCF inputs should be called out before users submit jobs."""
    from ase import Atoms
    from ase.io import write

    generated_dir = tmp_path / "generated"
    source_dir = generated_dir / "large"
    source_dir.mkdir(parents=True)
    source_path = source_dir / "structure_000000.xyz"
    atoms = Atoms(
        "Te251",
        positions=[(float(index), 0.0, 0.0) for index in range(251)],
        cell=[300.0, 10.0, 10.0],
        pbc=True,
    )
    write(source_path, atoms, format="extxyz")
    (generated_dir / "manifest.jsonl").write_text(
        json.dumps({"path": str(source_path), "atom_count": 251}) + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "pesmaker.yaml"
    config_path.write_text(
        f"""project: large_scf_warning
generation:
  output_dir: {generated_dir.as_posix()}
labeling:
  output_dir: {(tmp_path / 'labeling').as_posix()}
jobs:
  cores_cpu: 36
""",
        encoding="utf-8",
    )

    assert main(["scf-setup", str(config_path)]) == 0
    output = capsys.readouterr().out

    poscar = tmp_path / "labeling" / "large" / "structure_000000" / "POSCAR"
    assert "Warnings:" in output
    assert f"Large SCF job: {source_path}" in output
    assert f"-> {poscar}" in output
    assert "has 251 atoms" in output
    assert "single-point calculation may be expensive" in output


def test_labeling_setup_does_not_warn_for_250_atom_scf_jobs(tmp_path, capsys):
    """The large-job warning should only trigger above 250 atoms."""
    from ase import Atoms
    from ase.io import write

    generated_dir = tmp_path / "generated"
    source_dir = generated_dir / "medium"
    source_dir.mkdir(parents=True)
    source_path = source_dir / "structure_000000.xyz"
    atoms = Atoms(
        "Te250",
        positions=[(float(index), 0.0, 0.0) for index in range(250)],
        cell=[300.0, 10.0, 10.0],
        pbc=True,
    )
    write(source_path, atoms, format="extxyz")
    (generated_dir / "manifest.jsonl").write_text(
        json.dumps({"path": str(source_path), "atom_count": 250}) + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "pesmaker.yaml"
    config_path.write_text(
        f"""project: no_large_scf_warning
generation:
  output_dir: {generated_dir.as_posix()}
labeling:
  output_dir: {(tmp_path / 'labeling').as_posix()}
jobs:
  cores_cpu: 36
""",
        encoding="utf-8",
    )

    assert main(["scf-setup", str(config_path)]) == 0
    output = capsys.readouterr().out

    assert "Warnings:" not in output
    assert "Large SCF job" not in output


def test_labeling_setup_normalizes_literal_submit_template(tmp_path):
    """User submit scripts should inherit generated job names and resources."""
    generated_dir = tmp_path / "generated"
    source_dir = generated_dir / "mp-105_Te"
    source_dir.mkdir(parents=True)
    source_path = source_dir / "perturb_000000.vasp"
    source_path.write_text(
        "Te\n1.0\n1 0 0\n0 1 0\n0 0 1\nTe\n1\nDirect\n0 0 0\n",
        encoding="utf-8",
    )
    (generated_dir / "manifest.jsonl").write_text(
        json.dumps({"path": str(source_path)}) + "\n",
        encoding="utf-8",
    )
    sub_file = tmp_path / "sub.sh"
    sub_file.write_text(
        """#!/bin/bash -l
#SBATCH --job-name=VASP-CPU
#SBATCH --output=out.%j
#SBATCH --error=err.%j
#SBATCH --nodes=1
#SBATCH --ntasks=12              # total MPI ranks
#SBATCH --cpus-per-task=1

echo "Running on node: ${SLURM_NODELIST:-unknown}"
mpirun /old/software/vasp_std
""",
        encoding="utf-8",
    )
    config_path = tmp_path / "pesmaker.yaml"
    config_path.write_text(
        f"""project: submit_template_test
generation:
  output_dir: {generated_dir.as_posix()}
labeling:
  output_dir: {(tmp_path / 'labeling').as_posix()}
  command: /opt/vasp/vasp_std
jobs:
  cores_cpu: 36
  sub_file: {sub_file.as_posix()}
""",
        encoding="utf-8",
    )

    assert main(["scf-setup", str(config_path)]) == 0

    workdir = tmp_path / "labeling" / "mp-105_Te" / "perturb_000000"
    submit_text = (workdir / "submit.sh").read_text(encoding="utf-8")
    assert "#SBATCH --job-name=perturb_000000" in submit_text
    assert "#SBATCH --ntasks=36" in submit_text
    assert "# total MPI ranks" in submit_text
    assert "mpirun -np 36 /opt/vasp/vasp_std" in submit_text
    assert "/old/software/vasp_std" not in submit_text
    assert "${SLURM_NODELIST:-unknown}" in submit_text


def test_labeling_setup_preserves_literal_submit_template_without_resources(
    tmp_path,
):
    """Literal user submit scripts should stay untouched without resource fields."""
    generated_dir = tmp_path / "generated"
    source_dir = generated_dir / "mp-105_Te"
    source_dir.mkdir(parents=True)
    source_path = source_dir / "perturb_000000.vasp"
    source_path.write_text(
        "Te\n1.0\n1 0 0\n0 1 0\n0 0 1\nTe\n1\nDirect\n0 0 0\n",
        encoding="utf-8",
    )
    (generated_dir / "manifest.jsonl").write_text(
        json.dumps({"path": str(source_path)}) + "\n",
        encoding="utf-8",
    )
    sub_file = tmp_path / "sub.sh"
    template_text = """#!/bin/bash
#SBATCH --job-name=manual_vasp
#SBATCH --ntasks=12
mpirun /cluster/vasp_std
"""
    sub_file.write_text(template_text, encoding="utf-8")
    config_path = tmp_path / "pesmaker.yaml"
    config_path.write_text(
        f"""project: preserve_submit_template_test
generation:
  output_dir: {generated_dir.as_posix()}
labeling:
  output_dir: {(tmp_path / 'labeling').as_posix()}
  command: /new/vasp_std
jobs:
  sub_file: {sub_file.as_posix()}
""",
        encoding="utf-8",
    )

    assert main(["scf-setup", str(config_path)]) == 0

    workdir = tmp_path / "labeling" / "mp-105_Te" / "perturb_000000"
    assert (workdir / "submit.sh").read_text(encoding="utf-8") == template_text


def test_labeling_setup_scans_explicit_input_dir_without_manifest(tmp_path):
    """Users can point SCF setup at a folder of generated structure files."""
    input_dir = tmp_path / "generated"
    source_dir = input_dir / "mp-105_Te"
    source_dir.mkdir(parents=True)
    source_path = source_dir / "structure_000000.vasp"
    source_text = "Te\n1.0\n1 0 0\n0 1 0\n0 0 1\nTe\n1\nDirect\n0 0 0\n"
    source_path.write_text(source_text, encoding="utf-8")
    (source_dir / "notes.txt").write_text("ignored note\n", encoding="utf-8")
    config_path = tmp_path / "sub.yaml"
    config_path.write_text(
        f"""project: Te_bulk_mp
labeling:
  engine: vasp
  input_dir: {input_dir.as_posix()}
  output_dir: {(tmp_path / 'labeling').as_posix()}
""",
        encoding="utf-8",
    )

    assert main(["scf-setup", str(config_path)]) == 0

    manifest_records = [
        json.loads(line)
        for line in (tmp_path / "labeling" / "labeling_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(manifest_records) == 1
    assert manifest_records[0]["source"] == str(source_path)
    assert manifest_records[0]["input_dir"] == str(input_dir)
    assert manifest_records[0]["input_mode"] == "input_dir_scan"
    assert manifest_records[0]["input_relative_path"] == (
        "mp-105_Te/structure_000000.vasp"
    )
    assert manifest_records[0]["cores_cpu"] == 1
    assert manifest_records[0]["gpus"] == 0
    assert manifest_records[0]["vasp_kpar"] == 1
    assert manifest_records[0]["vasp_ncore"] == 1


def test_labeling_setup_scans_generic_xyz_without_manifest(tmp_path):
    """Non-PESMaker structure filenames should be valid SCF inputs."""
    from ase import Atoms
    from ase.io import write

    input_dir = tmp_path / "generated"
    source_dir = input_dir / "mp-105_Te"
    source_dir.mkdir(parents=True)
    source_path = source_dir / "manual_candidate.xyz"
    write(
        source_path,
        Atoms("Te", positions=[(0.0, 0.0, 0.0)], cell=[1.0, 1.0, 1.0], pbc=True),
        format="extxyz",
    )
    (source_dir / "notes.txt").write_text("ignored note\n", encoding="utf-8")
    config_path = tmp_path / "sub.yaml"
    config_path.write_text(
        f"""project: Te_bulk_mp
labeling:
  engine: vasp
  input_dir: {input_dir.as_posix()}
  output_dir: {(tmp_path / 'labeling').as_posix()}
""",
        encoding="utf-8",
    )

    assert main(["scf-setup", str(config_path)]) == 0

    manifest_records = [
        json.loads(line)
        for line in (tmp_path / "labeling" / "labeling_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(manifest_records) == 1
    assert manifest_records[0]["source"] == str(source_path)
    assert manifest_records[0]["input_relative_path"] == (
        "mp-105_Te/manual_candidate.xyz"
    )
    assert (tmp_path / "labeling" / "mp-105_Te" / "manual_candidate").exists()


def test_labeling_setup_expands_multiframe_extxyz_without_manifest(tmp_path):
    """Manual multi-frame structure files should become one SCF job per frame."""
    from ase import Atoms
    from ase.io import read, write

    input_dir = tmp_path / "selected"
    input_dir.mkdir()
    source_path = input_dir / "manual_selected_frames.extxyz"
    write(
        source_path,
        [
            Atoms("Te", positions=[(0.0, 0.0, 0.0)], cell=[5, 5, 5], pbc=True),
            Atoms("Te", positions=[(2.0, 0.0, 0.0)], cell=[5, 5, 5], pbc=True),
        ],
        format="extxyz",
    )
    config_path = tmp_path / "sub.yaml"
    config_path.write_text(
        f"""project: manual_selected_frames
labeling:
  engine: vasp
  input_dir: {input_dir.as_posix()}
  output_dir: {(tmp_path / 'labeling').as_posix()}
""",
        encoding="utf-8",
    )

    assert main(["scf-setup", str(config_path)]) == 0

    poscar_0 = tmp_path / "labeling" / "manual_selected_frames_000000" / "POSCAR"
    poscar_1 = tmp_path / "labeling" / "manual_selected_frames_000001" / "POSCAR"
    assert read(poscar_0).get_positions()[0, 0] == 0.0
    assert read(poscar_1).get_positions()[0, 0] == 2.0
    manifest_records = [
        json.loads(line)
        for line in (tmp_path / "labeling" / "labeling_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(manifest_records) == 2
    assert [record["source"] for record in manifest_records] == [
        str(source_path),
        str(source_path),
    ]
    assert [record["frame_index"] for record in manifest_records] == [0, 1]
    assert [record["source_frame"] for record in manifest_records] == [0, 1]
    assert {record["input_mode"] for record in manifest_records} == {
        "input_dir_scan"
    }


def test_labeling_setup_writes_cpu_resources_to_incar_and_submit(tmp_path):
    """Default VASP setup should match CPU Slurm resources."""
    generated_dir = tmp_path / "generated"
    source_dir = generated_dir / "mp-105_Te"
    source_dir.mkdir(parents=True)
    source_path = source_dir / "structure_000000.vasp"
    source_path.write_text(
        "Te\n1.0\n1 0 0\n0 1 0\n0 0 1\nTe\n1\nDirect\n0 0 0\n",
        encoding="utf-8",
    )
    (generated_dir / "manifest.jsonl").write_text(
        json.dumps({"path": str(source_path)}) + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "pesmaker.yaml"
    config_path.write_text(
        f"""project: cpu_resources_test
generation:
  output_dir: {generated_dir.as_posix()}
labeling:
  output_dir: {(tmp_path / 'labeling').as_posix()}
  command: /opt/vasp/vasp_std
jobs:
  cores_cpu: 36
""",
        encoding="utf-8",
    )

    assert main(["scf-setup", str(config_path)]) == 0

    workdir = tmp_path / "labeling" / "mp-105_Te" / "structure_000000"
    incar_text = (workdir / "INCAR").read_text(encoding="utf-8")
    submit_text = (workdir / "submit.sh").read_text(encoding="utf-8")
    assert "ENCUT = 650" in incar_text
    assert "KSPACING = 0.2" in incar_text
    assert "EDIFF = 1E-06" in incar_text
    assert "SIGMA = 0.02" in incar_text
    assert "ISMEAR = 0" in incar_text
    assert "IVDW" not in incar_text
    assert "LWAVE = .FALSE." in incar_text
    assert "LCHARG = .FALSE." in incar_text
    assert "KPAR = 2" in incar_text
    assert "NCORE = 3" in incar_text
    assert "#!/bin/bash -l" in submit_text
    assert "#SBATCH --output=out.%j" in submit_text
    assert "#SBATCH --error=err.%j" in submit_text
    assert "#SBATCH --ntasks=36" in submit_text
    assert "#SBATCH --cpus-per-task=1" in submit_text
    assert "#SBATCH --gres" not in submit_text
    assert "#SBATCH --time" not in submit_text
    assert "set -euo pipefail" in submit_text
    assert "cd " not in submit_text
    assert "export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}" in submit_text
    assert "ulimit -s unlimited" in submit_text
    assert "mpirun -np 36 /opt/vasp/vasp_std" in submit_text


def test_labeling_setup_adds_np_to_explicit_cpu_mpirun_command(tmp_path):
    """CPU VASP commands with mpirun but no rank count should get -np."""
    generated_dir = tmp_path / "generated"
    source_dir = generated_dir / "mp-105_Te"
    source_dir.mkdir(parents=True)
    source_path = source_dir / "structure_000000.vasp"
    source_path.write_text(
        "Te\n1.0\n1 0 0\n0 1 0\n0 0 1\nTe\n1\nDirect\n0 0 0\n",
        encoding="utf-8",
    )
    (generated_dir / "manifest.jsonl").write_text(
        json.dumps({"path": str(source_path)}) + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "pesmaker.yaml"
    config_path.write_text(
        f"""project: cpu_explicit_mpirun_test
generation:
  output_dir: {generated_dir.as_posix()}
labeling:
  output_dir: {(tmp_path / 'labeling').as_posix()}
  command: mpirun /opt/vasp/vasp_std
jobs:
  cores_cpu: 36
""",
        encoding="utf-8",
    )

    assert main(["scf-setup", str(config_path)]) == 0

    workdir = tmp_path / "labeling" / "mp-105_Te" / "structure_000000"
    submit_text = (workdir / "submit.sh").read_text(encoding="utf-8")
    assert submit_text.count("mpirun -np 36 /opt/vasp/vasp_std") == 1
    assert "mpirun -np 36 mpirun" not in submit_text


def test_labeling_setup_uses_manual_vasp_parallel_options(tmp_path):
    """VASP-specific parallel overrides should use explicit jobs.vasp_* keys."""
    generated_dir = tmp_path / "generated"
    source_dir = generated_dir / "mp-105_Te"
    source_dir.mkdir(parents=True)
    source_path = source_dir / "structure_000000.vasp"
    source_path.write_text(
        "Te\n1.0\n1 0 0\n0 1 0\n0 0 1\nTe\n1\nDirect\n0 0 0\n",
        encoding="utf-8",
    )
    (generated_dir / "manifest.jsonl").write_text(
        json.dumps({"path": str(source_path)}) + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "pesmaker.yaml"
    config_path.write_text(
        f"""project: manual_vasp_parallel_test
generation:
  output_dir: {generated_dir.as_posix()}
labeling:
  output_dir: {(tmp_path / 'labeling').as_posix()}
jobs:
  cores_cpu: 36
  vasp_kpar: 2
  vasp_ncore: 6
""",
        encoding="utf-8",
    )

    assert main(["scf-setup", str(config_path)]) == 0

    workdir = tmp_path / "labeling" / "mp-105_Te" / "structure_000000"
    incar_text = (workdir / "INCAR").read_text(encoding="utf-8")
    manifest_record = json.loads(
        (tmp_path / "labeling" / "labeling_manifest.jsonl").read_text(
            encoding="utf-8"
        )
    )
    assert "KPAR = 2" in incar_text
    assert "NCORE = 6" in incar_text
    assert manifest_record["vasp_kpar"] == 2
    assert manifest_record["vasp_ncore"] == 6


def test_labeling_setup_rejects_old_parallel_option_names(tmp_path, capsys):
    """Old generic parallel option names should fail with the new VASP names."""
    generated_dir = tmp_path / "generated"
    source_dir = generated_dir / "mp-105_Te"
    source_dir.mkdir(parents=True)
    source_path = source_dir / "structure_000000.vasp"
    source_path.write_text(
        "Te\n1.0\n1 0 0\n0 1 0\n0 0 1\nTe\n1\nDirect\n0 0 0\n",
        encoding="utf-8",
    )
    (generated_dir / "manifest.jsonl").write_text(
        json.dumps({"path": str(source_path)}) + "\n",
        encoding="utf-8",
    )

    for old_key, new_key in (("kpar", "vasp_kpar"), ("ncore", "vasp_ncore")):
        config_path = tmp_path / f"{old_key}.yaml"
        config_path.write_text(
            f"""project: old_parallel_option_test
generation:
  output_dir: {generated_dir.as_posix()}
labeling:
  output_dir: {(tmp_path / f'labeling_{old_key}').as_posix()}
jobs:
  cores_cpu: 36
  {old_key}: 2
""",
            encoding="utf-8",
        )

        assert main(["scf-setup", str(config_path)]) == 2
        captured = capsys.readouterr()
        assert f"jobs.{old_key} has been renamed to jobs.{new_key}" in captured.err


def test_vasp_parallel_factors_follow_requested_cpu_cores():
    """KPAR/NCORE should be calculated from jobs.cores_cpu, not fixed."""
    expected = {
        24: (2, 3),
        32: (2, 4),
        36: (2, 3),
        40: (2, 4),
        48: (2, 4),
        64: (2, 4),
    }

    for cores_cpu, factors in expected.items():
        assert _vasp_parallel_factors(cores_cpu) == factors


def test_vasp_parallel_factors_keep_manual_kpar_and_large_cell_ncore():
    """KPAR is user-controlled, while large cells prefer larger NCORE."""
    assert _vasp_parallel_factors(48, kpar=4) == (4, 3)
    assert _vasp_parallel_factors(36, atom_count=500) == (2, 9)
    assert _vasp_parallel_factors(48, atom_count=500) == (2, 12)


def test_labeling_setup_writes_gpu_resources_to_submit_without_cpu_incar(tmp_path):
    """GPU jobs should request GPUs without adding VASP CPU parallel tags."""
    generated_dir = tmp_path / "generated"
    source_dir = generated_dir / "mp-105_Te"
    source_dir.mkdir(parents=True)
    source_path = source_dir / "structure_000000.vasp"
    source_path.write_text(
        "Te\n1.0\n1 0 0\n0 1 0\n0 0 1\nTe\n1\nDirect\n0 0 0\n",
        encoding="utf-8",
    )
    (generated_dir / "manifest.jsonl").write_text(
        json.dumps({"path": str(source_path)}) + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "pesmaker.yaml"
    config_path.write_text(
        f"""project: gpu_resources_test
generation:
  output_dir: {generated_dir.as_posix()}
labeling:
  output_dir: {(tmp_path / 'labeling').as_posix()}
  command: /opt/vasp/vasp_std
jobs:
  cores_cpu: 8
  gpus: 2
""",
        encoding="utf-8",
    )

    assert main(["scf-setup", str(config_path)]) == 0

    workdir = tmp_path / "labeling" / "mp-105_Te" / "structure_000000"
    incar_text = (workdir / "INCAR").read_text(encoding="utf-8")
    submit_text = (workdir / "submit.sh").read_text(encoding="utf-8")
    assert "KPAR =" not in incar_text
    assert "NCORE =" not in incar_text
    assert "#SBATCH --ntasks=8" in submit_text
    assert "#SBATCH --gres=gpu:2" in submit_text
    assert "mpirun -np 2 /opt/vasp/vasp_std" in submit_text


def test_labeling_setup_wraps_gpu_vasp_command_in_submit_template(tmp_path):
    """GPU VASP templates should run one MPI rank per requested GPU."""
    generated_dir = tmp_path / "generated"
    source_dir = generated_dir / "mp-105_Te"
    source_dir.mkdir(parents=True)
    source_path = source_dir / "structure_000000.vasp"
    source_path.write_text(
        "Te\n1.0\n1 0 0\n0 1 0\n0 0 1\nTe\n1\nDirect\n0 0 0\n",
        encoding="utf-8",
    )
    (generated_dir / "manifest.jsonl").write_text(
        json.dumps({"path": str(source_path)}) + "\n",
        encoding="utf-8",
    )
    sub_file = tmp_path / "sub_gpu.sh"
    sub_file.write_text(
        """#!/bin/bash
#SBATCH --job-name=old_name
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
{command}
""",
        encoding="utf-8",
    )
    config_path = tmp_path / "pesmaker.yaml"
    config_path.write_text(
        f"""project: gpu_template_test
generation:
  output_dir: {generated_dir.as_posix()}
labeling:
  output_dir: {(tmp_path / 'labeling').as_posix()}
  command: /data/software/vasp6.4-gpu/bin/vasp_std
jobs:
  cores_cpu: 6
  gpus: 1
  sub_file: {sub_file.as_posix()}
""",
        encoding="utf-8",
    )

    assert main(["scf-setup", str(config_path)]) == 0

    workdir = tmp_path / "labeling" / "mp-105_Te" / "structure_000000"
    submit_text = (workdir / "submit.sh").read_text(encoding="utf-8")
    assert "#SBATCH --ntasks=6" in submit_text
    assert "#SBATCH --gres=gpu:1" in submit_text
    assert "mpirun -np 1 /data/software/vasp6.4-gpu/bin/vasp_std" in submit_text


def test_labeling_setup_keeps_explicit_gpu_vasp_mpirun_command(tmp_path):
    """Explicit GPU VASP launch commands should not be wrapped twice."""
    generated_dir = tmp_path / "generated"
    source_dir = generated_dir / "mp-105_Te"
    source_dir.mkdir(parents=True)
    source_path = source_dir / "structure_000000.vasp"
    source_path.write_text(
        "Te\n1.0\n1 0 0\n0 1 0\n0 0 1\nTe\n1\nDirect\n0 0 0\n",
        encoding="utf-8",
    )
    (generated_dir / "manifest.jsonl").write_text(
        json.dumps({"path": str(source_path)}) + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "pesmaker.yaml"
    config_path.write_text(
        f"""project: gpu_explicit_mpirun_test
generation:
  output_dir: {generated_dir.as_posix()}
labeling:
  output_dir: {(tmp_path / 'labeling').as_posix()}
  command: mpirun -np 1 /data/software/vasp6.4-gpu/bin/vasp_std
jobs:
  cores_cpu: 6
  gpus: 1
""",
        encoding="utf-8",
    )

    assert main(["scf-setup", str(config_path)]) == 0

    workdir = tmp_path / "labeling" / "mp-105_Te" / "structure_000000"
    submit_text = (workdir / "submit.sh").read_text(encoding="utf-8")
    assert (
        submit_text.count("mpirun -np 1 /data/software/vasp6.4-gpu/bin/vasp_std")
        == 1
    )
    assert "mpirun -np 1 mpirun" not in submit_text


def test_labeling_setup_adds_np_to_explicit_gpu_mpirun_command(tmp_path):
    """GPU VASP commands with mpirun but no rank count should get -np."""
    generated_dir = tmp_path / "generated"
    source_dir = generated_dir / "mp-105_Te"
    source_dir.mkdir(parents=True)
    source_path = source_dir / "structure_000000.vasp"
    source_path.write_text(
        "Te\n1.0\n1 0 0\n0 1 0\n0 0 1\nTe\n1\nDirect\n0 0 0\n",
        encoding="utf-8",
    )
    (generated_dir / "manifest.jsonl").write_text(
        json.dumps({"path": str(source_path)}) + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "pesmaker.yaml"
    config_path.write_text(
        f"""project: gpu_explicit_mpirun_np_test
generation:
  output_dir: {generated_dir.as_posix()}
labeling:
  output_dir: {(tmp_path / 'labeling').as_posix()}
  command: mpirun /data/software/vasp6.4-gpu/bin/vasp_std
jobs:
  cores_cpu: 6
  gpus: 1
""",
        encoding="utf-8",
    )

    assert main(["scf-setup", str(config_path)]) == 0

    workdir = tmp_path / "labeling" / "mp-105_Te" / "structure_000000"
    submit_text = (workdir / "submit.sh").read_text(encoding="utf-8")
    assert (
        submit_text.count("mpirun -np 1 /data/software/vasp6.4-gpu/bin/vasp_std")
        == 1
    )
    assert "mpirun -np 1 mpirun" not in submit_text


def test_labeling_setup_can_generate_potcar_from_library(tmp_path):
    """VASP SCF setup can concatenate POTCAR files from a local library."""
    generated_dir = tmp_path / "generated"
    source_dir = generated_dir / "mp-105_Te"
    source_dir.mkdir(parents=True)
    source_path = source_dir / "structure_000000.vasp"
    source_path.write_text(
        "Te\n1.0\n1 0 0\n0 1 0\n0 0 1\nTe\n1\nDirect\n0 0 0\n",
        encoding="utf-8",
    )
    (generated_dir / "manifest.jsonl").write_text(
        json.dumps({"path": str(source_path)}) + "\n",
        encoding="utf-8",
    )
    library = tmp_path / "potentials"
    (library / "Te").mkdir(parents=True)
    (library / "Te" / "POTCAR").write_text("POTCAR Te\n", encoding="utf-8")
    config_path = tmp_path / "pesmaker.yaml"
    config_path.write_text(
        f"""project: potcar_test
structures:
  - POSCAR
generation:
  output_dir: {generated_dir.as_posix()}
labeling:
  output_dir: {(tmp_path / 'labeling').as_posix()}
  potcar_library: {library.as_posix()}
""",
        encoding="utf-8",
    )

    assert main(["scf-setup", str(config_path)]) == 0

    workdir = tmp_path / "labeling" / "mp-105_Te" / "structure_000000"
    assert (workdir / "POTCAR").read_text(encoding="utf-8") == "POTCAR Te\n"
    assert (workdir / "POTCAR.spec").read_text(encoding="utf-8") == "Te\n"


def test_labeling_setup_uses_recommended_pbe_potcar(tmp_path):
    """Default POTCAR generation should use recommended PBE variants."""
    generated_dir = tmp_path / "generated"
    source_dir = generated_dir / "mp-Na"
    source_dir.mkdir(parents=True)
    source_path = source_dir / "structure_000000.vasp"
    source_path.write_text(
        "Na\n1.0\n1 0 0\n0 1 0\n0 0 1\nNa\n1\nDirect\n0 0 0\n",
        encoding="utf-8",
    )
    (generated_dir / "manifest.jsonl").write_text(
        json.dumps({"path": str(source_path)}) + "\n",
        encoding="utf-8",
    )
    library = tmp_path / "potentials"
    (library / "Na_pv").mkdir(parents=True)
    (library / "Na_pv" / "POTCAR").write_text("POTCAR Na pv\n", encoding="utf-8")
    config_path = tmp_path / "pesmaker.yaml"
    config_path.write_text(
        f"""project: recommended_potcar_test
structures:
  - POSCAR
generation:
  output_dir: {generated_dir.as_posix()}
labeling:
  output_dir: {(tmp_path / 'labeling').as_posix()}
  potcar_library: {library.as_posix()}
""",
        encoding="utf-8",
    )

    assert main(["scf-setup", str(config_path)]) == 0

    workdir = tmp_path / "labeling" / "mp-Na" / "structure_000000"
    assert (workdir / "POTCAR").read_text(encoding="utf-8") == "POTCAR Na pv\n"
    assert (workdir / "POTCAR.spec").read_text(encoding="utf-8") == "Na_pv\n"


def test_labeling_setup_can_generate_gw_potcar_from_library(tmp_path):
    """GW POTCAR selection should use the element_GW directory by default."""
    generated_dir = tmp_path / "generated"
    source_dir = generated_dir / "mp-105_Te"
    source_dir.mkdir(parents=True)
    source_path = source_dir / "structure_000000.vasp"
    source_path.write_text(
        "Te\n1.0\n1 0 0\n0 1 0\n0 0 1\nTe\n1\nDirect\n0 0 0\n",
        encoding="utf-8",
    )
    (generated_dir / "manifest.jsonl").write_text(
        json.dumps({"path": str(source_path)}) + "\n",
        encoding="utf-8",
    )
    library = tmp_path / "potentials"
    (library / "Te_GW").mkdir(parents=True)
    (library / "Te_GW" / "POTCAR").write_text("POTCAR Te GW\n", encoding="utf-8")
    config_path = tmp_path / "pesmaker.yaml"
    config_path.write_text(
        f"""project: potcar_gw_test
structures:
  - POSCAR
generation:
  output_dir: {generated_dir.as_posix()}
labeling:
  output_dir: {(tmp_path / 'labeling').as_posix()}
  potcar_library: {library.as_posix()}
  gw_potcar: true
""",
        encoding="utf-8",
    )

    assert main(["scf-setup", str(config_path)]) == 0

    workdir = tmp_path / "labeling" / "mp-105_Te" / "structure_000000"
    assert (workdir / "POTCAR").read_text(encoding="utf-8") == "POTCAR Te GW\n"
    assert (workdir / "POTCAR.spec").read_text(encoding="utf-8") == "Te_GW\n"


def test_labeling_setup_uses_recommended_gw_potcar(tmp_path):
    """GW POTCAR generation should use recommended GW variants."""
    generated_dir = tmp_path / "generated"
    source_dir = generated_dir / "mp-Na"
    source_dir.mkdir(parents=True)
    source_path = source_dir / "structure_000000.vasp"
    source_path.write_text(
        "Na\n1.0\n1 0 0\n0 1 0\n0 0 1\nNa\n1\nDirect\n0 0 0\n",
        encoding="utf-8",
    )
    (generated_dir / "manifest.jsonl").write_text(
        json.dumps({"path": str(source_path)}) + "\n",
        encoding="utf-8",
    )
    library = tmp_path / "potentials"
    (library / "Na_sv_GW").mkdir(parents=True)
    (library / "Na_sv_GW" / "POTCAR").write_text(
        "POTCAR Na sv GW\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "pesmaker.yaml"
    config_path.write_text(
        f"""project: recommended_gw_potcar_test
structures:
  - POSCAR
generation:
  output_dir: {generated_dir.as_posix()}
labeling:
  output_dir: {(tmp_path / 'labeling').as_posix()}
  potcar_library: {library.as_posix()}
  gw_potcar: true
""",
        encoding="utf-8",
    )

    assert main(["scf-setup", str(config_path)]) == 0

    workdir = tmp_path / "labeling" / "mp-Na" / "structure_000000"
    assert (workdir / "POTCAR").read_text(encoding="utf-8") == "POTCAR Na sv GW\n"
    assert (workdir / "POTCAR.spec").read_text(encoding="utf-8") == "Na_sv_GW\n"


def test_recommended_potcar_tables_are_applied_for_all_entries():
    """Every built-in VASP recommendation should be used by POTCAR selection."""
    assert len(RECOMMENDED_PBE_POTCARS) == 96
    assert len(RECOMMENDED_GW_POTCARS) == 73

    for symbol, directory in RECOMMENDED_PBE_POTCARS.items():
        assert _potcar_directory_name(symbol, mapping={}, use_gw=False) == directory

    for symbol, directory in RECOMMENDED_GW_POTCARS.items():
        assert _potcar_directory_name(symbol, mapping={}, use_gw=True) == directory


def test_explicit_potcar_map_overrides_recommended_tables():
    """User overrides must still win over built-in recommended variants."""
    mapping = {"Na": "Na"}

    assert _potcar_directory_name("Na", mapping=mapping, use_gw=False) == "Na"
    assert _potcar_directory_name("Na", mapping=mapping, use_gw=True) == "Na"


def test_submit_jobs_dry_run_uses_labeling_manifest(tmp_path):
    """Batch submission should follow prepared labeling workdirs."""
    workdir = tmp_path / "labeling" / "calc_000000"
    workdir.mkdir(parents=True)
    (workdir / "submit.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    (tmp_path / "labeling" / "labeling_manifest.jsonl").write_text(
        json.dumps({"index": 0, "source": "a.vasp", "workdir": str(workdir)}) + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "pesmaker.yaml"
    config_path.write_text(
        f"""project: submit_test
structures:
  - POSCAR
labeling:
  output_dir: {(tmp_path / 'labeling').as_posix()}
jobs:
  submit_command: sbatch
""",
        encoding="utf-8",
    )

    result = submit_jobs(load_config(config_path), dry_run=True)

    assert result.message == "Would submit 1 scf job(s)"
    log = tmp_path / "labeling" / "scf_submitted_jobs.txt"
    assert "DRY-RUN" in log.read_text(encoding="utf-8")


def test_submit_jobs_skips_completed_vasp_jobs_by_default(tmp_path):
    """Normally terminated and converged VASP jobs should not be resubmitted."""
    labeling_dir = tmp_path / "labeling"
    complete = labeling_dir / "calc_complete"
    incomplete = labeling_dir / "calc_incomplete"
    for workdir in (complete, incomplete):
        workdir.mkdir(parents=True)
        (workdir / "submit.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    (complete / "OUTCAR").write_text(
        "General timing and accounting informations for this job:\n",
        encoding="utf-8",
    )
    (incomplete / "OUTCAR").write_text("DAV:  20\n", encoding="utf-8")
    (labeling_dir / "labeling_manifest.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"index": 0, "workdir": str(complete)}),
                json.dumps({"index": 1, "workdir": str(incomplete)}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "pesmaker.yaml"
    config_path.write_text(
        f"""project: submit_test
labeling:
  engine: vasp
  output_dir: {labeling_dir.as_posix()}
jobs:
  submit_command: sbatch
""",
        encoding="utf-8",
    )

    result = submit_jobs(load_config(config_path), dry_run=True)

    assert result.message == (
        "Would submit 1 scf job(s); skipped 1 completed VASP job(s)"
    )
    assert result.submission is not None
    assert result.submission.total_jobs == 2
    assert result.submission.completed_jobs == 1
    assert result.submission.pending_jobs == 1
    log_text = (labeling_dir / "scf_submitted_jobs.txt").read_text(
        encoding="utf-8"
    )
    assert f"SKIPPED completed VASP job: {complete}" in log_text
    assert f"cd {incomplete}" in log_text
    assert f"cd {complete}" not in log_text


def test_submit_jobs_refreshes_only_vasp_jobs_that_need_retry(tmp_path):
    """Migrated VASP folders should reuse results and get current scripts."""
    labeling_dir = tmp_path / "labeling"
    complete = labeling_dir / "calc_complete"
    not_converged = labeling_dir / "calc_not_converged"
    incomplete = labeling_dir / "calc_incomplete"
    not_started = labeling_dir / "calc_not_started"
    old_script = "#!/bin/bash\n#SBATCH --partition=old\n/old/vasp_std\n"
    poscar = """SiC
1.0
3.0 0.0 0.0
0.0 3.0 0.0
0.0 0.0 3.0
Si C
1 1
Direct
0.0 0.0 0.0
0.25 0.25 0.25
"""
    for workdir in (complete, not_converged, incomplete, not_started):
        workdir.mkdir(parents=True)
        (workdir / "POSCAR").write_text(poscar, encoding="utf-8")
    for workdir in (complete, not_converged, incomplete):
        (workdir / "submit.sh").write_text(old_script, encoding="utf-8")
    (complete / "OUTCAR").write_text(
        "General timing and accounting informations for this job:\n",
        encoding="utf-8",
    )
    (not_converged / "OUTCAR").write_text(
        "The electronic self-consistency was not achieved in the given "
        "number of steps\n"
        "General timing and accounting informations for this job:\n",
        encoding="utf-8",
    )
    (incomplete / "OUTCAR").write_text("DAV:  20\n", encoding="utf-8")
    submit_template = tmp_path / "new_sub.sh"
    submit_template.write_text(
        """#!/bin/bash
#SBATCH --job-name=old_name
#SBATCH --ntasks=1
mpirun /old/vasp_std
""",
        encoding="utf-8",
    )
    config_path = tmp_path / "pesmaker.yaml"
    config_path.write_text(
        f"""project: migrated_scf
labeling:
  engine: vasp
  output_dir: {labeling_dir.as_posix()}
  command: /new/vasp_std
jobs:
  submit_command: sbatch
  cores_cpu: 36
  vasp_kpar: 3
  vasp_ncore: 6
  skip_completed: true
  check_scf_convergence: true
  sub_file: {submit_template.as_posix()}
""",
        encoding="utf-8",
    )

    result = submit_jobs(load_config(config_path), dry_run=True)

    assert result.message == (
        "Would submit 3 scf job(s); skipped 1 completed VASP job(s)"
    )
    assert (complete / "submit.sh").read_text(encoding="utf-8") == old_script
    for workdir in (not_converged, incomplete, not_started):
        submit_text = (workdir / "submit.sh").read_text(encoding="utf-8")
        assert f"#SBATCH --job-name={workdir.name}" in submit_text
        assert "#SBATCH --ntasks=36" in submit_text
        assert "mpirun -np 36 /new/vasp_std" in submit_text
        assert "#SBATCH --partition=old" not in submit_text
    log_text = (labeling_dir / "scf_submitted_jobs.txt").read_text(
        encoding="utf-8"
    )
    assert f"SKIPPED completed VASP job: {complete}" in log_text
    assert f"RETRY electronic SCF not converged: {not_converged}" in log_text
    assert f"RETRY incomplete VASP output: {incomplete}" in log_text
    assert log_text.count("REFRESHED submit script:") == 3
    assert log_text.count("DRY-RUN") == 3
    assert not (labeling_dir / "labeling_manifest.jsonl").exists()


def test_submit_jobs_can_ignore_scf_convergence_marker(tmp_path):
    """Users can treat a normally terminated nonconverged OUTCAR as complete."""
    workdir = tmp_path / "labeling" / "calc_nonconverged"
    workdir.mkdir(parents=True)
    old_script = "#!/bin/bash\nkeep this script\n"
    (workdir / "submit.sh").write_text(old_script, encoding="utf-8")
    (workdir / "OUTCAR").write_text(
        "The electronic self-consistency was not achieved in 150 steps\n"
        "General timing and accounting informations for this job:\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "pesmaker.yaml"
    config_path.write_text(
        f"""project: submit_test
labeling:
  engine: vasp
  output_dir: {(tmp_path / 'labeling').as_posix()}
jobs:
  check_scf_convergence: false
""",
        encoding="utf-8",
    )

    result = submit_jobs(load_config(config_path), dry_run=True)

    assert result.message == (
        "Would submit 0 scf job(s); skipped 1 completed VASP job(s)"
    )
    assert (workdir / "submit.sh").read_text(encoding="utf-8") == old_script


def test_submit_jobs_can_resubmit_completed_vasp_jobs(tmp_path):
    """Users should be able to disable completed-job filtering explicitly."""
    workdir = tmp_path / "labeling" / "calc_complete"
    workdir.mkdir(parents=True)
    (workdir / "submit.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    (workdir / "OUTCAR").write_text(
        "General timing and accounting informations for this job:\n",
        encoding="utf-8",
    )
    (tmp_path / "labeling" / "labeling_manifest.jsonl").write_text(
        json.dumps({"index": 0, "workdir": str(workdir)}) + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "pesmaker.yaml"
    config_path.write_text(
        f"""project: submit_test
labeling:
  engine: vasp
  output_dir: {(tmp_path / 'labeling').as_posix()}
jobs:
  submit_command: sbatch
  skip_completed: false
""",
        encoding="utf-8",
    )

    result = submit_jobs(load_config(config_path), dry_run=True)

    assert result.message == "Would submit 1 scf job(s)"
    log_text = (tmp_path / "labeling" / "scf_submitted_jobs.txt").read_text(
        encoding="utf-8"
    )
    assert "DRY-RUN" in log_text
    assert "SKIPPED" not in log_text
    assert (workdir / "submit.sh").read_text(encoding="utf-8") == "#!/bin/bash\n"


def test_submit_jobs_rejects_non_boolean_skip_completed(tmp_path):
    """The completed-job option should not accept ambiguous string values."""
    workdir = tmp_path / "labeling" / "calc"
    workdir.mkdir(parents=True)
    (workdir / "submit.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    config_path = tmp_path / "pesmaker.yaml"
    config_path.write_text(
        f"""project: submit_test
labeling:
  engine: vasp
  output_dir: {(tmp_path / 'labeling').as_posix()}
jobs:
  skip_completed: "false"
""",
        encoding="utf-8",
    )

    try:
        submit_jobs(load_config(config_path), dry_run=True)
    except ValueError as exc:
        assert "jobs.skip_completed must be true or false" in str(exc)
    else:
        raise AssertionError("string skip_completed values should fail")


def test_submit_jobs_rejects_non_boolean_scf_convergence_check(tmp_path):
    """The SCF convergence option should require a YAML boolean."""
    workdir = tmp_path / "labeling" / "calc"
    workdir.mkdir(parents=True)
    (workdir / "submit.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    config_path = tmp_path / "pesmaker.yaml"
    config_path.write_text(
        f"""project: submit_test
labeling:
  engine: vasp
  output_dir: {(tmp_path / 'labeling').as_posix()}
jobs:
  check_scf_convergence: "true"
""",
        encoding="utf-8",
    )

    try:
        submit_jobs(load_config(config_path), dry_run=True)
    except ValueError as exc:
        assert "jobs.check_scf_convergence must be true or false" in str(exc)
    else:
        raise AssertionError("string check_scf_convergence values should fail")


def test_submit_jobs_nohup_uses_bash_and_out_log(tmp_path):
    """Local nohup submission should launch submit.sh with bash and out log."""
    workdir = tmp_path / "sampling" / "md_000000_temp_300K"
    workdir.mkdir(parents=True)
    (workdir / "submit.sh").write_text("#!/bin/bash\ngpumd\n", encoding="utf-8")
    (tmp_path / "sampling" / "sampling_manifest.jsonl").write_text(
        json.dumps({"index": 0, "workdir": str(workdir)}) + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "pesmaker.yaml"
    config_path.write_text(
        f"""project: nohup_submit
sampling:
  engine: gpumd
  output_dir: {(tmp_path / 'sampling').as_posix()}
jobs:
  submit_command: nohup
""",
        encoding="utf-8",
    )

    result = submit_jobs(load_config(config_path), stage="sampling", dry_run=True)

    assert result.message == "Would submit 1 sampling job(s)"
    log = tmp_path / "sampling" / "sampling_submitted_jobs.txt"
    log_text = log.read_text(encoding="utf-8")
    assert "DRY-RUN" in log_text
    assert f"cd {workdir}" in log_text
    assert "nohup bash submit.sh > out 2>&1 &" in log_text


def test_submit_jobs_nohup_starts_background_process(tmp_path, monkeypatch):
    """Real nohup submission should start a detached process and return."""
    workdir = tmp_path / "sampling" / "md_000000_temp_300K"
    workdir.mkdir(parents=True)
    (workdir / "submit.sh").write_text("#!/bin/bash\ngpumd\n", encoding="utf-8")
    (tmp_path / "sampling" / "sampling_manifest.jsonl").write_text(
        json.dumps({"index": 0, "workdir": str(workdir)}) + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "pesmaker.yaml"
    config_path.write_text(
        f"""project: nohup_submit
sampling:
  engine: gpumd
  output_dir: {(tmp_path / 'sampling').as_posix()}
jobs:
  submit_command: nohup
""",
        encoding="utf-8",
    )
    calls = []

    class FakeProcess:
        pid = 12345

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        return FakeProcess()

    monkeypatch.setattr("pesmaker.jobs.submit.subprocess.Popen", fake_popen)

    result = submit_jobs(load_config(config_path), stage="sampling", dry_run=False)

    assert result.message == "Submitted 1 sampling job(s)"
    assert calls
    command, kwargs = calls[0]
    assert command == ["nohup", "bash", "submit.sh"]
    assert kwargs["cwd"] == workdir
    assert kwargs["stderr"] == subprocess.STDOUT
    assert kwargs["start_new_session"] is True
    assert Path(kwargs["stdout"].name) == workdir / "out"
    log_text = (tmp_path / "sampling" / "sampling_submitted_jobs.txt").read_text(
        encoding="utf-8"
    )
    assert "started PID 12345; log: out" in log_text


def test_cli_submit_dry_run_prints_clear_summary(tmp_path, capsys):
    """Dry-run submission output should point users at the command log."""
    workdir = tmp_path / "labeling" / "calc_000000"
    workdir.mkdir(parents=True)
    (workdir / "submit.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    (tmp_path / "labeling" / "labeling_manifest.jsonl").write_text(
        json.dumps({"index": 0, "source": "a.vasp", "workdir": str(workdir)}) + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "pesmaker.yaml"
    config_path.write_text(
        f"""project: submit_test
structures:
  - POSCAR
labeling:
  output_dir: {(tmp_path / 'labeling').as_posix()}
jobs:
  submit_command: sbatch
""",
        encoding="utf-8",
    )

    assert main(["submit", str(config_path), "--dry-run"]) == 0
    output = capsys.readouterr().out
    log = tmp_path / "labeling" / "scf_submitted_jobs.txt"

    assert "Submission preview complete." in output
    assert "Jobs total       : 1" in output
    assert "Jobs completed   : 0" in output
    assert "Jobs to submit   : 1" in output
    assert f"Output directory : {tmp_path / 'labeling'}" in output
    assert f"Log              : {log}" in output
    assert f"Review commands in {log}" in output
    assert f"Submit SCF jobs: pesmaker submit {config_path}" in output
    assert "Files written" not in output


def test_cli_submit_dry_run_reports_all_completed_jobs(tmp_path, capsys):
    """A zero-job preview should still show how many folders are complete."""
    labeling_dir = tmp_path / "labeling"
    for index in range(3):
        workdir = labeling_dir / f"calc_{index:06d}"
        workdir.mkdir(parents=True)
        (workdir / "submit.sh").write_text("#!/bin/bash\n", encoding="utf-8")
        (workdir / "OUTCAR").write_text(
            "General timing and accounting informations for this job:\n",
            encoding="utf-8",
        )
    config_path = tmp_path / "sub.yaml"
    config_path.write_text(
        f"""project: completed_scf
labeling:
  engine: vasp
  output_dir: {labeling_dir.as_posix()}
jobs:
  skip_completed: true
  check_scf_convergence: true
""",
        encoding="utf-8",
    )

    assert main(["submit", str(config_path), "--dry-run"]) == 0
    output = capsys.readouterr().out

    assert "Submission preview complete." in output
    assert "Jobs total       : 3" in output
    assert "Jobs completed   : 3" in output
    assert "Jobs to submit   : 0" in output


def test_cli_submit_prints_collect_next_step(tmp_path, capsys):
    """Real SCF submission output should point to queue checks and collection."""
    log = tmp_path / "labeling" / "scf_submitted_jobs.txt"
    log.parent.mkdir()
    result = StageResult(
        output_dir=tmp_path / "labeling",
        files=(log,),
        message="Submitted 2 scf job(s)",
    )
    config_path = tmp_path / "sub.yaml"

    _print_submit_result(result, config_path=config_path, stage="scf", dry_run=False)
    output = capsys.readouterr().out

    assert "Job submission complete." in output
    assert "Jobs submitted  : 2" in output
    assert f"Log              : {log}" in output
    assert "Check queue: squeue" in output
    assert f"Collect finished results: pesmaker collect {config_path}" in output
    assert f"Submit more prepared SCF jobs: pesmaker submit {config_path}" in output
    assert "--stage scf" not in output


def test_cli_submit_nohup_prints_nvidia_smi_next_step(tmp_path, capsys):
    """Local nohup submission output should point to GPU process checks."""
    log = tmp_path / "sampling" / "sampling_submitted_jobs.txt"
    log.parent.mkdir()
    result = StageResult(
        output_dir=tmp_path / "sampling",
        files=(log,),
        message="Submitted 1 sampling job(s)",
    )
    config_path = tmp_path / "run.yaml"

    _print_submit_result(
        result,
        config_path=config_path,
        stage="sampling",
        dry_run=False,
        submit_command="nohup",
    )
    output = capsys.readouterr().out

    assert "Job submission complete." in output
    assert "Jobs submitted  : 1" in output
    assert "Check GPU process: nvidia-smi" in output
    assert "Check queue: squeue" not in output


def test_sampling_setup_writes_temperature_jobs(tmp_path):
    """GPUMD setup should expand a temperature list into independent jobs."""
    from ase import Atoms
    from ase.io import write

    structure_path = tmp_path / "structure.xyz"
    write(
        structure_path,
        Atoms("Te", positions=[(0.0, 0.0, 0.0)], cell=[3, 3, 20], pbc=True),
        format="extxyz",
    )
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    source_path = tmp_path / "2D_beta_Te.cif"
    source_path.write_text("data_beta\n", encoding="utf-8")
    (generated_dir / "manifest.jsonl").write_text(
        json.dumps({"path": str(structure_path), "source": str(source_path)}) + "\n",
        encoding="utf-8",
    )
    potential = tmp_path / "nep89_20250409.txt"
    potential.write_text("dummy potential\n", encoding="utf-8")
    config_path = tmp_path / "pesmaker.yaml"
    config_path.write_text(
        f"""project: temp_test
structures:
  - {structure_path.as_posix()}
generation:
  output_dir: {generated_dir.as_posix()}
sampling:
  engine: gpumd
  output_dir: {(tmp_path / 'sampling').as_posix()}
  potential: {potential.as_posix()}
  temperatures: [300, 600, 900]
""",
        encoding="utf-8",
    )

    assert main(["sample-setup", str(config_path)]) == 0

    for temperature in (300, 600, 900):
        workdir = tmp_path / "sampling" / f"2D_beta_Te_temp_{temperature}K"
        assert (workdir / "model.xyz").exists()
        assert (workdir / "nep89_20250409.txt").exists()
        run_in = (workdir / "run.in").read_text(encoding="utf-8")
        assert "potential      nep89_20250409.txt" in run_in
        assert f"velocity       {temperature}" in run_in
        assert f"ensemble       npt_scr {temperature} {temperature}" in run_in
    assert len(
        (tmp_path / "sampling" / "sampling_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ) == 3


def test_sampling_setup_resolves_default_nep89_relative_to_gpumd_dir(tmp_path):
    """Omitted potential should resolve to GPUMD's bundled NEP89 path."""
    from ase import Atoms
    from ase.io import write

    gpumd_dir = tmp_path / "GPUMD" / "src"
    gpumd_dir.mkdir(parents=True)
    potential = (
        tmp_path
        / "GPUMD"
        / "potentials"
        / "nep"
        / "nep89_20250409"
        / "nep89_20250409.txt"
    )
    potential.parent.mkdir(parents=True)
    potential.write_text("dummy nep89 potential\n", encoding="utf-8")
    structure_path = tmp_path / "structure.xyz"
    write(
        structure_path,
        Atoms("Te", positions=[(0.0, 0.0, 0.0)], cell=[3, 3, 20], pbc=True),
        format="extxyz",
    )
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    (generated_dir / "manifest.jsonl").write_text(
        json.dumps({"path": str(structure_path)}) + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "pesmaker.yaml"
    config_path.write_text(
        f"""project: default_nep89
generation:
  output_dir: {generated_dir.as_posix()}
sampling:
  engine: gpumd
  output_dir: {(tmp_path / 'sampling').as_posix()}
  gpumd_dir: {gpumd_dir.as_posix()}
  temperature: 300
""",
        encoding="utf-8",
    )

    assert main(["sample-setup", str(config_path)]) == 0

    workdir = tmp_path / "sampling" / "structure_temp_300K"
    assert (workdir / "nep89_20250409.txt").exists()
    run_in = (workdir / "run.in").read_text(encoding="utf-8")
    assert "potential      nep89_20250409.txt" in run_in
    submit = (workdir / "submit.sh").read_text(encoding="utf-8")
    assert str(gpumd_dir / "gpumd") in submit
    assert "#SBATCH --ntasks" not in submit
    assert "#SBATCH --cpus-per-task" not in submit


def test_sampling_setup_preserves_gpumd_submit_template_resources(tmp_path):
    """GPUMD user submit templates should not be rewritten with CPU resources."""
    from ase import Atoms
    from ase.io import write

    structure_path = tmp_path / "structure.xyz"
    write(
        structure_path,
        Atoms("Te", positions=[(0.0, 0.0, 0.0)], cell=[3, 3, 20], pbc=True),
        format="extxyz",
    )
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    (generated_dir / "manifest.jsonl").write_text(
        json.dumps({"path": str(structure_path)}) + "\n",
        encoding="utf-8",
    )
    gpumd_template = tmp_path / "gpumd.sh"
    gpumd_template.write_text(
        """#!/bin/bash
#SBATCH --job-name=user_gpumd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

cd "{workdir}"
{command}
""",
        encoding="utf-8",
    )
    config_path = tmp_path / "pesmaker.yaml"
    config_path.write_text(
        f"""project: preserve_gpumd_template
generation:
  output_dir: {generated_dir.as_posix()}
sampling:
  engine: gpumd
  output_dir: {(tmp_path / 'sampling').as_posix()}
  command: /opt/gpumd/gpumd
  temperature: 300
jobs:
  submit_command: sbatch
  nodes: 2
  cores_cpu: 64
  gpus: 4
  sub_file:
    sampling: {gpumd_template.as_posix()}
""",
        encoding="utf-8",
    )

    assert main(["sample-setup", str(config_path)]) == 0

    workdir = tmp_path / "sampling" / "structure_temp_300K"
    assert (workdir / "gpumd.sh").exists()
    assert (workdir / "submit.sh").exists()
    submit = (workdir / "gpumd.sh").read_text(encoding="utf-8")
    assert "#SBATCH --job-name=user_gpumd" in submit
    assert "#SBATCH --ntasks=1" in submit
    assert "#SBATCH --gres=gpu:1" in submit
    assert "#SBATCH --ntasks=128" not in submit
    assert "#SBATCH --gres=gpu:4" not in submit
    assert f'cd "{workdir}"' in submit
    assert "/opt/gpumd/gpumd" in submit
    manifest = json.loads(
        (tmp_path / "sampling" / "sampling_manifest.jsonl").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["submit_script"] == str(workdir / "gpumd.sh")


def test_submit_jobs_bash_uses_rendered_gpumd_template_name(tmp_path):
    """Local bash submission should run the rendered GPUMD template by name."""
    from ase import Atoms
    from ase.io import write

    structure_path = tmp_path / "structure.xyz"
    write(
        structure_path,
        Atoms("Te", positions=[(0.0, 0.0, 0.0)], cell=[3, 3, 20], pbc=True),
        format="extxyz",
    )
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    (generated_dir / "manifest.jsonl").write_text(
        json.dumps({"path": str(structure_path)}) + "\n",
        encoding="utf-8",
    )
    gpumd_template = tmp_path / "gpumd.sh"
    gpumd_template.write_text(
        """#!/bin/bash
{command}
""",
        encoding="utf-8",
    )
    config_path = tmp_path / "pesmaker.yaml"
    config_path.write_text(
        f"""project: bash_gpumd_template
generation:
  output_dir: {generated_dir.as_posix()}
sampling:
  engine: gpumd
  output_dir: {(tmp_path / 'sampling').as_posix()}
  command: /opt/gpumd/gpumd
  temperature: 300
jobs:
  submit_command: bash
  sub_file: {gpumd_template.as_posix()}
""",
        encoding="utf-8",
    )

    assert main(["sample-setup", str(config_path)]) == 0
    result = submit_jobs(load_config(config_path), stage="sampling", dry_run=True)

    assert result.message == "Would submit 1 sampling job(s)"
    workdir = tmp_path / "sampling" / "structure_temp_300K"
    assert (workdir / "gpumd.sh").exists()
    assert (workdir / "submit.sh").exists()
    log = tmp_path / "sampling" / "sampling_submitted_jobs.txt"
    assert f"DRY-RUN (cd {workdir} && bash gpumd.sh)" in log.read_text(
        encoding="utf-8"
    )
    assert "bash submit.sh" not in log.read_text(encoding="utf-8")


def test_mace_sampling_setup_writes_lammps_inputs_and_preserves_submit_template(
    tmp_path,
):
    """MACE sampling should render LAMMPS inputs while preserving lammps.sh."""
    from ase import Atoms
    from ase.io import write

    structure_path = tmp_path / "structure.xyz"
    write(
        structure_path,
        Atoms(
            symbols=["Te", "Pd", "Te"],
            positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            cell=[5.0, 5.0, 5.0],
            pbc=True,
        ),
        format="extxyz",
    )
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    (generated_dir / "manifest.jsonl").write_text(
        json.dumps({"path": str(structure_path)}) + "\n",
        encoding="utf-8",
    )
    model = tmp_path / "mace-omat-0-small.model-mliap_lammps.pt"
    model.write_text("fake model\n", encoding="utf-8")
    run_template = tmp_path / "in.run_mace_npt"
    run_template.write_text(
        """variable Tstart equal {temperature_start}
variable Tstop equal {temperature_end}
variable Tdamp equal ${ts}*100
read_data {data_file}
pair_style mliap unified {potential} 0
pair_coeff * * {elements}
dump myDump all custom 3000 {trajectory} id element x y z
dump_modify myDump sort id element {elements}
fix MD all npt temp ${Tstart} ${Tstop} ${Tdamp} x 0 0 ${Pdamp} y 0 0 ${Pdamp} z 0 0 ${Pdamp} couple none
""",
        encoding="utf-8",
    )
    lammps_template = tmp_path / "lammps.sh"
    lammps_template.write_text(
        """#!/bin/bash
#SBATCH --job-name=user_lammps
#SBATCH --ntasks=1
export CUDA_VISIBLE_DEVICES=0
export MACE_TIME=true
mpirun -np 1 /path/to/lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.run_mace_npt
""",
        encoding="utf-8",
    )
    config_path = tmp_path / "pesmaker.yaml"
    config_path.write_text(
        f"""project: mace_sampling
generation:
  output_dir: {generated_dir.as_posix()}
sampling:
  engine: mace
  output_dir: {(tmp_path / 'sampling').as_posix()}
  potential: {model.as_posix()}
  run_in: {run_template.as_posix()}
  temperature: 300-1200
  trajectory: mace.lammpstrj
jobs:
  submit_command: nohup
  sub_file: {lammps_template.as_posix()}
""",
        encoding="utf-8",
    )

    assert main(["sample-setup", str(config_path)]) == 0

    workdir = tmp_path / "sampling" / "structure_ramp_300K_to_1200K"
    data_in = (workdir / "data.in").read_text(encoding="utf-8")
    run_in = (workdir / "in.run_mace_npt").read_text(encoding="utf-8")
    submit = (workdir / "lammps.sh").read_text(encoding="utf-8")

    assert "1      127.599" in data_in
    assert "# Te" in data_in
    assert "# Pd" in data_in
    assert "variable Tdamp equal ${ts}*100" in run_in
    assert f"pair_style    mliap unified {model.resolve()} 0" in run_in
    assert "pair_coeff    * * Te Pd" in run_in
    assert "dump_modify  myDump  sort  id  element  Te Pd" in run_in
    assert "mace.lammpstrj" in run_in
    assert "#SBATCH --job-name=user_lammps" in submit
    assert "#SBATCH --ntasks=1" in submit
    assert "#SBATCH --ntasks=36" not in submit
    assert (workdir / "submit.sh").read_text(encoding="utf-8") == submit
    manifest = json.loads(
        (tmp_path / "sampling" / "sampling_manifest.jsonl").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["engine"] == "mace"
    assert manifest["elements"] == ["Te", "Pd"]
    assert manifest["data_file"] == str(workdir / "data.in")
    assert manifest["run_in"] == str(workdir / "in.run_mace_npt")
    assert manifest["submit_script"] == str(workdir / "lammps.sh")

    result = submit_jobs(load_config(config_path), stage="sampling", dry_run=True)
    assert result.message == "Would submit 1 sampling job(s)"
    log = tmp_path / "sampling" / "sampling_submitted_jobs.txt"
    assert f"DRY-RUN (cd {workdir} && nohup bash lammps.sh > out 2>&1 &)" in (
        log.read_text(encoding="utf-8")
    )


def test_mace_sampling_renders_d3_template_without_special_yaml(tmp_path):
    """D3 is controlled by the user's LAMMPS template, not extra YAML fields."""
    from ase import Atoms
    from ase.io import write

    structure_path = tmp_path / "structure.xyz"
    write(
        structure_path,
        Atoms(
            "Te",
            positions=[(0.0, 0.0, 0.0)],
            cell=[5.0, 5.0, 5.0],
            pbc=True,
        ),
        format="extxyz",
    )
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    (generated_dir / "manifest.jsonl").write_text(
        json.dumps({"path": str(structure_path)}) + "\n",
        encoding="utf-8",
    )
    run_template = tmp_path / "in.run_mace_d3"
    run_template.write_text(
        """read_data {data_file}
pair_style hybrid/overlay &
           mliap unified {potential} 0 &
           dispersion/d3 bj pbe 12.0 6.0
pair_coeff * * mliap {elements}
pair_coeff * * dispersion/d3 {elements}
""",
        encoding="utf-8",
    )
    lammps_template = tmp_path / "lammps.sh"
    lammps_template.write_text("#!/bin/bash\n/path/to/lmp -in in.run_mace_d3\n")
    config_path = tmp_path / "pesmaker.yaml"
    config_path.write_text(
        f"""project: mace_d3
generation:
  output_dir: {generated_dir.as_posix()}
sampling:
  engine: lammps_mace
  output_dir: {(tmp_path / 'sampling').as_posix()}
  potential: /models/mace-omat-0-small.model-mliap_lammps.pt
  run_in: {run_template.as_posix()}
  temperature: 300
jobs:
  sub_file: {lammps_template.as_posix()}
""",
        encoding="utf-8",
    )

    assert main(["sample-setup", str(config_path)]) == 0

    run_in = (
        tmp_path / "sampling" / "structure_temp_300K" / "in.run_mace_d3"
    ).read_text(encoding="utf-8")
    assert "mliap unified /models/mace-omat-0-small.model-mliap_lammps.pt 0" in run_in
    assert "pair_coeff    * * mliap Te" in run_in
    assert "pair_coeff    * * dispersion/d3 Te" in run_in


def test_mace_sampling_rewrites_common_literal_lammps_template(tmp_path):
    """Existing MACE/LAMMPS inputs should be adjusted even without placeholders."""
    from ase import Atoms
    from ase.io import write

    structure_path = tmp_path / "structure.xyz"
    write(
        structure_path,
        Atoms(
            symbols=["Te", "Pd", "Te"],
            positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            cell=[5.0, 5.0, 5.0],
            pbc=True,
        ),
        format="extxyz",
    )
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    (generated_dir / "manifest.jsonl").write_text(
        json.dumps({"path": str(structure_path)}) + "\n",
        encoding="utf-8",
    )
    model = tmp_path / "mace-omat-0-small.model-mliap_lammps.pt"
    model.write_text("fake model\n", encoding="utf-8")
    run_template = tmp_path / "in.run_mace_npt"
    run_template.write_text(
        """units metal
variable      T          equal  300
variable      Tdamp      equal ${ts}*100
variable      Pdamp      equal ${ts}*1000
variable      P_0        equal 0.0
read_data     old.data
pair_style    hybrid/overlay &
              mliap unified ../foundation-model/mace-omat-0-small.model-mliap_lammps.pt 0 &
              dispersion/d3 bj pbe 12.0 6.0
pair_coeff    * * mliap &
              Mn Cr Fe Co Ni Cu Ag
pair_coeff    * * dispersion/d3 &
              Mn Cr Fe Co Ni Cu Ag
dump_modify  myDump  sort id  element  Mn Cr Fe Co Ni Cu Ag
velocity     all create  ${T} 123456  mom  yes  rot  yes  dist  gaussian
fix          NVT all nvt temp ${T} ${T} ${Tdamp}
unfix        NVT
""",
        encoding="utf-8",
    )
    lammps_template = tmp_path / "run.sh"
    lammps_template.write_text("#!/bin/bash\n/path/to/lmp -in in.run_mace_npt\n")
    config_path = tmp_path / "pesmaker.yaml"
    config_path.write_text(
        f"""project: mace_literal
generation:
  output_dir: {generated_dir.as_posix()}
sampling:
  engine: mace
  output_dir: {(tmp_path / 'sampling').as_posix()}
  potential: {model.as_posix()}
  run_in: {run_template.as_posix()}
  temperature: 300-900
jobs:
  submit_command: nohup
  sub_file: {lammps_template.as_posix()}
""",
        encoding="utf-8",
    )

    assert main(["sample-setup", str(config_path)]) == 0

    run_in = (
        tmp_path / "sampling" / "structure_ramp_300K_to_900K" / "in.run_mace_npt"
    ).read_text(encoding="utf-8")
    assert "variable      T          equal  300" in run_in
    assert "variable      Tstart     equal  300" in run_in
    assert "variable      Tstop      equal  900" in run_in
    assert "read_data     data.in" in run_in
    assert f"mliap unified {model.resolve()} 0 &" in run_in
    assert "pair_coeff    * * mliap Te Pd" in run_in
    assert "pair_coeff    * * dispersion/d3 Te Pd" in run_in
    assert "dump_modify  myDump  sort  id  element  Te Pd" in run_in
    assert "velocity  all  create  ${Tstart}" in run_in
    assert "fix          NVT all npt temp ${Tstart} ${Tstop} ${Tdamp}" in run_in
    assert "all nvt" not in run_in
    assert "unfix        NVT" in run_in


def test_mace_sampling_can_preserve_user_run_in_verbatim(tmp_path):
    """Users can opt out of all MACE run input rewriting."""
    from ase import Atoms
    from ase.io import write

    structure_path = tmp_path / "structure.xyz"
    write(
        structure_path,
        Atoms("Te", positions=[(0.0, 0.0, 0.0)], cell=[5.0, 5.0, 5.0], pbc=True),
        format="extxyz",
    )
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    (generated_dir / "manifest.jsonl").write_text(
        json.dumps({"path": str(structure_path)}) + "\n",
        encoding="utf-8",
    )
    run_template = tmp_path / "in.run_mace_npt"
    literal_text = """read_data old.data
pair_style mliap unified old.model 0
pair_coeff * * Mn Cr Fe
velocity all create 300 123456
fix NVT all nvt temp 300 300 ${Tdamp}
"""
    run_template.write_text(literal_text, encoding="utf-8")
    lammps_template = tmp_path / "run.sh"
    lammps_template.write_text("#!/bin/bash\n/path/to/lmp -in in.run_mace_npt\n")
    config_path = tmp_path / "pesmaker.yaml"
    config_path.write_text(
        f"""project: mace_preserve
generation:
  output_dir: {generated_dir.as_posix()}
sampling:
  engine: mace
  output_dir: {(tmp_path / 'sampling').as_posix()}
  potential: /models/new.model
  run_in: {run_template.as_posix()}
  preserve_run_in: true
  temperature: 300-900
jobs:
  sub_file: {lammps_template.as_posix()}
""",
        encoding="utf-8",
    )

    assert main(["sample-setup", str(config_path)]) == 0

    workdir = tmp_path / "sampling" / "structure_ramp_300K_to_900K"
    assert (workdir / "data.in").exists()
    assert (workdir / "in.run_mace_npt").read_text(encoding="utf-8") == literal_text


def test_gpumd_sampling_can_preserve_user_run_in_verbatim(tmp_path):
    """Users can opt out of all GPUMD run.in rewriting."""
    from ase import Atoms
    from ase.io import write

    structure_path = tmp_path / "structure.xyz"
    write(
        structure_path,
        Atoms("Te", positions=[(0.0, 0.0, 0.0)], cell=[5.0, 5.0, 5.0], pbc=True),
        format="extxyz",
    )
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    (generated_dir / "manifest.jsonl").write_text(
        json.dumps({"path": str(structure_path)}) + "\n",
        encoding="utf-8",
    )
    run_template = tmp_path / "run.in"
    literal_text = """potential old_nep.txt
velocity 100
ensemble nvt 100 100 100
run 42
"""
    run_template.write_text(literal_text, encoding="utf-8")
    config_path = tmp_path / "pesmaker.yaml"
    config_path.write_text(
        f"""project: gpumd_preserve
generation:
  output_dir: {generated_dir.as_posix()}
sampling:
  engine: gpumd
  output_dir: {(tmp_path / 'sampling').as_posix()}
  potential: new_nep.txt
  run_in: {run_template.as_posix()}
  preserve_run_in: true
  temperature: 300-900
""",
        encoding="utf-8",
    )

    assert main(["sample-setup", str(config_path)]) == 0

    run_in = (
        tmp_path / "sampling" / "structure_ramp_300K_to_900K" / "run.in"
    ).read_text(encoding="utf-8")
    assert run_in == literal_text


def test_sampling_setup_writes_temperature_ramp(tmp_path):
    """A temperature range should produce one ramp MD job."""
    from ase import Atoms
    from ase.io import write

    structure_path = tmp_path / "structure.xyz"
    write(
        structure_path,
        Atoms("Te", positions=[(0.0, 0.0, 0.0)], cell=[3, 3, 20], pbc=True),
        format="extxyz",
    )
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    (generated_dir / "manifest.jsonl").write_text(
        json.dumps({"path": str(structure_path)}) + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "pesmaker.yaml"
    config_path.write_text(
        f"""project: ramp_test
structures:
  - {structure_path.as_posix()}
generation:
  output_dir: {generated_dir.as_posix()}
sampling:
  engine: gpumd
  output_dir: {(tmp_path / 'sampling').as_posix()}
  potential: nep89_20250409.txt
  temperatures: [300-1500]
""",
        encoding="utf-8",
    )

    assert main(["sample-setup", str(config_path)]) == 0

    workdir = tmp_path / "sampling" / "structure_ramp_300K_to_1500K"
    run_in = (workdir / "run.in").read_text(encoding="utf-8")
    assert "velocity       300" in run_in
    assert "ensemble       npt_scr 300 1500" in run_in


def test_sampling_setup_preserves_user_run_line_and_warns_for_triclinic_npt_scr(
    tmp_path,
    capsys,
):
    """A user run.in should keep its run count while fixing triclinic npt_scr."""
    from ase import Atoms
    from ase.io import write

    structure_path = tmp_path / "structure.xyz"
    write(
        structure_path,
        Atoms(
            "Te2",
            positions=[(0.0, 0.0, 0.0), (2.0, 2.0, 2.0)],
            cell=[(3.0, 0.0, 0.0), (1.0, 3.0, 0.0), (0.0, 0.0, 3.0)],
            pbc=True,
        ),
        format="extxyz",
    )
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    (generated_dir / "manifest.jsonl").write_text(
        json.dumps({"path": str(structure_path)}) + "\n",
        encoding="utf-8",
    )
    run_template = tmp_path / "run.in"
    run_template.write_text(
        """potential      {potential}
velocity       {temperature_start}

ensemble       npt_scr {temperature_start} {temperature_end} 100 0 0 0 50 50 50 1000
time_step      1
dump_thermo    1000
dump_position  4000
run            4000000
""",
        encoding="utf-8",
    )
    config_path = tmp_path / "pesmaker.yaml"
    config_path.write_text(
        f"""project: user_run_in
generation:
  output_dir: {generated_dir.as_posix()}
sampling:
  engine: gpumd
  output_dir: {(tmp_path / 'sampling').as_posix()}
  potential: nep89_20250409.txt
  temperature: 300-1200
  run_in: {run_template.as_posix()}
""",
        encoding="utf-8",
    )

    assert main(["sample-setup", str(config_path)]) == 0
    output = capsys.readouterr().out

    run_in = (
        tmp_path / "sampling" / "structure_ramp_300K_to_1200K" / "run.in"
    ).read_text(encoding="utf-8")
    assert (
        "ensemble       npt_scr 300 1200 100 0 0 0 0 0 0 "
        "50 50 50 50 50 50 1000"
    ) in run_in
    assert "run            4000000" in run_in
    assert "run            3000000" not in run_in
    assert "GPUMD run.in npt_scr was adjusted for triclinic cell format." in output


def test_sampling_setup_selects_gpumd_ensemble_by_cell_shape(tmp_path):
    """GPUMD run.in generation should adapt NPT parameters to cell shape."""
    from ase import Atoms
    from ase.io import write

    structures = [
        Atoms(
            "Te2",
            positions=[(0.0, 0.0, 0.0), (2.0, 2.0, 2.0)],
            cell=[3.0, 3.0, 3.0],
            pbc=True,
        ),
        Atoms(
            "Te2",
            positions=[(0.0, 0.0, 0.0), (2.0, 2.0, 2.0)],
            cell=[(3.0, 0.0, 0.0), (1.0, 3.0, 0.0), (0.0, 0.0, 3.0)],
            pbc=True,
        ),
        Atoms(
            "Te2",
            positions=[(0.0, 0.0, 10.0), (1.0, 1.0, 11.0)],
            cell=[5.0, 5.0, 40.0],
            pbc=[True, True, False],
        ),
        Atoms(
            "Te2",
            positions=[(0.0, 0.0, 10.0), (1.0, 1.0, 11.0)],
            cell=[(5.0, 0.0, 0.0), (1.0, 5.0, 0.0), (0.0, 0.0, 40.0)],
            pbc=[True, True, False],
        ),
    ]
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    manifest_lines = []
    for index, atoms in enumerate(structures):
        path = tmp_path / f"structure_{index}.xyz"
        write(path, atoms, format="extxyz")
        manifest_lines.append(json.dumps({"path": str(path)}))
    (generated_dir / "manifest.jsonl").write_text(
        "\n".join(manifest_lines) + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "pesmaker.yaml"
    config_path.write_text(
        f"""project: ensemble_modes
generation:
  output_dir: {generated_dir.as_posix()}
sampling:
  engine: gpumd
  output_dir: {(tmp_path / 'sampling').as_posix()}
  temperatures: [300]
""",
        encoding="utf-8",
    )

    assert main(["sample-setup", str(config_path)]) == 0

    orthogonal = (
        tmp_path / "sampling" / "structure_0_temp_300K" / "run.in"
    ).read_text(encoding="utf-8")
    triclinic = (
        tmp_path / "sampling" / "structure_1_temp_300K" / "run.in"
    ).read_text(encoding="utf-8")
    two_dimensional = (
        tmp_path / "sampling" / "structure_2_temp_300K" / "run.in"
    ).read_text(encoding="utf-8")
    two_dimensional_triclinic = (
        tmp_path / "sampling" / "structure_3_temp_300K" / "run.in"
    ).read_text(encoding="utf-8")

    assert "ensemble       npt_scr 300 300 100 0 0 0 50 50 50 1000" in orthogonal
    assert (
        "ensemble       npt_scr 300 300 100 0 0 0 0 0 0 "
        "50 50 50 50 50 50 1000"
    ) in triclinic
    assert (
        "ensemble       npt_scr 300 300 100 0 0 0 50 50 500 1000"
        in two_dimensional
    )
    assert (
        "ensemble       npt_scr 300 300 100 0 0 0 0 0 0 "
        "50 50 500 500 500 50 1000"
    ) in two_dimensional_triclinic
    assert "dump_position  1000" in two_dimensional_triclinic
    assert "run            1000000" in two_dimensional_triclinic


def test_sampling_setup_respects_run_steps_and_forced_gpumd_cell_mode(tmp_path):
    """Users can force the GPUMD NPT cell mode and total MD step count."""
    from ase import Atoms
    from ase.io import write

    structure_path = tmp_path / "structure.xyz"
    write(
        structure_path,
        Atoms(
            "Te2",
            positions=[(0.0, 0.0, 0.0), (2.0, 2.0, 2.0)],
            cell=[3.0, 3.0, 3.0],
            pbc=True,
        ),
        format="extxyz",
    )
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    (generated_dir / "manifest.jsonl").write_text(
        json.dumps({"path": str(structure_path)}) + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "pesmaker.yaml"
    config_path.write_text(
        f"""project: forced_mode
generation:
  output_dir: {generated_dir.as_posix()}
sampling:
  engine: gpumd
  output_dir: {(tmp_path / 'sampling').as_posix()}
  temperature: 300
  ensemble_mode: triclinic
  run_steps: 12345
""",
        encoding="utf-8",
    )

    assert main(["sample-setup", str(config_path)]) == 0

    run_in = (
        tmp_path / "sampling" / "structure_temp_300K" / "run.in"
    ).read_text(encoding="utf-8")
    assert (
        "ensemble       npt_scr 300 300 100 0 0 0 0 0 0 "
        "50 50 50 50 50 50 1000"
    ) in run_in
    assert "run            12345" in run_in


def test_select_uses_farthest_point_sampling(tmp_path, monkeypatch, capsys):
    """Selection should write a compact extxyz file and manifest."""
    from ase import Atoms
    from ase.io import write

    frames = [
        Atoms("Te", positions=[(0.0, 0.0, 0.0)], cell=[3, 3, 20], pbc=True),
        Atoms("Te", positions=[(0.1, 0.0, 0.0)], cell=[3, 3, 20], pbc=True),
        Atoms("Te", positions=[(1.5, 0.0, 0.0)], cell=[3, 3, 20], pbc=True),
    ]
    trajectory = tmp_path / "movie.xyz"
    write(trajectory, frames, format="extxyz")
    config_path = tmp_path / "pesmaker.yaml"
    selected_dir = tmp_path / "selected"
    config_path.write_text(
        f"""project: select_test
structures:
  - {trajectory.as_posix()}
sampling:
  engine: gpumd
  selection:
    trajectory_pattern: {trajectory.as_posix()}
    output_dir: {selected_dir.as_posix()}
    descriptor: simple
    min_distance: 0.2
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    assert main(["select", str(config_path)]) == 0
    output = capsys.readouterr().out

    assert (selected_dir / "selected.xyz").exists()
    assert (selected_dir / "selection_features.npy").exists()
    assert (selected_dir / "fps_selection.png").exists()
    from PIL import Image

    assert Image.open(selected_dir / "fps_selection.png").info["dpi"][0] >= 590
    assert list(selected_dir.glob("selected_*.xyz")) == []
    assert "Trajectory selection" in output
    assert "Mode             : separate_trajectories" in output
    assert "Trajectories     : 1" in output
    assert "Descriptor       : simple geometry" in output
    assert "Trajectory       : 1/1" in output
    assert "Selection completed: Selected 2 of 3 frame(s) from 1 trajectory file(s)." in output
    assert "Selected 2 of 3 MD frame(s)" in output
    assert "Selection stopped because remaining frames are closer" in output
    assert "edit your YAML under sampling.selection" in output
    assert "min_distance: 0.12" in output
    records = [
        json.loads(line)
        for line in (selected_dir / "manifest.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert len(records) == 2
    assert {record["path"] for record in records} == {str(selected_dir / "selected.xyz")}
    assert [record["frame_index"] for record in records] == [0, 1]
    assert [record["source_frame"] for record in records] == [0, 2]
    assert {record["descriptor"] for record in records} == {"simple"}


def test_select_uses_interval_sampling_without_descriptor_model(
    tmp_path, monkeypatch, capsys
):
    """Interval selection should not require a NEP potential or descriptors."""
    from ase import Atoms
    from ase.io import write

    frames = [
        Atoms("Te", positions=[(float(index), 0.0, 0.0)], cell=[10, 10, 10], pbc=True)
        for index in range(7)
    ]
    trajectory = tmp_path / "XDATCAR"
    write(trajectory, frames, format="vasp-xdatcar")
    config_path = tmp_path / "pesmaker.yaml"
    selected_dir = tmp_path / "selected"
    config_path.write_text(
        f"""project: interval_select_test
sampling:
  selection:
    method: interval
    trajectory_pattern: {trajectory.as_posix()}
    output_dir: {selected_dir.as_posix()}
    interval: 3
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    assert main(["select", str(config_path)]) == 0
    output = capsys.readouterr().out

    assert (selected_dir / "selected.xyz").exists()
    assert (selected_dir / "manifest.jsonl").exists()
    assert not (selected_dir / "selection_features.npy").exists()
    assert not (selected_dir / "fps_selection.png").exists()
    assert "Trajectory selection" in output
    assert "Trajectories     : 1" in output
    assert "Method           : interval sampling" in output
    assert "Trajectory       : 1/1" in output
    assert "Selection completed: Selected 3 of 7 frame(s) from 1 trajectory file(s)." in output
    assert "Selected 3 of 7 MD frame(s) using interval sampling every 3 frame(s)" in output
    records = [
        json.loads(line)
        for line in (selected_dir / "manifest.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert [record["source_frame"] for record in records] == [0, 3, 6]
    assert [record["frame_index"] for record in records] == [0, 1, 2]
    assert {record["selection_method"] for record in records} == {"interval"}
    assert {record["interval"] for record in records} == {3}


def test_select_samples_each_trajectory_separately_by_default(
    tmp_path, monkeypatch, capsys
):
    """Multiple trajectory files should apply max_count per trajectory by default."""
    from ase import Atoms
    from ase.io import write

    trajectory_a = tmp_path / "mp-1_temp_300K" / "movie.xyz"
    trajectory_b = tmp_path / "mp-2_temp_300K" / "movie.xyz"
    trajectory_a.parent.mkdir()
    trajectory_b.parent.mkdir()
    frames_a = [
        Atoms("Te", positions=[(0.0, 0.0, 0.0)], cell=[3, 3, 20], pbc=True),
        Atoms("Te", positions=[(0.1, 0.0, 0.0)], cell=[3, 3, 20], pbc=True),
        Atoms("Te", positions=[(1.5, 0.0, 0.0)], cell=[3, 3, 20], pbc=True),
    ]
    frames_b = [
        Atoms("Te", positions=[(0.0, 0.0, 0.0)], cell=[4, 4, 20], pbc=True),
        Atoms("Te", positions=[(0.2, 0.0, 0.0)], cell=[4, 4, 20], pbc=True),
        Atoms("Te", positions=[(2.5, 0.0, 0.0)], cell=[4, 4, 20], pbc=True),
    ]
    write(trajectory_a, frames_a, format="extxyz")
    write(trajectory_b, frames_b, format="extxyz")
    selected_dir = tmp_path / "selected"
    config_path = tmp_path / "pesmaker.yaml"
    config_path.write_text(
        f"""project: separate_select_test
sampling:
  engine: gpumd
  selection:
    trajectory_pattern: {(tmp_path / '*' / 'movie.xyz').as_posix()}
    output_dir: {selected_dir.as_posix()}
    descriptor: simple
    min_distance: 0.2
    max_count: 2
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    assert main(["select", str(config_path)]) == 0
    output = capsys.readouterr().out

    assert "Trajectory selection" in output
    assert "Mode             : separate_trajectories" in output
    assert "Trajectories     : 2" in output
    assert "Per trajectory   : max_count=2, min_distance=0.2" in output
    assert "Selection completed: Selected 4 of 6 frame(s)" in output
    assert (selected_dir / "mp-1_temp_300K" / "selected.xyz").exists()
    assert (selected_dir / "mp-2_temp_300K" / "selected.xyz").exists()
    assert (selected_dir / "mp-1_temp_300K" / "selection_features.npy").exists()
    assert (selected_dir / "mp-2_temp_300K" / "selection_features.npy").exists()
    records = [
        json.loads(line)
        for line in (selected_dir / "manifest.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert len(records) == 4
    assert [record["index"] for record in records] == [0, 1, 2, 3]
    assert [record["source_frame"] for record in records] == [0, 2, 0, 2]
    assert [record["frame_index"] for record in records] == [0, 1, 0, 1]
    assert {record["trajectory_name"] for record in records} == {
        "mp-1_temp_300K",
        "mp-2_temp_300K",
    }
    assert {record["descriptor"] for record in records} == {"simple"}


def test_select_can_combine_multiple_trajectories_when_requested(
    tmp_path, monkeypatch, capsys
):
    """Users can still opt into one global selection over all matched frames."""
    from ase import Atoms
    from ase.io import write

    trajectory_a = tmp_path / "traj_a" / "movie.xyz"
    trajectory_b = tmp_path / "traj_b" / "movie.xyz"
    trajectory_a.parent.mkdir()
    trajectory_b.parent.mkdir()
    write(
        trajectory_a,
        [
            Atoms("Te", positions=[(0.0, 0.0, 0.0)], cell=[5, 5, 5], pbc=True),
            Atoms("Te", positions=[(1.0, 0.0, 0.0)], cell=[5, 5, 5], pbc=True),
        ],
        format="extxyz",
    )
    write(
        trajectory_b,
        [
            Atoms("Te", positions=[(2.0, 0.0, 0.0)], cell=[5, 5, 5], pbc=True),
            Atoms("Te", positions=[(3.0, 0.0, 0.0)], cell=[5, 5, 5], pbc=True),
        ],
        format="extxyz",
    )
    selected_dir = tmp_path / "selected"
    config_path = tmp_path / "pesmaker.yaml"
    config_path.write_text(
        f"""project: combined_select_test
sampling:
  engine: gpumd
  selection:
    trajectory_pattern: {(tmp_path / 'traj_*' / 'movie.xyz').as_posix()}
    output_dir: {selected_dir.as_posix()}
    separate_trajectories: false
    descriptor: simple
    min_distance: 0.0
    max_count: 2
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    assert main(["select", str(config_path)]) == 0
    output = capsys.readouterr().out

    assert "Trajectory selection" not in output
    assert "Selected 2 of 4 MD frame(s)" in output
    assert (selected_dir / "selected.xyz").exists()
    assert not (selected_dir / "traj_a" / "selected.xyz").exists()
    records = [
        json.loads(line)
        for line in (selected_dir / "manifest.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert len(records) == 2
    assert {record["path"] for record in records} == {str(selected_dir / "selected.xyz")}


def test_separate_gpumd_select_prints_descriptor_summary_once(
    tmp_path, monkeypatch, capsys
):
    """Separate GPUMD selection should avoid repeated model header blocks."""
    import sys
    import types

    from ase import Atoms
    from ase.io import write

    def fake_get_descriptors(atoms, *, model_filename):
        x_mean = float(np.mean(atoms.get_positions()[:, 0]))
        return np.array([[x_mean, x_mean**2]])

    calorine_module = types.ModuleType("calorine")
    nep_module = types.ModuleType("calorine.nep")
    nep_module.get_descriptors = fake_get_descriptors
    monkeypatch.setitem(sys.modules, "calorine", calorine_module)
    monkeypatch.setitem(sys.modules, "calorine.nep", nep_module)

    trajectory_a = tmp_path / "traj_a" / "movie.xyz"
    trajectory_b = tmp_path / "traj_b" / "movie.xyz"
    trajectory_a.parent.mkdir()
    trajectory_b.parent.mkdir()
    frames_a = [
        Atoms("Te", positions=[(0.0, 0.0, 0.0)], cell=[5, 5, 5], pbc=True),
        Atoms("Te", positions=[(1.0, 0.0, 0.0)], cell=[5, 5, 5], pbc=True),
    ]
    frames_b = [
        Atoms("Te", positions=[(2.0, 0.0, 0.0)], cell=[5, 5, 5], pbc=True),
        Atoms("Te", positions=[(3.0, 0.0, 0.0)], cell=[5, 5, 5], pbc=True),
    ]
    write(trajectory_a, frames_a, format="extxyz")
    write(trajectory_b, frames_b, format="extxyz")
    potential = tmp_path / "nep89.txt"
    potential.write_text("fake potential\n", encoding="utf-8")
    config_path = tmp_path / "pesmaker.yaml"
    config_path.write_text(
        f"""project: separate_gpumd_output
sampling:
  engine: gpumd
  potential: {potential.as_posix()}
  selection:
    trajectory_pattern: {(tmp_path / 'traj_*' / 'movie.xyz').as_posix()}
    output_dir: {(tmp_path / 'selected').as_posix()}
    min_distance: 0.0
    max_count: 1
""",
        encoding="utf-8",
    )

    assert main(["select", str(config_path)]) == 0
    output = capsys.readouterr().out

    assert "Descriptor       : GPUMD / Calorine-calculated NEP descriptors" in output
    assert "Trajectory selection" in output
    assert output.count("Potential        :") == 1
    assert "Engine           : GPUMD" not in output
    assert "Backend          : Calorine-calculated NEP descriptors" not in output
    assert "Status           : Loading the model" not in output
    assert output.count("Frames           : 2") == 2
    assert output.count("Descriptor matrix: 2 frame(s) x 2 feature(s)") == 2


def test_selection_warning_suggests_lower_min_distance():
    """Min-distance warnings should suggest a smaller threshold."""
    from pesmaker.samplers.selection import _suggest_lower_min_distance

    assert _suggest_lower_min_distance(0.005) == "0.003"


def test_gpumd_manifest_descriptor_name_follows_nep_potential():
    """Selection manifests should record the NEP family, not Calorine backend."""
    from pesmaker.samplers.selection import _nep_descriptor_name

    assert _nep_descriptor_name("nep89_20250409.txt") == "nep89"
    assert _nep_descriptor_name("nep89.txt") == "nep89"
    assert _nep_descriptor_name("nep.txt") == "nep"
    assert _nep_descriptor_name("custom_nep.txt") == "nep"


def test_scf_setup_reads_selected_frames_from_combined_xyz(tmp_path, monkeypatch):
    """SCF setup should split selected.xyz frames using manifest frame_index."""
    from ase import Atoms
    from ase.io import write

    selected_dir = tmp_path / "selected"
    selected_dir.mkdir()
    selected_xyz = selected_dir / "selected.xyz"
    frames = [
        Atoms("Te", positions=[(0.0, 0.0, 0.0)], cell=[5, 5, 5], pbc=True),
        Atoms("Te", positions=[(2.0, 0.0, 0.0)], cell=[5, 5, 5], pbc=True),
    ]
    write(selected_xyz, frames, format="extxyz")
    (selected_dir / "manifest.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "index": 0,
                        "source_frame": 3,
                        "frame_index": 0,
                        "path": str(selected_xyz),
                        "atom_count": 1,
                    }
                ),
                json.dumps(
                    {
                        "index": 1,
                        "source_frame": 7,
                        "frame_index": 1,
                        "path": str(selected_xyz),
                        "atom_count": 1,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "pesmaker.yaml"
    config_path.write_text(
        f"""project: selected_labeling
labeling:
  output_dir: {(tmp_path / 'labeling').as_posix()}
  input_manifest: {(selected_dir / 'manifest.jsonl').as_posix()}
jobs:
  cores_cpu: 1
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    assert main(["scf-setup", str(config_path)]) == 0

    poscar_0 = tmp_path / "labeling" / "selected_000000" / "POSCAR"
    poscar_1 = tmp_path / "labeling" / "selected_000001" / "POSCAR"
    from ase.io import read

    assert read(poscar_0).get_positions()[0, 0] == 0.0
    assert read(poscar_1).get_positions()[0, 0] == 2.0
    records = [
        json.loads(line)
        for line in (tmp_path / "labeling" / "labeling_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["frame_index"] for record in records] == [0, 1]
    assert [record["source_frame"] for record in records] == [3, 7]


def test_gpumd_select_uses_calorine_nep_descriptors_by_default(
    tmp_path, monkeypatch, capsys
):
    """GPUMD selection should use its NEP potential without a descriptor key."""
    import sys
    import types

    from ase import Atoms
    from ase.io import write

    def fake_get_descriptors(atoms, *, model_filename):
        positions = atoms.get_positions()
        value = float(np.mean(positions[:, 0]))
        return np.array([[value, value**2], [value + 0.1, value**2 + 0.1]])

    calorine_module = types.ModuleType("calorine")
    nep_module = types.ModuleType("calorine.nep")
    nep_module.get_descriptors = fake_get_descriptors
    monkeypatch.setitem(sys.modules, "calorine", calorine_module)
    monkeypatch.setitem(sys.modules, "calorine.nep", nep_module)

    frames = [
        Atoms("Te2", positions=[(0.0, 0.0, 0.0), (0.2, 0.0, 0.0)], cell=[3, 3, 20], pbc=True),
        Atoms("Te2", positions=[(0.1, 0.0, 0.0), (0.3, 0.0, 0.0)], cell=[3, 3, 20], pbc=True),
        Atoms("Te2", positions=[(1.5, 0.0, 0.0), (1.7, 0.0, 0.0)], cell=[3, 3, 20], pbc=True),
    ]
    trajectory = tmp_path / "movie.xyz"
    write(trajectory, frames, format="extxyz")
    potential = tmp_path / "nep89.txt"
    potential.write_text("fake potential\n", encoding="utf-8")
    selected_dir = tmp_path / "selected"
    config_path = tmp_path / "pesmaker.yaml"
    config_path.write_text(
        f"""project: calorine_select
sampling:
  engine: gpumd
  potential: {potential.as_posix()}
  selection:
    trajectory_pattern: {trajectory.as_posix()}
    output_dir: {selected_dir.as_posix()}
    min_distance: 0.2
    max_count: 2
""",
        encoding="utf-8",
    )

    assert main(["select", str(config_path)]) == 0
    output = capsys.readouterr().out

    features = np.load(selected_dir / "selection_features.npy")
    assert features.shape == (3, 2)
    assert (selected_dir / "fps_selection.png").exists()
    assert "Trajectory selection" in output
    assert "Mode             : separate_trajectories" in output
    assert "Trajectories     : 1" in output
    assert "Descriptor       : GPUMD / Calorine-calculated NEP descriptors" in output
    assert "Potential        :" in output
    assert "FPS descriptor calculation" not in output
    assert "Engine           : GPUMD" not in output
    assert "Backend          : Calorine-calculated NEP descriptors" not in output
    assert "Frames           : 3" in output
    assert "Progress         : [##############################]" in output
    assert "3/3 frame(s) (100.0%)" in output
    assert "Descriptor matrix: 3 frame(s) x 2 feature(s)" in output
    assert "Selection completed: Selected 2 of 3 frame(s) from 1 trajectory file(s)." in output
    assert "using NEP89 descriptors calculated from the GPUMD potential" in output
    records = [
        json.loads(line)
        for line in (selected_dir / "manifest.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert [record["source_frame"] for record in records] == [0, 2]
    assert {record["descriptor"] for record in records} == {"nep89"}


def test_mace_select_uses_invariant_model_descriptors_by_default(
    tmp_path, monkeypatch, capsys, caplog
):
    """MACE selection should use invariant, element-pooled model descriptors."""
    import sys
    import types

    from ase import Atoms
    from ase.io import write

    calculator_calls = []
    descriptor_calls = []

    class FakeMACECalculator:
        def __init__(self, **kwargs):
            warnings.warn(
                "Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD "
                "detected, forcing weights_only=False.",
                UserWarning,
            )
            warnings.warn(
                "`torch.jit.load` is deprecated. Please switch to torch.export.",
                DeprecationWarning,
            )
            logging.warning(
                "No dtype selected, switching to float64 to match model dtype."
            )
            calculator_calls.append(kwargs)

        def get_descriptors(self, atoms, *, invariants_only, num_layers):
            descriptor_calls.append((invariants_only, num_layers))
            x_mean = float(np.mean(atoms.get_positions()[:, 0]))
            return np.array(
                [
                    [x_mean, 1.0],
                    [x_mean + 10.0, 2.0],
                ]
            )

    mace_module = types.ModuleType("mace")
    calculators_module = types.ModuleType("mace.calculators")
    calculators_module.MACECalculator = FakeMACECalculator
    monkeypatch.setitem(sys.modules, "mace", mace_module)
    monkeypatch.setitem(sys.modules, "mace.calculators", calculators_module)

    frames = [
        Atoms(
            ["Te", "O"],
            positions=[(0.0, 0.0, 0.0), (0.2, 0.0, 0.0)],
            cell=[3, 3, 20],
            pbc=True,
        ),
        Atoms(
            ["Te", "O"],
            positions=[(0.1, 0.0, 0.0), (0.3, 0.0, 0.0)],
            cell=[3, 3, 20],
            pbc=True,
        ),
        Atoms(
            ["Te", "O"],
            positions=[(1.5, 0.0, 0.0), (1.7, 0.0, 0.0)],
            cell=[3, 3, 20],
            pbc=True,
        ),
    ]
    trajectory = tmp_path / "mace.xyz"
    write(trajectory, frames, format="extxyz")
    descriptor_model = tmp_path / "mace-omat-small.model"
    descriptor_model.write_text("fake MACE model\n", encoding="utf-8")
    selected_dir = tmp_path / "selected"
    config_path = tmp_path / "pesmaker.yaml"
    config_path.write_text(
        f"""project: mace_select
sampling:
  engine: mace
  selection:
    trajectory_pattern: {trajectory.as_posix()}
    output_dir: {selected_dir.as_posix()}
    descriptor_model: {descriptor_model.as_posix()}
    min_distance: 0.0
    max_count: 2
""",
        encoding="utf-8",
    )

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        assert main(["select", str(config_path)]) == 0
    output = capsys.readouterr().out

    assert caught_warnings == []
    assert "No dtype selected" not in caplog.text
    features = np.load(selected_dir / "selection_features.npy")
    assert features.shape == (3, 4)
    assert calculator_calls == [
        {
            "model_paths": str(descriptor_model.resolve()),
            "device": "cuda",
        }
    ]
    assert descriptor_calls == [(True, -1), (True, -1), (True, -1)]
    assert np.allclose(features[0], [10.1, 2.0, 0.1, 1.0])
    assert "Trajectory selection" in output
    assert "Mode             : separate_trajectories" in output
    assert "Trajectories     : 1" in output
    assert "Descriptor       : MACE invariant descriptors" in output
    assert f"Model            : {descriptor_model.resolve()}" in output
    assert "Device           : cuda" in output
    assert "FPS descriptor calculation" not in output
    assert "Engine           : MACE" not in output
    assert "Backend          : MACECalculator invariant descriptors" not in output
    assert "Frames           : 3" in output
    assert "Progress         : [##############################]" in output
    assert "3/3 frame(s) (100.0%)" in output
    assert "Descriptor matrix: 3 frame(s) x 4 feature(s)" in output
    assert "Selection completed: Selected 2 of 3 frame(s) from 1 trajectory file(s)." in output
    assert "using invariant descriptors output by the MACE model" in output
    records = [
        json.loads(line)
        for line in (selected_dir / "manifest.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert [record["source_frame"] for record in records] == [0, 2]
    assert {record["descriptor"] for record in records} == {"mace"}
