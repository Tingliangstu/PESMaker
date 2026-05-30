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

from pesmaker.cli import main
from pesmaker.config.io import load_config
from pesmaker.workflow.stages import (
    RECOMMENDED_GW_POTCARS,
    RECOMMENDED_PBE_POTCARS,
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

    assert (tmp_path / "sampling" / "md_000000_temp_300K" / "run.in").exists()
    assert (tmp_path / "sampling" / "md_000000_temp_300K" / "submit.sh").exists()
    assert (tmp_path / "labeling" / "calc_000000" / "POSCAR").exists()
    incar_text = (tmp_path / "labeling" / "calc_000000" / "INCAR").read_text(
        encoding="utf-8"
    )
    assert "NSW = 0\n" in incar_text
    assert "KPAR = 1" in incar_text
    assert "NCORE = 1" in incar_text
    assert (tmp_path / "training" / "nep.in").exists()


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
    assert (workdir / "POSCAR").read_text(encoding="utf-8") == source_text
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
    assert (workdir / "POSCAR").read_text(encoding="utf-8") == source_text
    incar_text = (workdir / "INCAR").read_text(encoding="utf-8")
    assert "NSW = 0\n" in incar_text
    assert "KPAR = 2" in incar_text
    assert "NCORE = 3" in incar_text
    assert "/opt/vasp/vasp_std" in (workdir / "submit.sh").read_text(
        encoding="utf-8"
    )


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
    assert manifest_records[0]["kpar"] == 1
    assert manifest_records[0]["ncore"] == 1


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
    assert "KPAR = 2" in incar_text
    assert "NCORE = 3" in incar_text
    assert "#SBATCH --ntasks-per-node=36" in submit_text
    assert "#SBATCH --gres" not in submit_text
    assert "#SBATCH --time" not in submit_text
    assert "set -euo pipefail" not in submit_text
    assert 'cd "$(dirname "$0")"' not in submit_text
    assert "srun /opt/vasp/vasp_std" in submit_text


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
    assert "#SBATCH --ntasks-per-node=8" in submit_text
    assert "#SBATCH --gres=gpu:2" in submit_text


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
    (generated_dir / "manifest.jsonl").write_text(
        json.dumps({"path": str(structure_path)}) + "\n",
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
        workdir = tmp_path / "sampling" / f"md_000000_temp_{temperature}K"
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
  temperature: 300-1500
""",
        encoding="utf-8",
    )

    assert main(["sample-setup", str(config_path)]) == 0

    workdir = tmp_path / "sampling" / "md_000000_ramp_300K_to_1500K"
    run_in = (workdir / "run.in").read_text(encoding="utf-8")
    assert "velocity       300" in run_in
    assert "ensemble       npt_scr 300 1500" in run_in


def test_select_uses_farthest_point_sampling(tmp_path, monkeypatch):
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
    min_distance: 0.2
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    assert main(["select", str(config_path)]) == 0

    assert (selected_dir / "selected.xyz").exists()
    assert len((selected_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()) == 2
