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
    assert main(["label-setup", str(config_path)]) == 0
    assert main(["train-setup", str(config_path)]) == 0

    assert (tmp_path / "sampling" / "md_000000_temp_300K" / "run.in").exists()
    assert (tmp_path / "sampling" / "md_000000_temp_300K" / "submit.sh").exists()
    assert (tmp_path / "labeling" / "calc_000000" / "POSCAR").exists()
    assert (tmp_path / "labeling" / "calc_000000" / "INCAR").read_text(
        encoding="utf-8"
    ) == "NSW = 0\n"
    assert (tmp_path / "training" / "nep.in").exists()


def test_labeling_setup_can_preserve_generated_vasp_source_tree(tmp_path):
    """VASP labeling setup can keep generated path identity for batch jobs."""
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

    assert main(["label-setup", str(config_path)]) == 0

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


def test_labeling_setup_can_generate_potcar_from_library(tmp_path):
    """VASP labeling setup can concatenate POTCAR files from a local library."""
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

    assert main(["label-setup", str(config_path)]) == 0

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

    assert main(["label-setup", str(config_path)]) == 0

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

    assert main(["label-setup", str(config_path)]) == 0

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

    assert main(["label-setup", str(config_path)]) == 0

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

    assert result.message == "Would submit 1 labeling job(s)"
    log = tmp_path / "labeling" / "labeling_submitted_jobs.txt"
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
