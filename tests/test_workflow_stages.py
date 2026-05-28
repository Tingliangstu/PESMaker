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
training:
  model: nep
  output_dir: {(tmp_path / 'training').as_posix()}
  dataset: {train_xyz.as_posix()}
""",
        encoding="utf-8",
    )

    assert main(["md_sampling", str(config_path)]) == 0
    assert main(["label-setup", str(config_path)]) == 0
    assert main(["train-setup", str(config_path)]) == 0

    assert (tmp_path / "sampling" / "md_000000_temp_300K" / "run.in").exists()
    assert (tmp_path / "sampling" / "md_000000_temp_300K" / "submit.sh").exists()
    assert (tmp_path / "labeling" / "calc_000000" / "POSCAR").exists()
    assert (tmp_path / "labeling" / "calc_000000" / "INCAR").read_text(
        encoding="utf-8"
    ) == "NSW = 0\n"
    assert (tmp_path / "training" / "nep.in").exists()


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

    assert main(["sampling", str(config_path)]) == 0

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

    assert main(["md-sampling", str(config_path)]) == 0

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
