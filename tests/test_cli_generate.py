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
"""CLI integration tests for structure generation."""

import json

from pesmaker.cli import main


def test_cli_help_shows_simple_public_commands(capsys):
    """Top-level help should show only concise public workflow commands."""
    try:
        main(["-h"])
    except SystemExit as exc:
        assert exc.code == 0
    output = capsys.readouterr().out

    assert "sample-setup" in output
    assert "scf-setup" in output
    assert "plan" not in output
    assert "Prepare SCF calculation job folders" in output


def test_cli_generate_writes_structures(tmp_path, capsys):
    """The generate command should write perturbed structures and a manifest.

    Args:
        tmp_path: Pytest temporary directory used for config and outputs.
        capsys: Pytest fixture used to capture command output.
    """
    cif_path = tmp_path / "te.cif"
    cif_path.write_text(
        """data_te
_symmetry_space_group_name_H-M    'P 1'
_cell_length_a    3.0
_cell_length_b    3.0
_cell_length_c    3.0
_cell_angle_alpha 90
_cell_angle_beta  90
_cell_angle_gamma 90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Te1 Te 0 0 0
""",
        encoding="utf-8",
    )
    output_dir = tmp_path / "generated"
    config_path = tmp_path / "pesmaker.yaml"
    config_path.write_text(
        f"""project: test_generate
structures:
  - {cif_path.as_posix()}
generation:
  supercell: [4, 4, 4]
  output_dir: {output_dir.as_posix()}
  perturb:
    pert_num: 2
    cell_pert_fraction: 0.03
    atom_pert_distance: 0.1
    atom_pert_style: normal
    seed: 7
    format: vasp
""",
        encoding="utf-8",
    )

    exit_code = main(["generate", str(config_path)])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert (output_dir / "te" / "perturb_000000.vasp").exists()
    assert (output_dir / "te" / "perturb_000001.vasp").exists()
    assert (output_dir / "manifest.jsonl").exists()
    assert len(list(output_dir.glob("te/perturb_*.vasp"))) == 2
    records = [
        json.loads(line)
        for line in (output_dir / "manifest.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert {record["generation_type"] for record in records} == {"perturb"}
    assert "Structure generation complete." in output
    assert "\n\nStructure generation complete." in output
    assert "Input structures     : 1" in output
    assert "Generated structures : 2" in output
    assert f"Output directory     : {output_dir}" in output
    assert f"Manifest             : {output_dir / 'manifest.jsonl'}" in output
    assert "  - 1 input(s) -> 2 structure(s), supercell=(4, 4, 4)" in output
    assert "    per input: 2 perturb structure(s)" in output
    assert f"{cif_path}: 2 perturb structure(s)" in output
    assert f"-> {output_dir / 'te'}" in output
    assert f"perturb files -> {output_dir / 'te' / 'perturb_*.vasp'} (2)" in output
    assert "pristine ->" not in output
    assert output.endswith("\n\n")


def test_cli_generate_uses_unique_folders_for_duplicate_stems(tmp_path):
    """Duplicate input stems should not overwrite each other's outputs.

    Args:
        tmp_path: Pytest temporary directory used for config and outputs.
    """
    cif_path = tmp_path / "te.cif"
    cif_path.write_text(
        """data_te
_symmetry_space_group_name_H-M    'P 1'
_cell_length_a    3.0
_cell_length_b    3.0
_cell_length_c    3.0
_cell_angle_alpha 90
_cell_angle_beta  90
_cell_angle_gamma 90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Te1 Te 0 0 0
""",
        encoding="utf-8",
    )
    output_dir = tmp_path / "generated"
    config_path = tmp_path / "pesmaker.yaml"
    config_path.write_text(
        f"""project: test_generate
structures:
  - {cif_path.as_posix()}
  - {cif_path.as_posix()}
generation:
  output_dir: {output_dir.as_posix()}
  perturb:
    pert_num: 1
    seed: 7
""",
        encoding="utf-8",
    )

    exit_code = main(["generate", str(config_path)])

    assert exit_code == 0
    assert (output_dir / "te" / "perturb_000000.vasp").exists()
    assert (output_dir / "te_2" / "perturb_000000.vasp").exists()


def test_cli_generate_writes_surface_defect_folders(tmp_path):
    """Surface and defect generation should write separate variant folders."""
    from ase import Atoms
    from ase.io import write

    atoms = Atoms(
        "Te4",
        positions=[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
        ],
        cell=[2.0, 2.0, 6.0],
        pbc=[True, True, False],
    )
    structure_path = tmp_path / "te2d.xyz"
    write(structure_path, atoms, format="extxyz")
    output_dir = tmp_path / "generated"
    config_path = tmp_path / "pesmaker.yaml"
    config_path.write_text(
        f"""project: defect_generate
structures:
  - {structure_path.as_posix()}
generation:
  supercell: [1, 1, 1]
  output_dir: {output_dir.as_posix()}
  surface:
    vacuum: 30.0
    axis: 2
    defects:
      include_pristine: true
      single_vacancies:
        elements: [Te]
        max_count: 1
      double_vacancies:
        elements: [Te]
        max_count: 1
      line_defects:
        elements: [Te]
        max_count: 1
    perturb:
      pert_num: 1
      seed: 7
""",
        encoding="utf-8",
    )

    exit_code = main(["generate", str(config_path)])

    assert exit_code == 0
    assert (output_dir / "te2d" / "pristine" / "surface_000000.vasp").exists()
    assert (
        output_dir / "te2d" / "single_vacancy_Te_000000" / "defect_000000.vasp"
    ).exists()
    assert len(list(output_dir.glob("te2d/*/*.vasp"))) == 4
    summary = (output_dir / "generation_summary.txt").read_text(encoding="utf-8")
    assert "Input structures     : 1" in summary
    assert "per input: 1 surface, 3 defect structure(s)" in summary
    assert f"{structure_path}: 1 surface, 3 defect structure(s)" in summary
    assert "surface files ->" in summary
    assert "defect:single_vacancy_Te_000000 files ->" in summary
    assert "pristine ->" not in summary


def test_cli_generate_writes_multiple_task_folders(tmp_path):
    """Multiple generation tasks should be grouped by task name."""
    from ase import Atoms
    from ase.io import write

    atoms = Atoms(
        "Te2",
        positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        cell=[2.0, 2.0, 6.0],
        pbc=[True, True, False],
    )
    structure_path = tmp_path / "te.xyz"
    write(structure_path, atoms, format="extxyz")
    output_dir = tmp_path / "generated"
    config_path = tmp_path / "pesmaker.yaml"
    config_path.write_text(
        f"""project: multi_task
structures:
  - {structure_path.as_posix()}
generation:
  output_dir: {output_dir.as_posix()}
  tasks:
    - name: surface_111
      supercell: [1, 1, 1]
      surface:
        vacuum: 20.0
        defects:
          single_vacancies:
            elements: [Te]
            max_count: 1
          perturb:
            pert_num: 2
            seed: 1
    - name: bulk_221
      supercell: [2, 2, 1]
      perturb:
        pert_num: 1
        seed: 2
""",
        encoding="utf-8",
    )

    assert main(["generate", str(config_path)]) == 0

    assert (
        output_dir
        / "surface_111"
        / "te"
        / "single_vacancy_Te_000000"
        / "defect_000001.vasp"
    ).exists()
    assert (
        output_dir / "bulk_221" / "te" / "pristine" / "perturb_000000.vasp"
    ).exists()
    records = [
        json.loads(line)
        for line in (output_dir / "manifest.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert {record["task"] for record in records} == {"surface_111", "bulk_221"}
    assert {record["generation_type"] for record in records} == {
        "defect",
        "surface",
        "perturb",
    }
    assert {tuple(record["supercell"]) for record in records} == {
        (1, 1, 1),
        (2, 2, 1),
    }
    summary = (output_dir / "generation_summary.txt").read_text(encoding="utf-8")
    assert "Generation plan:" in summary
    assert "surface_111: 1 input(s) ->" in summary
    assert "bulk_221: 1 input(s) ->" in summary


def test_cli_prints_banner_for_commands(tmp_path, capsys):
    """Every executed CLI command should print version and contact information.

    Args:
        tmp_path: Pytest temporary directory used for a starter config.
        capsys: Pytest fixture used to capture command output.
    """
    config_path = tmp_path / "pesmaker.yaml"

    exit_code = main(["init", str(config_path)])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Potential Energy Surface Maker" in output
    assert "v-0.1.0" in output
    assert "Author: liangting.zj@gmail.com" in output
