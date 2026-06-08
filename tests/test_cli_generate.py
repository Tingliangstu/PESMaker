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

    assert "Potential Energy Surface Maker" in output
    assert "v-0.1.0" in output
    assert "sample-setup" in output
    assert "scf-setup" in output
    assert "plan" not in output
    assert "Prepare SCF calculation job folders" in output


def test_cli_missing_config_names_the_missing_file(tmp_path, capsys):
    """Missing YAML paths should produce a clear user-facing error."""
    config_path = tmp_path / "run.yaml"

    exit_code = main(["next", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "Potential Energy Surface Maker" in captured.out
    assert f"Error: config file does not exist: {config_path}" in captured.err
    assert captured.err.strip() != f"Error: {config_path}"


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
    assert (output_dir / "te" / "pristine_4x4x4.vasp").exists()
    assert (output_dir / "te" / "perturb_000000.vasp").exists()
    assert (output_dir / "te" / "perturb_000001.vasp").exists()
    assert (output_dir / "manifest.jsonl").exists()
    assert len(list(output_dir.glob("te/*.vasp"))) == 3
    records = [
        json.loads(line)
        for line in (output_dir / "manifest.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert [record["generation_type"] for record in records] == [
        "pristine",
        "perturb",
        "perturb",
    ]
    assert "Perturbation generation complete." in output
    assert "\n\nPerturbation generation complete." in output
    assert "Input structures" not in output
    assert "Generated structures" not in output
    assert f"Output directory : {output_dir}" in output
    assert f"Manifest         : {output_dir / 'manifest.jsonl'}" in output
    assert f"Details          : {output_dir / 'generation_summary.txt'}" in output
    assert "Generation tasks:" in output
    assert "  - 1 input(s) -> 3 structure(s), supercell=(4, 4, 4)" in output
    assert "    per input:" in output
    assert "      pristine: 3 structure(s) (1 pristine, 2 perturbed)" in output
    assert "    details:" not in output
    assert f"      - input: {cif_path}" not in output
    assert "        generated: 2 perturb structure(s)" not in output
    assert "        outputs:" not in output
    assert f"          - perturb -> {output_dir / 'te'} (2)" not in output
    assert "perturb files ->" not in output
    assert "pristine ->" not in output
    assert output.endswith("\n\n")


def test_cli_generate_omitted_perturb_writes_only_pristine_supercell(
    tmp_path, capsys
):
    """A config without `perturb` should only expand and write the structure."""
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
  include:
    - {cif_path.as_posix()}
generation:
  supercell: [2, 2, 1]
  output_dir: {output_dir.as_posix()}
""",
        encoding="utf-8",
    )

    exit_code = main(["generate", str(config_path)])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert (output_dir / "te" / "pristine_2x2x1.vasp").exists()
    assert list(output_dir.glob("te/perturb_*.vasp")) == []
    records = [
        json.loads(line)
        for line in (output_dir / "manifest.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert len(records) == 1
    assert records[0]["generation_type"] == "pristine"
    assert records[0]["atom_count"] == 4
    assert "Structure generation complete." in output
    assert "Perturbation generation complete." not in output
    assert "  - 1 input(s) -> 1 structure(s), supercell=(2, 2, 1)" in output
    assert "      pristine: 1 structure(s) (1 pristine)" in output


def test_cli_generate_explains_scf_only_config(tmp_path, capsys):
    """SCF-only configs should point users at the SCF setup command."""
    config_path = tmp_path / "sub.yaml"
    config_path.write_text(
        """project: Te_bulk_mp
labeling:
  engine: vasp
  output_dir: run_vasp_scf
  input_dir: generated
  incar: INCAR
  command: vasp_std

jobs:
  submit_command: sbatch
""",
        encoding="utf-8",
    )

    exit_code = main(["generate", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "Traceback" not in captured.err
    assert "this config is for SCF setup" in captured.err
    assert f"pesmaker scf-setup {config_path}" in captured.err
    assert f"pesmaker submit {config_path}" in captured.err
    assert "--stage scf" not in captured.err


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
      include_pristine: true
      pert_num: 1
      seed: 7
""",
        encoding="utf-8",
    )

    exit_code = main(["generate", str(config_path)])

    assert exit_code == 0
    assert (output_dir / "te2d" / "pristine" / "pristine_1x1x1.vasp").exists()
    assert (output_dir / "te2d" / "pristine" / "surface_000000.vasp").exists()
    assert (
        output_dir / "te2d" / "single_vacancy_Te_000001" / "defect_000000.vasp"
    ).exists()
    assert (
        output_dir
        / "te2d"
        / "single_vacancy_Te_000001"
        / "pristine_1x1x1_single_vacancy_Te_000001.vasp"
    ).exists()
    assert (
        output_dir
        / "te2d"
        / "line_defect_Te_const_a_000001"
        / "pristine_1x1x1_line_defect_Te_const_a_000001.vasp"
    ).exists()
    assert len(list(output_dir.glob("te2d/*/*.vasp"))) == 8
    summary = (output_dir / "generation_summary.txt").read_text(encoding="utf-8")
    assert "Structure generation complete." in summary
    assert f"Details          : {output_dir / 'generation_summary.txt'}" in summary
    assert "Generation tasks:" in summary
    assert "Input structures" not in summary
    assert "    per input:" in summary
    assert "      pristine: 2 structure(s) (1 pristine, 1 perturbed)" in summary
    assert (
        "      single vacancies: 1 variant(s), 2 structure(s) "
        "(1 pristine, 1 perturbed)"
    ) in summary
    assert (
        "      double vacancies: 1 variant(s), 2 structure(s) "
        "(1 pristine, 1 perturbed)"
    ) in summary
    assert (
        "      line defects: 1 variant(s), 2 structure(s) "
        "(1 pristine, 1 perturbed)"
    ) in summary
    assert "    details:" in summary
    assert f"      - input: {structure_path}" in summary
    assert "        generated:" in summary
    assert "surface ->" in summary
    assert "defect:single_vacancy_Te_000001 ->" in summary
    assert "files ->" not in summary


def test_cli_generate_defects_without_perturb_writes_pristine_variants(tmp_path):
    """Defect-only generation should not require random perturbation settings."""
    from ase import Atoms
    from ase.io import write

    atoms = Atoms(
        "Te2",
        positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        cell=[3.0, 3.0, 3.0],
        pbc=True,
    )
    structure_path = tmp_path / "te.xyz"
    write(structure_path, atoms, format="extxyz")
    output_dir = tmp_path / "generated"
    config_path = tmp_path / "pesmaker.yaml"
    config_path.write_text(
        f"""project: defect_no_perturb
structures:
  - {structure_path.as_posix()}
generation:
  output_dir: {output_dir.as_posix()}
  defects:
    single_vacancies:
      elements: [Te]
      max_count: 1
""",
        encoding="utf-8",
    )

    assert main(["generate", str(config_path)]) == 0

    assert (output_dir / "te" / "pristine" / "pristine_1x1x1.vasp").exists()
    assert (
        output_dir
        / "te"
        / "single_vacancy_Te_000001"
        / "pristine_1x1x1_single_vacancy_Te_000001.vasp"
    ).exists()
    assert set(output_dir.glob("te/**/*.vasp")) == {
        output_dir / "te" / "pristine" / "pristine_1x1x1.vasp",
        output_dir
        / "te"
        / "single_vacancy_Te_000001"
        / "pristine_1x1x1_single_vacancy_Te_000001.vasp",
    }
    records = [
        json.loads(line)
        for line in (output_dir / "manifest.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert [record["variant"] for record in records] == [
        "pristine",
        "single_vacancy_Te_000001",
    ]
    assert {record["generation_type"] for record in records} == {"pristine"}


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
        / "single_vacancy_Te_000001"
        / "defect_000001.vasp"
    ).exists()
    assert (
        output_dir / "bulk_221" / "te" / "pristine" / "perturb_000000.vasp"
    ).exists()
    assert (
        output_dir / "bulk_221" / "te" / "pristine" / "pristine_2x2x1.vasp"
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
        "pristine",
    }
    assert {tuple(record["supercell"]) for record in records} == {
        (1, 1, 1),
        (2, 2, 1),
    }
    summary = (output_dir / "generation_summary.txt").read_text(encoding="utf-8")
    assert "Generation tasks:" in summary
    assert "surface_111: 1 input(s) ->" in summary
    assert "bulk_221: 1 input(s) ->" in summary


def test_cli_generate_writes_named_pristine_supercell(tmp_path):
    """Generation writes the expanded pristine structure before perturbations."""
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
  supercell: [2, 2, 1]
  output_dir: {output_dir.as_posix()}
  perturb:
    pert_num: 2
    seed: 7
    format: vasp
""",
        encoding="utf-8",
    )

    assert main(["generate", str(config_path)]) == 0

    assert (output_dir / "te" / "pristine_2x2x1.vasp").exists()
    assert (output_dir / "te" / "perturb_000000.vasp").exists()
    assert (output_dir / "te" / "perturb_000001.vasp").exists()
    records = [
        json.loads(line)
        for line in (output_dir / "manifest.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert [record["generation_type"] for record in records] == [
        "pristine",
        "perturb",
        "perturb",
    ]
    assert records[0]["atom_count"] == 4


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
