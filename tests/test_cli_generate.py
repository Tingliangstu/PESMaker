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

from pesmaker.cli import main


def test_cli_generate_writes_structures(tmp_path):
    """The generate command should write perturbed structures and a manifest."""
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

    assert exit_code == 0
    assert (output_dir / "te" / "structure_000000.vasp").exists()
    assert (output_dir / "te" / "structure_000001.vasp").exists()
    assert (output_dir / "manifest.jsonl").exists()
    assert len(list(output_dir.glob("te/structure_*.vasp"))) == 2


def test_cli_generate_uses_unique_folders_for_duplicate_stems(tmp_path):
    """Duplicate input stems should not overwrite each other's outputs."""
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
    assert (output_dir / "te" / "structure_000000.vasp").exists()
    assert (output_dir / "te_2" / "structure_000000.vasp").exists()
