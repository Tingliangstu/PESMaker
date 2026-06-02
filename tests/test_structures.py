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
"""Tests for structure supercell and perturbation utilities."""

import numpy as np

from pesmaker.structures.io import write_structure
from pesmaker.structures.perturb import (
    PerturbationSettings,
    get_atom_perturb_vector,
    get_cell_perturb_matrix,
    make_supercell,
    perturb_structures,
)
from pesmaker.structures.defects import (
    apply_surface_settings,
    generate_defect_variants,
)


def test_make_supercell_multiplies_atom_count():
    """A 4x4x4 supercell should contain 64 times the original atoms."""
    from ase import Atoms

    atoms = Atoms("Te", positions=[[0.0, 0.0, 0.0]], cell=[3.0, 3.0, 3.0], pbc=True)

    supercell = make_supercell(atoms, (4, 4, 4))

    assert len(supercell) == 64


def test_vasp_writer_groups_repeated_element_blocks(tmp_path):
    """VASP output should use one compact count block per element."""
    from ase import Atoms

    atoms = Atoms(
        ["Te", "Te", "Pd", "Te", "Te", "Pd"],
        positions=[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (3.0, 0.0, 0.0),
            (4.0, 0.0, 0.0),
            (5.0, 0.0, 0.0),
        ],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    path = tmp_path / "tepd.vasp"

    write_structure(atoms, path, fmt="vasp")

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[5].split() == ["Te", "Pd"]
    assert lines[6].split() == ["4", "2"]


def test_cell_perturb_matrix_matches_dpdata_shape_and_bounds():
    """Cell perturbation matrices should follow the dpdata-style bounds."""
    rng = np.random.default_rng(1)

    matrix = get_cell_perturb_matrix(0.03, rng)

    assert matrix.shape == (3, 3)
    assert np.allclose(matrix, matrix.T)
    assert np.all(np.abs(np.diag(matrix) - 1.0) <= 0.03)
    off_diagonal = matrix[~np.eye(3, dtype=bool)]
    assert np.all(np.abs(off_diagonal) <= 0.015)


def test_normal_atom_perturb_vector_is_seeded():
    """Normal atomic perturbations should be reproducible with a seed."""
    rng = np.random.default_rng(3)

    vector = get_atom_perturb_vector(0.1, "normal", rng)

    assert vector.shape == (3,)
    assert np.allclose(vector, [0.11783252, -0.14755139, 0.02413895])


def test_perturb_structures_keeps_atom_count_and_changes_cell():
    """Perturbed structures should keep atom count while changing geometry."""
    from ase import Atoms

    atoms = Atoms("Te", positions=[[0.0, 0.0, 0.0]], cell=[3.0, 3.0, 3.0], pbc=True)
    settings = PerturbationSettings(
        pert_num=2,
        cell_pert_fraction=0.03,
        atom_pert_distance=0.1,
        atom_pert_style="normal",
        seed=5,
    )

    generated = list(perturb_structures(atoms, settings))

    assert len(generated) == 2
    assert all(len(item) == len(atoms) for item in generated)
    assert not np.allclose(generated[0].cell.array, atoms.cell.array)


def test_default_perturbation_settings_do_not_generate_random_structures():
    """Omitted perturb settings should leave generation in supercell-only mode."""
    from ase import Atoms

    atoms = Atoms("Te", positions=[[0.0, 0.0, 0.0]], cell=[3.0, 3.0, 3.0], pbc=True)
    settings = PerturbationSettings.from_mapping({})

    assert settings.pert_num == 0
    assert list(perturb_structures(atoms, settings)) == []


def test_negative_perturbation_count_is_rejected():
    """Invalid perturbation counts should fail instead of silently doing nothing."""
    from ase import Atoms

    atoms = Atoms("Te", positions=[[0.0, 0.0, 0.0]], cell=[3.0, 3.0, 3.0], pbc=True)
    settings = PerturbationSettings.from_mapping({"pert_num": -1})

    try:
        list(perturb_structures(atoms, settings))
    except ValueError as exc:
        assert "pert_num can not be negative" in str(exc)
    else:
        raise AssertionError("negative pert_num should fail")


def test_surface_and_defect_variants_are_generated():
    """2D surface and defect settings should create concrete variants."""
    from ase import Atoms

    atoms = Atoms(
        "Te4",
        positions=[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
        ],
        cell=[2.0, 2.0, 4.0],
        pbc=[True, True, False],
    )

    slab = apply_surface_settings(atoms, {"vacuum": 30.0, "axis": 2})
    variants = generate_defect_variants(
        slab,
        {
            "single_vacancies": {"elements": ["Te"], "max_count": 1},
            "double_vacancies": {"elements": ["Te"], "max_count": 1},
            "line_defects": {
                "elements": ["Te"],
                "max_count": 1,
            },
        },
    )

    assert np.isclose(slab.cell.lengths()[2], 30.0)
    assert [variant.name for variant in variants] == [
        "pristine",
        "single_vacancy_Te_000001",
        "double_vacancy_Te_000001",
        "line_defect_Te_const_a_000001",
    ]
    assert [len(variant.atoms) for variant in variants] == [4, 3, 2, 2]


def test_surface_vacuum_replaces_existing_vacuum():
    """Surface vacuum is total empty space, not vacuum added on both sides."""
    from ase import Atoms

    atoms = Atoms(
        "Te2",
        positions=[
            (0.0, 0.0, 20.0),
            (0.0, 0.0, 22.0),
        ],
        cell=[3.0, 3.0, 80.0],
        pbc=[True, True, False],
    )

    slab = apply_surface_settings(
        atoms,
        {"vacuum": 30.0, "axis": 2, "center": True},
    )
    positions = slab.get_positions()

    assert np.isclose(slab.cell.lengths()[2], 32.0)
    assert np.allclose(positions[:, 2], [15.0, 17.0])


def test_random_vacancies_are_seeded_and_reproducible():
    """Random vacancy mode should be reproducible with a user seed."""
    from ase import Atoms

    atoms = Atoms(
        "Te6",
        positions=[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
            (2.0, 1.0, 0.0),
        ],
        cell=[3.0, 2.0, 10.0],
        pbc=[True, True, False],
    )
    settings = {
        "mode": "random",
        "seed": 42,
        "include_pristine": False,
        "single_vacancies": {"elements": ["Te"], "max_count": 2},
        "double_vacancies": {"elements": ["Te"], "max_count": 2},
    }

    first = generate_defect_variants(atoms, settings)
    second = generate_defect_variants(atoms, settings)
    different_seed = generate_defect_variants(atoms, {**settings, "seed": 7})

    assert [variant.name for variant in first] == [variant.name for variant in second]
    assert [variant.description for variant in first] == [
        variant.description for variant in second
    ]
    assert [variant.description for variant in first] != [
        variant.description for variant in different_seed
    ]
    assert [variant.name for variant in first] == [
        "single_vacancy_Te_000001",
        "single_vacancy_Te_000002",
        "double_vacancy_Te_000001",
        "double_vacancy_Te_000002",
    ]


def test_perturbation_settings_can_include_pristine_base():
    """Perturbation settings can request an unperturbed generated structure."""
    settings = PerturbationSettings.from_mapping(
        {"pert_num": 3, "include_pristine": True}
    )
    alias_settings = PerturbationSettings.from_mapping({"include_unperturbed": True})

    assert settings.include_pristine is True
    assert settings.pert_num == 3
    assert alias_settings.include_pristine is True
