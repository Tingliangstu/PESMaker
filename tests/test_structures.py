# Copyright (c) 2026 Ting Liang.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Tests for structure supercell and perturbation utilities."""

import numpy as np

from pesmaker.structures.perturb import (
    PerturbationSettings,
    get_atom_perturb_vector,
    get_cell_perturb_matrix,
    make_supercell,
    perturb_structures,
)


def test_make_supercell_multiplies_atom_count():
    """A 4x4x4 supercell should contain 64 times the original atoms."""
    from ase import Atoms

    atoms = Atoms("Te", positions=[[0.0, 0.0, 0.0]], cell=[3.0, 3.0, 3.0], pbc=True)

    supercell = make_supercell(atoms, (4, 4, 4))

    assert len(supercell) == 64


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
