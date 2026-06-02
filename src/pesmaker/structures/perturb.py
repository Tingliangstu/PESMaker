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
"""Supercell construction and dpdata-style structure perturbation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PerturbationSettings:
    """Parameters controlling cell and atomic coordinate perturbations.

    Attributes:
        pert_num: Number of randomly perturbed structures to generate per input
            structure. The default is zero so generation can be used for
            supercell-only structure expansion.
        cell_pert_fraction: Maximum absolute cell perturbation fraction.
        atom_pert_distance: Atomic displacement scale in Angstrom.
        atom_pert_style: Distribution used for atomic displacement. Supported
            values are `normal`, `uniform`, and `const`.
        atom_pert_prob: Fraction of atoms selected for atomic displacement.
        seed: Optional random seed for reproducible perturbations.
        include_pristine: Whether to also write a pristine structure for
            every generated variant. The pristine variant is always written
            once in generation workflows.
    """

    pert_num: int = 0
    cell_pert_fraction: float = 0.03
    atom_pert_distance: float = 0.1
    atom_pert_style: str = "normal"
    atom_pert_prob: float = 1.0
    seed: int | None = None
    include_pristine: bool = False

    @classmethod
    def from_mapping(cls, data: dict | None) -> "PerturbationSettings":
        """Parse perturbation settings from the `generation.perturb` section.

        Args:
            data: Optional raw perturbation mapping from the config file.

        Returns:
            Parsed perturbation settings with defaults for omitted values.
        """
        data = data or {}
        return cls(
            pert_num=int(data.get("pert_num", 0)),
            cell_pert_fraction=float(data.get("cell_pert_fraction", 0.03)),
            atom_pert_distance=float(data.get("atom_pert_distance", 0.1)),
            atom_pert_style=str(data.get("atom_pert_style", "normal")),
            atom_pert_prob=float(data.get("atom_pert_prob", 1.0)),
            seed=int(data["seed"]) if data.get("seed") is not None else None,
            include_pristine=bool(
                data.get("include_pristine", data.get("include_unperturbed", False))
            ),
        )


def make_supercell(atoms, supercell: tuple[int, int, int]):
    """Return an ASE supercell using three positive expansion factors.

    Args:
        atoms: ASE `Atoms` object to expand.
        supercell: Three integer replication factors along lattice directions.

    Returns:
        New ASE `Atoms` object containing the repeated supercell.

    Raises:
        ValueError: If `supercell` does not contain three positive integers.
    """
    if len(supercell) != 3:
        raise ValueError("supercell must contain three integers")
    if any(value < 1 for value in supercell):
        raise ValueError("supercell values must be positive")
    return atoms.repeat(supercell)


def perturb_structure(
    atoms,
    settings: PerturbationSettings,
    *,
    rng: np.random.Generator | None = None,
):
    """Apply one random cell perturbation and one random atomic perturbation.

    Args:
        atoms: ASE `Atoms` object to perturb. The input object is not modified.
        settings: Perturbation parameters.
        rng: Optional NumPy random generator. When omitted, one is created from
            `settings.seed`.

    Returns:
        New perturbed ASE `Atoms` object.

    Raises:
        ValueError: If perturbation amplitudes or probabilities are invalid, or
            if `settings.atom_pert_style` is unsupported.
    """
    if settings.cell_pert_fraction < 0:
        raise ValueError("cell_pert_fraction can not be negative")
    if settings.atom_pert_distance < 0:
        raise ValueError("atom_pert_distance can not be negative")
    if not 0.0 <= settings.atom_pert_prob <= 1.0:
        raise ValueError("atom_pert_prob must be in [0, 1]")

    rng = rng or np.random.default_rng(settings.seed)
    perturbed = atoms.copy()
    matrix = get_cell_perturb_matrix(settings.cell_pert_fraction, rng)

    new_cell = np.asarray(perturbed.cell.array) @ matrix
    new_positions = perturbed.get_positions() @ matrix
    perturbed.set_cell(new_cell, scale_atoms=False)
    perturbed.set_positions(new_positions)

    atom_count = len(perturbed)
    perturbed_count = int(settings.atom_pert_prob * atom_count)
    if perturbed_count:
        atom_ids = rng.choice(atom_count, size=perturbed_count, replace=False)
        positions = perturbed.get_positions()
        for atom_id in sorted(atom_ids.tolist()):
            positions[atom_id] += get_atom_perturb_vector(
                settings.atom_pert_distance,
                settings.atom_pert_style,
                rng,
            )
        perturbed.set_positions(positions)

    return perturbed


def perturb_structures(atoms, settings: PerturbationSettings) -> Iterable:
    """Yield the requested number of perturbed structures.

    Args:
        atoms: ASE `Atoms` object used as the base structure.
        settings: Perturbation parameters.

    Yields:
        Perturbed ASE `Atoms` objects.
    """
    if settings.pert_num < 0:
        raise ValueError("pert_num can not be negative")

    rng = np.random.default_rng(settings.seed)
    for _ in range(settings.pert_num):
        yield perturb_structure(atoms, settings, rng=rng)


def get_cell_perturb_matrix(
    cell_pert_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create the symmetric dpdata-style cell perturbation matrix.

    Args:
        cell_pert_fraction: Maximum absolute cell perturbation fraction.
        rng: NumPy random generator used to draw six independent values.

    Returns:
        A 3x3 symmetric perturbation matrix. Diagonal values perturb
        compression/extension, and off-diagonal values perturb shear.
    """
    values = rng.random(6) * 2 * cell_pert_fraction - cell_pert_fraction
    return np.array(
        [
            [1 + values[0], 0.5 * values[5], 0.5 * values[4]],
            [0.5 * values[5], 1 + values[1], 0.5 * values[3]],
            [0.5 * values[4], 0.5 * values[3], 1 + values[2]],
        ],
        dtype=float,
    )


def get_atom_perturb_vector(
    atom_pert_distance: float,
    atom_pert_style: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create one random atomic displacement vector.

    Args:
        atom_pert_distance: Displacement scale in Angstrom.
        atom_pert_style: Distribution style. `normal` draws independent normal
            components scaled by `atom_pert_distance / sqrt(3)`, `uniform`
            samples uniformly inside a sphere, and `const` samples a fixed
            radius direction.
        rng: NumPy random generator.

    Returns:
        Three-component Cartesian displacement vector.

    Raises:
        ValueError: If `atom_pert_style` is not supported.
    """
    if atom_pert_style == "normal":
        return (atom_pert_distance / np.sqrt(3.0)) * rng.normal(size=3)
    if atom_pert_style == "uniform":
        return atom_pert_distance * np.cbrt(rng.random()) * _random_unit_vector(rng)
    if atom_pert_style == "const":
        return atom_pert_distance * _random_unit_vector(rng)
    raise ValueError(f"unsupported atom_pert_style: {atom_pert_style}")


def _random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    """Draw a random unit vector from a normal distribution.

    Args:
        rng: NumPy random generator.

    Returns:
        Three-component vector with Euclidean norm equal to one.
    """
    vector = rng.normal(size=3)
    while np.linalg.norm(vector) < 0.1:
        vector = rng.normal(size=3)
    return vector / np.linalg.norm(vector)
