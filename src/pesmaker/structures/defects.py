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
"""Surface and defect builders for generated candidate structures."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np


@dataclass(frozen=True)
class StructureVariant:
    """One generated structural variant before random perturbation.

    Attributes:
        name: Filesystem-safe variant name.
        atoms: ASE `Atoms` object for this variant.
        description: Human-readable variant description for manifests.
    """

    name: str
    atoms: Any
    description: str


def apply_surface_settings(atoms, settings: dict[str, Any] | None):
    """Apply optional 2D slab centering and vacuum settings.

    Args:
        atoms: ASE `Atoms` object.
        settings: Mapping such as `{vacuum: 30.0, axis: 2, center: true}`.

    Returns:
        A copied ASE `Atoms` object with the requested vacuum applied.
    """
    result = atoms.copy()
    settings = settings or {}
    if not settings:
        return result

    axis = int(settings.get("axis", 2))
    vacuum = settings.get("vacuum")
    center = bool(settings.get("center", True))
    if axis not in {0, 1, 2}:
        raise ValueError("generation.surface.axis must be 0, 1, or 2")
    if vacuum is not None:
        vacuum_value = float(vacuum)
        if vacuum_value < 0:
            raise ValueError("generation.surface.vacuum can not be negative")
        _set_total_vacuum(result, axis=axis, vacuum=vacuum_value, center=center)
    return result


def _set_total_vacuum(atoms, *, axis: int, vacuum: float, center: bool) -> None:
    cell = atoms.cell.array.copy()
    axis_vector = cell[axis]
    axis_length = np.linalg.norm(axis_vector)
    if axis_length <= 0:
        raise ValueError("generation.surface.axis cell vector has zero length")

    axis_unit = axis_vector / axis_length
    positions = atoms.get_positions()
    if len(atoms):
        projections = positions @ axis_unit
        slab_min = float(np.min(projections))
        slab_max = float(np.max(projections))
    else:
        slab_min = 0.0
        slab_max = 0.0

    target_length = (slab_max - slab_min) + vacuum
    if target_length <= 0:
        raise ValueError("generation.surface.vacuum is too small for this slab")

    cell[axis] = axis_unit * target_length
    atoms.set_cell(cell, scale_atoms=False)
    if center and len(atoms):
        target_center = float(np.dot(cell.sum(axis=0) * 0.5, axis_unit))
        slab_center = 0.5 * (slab_min + slab_max)
        shifted = positions + axis_unit * (target_center - slab_center)
        atoms.set_positions(shifted)


def generate_defect_variants(
    atoms,
    settings: dict[str, Any] | None,
) -> tuple[StructureVariant, ...]:
    """Build pristine, vacancy, and line-defect variants from a base structure.

    Args:
        atoms: ASE `Atoms` object after supercell/surface preparation.
        settings: Defect settings from `generation.defects`.

    Returns:
        Tuple of structural variants. Missing settings return only pristine.
    """
    settings = settings or {}
    variants: list[StructureVariant] = []
    include_pristine = bool(settings.get("include_pristine", True))
    if include_pristine or not settings:
        variants.append(StructureVariant("pristine", atoms.copy(), "pristine"))

    defaults = _defect_defaults(settings)
    variants.extend(
        _single_vacancies(
            atoms,
            _merge_defect_options(defaults, settings.get("single_vacancies")),
        )
    )
    variants.extend(
        _double_vacancies(
            atoms,
            _merge_defect_options(defaults, settings.get("double_vacancies")),
        )
    )
    variants.extend(
        _line_defects(
            atoms,
            _merge_defect_options(defaults, settings.get("line_defects")),
        )
    )
    return tuple(variants)


def _single_vacancies(atoms, settings: Any) -> list[StructureVariant]:
    options = _normalize_defect_options(settings, default_max=len(atoms))
    if not options["enabled"]:
        return []

    variants: list[StructureVariant] = []
    indices = _candidate_indices(atoms, options)
    if _selection_mode(options) == "random":
        indices = _sample_items(indices, options)
    else:
        indices = indices[: options["max_count"]]
    for serial, index in enumerate(indices, start=1):
        variant = atoms.copy()
        symbol = variant[index].symbol
        del variant[index]
        variants.append(
            StructureVariant(
                f"single_vacancy_{_element_label([symbol])}_{serial:06d}",
                variant,
                f"single vacancy: remove {symbol} atom {index}",
            )
        )
    return variants


def _double_vacancies(atoms, settings: Any) -> list[StructureVariant]:
    options = _normalize_defect_options(settings, default_max=20)
    if not options["enabled"]:
        return []

    pairs = list(combinations(_candidate_indices(atoms, options), 2))
    selection = _selection_mode(options, default="nearest")
    if selection == "random":
        pairs = _sample_items(pairs, options)
    elif selection in {"nearest", "ordered"} and bool(options.get("nearest_first", True)):
        positions = atoms.get_positions()
        pairs.sort(
            key=lambda pair: np.linalg.norm(positions[pair[0]] - positions[pair[1]])
        )
        pairs = pairs[: options["max_count"]]
    else:
        raise ValueError(f"unsupported double_vacancies selection: {selection}")

    variants: list[StructureVariant] = []
    for serial, (first, second) in enumerate(pairs, start=1):
        variant = atoms.copy()
        symbols = (variant[first].symbol, variant[second].symbol)
        for index in sorted((first, second), reverse=True):
            del variant[index]
        variants.append(
            StructureVariant(
                f"double_vacancy_{_element_label(symbols)}_{serial:06d}",
                variant,
                f"double vacancy: remove atoms {first} and {second}",
            )
        )
    return variants


def _line_defects(atoms, settings: Any) -> list[StructureVariant]:
    options = _normalize_defect_options(settings, default_max=4)
    if not options["enabled"]:
        return []

    candidates = _candidate_indices(atoms, options)
    scaled = atoms.get_scaled_positions(wrap=True)
    coordinate_axis = options.get("coordinate_axis")
    split_axis = _line_split_axis(options)
    if coordinate_axis is None:
        coordinate_axis, rows = _infer_line_rows(scaled, candidates, options)
    else:
        coordinate_axis = int(coordinate_axis)
        if coordinate_axis not in {0, 1, 2}:
            raise ValueError("line_defects.coordinate_axis must be 0, 1, or 2")
        if split_axis is not None and split_axis == coordinate_axis:
            raise ValueError(
                "line_defects.split_by_axis must differ from coordinate_axis"
            )
        rows = _group_line_rows(scaled, candidates, coordinate_axis, options)

    variants: list[StructureVariant] = []
    ordered_rows = sorted(rows.values(), key=lambda row: (-len(row), row[0]))
    if _selection_mode(options) == "random":
        selected_rows = _sample_items(ordered_rows, options)
    else:
        selected_rows = ordered_rows[: options["max_count"]]
    axis_label = _coordinate_axis_label(coordinate_axis)
    split_label = _split_axis_label(split_axis)
    for serial, row in enumerate(selected_rows, start=1):
        variant = atoms.copy()
        symbols = [variant[index].symbol for index in row]
        for index in sorted(row, reverse=True):
            del variant[index]
        variants.append(
            StructureVariant(
                (
                    f"line_defect_{_element_label(symbols)}_{axis_label}"
                    f"{split_label}_{serial:06d}"
                ),
                variant,
                (
                    f"line defect: fixed fractional "
                    f"{axis_label.removeprefix('const_')} coordinate, "
                    f"{_split_description(split_axis)}"
                    f"remove atoms {sorted(row)}"
                ),
            )
        )
    return variants


def _defect_defaults(settings: dict[str, Any]) -> dict[str, Any]:
    defaults = {}
    for key in ("mode", "selection", "seed"):
        if key in settings:
            defaults[key] = settings[key]
    return defaults


def _merge_defect_options(defaults: dict[str, Any], settings: Any) -> Any:
    if not defaults or settings in (None, False):
        return settings
    if settings is True:
        return {**defaults, "enabled": True}
    if not isinstance(settings, dict):
        return settings
    return {**defaults, **settings}


def _infer_line_rows(
    scaled: np.ndarray,
    candidates: list[int],
    options: dict[str, Any],
) -> tuple[int, dict[Any, list[int]]]:
    best_axis = 1
    best_rows: dict[Any, list[int]] = {}
    best_score = (-1, 0)
    split_axis = _line_split_axis(options)
    for axis in (0, 1):
        if axis == split_axis:
            continue
        rows = _group_line_rows(scaled, candidates, axis, options)
        score = (max((len(row) for row in rows.values()), default=0), -len(rows))
        if score > best_score:
            best_axis = axis
            best_rows = rows
            best_score = score
    return best_axis, best_rows


def _group_line_rows(
    scaled: np.ndarray,
    candidates: list[int],
    coordinate_axis: int,
    options: dict[str, Any],
) -> dict[Any, list[int]]:
    tolerance = _line_axis_tolerance(
        scaled,
        candidates,
        coordinate_axis,
        options,
        option_name="tolerance",
    )
    split_axis = _line_split_axis(options)
    split_tolerance = (
        _line_axis_tolerance(
            scaled,
            candidates,
            split_axis,
            options,
            option_name="split_tolerance",
        )
        if split_axis is not None
        else None
    )

    rows: dict[Any, list[int]] = {}
    for index in candidates:
        row_key = _axis_row_key(scaled[index][coordinate_axis], tolerance)
        if split_axis is not None and split_tolerance is not None:
            row_key = (
                _axis_row_key(scaled[index][split_axis], split_tolerance),
                row_key,
            )
        rows.setdefault(row_key, []).append(index)
    return rows


def _line_axis_tolerance(
    scaled: np.ndarray,
    candidates: list[int],
    axis: int,
    options: dict[str, Any],
    *,
    option_name: str,
) -> float:
    tolerance = options.get(option_name)
    tolerance = float(tolerance) if tolerance is not None else _infer_tolerance(
        scaled,
        candidates,
        axis,
    )
    if tolerance <= 0:
        raise ValueError(f"line_defects.{option_name} must be positive")
    return tolerance


def _axis_row_key(value: float, tolerance: float) -> int:
    return int(round(float(value) / tolerance))


def _line_split_axis(options: dict[str, Any]) -> int | None:
    split_axis = options.get("split_by_axis")
    if split_axis is None:
        return None
    split_axis = int(split_axis)
    if split_axis not in {0, 1, 2}:
        raise ValueError("line_defects.split_by_axis must be 0, 1, or 2")
    return split_axis


def _infer_tolerance(
    scaled: np.ndarray,
    candidates: list[int],
    coordinate_axis: int,
) -> float:
    values = sorted(
        {round(float(scaled[index][coordinate_axis]), 8) for index in candidates}
    )
    if len(values) < 2:
        return 0.05
    spacings = [
        values[index + 1] - values[index]
        for index in range(len(values) - 1)
        if values[index + 1] - values[index] > 1e-8
    ]
    if not spacings:
        return 0.05
    return max(min(spacings) * 0.45, 1e-4)


def _normalize_defect_options(settings: Any, *, default_max: int) -> dict[str, Any]:
    if settings in (None, False):
        return {"enabled": False, "max_count": 0}
    if settings is True:
        return {"enabled": True, "max_count": default_max}
    if not isinstance(settings, dict):
        raise ValueError("defect settings must be a mapping or boolean")
    options = dict(settings)
    if "mode" in options and "selection" not in options:
        options["selection"] = options["mode"]
    options["enabled"] = bool(options.get("enabled", True))
    options["max_count"] = int(options.get("max_count", default_max))
    if options["max_count"] < 0:
        raise ValueError("defect max_count can not be negative")
    return options


def _selection_mode(options: dict[str, Any], *, default: str = "ordered") -> str:
    return str(options.get("selection", default)).lower()


def _element_label(symbols: list[str] | tuple[str, ...]) -> str:
    unique = []
    for symbol in symbols:
        if symbol not in unique:
            unique.append(symbol)
    return "-".join(unique) if unique else "all"


def _coordinate_axis_label(axis: int) -> str:
    return ("const_a", "const_b", "const_c")[axis]


def _split_axis_label(axis: int | None) -> str:
    if axis is None:
        return ""
    return f"_split_{'abc'[axis]}"


def _split_description(axis: int | None) -> str:
    if axis is None:
        return ""
    return f"split by fractional {'abc'[axis]} coordinate, "


def _sample_items(items: list[Any], options: dict[str, Any]) -> list[Any]:
    max_count = min(int(options["max_count"]), len(items))
    if max_count <= 0:
        return []
    rng = np.random.default_rng(
        int(options["seed"]) if options.get("seed") is not None else None
    )
    selected = rng.choice(len(items), size=max_count, replace=False)
    return [items[index] for index in selected.tolist()]


def _candidate_indices(atoms, options: dict[str, Any]) -> list[int]:
    explicit_indices = options.get("indices")
    if explicit_indices is not None:
        return [int(index) for index in explicit_indices]

    elements = options.get("elements")
    if elements is None:
        return list(range(len(atoms)))
    allowed = {str(element) for element in elements}
    return [index for index, atom in enumerate(atoms) if atom.symbol in allowed]
