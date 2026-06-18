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

"""Scheduler and VASP resource selection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pesmaker.config.schema import PESMakerConfig


@dataclass(frozen=True)
class JobResources:
    """Scheduler resource settings shared by generated submit scripts."""

    nodes: int
    cores_cpu: int
    gpus: int
    vasp_kpar: int
    vasp_ncore: int
    write_vasp_kpar: bool = False
    write_vasp_ncore: bool = False


def _job_resources(
    config: PESMakerConfig,
    *,
    atom_count: int | None = None,
) -> JobResources:
    options = config.jobs.options
    nodes = _positive_int_option(options, "nodes", default=1)
    cores_cpu = _positive_int_option(options, "cores_cpu", default=1)
    gpus = _nonnegative_int_option(
        options,
        "gpus",
        default=_nonnegative_int_option(options, "gpus_gpu", default=0),
    )
    _reject_legacy_vasp_parallel_options(options)
    write_vasp_kpar = "vasp_kpar" in options
    write_vasp_ncore = "vasp_ncore" in options
    vasp_kpar = _positive_int_option(
        options,
        "vasp_kpar",
        default=_default_vasp_kpar(cores_cpu),
    )
    if cores_cpu % vasp_kpar != 0:
        raise ValueError("jobs.vasp_kpar must divide jobs.cores_cpu")
    if "vasp_ncore" in options:
        vasp_ncore = _positive_int_option(options, "vasp_ncore", default=1)
        if (cores_cpu // vasp_kpar) % vasp_ncore != 0:
            raise ValueError(
                "jobs.vasp_ncore must divide jobs.cores_cpu / jobs.vasp_kpar"
            )
    else:
        _, vasp_ncore = _vasp_parallel_factors(
            cores_cpu,
            atom_count=atom_count,
            kpar=vasp_kpar,
        )
    return JobResources(
        nodes=nodes,
        cores_cpu=cores_cpu,
        gpus=gpus,
        vasp_kpar=vasp_kpar,
        vasp_ncore=vasp_ncore,
        write_vasp_kpar=write_vasp_kpar,
        write_vasp_ncore=write_vasp_ncore,
    )


def _positive_int_option(
    options: dict[str, Any],
    key: str,
    *,
    default: int,
) -> int:
    value = int(options.get(key, default))
    if value < 1:
        raise ValueError(f"jobs.{key} must be a positive integer")
    return value


def _reject_legacy_vasp_parallel_options(options: dict[str, Any]) -> None:
    if "kpar" in options:
        raise ValueError("jobs.kpar has been renamed to jobs.vasp_kpar")
    if "ncore" in options:
        raise ValueError("jobs.ncore has been renamed to jobs.vasp_ncore")


def _nonnegative_int_option(
    options: dict[str, Any],
    key: str,
    *,
    default: int,
) -> int:
    value = int(options.get(key, default))
    if value < 0:
        raise ValueError(f"jobs.{key} must be a non-negative integer")
    return value


def _vasp_parallel_factors(
    cores_cpu: int,
    *,
    atom_count: int | None = None,
    kpar: int | None = None,
) -> tuple[int, int]:
    """Choose conservative VASP CPU parallel settings.

    KPAR defaults to two groups when the requested rank count permits it.
    NCORE is selected from ranks available within each KPAR group.
    """
    if cores_cpu < 1:
        raise ValueError("cores_cpu must be a positive integer")
    if kpar is None:
        kpar = _default_vasp_kpar(cores_cpu)
    if kpar < 1:
        raise ValueError("kpar must be a positive integer")
    if cores_cpu % kpar != 0:
        raise ValueError("kpar must divide cores_cpu")

    ranks_per_kpar = cores_cpu // kpar
    return kpar, _vasp_ncore_factor(ranks_per_kpar, atom_count=atom_count)


def _default_vasp_kpar(cores_cpu: int) -> int:
    if cores_cpu >= 2 and cores_cpu % 2 == 0:
        return 2
    return 1


def _vasp_ncore_factor(
    ranks_per_kpar: int,
    *,
    atom_count: int | None,
) -> int:
    if ranks_per_kpar <= 8:
        return 1

    target = ranks_per_kpar**0.5
    prefer_below = False
    if atom_count is not None:
        if atom_count > 400:
            target = max(target, 16)
            prefer_below = True
        elif atom_count >= 100:
            target = max(target, 4)

    return _factor_near(ranks_per_kpar, target, prefer_below=prefer_below)


def _factor_near(value: int, target: float, *, prefer_below: bool = False) -> int:
    factors = [factor for factor in range(1, value + 1) if value % factor == 0]
    if prefer_below:
        below = [factor for factor in factors if factor <= target]
        if below:
            return max(below)
    return min(factors, key=lambda factor: (abs(factor - target), factor))
