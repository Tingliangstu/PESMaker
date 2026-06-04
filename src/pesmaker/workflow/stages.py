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
"""Backward-compatible workflow stage imports.

The concrete stage implementations now live in the domain packages:
`generators`, `samplers`, `labelers`, `jobs`, `dataset`, and `trainers`.
This module intentionally re-exports the previous symbols so existing user
code and tests can keep importing from `pesmaker.workflow.stages`.
"""

from __future__ import annotations

from importlib import import_module

_REEXPORT_MODULES = (
    "pesmaker.results",
    "pesmaker.artifacts",
    "pesmaker.samplers.gpumd",
    "pesmaker.samplers.selection",
    "pesmaker.labelers.vasp",
    "pesmaker.jobs.resources",
    "pesmaker.jobs.scripts",
    "pesmaker.jobs.submit",
    "pesmaker.dataset.extxyz",
    "pesmaker.trainers.nep",
)

for _module_name in _REEXPORT_MODULES:
    _module = import_module(_module_name)
    for _name in dir(_module):
        if not _name.startswith("__"):
            globals()[_name] = getattr(_module, _name)

__all__ = sorted(
    name
    for name in globals()
    if not name.startswith("__")
    and name not in {"import_module", "_module", "_module_name", "_REEXPORT_MODULES"}
)
