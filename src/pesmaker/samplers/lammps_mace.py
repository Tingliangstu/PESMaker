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

"""LAMMPS MLIAP/MACE sampling setup."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pesmaker.artifacts import _load_input_records, _section_output_dir
from pesmaker.config.schema import PESMakerConfig
from pesmaker.jobs.scripts import _write_submit_script
from pesmaker.results import StageResult
from pesmaker.samplers.gpumd import (
    SamplingCondition,
    _ensure_trailing_newline,
    _format_temperature,
    _preserve_sampling_run_in,
    _sampling_ensemble_mode,
    _sampling_conditions,
    _two_dimensional_axis,
)
from pesmaker.structures import load_structure

DEFAULT_MACE_RUN_IN_NAME = "in.run_mace_npt"

DEFAULT_MACE_TRAJECTORY = "mace.lammpstrj"

DEFAULT_MACE_RUN_IN = """units         metal
dimension     3
boundary      p p p
atom_style    atomic
atom_modify   map yes
newton        on

variable      ts         equal 0.001
variable      Tstart     equal {temperature_start}
variable      Tstop      equal {temperature_end}
variable      Tdamp      equal ${ts}*100
variable      Pdamp      equal ${ts}*1000
variable      P_0        equal 0.0
variable      dt_dump    equal 3000
variable      dt_thermo  equal 1000

read_data     {data_file}

pair_style    mliap unified {potential} 0
pair_coeff    * * {elements}

dump          myDump all custom ${dt_dump} {trajectory} id element x y z
dump_modify   myDump sort id element {elements}

thermo_style  custom step time cpu pe ke etotal temp press vol density
thermo        ${dt_thermo}
thermo_modify lost ignore

velocity      all create ${Tstart} 123456 mom yes rot yes dist gaussian
fix           MD all npt temp ${Tstart} ${Tstop} ${Tdamp} &
              x ${P_0} ${P_0} ${Pdamp} &
              y ${P_0} ${P_0} ${Pdamp} &
              z ${P_0} ${P_0} ${Pdamp} &
              couple none

timestep      ${ts}
run           3000000
"""


def setup_sampling(config: PESMakerConfig) -> StageResult:
    """Prepare LAMMPS-MACE sampling folders."""
    engine = config.sampling.engine.lower()
    options = config.sampling.options
    output_dir = _section_output_dir(config, options, "sampling")
    output_dir.mkdir(parents=True, exist_ok=True)

    records = _load_input_records(config, options)
    conditions = _sampling_conditions(options)
    if _preserve_sampling_run_in(options) and not options.get("run_in"):
        raise ValueError("sampling.run_in is required when preserve_run_in is true")
    run_template = _mace_run_template(options)
    run_in_name = _mace_run_in_name(options)
    potential = _mace_potential(options)
    trajectory = str(options.get("trajectory", DEFAULT_MACE_TRAJECTORY))

    files: list[Path] = []
    manifest_path = output_dir / "sampling_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        job_index = 0
        for record_index, record in enumerate(records):
            for condition in conditions:
                stage_dir = output_dir / f"md_{record_index:06d}_{condition.name}"
                stage_dir.mkdir(parents=True, exist_ok=True)
                atoms = load_structure(record["path"])
                elements = _first_seen_elements(atoms)
                mode = _sampling_ensemble_mode(options, atoms)

                data_path = stage_dir / "data.in"
                _write_lammps_data(
                    atoms,
                    data_path,
                    elements,
                    force_skew=mode in {"triclinic", "2d", "2d_triclinic"},
                )

                run_in_path = stage_dir / run_in_name
                run_in_path.write_text(
                    _render_mace_run_in(
                        options,
                        run_template,
                        condition,
                        atoms,
                        data_file=data_path.name,
                        potential=potential,
                        elements=elements,
                        mode=mode,
                        trajectory=trajectory,
                    ),
                    encoding="utf-8",
                )

                submit_path = _write_submit_script(
                    config,
                    stage_dir,
                    stage="sampling",
                    command=_mace_command(options, run_in_name),
                )
                files.extend((data_path, run_in_path, submit_path))
                manifest.write(
                    json.dumps(
                        {
                            "index": job_index,
                            "engine": engine,
                            "source": record["path"],
                            "condition": condition.name,
                            "potential": potential,
                            "elements": elements,
                            "temperature_start": condition.start,
                            "temperature_end": condition.end,
                            "workdir": str(stage_dir),
                            "data_file": str(data_path),
                            "run_in": str(run_in_path),
                            "trajectory": str(stage_dir / trajectory),
                            "submit_script": str(submit_path),
                        }
                    )
                    + "\n"
                )
                job_index += 1
    files.append(manifest_path)
    job_count = len(records) * len(conditions)
    return StageResult(
        output_dir,
        tuple(files),
        f"Prepared {job_count} MACE job(s)",
    )


def _mace_run_template(options: dict[str, Any]) -> str:
    run_in = options.get("run_in")
    if run_in:
        return Path(str(run_in)).read_text(encoding="utf-8")
    return DEFAULT_MACE_RUN_IN


def _mace_run_in_name(options: dict[str, Any]) -> str:
    run_in = options.get("run_in")
    if run_in:
        return Path(str(run_in)).name
    return DEFAULT_MACE_RUN_IN_NAME


def _mace_potential(options: dict[str, Any]) -> str:
    raw_potential = options.get("potential")
    if not raw_potential:
        raise ValueError("sampling.potential is required for MACE sampling")
    potential = Path(str(raw_potential))
    if potential.exists():
        return str(potential.resolve())
    return str(raw_potential)


def _mace_command(options: dict[str, Any], run_in_name: str) -> str:
    command = options.get("command")
    if command:
        return str(command)
    return f"lmp -in {run_in_name}"


def _render_mace_run_in(
    options: dict[str, Any],
    template: str,
    condition: SamplingCondition,
    atoms,
    *,
    data_file: str,
    potential: str,
    elements: list[str],
    mode: str,
    trajectory: str,
) -> str:
    if _preserve_sampling_run_in(options):
        return _ensure_trailing_newline(template)
    values = {
        "data_file": data_file,
        "potential": potential,
        "model": potential,
        "elements": " ".join(elements),
        "temperature": _format_temperature(condition.start),
        "temperature_start": _format_temperature(condition.start),
        "temperature_end": _format_temperature(condition.end),
        "trajectory": trajectory,
    }
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f"{{{key}}}", str(value))
    rendered = _rewrite_common_mace_lammps_lines(
        rendered,
        options,
        condition,
        atoms,
        data_file=data_file,
        potential=potential,
        elements=elements,
        mode=mode,
    )
    if not rendered.endswith("\n"):
        rendered += "\n"
    return rendered


def _rewrite_common_mace_lammps_lines(
    text: str,
    options: dict[str, Any],
    condition: SamplingCondition,
    atoms,
    *,
    data_file: str,
    potential: str,
    elements: list[str],
    mode: str,
) -> str:
    """Adjust common literal MACE/LAMMPS templates that lack placeholders."""
    rendered: list[str] = []
    skip_element_continuation = False
    skip_fix_continuation = False
    for line in text.splitlines():
        if skip_element_continuation and _is_element_continuation(line):
            continue
        skip_element_continuation = False
        if skip_fix_continuation and _is_fix_continuation(line):
            continue
        skip_fix_continuation = False

        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            rendered.append(line)
            continue

        variable_line = _rewrite_temperature_variable(line, condition)
        if variable_line is not None:
            rendered.append(variable_line)
            continue

        if _starts_with_lammps_command(stripped, "read_data"):
            rendered.append(f"read_data     {data_file}")
            continue

        pair_style_line = _rewrite_pair_style_mliap(line, potential)
        if pair_style_line is not None:
            rendered.append(pair_style_line)
            continue

        pair_coeff_line = _rewrite_pair_coeff(line, elements)
        if pair_coeff_line is not None:
            rendered.append(pair_coeff_line)
            skip_element_continuation = stripped.endswith("&")
            continue

        dump_modify_line = _rewrite_dump_modify(line, elements)
        if dump_modify_line is not None:
            rendered.append(dump_modify_line)
            continue

        velocity_line = _rewrite_velocity(line)
        if velocity_line is not None:
            rendered.append(velocity_line)
            continue

        fix_line = _rewrite_fix_npt(
            line,
            options,
            atoms,
            mode=mode,
        )
        if fix_line is not None:
            rendered.append(fix_line)
            skip_fix_continuation = stripped.endswith("&")
            continue

        rendered.append(line)
    return "\n".join(rendered)


def _rewrite_temperature_variable(
    line: str,
    condition: SamplingCondition,
) -> str | None:
    name = _lammps_variable_name(line)
    if name == "T":
        return "\n".join(
            [
                f"variable      T          equal  {_format_temperature(condition.start)}",
                f"variable      Tstart     equal  {_format_temperature(condition.start)}",
                f"variable      Tstop      equal  {_format_temperature(condition.end)}",
            ]
        )
    if name == "Tstart":
        return f"variable      Tstart     equal  {_format_temperature(condition.start)}"
    if name == "Tstop":
        return f"variable      Tstop      equal  {_format_temperature(condition.end)}"
    return None


def _lammps_variable_name(line: str) -> str | None:
    parts = line.split()
    if len(parts) >= 2 and parts[0].lower() == "variable":
        return parts[1]
    return None


def _rewrite_pair_style_mliap(line: str, potential: str) -> str | None:
    stripped = line.strip()
    if "mliap" not in stripped or "unified" not in stripped:
        return None
    has_continuation = stripped.endswith("&")
    prefix = _line_indent(line)
    if _starts_with_lammps_command(stripped, "pair_style"):
        rewritten = f"{prefix}pair_style    mliap unified {potential} 0"
    elif stripped.startswith("mliap "):
        rewritten = f"{prefix}mliap unified {potential} 0"
    else:
        return None
    if has_continuation:
        rewritten += " &"
    return rewritten


def _rewrite_pair_coeff(line: str, elements: list[str]) -> str | None:
    stripped = line.strip()
    if not _starts_with_lammps_command(stripped, "pair_coeff"):
        return None
    if "dispersion/d3" in stripped:
        return f"{_line_indent(line)}pair_coeff    * * dispersion/d3 {_elements_text(elements)}"
    if "mliap" in stripped:
        return f"{_line_indent(line)}pair_coeff    * * mliap {_elements_text(elements)}"
    return f"{_line_indent(line)}pair_coeff    * * {_elements_text(elements)}"


def _rewrite_dump_modify(line: str, elements: list[str]) -> str | None:
    parts = line.split()
    if not parts or parts[0].lower() != "dump_modify":
        return None
    lowered = [part.lower() for part in parts]
    if "element" not in lowered:
        return None
    index = lowered.index("element")
    prefix = "  ".join(parts[: index + 1])
    return f"{_line_indent(line)}{prefix}  {_elements_text(elements)}"


def _rewrite_velocity(line: str) -> str | None:
    parts = line.split()
    if len(parts) < 4 or parts[0].lower() != "velocity":
        return None
    lowered = [part.lower() for part in parts]
    if "create" not in lowered:
        return None
    create_index = lowered.index("create")
    if create_index + 1 >= len(parts):
        return None
    parts[create_index + 1] = "${Tstart}"
    return f"{_line_indent(line)}{'  '.join(parts)}"


def _rewrite_fix_npt(
    line: str,
    options: dict[str, Any],
    atoms,
    *,
    mode: str,
) -> str | None:
    parts = line.split()
    if len(parts) < 4 or parts[0].lower() != "fix":
        return None
    style = parts[3].lower()
    if style not in {"nvt", "npt"}:
        return None
    return _mace_npt_fix(
        options,
        atoms,
        fix_id=parts[1],
        mode=mode,
        indent=_line_indent(line),
    )


def _mace_npt_fix(
    options: dict[str, Any],
    atoms,
    *,
    fix_id: str,
    mode: str,
    indent: str,
) -> str:
    pressure = str(options.get("pressure", options.get("pressures", "${P_0}")))
    dimensions = _npt_dimensions(options, atoms, mode)
    lines = [
        (
            f"{indent}fix          {fix_id} all npt temp "
            "${Tstart} ${Tstop} ${Tdamp} &"
        )
    ]
    for index, dimension in enumerate(dimensions):
        suffix = " &" if index < len(dimensions) - 1 else " &"
        lines.append(
            (
                f"{indent}             {dimension} {pressure} "
                f"{pressure} ${{Pdamp}}{suffix}"
            )
        )
    lines.append(f"{indent}             couple none")
    return "\n".join(lines)


def _npt_dimensions(options: dict[str, Any], atoms, mode: str) -> list[str]:
    if mode == "triclinic":
        return ["x", "y", "z", "xy", "xz", "yz"]
    if mode in {"2d", "2d_triclinic"}:
        axis = _two_dimensional_axis(options, atoms)
        if axis == 0:
            return ["y", "z", "yz"]
        if axis == 1:
            return ["x", "z", "xz"]
        return ["x", "y", "xy"]
    return ["x", "y", "z"]


def _is_element_continuation(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    tokens = [token for token in stripped.rstrip("&").split() if token != "&"]
    return bool(tokens) and all(
        re.fullmatch(r"[A-Z][a-z]?|NULL", token) for token in tokens
    )


def _is_fix_continuation(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    first = stripped.split()[0].lower()
    return first in {"x", "y", "z", "xy", "xz", "yz", "couple"}


def _starts_with_lammps_command(stripped: str, command: str) -> bool:
    return stripped == command or stripped.startswith(f"{command} ")


def _line_indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _elements_text(elements: list[str]) -> str:
    return " ".join(elements)


def _first_seen_elements(atoms) -> list[str]:
    elements: list[str] = []
    for symbol in atoms.get_chemical_symbols():
        if symbol not in elements:
            elements.append(symbol)
    return elements


def _write_lammps_data(
    atoms,
    path: Path,
    elements: list[str],
    *,
    force_skew: bool = False,
) -> None:
    try:
        from ase.io.lammpsdata import write_lammps_data
    except ImportError as exc:
        raise RuntimeError("Writing LAMMPS data files requires ASE") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        write_lammps_data(
            handle,
            atoms,
            specorder=elements,
            force_skew=force_skew,
            masses=True,
            atom_style="atomic",
            units="metal",
        )
