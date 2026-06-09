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

"""GPUMD sampling setup and run.in rendering."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pesmaker.artifacts import (
    _load_input_records,
    _read_optional_file,
    _section_output_dir,
)
from pesmaker.config.schema import PESMakerConfig
from pesmaker.jobs.scripts import _write_submit_script
from pesmaker.results import StageResult
from pesmaker.structures import load_structure, write_structure

DEFAULT_GPUMD_RUN_IN = """potential      nep89_20250409.txt
velocity       300

ensemble       npt_scr 300 300 100 0 0 0 50 50 50 1000
time_step      1
dump_thermo    1000
dump_position  3000
run            {run_steps}
"""

DEFAULT_GPUMD_RUN_STEPS = 3000000

DEFAULT_GPUMD_DIR = Path("/home/tingliang/software/GPUMD/GPUMD-master-26-05-2026/src")

DEFAULT_GPUMD_NEP89_NAME = "nep89_20250409.txt"

DEFAULT_GPUMD_NEP89_RELATIVE_PATH = Path(
    "../potentials/nep/nep89_20250409/nep89_20250409.txt"
)

DEFAULT_GPUMD_TEMP_COUPLING = 100.0

DEFAULT_GPUMD_PRESSURE_COUPLING = 1000.0

DEFAULT_GPUMD_ORTHOGONAL_ELASTIC = (50.0, 50.0, 50.0)

DEFAULT_GPUMD_2D_ELASTIC = (50.0, 50.0, 200.0)

DEFAULT_GPUMD_TRICLINIC_ELASTIC = (50.0, 50.0, 50.0, 50.0, 50.0, 50.0)

DEFAULT_2D_VACUUM_THRESHOLD = 10.0

DEFAULT_2D_VACUUM_RATIO = 1.0


@dataclass(frozen=True)
class SamplingCondition:
    """One MD sampling temperature condition."""

    name: str
    start: float
    end: float


def setup_sampling(config: PESMakerConfig) -> StageResult:
    """Prepare MD sampling folders for GPUMD or future engines."""
    engine = config.sampling.engine.lower()
    output_dir = _section_output_dir(config, config.sampling.options, "sampling")
    output_dir.mkdir(parents=True, exist_ok=True)

    records = _load_input_records(config, config.sampling.options)
    conditions = _sampling_conditions(config.sampling.options)
    if _preserve_sampling_run_in(config.sampling.options) and not config.sampling.options.get(
        "run_in"
    ):
        raise ValueError("sampling.run_in is required when preserve_run_in is true")
    run_template = _read_optional_file(
        config.sampling.options.get("run_in"),
        default=DEFAULT_GPUMD_RUN_IN,
    )
    files: list[Path] = []
    warnings: list[str] = []
    manifest_path = output_dir / "sampling_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        job_index = 0
        for record_index, record in enumerate(records):
            for condition in conditions:
                stage_dir = output_dir / f"md_{record_index:06d}_{condition.name}"
                stage_dir.mkdir(parents=True, exist_ok=True)
                structure_path = stage_dir / "model.xyz"
                atoms = load_structure(record["path"])
                write_structure(atoms, structure_path, fmt="extxyz")
                potential_name, potential_path = _prepare_sampling_potential(
                    config.sampling.options,
                    stage_dir,
                )
                run_in_path = stage_dir / "run.in"
                run_in, run_in_warnings = _render_sampling_run_in(
                    config.sampling.options,
                    run_template,
                    condition,
                    atoms,
                    potential_name=potential_name,
                )
                for warning in run_in_warnings:
                    if warning not in warnings:
                        warnings.append(warning)
                run_in_path.write_text(run_in, encoding="utf-8")
                command = _sampling_command(config)
                submit_path = _write_submit_script(
                    config,
                    stage_dir,
                    stage="sampling",
                    command=command,
                )
                files.extend(
                    path
                    for path in (
                        structure_path,
                        potential_path,
                        run_in_path,
                        submit_path,
                    )
                    if path is not None
                )
                manifest.write(
                    json.dumps(
                        {
                            "index": job_index,
                            "engine": engine,
                            "source": record["path"],
                            "condition": condition.name,
                            "potential": potential_name,
                            "temperature_start": condition.start,
                            "temperature_end": condition.end,
                            "workdir": str(stage_dir),
                            "run_in": str(run_in_path),
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
        f"Prepared {job_count} GPUMD-MD job(s)",
        warnings=tuple(warnings),
    )


def _sampling_conditions(options: dict[str, Any]) -> tuple[SamplingCondition, ...]:
    temperatures = options.get("temperatures")
    if temperatures is not None:
        if isinstance(temperatures, str) and "-" in temperatures:
            start, end = _parse_temperature_range(temperatures)
            return (_ramp_temperature(start, end),)
        if not isinstance(temperatures, list) or not temperatures:
            raise ValueError("sampling.temperatures must be a non-empty list")
        if (
            len(temperatures) == 1
            and isinstance(temperatures[0], str)
            and "-" in temperatures[0]
        ):
            start, end = _parse_temperature_range(temperatures[0])
            return (_ramp_temperature(start, end),)
        return tuple(_constant_temperature(float(value)) for value in temperatures)

    temperature = options.get("temperature", options.get("temp", 300))
    if isinstance(temperature, str) and "-" in temperature:
        start, end = _parse_temperature_range(temperature)
        return (_ramp_temperature(start, end),)
    if isinstance(temperature, dict):
        start = float(temperature.get("start", temperature.get("from", 300)))
        end = float(temperature.get("end", temperature.get("to", start)))
        return (
            _ramp_temperature(start, end)
            if start != end
            else _constant_temperature(start),
        )
    if isinstance(temperature, list):
        if len(temperature) == 2:
            return (_ramp_temperature(float(temperature[0]), float(temperature[1])),)
        return tuple(_constant_temperature(float(value)) for value in temperature)
    return (_constant_temperature(float(temperature)),)


def _parse_temperature_range(value: str) -> tuple[float, float]:
    parts = [part.strip() for part in value.split("-", 1)]
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"invalid sampling temperature range: {value}")
    return float(parts[0]), float(parts[1])


def _constant_temperature(temperature: float) -> SamplingCondition:
    label = _temperature_label(temperature)
    return SamplingCondition(name=f"temp_{label}K", start=temperature, end=temperature)


def _ramp_temperature(start: float, end: float) -> SamplingCondition:
    return SamplingCondition(
        name=f"ramp_{_temperature_label(start)}K_to_{_temperature_label(end)}K",
        start=start,
        end=end,
    )


def _temperature_label(value: float) -> str:
    return (
        str(int(value)) if float(value).is_integer() else str(value).replace(".", "p")
    )


def _render_sampling_run_in(
    options: dict[str, Any],
    template: str,
    condition: SamplingCondition,
    atoms,
    *,
    potential_name: str | None = None,
) -> tuple[str, tuple[str, ...]]:
    if _preserve_sampling_run_in(options):
        return _ensure_trailing_newline(template), ()
    potential = potential_name or str(options.get("potential", "nep89_20250409.txt"))
    ensemble = _sampling_ensemble_line(options, condition, atoms)
    explicit_run_steps = _has_explicit_sampling_run_steps(options)
    run_steps = _sampling_run_steps(options)
    rendered = template.format(
        potential=potential,
        ensemble=ensemble,
        run_steps=run_steps,
        temperature=_format_temperature(condition.start),
        temperature_start=_format_temperature(condition.start),
        temperature_end=_format_temperature(condition.end),
    )
    warnings = _sampling_run_in_warnings(options, rendered, atoms)
    rendered = _rewrite_gpumd_run_line(
        rendered, "potential", f"potential      {potential}"
    )
    rendered = _rewrite_gpumd_run_line(
        rendered,
        "velocity",
        f"velocity       {_format_temperature(condition.start)}",
    )
    rendered = _rewrite_gpumd_run_line(rendered, "ensemble", ensemble)
    if explicit_run_steps or not _has_gpumd_run_line(rendered, "run"):
        rendered = _rewrite_gpumd_run_line(rendered, "run", f"run            {run_steps}")
    return rendered, warnings


def _sampling_ensemble_line(
    options: dict[str, Any],
    condition: SamplingCondition,
    atoms,
) -> str:
    mode = _sampling_ensemble_mode(options, atoms)
    start = _format_temperature(condition.start)
    end = _format_temperature(condition.end)
    tau_t = _format_gpumd_number(
        float(options.get("temperature_coupling", DEFAULT_GPUMD_TEMP_COUPLING))
    )
    tau_p = _format_gpumd_number(
        float(options.get("pressure_coupling", DEFAULT_GPUMD_PRESSURE_COUPLING))
    )

    if mode in {"triclinic", "2d_triclinic"}:
        pressures = _sampling_float_values(
            options.get("pressure", options.get("pressures", 0.0)),
            count=6,
            default=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            shear_default=0.0,
        )
        default_elastic = (
            _default_2d_triclinic_elastic(options, atoms)
            if mode == "2d_triclinic"
            else DEFAULT_GPUMD_TRICLINIC_ELASTIC
        )
        elastic = _sampling_float_values(
            options.get("elastic_constants", options.get("elastic", None)),
            count=6,
            default=default_elastic,
        )
    else:
        default_elastic = (
            DEFAULT_GPUMD_2D_ELASTIC
            if mode == "2d"
            else DEFAULT_GPUMD_ORTHOGONAL_ELASTIC
        )
        pressures = _sampling_float_values(
            options.get("pressure", options.get("pressures", 0.0)),
            count=3,
            default=(0.0, 0.0, 0.0),
        )
        elastic = _sampling_float_values(
            options.get("elastic_constants", options.get("elastic", None)),
            count=3,
            default=default_elastic,
        )

    values = [start, end, tau_t, *map(_format_gpumd_number, pressures)]
    values.extend(map(_format_gpumd_number, elastic))
    values.append(tau_p)
    return "ensemble       npt_scr " + " ".join(values)


def _sampling_ensemble_mode(options: dict[str, Any], atoms) -> str:
    raw_mode = str(
        options.get("ensemble_mode", options.get("cell_mode", "auto"))
    ).lower()
    aliases = {
        "auto": "auto",
        "orthogonal": "orthogonal",
        "orthorhombic": "orthogonal",
        "ortho": "orthogonal",
        "triclinic": "triclinic",
        "tri": "triclinic",
        "2d": "2d",
        "2d_orthogonal": "2d",
        "2d-orthogonal": "2d",
        "2d_triclinic": "2d_triclinic",
        "2d-triclinic": "2d_triclinic",
        "slab": "2d",
        "surface": "2d",
    }
    if raw_mode not in aliases:
        raise ValueError(
            "sampling.ensemble_mode must be one of: auto, orthogonal, "
            "triclinic, 2d, 2d_triclinic"
        )
    mode = aliases[raw_mode]
    if mode == "2d":
        return "2d" if _is_orthogonal_cell(atoms) else "2d_triclinic"
    if mode != "auto":
        return mode
    if _is_two_dimensional_sampling_cell(options, atoms):
        return "2d" if _is_orthogonal_cell(atoms) else "2d_triclinic"
    if _is_orthogonal_cell(atoms):
        return "orthogonal"
    return "triclinic"


def _default_2d_triclinic_elastic(options: dict[str, Any], atoms) -> tuple[float, ...]:
    values = [50.0, 50.0, 50.0, 50.0, 50.0, 50.0]
    axis = _two_dimensional_axis(options, atoms)
    if axis is None:
        axis = 2
    for index in _triclinic_elastic_indices_for_axis(axis):
        values[index] = 200.0
    return tuple(values)


def _triclinic_elastic_indices_for_axis(axis: int) -> tuple[int, ...]:
    # GPUMD triclinic order: C_xx, C_yy, C_zz, C_yz, C_xz, C_xy.
    if axis == 0:
        return (0, 4, 5)
    if axis == 1:
        return (1, 3, 5)
    return (2, 3, 4)


def _sampling_float_values(
    value: Any,
    *,
    count: int,
    default: tuple[float, ...],
    shear_default: float | None = None,
) -> tuple[float, ...]:
    if value is None:
        return default
    if isinstance(value, (int, float, str)):
        return tuple(float(value) for _ in range(count))
    if not isinstance(value, list):
        raise ValueError(
            "sampling pressure and elastic options must be scalars or lists"
        )
    values = tuple(float(item) for item in value)
    if len(values) == count:
        return values
    if count == 6 and len(values) == 3:
        fill = 0.0 if shear_default is None else shear_default
        return (*values, fill, fill, fill)
    raise ValueError(f"sampling option must contain {count} value(s)")


def _sampling_run_steps(options: dict[str, Any]) -> int:
    value = int(options.get("run_steps", options.get("steps", DEFAULT_GPUMD_RUN_STEPS)))
    if value < 1:
        raise ValueError("sampling.run_steps must be a positive integer")
    return value


def _has_explicit_sampling_run_steps(options: dict[str, Any]) -> bool:
    return "run_steps" in options or "steps" in options


def _sampling_run_in_warnings(
    options: dict[str, Any],
    rendered: str,
    atoms,
) -> tuple[str, ...]:
    if not options.get("run_in"):
        return ()
    mode = _sampling_ensemble_mode(options, atoms)
    if mode not in {"triclinic", "2d_triclinic"}:
        return ()
    line = _find_gpumd_run_line(rendered, "ensemble")
    if line is None or "npt_scr" not in line:
        return ()
    if _npt_scr_value_count(line) == 16:
        return ()
    return ("GPUMD run.in npt_scr was adjusted for triclinic cell format.",)


def _find_gpumd_run_line(text: str, keyword: str) -> str | None:
    for line in text.splitlines():
        if line.strip().startswith(keyword):
            return line
    return None


def _has_gpumd_run_line(text: str, keyword: str) -> bool:
    return _find_gpumd_run_line(text, keyword) is not None


def _npt_scr_value_count(line: str) -> int:
    parts = line.split()
    try:
        index = parts.index("npt_scr")
    except ValueError:
        return 0
    return len(parts) - index - 1


def _is_orthogonal_cell(atoms) -> bool:
    cell = np.asarray(atoms.cell.array, dtype=float)
    metric = cell @ cell.T
    off_diagonal = metric[~np.eye(3, dtype=bool)]
    lengths = np.linalg.norm(cell, axis=1)
    scale = max(float(np.max(lengths) ** 2), 1.0)
    return bool(np.all(np.abs(off_diagonal) <= 1e-8 * scale))


def _is_two_dimensional_sampling_cell(options: dict[str, Any], atoms) -> bool:
    return _two_dimensional_axis(options, atoms) is not None


def _two_dimensional_axis(options: dict[str, Any], atoms) -> int | None:
    pbc = np.asarray(atoms.pbc, dtype=bool)
    if pbc.shape == (3,) and not bool(np.all(pbc)):
        nonperiodic = np.where(~pbc)[0]
        if len(nonperiodic):
            return int(nonperiodic[0])

    if len(atoms) == 0:
        return None
    threshold = float(
        options.get("two_dimensional_vacuum_threshold", DEFAULT_2D_VACUUM_THRESHOLD)
    )
    ratio = float(options.get("two_dimensional_vacuum_ratio", DEFAULT_2D_VACUUM_RATIO))
    cell = np.asarray(atoms.cell.array, dtype=float)
    positions = np.asarray(atoms.get_positions(), dtype=float)
    for axis, axis_vector in enumerate(cell):
        length = float(np.linalg.norm(axis_vector))
        if length <= 0.0:
            continue
        axis_unit = axis_vector / length
        projections = positions @ axis_unit
        span = float(np.max(projections) - np.min(projections))
        vacuum = max(length - span, 0.0)
        if vacuum >= threshold and (span <= 0.0 or vacuum >= ratio * span):
            return int(axis)
    return None


def _prepare_sampling_potential(
    options: dict[str, Any],
    stage_dir: Path,
) -> tuple[str, Path | None]:
    potential = _resolve_sampling_potential_path(options)
    if potential is not None and potential.exists() and potential.is_file():
        target = stage_dir / potential.name
        shutil.copy2(potential, target)
        return potential.name, target
    return _sampling_potential_name(options), None


def _resolve_sampling_potential_path(options: dict[str, Any]) -> Path | None:
    raw_potential = options.get("potential")
    gpumd_dir = _sampling_gpumd_dir(options)
    candidates = []

    if raw_potential:
        potential = Path(str(raw_potential))
        candidates.append(potential)
        if not potential.is_absolute() and gpumd_dir is not None:
            candidates.append(gpumd_dir / potential)
            if potential.name == DEFAULT_GPUMD_NEP89_NAME:
                candidates.append(gpumd_dir / DEFAULT_GPUMD_NEP89_RELATIVE_PATH)
    elif gpumd_dir is not None:
        candidates.append(gpumd_dir / DEFAULT_GPUMD_NEP89_RELATIVE_PATH)
        candidates.append(Path(DEFAULT_GPUMD_NEP89_NAME))
    else:
        candidates.append(Path(DEFAULT_GPUMD_NEP89_NAME))

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return candidates[0] if candidates else None


def _sampling_potential_name(options: dict[str, Any]) -> str:
    potential = _resolve_sampling_potential_path(options)
    if potential is not None and potential.exists():
        return potential.name
    raw_potential = options.get("potential")
    if raw_potential:
        return Path(str(raw_potential)).name
    return DEFAULT_GPUMD_NEP89_NAME


def _sampling_gpumd_dir(options: dict[str, Any]) -> Path | None:
    value = options.get("gpumd_dir")
    if value:
        return Path(str(value))
    if DEFAULT_GPUMD_DIR.exists() and DEFAULT_GPUMD_DIR.is_dir():
        return DEFAULT_GPUMD_DIR
    return None


def _rewrite_gpumd_run_line(text: str, keyword: str, replacement: str) -> str:
    lines = text.splitlines()
    replaced = False
    output = []
    for line in lines:
        if line.strip().startswith(keyword):
            output.append(replacement)
            replaced = True
        else:
            output.append(line)
    if not replaced:
        output.insert(0, replacement)
    return "\n".join(output) + "\n"


def _format_temperature(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def _format_gpumd_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def _sampling_command(config: PESMakerConfig) -> str:
    if config.sampling.engine.lower() == "gpumd":
        command = config.sampling.options.get("command")
        if command:
            return str(command)
        gpumd_dir = _sampling_gpumd_dir(config.sampling.options)
        if gpumd_dir:
            return str(gpumd_dir / "gpumd")
        return "gpumd"
    return str(config.sampling.options.get("command", config.sampling.engine))


def _preserve_sampling_run_in(options: dict[str, Any]) -> bool:
    """Return true when a user-provided run input should be copied verbatim."""
    for key in ("preserve_run_in", "keep_run_in", "copy_run_in"):
        if _sampling_bool_option(options.get(key), default=False):
            return True
    if "rewrite_run_in" in options:
        return not _sampling_bool_option(options.get("rewrite_run_in"), default=True)
    if "render_run_in" in options:
        return not _sampling_bool_option(options.get("render_run_in"), default=True)
    return False


def _sampling_bool_option(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _ensure_trailing_newline(text: str) -> str:
    return text if text.endswith("\n") else f"{text}\n"
