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
"""Stage setup helpers for sampling, labeling, dataset, and training workflows."""

from __future__ import annotations

import gzip
import json
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np

from pesmaker.config.schema import PESMakerConfig
from pesmaker.structures import load_structure, write_structure


DEFAULT_GPUMD_RUN_IN = """potential      nep89_20250409.txt
velocity       300

ensemble       npt_scr 300 300 100 0 0 0 20 20 100 1000
time_step      1
dump_thermo    1000
dump_position  3000
run            3000000
"""

DEFAULT_INCAR = """SYSTEM = PESMaker single point
GGA = PE
LREAL = Auto
ENCUT = 650
KSPACING = 0.2
KGAMMA = .TRUE.
NSW = 1
IBRION = -1
ALGO = Normal
EDIFF = 1E-06
SIGMA = 0.02
ISMEAR = 0
PREC = Accurate
NELM = 150
"""

RECOMMENDED_PBE_POTCARS = {
    "H": "H",
    "He": "He",
    "Li": "Li_sv",
    "Be": "Be",
    "B": "B",
    "C": "C",
    "N": "N",
    "O": "O",
    "F": "F",
    "Ne": "Ne",
    "Na": "Na_pv",
    "Mg": "Mg",
    "Al": "Al",
    "Si": "Si",
    "P": "P",
    "S": "S",
    "Cl": "Cl",
    "Ar": "Ar",
    "K": "K_sv",
    "Ca": "Ca_sv",
    "Sc": "Sc_sv",
    "Ti": "Ti_sv",
    "V": "V_sv",
    "Cr": "Cr_pv",
    "Mn": "Mn_pv",
    "Fe": "Fe",
    "Co": "Co",
    "Ni": "Ni",
    "Cu": "Cu",
    "Zn": "Zn",
    "Ga": "Ga_d",
    "Ge": "Ge_d",
    "As": "As",
    "Se": "Se",
    "Br": "Br",
    "Kr": "Kr",
    "Rb": "Rb_sv",
    "Sr": "Sr_sv",
    "Y": "Y_sv",
    "Zr": "Zr_sv",
    "Nb": "Nb_sv",
    "Mo": "Mo_sv",
    "Tc": "Tc_pv",
    "Ru": "Ru_pv",
    "Rh": "Rh_pv",
    "Pd": "Pd",
    "Ag": "Ag",
    "Cd": "Cd",
    "In": "In_d",
    "Sn": "Sn_d",
    "Sb": "Sb",
    "Te": "Te",
    "I": "I",
    "Xe": "Xe",
    "Cs": "Cs_sv",
    "Ba": "Ba_sv",
    "La": "La",
    "Ce": "Ce",
    "Pr": "Pr_3",
    "Nd": "Nd_3",
    "Pm": "Pm_3",
    "Sm": "Sm_3",
    "Eu": "Eu_2",
    "Gd": "Gd_3",
    "Tb": "Tb_3",
    "Dy": "Dy_3",
    "Ho": "Ho_3",
    "Er": "Er_3",
    "Tm": "Tm_3",
    "Yb": "Yb_2",
    "Lu": "Lu_3",
    "Hf": "Hf_pv",
    "Ta": "Ta_pv",
    "W": "W_sv",
    "Re": "Re",
    "Os": "Os",
    "Ir": "Ir",
    "Pt": "Pt",
    "Au": "Au",
    "Hg": "Hg",
    "Tl": "Tl_d",
    "Pb": "Pb_d",
    "Bi": "Bi_d",
    "Po": "Po_d",
    "At": "At",
    "Rn": "Rn",
    "Fr": "Fr_sv",
    "Ra": "Ra_sv",
    "Ac": "Ac",
    "Th": "Th",
    "Pa": "Pa",
    "U": "U",
    "Np": "Np",
    "Pu": "Pu",
    "Am": "Am",
    "Cm": "Cm",
}

RECOMMENDED_GW_POTCARS = {
    "H": "H_GW",
    "He": "He_GW",
    "Li": "Li_sv_GW",
    "Be": "Be_sv_GW",
    "B": "B_GW",
    "C": "C_GW",
    "N": "N_GW",
    "O": "O_GW",
    "F": "F_GW",
    "Ne": "Ne_GW",
    "Na": "Na_sv_GW",
    "Mg": "Mg_sv_GW",
    "Al": "Al_GW",
    "Si": "Si_GW",
    "P": "P_GW",
    "S": "S_GW",
    "Cl": "Cl_GW",
    "Ar": "Ar_GW",
    "K": "K_sv_GW",
    "Ca": "Ca_sv_GW",
    "Sc": "Sc_sv_GW",
    "Ti": "Ti_sv_GW",
    "V": "V_sv_GW",
    "Cr": "Cr_sv_GW",
    "Mn": "Mn_sv_GW",
    "Fe": "Fe_sv_GW",
    "Co": "Co_sv_GW",
    "Ni": "Ni_sv_GW",
    "Cu": "Cu_sv_GW",
    "Zn": "Zn_sv_GW",
    "Ga": "Ga_d_GW",
    "Ge": "Ge_d_GW",
    "As": "As_GW",
    "Se": "Se_GW",
    "Br": "Br_GW",
    "Kr": "Kr_GW",
    "Rb": "Rb_sv_GW",
    "Sr": "Sr_sv_GW",
    "Y": "Y_sv_GW",
    "Zr": "Zr_sv_GW",
    "Nb": "Nb_sv_GW",
    "Mo": "Mo_sv_GW",
    "Tc": "Tc_sv_GW",
    "Ru": "Ru_sv_GW",
    "Rh": "Rh_sv_GW",
    "Pd": "Pd_sv_GW",
    "Ag": "Ag_sv_GW",
    "Cd": "Cd_sv_GW",
    "In": "In_d_GW",
    "Sn": "Sn_d_GW",
    "Sb": "Sb_d_GW",
    "Te": "Te_GW",
    "I": "I_GW",
    "Xe": "Xe_GW",
    "Cs": "Cs_sv_GW",
    "Ba": "Ba_sv_GW",
    "La": "La_GW",
    "Ce": "Ce_GW",
    "Hf": "Hf_sv_GW",
    "Ta": "Ta_sv_GW",
    "W": "W_sv_GW",
    "Re": "Re_sv_GW",
    "Os": "Os_sv_GW",
    "Ir": "Ir_sv_GW",
    "Pt": "Pt_sv_GW",
    "Au": "Au_sv_GW",
    "Hg": "Hg_sv_GW",
    "Tl": "Tl_d_GW",
    "Pb": "Pb_d_GW",
    "Bi": "Bi_d_GW",
    "Po": "Po_d_GW",
    "At": "At_d_GW",
    "Rn": "Rn_d_GW",
}

DEFAULT_NEP_IN = """type 1 Te
version 4
prediction 0
potential nep.txt
"""


@dataclass(frozen=True)
class StageResult:
    """Summary for a prepared or collected workflow stage."""

    output_dir: Path
    files: tuple[Path, ...]
    message: str


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
    run_template = _read_optional_file(
        config.sampling.options.get("run_in"),
        default=DEFAULT_GPUMD_RUN_IN,
    )
    files: list[Path] = []
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
                run_in = _render_sampling_run_in(
                    config.sampling.options,
                    run_template,
                    condition,
                    potential_name=potential_name,
                )
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
                    for path in (structure_path, potential_path, run_in_path, submit_path)
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
                        }
                    )
                    + "\n"
                )
                job_index += 1
    files.append(manifest_path)
    job_count = len(records) * len(conditions)
    return StageResult(output_dir, tuple(files), f"Prepared {job_count} MD job(s)")


def select_sampling_frames(config: PESMakerConfig) -> StageResult:
    """Select representative MD frames with farthest point sampling."""
    options = config.sampling.options.get("selection", {})
    if not isinstance(options, dict):
        raise ValueError("sampling.selection must be a mapping")
    pattern = str(options.get("trajectory_pattern", "runs/*/sampling/**/movie.xyz"))
    output_dir = Path(str(options.get("output_dir", "selected")))
    min_distance = float(options.get("min_distance", 0.0))
    max_count = options.get("max_count")
    max_count = int(max_count) if max_count is not None else None

    frames = _read_trajectory_frames(pattern)
    features = _structure_features(frames)
    selected_indices = _farthest_point_indices(
        features,
        min_distance=min_distance,
        max_count=max_count,
    )
    selected = [frames[index] for index in selected_indices]

    output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = output_dir / "selected.xyz"
    _write_extxyz_many(selected_path, selected)
    selected_files = []
    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index, (frame_index, atoms) in enumerate(zip(selected_indices, selected)):
            frame_path = output_dir / f"selected_{index:06d}.xyz"
            write_structure(atoms, frame_path, fmt="extxyz")
            selected_files.append(frame_path)
            manifest.write(
                json.dumps(
                    {
                        "index": index,
                        "source_frame": frame_index,
                        "path": str(frame_path),
                    }
                )
                + "\n"
            )
    return StageResult(
        output_dir,
        (selected_path, *selected_files, manifest_path),
        f"Selected {len(selected)} of {len(frames)} MD frame(s)",
    )


def setup_labeling(config: PESMakerConfig) -> StageResult:
    """Prepare VASP SCF calculation folders."""
    output_dir = _section_output_dir(config, config.labeling.options, "labeling")
    output_dir.mkdir(parents=True, exist_ok=True)
    records = _load_input_records(config, config.labeling.options)
    incar = _read_optional_file(config.labeling.options.get("incar"), default=DEFAULT_INCAR)
    files: list[Path] = []
    manifest_path = output_dir / "labeling_manifest.jsonl"
    source_root = _labeling_source_root(config, config.labeling.options)
    used_workdirs: set[Path] = set()
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index, record in enumerate(records):
            source_path = Path(record["path"])
            calc_dir = _labeling_workdir(
                output_dir,
                source_path,
                index=index,
                naming=str(config.labeling.options.get("workdir_naming", "source_tree")),
                source_root=source_root,
                used_workdirs=used_workdirs,
            )
            calc_dir.mkdir(parents=True, exist_ok=True)
            poscar_path = calc_dir / "POSCAR"
            _write_labeling_poscar(source_path, poscar_path)
            backup_path = _copy_labeling_source_backup(
                config.labeling.options,
                source_path,
                calc_dir,
            )
            incar_path = calc_dir / "INCAR"
            incar_path.write_text(incar, encoding="utf-8")
            _copy_optional_templates(config.labeling.options, calc_dir)
            potcar_paths = _write_potcar_from_library(
                config.labeling.options,
                poscar_path,
                calc_dir,
            )
            submit_path = _write_submit_script(
                config,
                calc_dir,
                stage="labeling",
                command=str(config.labeling.options.get("command", "vasp_std")),
            )
            files.extend(
                path
                for path in (
                    poscar_path,
                    backup_path,
                    incar_path,
                    *potcar_paths,
                    submit_path,
                )
                if path is not None
            )
            manifest.write(
                json.dumps(
                    {
                        "index": index,
                        "engine": config.labeling.engine,
                        "source": str(source_path),
                        "workdir": str(calc_dir),
                    }
                )
                + "\n"
            )
    files.append(manifest_path)
    return StageResult(
        output_dir,
        tuple(files),
        f"Prepared {len(records)} single-point job(s)",
    )


def submit_jobs(
    config: PESMakerConfig,
    *,
    stage: str = "labeling",
    dry_run: bool = False,
) -> StageResult:
    """Submit prepared stage jobs with the configured scheduler command."""
    submit_scripts = _stage_submit_scripts(config, stage)
    if not submit_scripts:
        raise ValueError(f"no submit.sh scripts found for stage: {stage}")

    submit_command = str(config.jobs.options.get("submit_command", "sbatch"))
    output_dir = _stage_output_dir(config, stage)
    output_dir.mkdir(parents=True, exist_ok=True)
    submitted_log = output_dir / f"{stage}_submitted_jobs.txt"
    lines: list[str] = []
    for script in submit_scripts:
        command = [*shlex.split(submit_command), script.name]
        display = f"(cd {script.parent} && {' '.join(command)})"
        if dry_run:
            lines.append(f"DRY-RUN {display}")
            continue
        result = subprocess.run(
            command,
            cwd=script.parent,
            check=True,
            capture_output=True,
            text=True,
        )
        message = result.stdout.strip() or result.stderr.strip()
        lines.append(f"{script.parent}: {message}")
    submitted_log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    action = "Would submit" if dry_run else "Submitted"
    return StageResult(
        output_dir,
        (submitted_log,),
        f"{action} {len(submit_scripts)} {stage} job(s)",
    )


def collect_labeled_dataset(config: PESMakerConfig) -> StageResult:
    """Collect completed VASP SCF calculations into `train.xyz`."""
    output_dir = _section_output_dir(config, config.dataset.__dict__, "dataset")
    output_dir.mkdir(parents=True, exist_ok=True)
    default_pattern = Path("runs") / config.project / "labeling" / "**" / "OUTCAR"
    pattern = str(config.labeling.options.get("outcar_pattern", default_pattern))
    output_path = Path(
        str(config.labeling.options.get("dataset_path", output_dir / "train.xyz"))
    )
    outputs = [Path(path) for path in sorted(glob(pattern, recursive=True))]
    if not outputs:
        raise ValueError(f"no VASP outputs matched pattern: {pattern}")

    frames = []
    for output in outputs:
        frames.extend(_read_trajectory_frames(str(output)))
    _write_extxyz_many(output_path, frames)
    return StageResult(
        output_dir,
        (output_path,),
        f"Collected {len(frames)} labeled frame(s) into {output_path}",
    )


def setup_training(config: PESMakerConfig) -> StageResult:
    """Prepare potential training inputs and a submission script."""
    output_dir = _section_output_dir(config, config.training.options, "training")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(str(config.training.options.get("dataset", "train.xyz")))
    target_dataset = output_dir / dataset_path.name
    if dataset_path.exists():
        shutil.copy2(dataset_path, target_dataset)

    if config.training.engine.lower() == "nep":
        input_name = "nep.in"
        default_input = DEFAULT_NEP_IN
        command = str(config.training.options.get("command", "nep"))
    else:
        input_name = "train.in"
        default_input = "# Add trainer-specific options here.\n"
        command = str(config.training.options.get("command", config.training.engine))
    input_text = _read_optional_file(
        config.training.options.get("input"),
        default=default_input,
    )
    input_path = output_dir / input_name
    input_path.write_text(input_text, encoding="utf-8")
    submit_path = _write_submit_script(
        config,
        output_dir,
        stage="training",
        command=command,
    )
    return StageResult(
        output_dir,
        tuple(path for path in (target_dataset, input_path, submit_path) if path.exists()),
        f"Prepared training folder for {config.training.engine}",
    )


def _section_output_dir(
    config: PESMakerConfig,
    options: dict[str, Any],
    leaf: str,
) -> Path:
    value = options.get("output_dir")
    return Path(str(value)) if value else Path("runs") / config.project / leaf


def _load_input_records(
    config: PESMakerConfig,
    options: dict[str, Any],
) -> list[dict[str, str]]:
    manifest = options.get("input_manifest")
    if manifest:
        return _read_manifest(Path(str(manifest)))
    generation_dir = (
        config.generation.output_dir or Path("runs") / config.project / "generated"
    )
    manifest_path = generation_dir / "manifest.jsonl"
    if manifest_path.exists():
        return _read_manifest(manifest_path)
    paths = sorted(generation_dir.rglob("structure_*.*"))
    if not paths:
        raise ValueError(f"no generated structures found in {generation_dir}")
    return [{"path": str(path)} for path in paths]


def _labeling_source_root(config: PESMakerConfig, options: dict[str, Any]) -> Path:
    source_root = options.get("input_root")
    if source_root:
        return Path(str(source_root))
    manifest = options.get("input_manifest")
    if manifest:
        return Path(str(manifest)).parent
    return config.generation.output_dir or Path("runs") / config.project / "generated"


def _labeling_workdir(
    output_dir: Path,
    source_path: Path,
    *,
    index: int,
    naming: str,
    source_root: Path,
    used_workdirs: set[Path],
) -> Path:
    if naming == "indexed":
        candidate = output_dir / f"calc_{index:06d}"
    elif naming == "source_stem":
        candidate = output_dir / _safe_path_part(source_path.stem)
    elif naming == "source_tree":
        relative = _relative_source_path(source_path, source_root)
        candidate = output_dir / relative.with_suffix("")
    else:
        raise ValueError(
            "labeling.workdir_naming must be one of: indexed, source_stem, source_tree"
        )
    return _unique_workdir(candidate, used_workdirs)


def _relative_source_path(path: Path, root: Path) -> Path:
    for candidate, candidate_root in ((path, root), (path.resolve(), root.resolve())):
        try:
            return candidate.relative_to(candidate_root)
        except ValueError:
            continue
    return Path(_safe_path_part(path.with_suffix("").as_posix()))


def _unique_workdir(candidate: Path, used_workdirs: set[Path]) -> Path:
    base = candidate
    counter = 2
    while candidate in used_workdirs:
        candidate = base.with_name(f"{base.name}_{counter}")
        counter += 1
    used_workdirs.add(candidate)
    return candidate


def _safe_path_part(value: str) -> str:
    safe = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_" for char in value
    )
    return safe.strip("_") or "structure"


def _write_labeling_poscar(source_path: Path, poscar_path: Path) -> None:
    if source_path.suffix.lower() in {".vasp", ".poscar"}:
        shutil.copy2(source_path, poscar_path)
        return
    atoms = load_structure(source_path)
    write_structure(atoms, poscar_path, fmt="vasp")


def _copy_labeling_source_backup(
    options: dict[str, Any],
    source_path: Path,
    calc_dir: Path,
) -> Path | None:
    if not bool(options.get("backup_source", True)):
        return None
    suffix = str(options.get("backup_suffix", f"{source_path.suffix}-bak"))
    backup_name = f"{source_path.stem}{suffix}"
    backup_path = calc_dir / backup_name
    shutil.copy2(source_path, backup_path)
    return backup_path


def _write_potcar_from_library(
    options: dict[str, Any],
    poscar_path: Path,
    calc_dir: Path,
) -> tuple[Path, ...]:
    """Build POTCAR from a VASP potential library when configured."""
    if options.get("potcar"):
        return ()
    library = _potcar_library_path(options)
    if library is None:
        return ()

    symbols = _ordered_structure_symbols(poscar_path)
    if not symbols:
        raise ValueError(f"no elements found in {poscar_path}")

    mapping = options.get("potcar_map", {})
    if mapping is None:
        mapping = {}
    if not isinstance(mapping, dict):
        raise ValueError("labeling.potcar_map must be a mapping")

    use_gw = bool(options.get("gw_potcar", options.get("potcar_gw", False)))
    chunks: list[bytes] = []
    chosen: list[str] = []
    for symbol in symbols:
        directory_name = _potcar_directory_name(
            symbol,
            mapping=mapping,
            use_gw=use_gw,
        )
        potcar_file = _find_potcar_file(library / directory_name)
        if potcar_file is None:
            raise ValueError(
                f"missing POTCAR for {symbol}: expected {library / directory_name / 'POTCAR'}"
            )
        chunks.append(_read_potcar_bytes(potcar_file))
        chosen.append(directory_name)

    potcar_path = calc_dir / "POTCAR"
    potcar_path.write_bytes(_join_potcar_chunks(chunks))
    spec_path = calc_dir / "POTCAR.spec"
    spec_path.write_text("\n".join(chosen) + "\n", encoding="utf-8")
    return (potcar_path, spec_path)


def _potcar_library_path(options: dict[str, Any]) -> Path | None:
    for key in ("potcar_library", "pbe_path", "potcar_path"):
        value = options.get(key)
        if value:
            return Path(str(value))
    return None


def _ordered_structure_symbols(poscar_path: Path) -> list[str]:
    atoms = load_structure(poscar_path)
    symbols: list[str] = []
    for symbol in atoms.get_chemical_symbols():
        if symbol not in symbols:
            symbols.append(symbol)
    return symbols


def _potcar_directory_name(
    symbol: str,
    *,
    mapping: dict[str, Any],
    use_gw: bool,
) -> str:
    mapped = mapping.get(symbol)
    if mapped:
        return str(mapped)
    if use_gw:
        return RECOMMENDED_GW_POTCARS.get(symbol, f"{symbol}_GW")
    return RECOMMENDED_PBE_POTCARS.get(symbol, symbol)


def _find_potcar_file(directory: Path) -> Path | None:
    for name in ("POTCAR", "POTCAR.gz"):
        candidate = directory / name
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _read_potcar_bytes(path: Path) -> bytes:
    if path.suffix == ".gz":
        return gzip.decompress(path.read_bytes())
    return path.read_bytes()


def _join_potcar_chunks(chunks: list[bytes]) -> bytes:
    output = bytearray()
    for chunk in chunks:
        output.extend(chunk)
        if chunk and not chunk.endswith(b"\n"):
            output.extend(b"\n")
    return bytes(output)


def _stage_submit_scripts(config: PESMakerConfig, stage: str) -> list[Path]:
    manifest_name = f"{stage}_manifest.jsonl"
    output_dir = _stage_output_dir(config, stage)
    manifest_path = output_dir / manifest_name
    if manifest_path.exists():
        scripts = []
        for record in _read_manifest(manifest_path):
            workdir = record.get("workdir")
            if workdir:
                script = Path(str(workdir)) / "submit.sh"
                if script.exists():
                    scripts.append(script)
        if scripts:
            return scripts
    return sorted(output_dir.rglob("submit.sh"))


def _stage_output_dir(config: PESMakerConfig, stage: str) -> Path:
    if stage == "sampling":
        return _section_output_dir(config, config.sampling.options, "sampling")
    if stage == "labeling":
        return _section_output_dir(config, config.labeling.options, "labeling")
    if stage == "training":
        return _section_output_dir(config, config.training.options, "training")
    raise ValueError("stage must be one of: sampling, labeling, training")


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if "path" in record:
                record = {"path": str(record["path"]), **record}
            records.append(record)
    return records


def _read_optional_file(value: Any, *, default: str) -> str:
    if value:
        return Path(str(value)).read_text(encoding="utf-8")
    return default


def _sampling_conditions(options: dict[str, Any]) -> tuple[SamplingCondition, ...]:
    temperatures = options.get("temperatures")
    if temperatures is not None:
        if isinstance(temperatures, str) and "-" in temperatures:
            start, end = _parse_temperature_range(temperatures)
            return (_ramp_temperature(start, end),)
        if not isinstance(temperatures, list) or not temperatures:
            raise ValueError("sampling.temperatures must be a non-empty list")
        return tuple(_constant_temperature(float(value)) for value in temperatures)

    temperature = options.get("temperature", options.get("temp", 300))
    if isinstance(temperature, str) and "-" in temperature:
        start, end = _parse_temperature_range(temperature)
        return (_ramp_temperature(start, end),)
    if isinstance(temperature, dict):
        start = float(temperature.get("start", temperature.get("from", 300)))
        end = float(temperature.get("end", temperature.get("to", start)))
        return (_ramp_temperature(start, end) if start != end else _constant_temperature(start),)
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
    return str(int(value)) if float(value).is_integer() else str(value).replace(".", "p")


def _render_sampling_run_in(
    options: dict[str, Any],
    template: str,
    condition: SamplingCondition,
    *,
    potential_name: str | None = None,
) -> str:
    potential = potential_name or str(options.get("potential", "nep89_20250409.txt"))
    rendered = template.format(
        potential=potential,
        temperature=_format_temperature(condition.start),
        temperature_start=_format_temperature(condition.start),
        temperature_end=_format_temperature(condition.end),
    )
    rendered = _rewrite_gpumd_run_line(rendered, "potential", f"potential      {potential}")
    rendered = _rewrite_gpumd_run_line(
        rendered,
        "velocity",
        f"velocity       {_format_temperature(condition.start)}",
    )
    return _rewrite_gpumd_ensemble_temperature(rendered, condition)


def _prepare_sampling_potential(
    options: dict[str, Any],
    stage_dir: Path,
) -> tuple[str, Path | None]:
    potential = Path(str(options.get("potential", "nep89_20250409.txt")))
    if potential.exists() and potential.is_file():
        target = stage_dir / potential.name
        shutil.copy2(potential, target)
        return potential.name, target
    return str(options.get("potential", "nep89_20250409.txt")), None


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


def _rewrite_gpumd_ensemble_temperature(
    text: str,
    condition: SamplingCondition,
) -> str:
    output = []
    start = _format_temperature(condition.start)
    end = _format_temperature(condition.end)
    for line in text.splitlines():
        tokens = line.split()
        if len(tokens) >= 4 and tokens[0] == "ensemble":
            tokens[2] = start
            tokens[3] = end
            output.append(_format_gpumd_tokens(tokens))
        else:
            output.append(line)
    return "\n".join(output) + "\n"


def _format_gpumd_tokens(tokens: list[str]) -> str:
    if len(tokens) < 2:
        return " ".join(tokens)
    return f"{tokens[0]:<14} {tokens[1]} " + " ".join(tokens[2:])


def _format_temperature(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def _sampling_command(config: PESMakerConfig) -> str:
    if config.sampling.engine.lower() == "gpumd":
        command = config.sampling.options.get("command")
        if command:
            return str(command)
        gpumd_dir = config.sampling.options.get("gpumd_dir")
        if gpumd_dir:
            return str(Path(str(gpumd_dir)) / "gpumd")
        return "gpumd"
    return str(config.sampling.options.get("command", config.sampling.engine))


def _write_submit_script(
    config: PESMakerConfig,
    workdir: Path,
    *,
    stage: str,
    command: str,
) -> Path:
    template_path = _job_template_path(config, stage)
    job_name = f"{config.project}-{stage}"
    if template_path:
        text = template_path.read_text(encoding="utf-8").format(
            command=command,
            job_name=job_name,
            workdir=workdir,
        )
    else:
        text = _default_submit_script(command=command, job_name=job_name)
    path = workdir / "submit.sh"
    path.write_text(text, encoding="utf-8")
    return path


def _job_template_path(config: PESMakerConfig, stage: str) -> Path | None:
    templates = config.jobs.options.get("sbatch_templates", {})
    if isinstance(templates, dict) and templates.get(stage):
        return Path(str(templates[stage]))
    template = config.jobs.options.get("sbatch_template")
    return Path(str(template)) if template else None


def _default_submit_script(*, command: str, job_name: str) -> str:
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

set -euo pipefail
cd "$(dirname "$0")"
{command}
"""


def _copy_optional_templates(options: dict[str, Any], calc_dir: Path) -> None:
    for key in ("potcar", "kpoints", "template_dir"):
        value = options.get(key)
        if not value:
            continue
        source = Path(str(value))
        if source.is_dir():
            for item in source.iterdir():
                if item.is_file():
                    shutil.copy2(item, calc_dir / item.name)
        elif source.exists():
            shutil.copy2(source, calc_dir / source.name.upper())


def _read_trajectory_frames(pattern: str):
    try:
        from ase.io import read
    except ImportError as exc:
        raise RuntimeError("Trajectory selection requires ASE") from exc

    paths = [Path(path) for path in sorted(glob(pattern, recursive=True))]
    if not paths and Path(pattern).exists():
        paths = [Path(pattern)]
    frames = []
    for path in paths:
        items = read(path, index=":")
        if not isinstance(items, list):
            items = [items]
        frames.extend(items)
    if not frames:
        raise ValueError(f"no trajectory frames matched pattern: {pattern}")
    return frames


def _write_extxyz_many(path: Path, frames) -> None:
    try:
        from ase.io import write
    except ImportError as exc:
        raise RuntimeError("Writing extxyz requires ASE") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    write(path, frames, format="extxyz")


def _structure_features(frames) -> np.ndarray:
    raw = []
    max_length = 0
    for atoms in frames:
        numbers = atoms.get_atomic_numbers().reshape(-1, 1)
        scaled = atoms.get_scaled_positions(wrap=True)
        feature = np.concatenate([numbers, scaled], axis=1).reshape(-1)
        feature = np.concatenate([np.array([len(atoms)]), feature])
        raw.append(feature)
        max_length = max(max_length, len(feature))
    features = np.zeros((len(raw), max_length), dtype=float)
    for index, feature in enumerate(raw):
        features[index, : len(feature)] = feature
    return features


def _farthest_point_indices(
    features: np.ndarray,
    *,
    min_distance: float,
    max_count: int | None,
) -> list[int]:
    selected = [0]
    distances = np.linalg.norm(features - features[0], axis=1)
    while True:
        next_index = int(np.argmax(distances))
        next_distance = float(distances[next_index])
        if next_index in selected or next_distance < min_distance:
            break
        if max_count is not None and len(selected) >= max_count:
            break
        selected.append(next_index)
        new_distances = np.linalg.norm(features - features[next_index], axis=1)
        distances = np.minimum(distances, new_distances)
    return selected
