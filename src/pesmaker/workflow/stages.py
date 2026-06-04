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
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np

from pesmaker.config.schema import PESMakerConfig
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

DEFAULT_INCAR = """SYSTEM = PESMaker single point
GGA = PE         # Use the PBE functional for exchange-correlation
LREAL = Auto     # Projection operators: automatic
ENCUT = 650      # Cut-off energy for plane-wave basis set, in eV
KSPACING = 0.2   # Automatically calculate k-points
KGAMMA = .TRUE.  # Include the Gamma point
NSW = 0          # Static SCF; no ionic steps
IBRION = -1      # Ions are not moved
ALGO = Normal    # Standard electronic minimization algorithm
EDIFF = 1E-06    # SCF energy convergence in eV
SIGMA = 0.02     # Smearing width in eV
ISMEAR = 0       # Gaussian smearing
PREC = Accurate  # High precision level
NELM = 150       # Maximum electronic SCF steps
"""

STRUCTURE_INPUT_SUFFIXES = {".cif", ".extxyz", ".poscar", ".vasp", ".xyz"}
STRUCTURE_INPUT_NAMES = {"CONTCAR", "POSCAR"}
LARGE_SCF_ATOM_WARNING_THRESHOLD = 250


@dataclass(frozen=True)
class JobResources:
    """Scheduler resource settings shared by generated submit scripts."""

    nodes: int
    cores_cpu: int
    gpus: int
    vasp_kpar: int
    vasp_ncore: int


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
    warnings: tuple[str, ...] = field(default_factory=tuple)


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
                    atoms,
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
    features, descriptor_backend = _selection_features(
        frames,
        options,
        sampling_options=config.sampling.options,
    )
    selected_indices, selection_distances = _farthest_point_indices(
        features,
        min_distance=min_distance,
        max_count=max_count,
    )
    selected = [frames[index] for index in selected_indices]

    output_dir.mkdir(parents=True, exist_ok=True)
    features_path = output_dir / "selection_features.npy"
    np.save(features_path, features)
    plot_path = _write_selection_plot(
        features,
        selected_indices,
        selection_distances,
        output_dir=output_dir,
        options=options,
    )
    selected_path = output_dir / "selected.xyz"
    _write_extxyz_many(selected_path, selected)
    selected_files = []
    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index, (frame_index, atoms, distance) in enumerate(
            zip(selected_indices, selected, selection_distances)
        ):
            frame_path = output_dir / f"selected_{index:06d}.xyz"
            write_structure(atoms, frame_path, fmt="extxyz")
            selected_files.append(frame_path)
            manifest.write(
                json.dumps(
                    {
                        "index": index,
                        "source_frame": frame_index,
                        "path": str(frame_path),
                        "descriptor": descriptor_backend,
                        "selection_distance": distance,
                    }
                )
                + "\n"
            )
    files = [selected_path, features_path, *selected_files, manifest_path]
    if plot_path is not None:
        files.append(plot_path)
    return StageResult(
        output_dir,
        tuple(files),
        f"Selected {len(selected)} of {len(frames)} MD frame(s)",
    )


def setup_labeling(config: PESMakerConfig) -> StageResult:
    """Prepare VASP SCF calculation folders."""
    output_dir = _section_output_dir(config, config.labeling.options, "labeling")
    output_dir.mkdir(parents=True, exist_ok=True)
    records = _load_input_records(config, config.labeling.options)
    incar_template = _read_optional_file(
        config.labeling.options.get("incar"),
        default=DEFAULT_INCAR,
    )
    files: list[Path] = []
    warnings: list[str] = []
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
            warning = _write_labeling_poscar(source_path, poscar_path)
            if warning:
                warnings.append(warning)
            atom_count = _labeling_atom_count(record, poscar_path)
            warning = _large_scf_atom_count_warning(
                source_path,
                poscar_path,
                atom_count,
            )
            if warning:
                warnings.append(warning)
            resources = _job_resources(config, atom_count=atom_count)
            backup_path = _copy_labeling_source_backup(
                config.labeling.options,
                source_path,
                calc_dir,
            )
            incar_path = calc_dir / "INCAR"
            incar = _prepare_labeling_incar(incar_template, resources)
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
                resources=resources,
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
            record_data = {
                "index": index,
                "engine": config.labeling.engine,
                "source": str(source_path),
                "workdir": str(calc_dir),
                "cores_cpu": resources.cores_cpu,
                "gpus": resources.gpus,
            }
            if not resources.gpus:
                record_data["vasp_kpar"] = resources.vasp_kpar
                record_data["vasp_ncore"] = resources.vasp_ncore
            for key in (
                "input_dir",
                "input_mode",
                "input_relative_path",
                "source_record_index",
            ):
                if key in record:
                    record_data[key] = record[key]
            manifest.write(json.dumps(record_data) + "\n")
    files.append(manifest_path)
    return StageResult(
        output_dir,
        tuple(files),
        f"Prepared {len(records)} SCF job(s)",
        warnings=tuple(warnings),
    )


def submit_jobs(
    config: PESMakerConfig,
    *,
    stage: str = "scf",
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
) -> list[dict[str, Any]]:
    manifest = options.get("input_manifest")
    if manifest:
        manifest_path = Path(str(manifest))
        return _mark_input_records(
            _read_manifest(manifest_path),
            input_dir=manifest_path.parent,
            input_mode="input_manifest",
        )
    input_dir = options.get("input_dir")
    if input_dir:
        return _load_input_dir_records(Path(str(input_dir)), input_mode="input_dir")
    generation_dir = _generated_structures_dir(config)
    return _load_input_dir_records(generation_dir, input_mode="generated_dir")


def _load_input_dir_records(input_dir: Path, *, input_mode: str) -> list[dict[str, Any]]:
    if not input_dir.exists():
        raise ValueError(f"input structure directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise ValueError(f"input structure path must be a directory: {input_dir}")

    manifest_path = input_dir / "manifest.jsonl"
    if manifest_path.exists():
        return _mark_input_records(
            _read_manifest(manifest_path),
            input_dir=input_dir,
            input_mode=f"{input_mode}_manifest",
        )

    paths = _discover_input_structure_files(input_dir)
    if not paths:
        raise ValueError(f"no structure files found in {input_dir}")
    return [
        {
            "path": str(path),
            "input_dir": str(input_dir),
            "input_mode": f"{input_mode}_scan",
            "input_relative_path": path.relative_to(input_dir).as_posix(),
        }
        for path in paths
    ]


def _mark_input_records(
    records: list[dict[str, Any]],
    *,
    input_dir: Path,
    input_mode: str,
) -> list[dict[str, Any]]:
    marked = []
    for source_index, record in enumerate(records):
        path = Path(str(record["path"]))
        if not path.is_absolute() and not path.exists():
            candidate = input_dir / path
            if candidate.exists():
                path = candidate
        marked_record = {
            **record,
            "path": str(path),
            "input_dir": str(input_dir),
            "input_mode": input_mode,
            "source_record_index": record.get("index", source_index),
        }
        try:
            marked_record["input_relative_path"] = path.relative_to(
                input_dir
            ).as_posix()
        except ValueError:
            pass
        marked.append(marked_record)
    return marked


def _discover_input_structure_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and _is_input_structure_file(path)
    )


def _is_input_structure_file(path: Path) -> bool:
    if path.name.upper() in STRUCTURE_INPUT_NAMES:
        return True
    return path.suffix.lower() in STRUCTURE_INPUT_SUFFIXES


def _labeling_source_root(config: PESMakerConfig, options: dict[str, Any]) -> Path:
    source_root = options.get("input_root")
    if source_root:
        return Path(str(source_root))
    manifest = options.get("input_manifest")
    if manifest:
        return Path(str(manifest)).parent
    input_dir = options.get("input_dir")
    if input_dir:
        return Path(str(input_dir))
    return _generated_structures_dir(config)


def _generated_structures_dir(config: PESMakerConfig) -> Path:
    if config.generation.output_dir:
        return config.generation.output_dir
    local_generated = Path("generated")
    if local_generated.exists():
        return local_generated
    return Path("runs") / config.project / "generated"


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


def _write_labeling_poscar(source_path: Path, poscar_path: Path) -> str | None:
    needs_normalization = _vasp_species_block_has_repeated_symbols(source_path)
    atoms = load_structure(source_path)
    write_structure(atoms, poscar_path, fmt="vasp")
    if needs_normalization:
        return (
            "Normalized non-compact VASP species block: "
            f"{source_path} -> {poscar_path}. Inspect POSCAR before submission."
        )
    return None


def _vasp_species_block_has_repeated_symbols(path: Path) -> bool:
    if path.suffix.lower() not in {".vasp", ".poscar"}:
        return False
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        lines = path.read_text().splitlines()
    if len(lines) < 7:
        return False
    symbols = lines[5].split()
    counts = lines[6].split()
    if not symbols or not counts or len(symbols) != len(counts):
        return False
    if not all(_looks_like_element_symbol(symbol) for symbol in symbols):
        return False
    return len(set(symbols)) != len(symbols)


def _looks_like_element_symbol(value: str) -> bool:
    return len(value) <= 2 and value[0].isalpha() and value[0].isupper()


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


def _labeling_atom_count(record: dict[str, Any], poscar_path: Path) -> int | None:
    atom_count = record.get("atom_count")
    if atom_count is not None:
        return int(atom_count)
    try:
        return len(load_structure(poscar_path))
    except Exception:
        return None


def _large_scf_atom_count_warning(
    source_path: Path,
    poscar_path: Path,
    atom_count: int | None,
) -> str | None:
    if atom_count is None or atom_count <= LARGE_SCF_ATOM_WARNING_THRESHOLD:
        return None
    return (
        f"Large SCF job: {source_path} -> {poscar_path} has {atom_count} atoms; "
        "single-point calculation may be expensive."
    )


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
    manifest_name = _stage_manifest_name(stage)
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
    if stage == "scf":
        return _section_output_dir(config, config.labeling.options, "labeling")
    if stage == "training":
        return _section_output_dir(config, config.training.options, "training")
    raise ValueError("stage must be one of: sampling, scf, training")


def _stage_manifest_name(stage: str) -> str:
    if stage == "scf":
        return "labeling_manifest.jsonl"
    return f"{stage}_manifest.jsonl"


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
    atoms,
    *,
    potential_name: str | None = None,
) -> str:
    potential = potential_name or str(options.get("potential", "nep89_20250409.txt"))
    ensemble = _sampling_ensemble_line(options, condition, atoms)
    run_steps = _sampling_run_steps(options)
    rendered = template.format(
        potential=potential,
        ensemble=ensemble,
        run_steps=run_steps,
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
    rendered = _rewrite_gpumd_run_line(rendered, "ensemble", ensemble)
    return _rewrite_gpumd_run_line(rendered, "run", f"run            {run_steps}")


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
        raise ValueError("sampling pressure and elastic options must be scalars or lists")
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


def _write_submit_script(
    config: PESMakerConfig,
    workdir: Path,
    *,
    stage: str,
    command: str,
    resources: JobResources | None = None,
) -> Path:
    template_path = _job_template_path(config, stage)
    job_name = workdir.name
    resources = resources or _job_resources(config)
    if template_path:
        ntasks = resources.nodes * resources.cores_cpu
        text = _format_submit_template(
            template_path.read_text(encoding="utf-8"),
            {
                "command": command,
                "job_name": job_name,
                "workdir": str(workdir),
                "nodes": resources.nodes,
                "ntasks": ntasks,
                "cores_cpu": resources.cores_cpu,
                "ntasks_per_node": resources.cores_cpu,
                "gpus": resources.gpus,
                "vasp_kpar": resources.vasp_kpar,
                "vasp_ncore": resources.vasp_ncore,
            },
        )
        text = _normalize_submit_template(
            text,
            command=command,
            job_name=job_name,
            workdir=workdir,
            stage=stage,
            engine=_stage_engine(config, stage),
            resources=resources,
        )
    else:
        text = _default_submit_script(
            command=command,
            job_name=job_name,
            stage=stage,
            engine=_stage_engine(config, stage),
            resources=resources,
        )
    path = workdir / "submit.sh"
    path.write_text(text, encoding="utf-8")
    return path


def _job_template_path(config: PESMakerConfig, stage: str) -> Path | None:
    sub_file = config.jobs.options.get("sub_file")
    if isinstance(sub_file, dict):
        stage_template = _stage_template_value(sub_file, stage)
        if stage_template:
            return Path(str(stage_template))
    if sub_file and not isinstance(sub_file, dict):
        return Path(str(sub_file))

    templates = config.jobs.options.get("sbatch_templates", {})
    if isinstance(templates, dict):
        stage_template = _stage_template_value(templates, stage)
        if stage_template:
            return Path(str(stage_template))
    template = config.jobs.options.get("sbatch_template")
    return Path(str(template)) if template else None


def _stage_template_value(templates: dict[str, Any], stage: str) -> Any:
    if templates.get(stage):
        return templates[stage]
    if stage == "labeling" and templates.get("scf"):
        return templates["scf"]
    return None


def _stage_engine(config: PESMakerConfig, stage: str) -> str:
    if stage == "sampling":
        return config.sampling.engine
    if stage == "labeling":
        return config.labeling.engine
    if stage == "training":
        return config.training.engine
    return stage


def _format_submit_template(text: str, values: dict[str, object]) -> str:
    for key, value in values.items():
        text = text.replace(f"{{{key}}}", str(value))
    return text


def _normalize_submit_template(
    text: str,
    *,
    command: str,
    job_name: str,
    workdir: Path,
    stage: str,
    engine: str,
    resources: JobResources,
) -> str:
    ntasks = resources.nodes * resources.cores_cpu
    lines: list[str] = []
    for line in text.splitlines():
        if _is_generated_workdir_cd(line, workdir):
            continue
        updated = _set_sbatch_directive(line, "--job-name", job_name)
        if updated is None:
            updated = _set_sbatch_directive(line, "--ntasks", str(ntasks))
        if updated is None:
            updated = _set_sbatch_directive(
                line,
                "--ntasks-per-node",
                str(resources.cores_cpu),
            )
        if updated is None:
            updated = _replace_vasp_run_command(
                line,
                command=command,
                stage=stage,
                engine=engine,
                resources=resources,
            )
        lines.append(updated if updated is not None else line)
    return "\n".join(lines) + "\n"


def _set_sbatch_directive(line: str, option: str, value: str) -> str | None:
    prefix = line[: len(line) - len(line.lstrip())]
    stripped = line.lstrip()
    if not stripped.startswith("#SBATCH"):
        return None
    rest = stripped[len("#SBATCH") :].lstrip()
    if rest.startswith(f"{option}="):
        suffix = _directive_suffix(rest[len(option) + 1 :])
        return f"{prefix}#SBATCH {option}={value}{suffix}"
    if rest == option or rest.startswith(f"{option} "):
        suffix = _directive_suffix(rest[len(option) :].lstrip())
        return f"{prefix}#SBATCH {option}={value}{suffix}"
    return None


def _directive_suffix(value_text: str) -> str:
    comment_index = value_text.find(" #")
    if comment_index >= 0:
        return value_text[comment_index:]
    return ""


def _is_generated_workdir_cd(line: str, workdir: Path) -> bool:
    stripped = line.strip()
    workdir_text = str(workdir)
    return stripped in {
        f'cd "{workdir_text}"',
        f"cd '{workdir_text}'",
        f"cd {workdir_text}",
    }


def _replace_vasp_run_command(
    line: str,
    *,
    command: str,
    stage: str,
    engine: str,
    resources: JobResources,
) -> str | None:
    if stage != "labeling" or engine.lower() != "vasp":
        return None
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    lower = stripped.lower()
    if lower.startswith(("echo ", "export ", "module ", "source ", "ulimit ", "set ")):
        return None
    if "vasp" not in lower:
        return None
    prefix = line[: len(line) - len(line.lstrip())]
    return prefix + _default_run_command(
        command,
        stage=stage,
        engine=engine,
        resources=resources,
    )


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


def _default_submit_script(
    *,
    command: str,
    job_name: str,
    stage: str,
    engine: str,
    resources: JobResources,
) -> str:
    ntasks = resources.nodes * resources.cores_cpu
    lines = [
        "#!/bin/bash -l",
        f"#SBATCH --job-name={job_name}",
        "#SBATCH --output=out.%j",
        "#SBATCH --error=err.%j",
        f"#SBATCH --nodes={resources.nodes}",
        f"#SBATCH --ntasks={ntasks}",
        "#SBATCH --cpus-per-task=1",
    ]
    if resources.gpus:
        lines.append(f"#SBATCH --gres=gpu:{resources.gpus}")
    lines.extend(
        [
            "",
            "set -euo pipefail",
            "",
            "export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}",
            "ulimit -s unlimited",
            "",
            'echo "--------------------------------"',
            'echo "Job started at $(date)"',
            'echo "Running on node(s): ${SLURM_NODELIST:-unknown}"',
            'echo "Using total tasks: ${SLURM_NTASKS:-unknown}"',
            'echo "Working directory: $(pwd)"',
            'echo "--------------------------------"',
            "",
            _default_run_command(command, stage=stage, engine=engine, resources=resources),
            "",
            'echo "Simulation finished at $(date)"',
            "",
        ]
    )
    return "\n".join(lines)


def _default_run_command(
    command: str,
    *,
    stage: str,
    engine: str,
    resources: JobResources,
) -> str:
    if stage == "labeling" and engine.lower() == "vasp" and not resources.gpus:
        return f"mpirun {command}"
    return command


def _prepare_labeling_incar(text: str, resources: JobResources) -> str:
    if resources.gpus:
        return _ensure_trailing_newline(text)
    text = _set_incar_value(
        text,
        "KPAR",
        str(resources.vasp_kpar),
        "K-point parallel groups",
    )
    text = _set_incar_value(
        text,
        "NCORE",
        str(resources.vasp_ncore),
        "MPI ranks per band group",
    )
    return _ensure_trailing_newline(text)


def _set_incar_value(text: str, key: str, value: str, comment: str) -> str:
    line = f"{key} = {value}       # {comment}"
    lines = text.splitlines()
    for index, existing in enumerate(lines):
        if existing.strip().upper().startswith(f"{key.upper()}"):
            lines[index] = line
            return "\n".join(lines)
    lines.append(line)
    return "\n".join(lines)


def _ensure_trailing_newline(text: str) -> str:
    return text if text.endswith("\n") else f"{text}\n"


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


def _selection_features(
    frames,
    options: dict[str, Any],
    *,
    sampling_options: dict[str, Any],
) -> tuple[np.ndarray, str]:
    descriptor = str(options.get("descriptor", "calorine")).lower()
    if descriptor in {"calorine", "nep", "calorine-nep", "calorine_nep"}:
        selection_potential = options.get("potential", options.get("model"))
        potential_options = dict(sampling_options)
        if selection_potential:
            potential_options["potential"] = selection_potential
        potential = _resolve_sampling_potential_path(potential_options)
        if potential is not None and potential.exists():
            potential = str(potential.resolve())
        features = _calorine_nep_structure_features(frames, potential, options)
        return features, "calorine"
    if descriptor in {"simple", "geometry"}:
        return _structure_features(frames), "simple"
    raise ValueError("sampling.selection.descriptor must be 'calorine' or 'simple'")


def _calorine_nep_structure_features(
    frames,
    potential: Any,
    options: dict[str, Any],
) -> np.ndarray:
    if not potential:
        raise ValueError(
            "sampling.selection.potential or sampling.potential is required "
            "for Calorine NEP descriptor selection"
        )
    try:
        from calorine.nep import get_descriptors
    except ImportError as exc:
        raise RuntimeError(
            "Calorine NEP descriptor selection requires calorine. Install it "
            'with `python -m pip install ".[selection]"` or `python -m pip '
            "install calorine`."
        ) from exc

    potential_path = Path(str(potential))
    if not potential_path.exists():
        raise ValueError(f"Calorine NEP potential file does not exist: {potential_path}")
    pooling = str(options.get("descriptor_pooling", options.get("pooling", "mean")))
    features = []
    for atoms in frames:
        descriptors = np.asarray(
            get_descriptors(atoms, model_filename=str(potential_path)),
            dtype=float,
        )
        if descriptors.ndim == 1:
            descriptors = descriptors.reshape(1, -1)
        features.append(_pool_atom_descriptors(descriptors, pooling))
    return np.vstack(features)


def _pool_atom_descriptors(descriptors: np.ndarray, pooling: str) -> np.ndarray:
    if pooling == "mean":
        return descriptors.mean(axis=0)
    if pooling == "sum":
        return descriptors.sum(axis=0)
    if pooling in {"mean_std", "mean+std"}:
        return np.concatenate([descriptors.mean(axis=0), descriptors.std(axis=0)])
    raise ValueError(
        "sampling.selection.descriptor_pooling must be mean, sum, or mean_std"
    )


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
) -> tuple[list[int], list[float]]:
    if len(features) == 0:
        return [], []
    if max_count is not None and max_count < 1:
        return [], []

    selected = [0]
    selected_distances = [0.0]
    distances = np.linalg.norm(features - features[0], axis=1)
    while True:
        if max_count is not None and len(selected) >= max_count:
            break
        next_index = int(np.argmax(distances))
        next_distance = float(distances[next_index])
        if next_index in selected or next_distance < min_distance:
            break
        selected.append(next_index)
        selected_distances.append(next_distance)
        new_distances = np.linalg.norm(features - features[next_index], axis=1)
        distances = np.minimum(distances, new_distances)
        distances[selected] = 0.0
    return selected, selected_distances


def _write_selection_plot(
    features: np.ndarray,
    selected_indices: list[int],
    selection_distances: list[float],
    *,
    output_dir: Path,
    options: dict[str, Any],
) -> Path | None:
    if not bool(options.get("plot", True)):
        return None
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Selection plotting requires matplotlib. Install it with "
            "`python -m pip install matplotlib` or set sampling.selection.plot: false."
        ) from exc

    points = _pca_2d(features)
    plot_path = Path(str(options.get("plot_path", output_dir / "fps_selection.png")))
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_points, ax_distances) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax_points.scatter(points[:, 0], points[:, 1], s=18, c="#9aa0a6", label="all frames")
    if selected_indices:
        selected_points = points[selected_indices]
        ax_points.scatter(
            selected_points[:, 0],
            selected_points[:, 1],
            s=42,
            c="#d62728",
            label="selected",
        )
        for order, (x_value, y_value) in enumerate(selected_points[:50]):
            ax_points.annotate(str(order), (x_value, y_value), fontsize=7)
    ax_points.set_title("FPS selection in descriptor PCA space")
    ax_points.set_xlabel("PC1")
    ax_points.set_ylabel("PC2")
    ax_points.legend(frameon=False)

    ax_distances.plot(range(len(selection_distances)), selection_distances, marker="o")
    ax_distances.set_title("Distance when selected")
    ax_distances.set_xlabel("Selection order")
    ax_distances.set_ylabel("Nearest-selected distance")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return plot_path


def _pca_2d(features: np.ndarray) -> np.ndarray:
    if len(features) == 0:
        return np.zeros((0, 2), dtype=float)
    centered = features - np.mean(features, axis=0)
    if centered.shape[0] == 1:
        return np.zeros((1, 2), dtype=float)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    points = centered @ vh[: min(2, len(vh))].T
    if points.shape[1] == 1:
        points = np.column_stack([points[:, 0], np.zeros(len(points))])
    return points[:, :2]
