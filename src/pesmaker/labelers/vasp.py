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

"""VASP SCF labeling setup, input writing, and POTCAR assembly."""

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path
from typing import Any

from pesmaker.artifacts import (
    _generated_structures_dir,
    _load_input_records,
    _read_optional_file,
    _section_output_dir,
)
from pesmaker.config.schema import PESMakerConfig
from pesmaker.jobs.resources import JobResources, _job_resources
from pesmaker.jobs.scripts import _write_submit_script
from pesmaker.results import StageResult
from pesmaker.structures import load_structure, write_structure

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

LARGE_SCF_ATOM_WARNING_THRESHOLD = 250

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
            source_atoms = _load_labeling_source(record, source_path)
            naming_source_path = _labeling_naming_source_path(record, source_path)
            calc_dir = _labeling_workdir(
                output_dir,
                naming_source_path,
                index=index,
                naming=str(
                    config.labeling.options.get("workdir_naming", "source_tree")
                ),
                source_root=source_root,
                used_workdirs=used_workdirs,
            )
            calc_dir.mkdir(parents=True, exist_ok=True)
            poscar_path = calc_dir / "POSCAR"
            warning = _write_labeling_poscar(source_path, poscar_path, source_atoms)
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
                source_atoms=source_atoms,
                frame_index=record.get("frame_index"),
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
                "frame_index",
                "source_frame",
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


def _load_labeling_source(record: dict[str, Any], source_path: Path):
    frame_index = record.get("frame_index")
    if frame_index is not None:
        return load_structure(source_path, index=int(frame_index))
    return load_structure(source_path)


def _labeling_naming_source_path(record: dict[str, Any], source_path: Path) -> Path:
    frame_index = record.get("frame_index")
    if frame_index is None:
        return source_path
    return source_path.with_name(f"{source_path.stem}_{int(frame_index):06d}.xyz")


def _write_labeling_poscar(source_path: Path, poscar_path: Path, atoms) -> str | None:
    needs_normalization = _vasp_species_block_has_repeated_symbols(source_path)
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
    *,
    source_atoms=None,
    frame_index: Any = None,
) -> Path | None:
    if not bool(options.get("backup_source", True)):
        return None
    suffix = str(options.get("backup_suffix", f"{source_path.suffix}-bak"))
    if frame_index is not None:
        backup_name = f"{source_path.stem}_{int(frame_index):06d}{suffix}"
        backup_path = calc_dir / backup_name
        write_structure(source_atoms, backup_path, fmt="extxyz")
        return backup_path
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
