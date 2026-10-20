#!/usr/bin/env python3
"""Create a compact NEP training set from one fixed-cell VASP AIMD run.

The toolkit deliberately stays outside PESMaker's workflow implementation.  It
uses the public ``python -m pesmaker select`` command for FPS on XDATCAR, then
streams ionic frames from OUTCAR and writes labels only for the selected steps.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable, TextIO

import numpy as np


OUTPUT_DIR_NAME = "PESMakerToolkit_AIMD_FPS_to_NEP"
CELL_TOLERANCE_ANGSTROM = 1.0e-4
POSITION_TOLERANCE_ANGSTROM = 1.0e-4
REQUIRED_FPS_FILES = (
    "selected.xyz",
    "manifest.jsonl",
    "selection_features.npy",
    "fps_selection.png",
)


class ToolkitError(RuntimeError):
    """A user-facing toolkit failure."""


@dataclass(frozen=True)
class SelectionEntry:
    """One FPS-selected XDATCAR frame and its expected OUTCAR ionic step."""

    fps_order: int
    source_frame: int
    ionic_step: int
    atoms: Any


@dataclass(frozen=True)
class GeometryCheck:
    """Maximum differences observed when matching one selected frame."""

    max_cell_difference: float
    max_position_difference: float


@dataclass
class ExtractionStats:
    """Dataset-level statistics accumulated during streaming extraction."""

    written: int = 0
    min_energy: float = math.inf
    max_energy: float = -math.inf
    max_force: float = 0.0
    max_abs_virial: float = 0.0
    max_cell_difference: float = 0.0
    max_position_difference: float = 0.0

    def update(
        self,
        *,
        energy: float,
        forces: np.ndarray,
        virial: np.ndarray,
        geometry: GeometryCheck,
    ) -> None:
        self.written += 1
        self.min_energy = min(self.min_energy, energy)
        self.max_energy = max(self.max_energy, energy)
        self.max_force = max(
            self.max_force,
            float(np.max(np.linalg.norm(forces, axis=1))),
        )
        self.max_abs_virial = max(
            self.max_abs_virial,
            float(np.max(np.abs(virial))),
        )
        self.max_cell_difference = max(
            self.max_cell_difference,
            geometry.max_cell_difference,
        )
        self.max_position_difference = max(
            self.max_position_difference,
            geometry.max_position_difference,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run PESMaker FPS on a VASP AIMD XDATCAR and stream matching "
            "energy/force/virial labels from OUTCAR into NEP train.xyz."
        )
    )
    parser.add_argument(
        "--aimd-dir",
        type=Path,
        required=True,
        help="VASP AIMD directory containing INCAR, XDATCAR, and OUTCAR.",
    )
    parser.add_argument(
        "--max-count",
        type=int,
        required=True,
        help="Maximum number of structures selected by FPS (must be positive).",
    )
    parser.add_argument(
        "--min-distance",
        type=float,
        default=0.0,
        help=(
            "Minimum FPS distance in PESMaker simple-descriptor space. "
            "Default: 0.0."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_cli_values(args.max_count, args.min_distance)
        run_toolkit(
            aimd_dir=args.aimd_dir,
            max_count=args.max_count,
            min_distance=args.min_distance,
        )
    except ToolkitError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Error: interrupted by user", file=sys.stderr)
        return 130
    return 0


def run_toolkit(*, aimd_dir: Path, max_count: int, min_distance: float) -> Path:
    """Run the complete XDATCAR -> FPS -> streamed OUTCAR conversion."""
    aimd_dir = aimd_dir.expanduser().resolve()
    incar, xdatcar, outcar = _validate_inputs(aimd_dir)
    output_dir = aimd_dir / OUTPUT_DIR_NAME
    if output_dir.exists():
        raise ToolkitError(
            f"output directory already exists; refusing to overwrite: {output_dir}"
        )

    output_dir.mkdir(parents=False, exist_ok=False)
    logger = _configure_logger(output_dir / "toolkit.log")
    logger.info("PESMaker AIMD FPS to NEP toolkit")
    logger.info("AIMD directory : %s", aimd_dir)
    logger.info("XDATCAR        : %s", xdatcar)
    logger.info("OUTCAR         : %s", outcar)
    logger.info("Output          : %s", output_dir)
    logger.info("max_count       : %d", max_count)
    logger.info("min_distance    : %.16g", min_distance)

    try:
        nblock = parse_nblock(incar)
        logger.info("NBLOCK          : %d", nblock)
        config_path = output_dir / "pesmaker_fps.yaml"
        _write_pesmaker_config(
            config_path,
            xdatcar=xdatcar,
            output_dir=output_dir,
            max_count=max_count,
            min_distance=min_distance,
        )
        _run_pesmaker_select(config_path, aimd_dir=aimd_dir, logger=logger)
        _verify_fps_outputs(output_dir)

        selections = load_selections(output_dir, nblock=nblock)
        logger.info("FPS selected     : %d frame(s)", len(selections))
        stats = extract_train_xyz(
            outcar=outcar,
            selections=selections,
            output_dir=output_dir,
            logger=logger,
        )
        _log_success(logger, stats, output_dir / "train.xyz")
        return output_dir / "train.xyz"
    except Exception as exc:
        if isinstance(exc, ToolkitError):
            logger.error("Toolkit failed: %s", exc)
            raise
        logger.exception("Toolkit failed with an unexpected error")
        raise ToolkitError(str(exc)) from exc


def _validate_cli_values(max_count: int, min_distance: float) -> None:
    if max_count < 1:
        raise ToolkitError("--max-count must be a positive integer")
    if not math.isfinite(min_distance) or min_distance < 0.0:
        raise ToolkitError("--min-distance must be a finite non-negative number")


def _validate_inputs(aimd_dir: Path) -> tuple[Path, Path, Path]:
    if not aimd_dir.is_dir():
        raise ToolkitError(f"AIMD directory does not exist: {aimd_dir}")
    paths = tuple(aimd_dir / name for name in ("INCAR", "XDATCAR", "OUTCAR"))
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise ToolkitError(
            f"AIMD directory is missing required file(s): {', '.join(missing)}"
        )
    return paths


def parse_nblock(incar: Path) -> int:
    """Return the last active NBLOCK value in INCAR, defaulting to one."""
    nblock = 1
    pattern = re.compile(r"^\s*NBLOCK\s*=\s*([^\s;]+)", re.IGNORECASE)
    for raw_line in incar.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.split("!", 1)[0].split("#", 1)[0]
        match = pattern.match(line)
        if not match:
            continue
        try:
            nblock = int(match.group(1))
        except ValueError as exc:
            raise ToolkitError(f"invalid NBLOCK value in {incar}: {match.group(1)}") from exc
    if nblock < 1:
        raise ToolkitError(f"NBLOCK must be positive in {incar}, got {nblock}")
    return nblock


def _configure_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger(f"pesmaker_aimd_toolkit.{id(path)}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _write_pesmaker_config(
    path: Path,
    *,
    xdatcar: Path,
    output_dir: Path,
    max_count: int,
    min_distance: float,
) -> None:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - PESMaker requires PyYAML
        raise ToolkitError("PyYAML is required to generate the FPS config") from exc
    config = {
        "project": "aimd_fps_to_nep",
        "sampling": {
            "engine": "none",
            "selection": {
                "method": "fps",
                "trajectory_pattern": str(xdatcar),
                "trajectory_format": "vasp-xdatcar",
                "output_dir": str(output_dir),
                "max_count": max_count,
                "min_distance": min_distance,
                "separate_trajectories": False,
                "plot": True,
            },
        },
    }
    path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _run_pesmaker_select(
    config_path: Path,
    *,
    aimd_dir: Path,
    logger: logging.Logger,
) -> None:
    command = [sys.executable, "-m", "pesmaker", "select", str(config_path)]
    logger.info("Running          : %s", " ".join(command))
    try:
        process = subprocess.Popen(
            command,
            cwd=aimd_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except OSError as exc:
        raise ToolkitError(f"could not start PESMaker: {exc}") from exc
    assert process.stdout is not None
    try:
        for line in process.stdout:
            logger.info("[pesmaker] %s", line.rstrip())
        returncode = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
        raise
    if returncode != 0:
        raise ToolkitError(f"PESMaker select failed with exit status {returncode}")


def _verify_fps_outputs(output_dir: Path) -> None:
    missing = [name for name in REQUIRED_FPS_FILES if not (output_dir / name).is_file()]
    if missing:
        raise ToolkitError(
            "PESMaker select completed without required output file(s): "
            + ", ".join(missing)
        )


def load_selections(output_dir: Path, *, nblock: int) -> list[SelectionEntry]:
    """Load the FPS manifest and selected frames in their common order."""
    manifest_path = output_dir / "manifest.jsonl"
    records = []
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ToolkitError(
                f"invalid JSON in {manifest_path} line {line_number}"
            ) from exc
        records.append(record)

    try:
        from ase.io import iread
    except ImportError as exc:  # pragma: no cover - PESMaker requires ASE
        raise ToolkitError("ASE is required to read selected.xyz") from exc
    selected_frames = list(
        iread(output_dir / "selected.xyz", index=":", format="extxyz")
    )
    if not records:
        raise ToolkitError("FPS manifest contains no selected frames")
    if len(records) != len(selected_frames):
        raise ToolkitError(
            "FPS manifest/selected.xyz frame count mismatch: "
            f"{len(records)} vs {len(selected_frames)}"
        )
    _validate_selected_trajectory_assumptions(selected_frames)

    selections = []
    seen_source_frames: set[int] = set()
    seen_ionic_steps: set[int] = set()
    for order, (record, atoms) in enumerate(zip(records, selected_frames)):
        if "source_frame" not in record:
            raise ToolkitError(f"manifest record {order} has no source_frame")
        try:
            source_frame = int(record["source_frame"])
        except (TypeError, ValueError) as exc:
            raise ToolkitError(
                f"manifest record {order} has invalid source_frame"
            ) from exc
        if source_frame < 0 or source_frame in seen_source_frames:
            raise ToolkitError(
                f"manifest contains invalid or duplicate source_frame: {source_frame}"
            )
        ionic_step = (source_frame + 1) * nblock
        if ionic_step in seen_ionic_steps:
            raise ToolkitError(f"duplicate mapped ionic step: {ionic_step}")
        seen_source_frames.add(source_frame)
        seen_ionic_steps.add(ionic_step)
        selections.append(
            SelectionEntry(
                fps_order=order,
                source_frame=source_frame,
                ionic_step=ionic_step,
                atoms=atoms,
            )
        )
    return selections


def _validate_selected_trajectory_assumptions(selected_frames: list[Any]) -> None:
    """Enforce the first version's fixed-cell, fixed-order scope."""
    reference = selected_frames[0]
    reference_symbols = reference.get_chemical_symbols()
    reference_cell = np.asarray(reference.cell.array, dtype=float)
    if reference_cell.shape != (3, 3) or not np.all(np.isfinite(reference_cell)):
        raise ToolkitError("selected XDATCAR reference cell is not a finite 3x3 matrix")

    for frame_index, atoms in enumerate(selected_frames[1:], start=1):
        if atoms.get_chemical_symbols() != reference_symbols:
            raise ToolkitError(
                "selected XDATCAR frames do not keep a fixed atom count/order "
                f"(selected frame {frame_index})"
            )
        cell = np.asarray(atoms.cell.array, dtype=float)
        if cell.shape != (3, 3) or not np.all(np.isfinite(cell)):
            raise ToolkitError(
                f"selected XDATCAR frame {frame_index} cell is not a finite 3x3 matrix"
            )
        max_difference = float(np.max(np.abs(cell - reference_cell)))
        if max_difference > CELL_TOLERANCE_ANGSTROM:
            raise ToolkitError(
                "selected XDATCAR is not a fixed-cell trajectory: maximum cell "
                f"difference {max_difference:.6g} A exceeds "
                f"{CELL_TOLERANCE_ANGSTROM:g} A at selected frame {frame_index}"
            )


def extract_train_xyz(
    *,
    outcar: Path,
    selections: list[SelectionEntry],
    output_dir: Path,
    logger: logging.Logger,
) -> ExtractionStats:
    """Stream OUTCAR and write only the ionic steps selected by FPS."""
    by_step = {selection.ionic_step: selection for selection in selections}
    wanted_steps = set(by_step)
    found_steps: set[int] = set()
    train_partial = output_dir / "train.xyz.partial"
    mapping_partial = output_dir / "frame_mapping.jsonl.partial"
    train_path = output_dir / "train.xyz"
    mapping_path = output_dir / "frame_mapping.jsonl"
    stats = ExtractionStats()

    logger.info("Streaming OUTCAR labels; the file is not loaded into memory.")
    try:
        frames = _iter_outcar_frames(outcar)
        with train_partial.open("x", encoding="utf-8", newline="\n") as train_handle, \
                mapping_partial.open("x", encoding="utf-8", newline="\n") as map_handle:
            for ionic_step, labeled_atoms in enumerate(frames, start=1):
                selection = by_step.get(ionic_step)
                if selection is None:
                    continue
                geometry = validate_geometry(selection.atoms, labeled_atoms)
                energy, forces, virial = extract_labels(labeled_atoms)
                _write_nep_frame(
                    train_handle,
                    atoms=labeled_atoms,
                    energy=energy,
                    forces=forces,
                    virial=virial,
                )
                mapping = {
                    "train_frame": stats.written,
                    "fps_order": selection.fps_order,
                    "source_frame": selection.source_frame,
                    "ionic_step": ionic_step,
                    "label_source": str(outcar),
                }
                map_handle.write(json.dumps(mapping, ensure_ascii=False) + "\n")
                found_steps.add(ionic_step)
                stats.update(
                    energy=energy,
                    forces=forces,
                    virial=virial,
                    geometry=geometry,
                )
                logger.info(
                    "Matched ionic step %d (%d/%d)",
                    ionic_step,
                    len(found_steps),
                    len(wanted_steps),
                )
                if found_steps == wanted_steps:
                    break
    except ToolkitError:
        raise
    except Exception as exc:
        raise ToolkitError(f"failed while streaming OUTCAR: {exc}") from exc

    missing = sorted(wanted_steps - found_steps)
    if missing:
        preview = ", ".join(str(step) for step in missing[:10])
        suffix = "..." if len(missing) > 10 else ""
        raise ToolkitError(
            f"OUTCAR did not provide {len(missing)} selected ionic step(s): "
            f"{preview}{suffix}"
        )

    train_partial.replace(train_path)
    mapping_partial.replace(mapping_path)
    return stats


def _iter_outcar_frames(outcar: Path) -> Iterable[Any]:
    """Yield ASE Atoms while parsing only labels required by this toolkit.

    ASE's default OUTCAR reader also parses k-point metadata. Some valid VASP
    AIMD OUTCAR files contain abbreviated k-point header blocks that make that
    unrelated parser fail before the first ionic frame. Building ASE's stream
    generator with the minimal header/chunk parser set avoids that failure and
    still uses ASE's native VASP unit and sign conversions.
    """
    try:
        from ase.io.vasp_parsers.vasp_outcar_parsers import (
            Cell,
            Energy,
            IonsPerSpecies,
            OutcarChunkParser,
            OutcarHeaderParser,
            PositionsAndForces,
            SpeciesTypes,
            Spinpol,
            Stress,
            outcarchunks,
        )
    except ImportError as exc:  # pragma: no cover - PESMaker requires ASE
        raise ToolkitError(
            "ASE with vasp_outcar_parsers support is required to stream OUTCAR"
        ) from exc

    header_parser = OutcarHeaderParser(
        parsers=[SpeciesTypes(), IonsPerSpecies(), Spinpol()]
    )
    chunk_parser = OutcarChunkParser(
        parsers=[Cell(), PositionsAndForces(), Stress(), Energy()]
    )
    with outcar.open("r", encoding="utf-8", errors="replace") as handle:
        for chunk in outcarchunks(
            handle,
            header_parser=header_parser,
            chunk_parser=chunk_parser,
        ):
            yield chunk.build()


def validate_geometry(selected_atoms: Any, labeled_atoms: Any) -> GeometryCheck:
    """Verify that an XDATCAR selection matches one OUTCAR ionic frame."""
    if len(selected_atoms) != len(labeled_atoms):
        raise ToolkitError(
            "selected/OUTCAR atom-count mismatch: "
            f"{len(selected_atoms)} vs {len(labeled_atoms)}"
        )
    if selected_atoms.get_chemical_symbols() != labeled_atoms.get_chemical_symbols():
        raise ToolkitError("selected/OUTCAR element order mismatch")

    selected_cell = np.asarray(selected_atoms.cell.array, dtype=float)
    labeled_cell = np.asarray(labeled_atoms.cell.array, dtype=float)
    if selected_cell.shape != (3, 3) or labeled_cell.shape != (3, 3):
        raise ToolkitError("selected/OUTCAR cell is not a 3x3 matrix")
    if not np.all(np.isfinite(selected_cell)) or not np.all(np.isfinite(labeled_cell)):
        raise ToolkitError("selected/OUTCAR cell contains non-finite values")
    max_cell_difference = float(np.max(np.abs(selected_cell - labeled_cell)))
    if max_cell_difference > CELL_TOLERANCE_ANGSTROM:
        raise ToolkitError(
            "selected/OUTCAR fixed-cell mismatch: maximum cell difference "
            f"{max_cell_difference:.6g} A exceeds {CELL_TOLERANCE_ANGSTROM:g} A"
        )

    try:
        selected_scaled = np.asarray(
            selected_atoms.get_scaled_positions(wrap=False), dtype=float
        )
        labeled_scaled = np.asarray(
            labeled_atoms.get_scaled_positions(wrap=False), dtype=float
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise ToolkitError(f"could not calculate fractional coordinates: {exc}") from exc
    delta_scaled = labeled_scaled - selected_scaled
    delta_scaled -= np.round(delta_scaled)
    delta_cartesian = delta_scaled @ selected_cell
    max_position_difference = float(
        np.max(np.linalg.norm(delta_cartesian, axis=1))
    )
    if max_position_difference > POSITION_TOLERANCE_ANGSTROM:
        raise ToolkitError(
            "selected/OUTCAR position mismatch: maximum minimum-image "
            f"difference {max_position_difference:.6g} A exceeds "
            f"{POSITION_TOLERANCE_ANGSTROM:g} A"
        )
    return GeometryCheck(max_cell_difference, max_position_difference)


def extract_labels(atoms: Any) -> tuple[float, np.ndarray, np.ndarray]:
    """Extract force-consistent energy, forces, and GPUMD-sign virial."""
    if atoms.calc is None:
        raise ToolkitError("OUTCAR frame has no ASE calculator results")
    results = atoms.calc.results
    if "free_energy" not in results or results["free_energy"] is None:
        raise ToolkitError("OUTCAR frame has no free_energy/TOTEN label")
    energy = float(results["free_energy"])
    if not math.isfinite(energy):
        raise ToolkitError("OUTCAR frame has a non-finite free_energy/TOTEN")

    if "forces" not in results or results["forces"] is None:
        raise ToolkitError("OUTCAR frame has no force labels")
    forces = np.asarray(results["forces"], dtype=float)
    if forces.shape != (len(atoms), 3):
        raise ToolkitError(
            f"OUTCAR force shape is {forces.shape}, expected ({len(atoms)}, 3)"
        )
    if not np.all(np.isfinite(forces)):
        raise ToolkitError("OUTCAR frame has non-finite force labels")

    if "stress" not in results or results["stress"] is None:
        raise ToolkitError("OUTCAR frame has no stress label")
    try:
        stress = np.asarray(atoms.get_stress(voigt=False), dtype=float)
    except Exception as exc:
        raise ToolkitError(f"could not obtain 3x3 ASE stress: {exc}") from exc
    if stress.shape != (3, 3) or not np.all(np.isfinite(stress)):
        raise ToolkitError("OUTCAR stress is not a finite 3x3 tensor")
    volume = float(atoms.get_volume())
    if not math.isfinite(volume) or volume <= 0.0:
        raise ToolkitError(f"OUTCAR frame has invalid cell volume: {volume}")
    virial = -volume * stress
    if not np.all(np.isfinite(virial)):
        raise ToolkitError("converted virial contains non-finite values")
    return energy, forces, virial


def _write_nep_frame(
    handle: TextIO,
    *,
    atoms: Any,
    energy: float,
    forces: np.ndarray,
    virial: np.ndarray,
) -> None:
    cell = np.asarray(atoms.cell.array, dtype=float)
    positions = np.asarray(atoms.positions, dtype=float)
    lattice_text = " ".join(_number(value) for value in cell.reshape(-1))
    virial_text = " ".join(_number(value) for value in virial.reshape(-1))
    handle.write(f"{len(atoms)}\n")
    handle.write(
        f'Lattice="{lattice_text}" Energy={_number(energy)} '
        "Properties=species:S:1:pos:R:3:force:R:3 "
        f'Virial="{virial_text}" pbc="T T T"\n'
    )
    for symbol, position, force in zip(
        atoms.get_chemical_symbols(), positions, forces
    ):
        values = " ".join(_number(value) for value in np.concatenate((position, force)))
        handle.write(f"{symbol} {values}\n")


def _number(value: float) -> str:
    return f"{float(value):.16g}"


def _log_success(logger: logging.Logger, stats: ExtractionStats, train_path: Path) -> None:
    logger.info("Dataset complete")
    logger.info("Structures       : %d", stats.written)
    logger.info("Energy range     : %.16g .. %.16g eV", stats.min_energy, stats.max_energy)
    logger.info("Maximum force    : %.16g eV/A", stats.max_force)
    logger.info("Maximum |Virial| : %.16g eV", stats.max_abs_virial)
    logger.info("Maximum cell diff: %.16g A", stats.max_cell_difference)
    logger.info("Maximum pos diff : %.16g A", stats.max_position_difference)
    logger.info("train.xyz        : %s", train_path)


if __name__ == "__main__":
    raise SystemExit(main())
