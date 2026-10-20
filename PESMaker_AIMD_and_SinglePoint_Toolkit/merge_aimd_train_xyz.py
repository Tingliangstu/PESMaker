#!/usr/bin/env python3
"""Merge validated AIMD toolkit train.xyz files from one sampling source.

The user must choose ``interval`` or ``fps`` explicitly. The script scans only
direct AIMD children and only the official output directory for that source;
it never searches recursively for arbitrary files named ``train.xyz``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import logging
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class SourceMode:
    """Directory mapping and display name for one sampling source."""

    key: str
    display_name: str
    output_directory: str


SOURCE_MODES = {
    "interval": SourceMode(
        key="interval",
        display_name="Interval",
        output_directory="PESMakerToolkit_AIMD_Interval_to_NEP",
    ),
    "fps": SourceMode(
        key="fps",
        display_name="FPS",
        output_directory="PESMakerToolkit_AIMD_FPS_to_NEP",
    ),
}


@dataclass(frozen=True)
class DatasetStats:
    """Labels and structure information observed in one extxyz dataset."""

    frame_count: int
    min_atom_count: int
    max_atom_count: int
    elements: tuple[str, ...]


@dataclass(frozen=True)
class SourceSummary:
    """One source dataset and its inclusive range in the merged dataset."""

    order: int
    aimd_directory: Path
    train_xyz: Path
    stats: DatasetStats
    merged_start_0based: int
    merged_end_0based: int


class MergeError(RuntimeError):
    """A user-facing merge failure."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge validated train.xyz files from the explicitly selected "
            "AIMD toolkit sampling source."
        )
    )
    parser.add_argument(
        "--aimd-root",
        type=Path,
        required=True,
        help=(
            "Directory whose direct children are VASP AIMD run directories; "
            "the merged output directory is created inside this directory."
        ),
    )
    parser.add_argument(
        "--source",
        choices=tuple(SOURCE_MODES),
        required=True,
        help="Sampling source to merge: interval or fps.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_merge(aimd_root=args.aimd_root, source=args.source)
    except MergeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Error: interrupted by user", file=sys.stderr)
        return 130
    return 0


def run_merge(*, aimd_root: Path, source: str) -> Path:
    """Validate and merge all direct AIMD children for one sampling source."""
    if source not in SOURCE_MODES:
        raise MergeError(
            f"unsupported source {source!r}; choose one of: "
            + ", ".join(SOURCE_MODES)
        )
    mode = SOURCE_MODES[source]
    aimd_root = aimd_root.expanduser().resolve()
    if not aimd_root.is_dir():
        raise MergeError(f"AIMD root directory does not exist: {aimd_root}")

    output_dir = aimd_root / f"Merged_{mode.display_name}_NEP_TrainXYZ"
    if output_dir.exists():
        raise MergeError(
            f"output directory already exists; refusing to overwrite: {output_dir}"
        )

    aimd_directories = discover_aimd_directories(aimd_root)
    source_paths = require_source_train_xyz(aimd_directories, mode)

    output_dir.mkdir(parents=False, exist_ok=False)
    logger = configure_logger(output_dir / "merge.log")
    logger.info("PESMaker AIMD train.xyz merge toolkit")
    logger.info("AIMD root       : %s", aimd_root)
    logger.info("Sampling source : %s", mode.key)
    logger.info("Source directory: %s", mode.output_directory)
    logger.info("AIMD directories: %d", len(aimd_directories))
    logger.info("Output          : %s", output_dir)

    train_partial = output_dir / "train.xyz.partial"
    ranges_partial = output_dir / "source_ranges.tsv.partial"
    readme_partial = output_dir / "README.md.partial"

    try:
        summaries = build_source_summaries(
            aimd_root=aimd_root,
            source_paths=source_paths,
            logger=logger,
        )
        total_frames = sum(summary.stats.frame_count for summary in summaries)
        concatenate_train_xyz(
            [summary.train_xyz for summary in summaries],
            train_partial,
        )
        write_source_ranges(
            ranges_partial,
            aimd_root=aimd_root,
            mode=mode,
            summaries=summaries,
        )
        write_output_readme(
            readme_partial,
            aimd_root=aimd_root,
            mode=mode,
            summaries=summaries,
            total_frames=total_frames,
        )

        merged_stats = validate_train_xyz(train_partial)
        if merged_stats.frame_count != total_frames:
            raise MergeError(
                "merged frame-count mismatch: expected "
                f"{total_frames}, got {merged_stats.frame_count}"
            )
        expected_elements = tuple(
            sorted(
                {
                    element
                    for summary in summaries
                    for element in summary.stats.elements
                }
            )
        )
        if merged_stats.elements != expected_elements:
            raise MergeError(
                "merged element set does not match the source datasets: "
                f"expected {expected_elements}, got {merged_stats.elements}"
            )

        ranges_partial.replace(output_dir / "source_ranges.tsv")
        readme_partial.replace(output_dir / "README.md")
        train_partial.replace(output_dir / "train.xyz")
        logger.info("Merge complete")
        logger.info("Source datasets : %d", len(summaries))
        logger.info("Total frames    : %d", total_frames)
        logger.info("Elements        : %s", " ".join(merged_stats.elements))
        logger.info("train.xyz       : %s", output_dir / "train.xyz")
        return output_dir / "train.xyz"
    except Exception as exc:
        if isinstance(exc, MergeError):
            logger.error("Merge failed: %s", exc)
            raise
        logger.exception("Merge failed with an unexpected error")
        raise MergeError(str(exc)) from exc


def discover_aimd_directories(aimd_root: Path) -> list[Path]:
    """Find direct children that contain the three required VASP files."""
    required_files = ("INCAR", "XDATCAR", "OUTCAR")
    directories = [
        child
        for child in aimd_root.iterdir()
        if child.is_dir()
        and all((child / filename).is_file() for filename in required_files)
    ]
    directories.sort(key=lambda path: path.name.casefold())
    if not directories:
        raise MergeError(
            "AIMD root has no direct child containing INCAR, XDATCAR, and OUTCAR: "
            f"{aimd_root}"
        )
    return directories


def require_source_train_xyz(
    aimd_directories: list[Path],
    mode: SourceMode,
) -> list[Path]:
    """Require every discovered AIMD child to have the chosen official source."""
    source_paths = [
        directory / mode.output_directory / "train.xyz"
        for directory in aimd_directories
    ]
    missing = [
        directory
        for directory, source_path in zip(aimd_directories, source_paths)
        if not source_path.is_file()
    ]
    if missing:
        names = ", ".join(directory.name for directory in missing)
        raise MergeError(
            f"{len(missing)} AIMD directorie(s) lack the selected "
            f"{mode.key} train.xyz in {mode.output_directory}: {names}"
        )
    return source_paths


def build_source_summaries(
    *,
    aimd_root: Path,
    source_paths: list[Path],
    logger: logging.Logger,
) -> list[SourceSummary]:
    """Validate source datasets and assign contiguous merged frame ranges."""
    summaries = []
    next_start = 0
    for order, train_xyz in enumerate(source_paths, start=1):
        stats = validate_train_xyz(train_xyz)
        merged_end = next_start + stats.frame_count - 1
        summary = SourceSummary(
            order=order,
            aimd_directory=train_xyz.parent.parent,
            train_xyz=train_xyz,
            stats=stats,
            merged_start_0based=next_start,
            merged_end_0based=merged_end,
        )
        summaries.append(summary)
        logger.info(
            "Source %d        : %s | %d frame(s) | merged %d..%d",
            order,
            train_xyz.relative_to(aimd_root).as_posix(),
            stats.frame_count,
            next_start,
            merged_end,
        )
        next_start = merged_end + 1
    return summaries


def validate_train_xyz(path: Path) -> DatasetStats:
    """Stream one extxyz dataset and require finite NEP labels on every frame."""
    try:
        from ase.io import iread
    except ImportError as exc:  # pragma: no cover - toolkit requires ASE
        raise MergeError("ASE is required to validate train.xyz") from exc

    frame_count = 0
    min_atom_count = sys.maxsize
    max_atom_count = 0
    elements: set[str] = set()
    try:
        frames: Iterable[Any] = iread(path, index=":", format="extxyz")
        for frame_index, atoms in enumerate(frames):
            validate_nep_frame(atoms, path=path, frame_index=frame_index)
            atom_count = len(atoms)
            frame_count += 1
            min_atom_count = min(min_atom_count, atom_count)
            max_atom_count = max(max_atom_count, atom_count)
            elements.update(atoms.get_chemical_symbols())
    except MergeError:
        raise
    except Exception as exc:
        raise MergeError(f"could not read extxyz dataset {path}: {exc}") from exc

    if frame_count == 0:
        raise MergeError(f"train.xyz contains no frames: {path}")
    return DatasetStats(
        frame_count=frame_count,
        min_atom_count=min_atom_count,
        max_atom_count=max_atom_count,
        elements=tuple(sorted(elements)),
    )


def validate_nep_frame(atoms: Any, *, path: Path, frame_index: int) -> None:
    """Validate labels and geometry required by GPUMD/NEP extxyz input."""
    prefix = f"{path} frame {frame_index}"
    if len(atoms) < 1:
        raise MergeError(f"{prefix} has no atoms")
    cell = np.asarray(atoms.cell.array, dtype=float)
    positions = np.asarray(atoms.positions, dtype=float)
    if cell.shape != (3, 3) or not np.all(np.isfinite(cell)):
        raise MergeError(f"{prefix} has a non-finite or invalid cell")
    if positions.shape != (len(atoms), 3) or not np.all(np.isfinite(positions)):
        raise MergeError(f"{prefix} has non-finite or invalid positions")

    if "Energy" not in atoms.info:
        raise MergeError(f"{prefix} has no Energy label")
    energy = float(atoms.info["Energy"])
    if not math.isfinite(energy):
        raise MergeError(f"{prefix} has a non-finite Energy label")

    if "force" not in atoms.arrays:
        raise MergeError(f"{prefix} has no force labels")
    forces = np.asarray(atoms.arrays["force"], dtype=float)
    if forces.shape != (len(atoms), 3) or not np.all(np.isfinite(forces)):
        raise MergeError(f"{prefix} has non-finite or invalid force labels")

    if "Virial" not in atoms.info:
        raise MergeError(f"{prefix} has no Virial label")
    virial = np.asarray(atoms.info["Virial"], dtype=float)
    if virial.size != 9 or not np.all(np.isfinite(virial)):
        raise MergeError(f"{prefix} has non-finite or invalid Virial labels")


def concatenate_train_xyz(source_paths: list[Path], destination: Path) -> None:
    """Concatenate validated source text without reformatting extxyz frames."""
    with destination.open("x", encoding="utf-8", newline="\n") as output:
        for source_path in source_paths:
            last_character = "\n"
            with source_path.open("r", encoding="utf-8") as source:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    last_character = chunk[-1]
            if last_character != "\n":
                output.write("\n")


def write_source_ranges(
    path: Path,
    *,
    aimd_root: Path,
    mode: SourceMode,
    summaries: list[SourceSummary],
) -> None:
    """Write a machine-readable source and merged-range table."""
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            (
                "order",
                "sampling_source",
                "aimd_directory",
                "source_train_xyz",
                "frame_count",
                "atom_count",
                "elements",
                "merged_start_0based",
                "merged_end_0based",
                "merged_start_1based",
                "merged_end_1based",
            )
        )
        for summary in summaries:
            writer.writerow(
                (
                    summary.order,
                    mode.key,
                    summary.aimd_directory.name,
                    summary.train_xyz.relative_to(aimd_root).as_posix(),
                    summary.stats.frame_count,
                    atom_count_label(summary.stats),
                    " ".join(summary.stats.elements),
                    summary.merged_start_0based,
                    summary.merged_end_0based,
                    summary.merged_start_0based + 1,
                    summary.merged_end_0based + 1,
                )
            )


def write_output_readme(
    path: Path,
    *,
    aimd_root: Path,
    mode: SourceMode,
    summaries: list[SourceSummary],
    total_frames: int,
) -> None:
    """Write a concise human-readable provenance and layout description."""
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    lines = [
        f"# Merged {mode.display_name} NEP train.xyz",
        "",
        "This directory was generated automatically by `merge_aimd_train_xyz.py`.",
        "",
        f"- AIMD root: `{aimd_root}`",
        f"- Sampling source: `{mode.key}`",
        f"- Source output directory: `{mode.output_directory}`",
        f"- Merge time: `{timestamp}`",
        f"- Number of source trajectories: `{len(summaries)}`",
        f"- Total merged frames: `{total_frames}`",
        "",
        "## File Layout",
        "",
        "```text",
        f"Merged_{mode.display_name}_NEP_TrainXYZ/",
        "├── train.xyz",
        "├── source_ranges.tsv",
        "├── README.md",
        "└── merge.log",
        "```",
        "",
        "- `train.xyz`: GPUMD/NEP training set concatenated in the order below.",
        "- `source_ranges.tsv`: machine-readable source paths, frame counts, and merged ranges.",
        "- `README.md`: this description.",
        "- `merge.log`: validation, merge progress, and error log.",
        "",
        "## Sources and Frame Ranges",
        "",
        "All ranges are inclusive. Zero-based ranges are intended for programmatic use; one-based ranges are intended for human counting.",
        "",
        "| Order | AIMD directory | Frames | Atoms | Elements | Zero-based range | One-based range |",
        "| ---: | --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for summary in summaries:
        lines.append(
            "| "
            f"{summary.order} | {markdown_cell(summary.aimd_directory.name)} | "
            f"{summary.stats.frame_count} | {atom_count_label(summary.stats)} | "
            f"{' '.join(summary.stats.elements)} | "
            f"{summary.merged_start_0based}–{summary.merged_end_0based} | "
            f"{summary.merged_start_0based + 1}–{summary.merged_end_0based + 1} |"
        )
    lines.extend(
        [
            "",
            "## Merge Rules",
            "",
            "- Read only direct children of the AIMD root.",
            f"- Read only `{mode.output_directory}/train.xyz`.",
            "- Do not recursively search for other `train.xyz` files or mix sampling sources.",
            "- Sort sources case-insensitively by AIMD directory name.",
            "- Do not shuffle, deduplicate, split train/test data, or convert label units.",
            "- Treat source files as read-only; merging does not modify original data.",
            "",
            "## Reproduction Command",
            "",
            "```bash",
            "python merge_aimd_train_xyz.py \\",
            f"  --aimd-root {aimd_root} \\",
            f"  --source {mode.key}",
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def atom_count_label(stats: DatasetStats) -> str:
    if stats.min_atom_count == stats.max_atom_count:
        return str(stats.min_atom_count)
    return f"{stats.min_atom_count}-{stats.max_atom_count}"


def markdown_cell(value: str) -> str:
    return value.replace("|", "\\|")


def configure_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger(f"pesmaker_aimd_merge_toolkit.{id(path)}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


if __name__ == "__main__":
    raise SystemExit(main())
