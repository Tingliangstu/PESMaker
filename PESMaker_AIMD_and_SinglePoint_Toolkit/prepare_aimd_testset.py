#!/usr/bin/env python3
"""Build a labeled NEP test.xyz from AIMD frames excluded from training.

The script is independent of PESMaker's core workflow.  It reads the
``frame_mapping.jsonl`` written by the FPS or interval Toolkit script, selects
unused XDATCAR frames across each trajectory, and reuses the Toolkit's
streaming OUTCAR label parser to produce one merged GPUMD/NEP ``test.xyz``.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
import csv
import json
import logging
from pathlib import Path
import sys
from typing import Any, Iterable, TextIO

from aimd_fps_to_nep import (
    ExtractionStats,
    SelectionEntry,
    ToolkitError,
    _iter_outcar_frames,
    _validate_selected_trajectory_assumptions,
    _write_nep_frame,
    extract_labels,
    parse_nblock,
    validate_geometry,
)
from aimd_interval_to_nep import count_xdatcar_frames


OUTPUT_DIR_NAME = "test_verification_testset"
SOURCE_OUTPUT_DIRS = {
    "interval": "PESMakerToolkit_AIMD_Interval_to_NEP",
    "fps": "PESMakerToolkit_AIMD_FPS_to_NEP",
}
REQUIRED_AIMD_FILES = ("INCAR", "XDATCAR", "OUTCAR")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a merged NEP test.xyz from labeled AIMD frames that are "
            "absent from an existing Toolkit-generated training set."
        )
    )
    parser.add_argument(
        "--aimd-root",
        type=Path,
        required=True,
        help="Parent directory whose direct children are VASP AIMD projects.",
    )
    parser.add_argument(
        "--training-source",
        choices=tuple(SOURCE_OUTPUT_DIRS),
        required=True,
        help="Training mapping to exclude: interval or fps.",
    )
    selection_group = parser.add_mutually_exclusive_group()
    selection_group.add_argument(
        "--count-per-trajectory",
        type=int,
        default=None,
        help="Number of unused test frames selected from each trajectory. Default: 20.",
    )
    selection_group.add_argument(
        "--all-unused",
        action="store_true",
        help="Use every non-training XDATCAR frame that has an OUTCAR label.",
    )
    parser.add_argument(
        "--output-dir-name",
        default=OUTPUT_DIR_NAME,
        help=(
            "Name of the output directory created directly below --aimd-root. "
            f"Default: {OUTPUT_DIR_NAME}."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        prepare_testset(
            aimd_root=args.aimd_root,
            training_source=args.training_source,
            count_per_trajectory=(
                20 if args.count_per_trajectory is None else args.count_per_trajectory
            ),
            all_unused=args.all_unused,
            output_dir_name=args.output_dir_name,
        )
    except ToolkitError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Error: interrupted by user", file=sys.stderr)
        return 130
    return 0


def prepare_testset(
    *,
    aimd_root: Path,
    training_source: str,
    count_per_trajectory: int = 20,
    all_unused: bool = False,
    output_dir_name: str = OUTPUT_DIR_NAME,
) -> Path:
    """Create ``<aimd_root>/test_verification_testset/test.xyz``."""
    if training_source not in SOURCE_OUTPUT_DIRS:
        raise ToolkitError(
            f"unsupported training source {training_source!r}; "
            f"choose from {', '.join(SOURCE_OUTPUT_DIRS)}"
        )
    if not all_unused and count_per_trajectory < 1:
        raise ToolkitError("--count-per-trajectory must be a positive integer")
    _validate_output_dir_name(output_dir_name)

    aimd_root = aimd_root.expanduser().resolve()
    if not aimd_root.is_dir():
        raise ToolkitError(f"AIMD root directory does not exist: {aimd_root}")

    projects = discover_projects(aimd_root, training_source)
    if not projects:
        raise ToolkitError(
            "no direct AIMD child directory contains INCAR, XDATCAR, OUTCAR, "
            f"and {SOURCE_OUTPUT_DIRS[training_source]}/frame_mapping.jsonl"
        )

    output_dir = aimd_root / output_dir_name
    _prepare_empty_output_directory(output_dir)
    logger = _configure_logger(output_dir / "toolkit.log")
    logger.info("PESMaker Toolkit AIMD verification test-set builder")
    logger.info("AIMD root             : %s", aimd_root)
    logger.info("Training source       : %s", training_source)
    logger.info(
        "Selection mode         : %s",
        "all unused labeled frames" if all_unused else "balanced fixed count",
    )
    if not all_unused:
        logger.info("Count per trajectory  : %d", count_per_trajectory)
    logger.info("Projects              : %d", len(projects))
    logger.info("Output                : %s", output_dir)

    test_partial = output_dir / "test.xyz.partial"
    mapping_partial = output_dir / "frame_mapping.jsonl.partial"
    test_path = output_dir / "test.xyz"
    mapping_path = output_dir / "frame_mapping.jsonl"
    skipped_path = output_dir / "skipped_unlabeled_frames.tsv"
    skipped_partial = output_dir / "skipped_unlabeled_frames.tsv.partial"
    summary_rows: list[dict[str, Any]] = []
    total_stats = ExtractionStats()
    global_test_frame = 0
    skipped_total = 0

    try:
        with test_partial.open("x", encoding="utf-8", newline="\n") as test_handle, \
                mapping_partial.open("x", encoding="utf-8", newline="\n") as map_handle, \
                skipped_partial.open("x", encoding="utf-8", newline="\n") as skipped_handle:
            skipped_writer = csv.DictWriter(
                skipped_handle,
                fieldnames=("source_project", "source_frame", "ionic_step", "reason"),
                delimiter="\t",
            )
            skipped_writer.writeheader()
            for project in projects:
                frame_count = count_xdatcar_frames(project / "XDATCAR")
                training_frames = load_training_frames(
                    project
                    / SOURCE_OUTPUT_DIRS[training_source]
                    / "frame_mapping.jsonl",
                    frame_count=frame_count,
                )
                nblock = parse_nblock(project / "INCAR")
                start_frame = global_test_frame
                if all_unused:
                    logger.info(
                        "[%s] XDATCAR=%d, training=%d, requested test=%d, NBLOCK=%d",
                        project.name,
                        frame_count,
                        len(training_frames),
                        frame_count - len(training_frames),
                        nblock,
                    )
                    written, skipped, source_min, source_max = extract_all_unused_project_labels(
                        project=project,
                        frame_count=frame_count,
                        training_frames=training_frames,
                        training_source=training_source,
                        nblock=nblock,
                        test_handle=test_handle,
                        mapping_handle=map_handle,
                        skipped_writer=skipped_writer,
                        first_test_frame=global_test_frame,
                        stats=total_stats,
                        logger=logger,
                    )
                    skipped_total += skipped
                else:
                    test_frames = select_balanced_unused_frames(
                        frame_count=frame_count,
                        excluded_frames=training_frames,
                        count=count_per_trajectory,
                    )
                    selections = load_xdatcar_selections(
                        project / "XDATCAR",
                        test_frames,
                        nblock=nblock,
                    )
                    logger.info(
                        "[%s] XDATCAR=%d, training=%d, test=%d, NBLOCK=%d",
                        project.name,
                        frame_count,
                        len(training_frames),
                        len(test_frames),
                        nblock,
                    )
                    written = extract_project_labels(
                        project=project,
                        selections=selections,
                        training_frames=training_frames,
                        training_source=training_source,
                        test_handle=test_handle,
                        mapping_handle=map_handle,
                        first_test_frame=global_test_frame,
                        stats=total_stats,
                        logger=logger,
                    )
                    skipped = 0
                    source_min, source_max = min(test_frames), max(test_frames)
                global_test_frame += written
                summary_rows.append(
                    {
                        "aimd_directory": project.name,
                        "frame_count": written,
                        "test_start_0based": start_frame,
                        "test_end_0based": global_test_frame - 1,
                        "source_frame_min": source_min,
                        "source_frame_max": source_max,
                    }
                )

        if not all_unused and global_test_frame != len(projects) * count_per_trajectory:
            raise ToolkitError(
                f"expected {len(projects) * count_per_trajectory} test frames, "
                f"but wrote {global_test_frame}"
            )
        _write_source_ranges(
            output_dir / "source_ranges.tsv",
            summary_rows,
        )
        _write_output_readme(
            output_dir / "README.md",
            aimd_root=aimd_root,
            training_source=training_source,
            selection_description=(
                "all untrained frames with available OUTCAR labels"
                if all_unused
                else f"balanced selection of {count_per_trajectory} frames per trajectory"
            ),
            rows=summary_rows,
            total_frames=global_test_frame,
            skipped_unlabeled_frames=skipped_total,
        )
        test_partial.replace(test_path)
        mapping_partial.replace(mapping_path)
        skipped_partial.replace(skipped_path)
    except Exception as exc:
        if isinstance(exc, ToolkitError):
            logger.error("Test-set creation failed: %s", exc)
            raise
        logger.exception("Test-set creation failed with an unexpected error")
        raise ToolkitError(str(exc)) from exc

    logger.info("Test set complete")
    logger.info("Structures            : %d", total_stats.written)
    logger.info("Skipped unlabeled      : %d", skipped_total)
    logger.info(
        "Energy range          : %.16g .. %.16g eV",
        total_stats.min_energy,
        total_stats.max_energy,
    )
    logger.info("Maximum force         : %.16g eV/A", total_stats.max_force)
    logger.info("Maximum |Virial|      : %.16g eV", total_stats.max_abs_virial)
    logger.info("Maximum cell diff     : %.16g A", total_stats.max_cell_difference)
    logger.info("Maximum position diff : %.16g A", total_stats.max_position_difference)
    logger.info("test.xyz              : %s", test_path)
    return test_path


def discover_projects(aimd_root: Path, training_source: str) -> list[Path]:
    """Find direct AIMD children with the requested training mapping."""
    source_dir = SOURCE_OUTPUT_DIRS[training_source]
    projects = []
    for child in aimd_root.iterdir():
        if not child.is_dir():
            continue
        if not all((child / name).is_file() for name in REQUIRED_AIMD_FILES):
            continue
        if not (child / source_dir / "frame_mapping.jsonl").is_file():
            continue
        projects.append(child)
    return sorted(projects, key=lambda path: path.name.casefold())


def load_training_frames(mapping_path: Path, *, frame_count: int) -> set[int]:
    """Read and validate zero-based XDATCAR frames already used for training."""
    frames: set[int] = set()
    with mapping_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                source_frame = int(record["source_frame"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ToolkitError(
                    f"invalid source_frame in {mapping_path} line {line_number}"
                ) from exc
            if source_frame < 0 or source_frame >= frame_count:
                raise ToolkitError(
                    f"training source_frame {source_frame} is outside 0.."
                    f"{frame_count - 1} in {mapping_path}"
                )
            if source_frame in frames:
                raise ToolkitError(
                    f"duplicate training source_frame {source_frame} in {mapping_path}"
                )
            frames.add(source_frame)
    if not frames:
        raise ToolkitError(f"training mapping is empty: {mapping_path}")
    return frames


def select_balanced_unused_frames(
    *,
    frame_count: int,
    excluded_frames: set[int],
    count: int,
) -> list[int]:
    """Select one well-separated unused frame from each equal time bin.

    Within each bin, preference is given to the frame farthest in time from
    every training frame.  Ties are resolved toward the bin center and then
    toward the smaller frame number.
    """
    if frame_count < 1:
        raise ToolkitError("XDATCAR frame count must be positive")
    if count < 1:
        raise ToolkitError("test-frame count must be positive")
    if count > frame_count - len(excluded_frames):
        raise ToolkitError(
            f"requested {count} test frames but only "
            f"{frame_count - len(excluded_frames)} unused frames are available"
        )

    selection_anchors = sorted(excluded_frames | {-1, frame_count})
    selected: list[int] = []
    for bin_index in range(count):
        start = bin_index * frame_count // count
        end = (bin_index + 1) * frame_count // count
        candidates = (
            frame for frame in range(start, end) if frame not in excluded_frames
        )
        center = (start + end - 1) / 2.0
        try:
            best = max(
                candidates,
                key=lambda frame: (
                    _nearest_distance(frame, selection_anchors),
                    -abs(frame - center),
                    -frame,
                ),
            )
        except ValueError as exc:
            raise ToolkitError(
                f"time bin {start}..{end - 1} contains no unused frame"
            ) from exc
        selected.append(best)
    if len(selected) != len(set(selected)) or excluded_frames.intersection(selected):
        raise ToolkitError("internal error: selected test frames are not disjoint")
    return selected


def _nearest_distance(frame: int, sorted_frames: list[int]) -> int:
    if not sorted_frames:
        return sys.maxsize
    position = bisect_left(sorted_frames, frame)
    distances = []
    if position:
        distances.append(frame - sorted_frames[position - 1])
    if position < len(sorted_frames):
        distances.append(sorted_frames[position] - frame)
    return min(distances)


def load_xdatcar_selections(
    xdatcar: Path,
    source_frames: list[int],
    *,
    nblock: int,
) -> list[SelectionEntry]:
    """Stream XDATCAR and retain only requested frames as geometry references."""
    try:
        from ase.io import iread
    except ImportError as exc:
        raise ToolkitError("ASE is required to read XDATCAR") from exc

    wanted = set(source_frames)
    selected_atoms: dict[int, Any] = {}
    try:
        for frame_index, atoms in enumerate(
            iread(str(xdatcar), index=":", format="vasp-xdatcar")
        ):
            if frame_index in wanted:
                selected_atoms[frame_index] = atoms
                if len(selected_atoms) == len(wanted):
                    break
    except Exception as exc:
        raise ToolkitError(f"failed to stream selected XDATCAR frames: {exc}") from exc

    missing = sorted(wanted - selected_atoms.keys())
    if missing:
        raise ToolkitError(
            f"XDATCAR did not provide selected frame(s): "
            + ", ".join(str(frame) for frame in missing[:10])
        )
    ordered_atoms = [selected_atoms[frame] for frame in source_frames]
    _validate_selected_trajectory_assumptions(ordered_atoms)
    return [
        SelectionEntry(
            fps_order=order,
            source_frame=source_frame,
            ionic_step=(source_frame + 1) * nblock,
            atoms=atoms,
        )
        for order, (source_frame, atoms) in enumerate(
            zip(source_frames, ordered_atoms)
        )
    ]


def extract_project_labels(
    *,
    project: Path,
    selections: list[SelectionEntry],
    training_frames: set[int],
    training_source: str,
    test_handle: TextIO,
    mapping_handle: TextIO,
    first_test_frame: int,
    stats: ExtractionStats,
    logger: logging.Logger,
) -> int:
    """Stream one OUTCAR and append selected labeled frames to shared outputs."""
    by_step = {selection.ionic_step: selection for selection in selections}
    wanted_steps = set(by_step)
    found_steps: set[int] = set()
    outcar = project / "OUTCAR"

    try:
        frames: Iterable[Any] = _iter_outcar_frames(outcar)
        for ionic_step, labeled_atoms in enumerate(frames, start=1):
            selection = by_step.get(ionic_step)
            if selection is None:
                continue
            if selection.source_frame in training_frames:
                raise ToolkitError(
                    f"selected test frame {selection.source_frame} is in training set"
                )
            geometry = validate_geometry(selection.atoms, labeled_atoms)
            energy, forces, virial = extract_labels(labeled_atoms)
            _write_nep_frame(
                test_handle,
                atoms=labeled_atoms,
                energy=energy,
                forces=forces,
                virial=virial,
            )
            mapping = {
                "test_frame": first_test_frame + len(found_steps),
                "source_project": project.name,
                "source_frame": selection.source_frame,
                "ionic_step": ionic_step,
                "nearest_training_frame_distance": _nearest_distance(
                    selection.source_frame,
                    sorted(training_frames),
                ),
                "training_source": training_source,
                "label_source": str(outcar),
            }
            mapping_handle.write(json.dumps(mapping, ensure_ascii=False) + "\n")
            found_steps.add(ionic_step)
            stats.update(
                energy=energy,
                forces=forces,
                virial=virial,
                geometry=geometry,
            )
            logger.info(
                "[%s] matched source frame %d (%d/%d)",
                project.name,
                selection.source_frame,
                len(found_steps),
                len(wanted_steps),
            )
            if found_steps == wanted_steps:
                break
    except ToolkitError:
        raise
    except Exception as exc:
        raise ToolkitError(f"failed while streaming {outcar}: {exc}") from exc

    missing = sorted(wanted_steps - found_steps)
    if missing:
        raise ToolkitError(
            f"{outcar} did not provide {len(missing)} selected ionic step(s): "
            + ", ".join(str(step) for step in missing[:10])
        )
    return len(found_steps)


def extract_all_unused_project_labels(
    *,
    project: Path,
    frame_count: int,
    training_frames: set[int],
    training_source: str,
    nblock: int,
    test_handle: TextIO,
    mapping_handle: TextIO,
    skipped_writer: csv.DictWriter,
    first_test_frame: int,
    stats: ExtractionStats,
    logger: logging.Logger,
) -> tuple[int, int, int, int]:
    """Stream matching XDATCAR/OUTCAR frames without retaining a trajectory.

    This path is for an all-unused test set.  It holds only the current XDATCAR
    and OUTCAR frames in memory, so it remains suitable for long AIMD runs.
    """
    try:
        from ase.io import iread
    except ImportError as exc:
        raise ToolkitError("ASE is required to read XDATCAR") from exc

    outcar = project / "OUTCAR"
    sorted_training_frames = sorted(training_frames)
    outcar_frames = iter(_iter_outcar_frames(outcar))
    current_ionic_step = 0
    written = 0
    skipped = 0
    source_min: int | None = None
    source_max: int | None = None
    progress_interval = 1000

    try:
        for source_frame, reference_atoms in enumerate(
            iread(str(project / "XDATCAR"), index=":", format="vasp-xdatcar")
        ):
            if source_frame >= frame_count:
                raise ToolkitError(
                    f"XDATCAR reader returned more than {frame_count} frames: {project}"
                )
            if source_frame in training_frames:
                continue
            wanted_step = (source_frame + 1) * nblock
            labeled_atoms: Any | None = None
            while current_ionic_step < wanted_step:
                try:
                    labeled_atoms = next(outcar_frames)
                except StopIteration:
                    break
                current_ionic_step += 1
            if current_ionic_step != wanted_step or labeled_atoms is None:
                for missing_frame in range(source_frame, frame_count):
                    if missing_frame not in training_frames:
                        skipped_writer.writerow(
                            {
                                "source_project": project.name,
                                "source_frame": missing_frame,
                                "ionic_step": (missing_frame + 1) * nblock,
                                "reason": "OUTCAR ended before this ionic step",
                            }
                        )
                        skipped += 1
                logger.warning(
                    "[%s] OUTCAR ended at ionic step %d; skipped %d unlabeled tail frame(s)",
                    project.name,
                    current_ionic_step,
                    skipped,
                )
                break

            geometry = validate_geometry(reference_atoms, labeled_atoms)
            energy, forces, virial = extract_labels(labeled_atoms)
            _write_nep_frame(
                test_handle,
                atoms=labeled_atoms,
                energy=energy,
                forces=forces,
                virial=virial,
            )
            mapping_handle.write(
                json.dumps(
                    {
                        "test_frame": first_test_frame + written,
                        "source_project": project.name,
                        "source_frame": source_frame,
                        "ionic_step": wanted_step,
                        "nearest_training_frame_distance": _nearest_distance(
                            source_frame,
                            sorted_training_frames,
                        ),
                        "training_source": training_source,
                        "label_source": str(outcar),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            stats.update(
                energy=energy,
                forces=forces,
                virial=virial,
                geometry=geometry,
            )
            written += 1
            source_min = source_frame if source_min is None else source_min
            source_max = source_frame
            if written == 1 or written % progress_interval == 0:
                logger.info(
                    "[%s] wrote %d all-unused test frame(s); latest source frame %d",
                    project.name,
                    written,
                    source_frame,
                )
    except ToolkitError:
        raise
    except Exception as exc:
        raise ToolkitError(
            f"failed while streaming XDATCAR/OUTCAR for {project}: {exc}"
        ) from exc

    if source_min is None or source_max is None:
        raise ToolkitError(f"no labeled unused frames were written for {project}")
    return written, skipped, source_min, source_max


def _prepare_empty_output_directory(output_dir: Path) -> None:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ToolkitError(f"output path exists and is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise ToolkitError(
                f"output directory is not empty; refusing to overwrite: {output_dir}"
            )
        return
    output_dir.mkdir(parents=False, exist_ok=False)


def _validate_output_dir_name(output_dir_name: str) -> None:
    candidate = Path(output_dir_name)
    if (
        not output_dir_name
        or candidate.name != output_dir_name
        or output_dir_name in {".", ".."}
    ):
        raise ToolkitError(
            "--output-dir-name must be one simple directory name, not a path"
        )


def _configure_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger(f"pesmaker_aimd_testset_toolkit.{id(path)}")
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


def _write_source_ranges(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "aimd_directory",
        "frame_count",
        "test_start_0based",
        "test_end_0based",
        "source_frame_min",
        "source_frame_max",
    ]
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _write_output_readme(
    path: Path,
    *,
    aimd_root: Path,
    training_source: str,
    selection_description: str,
    rows: list[dict[str, Any]],
    total_frames: int,
    skipped_unlabeled_frames: int,
) -> None:
    ranges = "\n".join(
        f"- `{row['aimd_directory']}`: test.xyz frames "
        f"{row['test_start_0based']}–{row['test_end_0based']}; "
        f"{row['frame_count']} frames total"
        for row in rows
    )
    text = f"""# AIMD verification test set

This directory was generated by `prepare_aimd_testset.py` for preliminary GPUMD/NEP testing.

- AIMD root: `{aimd_root}`
- Excluded training source: `{training_source}`
- Selection method: {selection_description}
- Total test configurations: {total_frames}
- Frames skipped because OUTCAR labels were unavailable: {skipped_unlabeled_frames}
- No test frame appears in the corresponding training `frame_mapping.jsonl`.
- Labels are streamed from each project's OUTCAR and use the same definitions as Toolkit-generated train.xyz.

## Files

- `test.xyz`: GPUMD/NEP extxyz test set with total energy, forces, and total Virial.
- `frame_mapping.jsonl`: per-frame AIMD project, XDATCAR frame, OUTCAR ionic step, and label provenance.
- `source_ranges.tsv`: frame-number range for each trajectory in the merged test.xyz.
- `skipped_unlabeled_frames.tsv`: frames without usable OUTCAR labels that cannot enter the supervised test set.
- `toolkit.log`: extraction and validation log.

## Source Ranges

{ranges}

## Interpretation Limits

Although these configurations were not used for training, they come from the same AIMD trajectories as the training set. They therefore measure primarily in-distribution interpolation error and are not equivalent to a strictly independent test set from new temperatures, compositions, or evolution pathways.
"""
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
