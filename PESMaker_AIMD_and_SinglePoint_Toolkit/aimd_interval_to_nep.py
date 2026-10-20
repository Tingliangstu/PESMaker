#!/usr/bin/env python3
"""Create a NEP training set by interval-sampling one VASP AIMD run.

The script calls the public ``python -m pesmaker select`` command for
interval selection, then reuses the validated OUTCAR streaming and label
conversion helpers from ``aimd_fps_to_nep.py`` in this toolkit directory.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import re
import subprocess
import sys

from aimd_fps_to_nep import (
    ExtractionStats,
    SelectionEntry,
    ToolkitError,
    extract_train_xyz,
    load_selections,
    parse_nblock,
)


OUTPUT_DIR_NAME = "PESMakerToolkit_AIMD_Interval_to_NEP"
REQUIRED_SELECTION_FILES = ("selected.xyz", "manifest.jsonl")
CONFIGURATION_PATTERN = re.compile(
    r"^\s*(?:Direct|Cartesian)\s+configuration\s*=",
    re.IGNORECASE,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Interval-sample a VASP AIMD XDATCAR and stream matching "
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
        "--interval",
        type=int,
        required=True,
        help="Select every Nth XDATCAR frame (must be positive).",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="First eligible zero-based XDATCAR frame. Default: 0.",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help=(
            "Inclusive last eligible zero-based XDATCAR frame. "
            "Default: the final frame."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_cli_values(
            interval=args.interval,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
        )
        run_toolkit(
            aimd_dir=args.aimd_dir,
            interval=args.interval,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
        )
    except ToolkitError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Error: interrupted by user", file=sys.stderr)
        return 130
    return 0


def run_toolkit(
    *,
    aimd_dir: Path,
    interval: int,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> Path:
    """Run interval selection and streamed OUTCAR label extraction."""
    _validate_cli_values(
        interval=interval,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    aimd_dir = aimd_dir.expanduser().resolve()
    incar, xdatcar, outcar = _validate_inputs(aimd_dir)
    output_dir = aimd_dir / OUTPUT_DIR_NAME
    if output_dir.exists():
        raise ToolkitError(
            f"output directory already exists; refusing to overwrite: {output_dir}"
        )

    frame_count = count_xdatcar_frames(xdatcar)
    expected_source_frames, resolved_end_frame = resolve_source_frames(
        frame_count=frame_count,
        interval=interval,
        start_frame=start_frame,
        end_frame=end_frame,
    )

    output_dir.mkdir(parents=False, exist_ok=False)
    logger = _configure_logger(output_dir / "toolkit.log")
    logger.info("PESMaker AIMD interval to NEP toolkit")
    logger.info("AIMD directory  : %s", aimd_dir)
    logger.info("XDATCAR         : %s", xdatcar)
    logger.info("OUTCAR          : %s", outcar)
    logger.info("Output          : %s", output_dir)
    logger.info("XDATCAR frames  : %d", frame_count)
    logger.info("interval        : %d frame(s)", interval)
    logger.info("start_frame     : %d", start_frame)
    logger.info("end_frame       : %d (inclusive)", resolved_end_frame)
    logger.info("expected frames : %d", len(expected_source_frames))

    try:
        nblock = parse_nblock(incar)
        logger.info("NBLOCK          : %d", nblock)
        config_path = output_dir / "pesmaker_interval.yaml"
        _write_pesmaker_config(
            config_path,
            xdatcar=xdatcar,
            output_dir=output_dir,
            interval=interval,
            start_frame=start_frame,
            max_count=len(expected_source_frames),
        )
        _run_pesmaker_select(config_path, aimd_dir=aimd_dir, logger=logger)
        _verify_selection_outputs(output_dir)

        selections = load_selections(output_dir, nblock=nblock)
        verify_selected_source_frames(selections, expected_source_frames)
        logger.info("Interval selected: %d frame(s)", len(selections))
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


def _validate_cli_values(
    *,
    interval: int,
    start_frame: int,
    end_frame: int | None,
) -> None:
    if interval < 1:
        raise ToolkitError("--interval must be a positive integer")
    if start_frame < 0:
        raise ToolkitError("--start-frame must be zero or a positive integer")
    if end_frame is not None:
        if end_frame < 0:
            raise ToolkitError("--end-frame must be zero or a positive integer")
        if end_frame < start_frame:
            raise ToolkitError("--end-frame must not be smaller than --start-frame")


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


def count_xdatcar_frames(xdatcar: Path) -> int:
    """Count standard VASP XDATCAR configuration markers without loading frames."""
    frame_count = 0
    with xdatcar.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if CONFIGURATION_PATTERN.match(line):
                frame_count += 1
    if frame_count < 1:
        raise ToolkitError(f"XDATCAR contains no configuration frames: {xdatcar}")
    return frame_count


def resolve_source_frames(
    *,
    frame_count: int,
    interval: int,
    start_frame: int,
    end_frame: int | None,
) -> tuple[list[int], int]:
    """Return the exact zero-based XDATCAR frames for inclusive bounds."""
    if frame_count < 1:
        raise ToolkitError("XDATCAR frame count must be positive")
    if start_frame >= frame_count:
        raise ToolkitError(
            f"--start-frame {start_frame} is outside XDATCAR frame range "
            f"0..{frame_count - 1}"
        )
    resolved_end = frame_count - 1 if end_frame is None else end_frame
    if resolved_end >= frame_count:
        raise ToolkitError(
            f"--end-frame {resolved_end} is outside XDATCAR frame range "
            f"0..{frame_count - 1}"
        )
    source_frames = list(range(start_frame, resolved_end + 1, interval))
    if not source_frames:
        raise ToolkitError("the requested interval range selected no XDATCAR frames")
    return source_frames, resolved_end


def _configure_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger(f"pesmaker_aimd_interval_toolkit.{id(path)}")
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


def _write_pesmaker_config(
    path: Path,
    *,
    xdatcar: Path,
    output_dir: Path,
    interval: int,
    start_frame: int,
    max_count: int,
) -> None:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - PESMaker requires PyYAML
        raise ToolkitError("PyYAML is required to generate the interval config") from exc
    config = {
        "project": "aimd_interval_to_nep",
        "sampling": {
            "engine": "none",
            "selection": {
                "method": "interval",
                "trajectory_pattern": str(xdatcar),
                "trajectory_format": "vasp-xdatcar",
                "output_dir": str(output_dir),
                "interval": interval,
                "offset": start_frame,
                "max_count": max_count,
                "separate_trajectories": False,
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


def _verify_selection_outputs(output_dir: Path) -> None:
    missing = [
        name for name in REQUIRED_SELECTION_FILES if not (output_dir / name).is_file()
    ]
    if missing:
        raise ToolkitError(
            "PESMaker select completed without required output file(s): "
            + ", ".join(missing)
        )


def verify_selected_source_frames(
    selections: list[SelectionEntry],
    expected_source_frames: list[int],
) -> None:
    actual_source_frames = [selection.source_frame for selection in selections]
    if actual_source_frames != expected_source_frames:
        mismatch_index = next(
            (
                index
                for index, (actual, expected) in enumerate(
                    zip(actual_source_frames, expected_source_frames)
                )
                if actual != expected
            ),
            min(len(actual_source_frames), len(expected_source_frames)),
        )
        expected_value = (
            expected_source_frames[mismatch_index]
            if mismatch_index < len(expected_source_frames)
            else "<missing>"
        )
        actual_value = (
            actual_source_frames[mismatch_index]
            if mismatch_index < len(actual_source_frames)
            else "<missing>"
        )
        raise ToolkitError(
            "PESMaker interval selection did not match the requested frame range: "
            f"expected {len(expected_source_frames)} frame(s), got "
            f"{len(actual_source_frames)}; first mismatch at selection "
            f"{mismatch_index}: expected {expected_value}, got {actual_value}"
        )


def _log_success(
    logger: logging.Logger,
    stats: ExtractionStats,
    train_path: Path,
) -> None:
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
