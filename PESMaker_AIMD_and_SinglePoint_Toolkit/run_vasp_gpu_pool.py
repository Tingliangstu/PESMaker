#!/usr/bin/env python3
"""Run prepared VASP single-point folders through a local GPU worker pool.

Purpose:
    Discover prepared calculation folders, wait for idle NVIDIA GPUs, and run
    one ``submit.sh`` per GPU.  Each GPU takes another queued folder as soon as
    its current process exits.  This is a standalone helper and does not modify
    PESMaker's package code.

Usage:
    python run_vasp_gpu_pool.py /path/to/labeling --gpus auto --dry-run
    python run_vasp_gpu_pool.py /path/to/labeling --gpus 0,1,2,3 --background
    tail -f /path/to/labeling/gpu_pool_driver.log

Outputs:
    ``gpu_pool_driver.log`` contains output from a background pool controller,
    and ``gpu_pool.pid`` contains its PID while that controller is alive.
    ``gpu_pool_events.jsonl`` below the calculation root records queue events.
    Each calculation folder receives ``gpu_pool.log`` containing stdout and
    stderr from its direct ``bash submit.sh`` execution.  VASP's usual output
    files remain controlled by ``submit.sh`` and VASP itself.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import IO, Sequence

try:
    import fcntl
except ImportError:  # pragma: no cover - the runner is intended for Linux.
    fcntl = None


NORMAL_VASP_FOOTERS = (
    b"General timing and accounting informations for this job",
    b"General timing and accounting information for this job",
)
SCF_FAILURE_MARKER = b"The electronic self-consistency was not achieved in"


class PoolError(RuntimeError):
    """User-facing GPU pool configuration or execution error."""


@dataclass(frozen=True)
class GPUStatus:
    """One row reported by ``nvidia-smi``."""

    index: str
    memory_used_mb: int
    memory_total_mb: int
    utilization_percent: int

    def is_idle(self, *, max_memory_used_mb: int, max_utilization: int) -> bool:
        return (
            self.memory_used_mb <= max_memory_used_mb
            and self.utilization_percent <= max_utilization
        )


@dataclass
class RunningJob:
    """A calculation process currently assigned to a GPU."""

    workdir: Path
    gpu: str
    process: subprocess.Popen[bytes]
    started_at: float
    output_handle: IO[bytes]
    lock_handle: IO[str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run prepared VASP folders with one serial worker per idle NVIDIA GPU."
        )
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Calculation root recursively containing POSCAR and submit.sh files.",
    )
    parser.add_argument(
        "--script-name",
        default="submit.sh",
        help="Per-calculation script executed with Bash. Default: submit.sh.",
    )
    parser.add_argument(
        "--gpus",
        default="auto",
        help="Comma-separated physical GPU indices or 'auto'. Default: auto.",
    )
    parser.add_argument(
        "--max-memory-used-mb",
        type=int,
        default=1024,
        help=(
            "A GPU is externally idle only when used memory is at most this value. "
            "Default: 1024 MB."
        ),
    )
    parser.add_argument(
        "--max-utilization",
        type=int,
        default=10,
        help=(
            "A GPU is externally idle only when utilization is at most this "
            "percentage. Default: 10."
        ),
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=10.0,
        help="Seconds between process/GPU checks. Default: 10.",
    )
    parser.add_argument(
        "--job-log-name",
        default="gpu_pool.log",
        help="Per-job Bash stdout/stderr log name. Default: gpu_pool.log.",
    )
    parser.add_argument(
        "--event-log",
        type=Path,
        default=None,
        help=(
            "JSONL event log. Relative paths are resolved below root. "
            "Default: gpu_pool_events.jsonl."
        ),
    )
    parser.add_argument(
        "--failure-policy",
        choices=("continue", "stop"),
        default="continue",
        help="Continue other queued jobs or stop launching after a failure.",
    )
    parser.add_argument(
        "--accept-exit-zero",
        action="store_true",
        help=(
            "Treat shell exit status 0 as success even without a normal VASP "
            "OUTCAR footer. By default the footer and SCF check are required."
        ),
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=None,
        help="Limit this run to the first N pending jobs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List pending jobs and GPU status without starting processes.",
    )
    parser.add_argument(
        "--background",
        action="store_true",
        help=(
            "Detach the GPU-pool controller from the terminal. Output is appended "
            "to ROOT/gpu_pool_driver.log and its PID is written to "
            "ROOT/gpu_pool.pid."
        ),
    )
    parser.add_argument(
        "--_controller-child",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def parse_gpu_selection(value: str) -> tuple[str, ...] | None:
    """Return selected GPU indices, or ``None`` for automatic selection."""
    if value.strip().lower() == "auto":
        return None
    selected = tuple(part.strip() for part in value.split(",") if part.strip())
    if not selected:
        raise PoolError("--gpus must be 'auto' or a comma-separated GPU list")
    if len(selected) != len(set(selected)):
        raise PoolError("--gpus contains duplicate GPU indices")
    if any(not item.isdigit() for item in selected):
        raise PoolError("--gpus accepts numeric physical GPU indices only")
    return selected


def parse_nvidia_smi(text: str) -> list[GPUStatus]:
    """Parse the CSV emitted by the query used in :func:`query_gpus`."""
    statuses: list[GPUStatus] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            raise PoolError(
                f"unexpected nvidia-smi row {line_number}: {line.strip()}"
            )
        try:
            statuses.append(
                GPUStatus(
                    index=parts[0],
                    memory_used_mb=int(parts[1]),
                    memory_total_mb=int(parts[2]),
                    utilization_percent=int(parts[3]),
                )
            )
        except ValueError as exc:
            raise PoolError(
                f"non-numeric nvidia-smi row {line_number}: {line.strip()}"
            ) from exc
    if not statuses:
        raise PoolError("nvidia-smi reported no NVIDIA GPUs")
    return statuses


def query_gpus() -> list[GPUStatus]:
    """Read physical GPU index, memory use, and utilization."""
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise PoolError("nvidia-smi was not found on PATH") from exc
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise PoolError(f"nvidia-smi failed: {message}") from exc
    return parse_nvidia_smi(result.stdout)


def select_statuses(
    statuses: Sequence[GPUStatus], selected: tuple[str, ...] | None
) -> list[GPUStatus]:
    """Filter statuses while preserving physical GPU order."""
    if selected is None:
        return list(statuses)
    by_index = {status.index: status for status in statuses}
    missing = [index for index in selected if index not in by_index]
    if missing:
        raise PoolError(f"requested GPU indices do not exist: {', '.join(missing)}")
    return [by_index[index] for index in selected]


def discover_jobs(root: Path, script_name: str) -> list[Path]:
    """Find prepared VASP folders containing POSCAR and the requested script."""
    if not root.exists():
        raise PoolError(f"calculation root does not exist: {root}")
    if not root.is_dir():
        raise PoolError(f"calculation root is not a directory: {root}")
    if Path(script_name).name != script_name:
        raise PoolError("--script-name must be one file name, not a path")
    jobs = {
        script.parent.resolve()
        for script in root.rglob(script_name)
        if script.is_file() and (script.parent / "POSCAR").is_file()
    }
    return sorted(jobs, key=lambda path: path.as_posix())


def _file_contains_markers(path: Path, markers: Sequence[bytes]) -> set[bytes]:
    """Find several markers in one streaming pass through a large file."""
    found: set[bytes] = set()
    overlap = max(max((len(marker) for marker in markers), default=1) - 1, 0)
    previous = b""
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            data = previous + chunk
            for marker in markers:
                if marker not in found and marker in data:
                    found.add(marker)
            if len(found) == len(markers):
                break
            previous = data[-overlap:] if overlap else b""
    return found


def outcar_state(workdir: Path) -> str:
    """Classify the final OUTCAR state used for skip/retry decisions."""
    path = workdir / "OUTCAR"
    if not path.is_file():
        return "missing"
    markers = (SCF_FAILURE_MARKER, *NORMAL_VASP_FOOTERS)
    found = _file_contains_markers(path, markers)
    if SCF_FAILURE_MARKER in found:
        return "scf-nonconverged"
    if any(marker in found for marker in NORMAL_VASP_FOOTERS):
        return "complete"
    return "incomplete"


def pending_jobs(jobs: Sequence[Path]) -> tuple[list[Path], list[Path]]:
    """Split discovered folders into pending and already-complete jobs."""
    pending: list[Path] = []
    completed: list[Path] = []
    for job in jobs:
        if outcar_state(job) == "complete":
            completed.append(job)
        else:
            pending.append(job)
    return pending, completed


def _event_log_path(root: Path, configured: Path | None) -> Path:
    if configured is None:
        return root / "gpu_pool_events.jsonl"
    return configured if configured.is_absolute() else root / configured


def record_event(path: Path, event: str, **details: object) -> None:
    """Append one machine-readable event and print a compact screen line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "event": event,
        **details,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()
    summary = " ".join(f"{key}={value}" for key, value in details.items())
    print(f"[{payload['time']}] {event.upper()} {summary}".rstrip(), flush=True)


def _lock_directory() -> Path:
    return Path(tempfile.gettempdir()) / "pesmaker_gpu_pool_locks"


def try_gpu_lock(gpu: str) -> IO[str] | None:
    """Acquire a nonblocking per-GPU lock shared by pool instances."""
    if fcntl is None:
        raise PoolError("GPU locking requires Linux/WSL with the fcntl module")
    directory = _lock_directory()
    directory.mkdir(parents=True, exist_ok=True)
    handle = (directory / f"gpu_{gpu}.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()}\n")
    handle.flush()
    return handle


def release_gpu_lock(handle: IO[str]) -> None:
    if fcntl is not None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    handle.close()


def _pid_is_alive(pid: int) -> bool:
    """Return whether a local process exists, including permission-denied cases."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_pid_file(path: Path) -> int | None:
    """Read a positive PID, treating a missing or malformed file as stale."""
    try:
        value = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    try:
        pid = int(value)
    except ValueError:
        return None
    return pid if pid > 0 else None


def _remove_own_pid_file(path: Path) -> None:
    """Remove the controller PID file only when it still names this process."""
    if _read_pid_file(path) == os.getpid():
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def start_background_controller(
    args: argparse.Namespace, raw_args: Sequence[str]
) -> int:
    """Start one detached copy of this program as the pool controller."""
    _validate_options(args)
    root = args.root.resolve()
    if not root.is_dir():
        raise PoolError(f"calculation root is not a directory: {root}")
    if args.dry_run:
        raise PoolError("--background cannot be combined with --dry-run")
    parse_gpu_selection(args.gpus)
    if not discover_jobs(root, args.script_name):
        raise PoolError(
            f"no folders containing POSCAR and {args.script_name} found below {root}"
        )

    pid_path = root / "gpu_pool.pid"
    log_path = root / "gpu_pool_driver.log"
    old_pid = _read_pid_file(pid_path)
    if old_pid is not None and _pid_is_alive(old_pid):
        raise PoolError(
            f"a GPU-pool controller is already running for this root: PID {old_pid}"
        )

    child_args = [item for item in raw_args if item != "--background"]
    child_args.append("--_controller-child")
    command = [sys.executable, str(Path(__file__).resolve()), *child_args]
    environment = os.environ.copy()
    environment["PESMAKER_GPU_POOL_CONTROLLER"] = "1"

    try:
        with log_path.open("ab") as output_handle:
            header = (
                f"\n[{datetime.now().astimezone().isoformat(timespec='seconds')}] "
                "Starting detached GPU-pool controller\n"
            )
            output_handle.write(header.encode("utf-8"))
            output_handle.flush()
            process = subprocess.Popen(
                command,
                # Keep the caller's directory because the positional root in
                # child_args may be relative to it.
                cwd=Path.cwd(),
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=output_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
    except OSError as exc:
        raise PoolError(f"could not start background controller: {exc}") from exc

    print(f"GPU-pool controller submitted in background: PID={process.pid}")
    print(f"Driver log: {log_path}")
    print(f"PID file  : {pid_path}")
    return 0


def launch_job(
    workdir: Path,
    *,
    gpu: str,
    script_name: str,
    job_log_name: str,
    lock_handle: IO[str],
) -> RunningJob:
    """Start one calculation with the selected physical GPU exposed."""
    output_path = workdir / job_log_name
    output_handle: IO[bytes] | None = None
    try:
        output_handle = output_path.open("ab")
        header = (
            f"\n[{datetime.now().astimezone().isoformat(timespec='seconds')}] "
            f"GPU pool starting on physical GPU {gpu}\n"
        )
        output_handle.write(header.encode("utf-8"))
        output_handle.flush()
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        environment["PESMAKER_GPU_POOL"] = "1"
        process = subprocess.Popen(
            ["bash", script_name],
            cwd=workdir,
            env=environment,
            stdout=output_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except Exception:
        if output_handle is not None:
            output_handle.close()
        release_gpu_lock(lock_handle)
        raise
    return RunningJob(
        workdir=workdir,
        gpu=gpu,
        process=process,
        started_at=time.monotonic(),
        output_handle=output_handle,
        lock_handle=lock_handle,
    )


def _validate_options(args: argparse.Namespace) -> None:
    if args.max_memory_used_mb < 0:
        raise PoolError("--max-memory-used-mb must be zero or positive")
    if not 0 <= args.max_utilization <= 100:
        raise PoolError("--max-utilization must be between 0 and 100")
    if args.poll_seconds <= 0:
        raise PoolError("--poll-seconds must be positive")
    if args.max_jobs is not None and args.max_jobs < 1:
        raise PoolError("--max-jobs must be positive")
    for value, name in (
        (args.script_name, "--script-name"),
        (args.job_log_name, "--job-log-name"),
    ):
        if Path(value).name != value:
            raise PoolError(f"{name} must be one file name, not a path")


def run_pool(args: argparse.Namespace) -> int:
    """Run the queue until all pending jobs finish or launching is stopped."""
    _validate_options(args)
    root = args.root.resolve()
    selected = parse_gpu_selection(args.gpus)
    statuses = select_statuses(query_gpus(), selected)
    jobs = discover_jobs(root, args.script_name)
    if not jobs:
        raise PoolError(
            f"no folders containing POSCAR and {args.script_name} found below {root}"
        )
    pending, completed = pending_jobs(jobs)
    if args.max_jobs is not None:
        pending = pending[: args.max_jobs]

    print(f"Calculation root : {root}")
    print(f"Jobs discovered  : {len(jobs)}")
    print(f"Jobs completed   : {len(completed)}")
    print(f"Jobs pending     : {len(pending)}")
    print(f"GPU candidates   : {', '.join(status.index for status in statuses)}")
    for status in statuses:
        state = "idle" if status.is_idle(
            max_memory_used_mb=args.max_memory_used_mb,
            max_utilization=args.max_utilization,
        ) else "busy"
        print(
            f"  GPU {status.index}: {state}, memory "
            f"{status.memory_used_mb}/{status.memory_total_mb} MB, "
            f"utilization {status.utilization_percent}%"
        )

    if args.dry_run:
        print("Pending job order:")
        for number, job in enumerate(pending, start=1):
            print(f"  {number:6d}  {job}")
        return 0
    if not pending:
        print("No pending jobs need to run.")
        return 0

    event_log = _event_log_path(root, args.event_log)
    queue = deque(pending)
    active: dict[str, RunningJob] = {}
    failed_count = 0
    completed_count = 0
    stop_launching = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop_launching
        stop_launching = True
        print(
            "Stop requested: no new jobs will start; running jobs are left alive.",
            flush=True,
        )

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    try:
        while queue or active:
            changed = False
            for gpu, running in list(active.items()):
                return_code = running.process.poll()
                if return_code is None:
                    continue
                changed = True
                running.output_handle.close()
                release_gpu_lock(running.lock_handle)
                del active[gpu]
                state = outcar_state(running.workdir)
                success = return_code == 0 and (
                    args.accept_exit_zero or state == "complete"
                )
                elapsed = round(time.monotonic() - running.started_at, 1)
                if success:
                    completed_count += 1
                    record_event(
                        event_log,
                        "completed",
                        gpu=gpu,
                        pid=running.process.pid,
                        workdir=str(running.workdir),
                        exit_status=return_code,
                        outcar_state=state,
                        elapsed_seconds=elapsed,
                    )
                else:
                    failed_count += 1
                    record_event(
                        event_log,
                        "failed",
                        gpu=gpu,
                        pid=running.process.pid,
                        workdir=str(running.workdir),
                        exit_status=return_code,
                        outcar_state=state,
                        elapsed_seconds=elapsed,
                    )
                    if args.failure_policy == "stop":
                        stop_launching = True

            if queue and not stop_launching:
                try:
                    latest = select_statuses(query_gpus(), selected)
                except PoolError as exc:
                    print(
                        f"GPU status query failed; will retry: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    time.sleep(args.poll_seconds)
                    continue
                latest_by_gpu = {status.index: status for status in latest}
                for gpu in [status.index for status in statuses]:
                    if not queue or gpu in active:
                        continue
                    status = latest_by_gpu[gpu]
                    if not status.is_idle(
                        max_memory_used_mb=args.max_memory_used_mb,
                        max_utilization=args.max_utilization,
                    ):
                        continue
                    lock_handle = try_gpu_lock(gpu)
                    if lock_handle is None:
                        continue
                    # Recheck after acquiring the cooperative lock to narrow the
                    # race window with non-pool GPU users.
                    try:
                        rechecked = {
                            item.index: item
                            for item in select_statuses(query_gpus(), selected)
                        }[gpu]
                    except PoolError as exc:
                        release_gpu_lock(lock_handle)
                        print(
                            f"GPU status recheck failed; will retry: {exc}",
                            file=sys.stderr,
                            flush=True,
                        )
                        continue
                    if not rechecked.is_idle(
                        max_memory_used_mb=args.max_memory_used_mb,
                        max_utilization=args.max_utilization,
                    ):
                        release_gpu_lock(lock_handle)
                        continue
                    workdir = queue.popleft()
                    try:
                        running = launch_job(
                            workdir,
                            gpu=gpu,
                            script_name=args.script_name,
                            job_log_name=args.job_log_name,
                            lock_handle=lock_handle,
                        )
                    except (OSError, subprocess.SubprocessError) as exc:
                        failed_count += 1
                        changed = True
                        record_event(
                            event_log,
                            "launch-failed",
                            gpu=gpu,
                            workdir=str(workdir),
                            error=str(exc),
                            remaining=len(queue),
                        )
                        if args.failure_policy == "stop":
                            stop_launching = True
                        continue
                    active[gpu] = running
                    changed = True
                    record_event(
                        event_log,
                        "started",
                        gpu=gpu,
                        pid=running.process.pid,
                        workdir=str(workdir),
                        remaining=len(queue),
                    )

            if stop_launching and not active:
                break
            if not changed:
                time.sleep(args.poll_seconds)
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        for running in active.values():
            running.output_handle.close()
            release_gpu_lock(running.lock_handle)

    print("GPU pool finished.")
    print(f"Completed this run : {completed_count}")
    print(f"Failed this run    : {failed_count}")
    print(f"Not started       : {len(queue)}")
    print(f"Event log          : {event_log}")
    return 1 if failed_count or queue else 0


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_args)
    try:
        if args.background and args._controller_child:
            raise PoolError("internal controller mode cannot use --background")
        if args.background:
            return start_background_controller(args, raw_args)
        if args._controller_child:
            pid_path = args.root.resolve() / "gpu_pool.pid"
            try:
                return run_pool(args)
            finally:
                _remove_own_pid_file(pid_path)
        return run_pool(args)
    except PoolError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
