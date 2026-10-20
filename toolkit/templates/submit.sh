#!/usr/bin/env bash
# Purpose: Run one GPU-VASP single-point calculation.  The GPU pool calls this
#          file in the foreground; a user may submit this one folder manually
#          in the background for an independent test.
# Usage:   GPU-pool child: bash submit.sh
#          Manual background: bash submit.sh --background --gpu 0
#          Manual foreground: bash submit.sh --gpu 0
#          Batch controller (run from its own location):
#            python run_vasp_gpu_pool.py /path/to/labeling --gpus 0,1 --background
# Outputs: Normal VASP files; manual background output in vasp.log; vasp.pid
#          while a manual job is alive; vasp.exitcode and vasp.done/vasp.failed
#          after an attempted calculation.

set -eo pipefail

script_path="$(readlink -f "${BASH_SOURCE[0]}")"
workdir="$(dirname "$script_path")"
cd "$workdir"

vasp_env_file="${VASP_ENV_FILE:-/home/flb/software/VASP/GPU_vasp.6.5.1/ENV}"
vasp_binary="${VASP_BINARY:-/home/flb/software/VASP/GPU_vasp.6.5.1/bin/vasp_std}"
mpi_launcher="${MPI_LAUNCHER:-mpirun}"
mpi_ranks="${MPI_RANKS:-1}"
log_file="${VASP_LOG_FILE:-vasp.log}"
pid_file="${VASP_PID_FILE:-vasp.pid}"
exitcode_file="${VASP_EXITCODE_FILE:-vasp.exitcode}"
done_file="${VASP_DONE_FILE:-vasp.done}"
failed_file="${VASP_FAILED_FILE:-vasp.failed}"
lock_file="${VASP_LOCK_FILE:-.vasp_run.lock}"

run_mode="foreground"
gpu_index=""

usage() {
    echo "Usage: bash submit.sh [--gpu INDEX] [--background|--foreground]"
    echo "  --gpu INDEX    Physical NVIDIA GPU index for a manual run."
    echo "  --background   Detach this one manual VASP job and write vasp.log."
    echo "  --foreground   Run in the current shell (the default and pool mode)."
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)
            if [[ $# -lt 2 ]]; then
                echo "Error: --gpu requires an index." >&2
                exit 2
            fi
            gpu_index="$2"
            shift 2
            ;;
        --background)
            run_mode="background"
            shift
            ;;
        --foreground)
            run_mode="foreground"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -n "$gpu_index" ]]; then
    assigned_gpu="$gpu_index"
else
    assigned_gpu="${CUDA_VISIBLE_DEVICES:-}"
fi
if [[ -z "$assigned_gpu" ]]; then
    echo "Error: select one GPU with --gpu INDEX or CUDA_VISIBLE_DEVICES." >&2
    exit 2
fi
if [[ ! "$assigned_gpu" =~ ^[0-9]+$ ]]; then
    echo "Error: this one-GPU script requires one numeric GPU index: $assigned_gpu" >&2
    exit 2
fi
if [[ ! -r "$vasp_env_file" ]]; then
    echo "Error: VASP environment file is not readable: $vasp_env_file" >&2
    exit 2
fi
if [[ ! -x "$vasp_binary" ]]; then
    echo "Error: VASP executable is not executable: $vasp_binary" >&2
    exit 2
fi
if ! [[ "$mpi_ranks" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: MPI_RANKS must be a positive integer: $mpi_ranks" >&2
    exit 2
fi
for required_input in INCAR POSCAR POTCAR; do
    if [[ ! -s "$required_input" ]]; then
        echo "Error: required input is missing or empty: $workdir/$required_input" >&2
        exit 2
    fi
done
if ! command -v flock >/dev/null 2>&1; then
    echo "Error: flock is required to prevent duplicate runs in one folder." >&2
    exit 2
fi

# The pool must retain a foreground child so it can release the GPU only after
# VASP really exits.  Manual --background mode is deliberately independent.
if [[ "$run_mode" == "background" ]]; then
    if [[ "${PESMAKER_GPU_POOL:-0}" == "1" ]]; then
        echo "Error: --background is forbidden for a GPU-pool child." >&2
        exit 2
    fi
    if [[ -f "$pid_file" ]]; then
        old_pid="$(<"$pid_file")"
        if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
            echo "Error: a process recorded in $pid_file is still running (PID $old_pid)." >&2
            exit 1
        fi
    fi

    child_command=(bash "$script_path" --foreground --gpu "$assigned_gpu")
    nohup "${child_command[@]}" >> "$log_file" 2>&1 < /dev/null &
    child_pid=$!
    printf '%s\n' "$child_pid" > "$pid_file"

    echo "Submitted GPU-VASP in background: PID=$child_pid GPU=$assigned_gpu"
    echo "Log: $workdir/$log_file"
    exit 0
fi

export CUDA_VISIBLE_DEVICES="$assigned_gpu"

# The vendor environment may not be compatible with nounset, so this script
# intentionally uses set -eo pipefail rather than set -euo pipefail.
# Make `conda deactivate` available to non-interactive Bash.
if [[ -n "${CONDA_EXE:-}" ]] && ! declare -F conda >/dev/null 2>&1; then
    conda_base="$(dirname "$(dirname "$CONDA_EXE")")"
    conda_sh="$conda_base/etc/profile.d/conda.sh"

    if [[ ! -r "$conda_sh" ]]; then
        echo "Error: Conda initialization script is not readable: $conda_sh" >&2
        exit 2
    fi

    source "$conda_sh"
fi

source "$vasp_env_file"

# Preserve the assignment made by the pool or --gpu even if ENV changes it.
export CUDA_VISIBLE_DEVICES="$assigned_gpu"

if ! command -v "$mpi_launcher" >/dev/null 2>&1; then
    echo "Error: MPI launcher was not found after loading ENV: $mpi_launcher" >&2
    exit 2
fi
exec 9>"$lock_file"
if ! flock -n 9; then
    echo "Error: another calculation is already running in $workdir" >&2
    exit 1
fi

cleanup_pid_file() {
    if [[ -f "$pid_file" && "$(<"$pid_file")" == "$$" ]]; then
        rm -f -- "$pid_file"
    fi
}
trap cleanup_pid_file EXIT

rm -f -- "$done_file" "$failed_file"

start_human="$(date '+%Y-%m-%d %H:%M:%S %z')"
start_epoch="$(date +%s)"

echo
echo "****************** Start GPU-VASP single point ******************"
echo "Workdir : $workdir"
echo "GPU     : $CUDA_VISIBLE_DEVICES"
echo "Started : $start_human"
echo "Command : $mpi_launcher -np $mpi_ranks $vasp_binary"
echo

set +e
"$mpi_launcher" -np "$mpi_ranks" "$vasp_binary"
vasp_status=$?
set -e

end_human="$(date '+%Y-%m-%d %H:%M:%S %z')"
end_epoch="$(date +%s)"
elapsed_seconds=$((end_epoch - start_epoch))

final_status="$vasp_status"
failure_reason=""
if [[ "$vasp_status" -ne 0 ]]; then
    failure_reason="VASP or MPI exited with status $vasp_status"
elif [[ ! -f OUTCAR ]] || ! grep -Fq \
    "General timing and accounting information" OUTCAR; then
    final_status=3
    failure_reason="OUTCAR does not contain the normal VASP timing footer"
elif grep -Fq "The electronic self-consistency was not achieved in" OUTCAR; then
    final_status=4
    failure_reason="OUTCAR reports electronic self-consistency failure"
fi

printf '%s\n' "$final_status" > "$exitcode_file"
if [[ "$final_status" -eq 0 ]]; then
    printf 'completed=%s\ngpu=%s\nelapsed_seconds=%s\n' \
        "$end_human" "$CUDA_VISIBLE_DEVICES" "$elapsed_seconds" > "$done_file"
else
    printf 'failed=%s\ngpu=%s\nelapsed_seconds=%s\nreason=%s\n' \
        "$end_human" "$CUDA_VISIBLE_DEVICES" "$elapsed_seconds" \
        "$failure_reason" > "$failed_file"
fi

echo
echo "******************* End GPU-VASP single point *******************"
echo "Finished: $end_human"
echo "Elapsed : ${elapsed_seconds} seconds"
echo "VASP exit : $vasp_status"
echo "Final exit: $final_status"
if [[ -n "$failure_reason" ]]; then
    echo "Reason    : $failure_reason"
fi

exit "$final_status"
