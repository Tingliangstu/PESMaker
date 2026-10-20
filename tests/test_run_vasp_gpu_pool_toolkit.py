"""Tests for the standalone VASP GPU worker-pool helper.

Purpose:
    Verify GPU parsing, job discovery, OUTCAR restart logic, and dry-run safety.
Usage:
    Run ``python -m pytest tests/test_run_vasp_gpu_pool_toolkit.py -q``.
Outputs:
    Pytest pass/fail diagnostics only; temporary calculation trees are removed
    automatically by pytest.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


TOOLKIT_DIR = (
    Path(__file__).resolve().parents[1]
    / "PESMaker_AIMD_and_SinglePoint_Toolkit"
)
SCRIPT_PATH = TOOLKIT_DIR / "run_vasp_gpu_pool.py"
SUBMIT_TEMPLATE_PATH = TOOLKIT_DIR / "templates" / "submit.sh"
SPEC = importlib.util.spec_from_file_location("run_vasp_gpu_pool", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tool
SPEC.loader.exec_module(tool)


def make_job(root: Path, name: str) -> Path:
    workdir = root / name
    workdir.mkdir(parents=True)
    (workdir / "POSCAR").write_text("structure\n", encoding="utf-8")
    (workdir / "submit.sh").write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    return workdir


def test_parse_nvidia_smi_rows():
    statuses = tool.parse_nvidia_smi("0, 512, 24576, 0\n2, 4096, 24576, 75\n")

    assert statuses == [
        tool.GPUStatus("0", 512, 24576, 0),
        tool.GPUStatus("2", 4096, 24576, 75),
    ]
    assert statuses[0].is_idle(max_memory_used_mb=1024, max_utilization=10)
    assert not statuses[1].is_idle(
        max_memory_used_mb=1024, max_utilization=10
    )


def test_parse_gpu_selection_and_status_filtering():
    statuses = [
        tool.GPUStatus("0", 0, 100, 0),
        tool.GPUStatus("1", 0, 100, 0),
        tool.GPUStatus("2", 0, 100, 0),
    ]

    assert tool.parse_gpu_selection("auto") is None
    assert tool.parse_gpu_selection("2,0") == ("2", "0")
    assert [item.index for item in tool.select_statuses(statuses, ("2", "0"))] == [
        "2",
        "0",
    ]

    with pytest.raises(tool.PoolError, match="do not exist"):
        tool.select_statuses(statuses, ("3",))


def test_discover_jobs_requires_poscar_and_submit_script(tmp_path: Path):
    job_b = make_job(tmp_path, "b/job")
    job_a = make_job(tmp_path, "a/job")
    template = tmp_path / "templates"
    template.mkdir()
    (template / "submit.sh").write_text("template\n", encoding="utf-8")

    assert tool.discover_jobs(tmp_path, "submit.sh") == [job_a, job_b]


def test_outcar_state_prioritizes_scf_failure(tmp_path: Path):
    job = make_job(tmp_path, "job")
    outcar = job / "OUTCAR"

    assert tool.outcar_state(job) == "missing"
    outcar.write_text("partial output\n", encoding="utf-8")
    assert tool.outcar_state(job) == "incomplete"
    outcar.write_text(
        "General timing and accounting informations for this job\n",
        encoding="utf-8",
    )
    assert tool.outcar_state(job) == "complete"
    outcar.write_text(
        "The electronic self-consistency was not achieved in\n"
        "General timing and accounting informations for this job\n",
        encoding="utf-8",
    )
    assert tool.outcar_state(job) == "scf-nonconverged"


def test_pending_jobs_skip_only_normal_complete_outcar(tmp_path: Path):
    complete = make_job(tmp_path, "complete")
    pending = make_job(tmp_path, "pending")
    (complete / "OUTCAR").write_text(
        "General timing and accounting informations for this job\n",
        encoding="utf-8",
    )

    pending_result, completed_result = tool.pending_jobs([complete, pending])

    assert pending_result == [pending]
    assert completed_result == [complete]


def test_dry_run_lists_pending_jobs_without_writing_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    complete = make_job(tmp_path, "complete")
    pending = make_job(tmp_path, "pending")
    (complete / "OUTCAR").write_text(
        "General timing and accounting informations for this job\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        tool,
        "query_gpus",
        lambda: [tool.GPUStatus("0", 100, 24576, 0)],
    )

    result = tool.main([str(tmp_path), "--gpus", "0", "--dry-run"])

    assert result == 0
    output = capsys.readouterr().out
    assert "Jobs discovered  : 2" in output
    assert "Jobs completed   : 1" in output
    assert "Jobs pending     : 1" in output
    assert str(pending) in output
    assert not (tmp_path / "gpu_pool_events.jsonl").exists()


def test_invalid_threshold_is_rejected(tmp_path: Path, monkeypatch, capsys):
    make_job(tmp_path, "job")
    monkeypatch.setattr(
        tool,
        "query_gpus",
        lambda: [tool.GPUStatus("0", 0, 100, 0)],
    )

    result = tool.main(
        [str(tmp_path), "--max-utilization", "101", "--dry-run"]
    )

    assert result == 2
    assert "--max-utilization" in capsys.readouterr().err


def test_launch_marks_submit_script_as_gpu_pool_child(tmp_path: Path, monkeypatch):
    workdir = make_job(tmp_path, "job")
    captured: dict[str, object] = {}

    class FakeProcess:
        pid = 1234

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    lock_handle = (tmp_path / "gpu.lock").open("a+", encoding="utf-8")

    running = tool.launch_job(
        workdir,
        gpu="2",
        script_name="submit.sh",
        job_log_name="gpu_pool.log",
        lock_handle=lock_handle,
    )
    running.output_handle.close()
    lock_handle.close()

    environment = captured["env"]
    assert environment["CUDA_VISIBLE_DEVICES"] == "2"
    assert environment["PESMAKER_GPU_POOL"] == "1"
    assert captured["command"] == ["bash", "submit.sh"]


def test_background_controller_starts_detached_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    make_job(tmp_path, "job")
    captured: dict[str, object] = {}

    class FakeProcess:
        pid = 4321

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    result = tool.main([str(tmp_path), "--gpus", "0,1", "--background"])

    assert result == 0
    command = captured["command"]
    assert "--background" not in command
    assert "--_controller-child" in command
    assert captured["stdin"] is subprocess.DEVNULL
    assert captured["start_new_session"] is True
    assert captured["cwd"] == Path.cwd()
    assert (tmp_path / "gpu_pool.pid").read_text(encoding="utf-8") == "4321\n"
    assert (tmp_path / "gpu_pool_driver.log").is_file()
    assert "PID=4321" in capsys.readouterr().out


def test_background_controller_rejects_dry_run(tmp_path: Path, capsys):
    result = tool.main([str(tmp_path), "--dry-run", "--background"])

    assert result == 2
    assert "cannot be combined" in capsys.readouterr().err


def test_submit_template_has_separate_pool_and_manual_modes():
    text = SUBMIT_TEMPLATE_PATH.read_text(encoding="utf-8")

    assert "bash submit.sh --background --gpu 0" in text
    assert 'environment["PESMAKER_GPU_POOL"]' not in text
    assert '"${PESMAKER_GPU_POOL:-0}" == "1"' in text
    assert 'run_mode="foreground"' in text
    assert 'nohup "${child_command[@]}"' in text


def test_submit_template_initializes_conda_before_oneapi_environment():
    text = SUBMIT_TEMPLATE_PATH.read_text(encoding="utf-8")

    conda_guard = '[[ -n "${CONDA_EXE:-}" ]] && ! declare -F conda'
    conda_source = 'source "$conda_sh"'
    vasp_source = 'source "$vasp_env_file"'
    assert conda_guard in text
    assert conda_source in text
    assert text.index(conda_source) < text.index(vasp_source)
