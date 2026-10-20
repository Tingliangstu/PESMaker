from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml


TOOLKIT_DIR = (
    Path(__file__).resolve().parents[1]
    / "PESMaker_AIMD_and_SinglePoint_Toolkit"
)
if str(TOOLKIT_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLKIT_DIR))
TOOL_PATH = TOOLKIT_DIR / "aimd_interval_to_nep.py"
SPEC = importlib.util.spec_from_file_location("aimd_interval_to_nep", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tool
SPEC.loader.exec_module(tool)


def test_count_xdatcar_frames_streams_configuration_markers(tmp_path: Path):
    xdatcar = tmp_path / "XDATCAR"
    xdatcar.write_text(
        "header\nDirect configuration=     1\npositions\n"
        " Cartesian configuration=     2\npositions\n",
        encoding="utf-8",
    )
    assert tool.count_xdatcar_frames(xdatcar) == 2


def test_resolve_source_frames_uses_inclusive_end_without_forcing_it():
    frames, resolved_end = tool.resolve_source_frames(
        frame_count=20,
        interval=4,
        start_frame=2,
        end_frame=13,
    )
    assert frames == [2, 6, 10]
    assert resolved_end == 13


def test_resolve_source_frames_defaults_to_final_frame():
    frames, resolved_end = tool.resolve_source_frames(
        frame_count=11,
        interval=5,
        start_frame=0,
        end_frame=None,
    )
    assert frames == [0, 5, 10]
    assert resolved_end == 10


@pytest.mark.parametrize(
    ("interval", "start_frame", "end_frame", "message"),
    [
        (0, 0, None, "interval"),
        (1, -1, None, "start-frame"),
        (1, 5, 4, "end-frame"),
    ],
)
def test_cli_value_validation(interval, start_frame, end_frame, message):
    with pytest.raises(tool.ToolkitError, match=message):
        tool._validate_cli_values(
            interval=interval,
            start_frame=start_frame,
            end_frame=end_frame,
        )


def test_explicit_end_frame_outside_trajectory_is_rejected():
    with pytest.raises(tool.ToolkitError, match="outside XDATCAR frame range"):
        tool.resolve_source_frames(
            frame_count=10,
            interval=2,
            start_frame=0,
            end_frame=10,
        )


def test_generated_interval_yaml_uses_exact_bounds(tmp_path: Path):
    xdatcar = (tmp_path / "XDATCAR").resolve()
    output_dir = (tmp_path / tool.OUTPUT_DIR_NAME).resolve()
    output_dir.mkdir()
    config_path = output_dir / "pesmaker_interval.yaml"
    tool._write_pesmaker_config(
        config_path,
        xdatcar=xdatcar,
        output_dir=output_dir,
        interval=100,
        start_frame=20,
        max_count=5,
    )

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    selection = config["sampling"]["selection"]
    assert selection == {
        "method": "interval",
        "trajectory_pattern": str(xdatcar),
        "trajectory_format": "vasp-xdatcar",
        "output_dir": str(output_dir),
        "interval": 100,
        "offset": 20,
        "max_count": 5,
        "separate_trajectories": False,
    }


def test_manifest_source_frames_must_match_requested_sequence():
    selections = [
        SimpleNamespace(source_frame=2),
        SimpleNamespace(source_frame=6),
        SimpleNamespace(source_frame=10),
    ]
    tool.verify_selected_source_frames(selections, [2, 6, 10])
    with pytest.raises(tool.ToolkitError, match="did not match"):
        tool.verify_selected_source_frames(selections, [2, 6, 11])


def test_existing_output_directory_is_refused(tmp_path: Path):
    for name in ("INCAR", "XDATCAR", "OUTCAR"):
        (tmp_path / name).write_text("", encoding="utf-8")
    (tmp_path / tool.OUTPUT_DIR_NAME).mkdir()
    with pytest.raises(tool.ToolkitError, match="refusing to overwrite"):
        tool.run_toolkit(
            aimd_dir=tmp_path,
            interval=10,
            start_frame=0,
            end_frame=None,
        )
