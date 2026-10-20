from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


TOOLKIT_DIR = (
    Path(__file__).resolve().parents[1]
    / "PESMaker_AIMD_and_SinglePoint_Toolkit"
)
if str(TOOLKIT_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLKIT_DIR))

SCRIPT_PATH = TOOLKIT_DIR / "prepare_aimd_testset.py"
SPEC = importlib.util.spec_from_file_location("prepare_aimd_testset", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tool
SPEC.loader.exec_module(tool)


def make_aimd_project(root: Path, name: str, source: str = "interval") -> Path:
    project = root / name
    project.mkdir()
    for filename in ("INCAR", "XDATCAR", "OUTCAR"):
        (project / filename).write_text("test\n", encoding="utf-8")
    source_dir = tool.SOURCE_OUTPUT_DIRS[source]
    (project / source_dir).mkdir()
    (project / source_dir / "frame_mapping.jsonl").write_text(
        '{"source_frame": 0}\n',
        encoding="utf-8",
    )
    return project


def test_discovery_requires_direct_aimd_project_and_selected_source(tmp_path: Path):
    root = tmp_path / "aimd"
    root.mkdir()
    case_b = make_aimd_project(root, "b_case")
    case_a = make_aimd_project(root, "A_case")
    make_aimd_project(root, "fps_only", source="fps")
    (root / "scripts").mkdir()

    assert tool.discover_projects(root, "interval") == [case_a, case_b]


def test_balanced_selection_is_disjoint_and_spans_time_bins():
    training = set(range(0, 1000, 100))
    selected = tool.select_balanced_unused_frames(
        frame_count=1000,
        excluded_frames=training,
        count=10,
    )

    assert selected == list(range(50, 1000, 100))
    assert not training.intersection(selected)


def test_balanced_selection_rejects_bin_without_unused_frame():
    with pytest.raises(tool.ToolkitError, match="contains no unused frame"):
        tool.select_balanced_unused_frames(
            frame_count=4,
            excluded_frames={0, 1},
            count=2,
        )


def test_existing_nonempty_output_is_refused(tmp_path: Path):
    output = tmp_path / tool.OUTPUT_DIR_NAME
    output.mkdir()
    (output / "keep.txt").write_text("existing\n", encoding="utf-8")

    with pytest.raises(tool.ToolkitError, match="refusing to overwrite"):
        tool._prepare_empty_output_directory(output)


def test_existing_empty_output_is_accepted(tmp_path: Path):
    output = tmp_path / tool.OUTPUT_DIR_NAME
    output.mkdir()
    tool._prepare_empty_output_directory(output)
    assert output.is_dir()


def test_output_directory_name_must_not_be_a_path():
    with pytest.raises(tool.ToolkitError, match="one simple directory name"):
        tool._validate_output_dir_name("nested/testset")


def test_parser_all_unused_is_mutually_exclusive_with_fixed_count():
    with pytest.raises(SystemExit):
        tool.build_parser().parse_args(
            [
                "--aimd-root",
                "root",
                "--training-source",
                "interval",
                "--all-unused",
                "--count-per-trajectory",
                "20",
            ]
        )
