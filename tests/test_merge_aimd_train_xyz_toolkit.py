from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from ase import Atoms
from ase.io import read


TOOLKIT_DIR = (
    Path(__file__).resolve().parents[1]
    / "toolkit"
)
if str(TOOLKIT_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLKIT_DIR))

MERGE_PATH = TOOLKIT_DIR / "merge_aimd_train_xyz.py"
MERGE_SPEC = importlib.util.spec_from_file_location(
    "merge_aimd_train_xyz", MERGE_PATH
)
assert MERGE_SPEC is not None and MERGE_SPEC.loader is not None
merge_tool = importlib.util.module_from_spec(MERGE_SPEC)
sys.modules[MERGE_SPEC.name] = merge_tool
MERGE_SPEC.loader.exec_module(merge_tool)

FPS_PATH = TOOLKIT_DIR / "aimd_fps_to_nep.py"
FPS_SPEC = importlib.util.spec_from_file_location("merge_test_fps_tool", FPS_PATH)
assert FPS_SPEC is not None and FPS_SPEC.loader is not None
fps_tool = importlib.util.module_from_spec(FPS_SPEC)
sys.modules[FPS_SPEC.name] = fps_tool
FPS_SPEC.loader.exec_module(fps_tool)


def make_aimd_directory(root: Path, name: str) -> Path:
    directory = root / name
    directory.mkdir()
    for filename in ("INCAR", "XDATCAR", "OUTCAR"):
        (directory / filename).write_text("test\n", encoding="utf-8")
    return directory


def write_train_xyz(path: Path, energies: list[float], *, atom_count: int = 2):
    path.parent.mkdir()
    atoms = Atoms(
        "H" * atom_count,
        positions=np.arange(atom_count * 3, dtype=float).reshape(atom_count, 3),
        cell=np.eye(3) * 10.0,
        pbc=True,
    )
    with path.open("w", encoding="utf-8") as handle:
        for energy in energies:
            fps_tool._write_nep_frame(
                handle,
                atoms=atoms,
                energy=energy,
                forces=np.zeros((atom_count, 3)),
                virial=np.eye(3) * energy,
            )


def test_source_mode_maps_to_exact_official_directories():
    assert (
        merge_tool.SOURCE_MODES["interval"].output_directory
        == "PESMakerToolkit_AIMD_Interval_to_NEP"
    )
    assert (
        merge_tool.SOURCE_MODES["fps"].output_directory
        == "PESMakerToolkit_AIMD_FPS_to_NEP"
    )


def test_discovery_uses_only_direct_vasp_aimd_children(tmp_path: Path):
    root = tmp_path / "aimd"
    root.mkdir()
    valid_b = make_aimd_directory(root, "b_case")
    valid_a = make_aimd_directory(root, "A_case")
    (root / "scripts").mkdir()
    nested = root / "container" / "nested"
    nested.mkdir(parents=True)
    for filename in ("INCAR", "XDATCAR", "OUTCAR"):
        (nested / filename).write_text("test\n", encoding="utf-8")

    assert merge_tool.discover_aimd_directories(root) == [valid_a, valid_b]


def test_missing_selected_source_is_rejected_without_output(tmp_path: Path):
    root = tmp_path / "aimd"
    root.mkdir()
    directory = make_aimd_directory(root, "case")
    write_train_xyz(
        directory
        / "PESMakerToolkit_AIMD_Interval_to_NEP"
        / "train.xyz",
        [-1.0],
    )

    with pytest.raises(merge_tool.MergeError, match="lack the selected fps"):
        merge_tool.run_merge(aimd_root=root, source="fps")
    assert not (root / "Merged_FPS_NEP_TrainXYZ").exists()


def test_merge_interval_sources_end_to_end(tmp_path: Path):
    root = tmp_path / "aimd"
    root.mkdir()
    case_b = make_aimd_directory(root, "b_case")
    case_a = make_aimd_directory(root, "A_case")
    source_directory = "PESMakerToolkit_AIMD_Interval_to_NEP"
    write_train_xyz(case_a / source_directory / "train.xyz", [-1.0, -2.0])
    write_train_xyz(case_b / source_directory / "train.xyz", [-3.0], atom_count=3)

    train_path = merge_tool.run_merge(aimd_root=root, source="interval")
    output = root / "Merged_Interval_NEP_TrainXYZ"

    assert train_path == output / "train.xyz"
    assert {path.name for path in output.iterdir()} == {
        "train.xyz",
        "source_ranges.tsv",
        "README.md",
        "merge.log",
    }
    frames = read(train_path, index=":", format="extxyz")
    assert [frame.info["Energy"] for frame in frames] == [-1.0, -2.0, -3.0]
    assert [len(frame) for frame in frames] == [2, 2, 3]

    with (output / "source_ranges.tsv").open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle, delimiter="\t"))
    assert [record["aimd_directory"] for record in records] == ["A_case", "b_case"]
    assert [record["frame_count"] for record in records] == ["2", "1"]
    assert [record["merged_start_0based"] for record in records] == ["0", "2"]
    assert [record["merged_end_0based"] for record in records] == ["1", "2"]
    readme = (output / "README.md").read_text(encoding="utf-8")
    assert "Sampling source: `interval`" in readme
    assert "0–1" in readme
    assert "3" in readme


def test_invalid_source_dataset_does_not_create_formal_train(tmp_path: Path):
    root = tmp_path / "aimd"
    root.mkdir()
    directory = make_aimd_directory(root, "case")
    train_xyz = (
        directory
        / "PESMakerToolkit_AIMD_Interval_to_NEP"
        / "train.xyz"
    )
    train_xyz.parent.mkdir()
    train_xyz.write_text(
        "1\n"
        'Lattice="10 0 0 0 10 0 0 0 10" Energy=-1 '
        'Properties=species:S:1:pos:R:3:force:R:3 pbc="T T T"\n'
        "H 0 0 0 0 0 0\n",
        encoding="utf-8",
    )

    with pytest.raises(merge_tool.MergeError, match="no Virial label"):
        merge_tool.run_merge(aimd_root=root, source="interval")
    output = root / "Merged_Interval_NEP_TrainXYZ"
    assert output.is_dir()
    assert not (output / "train.xyz").exists()
    assert (output / "merge.log").is_file()


def test_existing_output_directory_is_refused(tmp_path: Path):
    root = tmp_path / "aimd"
    root.mkdir()
    make_aimd_directory(root, "case")
    (root / "Merged_Interval_NEP_TrainXYZ").mkdir()
    with pytest.raises(merge_tool.MergeError, match="refusing to overwrite"):
        merge_tool.run_merge(aimd_root=root, source="interval")
