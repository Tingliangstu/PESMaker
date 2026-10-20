from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write


TOOL_PATH = (
    Path(__file__).resolve().parents[1]
    / "PESMaker_AIMD_and_SinglePoint_Toolkit"
    / "aimd_fps_to_nep.py"
)
SPEC = importlib.util.spec_from_file_location("aimd_fps_to_nep", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tool
SPEC.loader.exec_module(tool)


def test_parse_nblock_defaults_and_uses_last_active_value(tmp_path: Path):
    incar = tmp_path / "INCAR"
    incar.write_text(
        "NBLOCK = 2 ! old setting\n# NBLOCK = 50\n nblock = 5\n",
        encoding="utf-8",
    )
    assert tool.parse_nblock(incar) == 5


def test_parse_nblock_defaults_to_one(tmp_path: Path):
    incar = tmp_path / "INCAR"
    incar.write_text("ENCUT = 500\n", encoding="utf-8")
    assert tool.parse_nblock(incar) == 1


def test_validate_geometry_accepts_periodic_wrap():
    selected = Atoms(
        "H",
        scaled_positions=[[0.99999, 0.2, 0.3]],
        cell=np.eye(3) * 10.0,
        pbc=True,
    )
    labeled = Atoms(
        "H",
        scaled_positions=[[-0.00001, 0.2, 0.3]],
        cell=np.eye(3) * 10.0,
        pbc=True,
    )
    result = tool.validate_geometry(selected, labeled)
    assert result.max_position_difference < 1.0e-10


def test_validate_geometry_rejects_atom_order_mismatch():
    selected = Atoms("HO", positions=[[0, 0, 0], [1, 0, 0]], cell=np.eye(3) * 5)
    labeled = Atoms("OH", positions=[[0, 0, 0], [1, 0, 0]], cell=np.eye(3) * 5)
    with pytest.raises(tool.ToolkitError, match="element order"):
        tool.validate_geometry(selected, labeled)


def test_extract_labels_converts_stress_to_gpumd_virial():
    atoms = Atoms("H", positions=[[0, 0, 0]], cell=np.eye(3) * 2.0, pbc=True)
    stress = np.array([1.0, 2.0, 3.0, 0.4, 0.5, 0.6])
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=-1.9,
        free_energy=-2.0,
        forces=np.array([[0.1, 0.2, 0.3]]),
        stress=stress,
    )
    energy, forces, virial = tool.extract_labels(atoms)
    assert energy == pytest.approx(-2.0)
    assert forces.shape == (1, 3)
    assert np.allclose(virial, -8.0 * atoms.get_stress(voigt=False))


def test_load_selections_maps_source_frame_with_nblock(tmp_path: Path):
    output = tmp_path
    frames = [
        Atoms("H", positions=[[0, 0, 0]], cell=np.eye(3), pbc=True),
        Atoms("H", positions=[[0.1, 0, 0]], cell=np.eye(3), pbc=True),
    ]
    write(output / "selected.xyz", frames, format="extxyz")
    records = [{"source_frame": 1}, {"source_frame": 4}]
    (output / "manifest.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    selections = tool.load_selections(output, nblock=5)
    assert [selection.ionic_step for selection in selections] == [10, 25]


def test_load_selections_rejects_variable_cell(tmp_path: Path):
    frames = [
        Atoms("H", positions=[[0, 0, 0]], cell=np.eye(3), pbc=True),
        Atoms("H", positions=[[0, 0, 0]], cell=np.eye(3) * 1.01, pbc=True),
    ]
    write(tmp_path / "selected.xyz", frames, format="extxyz")
    (tmp_path / "manifest.jsonl").write_text(
        '{"source_frame": 0}\n{"source_frame": 1}\n',
        encoding="utf-8",
    )
    with pytest.raises(tool.ToolkitError, match="not a fixed-cell trajectory"):
        tool.load_selections(tmp_path, nblock=1)


def test_nep_frame_is_ase_readable_and_contains_required_labels(tmp_path: Path):
    atoms = Atoms("H", positions=[[0.1, 0.2, 0.3]], cell=np.eye(3) * 2, pbc=True)
    forces = np.array([[0.4, 0.5, 0.6]])
    virial = np.arange(9, dtype=float).reshape(3, 3)
    path = tmp_path / "train.xyz"
    with path.open("w", encoding="utf-8") as handle:
        tool._write_nep_frame(
            handle,
            atoms=atoms,
            energy=-1.25,
            forces=forces,
            virial=virial,
        )
    loaded = read(path, format="extxyz")
    assert loaded.info["Energy"] == pytest.approx(-1.25)
    assert np.allclose(loaded.arrays["force"], forces)
    assert np.allclose(np.asarray(loaded.info["Virial"]).reshape(3, 3), virial)


def test_cli_value_validation():
    with pytest.raises(tool.ToolkitError, match="max-count"):
        tool._validate_cli_values(0, 0.0)
    with pytest.raises(tool.ToolkitError, match="min-distance"):
        tool._validate_cli_values(1, -0.1)


def test_generated_fps_yaml_uses_absolute_paths(tmp_path: Path):
    xdatcar = (tmp_path / "XDATCAR").resolve()
    output_dir = (tmp_path / tool.OUTPUT_DIR_NAME).resolve()
    output_dir.mkdir()
    config_path = output_dir / "pesmaker_fps.yaml"
    tool._write_pesmaker_config(
        config_path,
        xdatcar=xdatcar,
        output_dir=output_dir,
        max_count=25,
        min_distance=0.0,
    )

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["sampling"]["engine"] == "none"
    selection = config["sampling"]["selection"]
    assert selection == {
        "method": "fps",
        "trajectory_pattern": str(xdatcar),
        "trajectory_format": "vasp-xdatcar",
        "output_dir": str(output_dir),
        "max_count": 25,
        "min_distance": 0.0,
        "separate_trajectories": False,
        "plot": True,
    }


def test_existing_output_directory_is_refused(tmp_path: Path):
    for name in ("INCAR", "XDATCAR", "OUTCAR"):
        (tmp_path / name).write_text("", encoding="utf-8")
    (tmp_path / tool.OUTPUT_DIR_NAME).mkdir()
    with pytest.raises(tool.ToolkitError, match="refusing to overwrite"):
        tool.run_toolkit(aimd_dir=tmp_path, max_count=1, min_distance=0.0)


def _labeled_hydrogen(position: float, energy: float) -> Atoms:
    atoms = Atoms(
        "H",
        positions=[[position, 0, 0]],
        cell=np.eye(3) * 5.0,
        pbc=True,
    )
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=energy,
        free_energy=energy,
        forces=np.array([[position, 0.0, 0.0]]),
        stress=np.zeros(6),
    )
    return atoms


def test_extract_train_xyz_streams_only_selected_steps(tmp_path: Path, monkeypatch):
    frames = [
        _labeled_hydrogen(0.0, -1.0),
        _labeled_hydrogen(0.1, -1.1),
        _labeled_hydrogen(0.2, -1.2),
    ]
    selections = [
        tool.SelectionEntry(0, 0, 1, frames[0].copy()),
        tool.SelectionEntry(1, 2, 3, frames[2].copy()),
    ]
    outcar = tmp_path / "OUTCAR"
    outcar.touch()

    def fake_outcar_frames(path):
        assert Path(path) == outcar
        yield from frames

    monkeypatch.setattr(tool, "_iter_outcar_frames", fake_outcar_frames)
    stats = tool.extract_train_xyz(
        outcar=outcar,
        selections=selections,
        output_dir=tmp_path,
        logger=logging.getLogger("test-streaming-success"),
    )

    assert stats.written == 2
    assert (tmp_path / "train.xyz").is_file()
    assert not (tmp_path / "train.xyz.partial").exists()
    mappings = [
        json.loads(line)
        for line in (tmp_path / "frame_mapping.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert [record["ionic_step"] for record in mappings] == [1, 3]
    assert [record["source_frame"] for record in mappings] == [0, 2]


def test_extract_failure_keeps_partial_but_not_train_xyz(tmp_path: Path, monkeypatch):
    atoms = Atoms("H", positions=[[0, 0, 0]], cell=np.eye(3) * 5.0, pbc=True)
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=-1.0,
        free_energy=-1.0,
        forces=np.zeros((1, 3)),
    )
    selection = tool.SelectionEntry(0, 0, 1, atoms.copy())
    outcar = tmp_path / "OUTCAR"
    outcar.touch()
    monkeypatch.setattr(tool, "_iter_outcar_frames", lambda path: iter([atoms]))

    with pytest.raises(tool.ToolkitError, match="no stress label"):
        tool.extract_train_xyz(
            outcar=outcar,
            selections=[selection],
            output_dir=tmp_path,
            logger=logging.getLogger("test-streaming-failure"),
        )
    assert not (tmp_path / "train.xyz").exists()
    assert (tmp_path / "train.xyz.partial").exists()
