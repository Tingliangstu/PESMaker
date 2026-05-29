# Copyright 2026 Ting Liang and PESMaker development team
# This file is part of PESMaker.
#
# PESMaker is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PESMaker is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PESMaker. If not, see <https://www.gnu.org/licenses/>.
"""Tests for PESMaker configuration parsing."""

from pesmaker.config import io as config_io
from pesmaker.config.schema import PESMakerConfig
from pesmaker.config.io import load_config


def test_config_from_mapping_minimal():
    """Minimal configs should get sensible default workflow engines."""
    config = PESMakerConfig.from_mapping(
        {
            "project": "demo",
            "structures": [{"path": "POSCAR"}],
        }
    )

    assert config.project == "demo"
    assert len(config.structures) == 1
    assert config.labeling.engine == "vasp"
    assert config.training.engine == "nep"


def test_dataset_split_must_sum_to_one():
    """Dataset split ratios must be normalized."""
    try:
        PESMakerConfig.from_mapping(
            {
                "project": "demo",
                "structures": [{"path": "POSCAR"}],
                "dataset": {"split": [0.8, 0.1, 0.2]},
            }
        )
    except ValueError as exc:
        assert "sum to 1.0" in str(exc)
    else:
        raise AssertionError("invalid split should fail")


def test_training_model_alias_selects_engine():
    """The concise `training.model` key should select the training engine."""
    config = PESMakerConfig.from_mapping(
        {
            "project": "demo",
            "structures": [{"path": "POSCAR"}],
            "training": {"model": "mace", "device": "cuda"},
        }
    )

    assert config.training.engine == "mace"
    assert config.training.options == {"device": "cuda"}


def test_generation_accepts_surface_defects_and_job_templates():
    """Extended workflow sections should remain structured config options."""
    config = PESMakerConfig.from_mapping(
        {
            "project": "demo",
            "structures": ["POSCAR"],
            "generation": {
                "supercell": [3, 3, 1],
                "surface": {"vacuum": 30, "axis": 2},
                "defects": {
                    "single_vacancies": {"elements": ["Te"], "max_count": 2}
                },
            },
            "jobs": {
                "machine": "cluster-a",
                "sbatch_templates": {"labeling": "templates/vasp.sh"},
            },
        }
    )

    assert config.generation.supercell == (3, 3, 1)
    assert config.generation.surface["vacuum"] == 30
    assert config.generation.defects["single_vacancies"]["elements"] == ["Te"]
    assert config.jobs.engine == "cluster-a"
    assert config.jobs.options["sbatch_templates"]["labeling"] == "templates/vasp.sh"


def test_generation_accepts_surface_nested_defects_and_perturb():
    """Surface children should be applied on top of the surface structure."""
    config = PESMakerConfig.from_mapping(
        {
            "project": "demo",
            "structures": ["POSCAR"],
            "generation": {
                "supercell": [3, 3, 1],
                "surface": {
                    "vacuum": 30,
                    "defects": {
                        "single_vacancies": {"elements": ["Te"], "max_count": 2}
                    },
                    "perturb": {"pert_num": 5, "format": "vasp"},
                },
            },
        }
    )

    assert config.generation.surface["vacuum"] == 30
    assert config.generation.defects["single_vacancies"]["max_count"] == 2
    assert config.generation.perturb["pert_num"] == 5


def test_generation_accepts_multiple_tasks_and_defect_perturb():
    """Multiple generation tasks should preserve independent operation chains."""
    config = PESMakerConfig.from_mapping(
        {
            "project": "demo",
            "structures": ["POSCAR"],
            "generation": {
                "output_dir": "generated",
                "tasks": [
                    {
                        "name": "surface 331",
                        "supercell": [3, 3, 1],
                        "surface": {
                            "vacuum": 30,
                            "defects": {
                                "single_vacancies": {"elements": ["Te"]},
                                "perturb": {"pert_num": 3},
                            },
                        },
                    },
                    {
                        "name": "bulk_333",
                        "supercell": [3, 3, 3],
                        "perturb": {"pert_num": 2},
                    },
                ],
            },
        }
    )

    assert [task.name for task in config.generation.tasks] == [
        "surface_331",
        "bulk_333",
    ]
    assert config.generation.tasks[0].supercell == (3, 3, 1)
    assert config.generation.tasks[0].defects["single_vacancies"] == {
        "elements": ["Te"]
    }
    assert config.generation.tasks[0].perturb["pert_num"] == 3
    assert config.generation.tasks[1].supercell == (3, 3, 3)
    assert config.generation.tasks[1].perturb["pert_num"] == 2


def test_generation_accepts_inline_surface_defect_keys():
    """A concise surface block can contain defect keys directly."""
    config = PESMakerConfig.from_mapping(
        {
            "project": "demo",
            "structures": ["POSCAR"],
            "generation": {
                "surface": {
                    "vacuum": 30,
                    "single_vacancies": {"elements": ["Te"], "max_count": 2},
                },
            },
        }
    )

    assert config.generation.defects["single_vacancies"]["elements"] == ["Te"]


def test_yaml_duplicate_keys_are_rejected(tmp_path):
    """Duplicate YAML keys should not silently overwrite earlier settings."""
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        """project: demo
structures:
  - POSCAR
generation:
  supercell: [3, 3, 1]
  supercell: [3, 3, 3]
""",
        encoding="utf-8",
    )

    try:
        load_config(config_path)
    except ValueError as exc:
        assert "duplicate YAML key: supercell" in str(exc)
    else:
        raise AssertionError("duplicate YAML keys should fail")


def test_yaml_loading_does_not_require_tomllib(tmp_path, monkeypatch):
    """Python 3.10 users should be able to load YAML without TOML support."""
    monkeypatch.setattr(config_io, "tomllib", None)
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        """project: demo
structures:
  - POSCAR
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.project == "demo"


def test_toml_loading_without_tomllib_has_clear_error(tmp_path, monkeypatch):
    """TOML configs should explain the optional dependency on Python 3.10."""
    monkeypatch.setattr(config_io, "tomllib", None)
    config_path = tmp_path / "run.toml"
    config_path.write_text('project = "demo"\nstructures = ["POSCAR"]\n', encoding="utf-8")

    try:
        load_config(config_path)
    except RuntimeError as exc:
        assert "TOML config files require Python 3.11+" in str(exc)
    else:
        raise AssertionError("TOML without tomllib should fail clearly")


def test_structures_accept_simple_path_list():
    """Users can list structure paths directly without `{path: ...}`."""
    config = PESMakerConfig.from_mapping(
        {
            "project": "demo",
            "structures": ["Te-mp-19.cif", "Te-mp-23.cif"],
        }
    )

    assert [item.path.name for item in config.structures] == [
        "Te-mp-19.cif",
        "Te-mp-23.cif",
    ]


def test_structures_accept_include_patterns(tmp_path, monkeypatch):
    """Users can collect many structures with an `include` glob pattern.

    Args:
        tmp_path: Pytest temporary directory used as a fake project root.
        monkeypatch: Pytest fixture used to run glob expansion from `tmp_path`.
    """
    structure_dir = tmp_path / "initial_structures"
    structure_dir.mkdir()
    (structure_dir / "a.cif").write_text("", encoding="utf-8")
    (structure_dir / "b.cif").write_text("", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    config = PESMakerConfig.from_mapping(
        {
            "project": "demo",
            "structures": {"include": ["initial_structures/*.cif"]},
        }
    )

    assert [item.path.as_posix() for item in config.structures] == [
        "initial_structures/a.cif",
        "initial_structures/b.cif",
    ]
