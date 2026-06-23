"""Tests for NEP training setup and plots."""

from __future__ import annotations

import json

import numpy as np

from pesmaker.cli import main


def test_train_setup_writes_gpumd_nep_input_from_train_xyz(tmp_path, monkeypatch):
    """NEP setup should infer elements and write documented default weights."""
    train_xyz = tmp_path / "train.xyz"
    _write_train_xyz(train_xyz)
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        f"""project: nep_single
training:
  model: nep
  output_dir: {(tmp_path / 'training').as_posix()}
  dataset: {train_xyz.as_posix()}
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    assert main(["train-setup", str(config_path)]) == 0

    nep_in = (tmp_path / "training" / "nep.in").read_text(encoding="utf-8")
    assert "type          2 Te Pb" in nep_in
    assert "version       4" in nep_in
    assert "cutoff        8 4" in nep_in
    assert "lambda_e      1.0" in nep_in
    assert "lambda_f      1.0" in nep_in
    assert "lambda_v      0.1" in nep_in
    manifest = json.loads(
        (tmp_path / "training" / "training_manifest.jsonl").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["workdir"] == str(tmp_path / "training")


def test_two_step_nep_training_prepares_step2_after_step1_finishes(
    tmp_path,
    monkeypatch,
):
    """Two-step training should switch lambda weights after step1 has nep.txt."""
    train_xyz = tmp_path / "train.xyz"
    _write_train_xyz(train_xyz)
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        f"""project: nep_two_step
training:
  model: nep
  output_dir: {(tmp_path / 'training').as_posix()}
  dataset: {train_xyz.as_posix()}
  two_step: true
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    assert main(["train-setup", str(config_path)]) == 0
    step1 = tmp_path / "training" / "step1"
    step1_nep = (step1 / "nep.in").read_text(encoding="utf-8")
    assert "lambda_e      0.2" in step1_nep
    assert "lambda_f      2" in step1_nep
    assert "lambda_v      0.1" in step1_nep
    (step1 / "keep.dat").write_text("copied\n", encoding="utf-8")
    (step1 / "nep.txt").write_text("trained model\n", encoding="utf-8")

    assert main(["train-setup", str(config_path)]) == 0

    step2 = tmp_path / "training" / "step2"
    step2_nep = (step2 / "nep.in").read_text(encoding="utf-8")
    assert (step2 / "keep.dat").read_text(encoding="utf-8") == "copied\n"
    assert "lambda_e      2" in step2_nep
    assert "lambda_f      1" in step2_nep
    assert "lambda_v      1" in step2_nep
    manifest = json.loads(
        (tmp_path / "training" / "training_manifest.jsonl").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["workdir"] == str(step2)
    assert manifest["training_step"] == "step2"


def test_next_two_step_nep_advances_to_step2_after_step1_output(
    tmp_path,
    monkeypatch,
    capsys,
):
    """Smart next should preview step1, then build and preview step2."""
    train_xyz = tmp_path / "train.xyz"
    _write_train_xyz(train_xyz)
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        f"""project: nep_two_step_next
training:
  model: nep
  output_dir: {(tmp_path / 'training').as_posix()}
  dataset: {train_xyz.as_posix()}
  two_step: true
jobs:
  submit_command: sbatch
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    assert main(["next", str(config_path)]) == 0
    first_output = capsys.readouterr().out
    assert (tmp_path / "training" / "step1" / "submit.sh").is_file()
    assert "Prepared step1 training folder for nep" in first_output
    state_path = tmp_path / ".pesmaker" / "nep_two_step_next" / "next_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert "training_step1" in state["dry_runs"]

    (tmp_path / "training" / "step1" / "nep.txt").write_text(
        "trained model\n",
        encoding="utf-8",
    )
    assert main(["next", str(config_path)]) == 0
    second_output = capsys.readouterr().out

    step2 = tmp_path / "training" / "step2"
    assert (step2 / "submit.sh").is_file()
    assert "Prepared step2 training folder for nep" in second_output
    assert "Submit training jobs" in second_output
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert "training_step2" in state["dry_runs"]
    manifest = json.loads(
        (tmp_path / "training" / "training_manifest.jsonl").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["workdir"] == str(step2)


def test_plot_train_writes_nep_training_figures(tmp_path, monkeypatch, capsys):
    """`pesmaker plot train` should write high-resolution training plots."""
    _write_training_outputs(tmp_path)
    monkeypatch.chdir(tmp_path)

    assert main(["plot", "train"]) == 0
    output = capsys.readouterr().out

    assert (tmp_path / "plot" / "nep_train.png").is_file()
    assert (tmp_path / "plot" / "nep_parity.png").is_file()
    assert "Summary:" in output
    assert "Total generations : 3" in output
    assert "Quantity" in output
    assert "Energy" in output
    assert "Force" in output
    assert "Stress" in output
    assert "meV/atom" in output
    assert "meV/A" in output
    assert "GPa" in output


def test_nep_plot_axes_are_closed_and_scaled():
    """NEP plots should use closed frames, log loss axes, and parity limits."""
    import matplotlib.pyplot as plt

    from pesmaker.plot.nep import (
        ENERGY_COLOR,
        ParityData,
        TOTAL_LOSS_COLOR,
        _loss_colors,
        _plot_loss_panel,
        _plot_marginal_parity,
        _plot_simple_parity,
    )

    loss = np.array(
        [
            [1, 1.0, 0.4, 0.5, 0.1],
            [10, 0.5, 0.2, 0.25, 0.05],
            [100, 0.2, 0.1, 0.11, 0.02],
        ]
    )
    fig, ax = plt.subplots()
    _plot_loss_panel(ax, loss)
    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"
    assert all(spine.get_visible() for spine in ax.spines.values())
    plt.close(fig)
    assert TOTAL_LOSS_COLOR != ENERGY_COLOR
    assert _loss_colors(["Total", "Energy"], [])[0] != _loss_colors(
        ["Total", "Energy"], []
    )[1]

    panel = ParityData(
        true=np.array([-1.0, 0.0, 2.0]),
        pred=np.array([-0.8, 0.1, 2.4]),
        title="Energy",
        xlabel="DFT",
        ylabel="NEP",
        mae_scale=1.0,
        rmse_scale=1.0,
        unit="unit",
        decimals=2,
        color="#2878B5",
    )
    fig, ax = plt.subplots()
    _plot_simple_parity(ax, panel)
    assert ax.get_xlim() == ax.get_ylim()
    assert ax.get_aspect() == "auto"
    assert all(spine.get_visible() for spine in ax.spines.values())
    plt.close(fig)

    fig, ax = plt.subplots()
    _plot_marginal_parity(ax, panel)
    assert len(fig.axes) == 3
    assert fig.axes[0].get_xlim() == fig.axes[0].get_ylim()
    assert fig.axes[0].get_aspect() == "auto"
    assert all(patch.get_edgecolor()[3] == 0 for patch in fig.axes[1].patches)
    assert all(not spine.get_visible() for spine in fig.axes[1].spines.values())
    assert all(not spine.get_visible() for spine in fig.axes[2].spines.values())
    assert not fig.axes[0].spines["top"].get_visible()
    assert not fig.axes[0].spines["right"].get_visible()
    plt.close(fig)


def _write_train_xyz(path):
    path.write_text(
        """2
Lattice="3 0 0 0 3 0 0 0 3" Properties=species:S:1:pos:R:3:force:R:3 energy=-1.0
Te 0 0 0 0.1 0.0 0.0
Pb 1 1 1 0.0 0.1 0.0
""",
        encoding="utf-8",
    )


def _write_training_outputs(path):
    loss = np.array(
        [
            [1, 1.0, 0.4, 0.5, 0.1],
            [2, 0.5, 0.2, 0.25, 0.05],
            [3, 0.2, 0.1, 0.11, 0.02],
        ]
    )
    energy = np.array([[-1.01, -1.0], [-1.95, -2.0], [-2.98, -3.0]])
    force = np.array(
        [
            [0.1, 0.0, 0.0, 0.11, 0.01, 0.0],
            [0.0, 0.2, 0.0, 0.02, 0.18, 0.01],
            [0.0, 0.0, 0.3, -0.01, 0.02, 0.32],
        ]
    )
    stress = np.array(
        [
            [1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 1.1, 1.9, 3.1, 0.1, 0.1, 0.4],
            [2.0, 3.0, 4.0, 0.2, 0.3, 0.4, 2.1, 2.9, 4.2, 0.3, 0.3, 0.5],
        ]
    )
    np.savetxt(path / "loss.out", loss)
    np.savetxt(path / "energy_train.out", energy)
    np.savetxt(path / "force_train.out", force)
    np.savetxt(path / "stress_train.out", stress)
