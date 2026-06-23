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
"""NEP training plot commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pesmaker.plot.result import PlotResult
from pesmaker.plot.style import apply_plot_style


@dataclass(frozen=True)
class ParityData:
    """Flattened data used by a parity panel."""

    true: np.ndarray
    pred: np.ndarray
    title: str
    xlabel: str
    ylabel: str
    mae_scale: float
    rmse_scale: float
    unit: str
    decimals: int
    color: str


ENERGY_COLOR = "#2878B5"
FORCE_COLOR = "#2F9E44"
TENSOR_COLOR = "#F28E2B"
TOTAL_LOSS_COLOR = "#4C72B0"
REG_LOSS_COLORS = ("#8172B3", "#8C564B")


def plot_nep_training(
    source_dir: Path | None = None,
    *,
    output_dir: Path | None = None,
    dpi: int = 650,
) -> PlotResult:
    """Write NEP training diagnostic figures from GPUMD output files."""
    source = _resolve_training_source(source_dir or Path("."))
    output = output_dir or Path("plot")
    output.mkdir(parents=True, exist_ok=True)

    energy = _load_matrix(source / "energy_train.out")
    force = _load_matrix(source / "force_train.out")
    stress_path = source / "stress_train.out"
    stress_label = "stress"
    stress_unit = "GPa"
    stress = None
    if stress_path.exists():
        stress = _load_matrix(stress_path)
        stress = _filter_invalid_tensor_rows(stress)
    elif (source / "virial_train.out").exists():
        stress = _load_matrix(source / "virial_train.out")
        stress = _filter_invalid_tensor_rows(stress)
        stress_label = "virial"
        stress_unit = "eV"

    panels = _parity_panels(energy, force, stress, stress_label, stress_unit)
    files: list[Path] = []
    if (source / "loss.out").exists():
        files.append(_write_train_overview(source, output, panels, dpi=dpi))
    files.append(_write_parity_with_marginals(output, panels, dpi=dpi))
    return PlotResult(
        output,
        tuple(files),
        f"Wrote {len(files)} NEP training plot(s) from {source}",
    )


def _resolve_training_source(path: Path) -> Path:
    if _has_training_outputs(path):
        return path
    candidates = [
        path / "training" / "step2",
        path / "training" / "step1",
        path / "training",
    ]
    for candidate in candidates:
        if _has_training_outputs(candidate):
            return candidate
    raise ValueError(
        "could not find NEP training outputs. Expected energy_train.out and "
        "force_train.out in the current directory, training/, training/step1, "
        "or training/step2."
    )


def _has_training_outputs(path: Path) -> bool:
    return (path / "energy_train.out").is_file() and (path / "force_train.out").is_file()


def _load_matrix(path: Path) -> np.ndarray:
    if not path.exists():
        raise ValueError(f"required NEP output file is missing: {path}")
    data = np.loadtxt(path)
    return np.atleast_2d(data)


def _filter_invalid_tensor_rows(data: np.ndarray) -> np.ndarray:
    if data.size == 0:
        return data
    columns = min(12, data.shape[1])
    valid = ~np.any(np.abs(data[:, :columns]) >= 1e6, axis=1)
    return data[valid]


def _parity_panels(
    energy: np.ndarray,
    force: np.ndarray,
    tensor: np.ndarray | None,
    tensor_label: str,
    tensor_unit: str,
) -> list[ParityData]:
    panels = [
        ParityData(
            true=energy[:, 1].reshape(-1),
            pred=energy[:, 0].reshape(-1),
            title="Energy",
            xlabel="DFT energy (eV/atom)",
            ylabel="NEP energy (eV/atom)",
            mae_scale=1000.0,
            rmse_scale=1000.0,
            unit="meV/atom",
            decimals=2,
            color=ENERGY_COLOR,
        ),
        ParityData(
            true=force[:, 3:6].reshape(-1),
            pred=force[:, 0:3].reshape(-1),
            title="Force",
            xlabel=r"DFT force (eV/$\mathrm{\AA}$)",
            ylabel=r"NEP force (eV/$\mathrm{\AA}$)",
            mae_scale=1000.0,
            rmse_scale=1000.0,
            unit=r"meV/$\mathrm{\AA}$",
            decimals=2,
            color=FORCE_COLOR,
        ),
    ]
    if tensor is not None and tensor.size and tensor.shape[1] >= 12:
        panels.append(
            ParityData(
                true=tensor[:, 6:12].reshape(-1),
                pred=tensor[:, 0:6].reshape(-1),
                title=tensor_label.capitalize(),
                xlabel=f"DFT {tensor_label} ({tensor_unit})",
                ylabel=f"NEP {tensor_label} ({tensor_unit})",
                mae_scale=1.0,
                rmse_scale=1.0,
                unit=tensor_unit,
                decimals=4 if tensor_unit == "GPa" else 3,
                color=TENSOR_COLOR,
            )
        )
    return panels


def _write_train_overview(
    source: Path,
    output: Path,
    panels: list[ParityData],
    *,
    dpi: int,
) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    apply_plot_style()
    loss = _load_matrix(source / "loss.out")
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.8))
    _plot_loss_panel(axes[0, 0], loss, panels)
    _label_panel(axes[0, 0], 0)
    for index, (ax, panel) in enumerate(zip(axes.flat[1:], panels), start=1):
        _plot_simple_parity(ax, panel)
        _label_panel(ax, index)
    if len(panels) < 3:
        axes[1, 1].axis("off")
    fig.subplots_adjust(
        top=0.94,
        bottom=0.10,
        left=0.10,
        right=0.985,
        hspace=0.34,
        wspace=0.30,
    )
    path = output / "nep_train.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_loss_panel(
    ax,
    loss: np.ndarray,
    panels: list[ParityData] | None = None,
) -> None:
    x = loss[:, 0]
    if x[0] == 100:
        labels = ["Total", "L1-Reg", "L2-Reg", "Energy", "Force", "Virial"]
        columns = range(1, min(loss.shape[1], 7))
    else:
        labels = ["Total", "Energy", "Force", "Virial"]
        columns = range(1, min(loss.shape[1], 5))
    ax.set_xlabel("Generation")
    colors = _loss_colors(labels, panels or [])
    for column, label, color in zip(columns, labels, colors):
        values = loss[:, column]
        mask = (x > 0.0) & (values > 0.0)
        if np.any(mask):
            ax.plot(x[mask], values[mask], linewidth=1.9, color=color, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss")
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.set_box_aspect(1)
    _close_axes(ax)


def _loss_colors(labels: list[str], panels: list[ParityData]) -> list[str]:
    panel_colors = {panel.title.lower(): panel.color for panel in panels}
    colors = {
        "total": TOTAL_LOSS_COLOR,
        "l1-reg": REG_LOSS_COLORS[0],
        "l2-reg": REG_LOSS_COLORS[1],
        "energy": panel_colors.get("energy", ENERGY_COLOR),
        "force": panel_colors.get("force", FORCE_COLOR),
        "virial": panel_colors.get("virial", panel_colors.get("stress", TENSOR_COLOR)),
        "stress": panel_colors.get("stress", TENSOR_COLOR),
    }
    return [colors.get(label.lower(), TOTAL_LOSS_COLOR) for label in labels]


def _plot_simple_parity(ax, panel: ParityData) -> None:
    xmin, xmax = _limits(panel.true, panel.pred, padding=0.06)
    ax.scatter(panel.true, panel.pred, s=18, color=panel.color, alpha=0.38, linewidths=0)
    ax.plot([xmin, xmax], [xmin, xmax], color="#7f7f7f", linestyle="--", linewidth=1.6)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)
    ax.set_xlabel(panel.xlabel)
    ax.set_ylabel(panel.ylabel)
    ax.set_title(panel.title)
    _add_metric_text(ax, panel, x=0.05, y=0.95)
    _close_axes(ax)


def _write_parity_with_marginals(
    output: Path,
    panels: list[ParityData],
    *,
    dpi: int,
) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    apply_plot_style()
    fig, axes = plt.subplots(1, len(panels), figsize=(4.1 * len(panels), 3.95))
    if len(panels) == 1:
        axes = [axes]
    for index, (ax, panel) in enumerate(zip(axes, panels)):
        _plot_marginal_parity(ax, panel)
        _label_panel(ax, index, x=-0.16, y=1.06)
    fig.subplots_adjust(
        top=0.84,
        bottom=0.18,
        left=0.07,
        right=0.985,
        wspace=0.28,
    )
    path = output / "nep_parity.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_marginal_parity(ax, panel: ParityData) -> None:
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    xmin, xmax = _limits(panel.true, panel.pred, padding=0.05)
    ax.scatter(
        panel.true,
        panel.pred,
        s=28,
        c=panel.color,
        alpha=0.34,
        edgecolors="none",
        rasterized=True,
    )
    ax.plot([xmin, xmax], [xmin, xmax], color="#8c8c8c", linestyle="--", linewidth=2.0)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)
    ax.set_xlabel(panel.xlabel)
    ax.set_ylabel(panel.ylabel)
    _add_metric_text(ax, panel, x=0.05, y=0.95)
    _open_axes(ax)

    divider = make_axes_locatable(ax)
    ax_top = divider.append_axes("top", size="18%", pad=0.06, sharex=ax)
    ax_right = divider.append_axes("right", size="18%", pad=0.06, sharey=ax)
    ax_top.hist(panel.true, bins=34, color=panel.color, alpha=0.5, edgecolor="#777777")
    ax_right.hist(
        panel.pred,
        bins=34,
        orientation="horizontal",
        color=panel.color,
        alpha=0.5,
        edgecolor="#777777",
    )
    _clean_marginal_axis(ax_top, axis="x")
    _clean_marginal_axis(ax_right, axis="y")


def _clean_marginal_axis(ax, *, axis: str) -> None:
    if axis == "x":
        ax.tick_params(axis="x", labelbottom=False, bottom=False)
        ax.tick_params(axis="y", left=False, labelleft=False)
    else:
        ax.tick_params(axis="y", labelleft=False, left=False)
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
    _open_axes(ax)
    ax.grid(False)


def _add_metric_text(ax, panel: ParityData, *, x: float, y: float) -> None:
    mae = _mae(panel.pred, panel.true) * panel.mae_scale
    rmse = _rmse(panel.pred, panel.true) * panel.rmse_scale
    r2 = _r2(panel.true, panel.pred)
    ax.text(
        x,
        y,
        f"$R^2 = {r2:.4f}$\n"
        f"MAE = {mae:.{panel.decimals}f} {panel.unit}\n"
        f"RMSE = {rmse:.{panel.decimals}f} {panel.unit}",
        transform=ax.transAxes,
        fontsize=10.5,
        va="top",
        ha="left",
    )


def _limits(true: np.ndarray, pred: np.ndarray, *, padding: float) -> tuple[float, float]:
    data_min = float(min(np.min(true), np.min(pred)))
    data_max = float(max(np.max(true), np.max(pred)))
    data_range = data_max - data_min
    if data_range == 0.0:
        data_range = 1.0
    pad = padding * data_range
    return data_min - pad, data_max + pad


def _rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def _mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))


def _r2(true: np.ndarray, pred: np.ndarray) -> float:
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _close_axes(ax) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)


def _open_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)


def _label_panel(ax, index: int, *, x: float = -0.16, y: float = 1.07) -> None:
    ax.text(
        x,
        y,
        f"({chr(97 + index)})",
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
