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

"""Trajectory frame selection for sampled structures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from pesmaker.config.schema import PESMakerConfig
from pesmaker.parsers.ase import read_frames, write_extxyz_many
from pesmaker.results import StageResult
from pesmaker.samplers.gpumd import _resolve_sampling_potential_path


def select_sampling_frames(config: PESMakerConfig) -> StageResult:
    """Select representative MD frames with farthest point sampling."""
    options = config.sampling.options.get("selection", {})
    if not isinstance(options, dict):
        raise ValueError("sampling.selection must be a mapping")
    pattern = str(options.get("trajectory_pattern", "runs/*/sampling/**/movie.xyz"))
    output_dir = Path(str(options.get("output_dir", "selected")))
    min_distance = float(options.get("min_distance", 0.0))
    max_count = options.get("max_count")
    max_count = int(max_count) if max_count is not None else None

    frames = _read_trajectory_frames(pattern)
    sampling_options = {"engine": config.sampling.engine, **config.sampling.options}
    features, descriptor_backend = _selection_features(
        frames,
        options,
        sampling_options=sampling_options,
    )
    print(
        "FPS status       : Descriptor calculation complete; selecting "
        "farthest points. Please wait.",
        flush=True,
    )
    selected_indices, selection_distances = _farthest_point_indices(
        features,
        min_distance=min_distance,
        max_count=max_count,
    )
    selected = [frames[index] for index in selected_indices]

    output_dir.mkdir(parents=True, exist_ok=True)
    features_path = output_dir / "selection_features.npy"
    np.save(features_path, features)
    plot_path = _write_selection_plot(
        features,
        selected_indices,
        selection_distances,
        output_dir=output_dir,
        options=options,
    )
    selected_path = output_dir / "selected.xyz"
    _write_extxyz_many(selected_path, selected)
    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index, (frame_index, atoms, distance) in enumerate(
            zip(selected_indices, selected, selection_distances)
        ):
            manifest.write(
                json.dumps(
                    {
                        "index": index,
                        "source_frame": frame_index,
                        "frame_index": index,
                        "path": str(selected_path),
                        "atom_count": len(atoms),
                        "descriptor": descriptor_backend,
                        "selection_distance": distance,
                    }
                )
                + "\n"
            )
    files = [selected_path, features_path, manifest_path]
    if plot_path is not None:
        files.append(plot_path)
    return StageResult(
        output_dir,
        tuple(files),
        (
            f"Selected {len(selected)} of {len(frames)} MD frame(s) using "
            f"{_selection_descriptor_label(descriptor_backend)}"
        ),
        warnings=tuple(
            _selection_limit_warnings(
                selected_count=len(selected),
                total_count=len(frames),
                min_distance=min_distance,
                max_count=max_count,
            )
        ),
    )


def _read_trajectory_frames(pattern: str):
    return read_frames(pattern)


def _write_extxyz_many(path: Path, frames) -> None:
    write_extxyz_many(path, frames)


def _selection_limit_warnings(
    *,
    selected_count: int,
    total_count: int,
    min_distance: float,
    max_count: int | None,
) -> list[str]:
    if selected_count >= total_count:
        return []
    if max_count is not None and selected_count >= max_count:
        suggested_max_count = max(max_count * 2, 1)
        return [
            (
                "Selection stopped at "
                f"sampling.selection.max_count={max_count}. To keep more "
                "structures, edit your YAML under sampling.selection: "
                f"max_count: {suggested_max_count} or another larger value."
            )
        ]
    if min_distance <= 0:
        return []
    return [
        (
            "Selection stopped because remaining frames are closer than "
            f"min_distance={min_distance:g}. To keep more structures, edit "
            "your YAML under sampling.selection and lower this threshold, "
            f"for example: min_distance: {_suggest_lower_min_distance(min_distance)}."
        )
    ]


def _suggest_lower_min_distance(min_distance: float) -> str:
    return f"{min_distance * 0.6:.3g}"


def _selection_features(
    frames,
    options: dict[str, Any],
    *,
    sampling_options: dict[str, Any],
) -> tuple[np.ndarray, str]:
    descriptor = str(
        options.get("descriptor", _default_selection_descriptor(sampling_options))
    ).lower()
    if descriptor in {"calorine", "nep", "calorine-nep", "calorine_nep"}:
        selection_potential = options.get("potential", options.get("model"))
        potential_options = dict(sampling_options)
        if selection_potential:
            potential_options["potential"] = selection_potential
        potential = _resolve_sampling_potential_path(potential_options)
        if potential is not None and potential.exists():
            potential = str(potential.resolve())
        features = _calorine_nep_structure_features(frames, potential, options)
        return features, "calorine"
    if descriptor in {"mace", "mace-descriptor", "mace_descriptor"}:
        features = _mace_structure_features(
            frames,
            options.get("descriptor_model"),
            options,
        )
        return features, "mace"
    if descriptor in {"simple", "geometry"}:
        return _structure_features(frames), "simple"
    raise ValueError(
        "sampling.selection.descriptor must be 'mace', 'calorine', or 'simple'"
    )


def _default_selection_descriptor(sampling_options: dict[str, Any]) -> str:
    engine = str(sampling_options.get("engine", "")).lower().replace("_", "-")
    if engine in {"mace", "lammps-mace"}:
        return "mace"
    return "calorine"


def _selection_descriptor_label(descriptor_backend: str) -> str:
    if descriptor_backend == "mace":
        return "invariant descriptors output by the MACE model"
    if descriptor_backend == "calorine":
        return "NEP descriptors calculated from the GPUMD potential"
    return "the simple geometry descriptor"


def _calorine_nep_structure_features(
    frames,
    potential: Any,
    options: dict[str, Any],
) -> np.ndarray:
    if not potential:
        raise ValueError(
            "sampling.selection.potential or sampling.potential is required "
            "for Calorine NEP descriptor selection"
        )
    try:
        from calorine.nep import get_descriptors
    except ImportError as exc:
        raise RuntimeError(
            "Calorine NEP descriptor selection requires calorine. Install it "
            'with `python -m pip install ".[selection]"` or `python -m pip '
            "install calorine`."
        ) from exc

    potential_path = Path(str(potential))
    if not potential_path.exists():
        raise ValueError(
            f"Calorine NEP potential file does not exist: {potential_path}"
        )
    _print_descriptor_start(
        engine="GPUMD",
        backend="Calorine NEP descriptors",
        source_label="Potential",
        source_path=potential_path.resolve(),
        frame_count=len(frames),
    )
    pooling = str(options.get("descriptor_pooling", options.get("pooling", "mean")))
    features = []
    for index, atoms in enumerate(frames, start=1):
        descriptors = np.asarray(
            get_descriptors(atoms, model_filename=str(potential_path)),
            dtype=float,
        )
        if descriptors.ndim == 1:
            descriptors = descriptors.reshape(1, -1)
        features.append(_pool_atom_descriptors(descriptors, pooling))
        _print_descriptor_progress(index, len(frames))
    feature_matrix = np.vstack(features)
    _print_descriptor_complete(feature_matrix)
    return feature_matrix


def _pool_atom_descriptors(descriptors: np.ndarray, pooling: str) -> np.ndarray:
    if pooling == "mean":
        return descriptors.mean(axis=0)
    if pooling == "sum":
        return descriptors.sum(axis=0)
    if pooling in {"mean_std", "mean+std"}:
        return np.concatenate([descriptors.mean(axis=0), descriptors.std(axis=0)])
    raise ValueError(
        "sampling.selection.descriptor_pooling must be mean, sum, or mean_std"
    )


def _mace_structure_features(
    frames,
    model: Any,
    options: dict[str, Any],
) -> np.ndarray:
    if not model:
        raise ValueError(
            "sampling.selection.descriptor_model is required for MACE "
            "descriptor selection"
        )
    try:
        from mace.calculators import MACECalculator
    except ImportError as exc:
        raise RuntimeError(
            "MACE descriptor selection requires mace-torch. Install it with "
            "`python -m pip install mace-torch`."
        ) from exc

    model_path = Path(str(model))
    if not model_path.exists() or not model_path.is_file():
        raise ValueError(f"MACE descriptor model does not exist: {model_path}")

    calculator_options: dict[str, Any] = {
        "model_paths": str(model_path.resolve()),
        "device": str(options.get("device", "cuda")),
    }
    if options.get("default_dtype"):
        calculator_options["default_dtype"] = str(options["default_dtype"])
    if options.get("head"):
        calculator_options["head"] = str(options["head"])
    _print_descriptor_start(
        engine="MACE",
        backend="MACECalculator invariant descriptors",
        source_label="Model",
        source_path=model_path.resolve(),
        frame_count=len(frames),
        device=str(calculator_options["device"]),
    )
    calculator = MACECalculator(**calculator_options)

    species = sorted(
        {
            int(number)
            for atoms in frames
            for number in atoms.get_atomic_numbers()
        }
    )
    num_layers = int(options.get("num_layers", -1))
    features = []
    for index, atoms in enumerate(frames, start=1):
        descriptors = calculator.get_descriptors(
            atoms,
            invariants_only=True,
            num_layers=num_layers,
        )
        if isinstance(descriptors, list):
            raise ValueError(
                "MACE descriptor selection supports one descriptor model only"
            )
        descriptors = np.asarray(descriptors, dtype=np.float32)
        if descriptors.ndim != 2 or descriptors.shape[0] != len(atoms):
            raise ValueError(
                "Unexpected MACE descriptor shape "
                f"{descriptors.shape}; expected ({len(atoms)}, feature_count)"
            )
        features.append(
            _pool_mace_descriptors_by_element(
                descriptors,
                atoms.get_atomic_numbers(),
                species,
            )
        )
        _print_descriptor_progress(index, len(frames))
    feature_matrix = np.vstack(features)
    _print_descriptor_complete(feature_matrix)
    return feature_matrix


def _print_descriptor_start(
    *,
    engine: str,
    backend: str,
    source_label: str,
    source_path: Path,
    frame_count: int,
    device: str | None = None,
) -> None:
    print("FPS descriptor calculation")
    print(f"Engine           : {engine}")
    print(f"Backend          : {backend}")
    print(f"{source_label:<17}: {source_path}")
    if device is not None:
        print(f"Device           : {device}")
    print(f"Frames           : {frame_count}")
    print(
        "Status           : Loading the model and calculating descriptors. "
        "This may take some time; please wait.",
        flush=True,
    )


def _print_descriptor_progress(completed: int, total: int) -> None:
    if total < 1:
        return
    interval = max((total + 9) // 10, 1)
    if completed != 1 and completed != total and completed % interval != 0:
        return
    percentage = f"{completed * 100 / total:.1f}".rstrip("0").rstrip(".")
    print(
        f"Descriptor progress: {completed}/{total} frame(s) ({percentage}%)",
        flush=True,
    )


def _print_descriptor_complete(features: np.ndarray) -> None:
    print(
        f"Descriptor matrix: {features.shape[0]} frame(s) x "
        f"{features.shape[1]} feature(s)",
        flush=True,
    )


def _pool_mace_descriptors_by_element(
    descriptors: np.ndarray,
    atomic_numbers: np.ndarray,
    species: list[int],
) -> np.ndarray:
    pooled = []
    for atomic_number in species:
        element_descriptors = descriptors[atomic_numbers == atomic_number]
        if len(element_descriptors) == 0:
            raise ValueError(
                "All trajectory frames must contain the same elements for "
                "MACE descriptor selection; missing atomic number "
                f"{atomic_number}"
            )
        pooled.append(element_descriptors.mean(axis=0))
    return np.concatenate(pooled)


def _structure_features(frames) -> np.ndarray:
    raw = []
    max_length = 0
    for atoms in frames:
        numbers = atoms.get_atomic_numbers().reshape(-1, 1)
        scaled = atoms.get_scaled_positions(wrap=True)
        feature = np.concatenate([numbers, scaled], axis=1).reshape(-1)
        feature = np.concatenate([np.array([len(atoms)]), feature])
        raw.append(feature)
        max_length = max(max_length, len(feature))
    features = np.zeros((len(raw), max_length), dtype=float)
    for index, feature in enumerate(raw):
        features[index, : len(feature)] = feature
    return features


def _farthest_point_indices(
    features: np.ndarray,
    *,
    min_distance: float,
    max_count: int | None,
) -> tuple[list[int], list[float]]:
    if len(features) == 0:
        return [], []
    if max_count is not None and max_count < 1:
        return [], []

    selected = [0]
    selected_distances = [0.0]
    distances = np.linalg.norm(features - features[0], axis=1)
    while True:
        if max_count is not None and len(selected) >= max_count:
            break
        next_index = int(np.argmax(distances))
        next_distance = float(distances[next_index])
        if next_index in selected or next_distance < min_distance:
            break
        selected.append(next_index)
        selected_distances.append(next_distance)
        new_distances = np.linalg.norm(features - features[next_index], axis=1)
        distances = np.minimum(distances, new_distances)
        distances[selected] = 0.0
    return selected, selected_distances


def _write_selection_plot(
    features: np.ndarray,
    selected_indices: list[int],
    selection_distances: list[float],
    *,
    output_dir: Path,
    options: dict[str, Any],
) -> Path | None:
    if not bool(options.get("plot", True)):
        return None
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Selection plotting requires matplotlib. Install it with "
            "`python -m pip install matplotlib` or set sampling.selection.plot: false."
        ) from exc
    _apply_plot_style()

    points = _pca_2d(features)
    plot_path = Path(str(options.get("plot_path", output_dir / "fps_selection.png")))
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_points, ax_distances) = plt.subplots(1, 2, figsize=(11.5, 4.8))
    ax_points.scatter(
        points[:, 0],
        points[:, 1],
        s=float(options.get("all_marker_size", 18)),
        c="#9aa3ad",
        alpha=float(options.get("all_alpha", 0.5)),
        linewidths=0,
        label="all frames",
    )
    if selected_indices:
        selected_points = points[selected_indices]
        ax_points.scatter(
            selected_points[:, 0],
            selected_points[:, 1],
            s=float(options.get("selected_marker_size", 11)),
            c="#d62728",
            alpha=float(options.get("selected_alpha", 0.72)),
            linewidths=0,
            label="selected",
        )
        if bool(options.get("annotate", True)):
            for order, (x_value, y_value) in enumerate(selected_points[:50]):
                ax_points.annotate(
                    str(order),
                    (x_value, y_value),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=float(options.get("annotate_fontsize", 6.5)),
                    color="#4d4d4d",
                    alpha=0.85,
                )
    ax_points.set_title("FPS selection in descriptor PCA space")
    ax_points.set_xlabel("PC1")
    ax_points.set_ylabel("PC2")
    ax_points.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        borderaxespad=0.0,
        handletextpad=0.4,
        columnspacing=1.0,
    )

    ax_distances.plot(
        range(len(selection_distances)),
        selection_distances,
        color="#2b7bbb",
        linewidth=1.6,
    )
    ax_distances.scatter(
        range(len(selection_distances)),
        selection_distances,
        s=10,
        color="#2b7bbb",
        linewidths=0,
    )
    ax_distances.set_title("Distance when selected")
    ax_distances.set_xlabel("Selection order")
    ax_distances.set_ylabel("Nearest-selected distance")

    for ax in (ax_points, ax_distances):
        ax.grid(False)
        ax.tick_params(axis="both", which="both", direction="out")
        for spine in ax.spines.values():
            spine.set_visible(True)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=int(options.get("plot_dpi", 600)))
    plt.close(fig)
    return plot_path


def _apply_plot_style() -> None:
    try:
        import seaborn as sns
    except ImportError:
        import matplotlib.pyplot as plt

        plt.style.use("seaborn-v0_8-ticks")
        return

    sns.set_theme(
        style="ticks",
        context="notebook",
        font_scale=1.08,
        rc={
            "axes.spines.top": True,
            "axes.spines.right": True,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": False,
        },
    )


def _pca_2d(features: np.ndarray) -> np.ndarray:
    if len(features) == 0:
        return np.zeros((0, 2), dtype=float)
    centered = features - np.mean(features, axis=0)
    if centered.shape[0] == 1:
        return np.zeros((1, 2), dtype=float)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    points = centered @ vh[: min(2, len(vh))].T
    if points.shape[1] == 1:
        points = np.column_stack([points[:, 0], np.zeros(len(points))])
    return points[:, :2]
