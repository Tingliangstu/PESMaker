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
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np

from pesmaker.config.schema import PESMakerConfig
from pesmaker.results import StageResult
from pesmaker.samplers.gpumd import _resolve_sampling_potential_path
from pesmaker.structures import write_structure


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
    features, descriptor_backend = _selection_features(
        frames,
        options,
        sampling_options=config.sampling.options,
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
    selected_files = []
    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index, (frame_index, atoms, distance) in enumerate(
            zip(selected_indices, selected, selection_distances)
        ):
            frame_path = output_dir / f"selected_{index:06d}.xyz"
            write_structure(atoms, frame_path, fmt="extxyz")
            selected_files.append(frame_path)
            manifest.write(
                json.dumps(
                    {
                        "index": index,
                        "source_frame": frame_index,
                        "path": str(frame_path),
                        "descriptor": descriptor_backend,
                        "selection_distance": distance,
                    }
                )
                + "\n"
            )
    files = [selected_path, features_path, *selected_files, manifest_path]
    if plot_path is not None:
        files.append(plot_path)
    return StageResult(
        output_dir,
        tuple(files),
        f"Selected {len(selected)} of {len(frames)} MD frame(s)",
    )


def _read_trajectory_frames(pattern: str):
    try:
        from ase.io import read
    except ImportError as exc:
        raise RuntimeError("Trajectory selection requires ASE") from exc

    paths = [Path(path) for path in sorted(glob(pattern, recursive=True))]
    if not paths and Path(pattern).exists():
        paths = [Path(pattern)]
    frames = []
    for path in paths:
        items = read(path, index=":")
        if not isinstance(items, list):
            items = [items]
        frames.extend(items)
    if not frames:
        raise ValueError(f"no trajectory frames matched pattern: {pattern}")
    return frames


def _write_extxyz_many(path: Path, frames) -> None:
    try:
        from ase.io import write
    except ImportError as exc:
        raise RuntimeError("Writing extxyz requires ASE") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    write(path, frames, format="extxyz")


def _selection_features(
    frames,
    options: dict[str, Any],
    *,
    sampling_options: dict[str, Any],
) -> tuple[np.ndarray, str]:
    descriptor = str(options.get("descriptor", "calorine")).lower()
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
    if descriptor in {"simple", "geometry"}:
        return _structure_features(frames), "simple"
    raise ValueError("sampling.selection.descriptor must be 'calorine' or 'simple'")


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
    pooling = str(options.get("descriptor_pooling", options.get("pooling", "mean")))
    features = []
    for atoms in frames:
        descriptors = np.asarray(
            get_descriptors(atoms, model_filename=str(potential_path)),
            dtype=float,
        )
        if descriptors.ndim == 1:
            descriptors = descriptors.reshape(1, -1)
        features.append(_pool_atom_descriptors(descriptors, pooling))
    return np.vstack(features)


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

    points = _pca_2d(features)
    plot_path = Path(str(options.get("plot_path", output_dir / "fps_selection.png")))
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_points, ax_distances) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax_points.scatter(points[:, 0], points[:, 1], s=18, c="#9aa0a6", label="all frames")
    if selected_indices:
        selected_points = points[selected_indices]
        ax_points.scatter(
            selected_points[:, 0],
            selected_points[:, 1],
            s=42,
            c="#d62728",
            label="selected",
        )
        for order, (x_value, y_value) in enumerate(selected_points[:50]):
            ax_points.annotate(str(order), (x_value, y_value), fontsize=7)
    ax_points.set_title("FPS selection in descriptor PCA space")
    ax_points.set_xlabel("PC1")
    ax_points.set_ylabel("PC2")
    ax_points.legend(frameon=False)

    ax_distances.plot(range(len(selection_distances)), selection_distances, marker="o")
    ax_distances.set_title("Distance when selected")
    ax_distances.set_xlabel("Selection order")
    ax_distances.set_ylabel("Nearest-selected distance")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return plot_path


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
