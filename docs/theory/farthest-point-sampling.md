# Farthest Point Sampling

PESMaker uses farthest point sampling (FPS) to choose representative MD frames
from a trajectory. The goal is to keep a compact set of structures that covers
the descriptor space without labeling every sampled frame with DFT.

## Basic Idea

Each trajectory frame is mapped to a feature vector:

$$
x_i \in \mathbb{R}^d
$$

where \(x_i\) is the descriptor for frame \(i\). PESMaker chooses the
production descriptor from the sampling engine:

- GPUMD sampling uses Calorine NEP descriptors calculated with the GPUMD
  sampling potential.
- MACE sampling uses invariant descriptors output by the native MACE model
  configured as `sampling.selection.descriptor_model`.

Users do not need to select the descriptor backend manually.

FPS starts from one frame, then repeatedly selects the frame whose nearest
distance to the already selected set is largest:

$$
i^* = \arg\max_i \min_{j \in S} \|x_i - x_j\|_2
$$

where \(S\) is the set of already selected frames. This picks frames that are
far from what has already been selected, so the final set spreads across the
trajectory descriptor space.

The method is a greedy coverage rule. Random selection can miss rare regions of
a trajectory, especially when most MD frames stay near one basin. FPS instead
keeps the frame that currently improves descriptor-space coverage the most.
This makes it useful for reducing many correlated MD frames into a smaller DFT
labeling set.

FPS is only as meaningful as the descriptor. If the descriptor does not capture
the chemistry or distortion that matters for the target potential, far-apart
points in descriptor space may not be the most important DFT labels. For
production runs, inspect the selected structures and the PCA plot instead of
treating the selector as fully automatic.

## Stop Criteria

PESMaker stops selection when either condition is met:

- `max_count` selected frames have been kept;
- the next farthest frame is closer than `min_distance` to the selected set.

Use `max_count` to control labeling cost. Use `min_distance` to avoid spending
DFT calculations on frames that are nearly redundant in descriptor space.
`max_count` is optional; if omitted, FPS keeps selecting until the distance rule
stops it or all frames have been selected.

## Descriptor Invariance And Pooling

For MACE, PESMaker calls:

```python
calculator.get_descriptors(
    atoms,
    invariants_only=True,
    num_layers=-1,
)
```

`invariants_only=True` is appropriate for FPS because the distance should not
change when a physically identical environment is rotated. Equivariant
components rotate with the structure and would introduce orientation-dependent
distance. PESMaker uses every interaction layer, averages atom descriptors
separately for each element, and concatenates those element vectors in atomic
number order. This follows the structure-level descriptor construction used by
MACE's own fine-tuning selection workflow.

NEP descriptors are atom-level descriptors. PESMaker converts them into one
structure-level vector before FPS:

- `descriptor_pooling: mean`: average atom descriptors, the default;
- `descriptor_pooling: sum`: sum atom descriptors;
- `descriptor_pooling: mean_std`: concatenate descriptor means and standard
  deviations.

`mean` is usually the safest first choice because it is less sensitive to atom
count. `mean_std` can preserve more information about local-environment spread,
but it increases descriptor dimensionality.

## Practical Use

Minimal selection settings:

```yaml
sampling:
  engine: gpumd
  potential: /path/to/nep.txt
  selection:
    trajectory_pattern: sampling/**/movie.xyz
    output_dir: selected
    min_distance: 0.2
    max_count: 200  # optional cap
    plot: true
```

For MACE:

```yaml
sampling:
  engine: mace
  selection:
    trajectory_pattern: sampling/**/*.lammpstrj
    output_dir: selected
    descriptor_model: /path/to/native-mace.model
    min_distance: 0.0
    max_count: 200
```

The MACE descriptor model is the native model loaded by ASE. It is separate
from the MLIAP-exported model used by LAMMPS.

Run:

```bash
pesmaker select run.yaml
```

The output `selected/fps_selection.png` shows a PCA projection of all frames and
selected frames. Use it as a sanity check: selected frames should be spread over
the cloud rather than concentrated in one small region.

PESMaker also writes `selected/selection_features.npy`. This file stores the
actual descriptor matrix used by FPS before PCA:

```text
shape = (number_of_md_frames, descriptor_dimension)
```

It is not a structure file and does not go into VASP. It is for diagnostics and
reproducibility: reload it with `numpy.load`, rerun PCA or clustering, compare
different selection thresholds, and check whether the descriptor space separates
physically distinct MD regions. The selected structures themselves are stored in
the combined `selected/selected.xyz`; the manifest records which frames were
kept and lets later SCF setup split them into single-point jobs.

References:

- [MACE descriptor extraction](https://mace-docs.readthedocs.io/en/latest/guide/descriptors.html)
- [MACE ASE calculator](https://mace-docs.readthedocs.io/en/latest/guide/ase.html)
