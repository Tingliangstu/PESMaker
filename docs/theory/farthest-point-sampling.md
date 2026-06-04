# Farthest Point Sampling

PESMaker uses farthest point sampling (FPS) to choose representative MD frames
from a trajectory. The goal is to keep a compact set of structures that covers
the descriptor space without labeling every sampled frame with DFT.

## Basic Idea

Each trajectory frame is mapped to a feature vector:

$$
x_i \in \mathbb{R}^d
$$

where \(x_i\) is the descriptor for frame \(i\). In production active-learning
runs, PESMaker can use Calorine NEP descriptors. For quick debugging,
`descriptor: simple` uses a small geometry-based feature vector.

FPS starts from one frame, then repeatedly selects the frame whose nearest
distance to the already selected set is largest:

$$
i^* = \arg\max_i \min_{j \in S} \|x_i - x_j\|_2
$$

where \(S\) is the set of already selected frames. This picks frames that are
far from what has already been selected, so the final set spreads across the
trajectory descriptor space.

## Stop Criteria

PESMaker stops selection when either condition is met:

- `max_count` selected frames have been kept;
- the next farthest frame is closer than `min_distance` to the selected set.

Use `max_count` to control labeling cost. Use `min_distance` to avoid spending
DFT calculations on frames that are nearly redundant in descriptor space.

## Descriptor Pooling

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
  selection:
    trajectory_pattern: sampling/**/movie.xyz
    output_dir: selected
    descriptor: calorine
    potential: /path/to/nep.txt
    min_distance: 0.2
    max_count: 200
    plot: true
```

Run:

```bash
pesmaker select run.yaml
```

The output `selected/fps_selection.png` shows a PCA projection of all frames and
selected frames. Use it as a sanity check: selected frames should be spread over
the cloud rather than concentrated in one small region.
