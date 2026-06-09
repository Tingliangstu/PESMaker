# `pesmaker select`

`select` chooses representative MD frames.

Normal users can let [`next`](next.md) run this stage after GPUMD has written
`movie.xyz`.

## Use

```bash
pesmaker select run.yaml
```

## Minimal YAML

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

## What It Does

PESMaker reads trajectory frames, builds one descriptor vector per frame, then
uses farthest point sampling to keep frames that spread across descriptor
space.

`min_distance` and `max_count` are two stop rules:

- `min_distance`: stop when the next farthest frame is still closer than this
  distance to the selected set. This avoids choosing very similar structures.
- `max_count`: optional cap on how many frames to keep. Omit it if you only
  want the distance rule to decide how many structures are different enough.

For example, `min_distance: 0.2` and `max_count: 200` means "keep at most 200
frames, but stop earlier if the remaining frames are too similar." The distance
is measured in descriptor space, not in Angstrom.

For quick debugging, use:

```yaml
descriptor: simple
```

For production selection, use Calorine NEP descriptors.

## Outputs

```text
selected/
  selected.xyz
  manifest.jsonl
  selection_features.npy
  fps_selection.png
```

`selected.xyz` is the combined extxyz file containing all selected frames.
PESMaker no longer writes one `selected_000000.xyz` file per frame. SCF setup
uses `selected/manifest.jsonl` to split frames from `selected.xyz` when it
prepares single-point job folders.

`selection_features.npy` is a NumPy array with shape
`(number_of_md_frames, descriptor_dimension)`. Row `i` is the descriptor vector
used for MD frame `i` before FPS selection. It is useful for debugging and
post-analysis: you can reload it with `numpy.load`, reproduce descriptor-space
PCA plots, compare different `min_distance` values, or check whether your
descriptor separates different trajectory regions.

`fps_selection.png` is the diagnostic plot. PESMaker uses a seaborn-styled
plot: all MD frames are shown as light points, while selected frames are drawn
smaller on top so you can see how they sit inside the full cloud.

## Next Step

```bash
pesmaker scf-setup run.yaml
```

If `labeling.input_manifest` is omitted, `next` uses
`selected/manifest.jsonl` after selection.
