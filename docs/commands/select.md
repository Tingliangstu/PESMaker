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
  engine: gpumd
  potential: /path/to/nep.txt
  selection:
    trajectory_pattern: sampling/**/movie.xyz
    output_dir: selected
    min_distance: 0.2
    max_count: 200
    plot: true
```

## What It Does

PESMaker reads trajectory frames, builds one descriptor vector per frame, then
uses farthest point sampling to keep frames that spread across descriptor
space.

The descriptor backend follows `sampling.engine` automatically:

- `gpumd`: Calorine calculates NEP descriptors with `sampling.potential`;
- `mace`, `lammps-mace`, or `lammps_mace`: ASE's `MACECalculator` calculates
  invariant MACE descriptors with `sampling.selection.descriptor_model`.

The terminal summary prints which model descriptor was used. You do not need
to set `sampling.selection.descriptor`.

Descriptor inference can take time for a long trajectory. PESMaker prints the
active engine, descriptor backend, model or potential path, frame count, MACE
device, and progress at regular intervals. For example:

```text
FPS descriptor calculation
Engine           : MACE
Backend          : MACECalculator invariant descriptors
Model            : /path/to/mace-omat-0-small.model
Device           : cuda
Frames           : 1501
Status           : Loading the model and calculating descriptors. This may take some time; please wait.
Descriptor progress: 1/1501 frame(s) (0.1%)
Descriptor progress: 151/1501 frame(s) (10.1%)
...
Descriptor progress: 1501/1501 frame(s) (100%)
Descriptor matrix: 1501 frame(s) x 512 feature(s)
FPS status       : Descriptor calculation complete; selecting farthest points. Please wait.
```

GPUMD prints the same progress block with `Engine: GPUMD`, the Calorine NEP
backend, and the NEP potential path. Output is flushed immediately so the
terminal remains informative during model loading and descriptor inference.

`min_distance` and `max_count` are two stop rules:

- `min_distance`: stop when the next farthest frame is still closer than this
  distance to the selected set. This avoids choosing very similar structures.
- `max_count`: optional cap on how many frames to keep. Omit it if you only
  want the distance rule to decide how many structures are different enough.

For example, `min_distance: 0.2` and `max_count: 200` means "keep at most 200
frames, but stop earlier if the remaining frames are too similar." The distance
is measured in descriptor space, not in Angstrom.

For LAMMPS-MACE trajectories, use:

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

The descriptor model must be a native MACE model loadable by
`MACECalculator`, not the `*.model-mliap_lammps.pt` file used by LAMMPS.
PESMaker defaults to CUDA and all MACE interaction layers. It requests
`invariants_only=True`, then averages atom descriptors separately for each
element and concatenates them into one structure descriptor. This matches
MACE's official fine-tuning FPS implementation. Invariant scalar channels are
used because PESMaker applies ordinary Euclidean distance directly; including
equivariant tensor components would require a rotation-aware metric.

Install the required backend for the selected engine:

```bash
# GPUMD/NEP descriptor backend
python -m pip install ".[selection]"

# MACE descriptor backend
python -m pip install ".[mace]"
```

Because MACE and NEP descriptor distances have different numerical scales,
start a new MACE selection with `min_distance: 0.0` and use `max_count` as the
initial limit. Inspect the distance curve before choosing a nonzero threshold.

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
