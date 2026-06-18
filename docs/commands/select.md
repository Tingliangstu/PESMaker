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
Progress         : [###---------------------------] 151/1501 frame(s) ( 10.1%)
...
Progress         : [##############################] 1501/1501 frame(s) (100.0%)
Descriptor matrix: 1501 frame(s) x 512 feature(s)
FPS completed    : Selected 200 of 1501 frame(s).
```

GPUMD prints the same progress block with `Engine: GPUMD`, the
`Calorine-calculated NEP descriptors` backend, and the NEP potential path. In
an interactive terminal the progress bar updates in place. When output is
redirected to a log, PESMaker prints progress at regular intervals instead.
This uses the Python standard library and does not add another package
dependency.

PESMaker suppresses only known upstream MACE/e3nn/PyTorch model-loading
compatibility messages, including the automatic model-dtype notice. Other
warnings and all calculation errors remain visible.

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

## Separate Trajectory Sampling

When `trajectory_pattern` matches several MD trajectory files, PESMaker samples
each trajectory independently by default. This keeps every initial structure
represented in the selected DFT labeling set.

```yaml
sampling:
  engine: gpumd
  potential: /path/to/nep.txt
  selection:
    trajectory_pattern: MD_run_2D_Pd_551/*/movie.xyz
    output_dir: selected
    min_distance: 0.004
    max_count: 50
```

Here `max_count: 50` means at most 50 frames from each trajectory, not 50
frames total. For example, two matched `movie.xyz` files can produce up to 100
selected frames.

The terminal output states that separate trajectory selection is active:

```text
Separate trajectory selection
Mode             : separate_trajectories
Trajectories     : 2
Method           : FPS
Per trajectory   : max_count=50, min_distance=0.004
Output directory : selected

Trajectory       : 1/2 MD_run_2D_Pd_551/mp-1186427_Pd_temp_300K/movie.xyz
Selected         : 50 of 1000 frame(s)

Trajectory       : 2/2 MD_run_2D_Pd_551/mp-2646997_Pd_temp_300K/movie.xyz
Selected         : 50 of 1000 frame(s)

Separate selection completed: Selected 100 of 2000 frame(s) from 2 trajectory file(s).
```

This default is useful when the trajectories come from different initial
structures. If several trajectories are replicas of the same structure and you
want one global FPS selection over all frames, disable separate sampling:

```yaml
sampling:
  selection:
    trajectory_pattern: MD_run_2D_Pd_551/*/movie.xyz
    output_dir: selected
    separate_trajectories: false
    min_distance: 0.004
    max_count: 100
```

## Interval Sampling

Use interval sampling when you want frames at a fixed trajectory stride and do
not want PESMaker to calculate descriptors. This mode does not require
`sampling.potential`, Calorine, MACE, `min_distance`, or a descriptor model.

```yaml
sampling:
  selection:
    method: interval
    trajectory_pattern: /path/to/XDATCAR
    output_dir: selected
    interval: 10
```

This keeps frames `0, 10, 20, ...`. Optional `offset` starts from a later frame,
and optional `max_count` caps the number of kept frames:

```yaml
sampling:
  selection:
    method: interval
    trajectory_pattern: sampling/**/movie.xyz
    output_dir: selected
    interval: 20
    offset: 5
    max_count: 50
```

The aliases `strategy` or `mode` may be used instead of `method`, and `stride`,
`step`, or `frame_interval` may be used instead of `interval`.

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

The manifest `descriptor` field records the descriptor family used for FPS:
`nep89` for GPUMD runs with a `nep89*` potential, `nep` for other GPUMD NEP
potentials, `mace` for MACE descriptors, and `simple` for the geometry fallback.
For GPUMD, Calorine is still the calculation backend; it is not written as the
descriptor family.

`selection_features.npy` is a NumPy array with shape
`(number_of_md_frames, descriptor_dimension)`. Row `i` is the descriptor vector
used for MD frame `i` before FPS selection. It is useful for debugging and
post-analysis: you can reload it with `numpy.load`, reproduce descriptor-space
PCA plots, compare different `min_distance` values, or check whether your
descriptor separates different trajectory regions.

`fps_selection.png` is the diagnostic plot. PESMaker uses a seaborn-styled
plot: all MD frames are shown as light points, while selected frames are drawn
smaller on top so you can see how they sit inside the full cloud.

For separate trajectory sampling, PESMaker writes a subdirectory per trajectory:

```text
selected/
  manifest.jsonl
  mp-1186427_Pd_temp_300K/
    selected.xyz
    manifest.jsonl
    selection_features.npy
    fps_selection.png
  mp-2646997_Pd_temp_300K/
    selected.xyz
    manifest.jsonl
    selection_features.npy
    fps_selection.png
```

The top-level `selected/manifest.jsonl` combines all selected frames and is the
file used by `pesmaker scf-setup` or `pesmaker next` for later labeling.

For interval sampling, PESMaker writes only `selected.xyz` and `manifest.jsonl`
because no descriptor matrix or FPS diagnostic plot is calculated.

## Next Step

```bash
pesmaker scf-setup run.yaml
```

If `labeling.input_manifest` is omitted, `next` uses
`selected/manifest.jsonl` after selection.
