# PESMaker Toolkit

This directory provides helper scripts that operate independently of the PESMaker workflow. They generate and merge GPUMD/NEP datasets from VASP AIMD calculations and prepare test sets that do not overlap existing training frames.

| Script | Selection method | Intended use |
| --- | --- | --- |
| `aimd_fps_to_nep.py` | PESMaker farthest-point sampling (FPS) | Select representative structures based on configurational differences |
| `aimd_interval_to_nep.py` | Fixed-interval sampling | Thin an AIMD time series at a fixed trajectory-frame interval |
| `merge_aimd_train_xyz.py` | Explicit-source merge | Merge multiple finalized FPS or Interval `train.xyz` files and record source ranges |
| `prepare_aimd_testset.py` | Balanced sampling after excluding training frames | Generate a labeled NEP `test.xyz` from multiple AIMD trajectories |
| `run_vasp_gpu_pool.py` | Local GPU worker pool | Queue prepared VASP single-point directories on idle GPUs without Slurm |

The FPS and fixed-interval scripts neither modify nor import private PESMaker source code. They invoke the public command:

```text
python -m pesmaker select <automatically generated YAML>
```

## Shared Workflow

```text
XDATCAR
  → PESMaker selects configurations
  → Locate OUTCAR ionic steps using manifest.jsonl and NBLOCK
  → Stream TOTEN, forces, and stress
  → Verify configuration matching and labels
  → train.xyz
```

OUTCAR is parsed one ionic step at a time rather than loading every frame into memory. FPS still requires PESMaker to read XDATCAR and calculate geometric features; the fixed-interval script first counts XDATCAR frames as a text stream.

## Scope

The current version supports:

- One VASP AIMD directory.
- Fixed-cell trajectories.
- A constant atom count and atom order throughout AIMD.
- XDATCAR and OUTCAR produced by the same calculation.
- XDATCAR and OUTCAR files that have not been concatenated manually.
- Three-dimensional periodic systems.

The current version does not handle:

- Variable-cell AIMD.
- Merging multiple AIMD directories during extraction.
- Concatenated trajectories or mixed restart segments.
- Arbitrary random train/test splits; the test-set script only excludes frames through existing frame mappings and performs balanced selection.
- Electronic-step convergence-quality filtering.
- Overwriting or resuming an existing output directory.

## Environment Requirements

Run these scripts with the same Python environment in which PESMaker is installed. The environment requires at least:

- Python 3.10 or later.
- PESMaker.
- ASE.
- NumPy.
- PyYAML.
- Matplotlib (required for FPS plots).

Check the environment with:

```bash
python -m pesmaker --help
python -c "import ase, numpy, yaml; print(ase.__version__)"
```

The scripts start PESMaker through `sys.executable -m pesmaker select`. The Python interpreter running the script must therefore belong to the environment where PESMaker is installed.

## VASP GPU Worker Pool on Servers Without Slurm

`run_vasp_gpu_pool.py` is independent of the PESMaker source code. It recursively
discovers calculation directories containing both `POSCAR` and `submit.sh` and
runs one `bash submit.sh` process at a time on each candidate GPU. When a process
exits, that GPU immediately takes the next directory from the queue; different
GPUs operate in parallel.

Preview the jobs and current GPU status first:

```bash
python run_vasp_gpu_pool.py /path/to/labeling \
  --gpus auto \
  --dry-run
```

Detach the Python controller from the terminal and use all GPUs detected as idle:

```bash
python run_vasp_gpu_pool.py /path/to/labeling \
  --gpus auto \
  --background
```

After the command returns, the controller PID is stored in `labeling/gpu_pool.pid`
and controller output is appended to `labeling/gpu_pool_driver.log`. The SSH or
terminal window may then be closed. Follow the controller output with:

```bash
tail -f /path/to/labeling/gpu_pool_driver.log
```

Allow only physical GPUs 0, 2, and 3:

```bash
python run_vasp_gpu_pool.py /path/to/labeling --gpus 0,2,3
```

By default, a job starts only when used GPU memory is no more than 1024 MB and
GPU utilization is no more than 10%. Adjust these thresholds for display
services, ECC, and resident driver memory on the server:

```bash
python run_vasp_gpu_pool.py /path/to/labeling \
  --gpus 0,1,2,3 \
  --max-memory-used-mb 1500 \
  --max-utilization 10 \
  --poll-seconds 10
```

The script writes the physical GPU index to `CUDA_VISIBLE_DEVICES` for each child
process and sets `PESMAKER_GPU_POOL=1`. Therefore, `submit.sh` must not hard-code
or override the GPU in pool mode, nor may it place VASP in the background in that
mode; otherwise, the scheduler would release the GPU too early.

`templates/submit.sh` supports both invocation modes. To test one directory
manually and submit it independently in the background, use:

```bash
bash submit.sh --background --gpu 0
```

This creates `vasp.log` and a `vasp.pid` file while the process is running; the
job is not controlled by the Python worker pool. When the pool executes the same
file, it automatically sets `PESMAKER_GPU_POOL=1` and waits in the foreground for
the actual VASP process to finish, so the job script does not need to change.
For manual foreground debugging, use:

```bash
bash submit.sh --gpu 0
```

The template defaults to the VASP 6.5.1 paths from the reference script. If the
server uses different paths, edit the defaults at the top of the template or set
`VASP_ENV_FILE` and `VASP_BINARY` before invocation. With one job per GPU, the
template ultimately runs the core command:

```bash
mpirun -np 1 /path/to/vasp_std
```

### oneAPI and Non-Interactive Conda Environments

The pool starts single-point jobs through non-interactive `bash submit.sh`
processes. If the Python controller starts inside a Conda environment, child
shells inherit environment variables such as `CONDA_PREFIX` and `CONDA_EXE`, but
usually do not inherit the `conda` function from the interactive shell. Some
Intel oneAPI `setvars.sh` scripts call `conda deactivate` while initializing
`intelpython`, which can produce:

```text
CondaError: Run 'conda init' before 'conda deactivate'
```

The bundled `templates/submit.sh` handles this case before loading the VASP
environment. Only when `CONDA_EXE` exists and the current shell lacks a `conda`
function does it load `etc/profile.d/conda.sh` from the Conda installation root.
Environments that do not use Conda or have already initialized the shell do not
repeat this step, and `conda init bash` is not required.

After moving to another server, first validate the environment in one calculation
directory in foreground mode:

```bash
bash submit.sh --gpu 0
```

When using an older custom template, place the same Conda initialization guard before:

```bash
source "$vasp_env_file"
```

Prepared single-point directories contain independent copies of the template;
editing only the source template does not update those existing copies.

Default restart behavior:

- Skip an OUTCAR that contains the normal VASP timing/accounting footer and no electronic non-convergence marker.
- Queue a directory when OUTCAR is missing, incomplete, or electronically non-converged.
- Still report failure when the shell returns 0 but OUTCAR lacks the normal footer.
- `--failure-policy continue`: continue with other jobs after an individual failure.
- `--failure-policy stop`: stop launching new jobs without killing jobs already running.

Primary outputs:

```text
labeling/gpu_pool_events.jsonl       # Global start, completion, and failure events
labeling/gpu_pool_driver.log         # Background Python controller output
labeling/gpu_pool.pid                # PID while the background Python controller is running
labeling/**/gpu_pool.log             # stdout/stderr from each submit.sh
labeling/**/OUTCAR                   # VASP output
labeling/**/vasp.exitcode            # Final exit status determined by the template
labeling/**/vasp.done|vasp.failed    # Per-job success or failure summary
```

The script coordinates multiple instances through Linux file locks in
`/tmp/pesmaker_gpu_pool_locks/`, but it cannot prevent software that does not use
those locks from claiming a GPU after the status check. On shared servers,
explicitly specify `--gpus` and coordinate GPU ownership with other users. Use a
formal scheduler such as Slurm for production multi-user resource management.

## Expected Input

`--aimd-dir` must explicitly point to a directory containing:

```text
VASP_AIMD/
├── INCAR
├── XDATCAR
└── OUTCAR
```

All three VASP files are opened read-only. The scripts do not modify, move,
delete, compress, or rename them.

## FPS Script Usage

```bash
python toolkit/aimd_fps_to_nep.py \
  --aimd-dir /absolute/path/to/VASP_AIMD \
  --max-count 200 \
  --min-distance 0.0
```

Arguments:

| Argument | Required | Description |
| --- | --- | --- |
| `--aimd-dir` | Yes | Directory containing INCAR, XDATCAR, and OUTCAR |
| `--max-count` | Yes | Maximum number of configurations selected by FPS; must be a positive integer |
| `--min-distance` | No | Minimum distance in PESMaker's simple geometric-descriptor space; defaults to `0.0` and is not measured in Å |

For an initial run, keep `--min-distance 0.0` and control the selection count only through `--max-count`.

The FPS output directory is fixed at:

```text
<aimd-dir>/PESMakerToolkit_AIMD_FPS_to_NEP/
```

Expected output after success:

```text
PESMakerToolkit_AIMD_FPS_to_NEP/
├── train.xyz
├── selected.xyz
├── manifest.jsonl
├── selection_features.npy
├── fps_selection.png
├── frame_mapping.jsonl
├── pesmaker_fps.yaml
└── toolkit.log
```

## Fixed-Interval Script Usage

Minimal invocation:

```bash
python toolkit/aimd_interval_to_nep.py \
  --aimd-dir /absolute/path/to/VASP_AIMD \
  --interval 100
```

Specify start and end frames:

```bash
python toolkit/aimd_interval_to_nep.py \
  --aimd-dir /absolute/path/to/VASP_AIMD \
  --interval 100 \
  --start-frame 1000 \
  --end-frame 10000
```

Arguments:

| Argument | Required | Default | Description |
| --- | --- | --- | --- |
| `--aimd-dir` | Yes | None | Directory containing INCAR, XDATCAR, and OUTCAR |
| `--interval` | Yes | None | Select one configuration every this many XDATCAR frames; must be a positive integer |
| `--start-frame` | No | `0` | First eligible XDATCAR frame, using zero-based indexing |
| `--end-frame` | No | Last frame | Last eligible XDATCAR frame, using an inclusive boundary |

### Frame Numbers and End Boundary

`start-frame=0` means the first frame stored in XDATCAR, not VASP ionic step 0.

For example:

```text
--start-frame 100
--end-frame 1000
--interval 200
```

Actual selection:

```text
100, 300, 500, 700, 900
```

`end-frame` is the inclusive upper bound of the eligible range, but only frames
on the fixed-interval sequence are selected. The script does not add an extra,
off-interval configuration solely to include the end frame.

If `--end-frame` is omitted, sampling extends through the last XDATCAR frame. An
explicit end frame beyond the trajectory raises an error rather than being
silently truncated.

### `interval` and `NBLOCK`

`--interval` is measured in XDATCAR frames, not VASP ionic steps. Candidate
OUTCAR ionic steps are calculated as:

```text
ionic_step = (source_frame + 1) × NBLOCK
```

For example:

```text
NBLOCK = 5
--interval 100
```

The approximate VASP ionic-step interval between adjacent training configurations is:

```text
100 × 5 = 500
```

The frame-number formula only locates candidates. Every candidate must still pass
comparisons of the cell, element order, and periodic coordinates.

### Expected Fixed-Interval Output

The output directory is fixed at:

```text
<aimd-dir>/PESMakerToolkit_AIMD_Interval_to_NEP/
```

Expected output after success:

```text
PESMakerToolkit_AIMD_Interval_to_NEP/
├── train.xyz
├── selected.xyz
├── manifest.jsonl
├── frame_mapping.jsonl
├── pesmaker_interval.yaml
└── toolkit.log
```

Fixed-interval sampling does not calculate FPS descriptors, so it does not
generate `selection_features.npy` or `fps_selection.png`.

## Test Set from Untrained AIMD Frames

`prepare_aimd_testset.py` prepares a labeled `test.xyz` from multiple AIMD
projects. It reads `frame_mapping.jsonl` from existing Toolkit training outputs
and uses each `source_frame` as an exclusion list, so frames already used for
training do not reappear in the test set.

The script neither invokes nor modifies PESMaker itself; XDATCAR and OUTCAR are
both read as streams. Each trajectory is divided into equal-width time bins. One
untrained configuration, as far as possible in time from the nearest training
frame, is selected from each bin to balance temporal coverage with train/test
separation.

```bash
python toolkit/prepare_aimd_testset.py \
  --aimd-root /absolute/path/to/AIMD_ROOT \
  --training-source interval \
  --count-per-trajectory 20
```

Arguments:

| Argument | Required | Default | Description |
| --- | --- | --- | --- |
| `--aimd-root` | Yes | None | Direct parent directory of multiple real AIMD projects |
| `--training-source` | Yes | None | Training source to exclude: `interval` or `fps` |
| `--count-per-trajectory` | No | `20` | Number of test frames selected from each trajectory |
| `--all-unused` | No | Disabled | Use every frame absent from the specified training set for which complete OUTCAR labels are available; cannot be combined with `--count-per-trajectory` |
| `--output-dir-name` | No | `test_verification_testset` | Name of the output directory created below the AIMD root |

For example, prepare the most complete possible test set of external Interval frames:

```bash
python toolkit/prepare_aimd_testset.py \\
  --aimd-root /absolute/path/to/AIMD_ROOT \\
  --training-source interval \\
  --all-unused \\
  --output-dir-name test_veirfication_testset99pct
```

`--all-unused` does not load every configuration into memory; the script processes
XDATCAR and OUTCAR synchronously as streams. If trailing XDATCAR frames lack
corresponding OUTCAR labels, for example because the final step was not fully
written, those frames cannot become supervised test samples and are recorded in
`skipped_unlabeled_frames.tsv`.

With `--training-source interval`, for example, the script recognizes only this structure:

```text
AIMD_ROOT/
├── AIMD_project_1/
│   ├── INCAR
│   ├── XDATCAR
│   ├── OUTCAR
│   └── PESMakerToolkit_AIMD_Interval_to_NEP/
│       └── frame_mapping.jsonl
└── AIMD_project_2/
    └── ...
```

The output directory is fixed at:

```text
<aimd-root>/test_verification_testset/
├── test.xyz
├── frame_mapping.jsonl
├── source_ranges.tsv
├── skipped_unlabeled_frames.tsv
├── README.md
└── toolkit.log
```

The script may use an existing but empty target directory; it refuses to
overwrite a directory containing any files. `test.xyz` is first written as a
`.partial` file and becomes final only after configuration and label validation
succeeds for every project.

Note that absence from training does not imply strict independence. These test
frames still come from the same AIMD trajectories as the training set and mainly
measure in-distribution interpolation error. Evaluating generalization to new
temperatures, compositions, structure types, or evolution pathways requires a
separate external validation set.

## Merge Script Purpose and Syntax

`merge_aimd_train_xyz.py` merges finalized, validated `train.xyz` files already
generated in multiple AIMD subdirectories. It does not reparse AIMD, invoke
PESMaker, or modify any PESMaker source code.

The merge source must be specified explicitly; the script never guesses or mixes FPS and Interval sources:

```bash
python toolkit/merge_aimd_train_xyz.py \
  --aimd-root /absolute/path/to/AIMD_ROOT \
  --source interval
```

Merge FPS sources:

```bash
python toolkit/merge_aimd_train_xyz.py \
  --aimd-root /absolute/path/to/AIMD_ROOT \
  --source fps
```

Arguments:

| Argument | Required | Description |
| --- | --- | --- |
| `--aimd-root` | Yes | Root directory directly containing multiple VASP AIMD run directories |
| `--source` | Yes | Must be `interval` or `fps`; there is no default |

### Expected Input Layout

Each production AIMD directory directly below `--aimd-root` must contain INCAR, XDATCAR, and OUTCAR:

```text
AIMD_ROOT/
├── aimd_case_01/
│   ├── INCAR
│   ├── XDATCAR
│   ├── OUTCAR
│   ├── PESMakerToolkit_AIMD_Interval_to_NEP/
│   │   └── train.xyz
│   └── PESMakerToolkit_AIMD_FPS_to_NEP/
│       └── train.xyz
├── aimd_case_02/
│   ├── INCAR
│   ├── XDATCAR
│   ├── OUTCAR
│   └── ...
└── scripts/                 # Ignored because it lacks the three VASP files
```

The source argument maps strictly to the lookup path:

```text
--source interval
→ <AIMD-subdirectory>/PESMakerToolkit_AIMD_Interval_to_NEP/train.xyz

--source fps
→ <AIMD-subdirectory>/PESMakerToolkit_AIMD_FPS_to_NEP/train.xyz
```

The script checks only direct children of the AIMD root and does not recursively
search for arbitrary `train.xyz` files. It therefore does not accidentally
collect `_smoke_test`, `_failed_*`, or other temporary directories.

Every recognized production AIMD subdirectory must contain `train.xyz` from the
selected source. If any directory lacks it, the script fails before creating
merged output and lists the missing directories instead of silently skipping
them.

### Merged Output Directory

Output is created inside the AIMD root alongside its AIMD project directories, with the source stated explicitly in the name:

```text
<aimd-root>/Merged_Interval_NEP_TrainXYZ/
<aimd-root>/Merged_FPS_NEP_TrainXYZ/
```

For example:

```text
/mnt/d/ResearchData/growthDatasetPreparation_AIMD/
├── AIMD_project_1/
├── AIMD_project_2/
└── Merged_Interval_NEP_TrainXYZ/
```

Output layout:

```text
Merged_Interval_NEP_TrainXYZ/
├── train.xyz
├── source_ranges.tsv
├── README.md
└── merge.log
```

- `train.xyz`: training sets concatenated directly after case-insensitive sorting by AIMD directory name, without reformatting labels.
- `source_ranges.tsv`: machine-readable sources, frame counts, elements, atom counts, and merged ranges.
- `README.md`: automatically generated, human-readable source table, directory description, merge rules, and reproduction command.
- `merge.log`: source validation, frame ranges, total frame count, and failure reasons.

`source_ranges.tsv` contains:

```text
order
sampling_source
aimd_directory
source_train_xyz
frame_count
atom_count
elements
merged_start_0based
merged_end_0based
merged_start_1based
merged_end_1based
```

Both zero-based and one-based ranges are inclusive. The script calculates ranges
dynamically from the frames actually read and does not hard-code counts from the
current dataset.

### Merge Validation and Safety Behavior

Every source and the final merged file are validated as ASE streams:

- The extxyz contains at least one frame.
- Cell, positions, Energy, force, and Virial are present and finite.
- The force array has exactly shape `N×3`.
- Virial has exactly nine components.
- The merged frame count equals the sum of all source frame counts.
- The merged element set matches the source element sets.

The merge first creates:

```text
train.xyz.partial
source_ranges.tsv.partial
README.md.partial
```

Files receive their final names only after every validation passes, with
`train.xyz` renamed last. The script refuses to overwrite an existing output
directory.

The initial implementation does not perform random shuffling, deduplication,
secondary FPS, train/test splitting, or label-unit conversion.

## WSL Invocation Example

The Windows D drive is usually mounted at `/mnt/d` in WSL. For example:

```bash
conda run -n pesmaker python \
  /mnt/d/ResearchData/PESMaker_litupVer/toolkit/aimd_interval_to_nep.py \
  --aimd-dir /mnt/d/ResearchData/example_AIMD \
  --interval 100 \
  --start-frame 0
```

Display command-line help:

```bash
python aimd_fps_to_nep.py --help
python aimd_interval_to_nep.py --help
python merge_aimd_train_xyz.py --help
```

## Output File Reference

| File | Purpose |
| --- | --- |
| `train.xyz` | Final NEP training data; each frame contains total energy, forces, and total Virial |
| `selected.xyz` | Unlabeled configurations selected by PESMaker |
| `manifest.jsonl` | PESMaker selection manifest containing the original zero-based `source_frame` |
| `frame_mapping.jsonl` | Mapping between `train.xyz` frames and XDATCAR/OUTCAR frame numbers |
| `pesmaker_fps.yaml` | PESMaker configuration generated automatically by the FPS script |
| `pesmaker_interval.yaml` | PESMaker configuration generated automatically by the fixed-interval script |
| `selection_features.npy` | Simple geometric features used by FPS |
| `fps_selection.png` | FPS selection-process plot |
| `toolkit.log` | Arguments, paths, NBLOCK, frame counts, label ranges, geometric deviations, and error reasons |

Both scripts refuse to overwrite an existing identity directory and do not automatically create numbered subdirectories.

## Configuration and Label Checks

Every selected frame must satisfy:

- Atom counts match exactly.
- Elements and atom order match exactly.
- The maximum difference between fixed-cell matrix elements is no more than `1e-4 Å`.
- The maximum atomic displacement under the periodic minimum-image convention is no more than `1e-4 Å`.
- `free_energy`, forces, and stress are present and finite.
- The force array has exactly shape `N×3`.
- The corresponding ionic step can be found in OUTCAR.

If any selected frame fails these requirements, the entire conversion fails and no final `train.xyz` is generated.

## `train.xyz` Label Definitions

Each configuration uses GPUMD/NEP extended XYZ format:

```text
Lattice="ax ay az bx by bz cx cy cz"
Energy=<total energy in eV>
Properties=species:S:1:pos:R:3:force:R:3
Virial="vxx vxy vxz vyx vyy vyz vzx vzy vzz"
pbc="T T T"
```

- `Energy`: VASP `free energy TOTEN`, in eV; this is the total energy.
- `force`: force on each atom, in eV/Å.
- `Virial`: total 3×3 Virial for the configuration, in eV; it is not a per-atom value.

Virial conversion uses:

```text
Virial = -V × stress_ASE
```

ASE uses the tensile-positive stress convention. The converted total Virial
follows the sign convention used by GPUMD/NEP.

## Safe Output and Failure Behavior

The scripts first write:

```text
train.xyz.partial
frame_mapping.jsonl.partial
```

These files receive their final names only after every selected frame matches and passes label validation:

```text
train.xyz
frame_mapping.jsonl
```

On failure:

- No final `train.xyz` is generated.
- Completed selection results and `toolkit.log` are retained.
- `.partial` files may remain but must not be used for training.
- The failure reason is written to the log and displayed in the terminal.

## Common Errors

### Output Directory Already Exists

The scripts do not overwrite existing results:

```text
Error: output directory already exists; refusing to overwrite
```

Inspect the existing `train.xyz` and `toolkit.log` before deciding whether to retain or move the directory.

### Configuration Mismatch

Common causes include:

- XDATCAR and OUTCAR come from different calculations.
- Files were concatenated or truncated.
- `NBLOCK` in INCAR does not match the actual trajectory output.
- AIMD used variable-cell settings.
- Another tool changed the atom order.

### Missing Energy, Forces, or Stress

The scripts do not substitute another energy definition for TOTEN and do not
generate training frames without Virial. Check whether OUTCAR is complete and
whether the corresponding ionic step ended normally.

## Completed Real-Data Validation

The FPS script has completed an end-to-end test on a fixed-cell VASP AIMD directory:

- XDATCAR: 16,851 frames.
- OUTCAR: approximately 1.46 GB.
- FPS test selection: 5 frames.
- Selected ionic steps: 1, 5883, 10596, 14036, and 16583.
- Atoms per frame: 156.
- `train.xyz` can be read back by ASE.
- TOTEN agrees with the PESMaker `collect` parser result.
- The maximum Virial difference is approximately `6.54×10⁻⁵ eV`, arising from OUTCAR text-printing precision.

The fixed-interval script completed an end-to-end test on the same 16,851-frame trajectory with:

```text
--start-frame 100
--end-frame 16500
--interval 4000
```

Validation results:

- Actual source frames: 100, 4100, 8100, 12100, and 16100.
- Corresponding OUTCAR ionic steps: 101, 4101, 8101, 12101, and 16101 (`NBLOCK=1`).
- `manifest.jsonl`, `frame_mapping.jsonl`, and the expected frame numbers agree exactly.
- All five structures contain finite Energy, `156×3` force, and nine Virial components.
- `train.xyz` can be read back by ASE.
- No `.partial` files remain.

Real-data tests demonstrate only that the workflow works for the corresponding
data within the current scope. They do not replace separate validation for other
VASP versions, variable-cell trajectories, or concatenated trajectories.
