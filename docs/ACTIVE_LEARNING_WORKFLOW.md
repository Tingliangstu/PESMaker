# PESMaker Workflow Manual

This manual is organized by function. Each command has its own section with
the same pattern: what it does, what it reads, what it writes, and which YAML
keys control it.

PESMaker is built around ordinary files and directories. Each command prepares
one visible stage of the workflow, so generated structures, sampling jobs, SCF
inputs, collected datasets, and training folders can be inspected and repaired
without rerunning the whole pipeline.

## Command Map

| Command | Purpose | Main output |
| --- | --- | --- |
| `pesmaker init` | Write a starter YAML file | `pesmaker.yaml` or a chosen path |
| `pesmaker validate` | Check YAML syntax and schema | terminal validation result |
| `pesmaker generate` | Build supercells, surfaces, defects, and optional perturbations | `generated/` and `manifest.jsonl` |
| `pesmaker sample-setup` | Prepare MD sampling jobs | `sampling/` |
| `pesmaker select` | Select representative MD frames | `selected/` |
| `pesmaker scf-setup` | Prepare VASP SCF folders | `labeling/` |
| `pesmaker submit` | Submit prepared `submit.sh` files | scheduler submissions or dry-run log |
| `pesmaker collect` | Collect completed SCF outputs | `train.xyz` or configured dataset path |
| `pesmaker train-setup` | Prepare model training inputs | `training/` |

## Common Workflow Paths

Use the direct SCF path when generated structures should go straight to VASP
labeling:

```bash
pesmaker validate run.yaml           # check the config before doing work
pesmaker generate run.yaml           # create structural candidates
pesmaker scf-setup run.yaml          # prepare VASP calculation folders
pesmaker submit run.yaml --dry-run   # preview SCF submissions
pesmaker submit run.yaml             # submit SCF jobs
pesmaker collect run.yaml            # collect finished SCF results
```

Use the sampling path when generated structures should first seed MD:

```bash
pesmaker validate run.yaml                  # check the config before doing work
pesmaker generate run.yaml                  # create initial structures
pesmaker sample-setup run.yaml              # prepare MD sampling folders
pesmaker submit run.yaml --stage sampling   # submit MD jobs
pesmaker select run.yaml                    # select representative MD frames
pesmaker scf-setup run.yaml                 # prepare SCF folders for selected frames
pesmaker submit run.yaml                    # submit SCF jobs
pesmaker collect run.yaml                   # collect finished SCF results
pesmaker train-setup run.yaml               # prepare model training inputs
pesmaker submit run.yaml --stage training   # submit training jobs
```

Typical stage directories are:

```text
generated/   # generated supercells, surfaces, defects, optional perturbations
sampling/    # MD job folders with model.xyz, run.in, submit.sh
selected/    # selected MD frames and manifest
labeling/    # VASP SCF folders with POSCAR, INCAR, POTCAR, submit.sh
training/    # model training inputs and submit.sh
```

## Configuration Basics

Create a starter file:

```bash
pesmaker init run.yaml
```

A full config can contain these sections:

```yaml
project: Te_Pd_rich_defect_md

structures:
  include:
    - initial_structures/*.cif

generation:
  output_dir: generated

sampling:
  engine: gpumd

labeling:
  engine: vasp

dataset:
  format: extxyz

training:
  model: nep

jobs:
  submit_command: sbatch
```

Important conventions:

- `project` names the campaign and is used in default output paths.
- `structures` is required for `generate`, but later stages can read generated
  manifests or directories without listing original structures again.
- `structures` can be a list of paths or an `include` glob map.
- `generation.tasks` should be used when one config needs several independent
  structure families.
- Do not repeat the same YAML key in one mapping. PESMaker rejects duplicate
  keys because silent overwrites are dangerous in production runs.

## `init`: Create a Starter Config

Run:

```bash
pesmaker init run.yaml
```

`init` writes a starter YAML file and refuses to overwrite an existing file.
Use it as a scaffold, then replace paths, potentials, and scheduler settings
with values for your project and cluster.

## `validate`: Check the Config

Run:

```bash
pesmaker validate run.yaml
```

`validate` checks that the YAML can be parsed and that top-level sections have
valid shapes. It is cheap and should be run before expensive generation,
sampling, or SCF setup.

## `generate`: Build Structures

Run:

```bash
pesmaker generate run.yaml
```

`generate` reads the `structures` and `generation` sections. It writes
generated structures, a machine-readable `manifest.jsonl`, and a human-readable
`generation_summary.txt`.

Inside each generation task, operations happen in this order:

```text
input structure -> supercell -> surface -> defects -> optional perturb
```

### Inputs

Use explicit paths:

```yaml
structures:
  - Te-mp-19.cif
  - Te-mp-23.cif
```

Or use glob patterns:

```yaml
structures:
  include:
    - initial_structures/*.cif
```

Input structures are read through ASE, so CIF, POSCAR, VASP, extxyz, and xyz
style files are suitable starting points.

### Tasks

Use one task for one structural family:

```yaml
generation:
  output_dir: generated
  tasks:
    - name: surface_331
      supercell: [3, 3, 1]
      surface:
        vacuum: 30.0
        axis: 2
        center: true
      perturb:
        pert_num: 3
        format: vasp
```

Use several tasks when one config should build several independent families:

```yaml
generation:
  output_dir: generated
  tasks:
    - name: surface_331
      supercell: [3, 3, 1]
      surface:
        vacuum: 30.0
        axis: 2
        center: true
      perturb:
        pert_num: 3
        format: vasp
    - name: bulk_333
      supercell: [3, 3, 3]
      perturb:
        pert_num: 3
        format: vasp
```

### Supercells

`supercell` contains replication factors along the three lattice directions:

```yaml
supercell: [4, 4, 1]
```

For a 2D material, a common choice is `[n, n, 1]`. For a bulk seed structure,
use all three directions as needed.

For pure supercell expansion, omit `perturb` entirely:

```yaml
project: Te_bulk_mp

structures:
  include:
    - initial_structures/*.cif

generation:
  supercell: [3, 3, 3]
  output_dir: generated
```

This writes one expanded `pristine_3x3x3.vasp` file for each input structure
and does not create `perturb_*.vasp` files.

### Surface Slabs and Vacuum

For slab or 2D systems:

```yaml
surface:
  vacuum: 30.0
  axis: 2
  center: true
```

`vacuum` is the total empty-space thickness in Angstrom. Existing vacuum in the
input structure is replaced. For example, if the slab thickness is about 4
Angstrom and `vacuum: 30.0`, the final cell length along the vacuum axis is
about 34 Angstrom.

`axis` chooses the lattice vector used as the vacuum direction:

- `axis: 0`: vacuum along lattice vector a;
- `axis: 1`: vacuum along lattice vector b;
- `axis: 2`: vacuum along lattice vector c, the usual setting for 2D slabs in
  the xy plane.

`center: true` places the slab near the middle of the vacuum direction.

### Perturbations

Perturbation settings live under `perturb`. Random perturbations are disabled
unless `pert_num` is greater than zero:

```yaml
perturb:
  pert_num: 3
  cell_pert_fraction: 0.03
  atom_pert_distance: 0.1
  atom_pert_style: normal
  seed: 421
  format: vasp
```

Key fields:

- `pert_num`: number of random perturbations generated from each variant.
  Default is `0`.
- `cell_pert_fraction`: random cell strain amplitude.
- `atom_pert_distance`: atomic displacement scale in Angstrom.
- `atom_pert_style`: `normal`, `uniform`, or `const`.
- `atom_pert_prob`: fraction of atoms displaced, default `1.0`.
- `seed`: reproducible random seed.
- `format`: `vasp` or `extxyz`.

The pristine variant always gets one named pristine file. When `pert_num` is
greater than zero, the random perturbation files are written after it:

```text
pristine/
  pristine_3x3x3.vasp
  perturb_000000.vasp
  perturb_000001.vasp
```

For surface tasks, the perturbed pristine files use the `surface_` prefix:

```text
pristine/
  pristine_3x3x1.vasp
  surface_000000.vasp
  surface_000001.vasp
```

When no random perturbations are requested, every generated variant is written
once as a named pristine file. The true pristine variant uses
`pristine_<supercell>.vasp`; defect variants append the variant name. When
random perturbations are requested, set
`include_pristine: true` when every defect variant should also receive its own
named pristine file:

```yaml
perturb:
  include_pristine: true
  pert_num: 3
  format: vasp
```

Then a defect folder looks like:

```text
single_vacancy_Te_000001/
  pristine_3x3x3_single_vacancy_Te_000001.vasp
  defect_000000.vasp
  defect_000001.vasp
  defect_000002.vasp
```

The true pristine variant keeps the short name, such as
`pristine_3x3x3.vasp`. Defect variants append the variant name so the file
still identifies the defect if it is copied out of its folder.

### Defects

Defects can be written under `generation.defects` or nested under
`generation.surface.defects`. For slab workflows, the nested form keeps the
operation order visually clear:

```yaml
surface:
  vacuum: 30.0
  axis: 2
  center: true
  defects:
    mode: random
    seed: 42
    single_vacancies:
      elements: [Te]
      max_count: 5
    double_vacancies:
      elements: [Te]
      max_count: 5
    line_defects:
      elements: [Te]
      max_count: 5
```

Supported families:

- `pristine`: no atoms removed;
- `single_vacancies`: remove one atom;
- `double_vacancies`: remove two atoms;
- `line_defects`: remove one row of atoms.

Shared defect options such as `mode`, `selection`, and `seed` are inherited by
the individual defect families unless a family overrides them.

Folder names use 1-based serial numbers within each family. They are not atom
IDs. Exact removed atom indices are written in `manifest.jsonl` as
`variant_description`.

Example:

```json
{
  "variant": "line_defect_Te_const_b_000001",
  "variant_description": "line defect: fixed fractional b coordinate, remove atoms [1, 4, 7, 10]"
}
```

### Single Vacancies

```yaml
single_vacancies:
  elements: [Te]
  max_count: 5
```

When `mode: random` or `selection: random` is active, PESMaker randomly samples
candidate atoms. Without random selection, it takes the first candidates in
atom order.

### Double Vacancies

```yaml
double_vacancies:
  elements: [Te]
  max_count: 5
```

By default, double vacancies are ordered by nearest atom-pair distance, then
the first `max_count` pairs are used. With `selection: random`, PESMaker
randomly samples atom pairs instead.

### Line Defects

Line-defect generation has two separate steps:

1. Group candidate atoms into rows.
2. Select which rows to remove.

`coordinate_axis` controls the grouping step. It is not the random-selection
switch.

```yaml
line_defects:
  elements: [Te]
  max_count: 5
  coordinate_axis: 1
```

The values are fractional-coordinate axes:

- `coordinate_axis: 0`: group atoms with similar fractional `a` coordinate;
- `coordinate_axis: 1`: group atoms with similar fractional `b` coordinate;
- `coordinate_axis: 2`: group atoms with similar fractional `c` coordinate,
  usually not useful for in-plane line defects in 2D slabs.

Folder names use `const_a`, `const_b`, or `const_c` for the fractional
coordinate held constant. In an orthogonal 2D cell, `const_a` usually removes a
row running along b/y, while `const_b` usually removes a row running along a/x.
For hexagonal or otherwise non-orthogonal cells, interpret the row using the
lattice vectors rather than Cartesian x/y labels.

If `coordinate_axis` is omitted, PESMaker tries fractional `a` and `b` and uses
the axis that gives the clearest row grouping. It does not automatically use
fractional `c` for 2D line defects.

`tolerance` controls how close fractional coordinates must be to count as the
same row:

```yaml
line_defects:
  elements: [Te]
  max_count: 5
  coordinate_axis: 1
  tolerance: 0.03
```

If `tolerance` is omitted, PESMaker infers it from the spacing between
candidate rows.

Row selection is controlled by `mode` or `selection`:

```yaml
defects:
  mode: random
  seed: 42
  line_defects:
    elements: [Te]
    max_count: 5
    coordinate_axis: 1
```

With `mode: random`, PESMaker randomly selects `max_count` rows from the
grouped rows. `seed` makes the selection reproducible. Running the same config
again gives the same rows.

Without random selection, PESMaker sorts rows by row size and atom index, then
takes the first `max_count` rows. This deterministic mode is useful for
debugging but is not a random sampling of line positions.

Per-family settings can override the global mode:

```yaml
defects:
  mode: random
  seed: 42
  line_defects:
    selection: ordered
    elements: [Te]
    max_count: 5
    coordinate_axis: 0
```

### Generate Outputs and Summary

Generated folders are grouped by task, input structure, and variant:

```text
generated/
  manifest.jsonl
  generation_summary.txt
  surface_331/
    Te-mp-19/
      pristine/
        pristine_3x3x1.vasp
        surface_000000.vasp
      single_vacancy_Te_000001/
        defect_000000.vasp
      double_vacancy_Te_000001/
        defect_000000.vasp
      line_defect_Te_const_b_000001/
        defect_000000.vasp
```

`manifest.jsonl` is the file later stages read. `generation_summary.txt` is the
fastest file to inspect by eye.

For `max_count: 5` and `pert_num: 3`, the summary is shaped like:

```text
per input:
  pristine: 4 structure(s) (1 pristine, 3 perturbed)
  single vacancies: 5 variant(s), 15 structure(s) (15 perturbed)
  double vacancies: 5 variant(s), 15 structure(s) (15 perturbed)
  line defects: 5 variant(s), 15 structure(s) (15 perturbed)
```

With `include_pristine: true`, each defect variant also gets one named pristine
file, so each defect family above becomes:

```text
single vacancies: 5 variant(s), 20 structure(s) (5 pristine, 15 perturbed)
```

The corresponding defect files are named with both the supercell and variant,
for example `pristine_3x3x1_line_defect_Te_const_b_000001.vasp`.

## `sample-setup`: Prepare MD Sampling Jobs

Run:

```bash
pesmaker sample-setup run.yaml
```

`sample-setup` prepares MD folders from generated structures or from an
explicit input directory or manifest.

Input priority:

1. `sampling.input_manifest`
2. `sampling.input_dir`
3. `generation.output_dir`
4. local `generated/`
5. `runs/<project>/generated`

Minimal GPUMD setup:

```yaml
sampling:
  engine: gpumd
  output_dir: sampling
  gpumd_dir: /path/to/GPUMD/src
  potential: nep89_20250409.txt
  temperatures: [300, 600, 900]
  run_in: templates/gpumd/run.in
```

Constant temperatures:

```yaml
temperatures: [300, 600, 900]
```

Heating ramp:

```yaml
temperature: 300-1500
```

Expected output:

```text
sampling/
  sampling_manifest.jsonl
  md_000000_temp_300K/
    model.xyz
    run.in
    submit.sh
  md_000000_temp_600K/
    model.xyz
    run.in
    submit.sh
```

If `potential` points to an existing file, PESMaker copies it into each MD
folder. If it is only a filename, PESMaker writes the filename into `run.in`
and assumes it will be available when the job runs.

Submit sampling jobs with:

```bash
pesmaker submit run.yaml --stage sampling
```

## `select`: Select Representative Frames

Run after sampling trajectories are available:

```bash
pesmaker select run.yaml
```

Configure frame selection under `sampling.selection`:

```yaml
sampling:
  selection:
    trajectory_pattern: sampling/**/movie.xyz
    output_dir: selected
    min_distance: 0.2
    max_count: 200
```

The current selector uses farthest point sampling on simple structural
features. It keeps adding the frame farthest from the selected set until
`max_count` is reached or the nearest-selected distance falls below
`min_distance`.

Expected output:

```text
selected/
  selected.xyz
  selected_000000.xyz
  selected_000001.xyz
  manifest.jsonl
```

Use `selected.xyz` for quick visual inspection. Use `selected/manifest.jsonl`
as the input to `scf-setup`.

## `scf-setup`: Prepare VASP Labeling Jobs

Run:

```bash
pesmaker scf-setup run.yaml
```

`scf-setup` turns generated structures or selected frames into independent VASP
SCF folders.

Input priority:

1. `labeling.input_manifest`
2. `labeling.input_dir`
3. `generation.output_dir`
4. local `generated/`
5. `runs/<project>/generated`

### Label Selected MD Frames

```yaml
labeling:
  engine: vasp
  output_dir: labeling
  input_manifest: selected/manifest.jsonl
  incar: templates/vasp/INCAR
  potcar_library: /path/to/VASP/potentials
  command: /path/to/vasp_std
```

### Label Generated Structures Directly

```yaml
labeling:
  engine: vasp
  output_dir: labeling
  input_dir: generated
  incar: templates/vasp/INCAR
  potcar_library: /path/to/VASP/potentials
  command: /path/to/vasp_std
```

When an input directory has `manifest.jsonl`, PESMaker reads it for
traceability. Without a manifest, it recursively scans for `POSCAR`, `CONTCAR`,
`*.vasp`, `*.poscar`, `*.cif`, `*.extxyz`, and `*.xyz`.

Expected output:

```text
labeling/
  labeling_manifest.jsonl
  surface_331/
    Te-mp-19/
      single_vacancy_Te_000001/
        defect_000000/
          POSCAR
          defect_000000.vasp-bak
          INCAR
          POTCAR
          POTCAR.spec
          submit.sh
```

The generated structure is backed up by default. Set `backup_source: false`
under `labeling` only if these backups are not wanted.

### VASP Input Controls

The default `examples/templates/vasp/INCAR` is a conservative static SCF
template. You can provide your own file:

```yaml
labeling:
  incar: templates/vasp/INCAR
```

For CPU VASP jobs, PESMaker writes `KPAR` and `NCORE` into `INCAR`. `KPAR`
defaults to `2` when `jobs.cores_cpu` is even, otherwise `1`. `NCORE` is chosen
as a factor within each KPAR group.

Override these when benchmarks or cluster rules require it:

```yaml
jobs:
  cores_cpu: 36
  vasp_kpar: 2
  vasp_ncore: 6
```

PESMaker writes `NCORE`, not legacy `NPAR`.

If `potcar_library` is set, PESMaker assembles `POTCAR` from the built-in
recommended VASP potential table. For ordinary PBE calculations this uses
standard recommendations such as `Te`, `Na_pv`, `K_sv`, and `Ga_d`.

For GW potentials:

```yaml
labeling:
  gw_potcar: true
```

Each folder also receives `POTCAR.spec`, recording the exact potential
directories concatenated into `POTCAR`.

Explicit files can also be copied in:

```yaml
labeling:
  potcar: templates/vasp/POTCAR
  kpoints: templates/vasp/KPOINTS
  template_dir: templates/vasp/static_scf
```

## `submit`: Submit Prepared Jobs

`submit` submits existing `submit.sh` files. It does not create structures,
sampling inputs, SCF folders, or training inputs.

Default behavior submits the SCF labeling stage:

```bash
pesmaker submit run.yaml
```

Preview first:

```bash
pesmaker submit run.yaml --dry-run
```

Submit other stages:

```bash
pesmaker submit run.yaml --stage sampling
pesmaker submit run.yaml --stage training
```

Stage meaning:

- `--stage sampling`: submit jobs prepared by `sample-setup`;
- no `--stage`, or `--stage scf`: submit jobs prepared by `scf-setup`;
- `--stage training`: submit jobs prepared by `train-setup`.

Scheduler settings live under `jobs`:

```yaml
jobs:
  machine: cluster-a
  submit_command: sbatch
  nodes: 1
  cores_cpu: 36
  gpus: 0
  sub_file:
    sampling: templates/sbatch/gpumd.sh
    labeling: templates/sbatch/vasp_cpu_36.sh
    training: templates/sbatch/nep.sh
```

For one-stage configs, `sub_file` can be a single path:

```yaml
jobs:
  submit_command: sbatch
  cores_cpu: 36
  sub_file: templates/sbatch/vasp_cpu_36.sh
```

Templates can use:

```text
{job_name}          # generated from the work directory name
{workdir}           # stage working directory
{command}           # engine command, such as gpumd, vasp_std, or nep
{nodes}             # jobs.nodes, default 1
{ntasks}            # nodes * cores_cpu
{cores_cpu}         # CPU cores per node
{ntasks_per_node}   # alias for cores_cpu
{gpus}              # GPUs per node
{vasp_kpar}         # generated VASP KPAR
{vasp_ncore}        # generated VASP NCORE
```

When a template already contains `#SBATCH --job-name`, `#SBATCH --ntasks`, or a
VASP run line such as `mpirun /path/to/vasp_std`, PESMaker rewrites those values
for each calculation folder while preserving site-specific partitions,
accounts, modules, and environment setup.

The default CPU VASP run command is:

```bash
mpirun {command}
```

For GPU jobs:

```yaml
jobs:
  cores_cpu: 8
  gpus: 1
```

`submit` runs the scheduler command from each job folder. Manual submission
should do the same:

```bash
cd labeling/path/to/calc
sbatch submit.sh
```

## `collect`: Build the Labeled Dataset

Run after SCF jobs finish:

```bash
pesmaker collect run.yaml
```

Configure completed-output discovery:

```yaml
labeling:
  outcar_pattern: labeling/**/OUTCAR
  dataset_path: train.xyz
```

`collect` reads matched VASP outputs with ASE and writes an extxyz dataset.
Collection is separate from SCF setup and submission so failed or partial jobs
can be inspected before rebuilding the dataset.

## `train-setup`: Prepare Training Inputs

Run:

```bash
pesmaker train-setup run.yaml
```

For NEP:

```yaml
training:
  model: nep
  output_dir: training
  dataset: train.xyz
  command: nep
```

Expected output:

```text
training/
  train.xyz
  nep.in
  submit.sh
```

Submit training with:

```bash
pesmaker submit run.yaml --stage training
```

For non-NEP trainers, set `training.model` and `training.command`. The stage
boundary remains the same, so new trainers can be added without changing
generation, sampling, labeling, or collection.

## Inspection Checklist

Before generation:

- Run `pesmaker validate run.yaml`.
- Check input chemistry, cell, dimensionality, and units.
- For 2D materials, confirm `surface.axis` and `surface.vacuum`.
- Start with small `max_count` and `pert_num` values.

After generation:

- Read `generated/generation_summary.txt`.
- Inspect representative `pristine`, vacancy, double-vacancy, and line-defect
  structures visually.
- Confirm atom counts and defect families match the intended campaign.
- Check `manifest.jsonl` when you need exact removed atom indices.

Before submission:

- Run `pesmaker submit run.yaml --dry-run`.
- Inspect at least one `submit.sh` per stage.
- For SCF jobs, inspect `POSCAR`, `INCAR`, `POTCAR`, `POTCAR.spec`, and resource
  settings.
- Confirm `labeling.command`, `jobs.cores_cpu`, modules, and cluster account
  settings match the target machine.

After SCF jobs:

- Check convergence before collection.
- Collect only completed outputs matched by `labeling.outcar_pattern`.
- Keep `train.xyz` versioned by project or campaign name.
- Prepare training only after confirming dataset contents.

## Current Scope

Implemented capabilities:

- multi-task structure generation;
- supercells, 2D vacuum setup, pristine structures, single vacancies, double
  vacancies, line defects, and optional perturbations;
- GPUMD sampling setup;
- farthest point trajectory-frame selection;
- VASP SCF folder setup with submit scripts and optional POTCAR assembly;
- scheduler submission with dry-run support;
- extxyz dataset collection through ASE;
- NEP training setup.

Planned extension points:

- LAMMPS-MACE sampling templates and engine-specific input writers;
- more chemically aware defect enumeration;
- stricter VASP convergence filtering during dataset collection;
- richer dataset splitting and training reports.
