# Active Learning Workflow Manual

This manual describes the PESMaker workflow for 2D defect structures, large
model driven molecular dynamics, single-point labeling, dataset assembly, and
potential training. The workflow is intentionally split into independent stages
so that structure generation, MD jobs, VASP jobs, dataset collection, and
training runs can be prepared, submitted, inspected, and repeated separately.

The recommended command order is:

```bash
pesmaker generate run.yaml
pesmaker sample-setup run.yaml
pesmaker select run.yaml
pesmaker label-setup run.yaml
pesmaker collect run.yaml
pesmaker train-setup run.yaml
```

Each stage writes to its own directory:

```text
generated/     # initial supercells, surface slabs, defects, perturbations
sampling/      # MD working directories and sampling submit scripts
selected/      # representative frames selected from MD trajectories
labeling/      # VASP single-point calculation directories
train.xyz      # collected labeled dataset
training/      # potential training input files and submit script
```

## Configuration Overview

Start from `examples/te_defect_md.yaml` and adapt paths for the machine you are
using. A typical 2D Te/Pd-rich workflow looks like this:

```yaml
project: Te_Pd_rich_defect_md

structures:
  include:
    - initial_structures/*.cif

generation:
  supercell: [3, 3, 1]
  output_dir: generated
  surface:
    vacuum: 30.0
    axis: 2
    center: true
  defects:
    include_pristine: true
    single_vacancies:
      elements: [Te]
      max_count: 8
    double_vacancies:
      elements: [Te]
      nearest_first: true
      max_count: 8
    line_defects:
      elements: [Te]
      coordinate_axis: 1
      tolerance: 0.05
      max_count: 4
  perturb:
    pert_num: 20
    cell_pert_fraction: 0.03
    atom_pert_distance: 0.1
    atom_pert_style: normal
    seed: 42
    format: vasp

sampling:
  engine: gpumd
  gpumd_dir: /home/tingliang/software/GPUMD/GPUMD-master-26-05-2026/src
  output_dir: sampling
  run_in: templates/gpumd/run.in
  selection:
    trajectory_pattern: sampling/**/movie.xyz
    output_dir: selected
    min_distance: 0.2
    max_count: 200

labeling:
  engine: vasp
  output_dir: labeling
  input_manifest: selected/manifest.jsonl
  incar: templates/vasp/INCAR
  command: vasp_std

dataset:
  format: extxyz
  split: [0.8, 0.1, 0.1]

training:
  model: nep
  output_dir: training
  dataset: train.xyz
  command: nep

jobs:
  machine: local
  sbatch_templates:
    sampling: templates/sbatch/gpumd.sh
    labeling: templates/sbatch/vasp.sh
    training: templates/sbatch/nep.sh
```

## Stage 1: Generate Surface and Defect Structures

The `generation` section prepares the structural candidates that seed MD or
single-point calculations.

`supercell` controls the repeated cell. For a 2D material, `[3, 3, 1]` expands
the in-plane lattice while keeping the out-of-plane direction unchanged.

`surface` controls slab handling:

```yaml
surface:
  vacuum: 30.0
  axis: 2
  center: true
```

This centers the structure and gives it 30 Angstrom vacuum along the z axis.
Use `axis: 0`, `1`, or `2` for x, y, or z.

`defects` controls structural variants:

```yaml
defects:
  include_pristine: true
  single_vacancies:
    elements: [Te]
    max_count: 8
  double_vacancies:
    elements: [Te]
    nearest_first: true
    max_count: 8
  line_defects:
    elements: [Te]
    coordinate_axis: 1
    tolerance: 0.05
    max_count: 4
```

Supported variant families are:

- `pristine`: the surface supercell without removed atoms.
- `single_vacancies`: remove one atom from the selected elements.
- `double_vacancies`: remove atom pairs, optionally prioritizing nearest pairs.
- `line_defects`: remove rows of atoms grouped by fractional coordinate.

Run:

```bash
pesmaker generate run.yaml
```

Expected output:

```text
generated/
  manifest.jsonl
  Te-mp-19/
    pristine/
      structure_000000.vasp
    single_vacancy_Te_000000/
      structure_000000.vasp
    double_vacancy_Te000000_Te000001/
      structure_000000.vasp
    line_defect_axis1_000/
      structure_000000.vasp
```

The manifest records the source structure, written path, variant name, variant
description, perturbation index, and atom count.

## Stage 2: Prepare Large Model MD Sampling

The `sample-setup` stage creates one MD working directory per generated
structure. For GPUMD:

```yaml
sampling:
  engine: gpumd
  gpumd_dir: /home/tingliang/software/GPUMD/GPUMD-master-26-05-2026/src
  output_dir: sampling
  run_in: templates/gpumd/run.in
```

The default `examples/templates/gpumd/run.in` is:

```text
potential      nep89_20250409.txt
velocity       300

ensemble       npt_scr 300 300 100 0 0 0 20 20 100 1000
time_step      1
dump_thermo    1000
dump_position  3000
run            3000000
```

Run:

```bash
pesmaker sample-setup run.yaml
```

Expected output:

```text
sampling/
  sampling_manifest.jsonl
  md_000000/
    model.xyz
    run.in
    submit.sh
  md_000001/
    model.xyz
    run.in
    submit.sh
```

PESMaker does not launch long MD jobs directly. It prepares reproducible job
folders and submission scripts, then you submit those jobs on the target
machine.

Future engines such as LAMMPS-MACE can use the same stage boundary:

```yaml
sampling:
  engine: lammps-mace
  command: lmp -in in.lammps
```

## Stage 3: Select Representative MD Frames

After MD finishes and trajectories such as `movie.xyz` are available, select a
compact, diverse subset:

```yaml
sampling:
  selection:
    trajectory_pattern: sampling/**/movie.xyz
    output_dir: selected
    min_distance: 0.2
    max_count: 200
```

Run:

```bash
pesmaker select run.yaml
```

This stage uses farthest point sampling. It keeps adding the frame farthest
from the current selected set until either `max_count` is reached or the
nearest-selected distance falls below `min_distance`.

Expected output:

```text
selected/
  selected.xyz
  selected_000000.xyz
  selected_000001.xyz
  manifest.jsonl
```

`selected.xyz` is a multi-frame file for inspection. The single-frame
`selected_XXXXXX.xyz` files are referenced by `manifest.jsonl` and are used by
the labeling setup stage.

## Stage 4: Prepare VASP Single-Point Calculations

The `label-setup` stage turns selected structures into independent VASP
single-point folders.

```yaml
labeling:
  engine: vasp
  output_dir: labeling
  input_manifest: selected/manifest.jsonl
  incar: templates/vasp/INCAR
  command: vasp_std
```

Run:

```bash
pesmaker label-setup run.yaml
```

Expected output:

```text
labeling/
  labeling_manifest.jsonl
  calc_000000/
    POSCAR
    INCAR
    submit.sh
  calc_000001/
    POSCAR
    INCAR
    submit.sh
```

The default `examples/templates/vasp/INCAR` is a conservative single-point
template:

```text
SYSTEM = PESMaker single point
ENCUT = 520
EDIFF = 1E-6
IBRION = -1
NSW = 0
ISMEAR = 0
SIGMA = 0.05
LREAL = Auto
```

You can also provide `potcar`, `kpoints`, or a complete `template_dir` under
`labeling`; these files are copied into every calculation folder.

## Stage 5: Collect the Labeled Dataset

After single-point jobs finish, collect the labeled frames:

```yaml
labeling:
  outcar_pattern: labeling/**/OUTCAR
  dataset_path: train.xyz
```

Run:

```bash
pesmaker collect run.yaml
```

The command reads matched VASP outputs with ASE and writes an extxyz dataset.
This keeps the collection stage separate from job preparation, so failed or
partial calculations can be inspected before rebuilding the dataset.

## Stage 6: Prepare Training

The `train-setup` stage prepares potential-training inputs:

```yaml
training:
  model: nep
  output_dir: training
  dataset: train.xyz
  command: nep
```

Run:

```bash
pesmaker train-setup run.yaml
```

Expected output:

```text
training/
  train.xyz
  nep.in
  submit.sh
```

For non-NEP training backends, set `training.model` and `training.command`.
The stage boundary remains the same, so new trainers can be added without
changing the earlier generation, sampling, labeling, or collection steps.

## Machine-Specific Submission Templates

Cluster submission settings belong in the `jobs` section:

```yaml
jobs:
  machine: cluster-a
  sbatch_templates:
    sampling: templates/sbatch/gpumd.sh
    labeling: templates/sbatch/vasp.sh
    training: templates/sbatch/nep.sh
```

Templates can use these placeholders:

```text
{job_name}    # generated stage job name
{workdir}     # stage working directory
{command}     # engine command, such as gpumd, vasp_std, or nep
```

A GPUMD template can look like:

```bash
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

set -euo pipefail
cd "{workdir}"
{command}
```

Use one template set per machine. This keeps the scientific workflow stable
while allowing queues, modules, partitions, and GPU options to differ across
clusters.

## Practical Checklist

Before running the full workflow:

- Validate the config with `pesmaker validate run.yaml`.
- Inspect the plan with `pesmaker plan run.yaml`.
- Confirm the input structures match the intended chemistry and dimensionality.
- Confirm `generation.surface.vacuum` is large enough for the slab.
- Limit defect `max_count` values at first, then expand after checking outputs.
- Submit a small MD batch before launching the full sampling campaign.
- Check selected frames visually before preparing VASP jobs.
- Confirm `INCAR`, `POTCAR`, and `KPOINTS` are suitable for the target system.
- Collect only completed and converged single-point outputs.
- Keep `train.xyz` and `training/` versioned by project or campaign name.

## Current Scope and Extension Points

The implemented workflow prepares directories, manifests, templates, and
selection outputs. It intentionally does not hide expensive cluster execution
behind one monolithic command.

Current capabilities:

- 2D surface vacuum setup.
- Pristine, single-vacancy, double-vacancy, and line-defect variants.
- Random perturbations for every variant.
- GPUMD sampling setup.
- Farthest point frame selection.
- VASP single-point setup.
- Extxyz dataset collection through ASE.
- NEP training setup.
- Machine-specific sbatch templates.

Planned extension points:

- LAMMPS-MACE sampling templates and engine-specific input writers.
- More chemically aware defect enumeration.
- Dedicated VASP-to-NEP conversion with stricter convergence checks.
- Dataset splitting and metadata-rich training campaign reports.
