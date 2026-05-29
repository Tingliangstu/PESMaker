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

Use `generation.tasks` when one run should prepare several independent
structure families. Do not repeat the same YAML key in one mapping. For
example, do not write two `generation.supercell` entries; use two task entries
instead.

```yaml
project: Te_Pd_rich_defect_md

structures:
  include:
    - initial_structures/*.cif

generation:
  output_dir: generated
  tasks:
    - name: surface_331
      supercell: [3, 3, 1]
      surface:
        vacuum: 30.0
        axis: 2
        center: true
        defects:
          mode: random
          seed: 42
          single_vacancies:
            elements: [Te]
            max_count: 8
          double_vacancies:
            elements: [Te]
            max_count: 8
          line_defects:
            elements: [Te]
            max_count: 4
        perturb:
          pert_num: 20
          cell_pert_fraction: 0.03
          atom_pert_distance: 0.1
          atom_pert_style: normal
          seed: 42
          format: vasp
    - name: bulk_333
      supercell: [3, 3, 3]
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
  potential: nep89_20250409.txt
  temperatures: [300, 600, 900]
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
single-point calculations. Each entry under `generation.tasks` is independent:
one task can build a 2D surface with random vacancies, while another task can
build a bulk supercell with only perturbations.

`supercell` controls the repeated cell. For a 2D material, `[3, 3, 1]` expands
the in-plane lattice while keeping the out-of-plane direction unchanged.

Example task layout:

```yaml
generation:
  output_dir: generated
  tasks:
    - name: surface_331
      supercell: [3, 3, 1]
      surface:
        vacuum: 30.0
        defects:
          mode: random
          seed: 42
          single_vacancies:
            elements: [Te]
            max_count: 8
          perturb:
            pert_num: 20
    - name: bulk_333
      supercell: [3, 3, 3]
      perturb:
        pert_num: 20
```

The operation order inside a task follows the nesting:

```text
input structure -> supercell -> surface -> defects -> perturb
```

If `perturb` is nested under `surface`, it is applied after the surface is
made. If `perturb` is nested under `defects`, it is applied after defect
structures are made.

`surface` controls slab handling:

```yaml
surface:
  vacuum: 30.0
  axis: 2
  center: true
```

`vacuum` is the vacuum thickness in Angstrom. For most 2D materials, use the
out-of-plane direction.

`axis` selects which cell direction receives the vacuum:

```text
axis: 0  -> x direction
axis: 1  -> y direction
axis: 2  -> z direction
```

Most 2D structures lie in the xy plane, so `axis: 2` is the usual choice.

`center: true` moves the slab to the middle of the vacuum direction. This gives
vacuum on both sides of the layer instead of leaving the structure close to one
cell boundary. For ordinary 2D slab calculations, keep:

```yaml
axis: 2
center: true
```

Only change these values if the material plane is not the xy plane.

Defect and perturbation settings can be nested under `surface`. This means the
children are created from the surface slab, not from the original primitive
structure:

```yaml
surface:
  vacuum: 30.0
  defects:
    mode: random
    seed: 42
    single_vacancies:
      elements: [Te]
      max_count: 8
    double_vacancies:
      elements: [Te]
      max_count: 8
    line_defects:
      elements: [Te]
      max_count: 4
  perturb:
    pert_num: 20
```

Supported variant families are:

- `pristine`: the surface supercell without removed atoms.
- `single_vacancies`: remove one atom from the selected elements.
- `double_vacancies`: remove atom pairs. Nearest pairs are tried first by
  default.
- `line_defects`: remove atom rows. PESMaker infers the row grouping by default.

Vacancies are deterministic unless random mode is requested. For random,
reproducible vacancy positions, set:

```yaml
defects:
  mode: random
  seed: 42
```

`mode: random` applies to all enabled defect families unless a family has its
own `selection`. The same seed gives the same vacancy positions every time,
which is useful for reproducible datasets.

Per-family random selection is also supported:

```yaml
defects:
  single_vacancies:
    selection: random
    seed: 7
    elements: [Te]
    max_count: 8
```

Advanced line-defect controls are available but usually not needed:

- `coordinate_axis`: fractional coordinate used to group rows. For example,
  `coordinate_axis: 1` groups atoms with similar y coordinates, giving line
  defects along the x direction.
- `tolerance`: fractional-coordinate bin width for deciding whether atoms are
  in the same row. If omitted, PESMaker estimates it from the structure.

Advanced double-vacancy control:

- `nearest_first`: when `true`, the closest atom pairs are generated first.
  This is the default, so it can be omitted in normal input files.

Run:

```bash
pesmaker generate run.yaml
```

Expected output:

```text
generated/
  manifest.jsonl
  generation_summary.txt
  surface_331/
    Te-mp-19/
      pristine/
        structure_000000.vasp
      single_vacancy_Te_000000/
        structure_000000.vasp
      double_vacancy_Te000000_Te000001/
        structure_000000.vasp
      line_defect_axis1_000/
        structure_000000.vasp
  bulk_333/
    Te-mp-19/
      pristine/
        structure_000000.vasp
```

The JSONL manifest records task name, source structure, output path, supercell,
variant name, variant description, perturbation index, and atom count. It is
machine-readable and can be dense. For humans, inspect
`generated/generation_summary.txt`, which groups the same information by task,
source structure, and variant folder.

## Stage 2: Prepare Large Model MD Sampling

The `sample-setup` stage creates one MD working directory per generated
structure. For GPUMD:

```yaml
sampling:
  engine: gpumd
  gpumd_dir: /home/tingliang/software/GPUMD/GPUMD-master-26-05-2026/src
  output_dir: sampling
  potential: nep89_20250409.txt
  temperatures: [300, 600, 900]
  run_in: templates/gpumd/run.in
```

`temperatures: [300, 600, 900]` creates one constant-temperature MD job per
temperature for every generated structure. For example, one structure becomes:

```text
sampling/
  md_000000_temp_300K/
  md_000000_temp_600K/
  md_000000_temp_900K/
```

For a heating ramp, use `temperature` instead:

```yaml
sampling:
  engine: gpumd
  potential: nep89_20250409.txt
  temperature: 300-1500
```

This creates one ramp job per generated structure and writes an ensemble line
with start temperature 300 K and end temperature 1500 K.

If `potential` points to an existing file, PESMaker copies it into each MD
working directory. If it is only a filename, PESMaker writes that filename into
`run.in` and assumes you will make the potential available when running GPUMD.

The default `examples/templates/gpumd/run.in` is a template:

```text
potential      {potential}
velocity       {temperature_start}

ensemble       npt_scr {temperature_start} {temperature_end} 100 0 0 0 20 20 100 1000
time_step      1
dump_thermo    1000
dump_position  3000
run            3000000
```

The placeholders are filled by `pesmaker sample-setup`. If you provide a plain
GPUMD `run.in` without placeholders, PESMaker still rewrites `potential`,
`velocity`, and the first two ensemble temperatures from the sampling config.

Run:

```bash
pesmaker sample-setup run.yaml
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
GGA = PE
LREAL = Auto
ENCUT = 650
KSPACING = 0.2
KGAMMA = .TRUE.
NSW = 1
IBRION = -1
ALGO = Normal
EDIFF = 1E-06
SIGMA = 0.02
ISMEAR = 0
PREC = Accurate
NELM = 150
```

You can also provide `potcar`, `kpoints`, or a complete `template_dir` under
`labeling`; these files are copied into every calculation folder.

For generated VASP files, PESMaker preserves the generated folder structure by
default while dropping the `.vasp` suffix from each calculation folder:

```yaml
labeling:
  engine: vasp
  output_dir: labeling
  incar: templates/vasp/INCAR
  potcar_library: /home/a4s5d/software/VASP/potentials
  command: /home/a4s5d/software/VASP/CPU_vasp.6.6.0/bin/vasp_std

jobs:
  submit_command: sbatch
  sbatch_templates:
    labeling: templates/sbatch/vasp_cpu_36.sh
```

This writes folders such as:

```text
labeling/
  mp-105_Te/
    structure_000000/
      POSCAR
      structure_000000.vasp-bak
      INCAR
      POTCAR
      POTCAR.spec
      submit.sh
```

The original generated structure is backed up by default. Set
`backup_source: false` under `labeling` only if those backups are not wanted.
If `potcar_library` is set, PESMaker writes `POTCAR` automatically from
the built-in VASP-recommended potential table for each element. The table is
based on the bold recommended entries in the VASP Wiki page
[`Choosing pseudopotentials`](https://www.vasp.at/wiki/index.php/Choosing_pseudopotentials),
under `Recommended PAW potentials`.

For ordinary PBE calculations, PESMaker follows the `Standard PBE potentials
(potpaw.64)` recommendations. For example, Te uses `Te`, Na uses `Na_pv`, K
uses `K_sv`, and Ga uses `Ga_d`; users do not need to manually handle common
`_pv`, `_sv`, or `_d` choices. For GW potentials, set `gw_potcar: true` and
PESMaker will use the recommended `GW potentials (potpaw.64)` directories such
as `Te_GW` or `Na_sv_GW`.

Each calculation folder also contains `POTCAR.spec`, which records the exact
potential directories concatenated into `POTCAR`.

Submit the prepared jobs with:

```bash
pesmaker submit run.yaml --stage labeling
```

Use `--dry-run` first to write the scheduler commands without calling `sbatch`.

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
- Multiple independent generation tasks under one `generation` section.
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
