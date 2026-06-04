# Usage

This page gives the short command-level workflow. For detailed configuration
and stage behavior, use the
[Active Learning Workflow](ACTIVE_LEARNING_WORKFLOW.md) manual.

## Install and Check

```bash
python -m pip install -e .
pesmaker --help
```

Create a starter YAML file:

```bash
pesmaker init run.yaml
```

Validate before running expensive stages:

```bash
pesmaker validate run.yaml
```

Run the smart workflow driver:

```bash
pesmaker next run.yaml
```

`next` infers the flow from the YAML sections and existing artifacts. It runs
local setup stages until it reaches a submit preview, waits for external
outputs, or completes the workflow. It never submits jobs for real; it writes
dry-run logs and prints the matching `pesmaker submit ...` command.

Check what would happen next without writing files:

```bash
pesmaker status run.yaml
```

Practical loop:

1. Write the YAML sections for the work you want: `generation`, optional
   `sampling.selection`, `labeling`, optional `training`.
2. Run `pesmaker validate run.yaml`.
3. Run `pesmaker next run.yaml`.
4. If `next` prints `Submit jobs`, inspect the dry-run log and run that exact
   command.
5. After the external jobs finish, run `pesmaker next run.yaml` again.

## Direct Generate to SCF Workflow

Use this path when generated structures should go directly to DFT labeling.
No `workflow` field is needed; include `generation` and `labeling` sections,
then run `next`.

Recommended command:

```bash
pesmaker validate run.yaml
pesmaker next run.yaml
```

Manual stage commands:

```bash
pesmaker generate run.yaml
pesmaker scf-setup run.yaml
pesmaker submit run.yaml --dry-run   # preview SCF/VASP submissions
pesmaker submit run.yaml             # submit SCF/VASP jobs
pesmaker collect run.yaml
```

Supercell-only generation does not need a `perturb` section:

```yaml
project: Te_bulk_mp

structures:
  include:
    - initial_structures/*.cif

generation:
  supercell: [3, 3, 3]
  output_dir: generated
```

Minimal structure-generation and SCF setup example:

```yaml
project: Te_surface_scf

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
            max_count: 4
        perturb:
          include_pristine: true
          pert_num: 10
          format: vasp

labeling:
  engine: vasp
  output_dir: labeling
  input_dir: generated
  incar: templates/vasp/INCAR
  potcar_library: /path/to/VASP/potentials
  command: /path/to/vasp_std

jobs:
  submit_command: sbatch
  cores_cpu: 36
  sub_file: templates/sbatch/vasp_cpu_36.sh
```

## Full Sampling and Training Workflow

Use this path when generated structures first seed MD sampling. No `workflow`
field is needed; include `sampling` and `sampling.selection`, and `next` will
insert sampling and selection before SCF.

Recommended command:

```bash
pesmaker validate run.yaml
pesmaker next run.yaml
```

Manual stage commands:

```bash
pesmaker generate run.yaml
pesmaker sample-setup run.yaml
pesmaker submit run.yaml --stage sampling   # submit MD sampling jobs
pesmaker select run.yaml
pesmaker scf-setup run.yaml
pesmaker submit run.yaml                    # submit SCF/VASP jobs
pesmaker collect run.yaml
pesmaker train-setup run.yaml
pesmaker submit run.yaml --stage training   # submit NEP training jobs
```

The three `submit` commands are different. `--stage sampling` submits MD jobs
prepared by `sample-setup`; the default `submit` submits SCF/VASP jobs prepared
by `scf-setup`; `--stage training` submits training jobs prepared by
`train-setup`.

Add these sections to the config:

```yaml
sampling:
  engine: gpumd
  output_dir: sampling
  gpumd_dir: /path/to/GPUMD/src
  potential: ../potentials/nep/nep89_20250409/nep89_20250409.txt
  temperatures: [300, 600, 900]
  run_steps: 3000000
  ensemble_mode: auto
  run_in: templates/gpumd/run.in
  selection:
    trajectory_pattern: sampling/**/movie.xyz
    output_dir: selected
    descriptor: calorine
    potential: ../potentials/nep/nep89_20250409/nep89_20250409.txt
    min_distance: 0.2
    max_count: 200
    plot: true

labeling:
  engine: vasp
  output_dir: labeling
  input_manifest: selected/manifest.jsonl
  incar: templates/vasp/INCAR
  potcar_library: /path/to/VASP/potentials
  command: /path/to/vasp_std

training:
  model: nep
  output_dir: training
  dataset: train.xyz
  command: nep

jobs:
  submit_command: sbatch
  cores_cpu: 36
  sub_file:
    sampling: templates/sbatch/gpumd.sh
    labeling: templates/sbatch/vasp_cpu_36.sh
    training: templates/sbatch/nep.sh
```

`sampling` is the canonical section name. PESMaker also accepts `MD_sampling`
and `md_sampling` as aliases for the same section. For the shortest FPS setup,
omit `selection.min_distance`; the default is `0.0`, so selection is controlled
by `selection.max_count` unless duplicate descriptor vectors leave no farther
frame to add.

## Common Outputs

```text
generated/   # generated structures and manifest
sampling/    # GPUMD job folders
selected/    # selected MD frames
labeling/    # VASP SCF job folders
train.xyz    # collected labeled dataset
training/    # NEP training setup
```

Use `pesmaker submit run.yaml --dry-run` whenever you change machine templates
or resource settings.
