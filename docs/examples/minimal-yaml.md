# Minimal YAML Examples

This page gives small working shapes for common tasks.

Replace paths such as `POSCAR`, `/path/to/vasp_std`, `/path/to/GPUMD/src`, and
`/path/to/nep.txt` with your local files.

Normal use:

```bash
pesmaker validate run.yaml
pesmaker next run.yaml
```

Do not add a `workflow` field for ordinary runs. PESMaker infers the flow from
the sections present in the YAML.

## Generate Structures Only

Use this when you only want generated structures.

```yaml
project: generate_only

structures:
  - POSCAR

generation:
  output_dir: generated
  supercell: [3, 3, 3]
```

Run:

```bash
pesmaker validate run.yaml
pesmaker generate run.yaml
```

## Generate Then VASP SCF

Use this when generated structures go directly to DFT labeling.

```yaml
project: direct_scf

structures:
  - POSCAR

generation:
  output_dir: generated
  supercell: [3, 3, 3]

labeling:
  engine: vasp
  output_dir: labeling
  incar: templates/vasp/INCAR
  potcar_library: /path/to/VASP/potentials
  command: /path/to/vasp_std
  dataset_path: train.xyz

jobs:
  submit_command: sbatch
  cores_cpu: 36
  sub_file: templates/sbatch/vasp_cpu_36.sh
```

Run:

```bash
pesmaker validate run.yaml
pesmaker next run.yaml
```

When `next` prints the submit command, run it. After VASP finishes, run
`pesmaker next run.yaml` again.

## GPUMD Sampling, Selection, SCF, Training

Use this when generated structures first seed MD sampling.

```yaml
project: sampling_training

structures:
  - POSCAR

generation:
  output_dir: generated
  supercell: [3, 3, 3]

sampling:
  engine: gpumd
  output_dir: sampling
  gpumd_dir: /path/to/GPUMD/src
  potential: /path/to/nep.txt
  temperatures: [300, 600]
  run_steps: 300000
  selection:
    trajectory_pattern: sampling/**/movie.xyz
    output_dir: selected
    descriptor: calorine
    potential: /path/to/nep.txt
    max_count: 200
    min_distance: 0.2
    plot: true

labeling:
  engine: vasp
  output_dir: labeling
  incar: templates/vasp/INCAR
  potcar_library: /path/to/VASP/potentials
  command: /path/to/vasp_std
  dataset_path: train.xyz

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

Run:

```bash
pesmaker validate run.yaml
pesmaker next run.yaml
```

Then follow the `What you should do next` block printed by `next`.

## SCF Setup From Existing Structures

Use this when structures already exist and you only want VASP folders.

```yaml
project: scf_from_existing

labeling:
  engine: vasp
  input_dir: generated
  output_dir: labeling
  incar: templates/vasp/INCAR
  potcar_library: /path/to/VASP/potentials
  command: /path/to/vasp_std

jobs:
  submit_command: sbatch
  cores_cpu: 36
  sub_file: templates/sbatch/vasp_cpu_36.sh
```

Run:

```bash
pesmaker validate run.yaml
pesmaker next run.yaml
```

## Collect Existing OUTCAR Files

Use this when VASP calculations are already finished.

```yaml
project: collect_existing

labeling:
  outcar_pattern: labeling/**/OUTCAR
  dataset_path: train.xyz
```

Run:

```bash
pesmaker validate run.yaml
pesmaker collect run.yaml
```

## Training From Existing Dataset

Use this when `train.xyz` already exists.

```yaml
project: train_existing

training:
  model: nep
  output_dir: training
  dataset: train.xyz
  command: nep

jobs:
  submit_command: sbatch
  cores_cpu: 36
  sub_file:
    training: templates/sbatch/nep.sh
```

Run:

```bash
pesmaker validate run.yaml
pesmaker next run.yaml
```
