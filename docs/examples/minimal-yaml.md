# Minimal YAML Examples

This page gives small working shapes for common tasks.

Replace paths such as `POSCAR`, `/path/to/vasp_std`, `/path/to/GPUMD/src`,
`/path/to/nep.txt`, `/path/to/lmp`, and MACE model paths with your local files.

Normal use:

```bash
pesmaker validate run.yaml
pesmaker next run.yaml
```

Do not add a `workflow` field for ordinary runs. PESMaker infers the flow from
the sections present in the YAML.

## Generate Structures First

Use this when you first want to generate structures, then decide how to label
them.

```yaml
project: 2D_Te_defect

structures:
  - POSCAR

generation:
  output_dir: generated
  supercell: [3, 3, 3]
```

Run:

```bash
pesmaker validate run.yaml
pesmaker next run.yaml
```

PESMaker generates `generated/` and writes `run.next.yaml`. Edit that file
before running the next stage:

```yaml
project: 2D_Te_defect

labeling:
  engine: vasp
  output_dir: run_vasp_scf
  input_dir: generated
  incar: /path/to/INCAR
  potcar_library: /path/to/VASP/potentials
  command: /path/to/vasp_std

jobs:
  submit_command: sbatch
  cores_cpu: 36
  vasp_kpar: 3
  vasp_ncore: 6
  skip_completed: true
  check_scf_convergence: true
  sub_file: /path/to/sub.sh
```

Then continue:

```bash
pesmaker validate run.next.yaml
pesmaker next run.next.yaml
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
  temperature: "300-1200"
  run_steps: 300000
  selection:
    trajectory_pattern: sampling/**/movie.xyz
    output_dir: selected
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

Then follow the `Next` block printed by `next`.

## LAMMPS-MACE Sampling, Selection, SCF

Use this when generated structures first seed MACE-omat-small or another MACE
MLIAP model through LAMMPS.

```yaml
project: mace_sampling_training

structures:
  - POSCAR

generation:
  output_dir: generated
  supercell: [4, 4, 4]

sampling:
  engine: mace
  output_dir: sampling
  potential: /path/to/mace-omat-0-small.model-mliap_lammps.pt
  run_in: templates/lammps/in.run_mace_npt
  # Set true if your LAMMPS input is fully configured and should be copied
  # without PESMaker placeholder replacement or automatic MACE/NPT edits.
  # preserve_run_in: true
  temperature: "300-1200"
  selection:
    trajectory_pattern: sampling/**/*.lammpstrj
    output_dir: selected
    descriptor_model: /path/to/mace-omat-0-small.model
    min_distance: 0.0
    max_count: 200

labeling:
  engine: vasp
  output_dir: labeling
  incar: templates/vasp/INCAR
  potcar_library: /path/to/VASP/potentials
  command: /path/to/vasp_std
  dataset_path: train.xyz

jobs:
  submit_command: nohup
  sub_file:
    sampling: templates/lammps/lammps.sh
    labeling: templates/sbatch/vasp_cpu_36.sh
```

`templates/lammps/lammps.sh` should contain the real LAMMPS command for your
machine:

```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export MACE_TIME=true

mpirun -np 1 /path/to/lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.run_mace_npt
```

The LAMMPS input template controls NPT/NVT, D3, dump frequency, thermo
frequency, and run length. PESMaker only fills `{data_file}`, `{potential}`,
`{elements}`, `{temperature_start}`, `{temperature_end}`, and `{trajectory}`.
The recommended workflow is to write and test `templates/lammps/in.run_mace_npt`
yourself for your LAMMPS/MACE build, then let PESMaker render that proven input
for every generated structure.
Set `sampling.preserve_run_in: true` if that proven input should be copied
verbatim.
See [`sample-setup`](../commands/sample-setup.md#lammps-mace-sampling) for
complete MACE templates and links to the MACE/LAMMPS references.

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
