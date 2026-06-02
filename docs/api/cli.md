# Command-Line Interface

PESMaker exposes a `pesmaker` command after installation.

## `pesmaker init`

Write a starter YAML configuration:

```bash
pesmaker init pesmaker.yaml
```

The command refuses to overwrite an existing file.

## `pesmaker validate`

Validate a YAML configuration file:

```bash
pesmaker validate examples/perturb.yaml
```

## `pesmaker generate`

Generate supercells, surfaces, defects, and optional perturbed structures:

```bash
pesmaker generate examples/perturb.yaml
```

The current implementation writes structure files and a `manifest.jsonl` file in
the configured `generation.output_dir`. Omit `generation.perturb`, or set
`generation.perturb.pert_num: 0`, when only expanded pristine structures are
needed. Pristine outputs are named with the supercell, such as
`pristine_3x3x3.vasp`.

## `pesmaker sample-setup`

Prepare sampling job directories, default `run.in` content, and `submit.sh`
files from generated structures:

```bash
pesmaker sample-setup examples/te_defect_md.yaml
```

## `pesmaker select`

Select representative MD trajectory frames with farthest point sampling:

```bash
pesmaker select examples/te_defect_md.yaml
```

## `pesmaker scf-setup`

Prepare SCF calculation folders:

```bash
pesmaker scf-setup examples/te_defect_md.yaml
```

For a follow-up run that only labels structures already written by
`pesmaker generate`, the config can omit `structures`. Use
`labeling.input_dir` to point at the existing generated-structure directory.

```yaml
project: Te_bulk_mp

labeling:
  engine: vasp
  output_dir: labeling
  input_dir: generated
  incar: templates/vasp/INCAR
  potcar_library: /home/a4s5d/software/VASP/potentials
  command: /home/a4s5d/software/VASP/CPU_vasp.6.6.0/bin/vasp_std

jobs:
  submit_command: sbatch
  cores_cpu: 36
  # vasp_kpar: 2
  # vasp_ncore: 6
  sub_file: templates/sbatch/vasp_cpu_36.sh
```

`labeling.input_dir` may contain a `manifest.jsonl`. If it does not, PESMaker
recursively scans that folder for `POSCAR`, `CONTCAR`, `*.vasp`, `*.poscar`,
`*.cif`, `*.extxyz`, and `*.xyz` files. Each prepared job keeps the source
path, uses the source path without its suffix as the calculation folder name,
and records resource fields such as `cores_cpu`, `gpus`, `vasp_kpar`, and
`vasp_ncore`.

For CPU VASP jobs, `KPAR` defaults to `2` when `jobs.cores_cpu` is even, and
PESMaker chooses `NCORE` inside each KPAR group; for example, `cores_cpu: 36`
generates `KPAR = 2` and `NCORE = 3`. PESMaker writes `NCORE`, not legacy
`NPAR`. Override them with `jobs.vasp_kpar` and `jobs.vasp_ncore` when needed.
The default CPU VASP submit script uses `mpirun {command}` and assumes it is
submitted from the calculation directory. `pesmaker submit` does this
automatically. For GPU jobs, set `gpus: <count>` under `jobs`.

For user-provided `jobs.sub_file` templates, placeholders are optional for
common Slurm fields. PESMaker rewrites existing `#SBATCH --job-name`,
`#SBATCH --ntasks`, and VASP run-command lines from the generated folder name,
`jobs.cores_cpu`, and `labeling.command`.

## `pesmaker submit`

Submit prepared `submit.sh` files. By default this submits SCF jobs:

```bash
pesmaker submit examples/te_defect_md.yaml
```

Use `--stage sampling` or `--stage training` for those stages. Use `--dry-run`
to record the commands without invoking the scheduler.

## `pesmaker collect`

Collect completed SCF outputs into an extxyz training set:

```bash
pesmaker collect examples/te_defect_md.yaml
```

## `pesmaker train-setup`

Prepare potential-training inputs and submission script:

```bash
pesmaker train-setup examples/te_defect_md.yaml
```
