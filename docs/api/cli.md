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

Generate supercells and perturbed structures:

```bash
pesmaker generate examples/perturb.yaml
```

The current implementation writes structure files and a `manifest.jsonl` file in
the configured `generation.output_dir`.

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
`pesmaker generate`, the config can omit `structures`. Without
`labeling.input_manifest` or `generation.output_dir`, `scf-setup` reads
`generated/manifest.jsonl` from the current working directory when it exists.

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
  sub_file: templates/sbatch/vasp_cpu_36.sh
```

`labeling.input_dir` may contain a `manifest.jsonl`. If it does not, PESMaker
recursively scans that folder for structure files and marks each prepared job
in `labeling/labeling_manifest.jsonl`.

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
