# `pesmaker next`

`next` is the normal workflow driver.

It answers one question:

> What can PESMaker do next with this YAML and the files currently on disk?

## Basic Use

```bash
pesmaker validate run.yaml
pesmaker next run.yaml
```

Run `next` again whenever the jobs it asked you to submit have finished.
Use `pesmaker next run.yaml --verbose` only when you want detailed flow
diagnostics.

## Do I Still Need `generate`?

Usually no.

If you use `next`, PESMaker runs local commands for you when their inputs are
ready:

```text
generate
sample-setup
select
scf-setup
collect
train-setup
```

You still run `submit` yourself, because submitting to a cluster is an external
action. `next` only previews submissions and prints the command.

## What It Prints

`next` first prints the PESMaker banner, then a compact workflow position,
what happened, and what to do next:

```text
Next flow
Flow             : generate -> config-needed
Current          : waiting for SCF settings

Work done:
  - Structure generation complete.

Next:
  1. Edit run.next.yaml and set INCAR, POTCAR, VASP, and submit script paths.
  2. Run: pesmaker validate run.next.yaml
  3. Run: pesmaker next run.next.yaml
```

If `next` needs external files, it prints a short waiting message:

```text
Waiting:
  - SCF OUTCAR files are not ready.

Next:
  1. If not submitted yet: pesmaker submit run.yaml
  2. After jobs finish: pesmaker next run.yaml
```

For detailed flow/status/state output, run:

```bash
pesmaker status run.yaml
pesmaker next run.yaml --verbose
```

If no task exists, `next` says:

```text
Complete:
  - No local PESMaker task needs to run now.
```

and exits without writing `.pesmaker/` state.

## Generate-Only Example

If your YAML only contains `structures` and `generation`, `next` generates the
structures and stops. It does not guess your VASP, GPUMD, LAMMPS-MACE, or
submit settings.

```text
generate -> config-needed
```

After generation, PESMaker writes `run.next.yaml`:

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

Then it prints:

```text
Next flow
Flow             : generate -> config-needed
Current          : waiting for SCF settings

Template written : run.next.yaml

Next:
  1. Edit run.next.yaml and set INCAR, POTCAR, VASP, and submit script paths.
  2. Run: pesmaker validate run.next.yaml
  3. Run: pesmaker next run.next.yaml
```

If `run.next.yaml` already exists, PESMaker does not overwrite it.

## Direct SCF Example

With `generation` and `labeling` sections, the first `next` run will usually:

```text
generate -> scf-setup -> submit --dry-run
```

It executes the local stages and stops at the SCF submission preview.

Then it prints something like:

```text
Next:
  1. Review the dry-run log: labeling/scf_submitted_jobs.txt
  2. Submit SCF jobs: pesmaker submit run.yaml
  3. After those jobs finish, run: pesmaker next run.yaml
```

You do not run `pesmaker generate run.yaml` again. It already ran.

After VASP finishes and `OUTCAR` files exist, run:

```bash
pesmaker next run.yaml
```

PESMaker will collect the dataset.

## Migrated Or Retry SCF Folders

If a YAML points directly to an existing VASP calculation tree, and it has no
new `structures`, `input_dir`, or `input_manifest`, `next` treats the current
stage as migration or retry submission rather than SCF setup.

For example:

```yaml
labeling:
  engine: vasp
  output_dir: labeling
  command: /current/path/to/vasp_std

jobs:
  submit_command: sbatch
  skip_completed: true
  check_scf_convergence: true
  sub_file: /current/path/to/sub.sh
```

When `labeling/` already contains calculation folders with `POSCAR`, running
`pesmaker next sub.yaml` does not run `scf-setup`, rewrite inputs, or submit
jobs. It prints:

```text
Next flow
Flow             : SCF-retry submission
Current          : SCF retry submission

Next:
  1. Preview and refresh retry scripts: pesmaker submit sub.yaml --dry-run
  2. Review: cat labeling/scf_submitted_jobs.txt
  3. Submit retry jobs: pesmaker submit sub.yaml
```

The dry-run command classifies each `OUTCAR` and refreshes `submit.sh` only for
folders that need submission or retry. Normally terminated and electronically
converged calculations are left unchanged.

## Sampling Example

With `sampling.engine: gpumd` or `sampling.engine: mace` and
`sampling.selection`, the first `next` run will usually:

```text
generate -> sample-setup -> submit --stage sampling --dry-run
```

It executes the local stages and stops at the sampling submission preview.

Then it prints:

```text
Submit sampling jobs: pesmaker submit run.yaml --stage sampling
```

After GPUMD writes `movie.xyz` or LAMMPS-MACE writes the trajectory configured
in `sampling.selection.trajectory_pattern`, run:

```bash
pesmaker next run.yaml
```

PESMaker will select frames, prepare SCF folders, and preview SCF submission.

## What `next` Never Does

`next` does not call `sbatch`, `qsub`, or any scheduler for real.

This is intentional. You can inspect `submit.sh` and the dry-run log before
spending cluster time.

## When To Use Manual Commands

Use manual commands only when you want direct control over one stage:

```bash
pesmaker generate run.yaml
pesmaker scf-setup run.yaml
pesmaker submit run.yaml --dry-run
```

For the ordinary workflow, use `next`.
