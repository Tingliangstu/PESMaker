# `pesmaker submit`

`submit` submits prepared stage scripts.

It does not create structures or setup folders. Run setup first, or let
[`next`](next.md) prepare the folders.

`next` prints the correct submit command. New users can simply copy the command
printed in the `Next` block.

## SCF Jobs

```bash
pesmaker submit run.yaml
```

No `--stage` means SCF labeling. This shortcut is kept for compatibility and
for experienced users.

## Sampling Jobs

```bash
pesmaker submit run.yaml --stage sampling
```

## Training Jobs

```bash
pesmaker submit run.yaml --stage training
```

## Dry Run

Use dry-run before spending cluster time:

```bash
pesmaker submit run.yaml --dry-run
pesmaker submit run.yaml --stage sampling --dry-run
pesmaker submit run.yaml --stage training --dry-run
```

Dry-run writes a log of the commands but does not call the scheduler.

## Stage Names

`submit` is one command with different stages:

```text
SCF single-point jobs   : pesmaker submit run.yaml
MD sampling jobs        : pesmaker submit run.yaml --stage sampling
NEP training jobs       : pesmaker submit run.yaml --stage training
```

These stages do not conflict. PESMaker reads different manifests and output
folders for each stage.

## Scheduler Settings

```yaml
jobs:
  submit_command: sbatch
  cores_cpu: 36
  gpus: 0
  skip_completed: true
  check_scf_convergence: true
  sub_file:
    sampling: templates/sbatch/gpumd.sh
    labeling: templates/sbatch/vasp_cpu_36.sh
    training: templates/sbatch/nep.sh
```

For VASP SCF jobs, `skip_completed` defaults to `true`. Before submission,
PESMaker classifies each calculation folder:

```text
No OUTCAR                                      -> submit
OUTCAR without the normal VASP footer          -> retry
Normal footer plus SCF nonconvergence marker   -> retry
Normal footer and no SCF failure marker        -> skip
```

The electronic SCF failure marker is:

```text
The electronic self-consistency was not achieved in
```

`check_scf_convergence` defaults to `true`. Set it to `false` only when the
normal VASP timing and accounting footer alone should count as completion.

For every VASP folder that will be submitted or retried, PESMaker rewrites
`submit.sh` from the current `jobs.sub_file`, resource settings, and
`labeling.command`. Completed folders are not modified. This refresh also
happens during `--dry-run`, while the scheduler itself is not called.

To intentionally submit every prepared VASP folder again:

```yaml
jobs:
  submit_command: sbatch
  skip_completed: false
```

With `skip_completed: false`, PESMaker submits every existing script and does
not refresh it. Skipped, retried, and refreshed folders are recorded as
`SKIPPED`, `RETRY`, and `REFRESHED` in `scf_submitted_jobs.txt`.

## Submit Migrated VASP Folders

Use this when a prepared `labeling/` calculation tree was copied from another
machine. Point `output_dir` directly at that existing tree. `input_dir` is not
needed because no new calculation folders are being created:

```yaml
project: migrated_scf

labeling:
  engine: vasp
  output_dir: labeling
  command: /current/machine/path/to/vasp_std

jobs:
  submit_command: sbatch
  cores_cpu: 36
  vasp_kpar: 3
  vasp_ncore: 6
  skip_completed: true
  check_scf_convergence: true
  sub_file: /current/machine/path/to/sub.sh
```

PESMaker discovers calculation folders from `labeling_manifest.jsonl`,
`POSCAR`, or an existing `submit.sh`. Preview first:

```bash
pesmaker submit migrated.yaml --dry-run
pesmaker submit migrated.yaml
```

For this migration workflow, run `submit` directly. Do not run `scf-setup` or
`next`, because those commands may prepare or rewrite calculation inputs.

If your YAML only uses one submit template, `sub_file` can be a single path:

```yaml
jobs:
  submit_command: sbatch
  sub_file: /home/a4s5d/LT/yixiu/MLP_structure/1.Te/1.Material_project_structure/sub.sh
```

If one YAML contains multiple stages, use a mapping so each stage gets the
right template:

```yaml
jobs:
  submit_command: sbatch
  sub_file:
    sampling: templates/sbatch/gpumd.sh
    labeling: templates/sbatch/vasp_cpu_36.sh
    training: templates/sbatch/nep.sh
```

For GPUMD sampling, PESMaker does not rewrite CPU resource directives in the
provided sampling submit template. Keep GPU, partition, time, and any other
cluster-specific settings in `templates/sbatch/gpumd.sh`.

The same rule applies to LAMMPS-MACE sampling templates such as `lammps.sh`.
PESMaker renders that script into every sampling job directory and keeps a
`submit.sh` compatibility copy.

For local server runs without a scheduler, use `bash` when you want the command
to run in the foreground:

```yaml
jobs:
  submit_command: bash
  sub_file: gpumd.sh
```

For GPUMD sampling, PESMaker renders `gpumd.sh` into each job directory and
runs:

```bash
bash gpumd.sh
```

For MACE sampling with `sub_file: lammps.sh`, PESMaker runs:

```bash
bash lammps.sh
```

Use `nohup` when you want the local job to continue in the background:

```yaml
jobs:
  submit_command: nohup
sub_file: gpumd.sh
```

Use `sub_file: lammps.sh` for a local LAMMPS-MACE script.

PESMaker then runs:

```bash
nohup bash gpumd.sh > out 2>&1 &
```

For MACE with `sub_file: lammps.sh`, the command is:

```bash
nohup bash lammps.sh > out 2>&1 &
```

The process runs in the background and writes its output to `out` inside that
job directory. After `nohup` submission, check the local GPU process with
`nvidia-smi` instead of `squeue`.

## Important Rule

Submit from PESMaker with `pesmaker submit ...`, or manually submit from inside
each prepared job directory. The generated scripts assume the calculation
directory is the working directory.
