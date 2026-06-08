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
GPUMD sampling jobs     : pesmaker submit run.yaml --stage sampling
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
  sub_file:
    sampling: templates/sbatch/gpumd.sh
    labeling: templates/sbatch/vasp_cpu_36.sh
    training: templates/sbatch/nep.sh
```

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

Use `nohup` when you want the local job to continue in the background:

```yaml
jobs:
  submit_command: nohup
  sub_file: gpumd.sh
```

PESMaker then runs:

```bash
nohup bash gpumd.sh > out 2>&1 &
```

The process runs in the background and writes its output to `out` inside that
job directory. After `nohup` submission, check the local GPU process with
`nvidia-smi` instead of `squeue`.

## Important Rule

Submit from PESMaker with `pesmaker submit ...`, or manually submit from inside
each prepared job directory. The generated scripts assume the calculation
directory is the working directory.
