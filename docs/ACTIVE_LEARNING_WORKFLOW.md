# Workflow Guide

This page explains the whole PESMaker loop in plain language.

For command syntax, use the [Command Manual](commands/index.md). For small YAML
files, use [Minimal YAML Examples](examples/minimal-yaml.md).

## The Short Answer

Run:

```bash
pesmaker validate run.yaml
pesmaker next run.yaml
```

Then follow the `What you should do next` block printed by `next`.

You normally do not run every stage command by hand. `next` runs local PESMaker
stages for you and stops only when an external scheduler job must be submitted
or finished.

Before `next` writes files, it prints `Plan before execution`. This tells you
the first local step it will start with, that it will continue through later
ready local stages, and that it will stop before real scheduler submission.

## What `next` Does

`next` checks two things:

1. What sections are in `run.yaml`.
2. What files already exist in the project folders.

Then it advances as far as it safely can.

```text
No generated manifest -> generate structures
Sampling configured   -> prepare GPUMD folders
Need sampling jobs    -> write dry-run log and print submit command
movie.xyz exists      -> select frames
Need SCF jobs         -> write dry-run log and print submit command
OUTCAR exists         -> collect train.xyz
Training configured   -> prepare training folder and print submit command
```

If everything local is ready, one `next` run may perform several stages. If an
external result is missing, `next` waits and tells you what file it needs.
If no stage is needed, `next` reports that no PESMaker task needs to run now.

## Direct SCF Flow

Use this when structures should go directly to VASP labeling.

YAML shape:

```text
structures
generation
labeling
jobs
```

User loop:

```bash
pesmaker validate run.yaml
pesmaker next run.yaml
```

The first `next` run usually generates structures, prepares SCF folders, and
writes a dry-run submit log. It then prints:

```bash
pesmaker submit run.yaml
```

After VASP finishes and `OUTCAR` files exist:

```bash
pesmaker next run.yaml
```

PESMaker collects `train.xyz`.

## Sampling and Training Flow

Use this when generated structures should first seed GPUMD MD sampling.

YAML shape:

```text
structures
generation
sampling
sampling.selection
labeling
training
jobs
```

User loop:

```bash
pesmaker validate run.yaml
pesmaker next run.yaml
pesmaker submit run.yaml --stage sampling
```

After GPUMD writes `movie.xyz`:

```bash
pesmaker next run.yaml
pesmaker submit run.yaml
```

After VASP writes `OUTCAR`:

```bash
pesmaker next run.yaml
pesmaker submit run.yaml --stage training
```

You do not need to remember this full chain. `next` prints the correct submit
command at each boundary.

## What If Everything Is Already Done?

If generated structures, trajectories, selected frames, OUTCAR files, and the
dataset already exist, `next` skips finished work and continues to the next
missing local stage.

For example, if `OUTCAR` files already exist and training is configured,
`next` can collect the dataset, prepare training inputs, and stop at the
training submit preview in one run.

## What If I Want Manual Control?

Use command pages:

- [generate](commands/generate.md)
- [sample-setup](commands/sample-setup.md)
- [select](commands/select.md)
- [scf-setup](commands/scf-setup.md)
- [submit](commands/submit.md)
- [collect](commands/collect.md)
- [train-setup](commands/train-setup.md)

Manual mode is useful for debugging templates or inspecting one stage. It is
not required for ordinary use.
