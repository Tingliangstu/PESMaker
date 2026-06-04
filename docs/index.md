# PESMaker Manual

PESMaker builds machine-learning potential datasets from structures you already
care about: bulk phases, surfaces, defects, interfaces, and sampled MD frames.

## Start Here

Most users should run:

```bash
pesmaker validate run.yaml
pesmaker next run.yaml
```

`next` is the workflow driver. It runs local PESMaker stages when they are
ready, stops before real scheduler submission, and prints exactly what you
should do next.

You do not need to remember the full stage chain for normal use.

## Read These First

- [Quick Start](usage.md): the shortest practical usage loop.
- [Workflow Guide](ACTIVE_LEARNING_WORKFLOW.md): what `next` does and why it
  stops.
- [Command Manual](commands/index.md): one page per command.
- [Minimal YAML Examples](examples/minimal-yaml.md): small configs by task
  type.

## Normal Flow

```text
input structures
  -> generated structures
  -> optional GPUMD sampling
  -> optional frame selection
  -> VASP SCF setup and submission
  -> extxyz dataset collection
  -> optional NEP training setup
```

`next` decides which part of this flow is active by reading the YAML sections
and checking files such as `generated/manifest.jsonl`, `movie.xyz`, `OUTCAR`,
`train.xyz`, and `training/submit.sh`.

## Manual Flow

Manual commands are still available:

```bash
pesmaker generate run.yaml
pesmaker sample-setup run.yaml
pesmaker select run.yaml
pesmaker scf-setup run.yaml
pesmaker submit run.yaml
pesmaker collect run.yaml
pesmaker train-setup run.yaml
```

Use them when you want to debug one stage. Use `next` when you want the simpler
workflow.
