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

`next` prints a plan before it writes files:

```text
Plan before execution
Start with       : Generate structures from the configured inputs.
Then             : continue through any later local PESMaker stages whose inputs are ready
Stop rule        : stop before real scheduler submission, or when external outputs are missing
Submit behavior  : dry-run only; PESMaker will print the submit command
```

After the run, it prints what actually happened:

```text
Work done this run:
```

These are the local stages that PESMaker already ran.

```text
Stopped because:
```

This tells you why it stopped. The usual reasons are:

- a submit preview was written;
- PESMaker is waiting for `movie.xyz`;
- PESMaker is waiting for `OUTCAR`;
- the workflow is complete.

```text
What you should do next:
```

This is the important part. Run the command printed there.

If no task exists, `next` says:

```text
No PESMaker task needs to run now.
```

and exits without writing `.pesmaker/` state.

## Direct SCF Example

With `generation` and `labeling` sections, the first `next` run will usually:

```text
generate -> scf-setup -> submit --dry-run
```

Before doing that work, it prints the plan. Then it executes the local stages
and stops at the SCF submission preview.

Then it prints something like:

```text
What you should do next:
  1. Review the dry-run log: labeling/scf_submitted_jobs.txt
  2. Submit the prepared jobs: pesmaker submit run.yaml
  3. After those jobs finish, run: pesmaker next run.yaml
```

You do not run `pesmaker generate run.yaml` again. It already ran.

After VASP finishes and `OUTCAR` files exist, run:

```bash
pesmaker next run.yaml
```

PESMaker will collect the dataset.

## Sampling Example

With `sampling.engine: gpumd` and `sampling.selection`, the first `next` run
will usually:

```text
generate -> sample-setup -> submit --stage sampling --dry-run
```

Before doing that work, it prints the plan. Then it executes the local stages
and stops at the sampling submission preview.

Then it prints:

```bash
pesmaker submit run.yaml --stage sampling
```

After GPUMD writes `movie.xyz`, run:

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
