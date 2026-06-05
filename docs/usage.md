# Quick Start

This is the shortest way to use PESMaker.

## 1. Create or Edit `run.yaml`

Start from a template:

```bash
pesmaker init run.yaml
```

Edit the paths for your structures, VASP, GPUMD, NEP potential, and scheduler
templates.

Minimal examples are here:

- [Minimal YAML Examples](examples/minimal-yaml.md)

## 2. Validate the YAML

```bash
pesmaker validate run.yaml
```

Fix any reported YAML problem before continuing.

## 3. Run Smart Next

```bash
pesmaker next run.yaml
```

This is the normal command.

You do not need to run `generate`, `sample-setup`, `select`, `scf-setup`,
`collect`, or `train-setup` one by one. `next` runs those local stages when
their inputs are ready.

Before it writes files, `next` prints `Plan before execution`. Read that block
if you want to see what PESMaker is about to do. It still stops before any real
cluster submission.

## 4. Follow the Printed Next Step

When jobs need to be submitted, `next` prints a block like:

```text
What you should do next:
  1. Review the dry-run log: labeling/scf_submitted_jobs.txt
  2. Submit the prepared jobs: pesmaker submit run.yaml
  3. After those jobs finish, run: pesmaker next run.yaml
```

Do exactly that.

PESMaker does not submit jobs automatically. It only previews submission and
prints the real submit command.

If there is no next task, `next` says that no PESMaker task needs to run now
and exits without doing extra work.

## 5. Repeat After External Jobs Finish

After GPUMD creates `movie.xyz` or VASP creates `OUTCAR`, run:

```bash
pesmaker next run.yaml
```

PESMaker continues from the files already on disk.

## Check Without Writing Files

If you are unsure what will happen:

```bash
pesmaker status run.yaml
```

`status` is read-only.

## Manual Mode

Manual commands are still available for debugging:

```bash
pesmaker generate run.yaml
pesmaker scf-setup run.yaml
pesmaker submit run.yaml --dry-run
```

For normal production work, use `next`.
