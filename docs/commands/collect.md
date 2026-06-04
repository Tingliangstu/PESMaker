# `pesmaker collect`

`collect` reads finished VASP outputs and writes an extxyz dataset.

Normal users can let [`next`](next.md) run this stage after `OUTCAR` files
exist.

## Use

```bash
pesmaker collect run.yaml
```

## Minimal YAML

```yaml
project: collect_run

labeling:
  outcar_pattern: labeling/**/OUTCAR
  dataset_path: train.xyz
```

## What It Does

PESMaker reads every matched `OUTCAR` through ASE and writes the frames to one
extxyz file.

## Output

```text
train.xyz
```

or the path set by:

```yaml
labeling:
  dataset_path: path/to/train.xyz
```

## Next Step

Prepare training:

```bash
pesmaker train-setup run.yaml
```

With `next`, training setup runs automatically when `training` is configured.
