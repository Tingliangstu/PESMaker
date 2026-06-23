# `pesmaker train`

`train` continues from a training YAML. It uses the same smart workflow driver as
`next`, but is easier to remember after dataset collection.

## Use

```bash
pesmaker validate train.yaml
pesmaker train train.yaml
```

For a training YAML, this prepares model-training inputs and then prints the
next submit step when a submit script is ready.

## Example

```yaml
project: train_initial_structure

training:
  model: nep
  output_dir: training
  dataset: train.xyz
  command: nep
```

For two-step NEP training, add:

```yaml
training:
  model: nep
  output_dir: training
  dataset: train.xyz
  two_step: true
```

Run `pesmaker next train.yaml` after the first training job produces
`training/step1/nep.txt`; PESMaker then creates `training/step2` from `step1`
and prepares the second-step input.

## Plot training results

After NEP writes `loss.out`, `energy_train.out`, `force_train.out`, and
`stress_train.out`, run:

```bash
pesmaker -plt train
```

PESMaker writes figures under `plot/`, auto-detecting outputs in the current
directory, `training/step2`, `training/step1`, or `training`.

