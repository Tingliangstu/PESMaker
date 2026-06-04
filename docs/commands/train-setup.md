# `pesmaker train-setup`

`train-setup` prepares model training inputs and a training `submit.sh`.

Normal users can let [`next`](next.md) run this stage after the dataset exists.

## Use

```bash
pesmaker train-setup run.yaml
```

## Minimal YAML

```yaml
project: train_run

training:
  model: nep
  output_dir: training
  dataset: train.xyz
  command: nep

jobs:
  submit_command: sbatch
  cores_cpu: 36
  sub_file:
    training: templates/sbatch/nep.sh
```

## Outputs

```text
training/
  train.xyz
  nep.in
  submit.sh
```

## Next Step

Preview or submit training:

```bash
pesmaker submit run.yaml --stage training --dry-run
pesmaker submit run.yaml --stage training
```

With `next`, PESMaker writes the dry-run log and prints the real submit
command.
