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
  training_manifest.jsonl
  submit.sh
```

For NEP, PESMaker writes a starter `nep.in` using GPUMD defaults and infers the
`type` line from `train.xyz` when that file exists. You can override NEP
parameters under `training.nep`:

```yaml
training:
  model: nep
  output_dir: training
  dataset: train.xyz
  nep:
    cutoff: [8, 4]
    neuron: 30
    generation: 100000
```

## Two-step NEP training

Enable two-step training with `training.two_step: true`:

```yaml
training:
  model: nep
  output_dir: training
  dataset: train.xyz
  two_step: true
```

The first run writes `training/step1` with:

```text
lambda_e      0.2
lambda_f      2
lambda_v      0.1
```

After `training/step1/nep.txt` exists, run `pesmaker next train.yaml` again.
PESMaker copies the full `step1` folder to `training/step2`, rewrites `nep.in`,
and switches to:

```text
lambda_e      2
lambda_f      1
lambda_v      1
```

## Next Step

Preview or submit training:

```bash
pesmaker submit run.yaml --stage training --dry-run
pesmaker submit run.yaml --stage training
```

With `next`, PESMaker writes the dry-run log and prints the real submit
command.
