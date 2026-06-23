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

