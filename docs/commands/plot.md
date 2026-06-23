# `pesmaker plot`

`plot` writes diagnostic figures from training or simulation output files.

## Use

```bash
pesmaker plot train
```

`pesmaker plot train` reads GPUMD/NEP training outputs and writes:

- `plot/nep_train.png`: loss curves plus energy, force, and stress parity plots.
- `plot/nep_parity.png`: larger parity plots with marginal distributions.

The command looks for `energy_train.out` and `force_train.out` in the current
directory first. If they are not there, it checks `training/step2`,
`training/step1`, and `training`.
Figures are written at 650 dpi.

## Options

```bash
pesmaker plot train --source training/step2 --output-dir plot
```

- `--source`: directory containing NEP output files. Default: current directory
  with automatic training-folder detection.
- `--output-dir`: directory for generated figures. Default: `plot`.

The loss panel uses logarithmic generation and loss axes. Parity panels use the
same x and y coordinate range so deviations from the diagonal are visually
comparable.
