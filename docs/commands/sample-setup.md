# `pesmaker sample-setup`

`sample-setup` prepares GPUMD MD sampling folders.

Normal users can let [`next`](next.md) run this stage. Use `sample-setup`
directly when you want to inspect GPUMD inputs before using the smart driver.

## Use

```bash
pesmaker sample-setup run.yaml
```

## Minimal YAML

```yaml
project: sampling_run

sampling:
  engine: gpumd
  output_dir: sampling
  gpumd_dir: /path/to/GPUMD/src
  potential: /path/to/nep.txt
  temperatures: [300, 600]
  run_steps: 300000
  run_in: templates/gpumd/run.in
```

## Inputs

PESMaker looks for structures in this order:

1. `sampling.input_manifest`
2. `sampling.input_dir`
3. `generation.output_dir`
4. local `generated/`
5. `runs/<project>/generated`

## Outputs

```text
sampling/
  sampling_manifest.jsonl
  md_000000_temp_300K/
    model.xyz
    run.in
    submit.sh
```

## Next Step

Preview or submit sampling jobs:

```bash
pesmaker submit run.yaml --stage sampling --dry-run
pesmaker submit run.yaml --stage sampling
```

With `next`, PESMaker writes the dry-run log and prints the real submit
command for you.
