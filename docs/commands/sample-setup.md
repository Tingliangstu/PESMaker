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
  temperature: "300-1200"
  run_steps: 300000
  run_in: templates/gpumd/run.in
  selection:
    trajectory_pattern: sampling/**/movie.xyz
    output_dir: selected
    max_count: 200

jobs:
  submit_command: sbatch
  sub_file:
    sampling: templates/sbatch/gpumd.sh
```

For GPUMD, `cores_cpu` is optional. If `jobs.sub_file.sampling` is provided,
PESMaker keeps that submit template's scheduler resource lines and only fills
placeholders such as `{command}`, `{workdir}`, and `{job_name}`. If no submit
template is provided, the generated `submit.sh` simply runs the resolved GPUMD
command, such as `/path/to/GPUMD/src/gpumd`. Put GPU, partition, and walltime
requests directly in `templates/sbatch/gpumd.sh`.

When a GPUMD sampling template is named `gpumd.sh`, PESMaker also writes
`gpumd.sh` in each MD job directory so local submission can run
`bash gpumd.sh`. A `submit.sh` compatibility copy is kept for older workflows.

## Temperature Jobs And Movie Paths

Use one temperature ramp when you want a single MD job that heats or cools:

```yaml
sampling:
  temperature: "300-1200"
```

This creates a folder like:

```text
sampling/md_000000_ramp_300K_to_1200K/movie.xyz
```

Use a temperature list when you want independent MD jobs:

```yaml
sampling:
  temperatures: [300, 600, 900]
```

This creates folders like:

```text
sampling/md_000000_temp_300K/movie.xyz
sampling/md_000000_temp_600K/movie.xyz
sampling/md_000000_temp_900K/movie.xyz
```

Set selection to:

```yaml
sampling:
  selection:
    trajectory_pattern: sampling/**/movie.xyz
```

The `**` means "match through subdirectories". Do not use
`sampling/movie.xyz` unless your `movie.xyz` file is directly inside
`sampling/`.

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
