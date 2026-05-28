# Command-Line Interface

PESMaker exposes a `pesmaker` command after installation.

## `pesmaker init`

Write a starter YAML configuration:

```bash
pesmaker init pesmaker.yaml
```

The command refuses to overwrite an existing file.

## `pesmaker validate`

Validate a YAML or TOML configuration file:

```bash
pesmaker validate examples/perturb.yaml
```

## `pesmaker plan`

Print a human-readable workflow plan:

```bash
pesmaker plan examples/perturb.yaml
```

## `pesmaker generate`

Generate supercells and perturbed structures:

```bash
pesmaker generate examples/perturb.yaml
```

The current implementation writes structure files and a `manifest.jsonl` file in
the configured `generation.output_dir`.

## `pesmaker md_sampling`

Prepare MD sampling directories, default `run.in` content, and `submit.sh`
files from generated structures:

```bash
pesmaker md_sampling examples/te_defect_md.yaml
```

Aliases: `sample-setup`, `sampling`, and `md-sampling`.

## `pesmaker select`

Select representative MD trajectory frames with farthest point sampling:

```bash
pesmaker select examples/te_defect_md.yaml
```

## `pesmaker label-setup`

Prepare VASP single-point calculation folders:

```bash
pesmaker label-setup examples/te_defect_md.yaml
```

## `pesmaker collect`

Collect completed single-point outputs into an extxyz training set:

```bash
pesmaker collect examples/te_defect_md.yaml
```

## `pesmaker train-setup`

Prepare potential-training inputs and submission script:

```bash
pesmaker train-setup examples/te_defect_md.yaml
```
