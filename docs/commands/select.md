# `pesmaker select`

`select` chooses representative MD frames.

Normal users can let [`next`](next.md) run this stage after GPUMD has written
`movie.xyz`.

## Use

```bash
pesmaker select run.yaml
```

## Minimal YAML

```yaml
sampling:
  selection:
    trajectory_pattern: sampling/**/movie.xyz
    output_dir: selected
    descriptor: calorine
    potential: /path/to/nep.txt
    max_count: 200
    min_distance: 0.2
    plot: true
```

## What It Does

PESMaker reads trajectory frames, builds one descriptor vector per frame, then
uses farthest point sampling to keep frames that spread across descriptor
space.

For quick debugging, use:

```yaml
descriptor: simple
```

For production selection, use Calorine NEP descriptors.

## Outputs

```text
selected/
  selected.xyz
  selected_000000.xyz
  manifest.jsonl
  selection_features.npy
  fps_selection.png
```

Use `selected/manifest.jsonl` as the SCF input manifest.

## Next Step

```bash
pesmaker scf-setup run.yaml
```

If `labeling.input_manifest` is omitted, `next` uses
`selected/manifest.jsonl` after selection.
