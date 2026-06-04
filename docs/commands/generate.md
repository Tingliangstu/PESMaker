# `pesmaker generate`

`generate` builds structure candidates.

Normal users can let [`next`](next.md) run this stage. Use `generate` directly
when you only want to inspect generated structures.

## Use

```bash
pesmaker generate run.yaml
```

## Minimal YAML

```yaml
project: generate_only

structures:
  - POSCAR

generation:
  output_dir: generated
  supercell: [3, 3, 3]
```

## What It Does

For each input structure, PESMaker can apply:

```text
supercell -> surface/vacuum -> defects -> perturbations
```

Use `generation.tasks` when one YAML file needs several structure families.

## Useful Fields

```yaml
generation:
  output_dir: generated
  tasks:
    - name: surface_331
      supercell: [3, 3, 1]
      surface:
        vacuum: 30.0
        axis: 2
        center: true
      defects:
        mode: random
        seed: 42
        single_vacancies:
          elements: [Te]
          max_count: 4
      perturb:
        include_pristine: true
        pert_num: 10
        format: vasp
```

## Outputs

```text
generated/
  manifest.jsonl
  generation_summary.txt
  ...
```

Read `generation_summary.txt` first. It is the quick human-readable summary.
Later stages read `manifest.jsonl`.

## Next Step

To prepare VASP folders manually:

```bash
pesmaker scf-setup run.yaml
```

For the normal flow:

```bash
pesmaker next run.yaml
```
