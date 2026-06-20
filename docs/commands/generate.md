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
        line_defects:
          elements: [Pd]
          coordinate_axis: 0
          split_by_axis: 2
          max_count: 5
      perturb:
        include_pristine: true
        pert_num: 10
        format: vasp
```

For `line_defects`, `coordinate_axis` is the fractional coordinate kept fixed
when defining the line:

```text
0 -> a axis
1 -> b axis
2 -> c axis
```

For example, `coordinate_axis: 0` builds `const_a` line defects. In a slab with
multiple layers along `c`, this can remove the same line in every layer. Add
`split_by_axis: 2` to split candidates by fractional `c` layer first, so each
line-defect variant removes one layer's line instead of all `c` layers at the
same `a` coordinate. The split tolerance is inferred automatically; set
`split_tolerance` only when the automatic layer grouping is too coarse or too
fine.

For a 2D Te-Pd slab where `axis: 2` is the vacuum direction, a typical Pd line
defect setup is:

```yaml
generation:
  output_dir: generated
  tasks:
    - name: surface_442
      supercell: [4, 4, 2]
      surface:
        vacuum: 30.0
        axis: 2
        center: true
        defects:
          mode: random
          seed: 12345
          line_defects:
            elements: [Pd]
            coordinate_axis: 0
            split_by_axis: 2
            # optional: split_tolerance: 0.03
            max_count: 5
        perturb:
          pert_num: 3
          cell_pert_fraction: 0.03
          atom_pert_distance: 0.1
          atom_pert_style: normal
          seed: 424
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
