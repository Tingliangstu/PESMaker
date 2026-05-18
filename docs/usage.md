# Usage

## Generate perturbed structures

Create a YAML input file:

```yaml
project: Te_mp_19_test

structures:
  - path/to/input.cif

generation:
  supercell: [4, 4, 4]
  output_dir: runs/Te_mp_19_test/generated
  perturb:
    pert_num: 49
    cell_pert_fraction: 0.03
    atom_pert_distance: 0.1
    atom_pert_style: normal
    seed: 42
    format: vasp
```

Validate the input:

```bash
pesmaker validate examples/perturb.yaml
```

Inspect the planned workflow:

```bash
pesmaker plan examples/perturb.yaml
```

Generate the structures:

```bash
pesmaker generate examples/perturb.yaml
```

The output directory contains generated VASP structure files and a manifest:

```text
runs/Te_mp_19_test/generated/
  input/
    structure_000000.vasp
    structure_000001.vasp
    ...
  manifest.jsonl
```

For multiple CIF files, list them directly:

```yaml
project: Te_batch

structures:
  - Te-mp-19.cif
  - Te-mp-23.cif
  - Te-mp-1009490.cif

generation:
  supercell: [4, 4, 4]
  perturb:
    pert_num: 20
```

For many CIF files in a directory, use an include pattern:

```yaml
structures:
  include:
    - initial_structures/*.cif
```

## Perturbation parameters

- `supercell`: expansion factors along the three lattice directions.
- `pert_num`: number of perturbed structures to generate.
- `cell_pert_fraction`: cell perturbation amplitude.
- `atom_pert_distance`: atomic displacement scale in Angstrom.
- `atom_pert_style`: atomic displacement distribution. Current options are
  `normal`, `uniform`, and `const`.
- `seed`: optional random seed for reproducibility.
- `format`: output format. Current options are `vasp` and `extxyz`.
