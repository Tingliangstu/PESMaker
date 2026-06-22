# `pesmaker collect`

`collect` reads finished VASP outputs and writes an extxyz dataset.

Normal users can let [`next`](next.md) run this stage after `OUTCAR` files
exist.

## Use

```bash
pesmaker collect run.yaml
```

## Minimal YAML

```yaml
project: collect_initial_structure

 collecting:
  dataset_path: train.xyz
  test_data_frames: 0
```

By default, `collect` recursively finds every `OUTCAR` below the directory where
you run the YAML. Use `outcar_pattern` or `outcar_patterns` only when you want
to restrict the search. `collecting` is the recommended section name for
collect-only configs; the older `labeling` section name is still accepted for
backward compatibility.

For example:

```yaml
project: collect_existing_scf

collecting:
  outcar_patterns:
    - "1.Te/**/run_vasp_scf/**/OUTCAR"
    - "2.Pb/**/run_vasp_scf/**/OUTCAR"
    - "3.Te-Pd/**/run_vasp_scf/**/OUTCAR"
    - "4.bulk_pristine/**/run_vasp_scf/**/OUTCAR"
  dataset_path: train.xyz
```

## What It Does

PESMaker reads every matched `OUTCAR` with a VASP parser based on the older
`vasp2nep.py` workflow, then writes `train.xyz` and optionally `test.xyz` in
labeled extended xyz format.

The second line of each frame contains `Lattice`, `Energy`,
`Properties=species:S:1:pos:R:3:force:R:3`, optional `Virial`, `pbc`,
`Config_type`, and optional `weight`. It also writes a plain text summary report
with collected counts and skipped nonconverged OUTCAR counts, grouped by the
closest ancestor directory containing `sub.yaml` and the first child directory
below it.

By default, `check_scf_convergence` is `true`, so OUTCAR files containing VASP's
electronic self-consistency failure marker are skipped. `include_virial: false`
removes virial fields from the written extxyz frames. `test_data_frames`
randomly moves that many collected frames from `dataset_path` into
`test_dataset_path`, using `test_seed` for reproducibility. `include_weight`
defaults to `false`; write it only when you intentionally need a per-frame
`weight` field.

The virial parser automatically detects whether the OUTCAR virial block looks
standard or VDW/MBD-adjusted. The command prints a dataset-level
`VDW/MBD detected` line so users can see whether the collected calculations are
consistent.

`Config_type` is inferred from the OUTCAR path. PESMaker keeps meaningful path
parts such as element/system names, `Material_project_structure`, `perturbed`,
`MD`, `bulk_pristine`, phase labels, and `mp-*` identifiers, while dropping
generic calculation folders such as `run_vasp_scf` and `calc_000000`. For
example, `1.Te/1.Material_project_structure/run_vasp_scf/mp-105_Te/calc_000000`
becomes `Config_type=1.Te_1.Material_project_structure_mp-105_Te`.

## Output

```text
train.xyz
train_collection_summary.txt
```

The command also prints a structure count table:

For MACE, this same extxyz file can be used by setting its data keys. MACE
defaults to `energy` and `forces`, but supports `--energy_key`, `--forces_key`,
`--stress_key`, and `--virials_key`.

```bash
mace_run_train \
  --train_file train.xyz \
  --energy_key Energy \
  --forces_key force \
  --virials_key Virial
```

```text
Labeled dataset collection complete.

Structures:
  Config_type                         Frames
  -----------                         ------
  1.Te_1.Material_project_structure_mp-105_Te   128
  2.Pb_3.Pd-bulk_MD_bulk                        96
  3.Te-Pd_2.2D-Te-Pd_1.perturbed_beta           240

Total structures : 464
Train structures : 464
Test structures  : 0
Train dataset    : train.xyz
Summary          : train_collection_summary.txt
VDW/MBD detected : no (464/464 OUTCAR files use the standard virial block)
Nonconverged OUTCAR skipped : 0
```

or the path set by:

```yaml
collecting:
  dataset_path: path/to/train.xyz
```

## Next Step

Prepare training:

```bash
pesmaker train-setup run.yaml
```

With `next`, training setup runs automatically when `training` is configured.
