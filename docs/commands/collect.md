# `pesmaker collect`

`collect` reads finished VASP outputs and writes an extxyz dataset.

Normal users can let [`next`](next.md) run this stage after `OUTCAR` files
exist.

## Use

```bash
pesmaker next collect.yaml
```

`pesmaker collect collect.yaml` is the manual single-stage command. The normal
workflow entry point is `pesmaker next collect.yaml`.

## Recommended YAML

```yaml
project: collect_initial_structure

collecting:
  dataset_path: train.xyz
  test_path: test.xyz
  test_data_frames: 0
  include_virial: true
```

By default, `collect` recursively finds every `OUTCAR` below the directory where
you run the YAML. Save the file as `collect.yaml` in the directory that contains
your finished VASP calculation folders, then run:

```bash
pesmaker validate collect.yaml
pesmaker next collect.yaml
```

For large folders, collection may take a while. Before parsing starts, PESMaker
prints:

```text
Starting dataset collection:
  - Engine      : VASP
  - Input files : VASP OUTCAR files
  - PESMaker is scanning files and parsing labeled results.
  - Large datasets can take several minutes. Please wait.
```

This writes:

```text
train.xyz
train_collection_summary.txt
```

`test_data_frames: 0` means no `test.xyz` is written. `test_path` is the test
dataset file name that will be used only when `test_data_frames` is positive.
Set `test_data_frames` to a positive integer only when you want PESMaker to
randomly move that many structures into `test.xyz`.

## Optional Filters

Use `outcar_pattern` or `outcar_patterns` only when you want to restrict the
search.

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
with collected counts and skipped OUTCAR counts, grouped by source directory.
For grouping, PESMaker first uses the closest ancestor directory containing
`sub.yaml`. If no `sub.yaml` is found, it falls back to the path before the
first calculation folder such as `run_vasp_scf` or `calc_000000`.

Defaults:

- `check_vasp_completion: true`, so OUTCAR files without VASP's normal
  completion marker, `General timing and accounting informations for this job`,
  are skipped as incomplete.
- `check_scf_convergence: true`, so OUTCAR files containing VASP's electronic
  self-consistency failure marker are skipped.
- `include_virial: true`, so PESMaker writes `Virial` when the OUTCAR contains
  the `FORCE on cell =-STRESS` block.
- `config_type: true`, so PESMaker writes a source label inferred from the
  OUTCAR path.
- `include_weight: false`; write it only when you intentionally need a
  per-frame `weight` field.

The virial parser automatically detects whether the OUTCAR virial block looks
standard or VDW/MBD-adjusted. The command prints a dataset-level
`Van der Waals correction` line inferred from the parsed virial block layout.
This is a consistency check from OUTCAR content; the INCAR remains the final
source of truth for how the VASP calculations were configured.

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

`dataset_path` is the output dataset file name. If you set it to another path,
PESMaker writes the training dataset there:

```yaml
collecting:
  dataset_path: path/to/train.xyz
```

The summary report is a normal text file. It records matched OUTCAR files,
collected OUTCAR files, total structures, train/test structures, skipped
incomplete OUTCAR files, skipped nonconverged OUTCAR files, unreadable OUTCAR
files, structure counts grouped by source directory, detailed `Config_type`
counts, and skipped OUTCAR paths.

Source grouping does not require PESMaker-created folders. PESMaker first uses
the closest ancestor directory containing `sub.yaml` when present. If no
`sub.yaml` exists, it infers the source from the path before folders such as
`run_vasp_scf`, `*_run_vasp_scf`, or `calc_*`.

The command prints a compact screen summary:

```text
Labeled dataset collection complete.

Totals:
  OUTCAR matched       : 669
  OUTCAR collected     : 669
  Structures written   : 669
  Incomplete skipped   : 0
  Nonconverged skipped : 0
  Unreadable skipped   : 0

Datasets:
  Train : train.xyz (669 structures)
  Test  : not written (test_data_frames = 0)

Summary file : train_collection_summary.txt

Sources:
  Source groups : 17
  source                                OUTCARs  structures
  ------------------------------------  ------  ----------
  1.Te/1.Material_project_structure         27          27
  1.Te/2.2D-Te/1.perturbed                 147         147
  1.Te/2.2D-Te/2.MD                         30          30
  1.Te/3.bulk_MD                            35          35
  2.Pb/1.Material_project_structure         12          12
  2.Pb/2.2D_Pb/1.perturbed                  98          98
  2.Pb/2.2D_Pb/2.MD                         40          40
  2.Pb/3.Pd-bulk_MD                         34          34
  3.Te-Pd/1.Material_project_structure      42          42
  3.Te-Pd/2.2D-Te-Pd/1.perturbed            98          98
  3.Te-Pd/2.2D-Te-Pd/2.MD                   40          40
  3.Te-Pd/3.Te-Pd-Bulk_MD                   39          39
  4.bulk_pristine/1.Te                       9           9
  4.bulk_pristine/2.Pb                       4           4
  4.bulk_pristine/3.small_Te-Pd              5           5
  4.bulk_pristine/4.large_Te-Pd              7           7
  4.bulk_pristine/5.1_large_Te-Pd            2           2

  Detected OUTCARs total : 669
  Collected structures total : 669

Van der Waals correction : not detected in 669/669 collected calculation(s)
```

The output is not tied to one training code. It is labeled extended xyz. GPUMD
can read the `Energy`, `force`, `Virial`, `Config_type`, and optional `weight`
fields used by NEP training. MACE can use the same file by setting its data keys.
MACE defaults to `energy` and `forces`, but supports `--energy_key`,
`--forces_key`, `--stress_key`, and `--virials_key`.

```bash
mace_run_train \
  --train_file train.xyz \
  --energy_key Energy \
  --forces_key force \
  --virials_key Virial
```

## Next Step

Create `train.yaml` with a `training` section. For example:

```yaml
project: train_initial_structure

training:
  model: nep
  output_dir: training
  dataset: train.xyz
  command: nep
```

Then run:

```bash
pesmaker validate train.yaml
pesmaker train train.yaml
```
