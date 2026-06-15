# `pesmaker scf-setup`

`scf-setup` prepares VASP SCF calculation folders.

Normal users can let [`next`](next.md) run this stage.

## Use

```bash
pesmaker scf-setup run.yaml
```

## Minimal YAML

```yaml
project: scf_run

labeling:
  engine: vasp
  output_dir: labeling
  input_dir: generated
  incar: templates/vasp/INCAR
  potcar_library: /path/to/VASP/potentials
  command: /path/to/vasp_std

jobs:
  submit_command: sbatch
  cores_cpu: 36
  sub_file: templates/sbatch/vasp_cpu_36.sh
```

## Inputs

PESMaker looks for structures in this order:

1. `labeling.input_manifest`
2. `labeling.input_dir`
3. `generation.output_dir`
4. local `generated/`
5. `runs/<project>/generated`

## Outputs

```text
labeling/
  labeling_manifest.jsonl
  ...
    POSCAR
    INCAR
    POTCAR
    submit.sh
```

## VASP Resource Fields

```yaml
jobs:
  cores_cpu: 36
  vasp_kpar: 2
  vasp_ncore: 6
  skip_completed: true
```

If `vasp_kpar` and `vasp_ncore` are omitted, PESMaker chooses conservative
values from `cores_cpu`.

`skip_completed` defaults to `true`. During SCF submission, PESMaker skips
folders whose `OUTCAR` contains the normal VASP completion footer.

## Next Step

Preview or submit SCF jobs:

```bash
pesmaker submit run.yaml --dry-run
pesmaker submit run.yaml
```

With `next`, PESMaker writes the dry-run log and prints the real submit
command.
