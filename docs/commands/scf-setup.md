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

When an input directory contains `manifest.jsonl`, PESMaker uses that manifest
because it preserves exact frame indices for multi-frame files created by
`select`. If no manifest is present, PESMaker recursively scans the directory
for structure files and prepares one SCF folder per discovered structure.

Scanned input files can be named `POSCAR`, `CONTCAR`, or `XDATCAR`, or use one
of these suffixes:

```text
.cif
.extxyz
.poscar
.traj
.vasp
.xyz
```

For scanned multi-frame `.xyz`, `.extxyz`, `.traj`, or `XDATCAR` files,
PESMaker prepares one SCF job per frame. This lets users point
`labeling.input_dir` at a manually prepared folder even when the folder was not
created by PESMaker and has no `manifest.jsonl`.

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
  check_scf_convergence: true
```

If `vasp_kpar` and `vasp_ncore` are omitted, PESMaker chooses conservative
values from `cores_cpu`.

When `jobs.sub_file` is provided without resource fields such as `cores_cpu`,
`nodes`, `gpus`, `vasp_kpar`, or `vasp_ncore`, PESMaker keeps the user submit
script content unchanged except for explicit placeholders such as `{command}`,
`{job_name}`, or `{workdir}`. Add resource fields only when you want PESMaker
to refresh matching scheduler lines and VASP launch commands in the generated
`submit.sh`.

For CPU VASP jobs with `cores_cpu` or `nodes` set, PESMaker uses
`mpirun -np <nodes * cores_cpu> <labeling.command>` when `labeling.command` is
only the VASP executable. If the command starts with `mpirun` or `mpiexec` but
does not include `-np`, `-n`, `--np`, or `--ntasks`, PESMaker inserts the same
rank count. Existing rank-count options are kept unchanged.

For GPU VASP, request GPUs with `jobs.gpus` and usually omit `vasp_kpar` and
`vasp_ncore`. When `gpus` is greater than zero, PESMaker does not add CPU VASP
parallel tags to `INCAR`. If `labeling.command` is only the VASP executable,
PESMaker runs it as `mpirun -np <jobs.gpus> <labeling.command>`. If the command
starts with `mpirun` or `mpiexec` but does not include an explicit rank count,
PESMaker inserts `-np <jobs.gpus>`. Commands that start with `srun` are kept
unchanged.

```yaml
project: 2D_Te_MD

labeling:
  engine: vasp
  output_dir: run_vasp_scf
  input_dir: selected
  incar: /home/a4s5d/LT/yixiu/MLP_structure/1.Te/1.Material_project_structure/INCAR
  potcar_library: /home/a4s5d/software/VASP/potentials
  command: /data/software/vasp6.4-gpu/bin/vasp_std

jobs:
  submit_command: sbatch
  cores_cpu: 6
  gpus: 1
  skip_completed: true
  check_scf_convergence: true
  sub_file: /home/a4s5d/LT/yixiu/MLP_structure/1.Te/1.Material_project_structure/sub_gpu.sh
```

This requests one GPU per prepared SCF folder. `cores_cpu: 6` should match the
CPU task count in the submit template, for example `#SBATCH -n 6`.

During SCF submission, `skip_completed` and `check_scf_convergence` both
default to `true`. PESMaker skips only VASP folders that terminated normally
without the electronic SCF nonconvergence marker.

## Next Step

Preview or submit SCF jobs:

```bash
pesmaker submit run.yaml --dry-run
pesmaker submit run.yaml
```

With `next`, PESMaker writes the dry-run log and prints the real submit
command.
