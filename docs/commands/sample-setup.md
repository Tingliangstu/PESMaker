# `pesmaker sample-setup`

`sample-setup` prepares MD sampling folders. PESMaker currently supports
GPUMD/NEP sampling and LAMMPS-MACE sampling for MACE-omat-small and other MACE
MLIAP models.

Normal users can let [`next`](next.md) run this stage. Use `sample-setup`
directly when you want to inspect generated MD inputs before using the smart
driver.

## Use

```bash
pesmaker sample-setup run.yaml
```

## GPUMD Minimal YAML

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

If `sampling.run_in` already contains a `run` line, PESMaker keeps that step
count unless `sampling.run_steps` is explicitly set in the YAML. For
non-orthogonal cells, PESMaker adjusts `ensemble npt_scr` to the triclinic
format and prints a short warning.

Set `sampling.preserve_run_in: true` when you want PESMaker to copy your GPUMD
`run_in` exactly as written. In that mode PESMaker still prepares `model.xyz`,
potential files, manifests, and submit scripts, but it does not change
`potential`, `velocity`, `ensemble`, or `run` lines. `sampling.run_in` is
required when this option is enabled.

## LAMMPS-MACE Sampling

Use `sampling.engine: mace` when generated structures should be sampled with a
MACE foundation model through LAMMPS MLIAP. PESMaker prepares `data.in`, fills a
small set of placeholders in your LAMMPS input, and copies your `lammps.sh`
submit script into every job folder. The LAMMPS command and MD physics remain
in your files.

Recommended workflow: write and test your own LAMMPS `run_in` for the target
machine and material first, then let PESMaker render it for each generated
structure. This keeps MACE model settings, D3, NPT/NVT choices, dump cadence,
thermo output, timestep, and total run length under your direct control.
PESMaker's job is to supply the generated `data.in`, the correct element order,
the selected MACE model path, and temperature values.

If you want PESMaker to leave the LAMMPS input completely untouched, set
`sampling.preserve_run_in: true`. Then PESMaker copies `sampling.run_in` exactly
as written into every sampling job directory and does not replace placeholders,
change temperatures, change elements, or convert NVT to NPT. Use this when your
LAMMPS input is already fully configured and self-contained. `sampling.run_in`
is required in this mode.

```yaml
project: mace_sampling

sampling:
  engine: mace
  output_dir: sampling
  potential: /path/to/mace-omat-0-small.model-mliap_lammps.pt
  run_in: templates/lammps/in.run_mace_npt
  # Optional: set true to copy run_in verbatim with no PESMaker edits.
  # preserve_run_in: true
  temperature: "300-1200"
  selection:
    trajectory_pattern: sampling/**/*.lammpstrj
    output_dir: selected
    descriptor_model: /path/to/mace-omat-0-small.model
    min_distance: 0.0
    max_count: 200

jobs:
  submit_command: nohup
  sub_file: templates/lammps/lammps.sh
```

`sampling.potential` and `sampling.selection.descriptor_model` serve different
purposes. `potential` is the MLIAP model exported for LAMMPS. The
`descriptor_model` must be the native MACE model that ASE's `MACECalculator`
can load. PESMaker uses that model's invariant descriptors for FPS. Install
MACE in the PESMaker environment with:

```bash
python -m pip install mace-torch
```

No `selection.descriptor`, `selection.device`, or `selection.num_layers` entry
is required. For `engine: mace`, PESMaker automatically calls
`MACECalculator.get_descriptors(..., invariants_only=True, num_layers=-1)`.
The default device is `cuda`, and descriptors are averaged separately for each
element before FPS. The command output states that MACE model descriptors were
used and shows descriptor progress while the user waits.

`lammps.sh` can contain the full machine-specific LAMMPS command:

```bash
#!/bin/bash

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MACE_TIME=true

mpirun -np 1 /path/to/lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.run_mace_npt
```

Before using `pesmaker next`, it is a good idea to run one rendered sampling
folder manually with `bash lammps.sh` or the exact LAMMPS command from the
script. This catches LAMMPS build, Kokkos, GPU, MACE model, and D3 availability
issues before submitting a batch of jobs.

For MACE sampling templates, the recommended style is to use placeholders.
PESMaker replaces these placeholders:

| Placeholder | Meaning |
| --- | --- |
| `{data_file}` | Generated LAMMPS data file, usually `data.in`. |
| `{potential}` | MACE `.model-mliap_lammps.pt` path. |
| `{elements}` | Element list in LAMMPS atom-type order. |
| `{temperature_start}` | Start temperature from `sampling.temperature`. |
| `{temperature_end}` | End temperature from `sampling.temperature`. |
| `{trajectory}` | Trajectory filename, default `mace.lammpstrj`. |

LAMMPS variables such as `${Tstart}`, `${Pdamp}`, and `${dt_dump}` are left
untouched.

If you use an older literal LAMMPS input without placeholders, PESMaker also
adjusts common MACE lines conservatively. Treat this as a compatibility helper
for existing scripts, not the preferred interface:

- `read_data` is changed to the generated `data.in`.
- `mliap unified ... 0` is changed to `sampling.potential`.
- `pair_coeff` and `dump_modify ... element ...` element lists are changed to
  the element order in `data.in`.
- `variable T`, `variable Tstart`, and `variable Tstop` are changed from
  `sampling.temperature`.
- `fix ... nvt ...` or `fix ... npt ...` is changed to an NPT fix using the
  original fix ID, so a later `unfix` line remains valid.
  PESMaker uses the same cell-shape detection as GPUMD setup: 3D orthogonal
  uses `x/y/z`, 3D triclinic uses `x/y/z/xy/xz/yz`, and 2D cells use the two
  in-plane lengths plus the in-plane tilt term.

For anything outside those common lines, PESMaker keeps the user input as
written.

### MACE input without D3

```lammps
units         metal
dimension     3
boundary      p p p
atom_style    atomic
atom_modify   map yes
newton        on

variable      ts         equal 0.001
variable      Tstart     equal {temperature_start}
variable      Tstop      equal {temperature_end}
variable      Tdamp      equal ${ts}*100
variable      Pdamp      equal ${ts}*1000
variable      P_0        equal 0.0
variable      dt_dump    equal 3000
variable      dt_thermo  equal 1000

read_data     {data_file}

pair_style    mliap unified {potential} 0
pair_coeff    * * {elements}

dump          myDump all custom ${dt_dump} {trajectory} id element x y z
dump_modify   myDump sort id element {elements}
```

### MACE input with D3

```lammps
pair_style    hybrid/overlay &
              mliap unified {potential} 0 &
              dispersion/d3 bj pbe 12.0 6.0

pair_coeff    * * mliap {elements}
pair_coeff    * * dispersion/d3 {elements}
```

### NPT examples

For 2D materials with the material in the `x-y` plane:

```lammps
fix MD all npt temp ${Tstart} ${Tstop} ${Tdamp} &
    x  ${P_0} ${P_0} ${Pdamp} &
    y  ${P_0} ${P_0} ${Pdamp} &
    xy ${P_0} ${P_0} ${Pdamp} &
    couple none
```

For 3D orthogonal cells:

```lammps
fix MD all npt temp ${Tstart} ${Tstop} ${Tdamp} &
    x ${P_0} ${P_0} ${Pdamp} &
    y ${P_0} ${P_0} ${Pdamp} &
    z ${P_0} ${P_0} ${Pdamp} &
    couple none
```

For 3D triclinic cells:

```lammps
fix MD all npt temp ${Tstart} ${Tstop} ${Tdamp} &
    x  ${P_0} ${P_0} ${Pdamp} &
    y  ${P_0} ${P_0} ${Pdamp} &
    z  ${P_0} ${P_0} ${Pdamp} &
    xy ${P_0} ${P_0} ${Pdamp} &
    xz ${P_0} ${P_0} ${Pdamp} &
    yz ${P_0} ${P_0} ${Pdamp} &
    couple none
```

The MACE/LAMMPS interface is documented in the
[MACE LAMMPS MLIAP guide](https://mace-docs.readthedocs.io/en/latest/guide/lammps_mliap.html).
The NPT thermostat/barostat keywords are documented in
[LAMMPS fix npt](https://docs.lammps.org/fix_nh.html).

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

For GPUMD, set selection to:

```yaml
sampling:
  selection:
    trajectory_pattern: sampling/**/movie.xyz
```

PESMaker automatically calculates Calorine NEP descriptors with the same
`sampling.potential` used for GPUMD. The YAML does not need a separate
`descriptor` or selection potential. The command output states that GPUMD/NEP
descriptors were used and shows descriptor progress while the user waits.

The `**` means "match through subdirectories". Do not use
`sampling/movie.xyz` unless your `movie.xyz` file is directly inside
`sampling/`.

For MACE, set selection to the LAMMPS trajectory name used by your template:

```yaml
sampling:
  selection:
    trajectory_pattern: sampling/**/*.lammpstrj
    descriptor_model: /path/to/native-mace.model
```

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
    data.in                 # MACE/LAMMPS
    in.run_mace_npt         # MACE/LAMMPS
    model.xyz               # GPUMD
    run.in                  # GPUMD
    lammps.sh               # MACE when jobs.sub_file is lammps.sh
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
