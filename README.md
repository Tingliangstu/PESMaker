# PESMaker

PESMaker, short for **Potential Energy Surface Maker**, is a lightweight
workflow package for building application-oriented datasets for
machine-learned interatomic potentials from user-provided atomistic
structures.

It is designed for practical materials workflows where you already have
meaningful structures, such as bulk phases, surfaces, defects, interfaces, or
reaction candidates, and need to turn them into reproducible DFT labeling jobs
and training inputs.

## Why PESMaker

PESMaker helps you move from structures to MLIP training data without turning
the workflow into one large hidden script:

- generate supercells, surface slabs, vacancies, line defects, and optional
  perturbed structures from CIF, POSCAR, XYZ, and other ASE-readable inputs;
- keep every generated structure traceable through `manifest.jsonl` and
  human-readable summaries;
- prepare VASP SCF folders with `POSCAR`, `INCAR`, optional `POTCAR`, and
  `submit.sh`;
- submit prepared jobs through machine-specific Slurm templates;
- collect completed SCF outputs into an extxyz training set;
- prepare NEP training folders while keeping sampling, labeling, collection,
  and training as separate inspectable stages.

PESMaker is user-structure-driven rather than random-search-first. The intended
use case is targeted dataset construction for batteries, solid electrolytes,
thermal transport, alloys, 2D materials, defects, surfaces, catalysis, and
reactions.

## Workflow

For most runs, validate the YAML and then let `next` advance the workflow until
it reaches a submit preview, waits for external results, or finishes the local
steps:

```bash
pesmaker validate run.yaml
pesmaker next run.yaml
```

Set the high-level path in YAML:

```yaml
workflow: direct-scf          # generated structures -> VASP labeling -> collect
# or
workflow: sampling-training   # generate -> GPUMD sampling -> select -> label -> train
```

`workflow: auto` is the default. It uses `sampling-training` when GPUMD sampling
and `sampling.selection` are configured; otherwise it uses `direct-scf`.

`next` never submits jobs for real. At a sampling, SCF, or training submit
boundary it writes a dry-run log, records the gate in
`.pesmaker/<project>/next_state.json`, and prints the command to submit
manually.

Manual direct generation and DFT labeling:

```bash
pesmaker generate run.yaml
pesmaker scf-setup run.yaml
pesmaker submit run.yaml --dry-run
pesmaker submit run.yaml
pesmaker collect run.yaml
```

Manual sampling, labeling, and training loop:

```bash
pesmaker generate run.yaml
pesmaker sample-setup run.yaml
pesmaker submit run.yaml --stage sampling
pesmaker select run.yaml
pesmaker scf-setup run.yaml
pesmaker submit run.yaml
pesmaker collect run.yaml
pesmaker train-setup run.yaml
pesmaker submit run.yaml --stage training
```

`submit` always submits `submit.sh` files prepared by an earlier setup command.
Without `--stage`, it submits the SCF labeling stage by default.

```text
generated/   # supercells, surfaces, defects, optional perturbations
sampling/    # GPUMD MD job folders and submit scripts
selected/    # representative frames selected from trajectories
labeling/    # VASP SCF calculation folders
train.xyz    # collected labeled dataset
training/    # NEP training input folder and submit script
```

## Example Config

Minimal direct SCF run:

```yaml
project: direct_scf
workflow: direct-scf

structures:
  - POSCAR

generation:
  output_dir: generated
  supercell: [3, 3, 3]

labeling:
  engine: vasp
  output_dir: labeling
  incar: templates/vasp/INCAR
  command: /path/to/vasp_std
  dataset_path: train.xyz

jobs:
  submit_command: sbatch
  cores_cpu: 36
```

Run it with:

```bash
pesmaker validate run.yaml
pesmaker next run.yaml
```

Minimal sampling and training run:

```yaml
project: sampling_training
workflow: sampling-training

structures:
  - POSCAR

generation:
  output_dir: generated

sampling:
  engine: gpumd
  output_dir: sampling
  gpumd_dir: /path/to/GPUMD/src
  potential: /path/to/nep.txt
  temperatures: [300]
  selection:
    trajectory_pattern: sampling/**/movie.xyz
    output_dir: selected
    descriptor: calorine
    potential: /path/to/nep.txt
    max_count: 200

labeling:
  engine: vasp
  output_dir: labeling
  incar: templates/vasp/INCAR
  command: /path/to/vasp_std
  dataset_path: train.xyz

training:
  model: nep
  output_dir: training
  dataset: train.xyz

jobs:
  submit_command: sbatch
  cores_cpu: 36
```

Run `pesmaker next run.yaml` repeatedly after submitted jobs produce their
outputs. For more minimal YAML examples by task type, see
[`docs/ACTIVE_LEARNING_WORKFLOW.md`](docs/ACTIVE_LEARNING_WORKFLOW.md).

For pure supercell expansion, omit the `perturb` section. PESMaker writes one
expanded `pristine_<supercell>.vasp` file per input structure, such as
`pristine_3x3x3.vasp`:

```yaml
project: Te_bulk_mp
workflow: direct-scf

structures:
  include:
    - initial_structures/*.cif

generation:
  supercell: [3, 3, 3]
  output_dir: generated
```

For defect variants, the pristine file also includes the variant name, for
example `pristine_3x3x3_single_vacancy_Te_000001.vasp`.

```yaml
project: Te_Pd_rich_defect_md
workflow: sampling-training

structures:
  include:
    - initial_structures/*.cif

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
            max_count: 8
          double_vacancies:
            elements: [Te]
            max_count: 8
          line_defects:
            elements: [Te]
            max_count: 4
        perturb:
          include_pristine: true
          pert_num: 20
          cell_pert_fraction: 0.03
          atom_pert_distance: 0.1
          atom_pert_style: normal
          seed: 42
          format: vasp
    - name: bulk_333
      supercell: [3, 3, 3]
      perturb:
        include_pristine: true
        pert_num: 20
        cell_pert_fraction: 0.03
        atom_pert_distance: 0.1
        atom_pert_style: normal
        seed: 42
        format: vasp

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
  gpus: 0
  sub_file: templates/sbatch/vasp_cpu_36.sh
```

## Installation

```bash
git clone https://github.com/Tingliangstu/PESMaker.git
cd PESMaker
python -m pip install -e .
```

For development and documentation:

```bash
python -m pip install -e ".[dev,docs]"
```

For Calorine NEP descriptor-based frame selection:

```bash
python -m pip install -e ".[selection]"
```

Check the command-line interface:

```bash
pesmaker --help
```

On Windows, if `pesmaker` is not on `PATH`, run it through Python:

```powershell
python -m pesmaker --help
```

Minimum runtime dependencies are Python 3.10+, ASE, NumPy, and PyYAML.

## Documentation

The full manual is in [`docs/ACTIVE_LEARNING_WORKFLOW.md`](docs/ACTIVE_LEARNING_WORKFLOW.md).
The MkDocs site can be served locally with:

```bash
mkdocs serve
```

The intended GitHub Pages URL is:

```text
https://Tingliangstu.github.io/PESMaker/
```

## Current Scope

Current implemented stages cover structure generation, GPUMD sampling setup,
trajectory-frame selection, VASP SCF setup, scheduler submission, extxyz dataset
collection, and NEP training setup. Future backends such as LAMMPS-MACE can use
the same stage boundaries.

## License

PESMaker is free software distributed under the GNU General Public License,
version 3 of the License, or (at your option) any later version. See
[`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) for details.
