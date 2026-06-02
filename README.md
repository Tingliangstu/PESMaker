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

Direct generation and DFT labeling. Use this when generated structures should
go straight to VASP SCF labeling:

```bash
pesmaker validate run.yaml           # check YAML syntax and required fields
pesmaker generate run.yaml           # build supercells, surfaces, defects, and optional perturbations
pesmaker scf-setup run.yaml          # prepare VASP SCF folders with POSCAR/INCAR/submit.sh
pesmaker submit run.yaml --dry-run   # preview SCF/VASP submission commands
pesmaker submit run.yaml             # submit prepared SCF/VASP jobs
pesmaker collect run.yaml            # collect finished SCF outputs into a training dataset
```

Full sampling, labeling, and training loop. Use this when generated structures
first seed MD sampling before DFT labeling:

```bash
pesmaker validate run.yaml                  # check YAML syntax and required fields
pesmaker generate run.yaml                  # build initial structures for sampling
pesmaker sample-setup run.yaml              # prepare GPUMD MD folders and submit.sh files
pesmaker submit run.yaml --stage sampling   # submit prepared MD sampling jobs
pesmaker select run.yaml                    # select representative frames from MD trajectories
pesmaker scf-setup run.yaml                 # prepare VASP SCF folders for selected structures
pesmaker submit run.yaml                    # submit prepared SCF/VASP jobs
pesmaker collect run.yaml                   # collect finished SCF outputs into a training dataset
pesmaker train-setup run.yaml               # prepare NEP training input files and submit.sh
pesmaker submit run.yaml --stage training   # submit prepared NEP training jobs
```

`submit` always submits `submit.sh` files that were prepared by an earlier
setup command. Without `--stage`, it submits the SCF labeling stage by default.
Use `--stage sampling` for MD jobs and `--stage training` for training jobs.
Each stage writes normal files and folders that can be inspected, edited, and
rerun independently.

```text
generated/   # supercells, surfaces, defects, optional perturbations
sampling/    # GPUMD MD job folders and submit scripts
selected/    # representative frames selected from trajectories
labeling/    # VASP SCF calculation folders
train.xyz    # collected labeled dataset
training/    # NEP training input folder and submit script
```

## Example Config

For pure supercell expansion, omit the `perturb` section. PESMaker writes one
expanded `pristine_<supercell>.vasp` file per input structure, such as
`pristine_3x3x3.vasp`:

```yaml
project: Te_bulk_mp

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
