![PESMaker potential energy surface banner](https://raw.githubusercontent.com/Tingliangstu/PESMaker/main/docs/assets/pesmaker-banner.svg)

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

## Installation

PESMaker requires Python 3.10 or newer.

### Install From PyPI

After the first PESMaker release is published to PyPI, install the stable
package with:

```bash
python -m pip install pesmaker
pesmaker --help
```

Install an optional descriptor backend when needed:

```bash
# GPUMD trajectories: Calorine NEP descriptors
python -m pip install "pesmaker[selection]"

# MACE trajectories: MACECalculator descriptors
python -m pip install "pesmaker[mace]"

# Both descriptor backends
python -m pip install "pesmaker[selection,mace]"
```

### Install The Latest GitHub Version

This works before the first PyPI release and installs the current `main`
branch:

```bash
python -m pip install "git+https://github.com/Tingliangstu/PESMaker.git@main"
pesmaker --help
```

Optional extras also work with the GitHub URL:

```bash
python -m pip install "pesmaker[mace] @ git+https://github.com/Tingliangstu/PESMaker.git@main"
```

### Install From A Source Checkout

Use this method for development or offline installation:

```bash
git clone https://github.com/Tingliangstu/PESMaker.git
cd PESMaker
python -m pip install .
pesmaker --help
```

No internet: copy or unzip the PESMaker source folder, enter that folder, then
run `python -m pip install .`.

Installation test:

```bash
python -m pytest
```

This runs the test files in `tests/` and checks that the installed Python
package, config parser, structure tools, CLI functions, and workflow logic work.
If `pytest` is not installed, install the small test dependency once:

```bash
python -m pip install ".[dev]"
python -m pytest
```

## Updating an Existing Checkout

For a PyPI installation:

```bash
python -m pip install --upgrade pesmaker
```

For a source checkout on `main`:

```bash
git pull --ff-only
python -m pip install .
```

If you are not sure where you are:

```bash
cd ~/software/PESMaker
git switch main
git pull --ff-only
python -m pip install .
```

No internet: copy or unzip a newer PESMaker source folder, then reinstall:

```bash
cd /path/to/PESMaker
python -m pip install .
```

## Workflow

For most runs, validate the YAML and then let `next` advance the workflow until
it reaches a submit preview, waits for external results, or finishes the local
steps:

```bash
pesmaker validate run.yaml
pesmaker next run.yaml
```

You do not need to write a workflow name. PESMaker infers the flow from the
YAML sections and existing artifacts. For example, a config with
`sampling.engine` and `sampling.selection` will prepare sampling, wait for MD
trajectories, select frames, then continue to SCF and training if those
sections are configured.

If the YAML only contains structure generation settings, `next` generates the
structures and stops. It writes `run.next.yaml` as a simple VASP SCF template;
edit the INCAR, POTCAR, VASP, and submit-script paths there, then run
`pesmaker next run.next.yaml`.

`next` never submits jobs for real. At a sampling, SCF, or training submit
boundary it writes a dry-run log, records the gate in
`.pesmaker/<project>/next_state.json`, and prints the command to submit
manually. The printed line names the stage, for example `Submit SCF jobs` or
`Submit sampling jobs`.

The default `next` output is intentionally short: it shows the current
`Next flow`, `Work done`, and `Next`. Use `pesmaker status run.yaml` or
`pesmaker next run.yaml --verbose` when you want detailed flow diagnostics.

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

`submit` always submits the stage scripts prepared by an earlier setup command.
Without `--stage`, it submits the SCF labeling stage by default.

```text
generated/   # supercells, surfaces, defects, optional perturbations
sampling/    # GPUMD or LAMMPS-MACE MD job folders and submit scripts
selected/    # representative frames selected from trajectories
labeling/    # VASP SCF calculation folders
train.xyz    # collected labeled dataset
training/    # NEP training input folder and submit script
```

## Examples

Minimal YAML examples are grouped by task type in the documentation:

See the [minimal YAML examples](https://Tingliangstu.github.io/PESMaker/examples/minimal-yaml/).

## Documentation

Start with the [Quick Start](https://Tingliangstu.github.io/PESMaker/usage/).
The online manual also contains the
[command reference](https://Tingliangstu.github.io/PESMaker/commands/) and
[minimal YAML examples](https://Tingliangstu.github.io/PESMaker/examples/minimal-yaml/).

The intended GitHub Pages URL is:

```text
https://Tingliangstu.github.io/PESMaker/
```

## Current Scope

Current implemented stages cover structure generation, GPUMD sampling setup,
LAMMPS-MACE sampling setup, engine-matched NEP or MACE descriptor FPS,
VASP SCF setup, scheduler submission, extxyz dataset collection, and NEP
training setup.

## License

PESMaker is free software distributed under the GNU General Public License,
version 3 of the License, or (at your option) any later version. See
[LICENSE](https://github.com/Tingliangstu/PESMaker/blob/main/LICENSE) and
[NOTICE](https://github.com/Tingliangstu/PESMaker/blob/main/NOTICE) for details.
