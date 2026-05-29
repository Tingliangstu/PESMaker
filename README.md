# PESMaker

PESMaker is a planned lightweight workflow package for building application-oriented
datasets and machine-learned interatomic potentials from user-provided atomistic
structures.

The working full name is **Potential Energy Surface Maker**.

## License

PESMaker is free software distributed under the **GNU General Public License**,
version 3 of the License, or (at your option) any later version.

See [LICENSE](LICENSE) and [NOTICE](NOTICE) for details.

## Scope

PESMaker focuses on this path:

```text
initial structures
  -> structure generation and targeted sampling
  -> candidate filtering
  -> DFT SCF labeling
  -> dataset assembly
  -> NEP or MACE training
  -> model report and deployable potential
```

The first implementation target is VASP SCF labeling, followed by
GPUMD/NEP and MACE training workflows.

## Differentiation

PESMaker is designed to be:

- user-structure-driven rather than random-structure-search first;
- foundation-potential-assisted for affordable sampling before DFT labeling;
- NEP/GPUMD and MACE friendly from the beginning;
- lightweight enough to run without a mandatory database service;
- application-oriented for batteries, solid electrolytes, thermal transport,
  alloys, defects, surfaces, catalysis, and reactions.

## Current skeleton

```text
src/pesmaker/
  config/        # YAML input parsing and validation
  structures/    # POSCAR/CIF/XYZ IO, supercells, perturbations
  samplers/      # LAMMPS+MACE, GPUMD+NEP sampling interfaces
  generators/    # defects, vacancies, surfaces, transition states
  labelers/      # VASP first, CP2K later
  jobs/          # Slurm/PBS/local submission and monitoring
  parsers/       # VASP, GPUMD, MACE output parsing
  dataset/       # extxyz, NEP train.xyz, HDF5, metadata
  trainers/      # NEP and MACE training interfaces
  workflow/      # stage setup and workflow execution helpers
  cli.py         # command line interface
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Tingliangstu/PESMaker.git
cd PESMaker
```

Install PESMaker in editable mode:

```bash
python -m pip install -e .
```

For development, install the test and lint tools too:

```bash
python -m pip install -e ".[dev]"
```

After installation, the command-line interface should be available as:

```bash
pesmaker --help
```

On Windows, `pip` may install `pesmaker.exe` into a user script directory that
is not on `PATH`, for example:

```text
C:\Users\<user>\AppData\Roaming\Python\Python313\Scripts
```

If `pesmaker --help` is not recognized, either add that directory to your user
`PATH`, or run the executable by its full path:

```powershell
& 'C:\Users\<user>\AppData\Roaming\Python\Python313\Scripts\pesmaker.exe' --help
```

Current runtime dependencies are PyYAML, NumPy, and ASE. Pymatgen is optional
for later atomistic utilities.

Minimum runtime dependencies:

```text
python >= 3.10
ase >= 3.23
numpy >= 1.24
PyYAML >= 6.0
```

Editable installation also needs the build tools listed in `pyproject.toml`:

```text
setuptools >= 68
wheel
```

PESMaker uses YAML configuration files only. This keeps the command-line
workflow simple and avoids optional config parser dependencies.

## Documentation

The documentation is built with MkDocs and is intended to be published with
GitHub Pages:

```bash
python -m pip install -e ".[docs]"
mkdocs serve
```

After GitHub Pages is enabled, the online manual will be available at:

```text
https://Tingliangstu.github.io/PESMaker/
```

## Try the Scaffold

```bash
python -m pesmaker validate examples/pesmaker.yaml
python -m pesmaker generate examples/pesmaker.yaml
```

## Generate perturbed structures

PESMaker can generate supercells and perturbed structures from CIF, POSCAR, and
other ASE-readable structure files:

```bash
pesmaker generate examples/perturb.yaml
```

The first perturbation backend follows the common `dpdata.System.perturb` style
parameters:

```yaml
generation:
  supercell: [4, 4, 4]
  perturb:
    pert_num: 49
    cell_pert_fraction: 0.03
    atom_pert_distance: 0.1
    atom_pert_style: normal
    seed: 42
    format: vasp
```

For multiple structures, list each file directly:

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

Each input file gets its own output folder under
`runs/<project>/generated/<input-file-name>/`.

For many structures in one directory, use `include`:

```yaml
structures:
  include:
    - initial_structures/*.cif
```

