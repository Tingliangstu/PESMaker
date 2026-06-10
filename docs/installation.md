# Installation

PESMaker requires Python 3.10 or newer.

## Install From PyPI

After the first release is published, the normal installation is:

```bash
python -m pip install pesmaker
pesmaker --help
```

Upgrade an existing PyPI installation:

```bash
python -m pip install --upgrade pesmaker
```

Optional descriptor backends are installed as extras:

```bash
# GPUMD trajectories: Calorine NEP descriptors
python -m pip install "pesmaker[selection]"

# MACE trajectories: ASE MACECalculator descriptors
python -m pip install "pesmaker[mace]"

# Both descriptor backends
python -m pip install "pesmaker[selection,mace]"
```

## Install From GitHub

Before the first PyPI release, or when you need the latest merged changes,
install the `main` branch directly:

```bash
python -m pip install "git+https://github.com/Tingliangstu/PESMaker.git@main"
pesmaker --help
```

Install a GitHub version with optional dependencies by using a direct
requirement:

```bash
python -m pip install "pesmaker[mace] @ git+https://github.com/Tingliangstu/PESMaker.git@main"
```

Installing directly from GitHub requires `git` and internet access.

## Install From A Source Checkout

Clone the repository and install the local checkout:

```bash
git clone https://github.com/Tingliangstu/PESMaker.git
cd PESMaker
python -m pip install .
pesmaker --help
```

No internet: copy or unzip the PESMaker source folder, enter that folder, then
run `python -m pip install .`.

## Quick Test

After a fresh install or update, stay in the PESMaker repository root and run
the test files:

```bash
python -m pytest
```

This checks that the installed Python package, config parser, structure tools,
CLI functions, and workflow logic work.

If `pytest` is not installed, install the small test dependency once:

```bash
python -m pip install ".[dev]"
python -m pytest
```

## Update an Existing Checkout

For a PyPI installation:

```bash
python -m pip install --upgrade pesmaker
```

If you are using a source checkout on `main`:

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

## Dependencies

The current runtime dependencies are:

- `PyYAML`: YAML input files.
- `NumPy`: random perturbation and matrix operations.
- `ASE`: reading and writing atomistic structure files.

The optional `atomistic` extra currently reserves room for heavier atomistic
utilities such as `pymatgen`.

From a source checkout, trajectory FPS extras can also be installed with:

```bash
# GPUMD trajectories: Calorine NEP descriptors
python -m pip install ".[selection]"

# MACE trajectories: ASE MACECalculator descriptors
python -m pip install ".[mace]"
```

The MACE extra installs `mace-torch`. It is only needed in the Python
environment that runs `pesmaker select`; LAMMPS sampling itself still uses the
separate MLIAP model and LAMMPS executable.

## Windows command path

On Windows, `pip` may install `pesmaker.exe` into a user script directory that is
not on `PATH`, for example:

```text
C:\Users\<user>\AppData\Roaming\Python\Python313\Scripts
```

If `pesmaker --help` is not recognized, add that directory to the user `PATH`, or
run the executable with its full path:

```powershell
& 'C:\Users\<user>\AppData\Roaming\Python\Python313\Scripts\pesmaker.exe' --help
```
