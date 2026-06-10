# Installation

Clone the repository:

```bash
git clone https://github.com/Tingliangstu/PESMaker.git
cd PESMaker
```

Install PESMaker:

```bash
python -m pip install .
pesmaker --help
```

No internet: copy or unzip the PESMaker source folder, then run those two
commands inside it.

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

If you are already inside the PESMaker repository on `main`:

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

Trajectory FPS uses an engine-specific optional dependency:

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
