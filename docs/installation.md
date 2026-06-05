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

After a fresh install or update, use a new empty folder and run:

```bash
mkdir pesmaker-check
cd pesmaker-check
pesmaker init run.yaml
pesmaker validate run.yaml
```

The final line should look like:

```text
OK: run.yaml describes project 'example_project'.
```

This is only a small command-line and YAML validation check. It does not run
VASP, GPUMD, NEP, or any scheduler command.

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
