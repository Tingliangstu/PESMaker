# Installation

Clone the repository:

```bash
git clone https://github.com/Tingliangstu/PESMaker.git
cd PESMaker
```

Install PESMaker in editable mode:

```bash
python -m pip install -e .
```

For development, install test and lint tools:

```bash
python -m pip install -e ".[dev]"
```

For building the documentation locally:

```bash
python -m pip install -e ".[docs]"
mkdocs serve
```

## Update an Existing Checkout

If the repository already exists locally:

```bash
cd ~/software/PESMaker
git switch main
git fetch origin
git pull --ff-only origin main
python -m pip install -e .
```

For a developer checkout with tests and documentation tools:

```bash
git switch main
git fetch origin
git pull --ff-only origin main
python -m pip install -e ".[dev,docs]"
python -m pytest -q
```

If Git refuses to pull because local files changed, inspect them first:

```bash
git status
```

Then commit the local work or temporarily stash it:

```bash
git stash push -m "work before updating main"
git pull --ff-only origin main
git stash pop
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
