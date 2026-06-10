# Publishing PESMaker To PyPI

PESMaker uses
[PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/). GitHub
Actions authenticates with OIDC, so no long-lived PyPI API token is stored in
the repository.

## One-Time PyPI Setup

Before the first release, create a
[pending publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
in the PyPI account:

1. Open the [PyPI publishing settings](https://pypi.org/manage/account/publishing/).
2. Add a pending GitHub publisher.
3. Enter these values:

```text
PyPI project name : pesmaker
GitHub owner       : Tingliangstu
Repository         : PESMaker
Workflow filename  : publish.yml
Environment name   : pypi
```

Also create a GitHub repository environment named `pypi` under
**Settings -> Environments**. Required reviewers can be enabled there if every
production upload should require manual approval.

The project does not need to exist on PyPI before adding a pending publisher.
The first successful trusted publication creates it.

## Prepare A Release

The package version has one source of truth:

```python
src/pesmaker/__init__.py
```

Update `__version__`, run the checks, and merge the release commit:

```bash
python -m pip install -e ".[dev,docs,release]"
python -m pytest
python -m ruff check src tests
python -m mkdocs build --strict
python -m build
python -m twine check --strict dist/*
```

Install the built wheel in a clean environment when validating a release:

```bash
python -m pip install dist/pesmaker-*.whl
pesmaker --help
```

## Publish

Create a GitHub Release whose tag exactly matches the package version with a
leading `v`. For version `0.1.0`, use:

```text
v0.1.0
```

Publishing the GitHub Release starts
`.github/workflows/publish.yml`. The workflow:

1. builds the wheel and source distribution;
2. runs strict Twine metadata checks;
3. verifies that the release tag matches the package version;
4. uploads the distributions to PyPI through Trusted Publishing.

PyPI does not allow replacing a file or reusing a published version. If a
release is incorrect, increment `__version__` and publish a new version.

## Verify The Published Package

After the workflow finishes:

```bash
python -m pip index versions pesmaker
python -m pip install --upgrade pesmaker
pesmaker --help
```

The project page will be:

```text
https://pypi.org/project/pesmaker/
```
