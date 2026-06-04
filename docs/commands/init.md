# `pesmaker init`

`init` writes a starter YAML file.

## Use

```bash
pesmaker init run.yaml
```

If the file already exists, PESMaker refuses to overwrite it.

## Next Step

Edit the paths in `run.yaml`, then run:

```bash
pesmaker validate run.yaml
pesmaker next run.yaml
```

The starter file is only a scaffold. Replace structure paths, potentials,
VASP paths, and scheduler templates before using it on a cluster.
