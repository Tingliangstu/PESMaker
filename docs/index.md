# PESMaker

PESMaker is a lightweight Python workflow package for building
application-oriented datasets and machine-learned interatomic potentials from
user-provided atomistic structures.

It supports a staged workflow:

```text
initial structures
  -> supercells, surfaces, defects, optional perturbations
  -> optional GPUMD sampling and frame selection
  -> VASP SCF labeling setup and submission
  -> extxyz dataset collection
  -> NEP training setup
```

## Start Here

For the complete workflow, read the
[Active Learning Workflow](ACTIVE_LEARNING_WORKFLOW.md) manual.

For a short command overview, see [Usage](usage.md).

## Main Command Flow

Direct generation and SCF labeling:

```bash
pesmaker validate run.yaml
pesmaker generate run.yaml
pesmaker scf-setup run.yaml
pesmaker submit run.yaml             # submit SCF/VASP jobs
pesmaker collect run.yaml
```

Full sampling, labeling, and training loop:

```bash
pesmaker generate run.yaml
pesmaker sample-setup run.yaml
pesmaker submit run.yaml --stage sampling   # submit MD sampling jobs
pesmaker select run.yaml
pesmaker scf-setup run.yaml
pesmaker submit run.yaml                    # submit SCF/VASP jobs
pesmaker collect run.yaml
pesmaker train-setup run.yaml
pesmaker submit run.yaml --stage training   # submit NEP training jobs
```

PESMaker is designed to keep every stage inspectable through ordinary folders,
manifests, and scheduler scripts.
