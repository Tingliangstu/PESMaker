# PESMaker Roadmap

## MVP 1: structure generation and VASP task preparation

- Parse a simple YAML input file.
- Read user structures from POSCAR, CIF, and extxyz.
- Generate supercells and small perturbations.
- Create VASP SCF calculation folders from a template.
- Produce a local task manifest.

## MVP 2: job submission and result collection

- Support local, Slurm, and PBS job backends.
- Track submitted, running, completed, failed, and parsed jobs.
- Parse VASP energy, force, stress, convergence, and failure metadata.
- Resume interrupted workflows from the manifest.

## MVP 3: dataset assembly and quality checks

- Export extxyz and GPUMD NEP `train.xyz`.
- Deduplicate near-identical structures.
- Detect short bonds, abnormal energies, abnormal forces, and failed labels.
- Generate a dataset report with source tags and split metadata.

## MVP 4: training workflows

- Train GPUMD/NEP from generated datasets.
- Train MACE from generated datasets.
- Summarize train/validation/test errors.
- Package the final potential with a model card.

## Later modules

- CP2K labeler.
- MACE-OMAT, MACE-MP, MatterSim, Orb, SevenNet, and NEP sampling backends.
- Application recipes for diffusion, thermal transport, alloys, defects,
  surfaces, adsorption, transition states, and reactions.
- Active-learning loop with uncertainty or committee-based candidate selection.

## Position relative to autoplex

autoplex is a strong reference for automated MLIP workflows, especially
random-structure-search-driven potential-landscape exploration. PESMaker should
avoid becoming a simplified clone. Its target identity is:

```text
known material or task
  -> targeted sampling with foundation potentials
  -> DFT labels
  -> application-specific MLIP
```

This is different from:

```text
chemical formula
  -> random structures
  -> DFT labels
  -> broad PES model
```
