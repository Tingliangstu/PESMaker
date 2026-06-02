# PESMaker Project Brief

## Name

**PESMaker**: Potential Energy Surface Maker.

Working paper title:

```text
PESMaker: a lightweight foundation-potential-assisted workflow for
application-oriented machine-learning potential dataset generation
```

## Core idea

PESMaker starts from user-provided atomistic structures and turns them into
DFT-labeled datasets and trained machine-learned interatomic potentials.

The intended path is:

```text
known material or task
  -> targeted structure generation and sampling
  -> DFT SCF labeling
  -> dataset quality checks
  -> NEP or MACE training
  -> final potential
```

## Target users

- battery materials researchers;
- solid electrolyte researchers;
- thermal transport researchers;
- alloy and defect researchers;
- surface and catalysis researchers;
- users who already have VASP/GPUMD/LAMMPS scripts but need an automated,
  reproducible workflow.

## Differentiation from autoplex

autoplex is a strong reference and should be cited. Its strongest identity is
random-structure-search-driven potential-landscape exploration, especially with
AIRSS/buildcell and heavy workflow infrastructure.

PESMaker should be different:

- user-provided structures first;
- physical application recipes first;
- foundation potentials for affordable sampling before DFT labeling;
- NEP/GPUMD and MACE as first-class training targets;
- lightweight local execution first, with Slurm/PBS support;
- database services optional rather than mandatory.

## First development target

MVP 1 should not try to implement all planned science modules. It should support:

1. read one or more initial structures;
2. generate supercells and optional perturbations;
3. create SCF calculation folders from a user template;
4. write a manifest of all generated calculation tasks;
5. provide a CLI that can validate configs and prepare workflow stages.

After this is stable, add job submission, VASP parsing, extxyz export, and NEP
training.
