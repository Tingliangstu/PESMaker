# PESMaker Architecture

PESMaker is organized around replaceable workflow stages. The stage
implementations live in domain packages, while `pesmaker.workflow` is now an
orchestration and compatibility layer.

## Core objects

- `PESMakerConfig`: validated project configuration.
- `WorkflowConfig`: high-level `next` mode, either `auto`, `direct-scf`, or
  `sampling-training`.
- `StageResult`: small return object for setup, submit, collect, and training
  stages.
- JSON Lines manifests: persistent file records for generated structures,
  sampling jobs, selected frames, and SCF jobs.

## Module Boundaries

- `pesmaker.generators.structures`: supercells, surfaces, defects,
  perturbations, generated-structure manifests, and generation summaries.
- `pesmaker.samplers.gpumd`: GPUMD sampling folders, `run.in`, potential copy,
  and sampling submit scripts.
- `pesmaker.samplers.selection`: trajectory reading, descriptors, farthest
  point selection, and diagnostic plots.
- `pesmaker.labelers.vasp`: VASP SCF folders, POSCAR normalization, INCAR,
  POTCAR assembly, and SCF warnings.
- `pesmaker.jobs.resources`: CPU/GPU and VASP parallel-resource decisions.
- `pesmaker.jobs.scripts`: submit-script template rendering and normalization.
- `pesmaker.jobs.submit`: dry-run or real submission of prepared `submit.sh`
  files.
- `pesmaker.dataset.extxyz`: labeled-output collection into extxyz datasets.
- `pesmaker.trainers.nep`: NEP and generic training input setup.
- `pesmaker.workflow.next`: smart-next state machine.
- `pesmaker.workflow.plan`: workflow mode and artifact checks.
- `pesmaker.workflow.state`: `.pesmaker/<project>/next_state.json`.

`pesmaker.workflow.stages` and `pesmaker.workflow.generate` remain
backward-compatible re-export modules for older imports.

## Stage Interfaces

Each concrete stage exposes a small Python function such as
`generate_structures(config)`, `setup_sampling(config)`,
`setup_labeling(config)`, `submit_jobs(config, stage=..., dry_run=...)`,
`collect_labeled_dataset(config)`, or `setup_training(config)`. The CLI and
`next` orchestration call these functions directly.

## State Model

Stage data stays file-backed:

- generated structures: `generated/manifest.jsonl`;
- sampling jobs: `sampling/sampling_manifest.jsonl`;
- selected frames: `selected/manifest.jsonl`;
- SCF jobs: `labeling/labeling_manifest.jsonl`;
- smart-next dry-run gates: `.pesmaker/<project>/next_state.json`.

A database service remains optional and is not required for the current
workflow.

## Dependency policy

The base package should stay light. Heavy scientific tools should be optional
extras or external executables:

- base: configuration, stage setup, file manifests, CLI;
- atomistic extra: ASE and pymatgen;
- workflow extras: jobflow or AiiDA only if a user chooses those integrations;
- engines: VASP, CP2K, GPUMD, LAMMPS, MACE as external programs.
