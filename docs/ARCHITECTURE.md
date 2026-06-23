# PESMaker Architecture

PESMaker is organized around replaceable workflow stages. The stage
implementations live in domain packages, while `pesmaker.workflow` is now an
orchestration and compatibility layer.

## Core objects

- `PESMakerConfig`: validated project configuration.
- `WorkflowConfig`: optional compatibility override for `next`; normal configs
  omit it and let artifacts plus YAML sections determine the flow.
- `StageResult`: small return object for setup, submit, collect, and training
  stages.
- JSON Lines manifests: persistent file records for generated structures,
  sampling jobs, selected frames, SCF jobs, and the active training job.

## Module Boundaries

- `pesmaker.generators.structures`: supercells, surfaces, defects,
  perturbations, generated-structure manifests, and generation summaries.
- `pesmaker.samplers.gpumd`: GPUMD sampling folders, `run.in`, potential copy,
  and sampling submit scripts.
- `pesmaker.samplers.lammps_mace`: LAMMPS-MACE sampling folders, `data.in`,
  user LAMMPS input rendering, and sampling submit scripts.
- `pesmaker.samplers`: sampling-engine dispatcher used by CLI and `next`.
- `pesmaker.samplers.selection`: descriptors, farthest point selection, and
  diagnostic plots.
- `pesmaker.parsers.ase`: ASE-backed frame reading and extxyz writing.
- `pesmaker.parsers.vasp`: VASP output readers used by collection.
- `pesmaker.labelers.vasp`: VASP SCF folders, POSCAR normalization, INCAR,
  POTCAR assembly, and SCF warnings.
- `pesmaker.jobs.resources`: CPU/GPU and VASP parallel-resource decisions.
- `pesmaker.jobs.scripts`: submit-script template rendering and normalization.
- `pesmaker.jobs.submit`: dry-run or real submission of prepared `submit.sh`
  files.
- `pesmaker.dataset.extxyz`: labeled-output collection into extxyz datasets.
- `pesmaker.trainers.nep`: NEP and generic training input setup.
- `pesmaker.trainers.layout`: shared training output paths, two-step state,
  and training manifest locations.
- `pesmaker.plot`: plotting command package. `pesmaker.plot.commands` is the
  CLI registry, engine-specific plots live in modules such as
  `pesmaker.plot.nep`, and shared result/style helpers stay in
  `pesmaker.plot.result` and `pesmaker.plot.style`.
- `pesmaker.workflow.next`: artifact-driven smart-next state machine.
- `pesmaker.workflow.plan`: artifact path and file-presence checks.
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
- active training job: `training/training_manifest.jsonl`;
- follow-up config template after generation-only runs: `run.next.yaml`;
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
