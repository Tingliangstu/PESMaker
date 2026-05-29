# PESMaker Code Logic

This document describes the current scaffold-level code flow. It should be
updated as real workflow stages are added.

## Current entry points

```mermaid
flowchart TD
    A["python -m pesmaker"] --> B["pesmaker.__main__.main()"]
    C["pesmaker CLI command"] --> D["pesmaker.cli.main()"]
    B --> D

    D --> E{"command"}
    E -->|"init"| F["write starter YAML"]
    E -->|"validate"| G["load_config(path)"]
    E -->|"generate"| G

    G --> H["_load_mapping(path)"]
    H --> I{"file suffix"}
    I -->|".yaml/.yml"| J["yaml.safe_load"]
    J --> L["PESMakerConfig.from_mapping"]

    L --> M["StructureInput.from_mapping"]
    L --> N["GenerationConfig.from_mapping"]
    L --> O["EngineConfig.from_mapping: sampling"]
    L --> P["EngineConfig.from_mapping: labeling"]
    L --> Q["DatasetConfig.from_mapping"]
    L --> R["EngineConfig.from_mapping: training"]

    G --> S["PESMakerConfig"]
    S --> T["validate: print OK"]
    S --> X["generate_structures(config)"]
    X --> Y["load_structure -> make_supercell -> perturb_structures -> write_structure"]
```

## Module dependencies

```mermaid
flowchart LR
    Main["pesmaker.__main__"] --> CLI["pesmaker.cli"]
    Init["pesmaker.__init__"] --> Schema["pesmaker.config.schema"]
    CLI --> IO["pesmaker.config.io"]
    CLI --> Generate["pesmaker.workflow.generate"]
    IO --> Schema
    Generate --> Schema
    Generate --> Structures["pesmaker.structures"]
    Tests["tests/test_config.py"] --> Schema
```

## Current responsibilities

- `pesmaker.cli`: command-line parsing and user-facing commands.
- `pesmaker.config.io`: YAML file loading.
- `pesmaker.config.schema`: typed configuration objects and validation.
- Empty stage packages: future homes for structures, samplers, generators,
  labelers, jobs, parsers, dataset assembly, and trainers.

## Intended stage flow

```mermaid
flowchart TD
    A["input structures"] --> B["structures: read, normalize, supercell"]
    B --> C["generators: perturb, defects, surfaces, reactions"]
    C --> D["samplers: GPUMD/NEP, LAMMPS/MACE"]
    D --> E["labelers: VASP first, CP2K later"]
    E --> F["jobs: local/Slurm/PBS submit and monitor"]
    F --> G["parsers: energies, forces, stress, convergence"]
    G --> H["dataset: extxyz, NEP train.xyz, metadata"]
    H --> I["trainers: NEP, MACE"]
    I --> J["potential + report"]
```
