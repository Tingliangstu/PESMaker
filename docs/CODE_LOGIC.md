# PESMaker Code Logic

This document summarizes the current code flow after the stage split.

## Entry Points

```mermaid
flowchart TD
    A["python -m pesmaker"] --> B["pesmaker.__main__.main()"]
    C["pesmaker CLI command"] --> D["pesmaker.cli.main()"]
    B --> D

    D --> E{"command"}
    E -->|"init"| F["write starter YAML"]
    E -->|"validate"| G["load_config(path)"]
    E -->|"next"| N["workflow.next.run_next(config, path)"]
    E -->|"generate"| GEN["generators.structures.generate_structures"]
    E -->|"sample-setup"| SAMP["samplers.gpumd.setup_sampling"]
    E -->|"select"| SEL["samplers.selection.select_sampling_frames"]
    E -->|"scf-setup"| VASP["labelers.vasp.setup_labeling"]
    E -->|"submit"| SUB["jobs.submit.submit_jobs"]
    E -->|"collect"| DATA["dataset.extxyz.collect_labeled_dataset"]
    E -->|"train-setup"| TRAIN["trainers.nep.setup_training"]
```

## Config Flow

```mermaid
flowchart TD
    A["load_config(path)"] --> B["_load_mapping(path)"]
    B --> C["yaml.safe_load with duplicate-key rejection"]
    C --> D["PESMakerConfig.from_mapping"]
    D --> E["StructureInput"]
    D --> F["GenerationConfig"]
    D --> G["EngineConfig: sampling/labeling/training/jobs"]
    D --> H["DatasetConfig"]
    D --> I["WorkflowConfig"]
```

`WorkflowConfig.mode` accepts `auto`, `direct-scf`, and `sampling-training`.
`auto` resolves to `sampling-training` when sampling and selection are
configured; otherwise it resolves to `direct-scf`.

## Module Dependencies

```mermaid
flowchart LR
    CLI["pesmaker.cli"] --> Config["pesmaker.config"]
    CLI --> Next["pesmaker.workflow.next"]
    CLI --> Gen["pesmaker.generators.structures"]
    CLI --> Gpumd["pesmaker.samplers.gpumd"]
    CLI --> Select["pesmaker.samplers.selection"]
    CLI --> Vasp["pesmaker.labelers.vasp"]
    CLI --> Submit["pesmaker.jobs.submit"]
    CLI --> Data["pesmaker.dataset.extxyz"]
    CLI --> Train["pesmaker.trainers.nep"]

    Gen --> Structures["pesmaker.structures"]
    Gpumd --> JobsScripts["pesmaker.jobs.scripts"]
    Select --> Gpumd
    Vasp --> JobsResources["pesmaker.jobs.resources"]
    Vasp --> JobsScripts
    Submit --> Artifacts["pesmaker.artifacts"]
    Data --> Select
    Train --> JobsScripts
    Next --> Gen
    Next --> Gpumd
    Next --> Select
    Next --> Vasp
    Next --> Submit
    Next --> Data
    Next --> Train
```

## Smart Next Flow

```mermaid
flowchart TD
    A["pesmaker next run.yaml"] --> B["resolve workflow mode"]
    B --> C{"generated/manifest.jsonl exists?"}
    C -->|no| D["generate structures"]
    D --> C
    C -->|yes| E{"sampling-training?"}
    E -->|yes| F{"sampling_manifest exists?"}
    F -->|no| G["setup GPUMD sampling"]
    G --> F
    F -->|yes| H{"sampling dry-run recorded?"}
    H -->|no| I["submit --stage sampling --dry-run; record state"]
    H -->|yes| J{"movie.xyz exists?"}
    J -->|no| K["wait and print sampling submit command"]
    J -->|yes| L{"selected manifest exists?"}
    L -->|no| M["select representative frames"]
    M --> L
    E -->|no| N["SCF path"]
    L -->|yes| N
    N --> O{"labeling_manifest exists?"}
    O -->|no| P["setup VASP SCF folders"]
    P --> O
    O -->|yes| Q{"SCF dry-run recorded?"}
    Q -->|no| R["submit --dry-run; record state"]
    Q -->|yes| S{"OUTCAR files exist?"}
    S -->|no| T["wait and print SCF submit command"]
    S -->|yes| U{"dataset exists?"}
    U -->|no| V["collect labeled dataset"]
    V --> U
    U -->|yes| W{"training path?"}
    W -->|no| X["complete"]
    W -->|yes| Y{"training submit.sh exists?"}
    Y -->|no| Z["setup training"]
    Z --> Y
    Y -->|yes| AA{"training dry-run recorded?"}
    AA -->|no| AB["submit --stage training --dry-run; record state"]
    AA -->|yes| X
```

## Compatibility

Older imports remain valid:

```python
from pesmaker.workflow.generate import generate_structures
from pesmaker.workflow.stages import submit_jobs, StageResult
```

Those modules re-export symbols from the domain packages.
