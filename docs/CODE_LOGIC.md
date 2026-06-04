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
    D --> I["WorkflowConfig (optional override)"]
```

Normal user configs omit `workflow`. `pesmaker next` infers the flow from
configured sections and existing files. `WorkflowConfig.mode` still accepts
`auto`, `direct-scf`, and `sampling-training` for older configs and advanced
overrides; `direct-scf` forces `next` to skip sampling and training sections.

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
    Select --> ParsersAse["pesmaker.parsers.ase"]
    Vasp --> JobsResources["pesmaker.jobs.resources"]
    Vasp --> JobsScripts
    Submit --> Artifacts["pesmaker.artifacts"]
    Data --> ParsersAse
    Data --> ParsersVasp["pesmaker.parsers.vasp"]
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
    A["pesmaker next run.yaml"] --> B["inspect config sections and artifacts"]
    B --> C{"structures configured and generated/manifest.jsonl missing?"}
    C -->|yes| D["generate structures"]
    D --> C
    C -->|no| E{"sampling enabled?"}
    E -->|yes| F{"sampling_manifest exists?"}
    F -->|no| G["setup GPUMD sampling"]
    G --> F
    F -->|yes| H{"sampling dry-run recorded?"}
    H -->|no| I["submit --stage sampling --dry-run; record state"]
    H -->|yes| J{"selection configured?"}
    J -->|yes| K{"movie.xyz exists?"}
    K -->|no| L["wait and print sampling submit command"]
    K -->|yes| M{"selected manifest exists?"}
    M -->|no| N["select representative frames"]
    N --> M
    J -->|no| O["continue only if explicit labeling input exists"]
    E -->|no| P["SCF path"]
    M -->|yes| P
    O --> P
    P --> Q{"labeling enabled?"}
    Q -->|no| DONE["complete"]
    Q -->|yes| R{"labeling_manifest exists?"}
    R -->|no| S["setup VASP SCF folders"]
    S --> R
    R -->|yes| T{"SCF dry-run recorded?"}
    T -->|no| U["submit --dry-run; record state"]
    T -->|yes| V{"OUTCAR files exist?"}
    V -->|no| W["wait and print SCF submit command"]
    V -->|yes| X{"dataset exists?"}
    X -->|no| Y["collect labeled dataset"]
    Y --> X
    X -->|yes| Z{"training enabled?"}
    Z -->|no| DONE
    Z -->|yes| AA{"training submit.sh exists?"}
    AA -->|no| AB["setup training"]
    AB --> AA
    AA -->|yes| AC{"training dry-run recorded?"}
    AC -->|no| AD["submit --stage training --dry-run; record state"]
    AC -->|yes| DONE
```

## Compatibility

Older imports remain valid:

```python
from pesmaker.workflow.generate import generate_structures
from pesmaker.workflow.stages import submit_jobs, StageResult
```

Those modules re-export symbols from the domain packages.
