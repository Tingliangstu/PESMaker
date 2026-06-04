# `pesmaker validate`

`validate` checks the YAML before PESMaker does expensive work.

## Use

```bash
pesmaker validate run.yaml
```

## What It Checks

It checks:

- YAML syntax;
- duplicate YAML keys;
- section shapes such as `structures`, `generation`, and `jobs`;
- known enum values such as the optional advanced `workflow` override.

It does not run VASP, GPUMD, or NEP.

## Next Step

If validation passes:

```bash
pesmaker next run.yaml
```

If validation fails, fix the reported YAML line or field first.
