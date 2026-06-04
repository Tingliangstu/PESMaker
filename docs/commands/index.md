# Command Manual

This section is the command-by-command manual.

For normal work, you usually need only two commands:

```bash
pesmaker validate run.yaml
pesmaker next run.yaml
```

`next` runs the local PESMaker steps that are ready. You do not need to run
`generate`, `sample-setup`, `select`, `scf-setup`, `collect`, or `train-setup`
one by one unless you want manual control.

The one thing `next` does not do is real scheduler submission. When jobs need
to be submitted, `next` writes a dry-run log and prints the exact
`pesmaker submit ...` command to run.

## Command List

| Command | Use it for |
| --- | --- |
| [`init`](init.md) | Write a starter YAML file. |
| [`validate`](validate.md) | Check the YAML before doing work. |
| [`next`](next.md) | Continue the workflow until a submit, wait, or complete point. |
| [`status`](status.md) | Show what `next` would do without writing files. |
| [`generate`](generate.md) | Manually generate structures. |
| [`sample-setup`](sample-setup.md) | Manually prepare GPUMD sampling folders. |
| [`select`](select.md) | Manually select MD frames. |
| [`scf-setup`](scf-setup.md) | Manually prepare VASP SCF folders. |
| [`submit`](submit.md) | Submit prepared `submit.sh` files. |
| [`collect`](collect.md) | Build an extxyz dataset from finished VASP outputs. |
| [`train-setup`](train-setup.md) | Prepare model training inputs. |

## How `next` Thinks

`next` looks at the YAML sections and the files already on disk.

```text
structures configured + generated/manifest missing -> run generate
sampling configured                                -> prepare sampling
sampling submit not previewed                      -> preview sampling submit
movie.xyz exists                                   -> run select
labeling configured                                -> prepare SCF
SCF submit not previewed                           -> preview SCF submit
OUTCAR exists                                      -> collect train.xyz
training configured                                -> prepare training
training submit not previewed                      -> preview training submit
```

If the files already exist, `next` skips that step. If external results are
missing, it stops and tells you what it is waiting for.

## Manual Commands Are Still Useful

Use manual commands when you want to debug one stage:

```bash
pesmaker generate run.yaml
pesmaker scf-setup run.yaml
pesmaker submit run.yaml --dry-run
```

For production use, start with `next`. It is less to remember and it gives the
next command in the terminal output.
