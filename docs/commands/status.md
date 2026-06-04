# `pesmaker status`

`status` shows what `next` would do, without writing files.

## Use

```bash
pesmaker status run.yaml
```

## When To Use It

Use it before running `next` if you are not sure what PESMaker will do.

Example:

```text
Next action      : Generate structures from the configured inputs.
What you should do next:
  - Run: pesmaker next run.yaml
```

`status` is read-only. It does not create `generated/`, `labeling/`,
`.pesmaker/`, or dry-run logs.
