# Copyright (c) 2026 Ting Liang. All rights reserved.
"""Command-line interface for PESMaker."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pesmaker.config.io import load_config
from pesmaker.workflow.generate import generate_structures
from pesmaker.workflow.plan import build_plan


def main(argv: list[str] | None = None) -> int:
    """Run the PESMaker command-line interface."""
    parser = argparse.ArgumentParser(
        prog="pesmaker",
        description="Build application-oriented MLIP datasets and potentials.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate a config file.")
    validate_parser.add_argument("config", type=Path)

    plan_parser = subparsers.add_parser("plan", help="Print the workflow plan.")
    plan_parser.add_argument("config", type=Path)

    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate supercells and perturbed structures.",
    )
    generate_parser.add_argument("config", type=Path)

    init_parser = subparsers.add_parser("init", help="Write a starter config file.")
    init_parser.add_argument("path", type=Path, nargs="?", default=Path("pesmaker.yaml"))

    args = parser.parse_args(argv)

    if args.command == "init":
        return _write_starter_config(args.path)

    config = load_config(args.config)

    if args.command == "validate":
        print(f"OK: {args.config} describes project '{config.project}'.")
        return 0

    if args.command == "plan":
        print(build_plan(config).to_text())
        return 0

    if args.command == "generate":
        result = generate_structures(config)
        print(
            f"Generated {len(result.structures)} structure(s) in {result.output_dir}"
        )
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


def _write_starter_config(path: Path) -> int:
    """Write a minimal starter YAML configuration file."""
    if path.exists():
        print(f"Refusing to overwrite existing file: {path}", file=sys.stderr)
        return 1

    template = """project: example_project

structures:
  - POSCAR

generation:
  supercell: [1, 1, 1]
  perturb:
    pert_num: 10
    cell_pert_fraction: 0.03
    atom_pert_distance: 0.1
    atom_pert_style: normal
    format: vasp

sampling:
  engine: none

labeling:
  engine: vasp
  template: templates/vasp_singlepoint
  scheduler: local

dataset:
  format: extxyz
  split: [0.8, 0.1, 0.1]

training:
  model: nep
"""
    path.write_text(template, encoding="utf-8")
    print(f"Wrote starter config: {path}")
    return 0

