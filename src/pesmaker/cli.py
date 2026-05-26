# Copyright 2026 Ting Liang and PESMaker development team
# This file is part of PESMaker.
#
# PESMaker is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PESMaker is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PESMaker. If not, see <https://www.gnu.org/licenses/>.
"""Command-line interface for PESMaker."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from pesmaker import __contact__, __version__
from pesmaker.config.io import load_config
from pesmaker.workflow.generate import GenerateResult, generate_structures
from pesmaker.workflow.plan import build_plan


def main(argv: list[str] | None = None) -> int:
    """Run the PESMaker command-line interface.

    Args:
        argv: Optional command-line arguments. When `None`, arguments are read
            from `sys.argv` by `argparse`.

    Returns:
        Process-style exit code. `0` means the selected command completed
        successfully, and nonzero values indicate user-facing errors.
    """
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
    init_parser.add_argument(
        "path", type=Path, nargs="?", default=Path("pesmaker.yaml")
    )

    args = parser.parse_args(argv)
    _print_banner()

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
        _print_generate_summary(result)
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


def _write_starter_config(path: Path) -> int:
    """Write a minimal starter YAML configuration file.

    Args:
        path: Destination path for the new YAML input file.

    Returns:
        `0` when the file is written. Returns `1` if the destination already
        exists because PESMaker refuses to overwrite user files.
    """
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


def _print_generate_summary(result: GenerateResult) -> None:
    """Print a concise summary of generated perturbation outputs.

    Args:
        result: Completed generation result returned by the workflow layer.
    """
    folder_counts: dict[tuple[Path, Path], int] = defaultdict(int)
    for structure in result.structures:
        folder_counts[(structure.source, structure.path.parent)] += 1

    print("Perturbation generation complete.")
    print(f"Generated structures : {len(result.structures)}")
    print(f"Output directory     : {result.output_dir}")
    print(f"Manifest             : {result.output_dir / 'manifest.jsonl'}")
    print("Structure folders:")
    for (source, folder), count in folder_counts.items():
        print(f"  - {source} -> {folder} ({count} structure(s))")


def _print_banner() -> None:
    """Print the PESMaker command banner with version and contact information.

    The banner is shown once at the beginning of every executable subcommand so
    users can identify the running PESMaker version in logs.
    """
    logo_lines = [
        r"  _____   ______   _____   __  __          _                 ",
        r" |  __ \ |  ____| / ____| |  \/  |        | |                ",
        r" | |__) || |__   | (___   | \  / |   __ _ | | __  ___  _ __ ",
        r" |  ___/ |  __|   \___ \  | |\/| |  / _` || |/ / / _ \| '__|",
        r" | |     | |____  ____) | | |  | | | (_| ||   < |  __/| |   ",
        r" |_|     |______||_____/  |_|  |_|  \__,_||_|\_\ \___||_|   ",
        f"                                                   v-{__version__}",
    ]
    for line in logo_lines:
        print(line)
    print("**************** Potential Energy Surface Maker ****************")
    print("***** Automated dataset generation for machine-learned potentials *****")
    print(f"**************** Author: {__contact__} ****************")
    print("****************************************************************")
