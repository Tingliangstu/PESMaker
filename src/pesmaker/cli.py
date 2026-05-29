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
from pathlib import Path

from pesmaker import __contact__, __version__
from pesmaker.config.io import load_config
from pesmaker.workflow.generate import (
    GenerateResult,
    format_generate_summary,
    generate_structures,
)
from pesmaker.workflow.plan import build_plan
from pesmaker.workflow.stages import (
    StageResult,
    collect_labeled_dataset,
    select_sampling_frames,
    setup_labeling,
    setup_sampling,
    setup_training,
    submit_jobs,
)


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

    sample_setup_parser = subparsers.add_parser(
        "sample-setup",
        help="Prepare MD sampling folders and submission scripts.",
    )
    sample_setup_parser.add_argument("config", type=Path)

    select_parser = subparsers.add_parser(
        "select",
        help="Select representative MD frames with farthest point sampling.",
    )
    select_parser.add_argument("config", type=Path)

    scf_setup_parser = subparsers.add_parser(
        "scf-setup",
        help="Prepare VASP SCF calculation folders and submission scripts.",
    )
    scf_setup_parser.add_argument("config", type=Path)

    collect_parser = subparsers.add_parser(
        "collect",
        help="Collect completed VASP SCF calculations into a training set.",
    )
    collect_parser.add_argument("config", type=Path)

    train_setup_parser = subparsers.add_parser(
        "train-setup",
        help="Prepare potential training inputs and submission script.",
    )
    train_setup_parser.add_argument("config", type=Path)

    submit_parser = subparsers.add_parser(
        "submit",
        help="Submit prepared stage jobs with the configured scheduler.",
    )
    submit_parser.add_argument("config", type=Path)
    submit_parser.add_argument(
        "--stage",
        choices=("sampling", "labeling", "training"),
        default="labeling",
    )
    submit_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the submission commands without calling the scheduler.",
    )

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

    if args.command == "sample-setup":
        _print_stage_result(setup_sampling(config))
        return 0

    if args.command == "select":
        _print_stage_result(select_sampling_frames(config))
        return 0

    if args.command == "scf-setup":
        _print_stage_result(setup_labeling(config))
        return 0

    if args.command == "collect":
        _print_stage_result(collect_labeled_dataset(config))
        return 0

    if args.command == "train-setup":
        _print_stage_result(setup_training(config))
        return 0

    if args.command == "submit":
        _print_stage_result(submit_jobs(config, stage=args.stage, dry_run=args.dry_run))
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
  output_dir: generated
  tasks:
    - name: surface_331
      supercell: [3, 3, 1]
      surface:
        vacuum: 30.0
        axis: 2
        center: true
        defects:
          mode: random
          seed: 42
          single_vacancies:
            elements: [Te]
            max_count: 4
          double_vacancies:
            elements: [Te]
            max_count: 4
          line_defects:
            elements: [Te]
            max_count: 2
        perturb:
          pert_num: 10
          cell_pert_fraction: 0.03
          atom_pert_distance: 0.1
          atom_pert_style: normal
          format: vasp
    - name: bulk_333
      supercell: [3, 3, 3]
      perturb:
        pert_num: 10
        cell_pert_fraction: 0.03
        atom_pert_distance: 0.1
        atom_pert_style: normal
        format: vasp

sampling:
  engine: gpumd
  gpumd_dir: /home/tingliang/software/GPUMD/GPUMD-master-26-05-2026/src
  output_dir: sampling
  potential: nep89_20250409.txt
  temperatures: [300, 600, 900]
  run_in: templates/gpumd/run.in
  selection:
    trajectory_pattern: sampling/**/movie.xyz
    output_dir: selected
    min_distance: 0.2
    max_count: 200

labeling:
  engine: vasp
  output_dir: labeling
  incar: templates/vasp/INCAR
  potcar_library: /home/a4s5d/software/VASP/potentials
  command: /home/a4s5d/software/VASP/CPU_vasp.6.6.0/bin/vasp_std

dataset:
  format: extxyz
  split: [0.8, 0.1, 0.1]

training:
  model: nep
  output_dir: training

jobs:
  machine: local
  submit_command: sbatch
  sbatch_templates:
    sampling: templates/sbatch/gpumd.sh
    labeling: templates/sbatch/vasp_cpu_36.sh
    training: templates/sbatch/nep.sh
"""
    path.write_text(template, encoding="utf-8")
    print(f"Wrote starter config: {path}")
    return 0


def _print_generate_summary(result: GenerateResult) -> None:
    """Print a concise summary of generated perturbation outputs.

    Args:
        result: Completed generation result returned by the workflow layer.
    """
    print(format_generate_summary(result), end="")
    print()


def _print_stage_result(result: StageResult) -> None:
    """Print a concise setup or collection result summary."""
    print(result.message)
    print(f"Output directory : {result.output_dir}")
    print(f"Files written    : {len(result.files)}")
    print()


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
    print()
