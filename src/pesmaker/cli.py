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
        epilog='Use "pesmaker COMMAND -h" for command-specific help.',
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate",
        help="Check a config file.",
        description="Check config syntax and required fields.",
    )
    _add_config_argument(validate_parser)

    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate supercells and perturbed structures.",
        description="Generate structures and write a manifest for later stages.",
    )
    _add_config_argument(generate_parser)

    sample_setup_parser = subparsers.add_parser(
        "sample-setup",
        help="Prepare sampling job folders and submission scripts.",
        description="Prepare sampling job folders from generated structures.",
    )
    _add_config_argument(sample_setup_parser)

    select_parser = subparsers.add_parser(
        "select",
        help="Select representative sampled structures.",
        description="Select representative structures from sampling trajectories.",
    )
    _add_config_argument(select_parser)

    scf_setup_parser = subparsers.add_parser(
        "scf-setup",
        help="Prepare SCF calculation job folders and submission scripts.",
        description=(
            "Prepare SCF calculation job folders from generated or selected "
            "structures. For VASP configs this writes POSCAR, INCAR, POTCAR, "
            "and submit.sh."
        ),
    )
    _add_config_argument(scf_setup_parser)

    collect_parser = subparsers.add_parser(
        "collect",
        help="Collect completed SCF calculations into a training set.",
        description="Collect completed SCF outputs into a training dataset.",
    )
    _add_config_argument(collect_parser)

    train_setup_parser = subparsers.add_parser(
        "train-setup",
        help="Prepare training job inputs and submission scripts.",
        description="Prepare model-training input files and submission scripts.",
    )
    _add_config_argument(train_setup_parser)

    submit_parser = subparsers.add_parser(
        "submit",
        help="Submit prepared jobs with the configured scheduler.",
        description=(
            "Submit prepared jobs with the configured scheduler. Defaults to "
            "the SCF stage."
        ),
    )
    _add_config_argument(submit_parser)
    submit_parser.add_argument(
        "--stage",
        choices=("sampling", "scf", "training"),
        default="scf",
        help="Workflow stage to submit. Default: scf.",
    )
    submit_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the submission commands without calling the scheduler.",
    )

    init_parser = subparsers.add_parser(
        "init",
        help="Write a starter YAML config.",
        description="Write a starter YAML config file.",
    )
    init_parser.add_argument(
        "path", type=Path, nargs="?", default=Path("pesmaker.yaml")
    )

    args = parser.parse_args(argv)
    _print_banner()

    if args.command == "init":
        return _write_starter_config(args.path)

    config = None
    try:
        config = load_config(args.config)

        if args.command == "validate":
            print(f"OK: {args.config} describes project '{config.project}'.")
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
            _print_labeling_result(setup_labeling(config), args.config)
            return 0

        if args.command == "collect":
            _print_stage_result(collect_labeled_dataset(config))
            return 0

        if args.command == "train-setup":
            _print_stage_result(setup_training(config))
            return 0

        if args.command == "submit":
            _print_stage_result(
                submit_jobs(config, stage=args.stage, dry_run=args.dry_run)
            )
            return 0
    except (OSError, ValueError) as exc:
        print(_format_cli_error(args.command, args.config, config, exc), file=sys.stderr)
        return 2

    parser.error(f"unknown command: {args.command}")
    return 2


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    """Add the common config-file argument to a subcommand parser."""
    parser.add_argument("config", type=Path, help="YAML config file.")


def _format_cli_error(
    command: str,
    config_path: Path,
    config,
    exc: OSError | ValueError,
) -> str:
    """Format user-facing command errors without exposing a traceback."""
    message = str(exc)
    if (
        command == "generate"
        and config is not None
        and not config.structures
        and config.labeling.options
        and "requires 'structures'" in message
    ):
        return "\n".join(
            [
                "Error: this config is for SCF setup, not structure generation.",
                "`pesmaker generate` creates new structures and requires a "
                "`structures:` section.",
                "For your labeling/input_dir workflow, run:",
                f"  pesmaker scf-setup {config_path}",
                "After the SCF folders are prepared, submit them with:",
                f"  pesmaker submit {config_path}",
            ]
        )
    return f"Error: {message}"


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
  cores_cpu: 36
  gpus: 0
  sub_file:
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
    print(format_generate_summary(result, include_details=False), end="")
    print()


def _print_stage_result(result: StageResult) -> None:
    """Print a concise setup or collection result summary."""
    print(result.message)
    print(f"Output directory : {result.output_dir}")
    print(f"Files written    : {len(result.files)}")
    print()


def _print_labeling_result(result: StageResult, config_path: Path) -> None:
    """Print a focused summary for prepared SCF calculation jobs."""
    manifest_path = result.output_dir / "labeling_manifest.jsonl"
    job_count = _manifest_line_count(manifest_path)
    print("SCF setup complete.")
    print(f"Jobs prepared    : {job_count}")
    print(f"Output directory : {result.output_dir}")
    print(f"Manifest         : {manifest_path}")
    print()
    print("Next steps:")
    print(
        f"  - Inspect one job folder under {result.output_dir} "
        "(INCAR, POSCAR, POTCAR, submit.sh)"
    )
    print(f"  - Preview submissions: pesmaker submit {config_path} --dry-run")
    print(f"  - Submit jobs       : pesmaker submit {config_path}")
    print()


def _manifest_line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line)


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
