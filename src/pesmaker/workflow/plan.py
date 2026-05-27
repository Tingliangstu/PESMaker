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
"""Build human-readable workflow plans from validated configuration."""

from __future__ import annotations

from dataclasses import dataclass

from pesmaker.config.schema import PESMakerConfig


@dataclass(frozen=True)
class WorkflowStep:
    """One planned workflow step shown to the user.

    Attributes:
        name: Short stage name.
        detail: Human-readable stage summary.
    """

    name: str
    detail: str


@dataclass(frozen=True)
class WorkflowPlan:
    """A complete human-readable plan for a PESMaker project.

    Attributes:
        project: Project name from the configuration.
        steps: Ordered workflow steps.
    """

    project: str
    steps: tuple[WorkflowStep, ...]

    def to_text(self) -> str:
        """Render the workflow plan as plain text.

        Returns:
            Multi-line text suitable for printing in the CLI.
        """
        lines = [f"PESMaker workflow plan for '{self.project}':"]
        for index, step in enumerate(self.steps, start=1):
            lines.append(f"{index}. {step.name}: {step.detail}")
        return "\n".join(lines)


def build_plan(config: PESMakerConfig) -> WorkflowPlan:
    """Build a high-level workflow plan from a PESMaker config.

    Args:
        config: Validated PESMaker configuration.

    Returns:
        Workflow plan summarizing the major stages implied by the config.
    """
    steps = (
        WorkflowStep(
            "load structures",
            f"{len(config.structures)} input structure(s)",
        ),
        WorkflowStep(
            "generate candidates",
            f"{len(config.generation.tasks)} task(s): "
            + ", ".join(
                f"{task.name} supercell={task.supercell}"
                for task in config.generation.tasks
            ),
        ),
        WorkflowStep(
            "prepare MD sampling",
            f"engine={config.sampling.engine}",
        ),
        WorkflowStep(
            "prepare single-point labeling",
            f"engine={config.labeling.engine}",
        ),
        WorkflowStep(
            "assemble dataset",
            f"format={config.dataset.format}, split={config.dataset.split}",
        ),
        WorkflowStep(
            "train potential",
            f"model={config.training.engine}",
        ),
    )
    return WorkflowPlan(project=config.project, steps=steps)
