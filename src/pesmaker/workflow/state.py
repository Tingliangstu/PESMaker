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
"""File-backed state for `pesmaker next`."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pesmaker.config.schema import PESMakerConfig


def next_state_path(config: PESMakerConfig) -> Path:
    """Return the local state path used by `pesmaker next`."""
    return Path(".pesmaker") / _safe_state_part(config.project) / "next_state.json"


def load_next_state(config: PESMakerConfig) -> dict[str, Any]:
    """Load next-state data, returning an empty initialized state if absent."""
    path = next_state_path(config)
    if not path.exists():
        return _empty_state(config)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return _empty_state(config)
    data.setdefault("project", config.project)
    data.setdefault("dry_runs", {})
    return data


def save_next_state(config: PESMakerConfig, state: dict[str, Any]) -> Path:
    """Persist next-state data."""
    path = next_state_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def record_dry_run(
    config: PESMakerConfig,
    state: dict[str, Any],
    *,
    stage: str,
    command: str,
    log_path: Path,
) -> Path:
    """Record that `next` already previewed a stage submission."""
    dry_runs = state.setdefault("dry_runs", {})
    dry_runs[stage] = {
        "command": command,
        "log": str(log_path),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    return save_next_state(config, state)


def dry_run_recorded(state: dict[str, Any], stage: str) -> bool:
    """Return whether a dry-run gate has already been recorded."""
    dry_run = state.get("dry_runs", {}).get(stage)
    if not isinstance(dry_run, dict):
        return False
    log = dry_run.get("log")
    return bool(log and Path(str(log)).exists())


def _empty_state(config: PESMakerConfig) -> dict[str, Any]:
    return {"project": config.project, "dry_runs": {}}


def _safe_state_part(value: str) -> str:
    safe = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_" for char in value
    )
    return safe.strip("_") or "project"
