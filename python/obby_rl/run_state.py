from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

RunStatus = Literal["running", "complete", "failed"]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def start_run(path: Path, *, target_timesteps: int) -> dict[str, Any]:
    state = {
        "status": "running",
        "started_at": utc_now(),
        "finished_at": None,
        "target_timesteps": int(target_timesteps),
        "error": None,
    }
    write_json(path, state)
    return state


def finish_run(
    path: Path, state: dict[str, Any], status: RunStatus, *, error: str | None = None
) -> None:
    if status == "running":
        raise ValueError("finish_run requires a terminal status")
    write_json(path, {**state, "status": status, "finished_at": utc_now(), "error": error})
