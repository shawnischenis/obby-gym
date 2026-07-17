from __future__ import annotations

import json
from pathlib import Path

import pytest
from obby_rl.run_state import finish_run, start_run


def read_state(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def test_run_state_completes_atomically(tmp_path: Path) -> None:
    path = tmp_path / "run_state.json"
    state = start_run(path, target_timesteps=128)
    assert read_state(path)["status"] == "running"
    finish_run(path, state, "complete")
    completed = read_state(path)
    assert completed["status"] == "complete"
    assert completed["finished_at"] is not None
    assert completed["error"] is None


def test_run_state_records_failure(tmp_path: Path) -> None:
    path = tmp_path / "run_state.json"
    state = start_run(path, target_timesteps=10)
    finish_run(path, state, "failed", error="transport timeout")
    failed = read_state(path)
    assert failed["status"] == "failed"
    assert failed["error"] == "transport timeout"


def test_finish_rejects_nonterminal_status(tmp_path: Path) -> None:
    path = tmp_path / "run_state.json"
    state = start_run(path, target_timesteps=10)
    with pytest.raises(ValueError):
        finish_run(path, state, "running")
