from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from obby_rl.env import OBSERVATION_SIZE


class FakeTransport:
    """Deterministic contract double used before a Studio worker is connected."""

    def __init__(self) -> None:
        self.step_id = 0
        self.closed = False
        self.actions: list[Mapping[str, float | bool]] = []

    def _response(self, reward: float = 0.0) -> Mapping[str, Any]:
        values = [0.0] * OBSERVATION_SIZE
        values[12] = min(self.step_id / 10, 1.0)
        return {
            "observation": {"schema": "obby-structured-v1", "values": values},
            "reward": reward,
            "terminated": self.step_id >= 10,
            "truncated": False,
            "info": {"step_id": self.step_id},
        }

    def reset(self, *, seed: int) -> Mapping[str, Any]:
        self.step_id = 0
        response = dict(self._response())
        response["info"] = {"step_id": 0, "course_seed": seed}
        return response

    def step(self, action: Mapping[str, float | bool]) -> Mapping[str, Any]:
        self.actions.append(dict(action))
        self.step_id += 1
        return self._response(reward=0.1)

    def close(self) -> None:
        self.closed = True
