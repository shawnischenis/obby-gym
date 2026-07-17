from __future__ import annotations

from typing import Any

import numpy as np
from obby_rl.vector_env import RobloxObbyBatch


class FakeVectorTransport:
    def vector_reset(self, *, seeds: list[int]) -> list[dict[str, Any]]:
        return [self._result(index) for index in range(len(seeds))]

    def vector_step(self, actions: list[dict[str, float | bool]]) -> list[dict[str, Any]]:
        return [self._result(index) for index in range(len(actions))]

    @staticmethod
    def _result(index: int) -> dict[str, Any]:
        return {
            "observation": {"schema": "obby-structured-v1", "values": [0.0] * 22},
            "reward": 1.0,
            "terminated": False,
            "truncated": False,
            "info": {"lane_index": index + 1, "reward_components": {}},
        }

    def close(self) -> None:
        return


def test_smoothness_penalty_ignores_first_action_and_penalizes_changes() -> None:
    batch = RobloxObbyBatch(FakeVectorTransport(), 2, smoothness_weight=0.1)  # type: ignore[arg-type]
    batch.reset([0, 1])
    first = np.asarray([[0.5, 0.5, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    _, rewards, _, _, infos = batch.step(first)
    np.testing.assert_allclose(rewards, [1, 1])
    second = np.asarray([[-0.5, 1, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    _, rewards, _, _, infos = batch.step(second)
    np.testing.assert_allclose(rewards, [0.85, 1.0])
    assert infos[0]["reward_components"]["smoothness"] == np.float32(-0.15)
    assert infos[1]["movement_action_delta"] == 0
