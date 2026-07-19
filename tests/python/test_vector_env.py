from __future__ import annotations

from typing import Any

import numpy as np
from obby_rl.vector_env import RobloxBatchedVecEnv, RobloxObbyBatch


class FakeVectorTransport:
    def __init__(self, hazard: bool = False) -> None:
        self.hazard = hazard

    def vector_reset(self, *, seeds: list[int]) -> list[dict[str, Any]]:
        return [self._result(index) for index in range(len(seeds))]

    def vector_step(self, actions: list[dict[str, float | bool]]) -> list[dict[str, Any]]:
        return [self._result(index) for index in range(len(actions))]

    def _result(self, index: int) -> dict[str, Any]:
        values = [0.0] * 22
        values[4] = 1.0
        values[8] = 0.25
        return {
            "observation": {"schema": "obby-structured-v1", "values": values},
            "reward": 1.0,
            "terminated": False,
            "truncated": False,
            "info": {
                "lane_index": index + 1,
                "reward_components": {},
                "hazard_recovered": self.hazard,
            },
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


def test_hazard_can_end_vector_trial() -> None:
    batch = RobloxObbyBatch(  # type: ignore[arg-type]
        FakeVectorTransport(hazard=True), 2, terminate_on_hazard=True
    )
    batch.reset([0, 1])
    actions = np.zeros((2, 4), dtype=np.float32)
    _, _, terminated, _, _ = batch.step(actions)
    assert terminated.tolist() == [True, True]


def test_jump_timing_bonus_requires_grounded_jump_in_window() -> None:
    batch = RobloxObbyBatch(  # type: ignore[arg-type]
        FakeVectorTransport(), 2, jump_threshold=0, jump_timing_weight=0.2
    )
    batch.reset([0, 1])
    actions = np.asarray([[0, 1, 0, 1], [0, 1, 0, -1]], dtype=np.float32)
    _, rewards, _, _, infos = batch.step(actions)
    np.testing.assert_allclose(rewards, [1.2, 1.0])
    assert infos[0]["reward_components"]["jump_timing"] == np.float32(0.2)
    assert infos[1]["reward_components"]["jump_timing"] == 0


def test_early_raw_jump_signal_is_penalized() -> None:
    batch = RobloxObbyBatch(  # type: ignore[arg-type]
        FakeVectorTransport(), 1, jump_threshold=0, jump_mistiming_weight=0.05
    )
    batch.reset([0])
    batch._last_observations[0, 8] = 0.35
    actions = np.asarray([[0, 1, 0, 1]], dtype=np.float32)
    _, rewards, _, _, infos = batch.step(actions)
    np.testing.assert_allclose(rewards, [0.95])
    assert infos[0]["reward_components"]["jump_mistiming"] == np.float32(-0.05)


def test_jump_timing_window_shifts_with_observed_gap() -> None:
    batch = RobloxObbyBatch(  # type: ignore[arg-type]
        FakeVectorTransport(), 2, jump_threshold=0, jump_timing_weight=0.2
    )
    batch.reset([0, 1])
    batch._last_observations[:, 8] = 14 / 64
    batch._last_observations[0, 9] = 0.7
    batch._last_observations[1, 9] = 1.0
    actions = np.asarray([[0, 1, 0, 1], [0, 1, 0, 1]], dtype=np.float32)
    _, rewards, _, _, _ = batch.step(actions)
    np.testing.assert_allclose(rewards, [1.2, 1.0])


def test_vector_env_assigns_new_seed_only_to_reset_lanes() -> None:
    batch = RobloxObbyBatch(FakeVectorTransport(), 2)  # type: ignore[arg-type]
    env = RobloxBatchedVecEnv(batch, course_seed=100, vary_course_seeds=True)
    assert env._course_seeds() == [100, 101]
    assert env._course_seeds(np.asarray([False, True])) == [100, 102]
    env.close()
