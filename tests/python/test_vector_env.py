from __future__ import annotations

from typing import Any

import numpy as np
from obby_rl.vector_env import RobloxBatchedVecEnv, RobloxObbyBatch


class FakeVectorTransport:
    def __init__(self, hazard: bool = False, terminated_index: int | None = None) -> None:
        self.hazard = hazard
        self.terminated_index = terminated_index
        self.actions: list[list[dict[str, float | bool]]] = []
        self.checkpoint_index = 0
        self.post_landing_masks: list[list[bool]] = []

    def vector_reset(
        self, *, seeds: list[int], post_landing_mask: list[bool] | None = None
    ) -> list[dict[str, Any]]:
        self.post_landing_masks.append(post_landing_mask or [False] * len(seeds))
        return [self._result(index) for index in range(len(seeds))]

    def vector_step(self, actions: list[dict[str, float | bool]]) -> list[dict[str, Any]]:
        self.actions.append(actions)
        return [self._result(index) for index in range(len(actions))]

    def _result(self, index: int) -> dict[str, Any]:
        values = [0.0] * 22
        values[4] = 1.0
        values[8] = 0.25
        return {
            "observation": {"schema": "obby-structured-v1", "values": values},
            "privileged_observation": {
                "schema": "obby-privileged-v1",
                "values": values + [0.0] * 26,
            },
            "reward": 1.0,
            "terminated": index == self.terminated_index,
            "truncated": False,
            "info": {
                "lane_index": index + 1,
                "checkpoint_index": self.checkpoint_index,
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


def test_stage23_can_reset_half_of_lanes_from_post_landing_states() -> None:
    transport = FakeVectorTransport()
    transport.curriculum_stage = 23
    batch = RobloxObbyBatch(transport, 8)  # type: ignore[arg-type]
    env = RobloxBatchedVecEnv(
        batch,
        post_landing_reset_probability=0.5,
        curriculum_sampler_seed=7,
    )
    env.reset()
    assert sum(transport.post_landing_masks[-1]) == 4


def test_checkpoint_advancement_can_receive_extra_training_credit() -> None:
    transport = FakeVectorTransport()
    batch = RobloxObbyBatch(  # type: ignore[arg-type]
        transport, 1, checkpoint_credit_weight=0.4
    )
    batch.reset([0])
    transport.checkpoint_index = 1
    _, rewards, _, _, infos = batch.step(np.zeros((1, 4), dtype=np.float32))
    np.testing.assert_allclose(rewards, [1.4])
    assert infos[0]["reward_components"]["checkpoint_credit"] == np.float32(0.4)
    _, rewards, _, _, _ = batch.step(np.zeros((1, 4), dtype=np.float32))
    np.testing.assert_allclose(rewards, [1.0])


def test_privileged_teacher_mode_uses_separate_48_value_contract() -> None:
    batch = RobloxObbyBatch(  # type: ignore[arg-type]
        FakeVectorTransport(), 2, privileged_observations=True
    )
    observations, _ = batch.reset([0, 1])
    assert observations.shape == (2, 48)
    assert batch.student_observations.shape == (2, 22)
    env = RobloxBatchedVecEnv(batch)
    assert env.observation_space.shape == (48,)
    env.close()


def test_privileged_teacher_observes_decision_phase_but_student_does_not() -> None:
    batch = RobloxObbyBatch(  # type: ignore[arg-type]
        FakeVectorTransport(), 2, privileged_observations=True
    )
    observations, _ = batch.reset([0, 1])
    np.testing.assert_allclose(observations[:, 44], 0)
    observations, _, _, _, _ = batch.step(np.zeros((2, 4), dtype=np.float32))
    np.testing.assert_allclose(observations[:, 44], 1 / 32)
    assert batch.student_observations.shape == (2, 22)


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


def test_held_jump_retriggers_after_cooldown() -> None:
    transport = FakeVectorTransport()
    batch = RobloxObbyBatch(  # type: ignore[arg-type]
        transport, 1, jump_threshold=0, jump_cooldown_steps=1
    )
    batch.reset([0])
    held = np.asarray([[0, 1, 0, 1]], dtype=np.float32)
    batch.step(held)
    batch.step(held)
    batch.step(held)
    assert [step[0]["jump"] for step in transport.actions] == [True, False, True]


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


def test_vector_env_can_reset_entire_cohort_when_one_lane_finishes() -> None:
    batch = RobloxObbyBatch(  # type: ignore[arg-type]
        FakeVectorTransport(terminated_index=0), 2
    )
    env = RobloxBatchedVecEnv(batch, reset_all_on_any_done=True)
    env.reset()
    _, _, dones, infos = env.step(np.zeros((2, 4), dtype=np.float32))
    assert dones.tolist() == [True, True]
    assert infos[1]["cohort_interrupted"] is True
    assert infos[1]["TimeLimit.truncated"] is True
    env.close()


def test_vector_env_can_wait_at_barrier_until_every_lane_finishes() -> None:
    batch = RobloxObbyBatch(  # type: ignore[arg-type]
        FakeVectorTransport(terminated_index=0), 2
    )
    env = RobloxBatchedVecEnv(batch, wait_for_all_done=True, max_episode_steps=2)
    env.reset()
    _, _, dones, _ = env.step(np.ones((2, 4), dtype=np.float32))
    assert dones.tolist() == [False, False]
    _, _, dones, infos = env.step(np.ones((2, 4), dtype=np.float32))
    assert dones.tolist() == [True, True]
    assert batch.transport.actions[1][0] == {
        "strafe": 0.0,
        "forward": 0.0,
        "yaw": 0.0,
        "jump": False,
        "jump_cooldown_remaining": 7,
    }
    assert infos[0]["TimeLimit.truncated"] is False
    assert infos[1]["TimeLimit.truncated"] is True
    env.close()


def test_vector_env_samples_one_curriculum_stage_per_full_reset() -> None:
    transport = FakeVectorTransport()
    transport.curriculum_stage = 1
    batch = RobloxObbyBatch(transport, 2)  # type: ignore[arg-type]
    env = RobloxBatchedVecEnv(
        batch,
        curriculum_replay=[(6, 1.0), (9, 0.000001)],
        curriculum_sampler_seed=7,
    )
    env.reset()
    assert env.current_curriculum_stage == 6
    assert transport.curriculum_stage == 6
    env.close()


def test_vector_env_can_replay_scripted_forward_cohorts() -> None:
    batch = RobloxObbyBatch(FakeVectorTransport(), 2)  # type: ignore[arg-type]
    env = RobloxBatchedVecEnv(batch, scripted_forward_replay_probability=1.0)
    env.reset()
    assert batch.scripted_forward is True
    env.close()
