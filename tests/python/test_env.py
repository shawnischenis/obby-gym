from __future__ import annotations

import numpy as np
from gymnasium.utils.env_checker import check_env
from obby_rl.config import validate_course_config
from obby_rl.env import RobloxObbyEnv
from obby_rl.testing import FakeTransport


def test_gap_course_config_is_valid() -> None:
    config = validate_course_config()
    assert config["gap_min"] <= config["gap_max"]


def test_env_passes_gymnasium_checker() -> None:
    env = RobloxObbyEnv(FakeTransport())
    check_env(env, skip_render_check=True)


def test_env_reset_and_step_contract() -> None:
    transport = FakeTransport()
    env = RobloxObbyEnv(transport)
    observation, info = env.reset(seed=123)
    assert observation.shape == (22,)
    assert info["course_seed"] == 123
    observation, reward, terminated, truncated, info = env.step(
        np.array([0, 1, 0, 1], dtype=np.float32)
    )
    assert observation[12] == np.float32(0.1)
    assert reward == 0.1
    assert not terminated and not truncated
    env.close()
    assert transport.closed


def test_jump_is_edge_triggered_and_cooled_down() -> None:
    transport = FakeTransport()
    env = RobloxObbyEnv(transport, jump_threshold=0.75, jump_cooldown_steps=2)
    env.reset(seed=1)
    low = np.array([0, 0, 0, 0.7], dtype=np.float32)
    high = np.array([0, 0, 0, 0.9], dtype=np.float32)
    env.step(low)
    env.step(high)
    env.step(low)
    env.step(high)
    env.step(low)
    env.step(high)
    assert [action["jump"] for action in transport.actions] == [
        False,
        True,
        False,
        False,
        False,
        True,
    ]
    env.close()


def test_yaw_scale_can_freeze_heading() -> None:
    transport = FakeTransport()
    env = RobloxObbyEnv(transport, yaw_scale=0)
    env.reset(seed=1)
    env.step(np.array([0, 0, 1, 0], dtype=np.float32))
    assert transport.actions[-1]["yaw"] == 0
    env.close()
