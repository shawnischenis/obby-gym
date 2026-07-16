from __future__ import annotations

from obby_rl.env import RobloxObbyEnv
from obby_rl.testing import FakeTransport
from obby_rl.training import make_fixed_course_env


def test_fixed_course_forces_seed_on_every_episode() -> None:
    transport = FakeTransport()
    env = make_fixed_course_env(RobloxObbyEnv(transport), course_seed=17, max_episode_steps=3)
    _, first_info = env.reset(seed=999)
    assert first_info["course_seed"] == 17
    assert first_info["fixed_course_seed"] == 17
    for _ in range(3):
        _, _, _, truncated, _ = env.step(env.action_space.sample())
    assert truncated
    _, second_info = env.reset()
    assert second_info["course_seed"] == 17
    env.close()


def test_fixed_course_rejects_invalid_time_limit() -> None:
    env = RobloxObbyEnv(FakeTransport())
    try:
        try:
            make_fixed_course_env(env, course_seed=0, max_episode_steps=0)
        except ValueError as error:
            assert "positive" in str(error)
        else:
            raise AssertionError("expected invalid time limit to fail")
    finally:
        env.close()
