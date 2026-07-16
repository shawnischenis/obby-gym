from __future__ import annotations

from typing import Any

import gymnasium as gym


class FixedCourse(gym.Wrapper[Any, Any, Any, Any]):
    """Force every episode reset to rebuild one fixed procedural course seed."""

    def __init__(self, env: gym.Env[Any, Any], course_seed: int):
        super().__init__(env)
        self.course_seed = int(course_seed)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        observation, info = self.env.reset(seed=self.course_seed, options=options)
        return observation, {**info, "fixed_course_seed": self.course_seed}


def make_fixed_course_env(
    env: gym.Env[Any, Any], *, course_seed: int, max_episode_steps: int
) -> gym.Env[Any, Any]:
    if max_episode_steps < 1:
        raise ValueError("max_episode_steps must be positive")
    fixed = FixedCourse(env, course_seed)
    return gym.wrappers.TimeLimit(fixed, max_episode_steps=max_episode_steps)
