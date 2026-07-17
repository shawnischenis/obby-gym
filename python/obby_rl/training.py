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


EPISODE_INFO_KEYS = (
    "reward_checkpoint",
    "reward_progress",
    "reward_checkpoint_bonus",
    "reward_finish",
    "reward_hazard",
    "reward_time",
    "hazard_count",
    "max_checkpoint_index",
)


class EpisodeMetrics(gym.Wrapper[Any, Any, Any, Any]):
    def __init__(self, env: gym.Env[Any, Any]):
        super().__init__(env)
        self.totals: dict[str, float] = {}
        self.hazards = 0
        self.max_checkpoint = 0

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        self.totals = {}
        self.hazards = 0
        self.max_checkpoint = 0
        return self.env.reset(**kwargs)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        components = info.get("reward_components", {})
        if isinstance(components, dict):
            for name, value in components.items():
                self.totals[str(name)] = self.totals.get(str(name), 0.0) + float(value)
        self.hazards += int(bool(info.get("hazard_recovered")))
        self.max_checkpoint = max(self.max_checkpoint, int(info.get("checkpoint_index", 0)))
        if terminated or truncated:
            info = dict(info)
            for name in ("checkpoint", "progress", "checkpoint_bonus", "finish", "hazard", "time"):
                info[f"reward_{name}"] = self.totals.get(name, 0.0)
            info["hazard_count"] = self.hazards
            info["max_checkpoint_index"] = self.max_checkpoint
        return observation, float(reward), bool(terminated), bool(truncated), info


def make_fixed_course_env(
    env: gym.Env[Any, Any], *, course_seed: int, max_episode_steps: int
) -> gym.Env[Any, Any]:
    if max_episode_steps < 1:
        raise ValueError("max_episode_steps must be positive")
    fixed = FixedCourse(env, course_seed)
    timed: gym.Env[Any, Any] = gym.wrappers.TimeLimit(fixed, max_episode_steps=max_episode_steps)
    return EpisodeMetrics(timed)
