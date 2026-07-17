from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn

from obby_rl.env import (
    DEFAULT_JUMP_COOLDOWN_STEPS,
    DEFAULT_JUMP_THRESHOLD,
    OBSERVATION_SIZE,
    RobloxObbyEnv,
)
from obby_rl.transport import StudioHTTPTransport


class RobloxObbyBatch:
    """One synchronous Python controller for multiple Roblox simulation lanes."""

    def __init__(
        self,
        transport: StudioHTTPTransport,
        num_envs: int = 8,
        *,
        jump_threshold: float = DEFAULT_JUMP_THRESHOLD,
        jump_cooldown_steps: int = DEFAULT_JUMP_COOLDOWN_STEPS,
        yaw_scale: float = 1.0,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.transport = transport
        self.num_envs = num_envs
        self.jump_threshold = jump_threshold
        self.jump_cooldown_steps = jump_cooldown_steps
        self.yaw_scale = yaw_scale
        self._jump_active = np.zeros(num_envs, dtype=np.bool_)
        self._jump_cooldown = np.zeros(num_envs, dtype=np.int32)

    @staticmethod
    def _observation(result: dict[str, Any]) -> np.ndarray:
        return RobloxObbyEnv._observation(result)

    def reset(self, seeds: list[int]) -> tuple[np.ndarray, list[dict[str, Any]]]:
        if len(seeds) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} seeds, got {len(seeds)}")
        self._jump_active.fill(False)
        self._jump_cooldown.fill(0)
        results = self.transport.vector_reset(seeds=seeds)
        observations = np.stack([self._observation(dict(result)) for result in results])
        infos = [dict(result.get("info", {})) for result in results]
        return observations, infos

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        checked = np.clip(np.asarray(actions, dtype=np.float32), -1.0, 1.0)
        if checked.shape != (self.num_envs, 4):
            raise ValueError(f"expected action shape ({self.num_envs}, 4), got {checked.shape}")
        commands: list[dict[str, float | bool]] = []
        for index, action in enumerate(checked):
            active = bool(action[3] > self.jump_threshold)
            jump = active and not self._jump_active[index] and self._jump_cooldown[index] == 0
            self._jump_active[index] = active
            if jump:
                self._jump_cooldown[index] = self.jump_cooldown_steps
            elif self._jump_cooldown[index] > 0:
                self._jump_cooldown[index] -= 1
            commands.append(
                {
                    "strafe": float(action[0]),
                    "forward": float(action[1]),
                    "yaw": float(action[2]) * self.yaw_scale,
                    "jump": bool(jump),
                }
            )
        results = self.transport.vector_step(commands)
        observations = np.stack([self._observation(dict(result)) for result in results])
        rewards = np.asarray([result["reward"] for result in results], dtype=np.float32)
        terminated = np.asarray([result["terminated"] for result in results], dtype=np.bool_)
        truncated = np.asarray([result["truncated"] for result in results], dtype=np.bool_)
        infos = [dict(result.get("info", {})) for result in results]
        assert observations.shape == (self.num_envs, OBSERVATION_SIZE)
        return observations, rewards, terminated, truncated, infos

    def reset_lanes(
        self, seeds: list[int], reset_mask: np.ndarray
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        checked_mask = np.asarray(reset_mask, dtype=np.bool_)
        if len(seeds) != self.num_envs or checked_mask.shape != (self.num_envs,):
            raise ValueError("lane reset seeds/mask do not match num_envs")
        self._jump_active[checked_mask] = False
        self._jump_cooldown[checked_mask] = 0
        results = self.transport.vector_reset_lanes(seeds=seeds, reset_mask=checked_mask.tolist())
        observations = np.stack([self._observation(dict(result)) for result in results])
        infos = [dict(result.get("info", {})) for result in results]
        return observations, infos

    def close(self) -> None:
        self.transport.close()


class RobloxBatchedVecEnv(VecEnv):
    """Stable-Baselines3 adapter backed by one synchronous Roblox batch."""

    def __init__(
        self,
        batch: RobloxObbyBatch,
        *,
        course_seed: int = 0,
        max_episode_steps: int = 400,
    ) -> None:
        self.batch = batch
        self.course_seed = int(course_seed)
        self.max_episode_steps = int(max_episode_steps)
        self._actions: np.ndarray | None = None
        self._episode_steps = np.zeros(batch.num_envs, dtype=np.int32)
        observation_space = spaces.Box(-1.0, 1.0, (OBSERVATION_SIZE,), np.float32)
        action_space = spaces.Box(-1.0, 1.0, (4,), np.float32)
        super().__init__(batch.num_envs, observation_space, action_space)

    def _course_seeds(self) -> list[int]:
        return [self.course_seed if seed is None else int(seed) for seed in self._seeds]

    def reset(self) -> np.ndarray:
        observations, infos = self.batch.reset(self._course_seeds())
        self.reset_infos = infos
        self._episode_steps.fill(0)
        self._reset_seeds()
        self._reset_options()
        return observations

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = np.asarray(actions, dtype=np.float32)

    def step_wait(self) -> VecEnvStepReturn:
        if self._actions is None:
            raise RuntimeError("step_async must be called before step_wait")
        observations, rewards, terminated, truncated, infos = self.batch.step(self._actions)
        self._actions = None
        self._episode_steps += 1
        time_limit = self._episode_steps >= self.max_episode_steps
        truncated = truncated | (time_limit & ~terminated)
        dones = terminated | truncated
        if np.any(dones):
            for index in np.flatnonzero(dones):
                infos[index]["terminal_observation"] = observations[index].copy()
                infos[index]["TimeLimit.truncated"] = bool(
                    truncated[index] and not terminated[index]
                )
            reset_observations, reset_infos = self.batch.reset_lanes(self._course_seeds(), dones)
            observations[dones] = reset_observations[dones]
            for index in np.flatnonzero(dones):
                self.reset_infos[index] = reset_infos[index]
            self._episode_steps[dones] = 0
        return observations, rewards, dones, infos

    def close(self) -> None:
        self.batch.close()

    def get_attr(self, attr_name: str, indices: Any = None) -> list[Any]:
        target = self._get_indices(indices)
        return [getattr(self, attr_name) for _ in target]

    def set_attr(self, attr_name: str, value: Any, indices: Any = None) -> None:
        for _ in self._get_indices(indices):
            setattr(self, attr_name, value)

    def env_method(
        self, method_name: str, *method_args: Any, indices: Any = None, **method_kwargs: Any
    ) -> list[Any]:
        method = getattr(self, method_name)
        return [method(*method_args, **method_kwargs) for _ in self._get_indices(indices)]

    def env_is_wrapped(self, wrapper_class: type, indices: Any = None) -> list[bool]:
        return [False for _ in self._get_indices(indices)]
