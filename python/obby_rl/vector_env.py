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
        smoothness_weight: float = 0.0,
        terminate_on_hazard: bool = False,
        jump_timing_weight: float = 0.0,
        jump_timing_distance: tuple[float, float] = (12.0, 18.0),
        mask_jump_to_takeoff_window: bool = False,
        scripted_forward: bool = False,
        jump_mistiming_weight: float = 0.0,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.transport = transport
        self.num_envs = num_envs
        self.jump_threshold = jump_threshold
        self.jump_cooldown_steps = jump_cooldown_steps
        self.yaw_scale = yaw_scale
        if smoothness_weight < 0:
            raise ValueError("smoothness_weight must be non-negative")
        self.smoothness_weight = float(smoothness_weight)
        self.terminate_on_hazard = bool(terminate_on_hazard)
        if jump_timing_weight < 0:
            raise ValueError("jump_timing_weight must be non-negative")
        if jump_timing_distance[0] >= jump_timing_distance[1]:
            raise ValueError("jump timing distance window must be increasing")
        self.jump_timing_weight = float(jump_timing_weight)
        self.jump_timing_distance = jump_timing_distance
        self.mask_jump_to_takeoff_window = bool(mask_jump_to_takeoff_window)
        self.scripted_forward = bool(scripted_forward)
        if jump_mistiming_weight < 0:
            raise ValueError("jump_mistiming_weight must be non-negative")
        self.jump_mistiming_weight = float(jump_mistiming_weight)
        self._jump_active = np.zeros(num_envs, dtype=np.bool_)
        self._jump_cooldown = np.zeros(num_envs, dtype=np.int32)
        self._previous_movement = np.zeros((num_envs, 2), dtype=np.float32)
        self._has_previous_movement = np.zeros(num_envs, dtype=np.bool_)
        self._last_observations = np.zeros((num_envs, OBSERVATION_SIZE), dtype=np.float32)

    @staticmethod
    def _observation(result: dict[str, Any]) -> np.ndarray:
        return RobloxObbyEnv._observation(result)

    def reset(self, seeds: list[int]) -> tuple[np.ndarray, list[dict[str, Any]]]:
        if len(seeds) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} seeds, got {len(seeds)}")
        self._jump_active.fill(False)
        self._jump_cooldown.fill(0)
        self._previous_movement.fill(0)
        self._has_previous_movement.fill(False)
        results = self.transport.vector_reset(seeds=seeds)
        observations = np.stack([self._observation(dict(result)) for result in results])
        observations[:, 5] = 1.0
        self._last_observations[:] = observations
        infos = [dict(result.get("info", {})) for result in results]
        return observations, infos

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        checked = np.clip(np.asarray(actions, dtype=np.float32), -1.0, 1.0)
        if checked.shape != (self.num_envs, 4):
            raise ValueError(f"expected action shape ({self.num_envs}, 4), got {checked.shape}")
        checkpoint_distance = np.linalg.norm(
            self._last_observations[:, 6:9] * np.asarray([64.0, 32.0, 64.0]), axis=1
        )
        gap = self._last_observations[:, 9] * 10.0
        gap_shift = np.where(gap > 0, gap - 7.0, 0.0)
        takeoff_window = (
            (checkpoint_distance >= self.jump_timing_distance[0] + gap_shift)
            & (checkpoint_distance <= self.jump_timing_distance[1] + gap_shift)
            & (self._last_observations[:, 4] > 0.5)
        )
        commands: list[dict[str, float | bool]] = []
        for index, action in enumerate(checked):
            active = bool(action[3] > self.jump_threshold)
            jump_allowed = not self.mask_jump_to_takeoff_window or bool(takeoff_window[index])
            jump = active and self._jump_cooldown[index] == 0 and jump_allowed
            self._jump_active[index] = active
            if jump:
                self._jump_cooldown[index] = self.jump_cooldown_steps
            elif self._jump_cooldown[index] > 0:
                self._jump_cooldown[index] -= 1
            commands.append(
                {
                    "strafe": 0.0 if self.scripted_forward else float(action[0]),
                    "forward": 1.0 if self.scripted_forward else float(action[1]),
                    "yaw": 0.0 if self.scripted_forward else float(action[2]) * self.yaw_scale,
                    "jump": bool(jump),
                }
            )
        results = self.transport.vector_step(commands)
        observations = np.stack([self._observation(dict(result)) for result in results])
        observations[:, 5] = (self._jump_cooldown == 0).astype(np.float32)
        rewards = np.asarray([result["reward"] for result in results], dtype=np.float32)
        jump_commands = np.asarray([bool(command["jump"]) for command in commands])
        raw_jump_active = checked[:, 3] > self.jump_threshold
        jump_timing_bonus = (self.jump_timing_weight * (jump_commands & takeoff_window)).astype(
            np.float32
        )
        jump_mistiming_penalty = (
            -self.jump_mistiming_weight
            * (raw_jump_active & ~takeoff_window & (self._last_observations[:, 4] > 0.5))
        ).astype(np.float32)
        rewards += jump_timing_bonus + jump_mistiming_penalty
        applied_movement = (
            np.tile(np.asarray([0.0, 1.0], dtype=np.float32), (self.num_envs, 1))
            if self.scripted_forward
            else checked[:, :2]
        )
        movement_delta = np.abs(applied_movement - self._previous_movement).sum(axis=1)
        smoothness_penalty = np.where(
            self._has_previous_movement, -self.smoothness_weight * movement_delta, 0.0
        ).astype(np.float32)
        rewards += smoothness_penalty
        infos = [dict(result.get("info", {})) for result in results]
        for index, info in enumerate(infos):
            components = dict(info.get("reward_components", {}))
            components["smoothness"] = float(smoothness_penalty[index])
            components["jump_timing"] = float(jump_timing_bonus[index])
            components["jump_mistiming"] = float(jump_mistiming_penalty[index])
            info["reward_components"] = components
            info["movement_action_delta"] = float(movement_delta[index])
            info["jump_takeoff_distance"] = float(checkpoint_distance[index])
            info["jump_takeoff_allowed"] = bool(takeoff_window[index])
        self._previous_movement[:] = applied_movement
        self._has_previous_movement.fill(True)
        self._last_observations[:] = observations
        terminated = np.asarray(
            [
                bool(result["terminated"])
                or (
                    self.terminate_on_hazard
                    and bool(result.get("info", {}).get("hazard_recovered"))
                )
                for result in results
            ],
            dtype=np.bool_,
        )
        truncated = np.asarray([result["truncated"] for result in results], dtype=np.bool_)
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
        self._previous_movement[checked_mask] = 0
        self._has_previous_movement[checked_mask] = False
        results = self.transport.vector_reset_lanes(seeds=seeds, reset_mask=checked_mask.tolist())
        observations = np.stack([self._observation(dict(result)) for result in results])
        observations[:, 5] = (self._jump_cooldown == 0).astype(np.float32)
        self._last_observations[checked_mask] = observations[checked_mask]
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
        vary_course_seeds: bool = False,
    ) -> None:
        self.batch = batch
        self.render_mode = None
        self.course_seed = int(course_seed)
        self.vary_course_seeds = bool(vary_course_seeds)
        self._next_course_seed = self.course_seed
        self.max_episode_steps = int(max_episode_steps)
        self._actions: np.ndarray | None = None
        self._episode_steps = np.zeros(batch.num_envs, dtype=np.int32)
        observation_space = spaces.Box(-1.0, 1.0, (OBSERVATION_SIZE,), np.float32)
        action_space = spaces.Box(-1.0, 1.0, (4,), np.float32)
        super().__init__(batch.num_envs, observation_space, action_space)

    def _course_seeds(self, reset_mask: np.ndarray | None = None) -> list[int]:
        if self.vary_course_seeds:
            mask = (
                np.ones(self.num_envs, dtype=np.bool_)
                if reset_mask is None
                else np.asarray(reset_mask, dtype=np.bool_)
            )
            seeds = [self.course_seed] * self.num_envs
            for index in np.flatnonzero(mask):
                seeds[index] = self._next_course_seed
                self._next_course_seed += 1
            return seeds
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
            reset_observations, reset_infos = self.batch.reset_lanes(
                self._course_seeds(dones), dones
            )
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
