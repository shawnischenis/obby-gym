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
    PRIVILEGED_OBSERVATION_SIZE,
    RobloxObbyEnv,
    privileged_observation,
)
from obby_rl.feasibility import JumpFeasibilityModel, predict_probabilities
from obby_rl.transport import StudioHTTPTransport

PRIVILEGED_DECISION_PHASE_INDEX = 44
PRIVILEGED_DECISION_PHASE_SCALE = 32.0


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
        privileged_observations: bool = False,
        jump_feasibility_model: JumpFeasibilityModel | None = None,
        jump_feasibility_threshold: float = 0.5,
        force_jump_when_feasible: bool = False,
        checkpoint_credit_weight: float = 0.0,
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
        self.privileged_observations = bool(privileged_observations)
        if jump_feasibility_model is not None and not privileged_observations:
            raise ValueError("jump feasibility shaping requires privileged observations")
        if not 0 < jump_feasibility_threshold < 1:
            raise ValueError("jump feasibility threshold must be within (0, 1)")
        self.jump_feasibility_model = jump_feasibility_model
        self.jump_feasibility_threshold = float(jump_feasibility_threshold)
        if force_jump_when_feasible and jump_feasibility_model is None:
            raise ValueError("forced feasible jumps require a feasibility model")
        self.force_jump_when_feasible = bool(force_jump_when_feasible)
        if checkpoint_credit_weight < 0:
            raise ValueError("checkpoint credit weight must be non-negative")
        self.checkpoint_credit_weight = float(checkpoint_credit_weight)
        self.observation_size = (
            PRIVILEGED_OBSERVATION_SIZE if self.privileged_observations else OBSERVATION_SIZE
        )
        self._jump_active = np.zeros(num_envs, dtype=np.bool_)
        self._jump_cooldown = np.zeros(num_envs, dtype=np.int32)
        self._previous_movement = np.zeros((num_envs, 2), dtype=np.float32)
        self._has_previous_movement = np.zeros(num_envs, dtype=np.bool_)
        self._last_observations = np.zeros((num_envs, self.observation_size), dtype=np.float32)
        self._last_student_observations = np.zeros((num_envs, OBSERVATION_SIZE), dtype=np.float32)
        self._decision_steps = np.zeros(num_envs, dtype=np.int32)
        self._checkpoint_indices = np.zeros(num_envs, dtype=np.int32)

    @property
    def student_observations(self) -> np.ndarray:
        """Current limited-sensing observations paired with teacher observations."""
        return self._last_student_observations.copy()

    def _update_student_observations(self, results: list[dict[str, Any]]) -> None:
        student = np.stack([RobloxObbyEnv._observation(result) for result in results])
        student[:, 5] = (self._jump_cooldown == 0).astype(np.float32)
        self._last_student_observations[:] = student

    def _observation(self, result: dict[str, Any]) -> np.ndarray:
        if self.privileged_observations:
            return privileged_observation(result)
        return RobloxObbyEnv._observation(result)

    def _add_privileged_decision_phase(self, observations: np.ndarray) -> None:
        """Expose episode timing to the teacher without changing student sensing."""
        if self.privileged_observations:
            observations[:, PRIVILEGED_DECISION_PHASE_INDEX] = np.clip(
                self._decision_steps / PRIVILEGED_DECISION_PHASE_SCALE, 0.0, 1.0
            )

    def reset(
        self, seeds: list[int], post_landing_mask: np.ndarray | None = None
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        if len(seeds) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} seeds, got {len(seeds)}")
        self._jump_active.fill(False)
        self._jump_cooldown.fill(0)
        self._previous_movement.fill(0)
        self._has_previous_movement.fill(False)
        self._decision_steps.fill(0)
        landing_mask = (
            np.zeros(self.num_envs, dtype=np.bool_)
            if post_landing_mask is None
            else np.asarray(post_landing_mask, dtype=np.bool_)
        )
        if landing_mask.shape != (self.num_envs,):
            raise ValueError("post-landing mask must match num_envs")
        self._checkpoint_indices[:] = landing_mask.astype(np.int32)
        if np.any(landing_mask):
            results = self.transport.vector_reset(
                seeds=seeds, post_landing_mask=landing_mask.tolist()
            )
        else:
            results = self.transport.vector_reset(seeds=seeds)
        checked_results = [dict(result) for result in results]
        observations = np.stack([self._observation(result) for result in checked_results])
        observations[:, 5] = 1.0
        self._add_privileged_decision_phase(observations)
        self._last_observations[:] = observations
        self._update_student_observations(checked_results)
        infos = [dict(result.get("info", {})) for result in results]
        for index, info in enumerate(infos):
            if "restored_jump_cooldown" in info:
                self._jump_cooldown[index] = int(info["restored_jump_cooldown"])
                observations[index, 5] = float(self._jump_cooldown[index] == 0)
                self._last_observations[index, 5] = observations[index, 5]
                self._last_student_observations[index, 5] = observations[index, 5]
        self._checkpoint_indices[:] = np.asarray(
            [int(info.get("checkpoint_index", 0)) for info in infos], dtype=np.int32
        )
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
        pre_step_local_velocity = self._last_observations[:, :3].copy()
        pre_step_grounded = self._last_observations[:, 4] > 0.5
        pre_step_route_features = self._last_observations[:, 9:12].copy()
        pre_step_progress = self._last_observations[:, 12].copy()
        gap = self._last_observations[:, 9] * 10.0
        gap_shift = np.where(gap > 0, gap - 7.0, 0.0)
        takeoff_window = (
            (checkpoint_distance >= self.jump_timing_distance[0] + gap_shift)
            & (checkpoint_distance <= self.jump_timing_distance[1] + gap_shift)
            & (self._last_observations[:, 4] > 0.5)
        )
        feasibility_probability: np.ndarray | None = None
        if self.jump_feasibility_model is not None:
            feasibility_probability = predict_probabilities(
                self.jump_feasibility_model, self._last_observations
            )
            takeoff_window = (feasibility_probability >= self.jump_feasibility_threshold) & (
                self._last_observations[:, 4] > 0.5
            )
        commands: list[dict[str, float | bool]] = []
        for index, action in enumerate(checked):
            active = bool(action[3] > self.jump_threshold) or (
                self.force_jump_when_feasible and bool(takeoff_window[index])
            )
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
                    "jump_cooldown_remaining": int(self._jump_cooldown[index]),
                }
            )
        results = self.transport.vector_step(commands)
        self._decision_steps += 1
        checked_results = [dict(result) for result in results]
        observations = np.stack([self._observation(result) for result in checked_results])
        observations[:, 5] = (self._jump_cooldown == 0).astype(np.float32)
        self._add_privileged_decision_phase(observations)
        rewards = np.asarray([result["reward"] for result in results], dtype=np.float32)
        current_checkpoint_indices = np.asarray(
            [result.get("info", {}).get("checkpoint_index", 0) for result in results],
            dtype=np.int32,
        )
        checkpoint_advances = np.maximum(
            current_checkpoint_indices - self._checkpoint_indices, 0
        )
        checkpoint_credit = (
            self.checkpoint_credit_weight * checkpoint_advances
        ).astype(np.float32)
        rewards += checkpoint_credit
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
            components["checkpoint_credit"] = float(checkpoint_credit[index])
            info["reward_components"] = components
            info["movement_action_delta"] = float(movement_delta[index])
            info["jump_takeoff_distance"] = float(checkpoint_distance[index])
            info["jump_takeoff_allowed"] = bool(takeoff_window[index])
            info["jump_intent_active"] = bool(raw_jump_active[index])
            info["jump_intent_value"] = float(checked[index, 3])
            info["jump_command_applied"] = bool(jump_commands[index])
            info["jump_cooldown_remaining"] = int(self._jump_cooldown[index])
            info["pre_step_local_velocity"] = [
                float(value) for value in pre_step_local_velocity[index]
            ]
            info["pre_step_grounded"] = bool(pre_step_grounded[index])
            info["pre_step_route_features"] = [
                float(value) for value in pre_step_route_features[index]
            ]
            info["pre_step_progress"] = float(pre_step_progress[index])
            if feasibility_probability is not None:
                info["jump_feasibility_probability"] = float(feasibility_probability[index])
        self._previous_movement[:] = applied_movement
        self._has_previous_movement.fill(True)
        self._last_observations[:] = observations
        self._checkpoint_indices[:] = current_checkpoint_indices
        self._update_student_observations(checked_results)
        hazard_recovered = np.asarray(
            [bool(info.get("hazard_recovered")) for info in infos], dtype=np.bool_
        )
        if np.any(hazard_recovered):
            # Roblox has already moved these rigs back to their latest recovery
            # checkpoint. Start the new life with neutral controller state too,
            # rather than carrying the failed jump's cooldown or held movement.
            self._jump_active[hazard_recovered] = False
            self._jump_cooldown[hazard_recovered] = 0
            self._previous_movement[hazard_recovered] = 0
            self._has_previous_movement[hazard_recovered] = False
            observations[hazard_recovered, 5] = 1.0
            self._last_observations[hazard_recovered, 5] = 1.0
            self._last_student_observations[hazard_recovered, 5] = 1.0
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
        assert observations.shape == (self.num_envs, self.observation_size)
        return observations, rewards, terminated, truncated, infos

    def reset_lanes(
        self,
        seeds: list[int],
        reset_mask: np.ndarray,
        post_landing_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        checked_mask = np.asarray(reset_mask, dtype=np.bool_)
        if len(seeds) != self.num_envs or checked_mask.shape != (self.num_envs,):
            raise ValueError("lane reset seeds/mask do not match num_envs")
        self._jump_active[checked_mask] = False
        self._jump_cooldown[checked_mask] = 0
        self._previous_movement[checked_mask] = 0
        self._has_previous_movement[checked_mask] = False
        self._decision_steps[checked_mask] = 0
        landing_mask = (
            np.zeros(self.num_envs, dtype=np.bool_)
            if post_landing_mask is None
            else np.asarray(post_landing_mask, dtype=np.bool_)
        )
        if landing_mask.shape != (self.num_envs,):
            raise ValueError("post-landing mask must match num_envs")
        landing_mask &= checked_mask
        self._checkpoint_indices[checked_mask] = landing_mask[checked_mask].astype(np.int32)
        if np.any(landing_mask):
            results = self.transport.vector_reset_lanes(
                seeds=seeds,
                reset_mask=checked_mask.tolist(),
                post_landing_mask=landing_mask.tolist(),
            )
        else:
            results = self.transport.vector_reset_lanes(
                seeds=seeds, reset_mask=checked_mask.tolist()
            )
        checked_results = [dict(result) for result in results]
        observations = np.stack([self._observation(result) for result in checked_results])
        observations[:, 5] = (self._jump_cooldown == 0).astype(np.float32)
        self._add_privileged_decision_phase(observations)
        self._last_observations[checked_mask] = observations[checked_mask]
        student = np.stack([RobloxObbyEnv._observation(result) for result in checked_results])
        student[:, 5] = (self._jump_cooldown == 0).astype(np.float32)
        self._last_student_observations[checked_mask] = student[checked_mask]
        infos = [dict(result.get("info", {})) for result in results]
        for index, info in enumerate(infos):
            if checked_mask[index] and "restored_jump_cooldown" in info:
                self._jump_cooldown[index] = int(info["restored_jump_cooldown"])
                observations[index, 5] = float(self._jump_cooldown[index] == 0)
                student[index, 5] = observations[index, 5]
                self._last_observations[index, 5] = observations[index, 5]
                self._last_student_observations[index, 5] = observations[index, 5]
        self._checkpoint_indices[checked_mask] = np.asarray(
            [int(info.get("checkpoint_index", 0)) for info in infos], dtype=np.int32
        )[checked_mask]
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
        reset_all_on_any_done: bool = False,
        wait_for_all_done: bool = False,
        curriculum_replay: list[tuple[int, float]] | None = None,
        curriculum_sampler_seed: int = 0,
        scripted_forward_replay_probability: float = 0.0,
        post_landing_reset_probability: float = 0.0,
    ) -> None:
        self.batch = batch
        self.render_mode = None
        self.course_seed = int(course_seed)
        self.vary_course_seeds = bool(vary_course_seeds)
        self.reset_all_on_any_done = bool(reset_all_on_any_done)
        self.wait_for_all_done = bool(wait_for_all_done)
        if self.reset_all_on_any_done and self.wait_for_all_done:
            raise ValueError("cohort reset modes are mutually exclusive")
        self.curriculum_replay = curriculum_replay or []
        if any(stage < 1 or weight <= 0 for stage, weight in self.curriculum_replay):
            raise ValueError("curriculum replay stages and weights must be positive")
        self._curriculum_rng = np.random.default_rng(curriculum_sampler_seed)
        if not 0 <= scripted_forward_replay_probability <= 1:
            raise ValueError("scripted forward replay probability must be within [0, 1]")
        self.scripted_forward_replay_probability = float(scripted_forward_replay_probability)
        if not 0 <= post_landing_reset_probability <= 1:
            raise ValueError("post-landing reset probability must be within [0, 1]")
        self.post_landing_reset_probability = float(post_landing_reset_probability)
        self.current_curriculum_stage = int(getattr(batch.transport, "curriculum_stage", 0))
        self._next_course_seed = self.course_seed
        self.max_episode_steps = int(max_episode_steps)
        self._actions: np.ndarray | None = None
        self._episode_steps = np.zeros(batch.num_envs, dtype=np.int32)
        self._cohort_finished = np.zeros(batch.num_envs, dtype=np.bool_)
        self._cohort_terminal_observations = np.zeros(
            (batch.num_envs, batch.observation_size), dtype=np.float32
        )
        self._cohort_terminal_infos: list[dict[str, Any] | None] = [None] * batch.num_envs
        self._cohort_was_terminated = np.zeros(batch.num_envs, dtype=np.bool_)
        observation_space = spaces.Box(-1.0, 1.0, (batch.observation_size,), np.float32)
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

    def _post_landing_mask(self, reset_mask: np.ndarray | None = None) -> np.ndarray:
        eligible = (
            np.ones(self.num_envs, dtype=np.bool_)
            if reset_mask is None
            else np.asarray(reset_mask, dtype=np.bool_).copy()
        )
        result = np.zeros(self.num_envs, dtype=np.bool_)
        if (
            self.current_curriculum_stage not in {20, 21, 22, 23}
            or self.post_landing_reset_probability <= 0
        ):
            return result
        indices = np.flatnonzero(eligible)
        if reset_mask is None:
            count = int(round(len(indices) * self.post_landing_reset_probability))
            if count > 0:
                selected = self._curriculum_rng.choice(indices, size=count, replace=False)
                result[selected] = True
        else:
            result[indices] = (
                self._curriculum_rng.random(len(indices))
                < self.post_landing_reset_probability
            )
        return result

    def _reset_batch(self, seeds: list[int]) -> tuple[np.ndarray, list[dict[str, Any]]]:
        if self.scripted_forward_replay_probability > 0:
            self.batch.scripted_forward = bool(
                self._curriculum_rng.random() < self.scripted_forward_replay_probability
            )
        if self.curriculum_replay:
            stages = np.asarray([stage for stage, _ in self.curriculum_replay], dtype=np.int32)
            weights = np.asarray([weight for _, weight in self.curriculum_replay], dtype=np.float64)
            weights /= weights.sum()
            self.current_curriculum_stage = int(self._curriculum_rng.choice(stages, p=weights))
            self.batch.transport.curriculum_stage = self.current_curriculum_stage
        return self.batch.reset(seeds, self._post_landing_mask())

    def reset(self) -> np.ndarray:
        observations, infos = self._reset_batch(self._course_seeds())
        self.reset_infos = infos
        self._episode_steps.fill(0)
        self._cohort_finished.fill(False)
        self._cohort_was_terminated.fill(False)
        self._cohort_terminal_infos = [None] * self.num_envs
        self._reset_seeds()
        self._reset_options()
        return observations

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = np.asarray(actions, dtype=np.float32)

    def step_wait(self) -> VecEnvStepReturn:
        if self._actions is None:
            raise RuntimeError("step_async must be called before step_wait")
        actions = self._actions.copy()
        if self.wait_for_all_done:
            actions[self._cohort_finished] = 0
        observations, rewards, terminated, truncated, infos = self.batch.step(actions)
        self._actions = None
        active_before_step = ~self._cohort_finished
        self._episode_steps[active_before_step] += 1
        if self.wait_for_all_done:
            rewards[~active_before_step] = 0
        time_limit = self._episode_steps >= self.max_episode_steps
        truncated = truncated | (time_limit & ~terminated)
        dones = terminated | truncated
        if self.wait_for_all_done:
            newly_finished = dones & active_before_step
            for index in np.flatnonzero(newly_finished):
                self._cohort_terminal_observations[index] = observations[index]
                self._cohort_terminal_infos[index] = dict(infos[index])
                self._cohort_was_terminated[index] = bool(terminated[index])
            self._cohort_finished |= newly_finished
            if not np.all(self._cohort_finished):
                return observations, rewards, np.zeros(self.num_envs, dtype=np.bool_), infos
            dones = np.ones(self.num_envs, dtype=np.bool_)
            terminated = self._cohort_was_terminated.copy()
            truncated = ~terminated
            for index in range(self.num_envs):
                stored_info = self._cohort_terminal_infos[index]
                if stored_info is not None:
                    infos[index] = stored_info
                observations[index] = self._cohort_terminal_observations[index]
        if self.reset_all_on_any_done and np.any(dones):
            cohort_interrupted = ~dones
            truncated = truncated | cohort_interrupted
            dones = np.ones(self.num_envs, dtype=np.bool_)
            for index in np.flatnonzero(cohort_interrupted):
                infos[index]["cohort_interrupted"] = True
        if np.any(dones):
            for index in np.flatnonzero(dones):
                infos[index]["terminal_observation"] = observations[index].copy()
                infos[index]["TimeLimit.truncated"] = bool(
                    truncated[index] and not terminated[index]
                )
            if self.reset_all_on_any_done or self.wait_for_all_done:
                reset_observations, reset_infos = self._reset_batch(self._course_seeds())
            else:
                reset_observations, reset_infos = self.batch.reset_lanes(
                    self._course_seeds(dones), dones, self._post_landing_mask(dones)
                )
            observations[dones] = reset_observations[dones]
            for index in np.flatnonzero(dones):
                self.reset_infos[index] = reset_infos[index]
            self._episode_steps[dones] = 0
            if self.wait_for_all_done:
                self._cohort_finished.fill(False)
                self._cohort_was_terminated.fill(False)
                self._cohort_terminal_infos = [None] * self.num_envs
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
