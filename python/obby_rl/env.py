from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

import gymnasium as gym
import numpy as np
from gymnasium import spaces

OBSERVATION_SIZE = 22
DEFAULT_JUMP_THRESHOLD = 0.75
DEFAULT_JUMP_COOLDOWN_STEPS = 8


class ObbyTransport(Protocol):
    def reset(self, *, seed: int) -> Mapping[str, Any]: ...

    def step(self, action: Mapping[str, float | bool]) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


class RobloxObbyEnv(gym.Env[np.ndarray, np.ndarray]):
    """Gymnasium adapter around a synchronous Roblox transport."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        transport: ObbyTransport,
        *,
        jump_threshold: float = DEFAULT_JUMP_THRESHOLD,
        jump_cooldown_steps: int = DEFAULT_JUMP_COOLDOWN_STEPS,
        yaw_scale: float = 1.0,
    ):
        super().__init__()
        if not -1 <= jump_threshold <= 1:
            raise ValueError("jump_threshold must be within [-1, 1]")
        if jump_cooldown_steps < 0:
            raise ValueError("jump_cooldown_steps must be non-negative")
        if not 0 <= yaw_scale <= 1:
            raise ValueError("yaw_scale must be within [0, 1]")
        self.transport = transport
        self.jump_threshold = float(jump_threshold)
        self.jump_cooldown_steps = int(jump_cooldown_steps)
        self.yaw_scale = float(yaw_scale)
        self._jump_active = False
        self._jump_cooldown = 0
        self.observation_space = spaces.Box(-1.0, 1.0, (OBSERVATION_SIZE,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (4,), np.float32)

    @staticmethod
    def _observation(response: Mapping[str, Any]) -> np.ndarray:
        observation = response["observation"]
        if observation["schema"] != "obby-structured-v1":
            raise ValueError(f"unsupported observation schema: {observation['schema']}")
        values = np.asarray(observation["values"], dtype=np.float32)
        if values.shape != (OBSERVATION_SIZE,):
            raise ValueError(f"expected {OBSERVATION_SIZE} observations, got {values.shape}")
        return np.clip(values, -1.0, 1.0)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._jump_active = False
        self._jump_cooldown = 0
        course_seed = int(seed if seed is not None else self.np_random.integers(0, 2**31))
        response = self.transport.reset(seed=course_seed)
        observation = self._observation(response)
        observation[5] = 1.0
        return observation, {"course_seed": course_seed, **response.get("info", {})}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        checked = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        if checked.shape != (4,):
            raise ValueError(f"expected action shape (4,), got {checked.shape}")
        jump_active = bool(checked[3] > self.jump_threshold)
        jump = jump_active and self._jump_cooldown == 0
        self._jump_active = jump_active
        if jump:
            self._jump_cooldown = self.jump_cooldown_steps
        elif self._jump_cooldown > 0:
            self._jump_cooldown -= 1
        response = self.transport.step(
            {
                "strafe": float(checked[0]),
                "forward": float(checked[1]),
                "yaw": float(checked[2]) * self.yaw_scale,
                "jump": jump,
            }
        )
        observation = self._observation(response)
        observation[5] = float(self._jump_cooldown == 0)
        return (
            observation,
            float(response["reward"]),
            bool(response["terminated"]),
            bool(response["truncated"]),
            dict(response.get("info", {})),
        )

    def close(self) -> None:
        self.transport.close()
