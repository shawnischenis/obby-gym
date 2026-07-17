from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol

import gymnasium as gym
import numpy as np


class PredictPolicy(Protocol):
    def predict(
        self,
        observation: np.ndarray,
        state: Any = None,
        episode_start: Any = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Any]: ...


def evaluate_seeds(
    policy: PredictPolicy,
    env: gym.Env[np.ndarray, np.ndarray],
    seeds: Iterable[int],
) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []

    def checkpoint_distance(observation: np.ndarray) -> float:
        return float(
            np.linalg.norm([observation[6] * 64, observation[7] * 32, observation[8] * 64])
        )

    for seed in seeds:
        observation, _ = env.reset(seed=int(seed))
        initial_distance = checkpoint_distance(observation)
        minimum_distance = initial_distance
        terminated = False
        truncated = False
        episode_return = 0.0
        length = 0
        hazards = 0
        previous_action: np.ndarray | None = None
        action_variation = 0.0
        direction_reversals = 0
        final_info: dict[str, Any] = {}
        while not terminated and not truncated:
            action, _ = policy.predict(observation, deterministic=True)
            checked_action = np.asarray(action, dtype=np.float32)
            if previous_action is not None:
                action_variation += float(np.abs(checked_action[:3] - previous_action[:3]).sum())
                direction_reversals += int(np.any(checked_action[:2] * previous_action[:2] < -0.05))
            previous_action = checked_action.copy()
            observation, reward, terminated, truncated, info = env.step(action)
            minimum_distance = min(minimum_distance, checkpoint_distance(observation))
            episode_return += float(reward)
            length += 1
            hazards += int(bool(info.get("hazard_recovered")))
            final_info = info
        episodes.append(
            {
                "seed": int(seed),
                "return": episode_return,
                "length": length,
                "completed": bool(terminated),
                "truncated": bool(truncated),
                "hazards": hazards,
                "checkpoint_index": int(final_info.get("checkpoint_index", 0)),
                "initial_checkpoint_distance": initial_distance,
                "minimum_checkpoint_distance": minimum_distance,
                "final_checkpoint_distance": checkpoint_distance(observation),
                "mean_action_variation": action_variation / max(1, length - 1),
                "direction_reversals": direction_reversals,
            }
        )
    count = len(episodes)
    if count == 0:
        raise ValueError("evaluation requires at least one seed")
    return {
        "episodes": episodes,
        "episode_count": count,
        "completion_rate": sum(int(item["completed"]) for item in episodes) / count,
        "mean_return": sum(float(item["return"]) for item in episodes) / count,
        "mean_length": sum(int(item["length"]) for item in episodes) / count,
        "mean_hazards": sum(int(item["hazards"]) for item in episodes) / count,
        "mean_checkpoint_index": sum(int(item["checkpoint_index"]) for item in episodes) / count,
        "mean_action_variation": sum(float(item["mean_action_variation"]) for item in episodes)
        / count,
        "mean_direction_reversals": sum(int(item["direction_reversals"]) for item in episodes)
        / count,
    }
