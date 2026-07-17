#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
from obby_rl.transport import StudioHTTPTransport
from obby_rl.vector_env import RobloxBatchedVecEnv, RobloxObbyBatch


def main() -> None:
    transport = StudioHTTPTransport(timeout=120, curriculum_stage=1)
    batch = RobloxObbyBatch(transport, 8, yaw_scale=0)
    env = RobloxBatchedVecEnv(batch, course_seed=0, max_episode_steps=100)
    try:
        observations = env.reset()
        idle_before = observations[1:].copy()
        for step in range(1, 101):
            actions = np.zeros((8, 4), dtype=np.float32)
            actions[0, 1] = 1
            step_observations, _, dones, infos = env.step(actions)
            observations = np.asarray(step_observations)
            if np.any(dones):
                done_lanes = (np.flatnonzero(dones) + 1).tolist()
                if done_lanes != [1]:
                    raise RuntimeError(f"expected only lane 1 to finish, got {done_lanes}")
                if "terminal_observation" not in infos[0]:
                    raise RuntimeError("lane 1 terminal observation was not preserved")
                if not np.all(np.isfinite(observations)):
                    raise RuntimeError("auto-reset returned non-finite observations")
                idle_drift = float(np.max(np.abs(observations[1:] - idle_before)))
                print(
                    f"vector autoreset passed step={step} reset_lane=1 "
                    f"untouched_lanes=7 max_idle_observation_drift={idle_drift:.6f}"
                )
                return
        raise RuntimeError("lane 1 did not finish within 100 vector steps")
    finally:
        env.close()


if __name__ == "__main__":
    main()
