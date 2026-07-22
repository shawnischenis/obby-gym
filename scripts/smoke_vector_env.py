#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import numpy as np
from obby_rl.transport import StudioHTTPTransport
from obby_rl.vector_env import RobloxObbyBatch


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test simultaneous Roblox vector lanes")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--curriculum-stage", type=int, choices=range(1, 24), default=1)
    parser.add_argument("--action-repeat-ticks", type=int, choices=range(1, 7), default=3)
    parser.add_argument("--privileged-observations", action="store_true")
    args = parser.parse_args()
    transport = StudioHTTPTransport(
        timeout=120,
        curriculum_stage=args.curriculum_stage,
        action_repeat_ticks=args.action_repeat_ticks,
    )
    batch = RobloxObbyBatch(
        transport,
        args.num_envs,
        yaw_scale=0,
        privileged_observations=args.privileged_observations,
    )
    try:
        observations, infos = batch.reset(list(range(args.num_envs)))
        started = time.perf_counter()
        total_transitions = 0
        for _ in range(args.steps):
            actions = np.zeros((args.num_envs, 4), dtype=np.float32)
            actions[:, 1] = 1
            observations, _, terminated, truncated, infos = batch.step(actions)
            total_transitions += args.num_envs
            if np.any(terminated | truncated):
                break
        elapsed = time.perf_counter() - started
        print(
            f"vector smoke passed envs={args.num_envs} transitions={total_transitions} "
            f"seconds={elapsed:.3f} transitions_per_second={total_transitions / elapsed:.1f} "
            f"observation_shape={observations.shape} lanes={[info['lane_index'] for info in infos]}"
        )
    finally:
        batch.close()


if __name__ == "__main__":
    main()
