#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from obby_rl.transport import StudioHTTPTransport
from obby_rl.vector_env import RobloxObbyBatch
from stable_baselines3 import PPO


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate deterministic Stage 2 vector policy")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--curriculum-stage", type=int, choices=range(2, 15), default=2)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--policy-movement",
        action="store_true",
        help="Apply the policy's strafe/forward outputs instead of forcing forward movement",
    )
    args = parser.parse_args()
    if args.episodes < 1:
        raise ValueError("episodes must be positive")

    model = PPO.load(args.model, device="cpu")
    transport = StudioHTTPTransport(timeout=120, curriculum_stage=args.curriculum_stage)
    batch = RobloxObbyBatch(
        transport,
        8,
        jump_threshold=0.0,
        jump_cooldown_steps=8,
        yaw_scale=0,
        terminate_on_hazard=True,
        scripted_forward=not args.policy_movement,
    )
    next_seed = args.seed_start + batch.num_envs
    observations, _ = batch.reset(
        list(range(args.seed_start, args.seed_start + batch.num_envs))
    )
    finished = 0
    clean = 0
    hazards = 0
    lengths = np.zeros(batch.num_envs, dtype=np.int32)
    completed_lengths: list[int] = []
    try:
        while finished < args.episodes:
            actions, _ = model.predict(observations, deterministic=True)
            next_observations, _, terminated, truncated, infos = batch.step(actions)
            lengths += 1
            dones = terminated | truncated
            for index in np.flatnonzero(dones):
                if finished >= args.episodes:
                    break
                hazard = bool(infos[index].get("hazard_recovered"))
                hazards += int(hazard)
                clean += int(not hazard and bool(terminated[index]))
                completed_lengths.append(int(lengths[index]))
                lengths[index] = 0
                finished += 1
            if np.any(dones) and finished < args.episodes:
                reset_seeds = [0] * batch.num_envs
                for index in np.flatnonzero(dones):
                    reset_seeds[index] = next_seed
                    next_seed += 1
                reset_observations, _ = batch.reset_lanes(reset_seeds, dones)
                next_observations[dones] = reset_observations[dones]
            observations = next_observations
    finally:
        batch.close()

    print(
        f"Stage {args.curriculum_stage} deterministic vector clean={clean}/{finished} "
        f"rate={clean / finished:.1%} hazards={hazards} "
        f"mean_length={np.mean(completed_lengths):.2f}"
    )


if __name__ == "__main__":
    main()
