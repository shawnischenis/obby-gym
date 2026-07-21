#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from obby_rl.feasibility import JumpFeasibilityModel
from obby_rl.transport import StudioHTTPTransport
from obby_rl.vector_env import RobloxObbyBatch
from stable_baselines3 import PPO


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate deterministic Stage 2 vector policy")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--curriculum-stage", type=int, choices=range(1, 23), default=2)
    parser.add_argument("--action-repeat-ticks", type=int, choices=range(1, 7), default=3)
    parser.add_argument("--jump-cooldown-steps", type=int, default=8)
    parser.add_argument("--jump-feasibility-model", type=Path)
    parser.add_argument("--jump-feasibility-threshold", type=float, default=0.5)
    parser.add_argument("--mask-jump-to-takeoff-window", action="store_true")
    parser.add_argument("--force-jump-when-feasible", action="store_true")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--policy-movement",
        action="store_true",
        help="Apply the policy's strafe/forward outputs instead of forcing forward movement",
    )
    parser.add_argument("--privileged-observations", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1:
        raise ValueError("episodes must be positive")

    model = PPO.load(args.model, device="cpu")
    feasibility_model: JumpFeasibilityModel | None = None
    if args.jump_feasibility_model:
        feasibility_model = JumpFeasibilityModel()
        feasibility_model.load_state_dict(
            torch.load(args.jump_feasibility_model, map_location="cpu", weights_only=True)
        )
        feasibility_model.eval()
    transport = StudioHTTPTransport(
        timeout=120,
        curriculum_stage=args.curriculum_stage,
        action_repeat_ticks=args.action_repeat_ticks,
    )
    batch = RobloxObbyBatch(
        transport,
        8,
        jump_threshold=0.0,
        jump_cooldown_steps=args.jump_cooldown_steps,
        yaw_scale=0,
        terminate_on_hazard=True,
        scripted_forward=not args.policy_movement,
        privileged_observations=args.privileged_observations,
        jump_feasibility_model=feasibility_model,
        jump_feasibility_threshold=args.jump_feasibility_threshold,
        mask_jump_to_takeoff_window=args.mask_jump_to_takeoff_window,
        force_jump_when_feasible=args.force_jump_when_feasible,
    )
    next_seed = args.seed_start + batch.num_envs
    observations, _ = batch.reset(list(range(args.seed_start, args.seed_start + batch.num_envs)))
    episode_gaps = observations[:, 9] * 10.0
    episode_angles = observations[:, 11] * 18.0
    active = np.zeros(batch.num_envs, dtype=np.bool_)
    active[: min(args.episodes, batch.num_envs)] = True
    finished = 0
    clean = 0
    hazards = 0
    lengths = np.zeros(batch.num_envs, dtype=np.int32)
    completed_lengths: list[int] = []
    completed_gaps: list[float] = []
    completed_clean: list[bool] = []
    completed_angles: list[float] = []
    completed_checkpoint_indices: list[int] = []
    try:
        while finished < args.episodes:
            actions, _ = model.predict(observations, deterministic=True)
            actions[~active] = 0
            next_observations, _, terminated, truncated, infos = batch.step(actions)
            lengths[active] += 1
            dones = (terminated | truncated) & active
            for index in np.flatnonzero(dones):
                if finished >= args.episodes:
                    break
                hazard = bool(infos[index].get("hazard_recovered"))
                hazards += int(hazard)
                clean += int(not hazard and bool(terminated[index]))
                completed_clean.append(not hazard and bool(terminated[index]))
                completed_lengths.append(int(lengths[index]))
                completed_gaps.append(float(episode_gaps[index]))
                completed_angles.append(float(episode_angles[index]))
                completed_checkpoint_indices.append(int(infos[index].get("checkpoint_index", 0)))
                lengths[index] = 0
                active[index] = False
                finished += 1
            if not np.any(active) and finished < args.episodes:
                observations, _ = batch.reset(list(range(next_seed, next_seed + batch.num_envs)))
                next_seed += batch.num_envs
                episode_gaps = observations[:, 9] * 10.0
                episode_angles = observations[:, 11] * 18.0
                active[: min(args.episodes - finished, batch.num_envs)] = True
            else:
                observations = next_observations
    finally:
        batch.close()

    print(
        f"Stage {args.curriculum_stage} deterministic vector clean={clean}/{finished} "
        f"rate={clean / finished:.1%} hazards={hazards} "
        f"mean_length={np.mean(completed_lengths):.2f}"
    )
    checkpoint_counts = {
        checkpoint: completed_checkpoint_indices.count(checkpoint)
        for checkpoint in sorted(set(completed_checkpoint_indices))
    }
    print(f"  terminal_checkpoint_indices={checkpoint_counts}")
    for lower, upper in zip([5.0, 6.0, 7.0, 8.0, 9.0], [6.0, 7.0, 8.0, 9.0, 10.01], strict=True):
        indices = [index for index, gap in enumerate(completed_gaps) if lower <= gap < upper]
        if indices:
            bin_clean = sum(int(completed_clean[index]) for index in indices)
            print(
                f"  gap=[{lower:.0f},{upper if upper < 10.01 else 10:.0f}) "
                f"clean={bin_clean}/{len(indices)} rate={bin_clean / len(indices):.1%}"
            )
    if any(abs(angle) > 0.01 for angle in completed_angles):
        angle_bins = (
            [
                (-18.01, -12.0),
                (-12.0, -8.0),
                (-8.0, -4.0),
                (-4.0, 0.0),
                (0.0, 4.0),
                (4.0, 8.0),
                (8.0, 12.0),
                (12.0, 18.01),
            ]
            if max(abs(angle) for angle in completed_angles) > 8.01
            else [(-8.01, -4.0), (-4.0, 0.0), (0.0, 4.0), (4.0, 8.01)]
        )
        for lower, upper in angle_bins:
            indices = [
                index for index, angle in enumerate(completed_angles) if lower <= angle < upper
            ]
            if indices:
                bin_clean = sum(int(completed_clean[index]) for index in indices)
                print(
                    f"  angle=[{lower:.0f},{upper:.0f}) clean={bin_clean}/{len(indices)} "
                    f"rate={bin_clean / len(indices):.1%}"
                )


if __name__ == "__main__":
    main()
