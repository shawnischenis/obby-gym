#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from obby_rl.transport import StudioHTTPTransport
from obby_rl.vector_env import RobloxObbyBatch
from sb3_contrib import RecurrentPPO


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate recurrent Roblox policy")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--curriculum-stage", type=int, choices=range(1, 24), default=23)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--seed-start", type=int, default=123000)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--jump-threshold", type=float, default=0.65)
    parser.add_argument("--mask-jump-to-takeoff-window", action="store_true")
    args = parser.parse_args()

    model = RecurrentPPO.load(args.model, device="cpu")
    transport = StudioHTTPTransport(timeout=120, curriculum_stage=args.curriculum_stage)
    batch = RobloxObbyBatch(
        transport,
        8,
        jump_threshold=args.jump_threshold,
        jump_cooldown_steps=8,
        yaw_scale=0,
        terminate_on_hazard=True,
        mask_jump_to_takeoff_window=args.mask_jump_to_takeoff_window,
    )
    finished = clean = hazards = 0
    next_seed = args.seed_start
    try:
        while finished < args.episodes:
            seeds = list(range(next_seed, next_seed + batch.num_envs))
            next_seed += batch.num_envs
            observations, _ = batch.reset(seeds)
            active = np.zeros(batch.num_envs, dtype=np.bool_)
            active[: min(batch.num_envs, args.episodes - finished)] = True
            episode_starts = np.ones(batch.num_envs, dtype=np.bool_)
            lstm_states = None
            lengths = np.zeros(batch.num_envs, dtype=np.int32)
            while np.any(active):
                actions, lstm_states = model.predict(
                    observations,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=True,
                )
                actions = np.asarray(actions, dtype=np.float32)
                actions[~active] = 0
                observations, _, terminated, truncated, infos = batch.step(actions)
                lengths[active] += 1
                dones = (terminated | truncated | (lengths >= args.max_steps)) & active
                episode_starts = dones.copy()
                for index in np.flatnonzero(dones):
                    hazard = bool(infos[index].get("hazard_recovered"))
                    completed = bool(terminated[index]) and not hazard
                    clean += int(completed)
                    hazards += int(hazard)
                    print(
                        f"seed={seeds[index]} completed={completed} "
                        f"checkpoint={infos[index].get('checkpoint_index')} "
                        f"steps={lengths[index]}",
                        flush=True,
                    )
                    active[index] = False
                    finished += 1
    finally:
        batch.close()
    print(
        f"Stage {args.curriculum_stage} recurrent clean={clean}/{finished} "
        f"rate={clean / finished:.1%} hazards={hazards}"
    )


if __name__ == "__main__":
    main()
