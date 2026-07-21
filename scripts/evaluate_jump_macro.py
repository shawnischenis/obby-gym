#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np
from obby_rl.transport import StudioHTTPTransport
from obby_rl.vector_env import RobloxObbyBatch


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a jump-on-first-step macro")
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--curriculum-stage", type=int, choices=range(2, 18), default=17)
    parser.add_argument("--seed-start", type=int, default=7_000_000)
    parser.add_argument("--action-repeat-ticks", type=int, choices=range(1, 7), default=3)
    parser.add_argument("--jump-cooldown-steps", type=int, default=8)
    parser.add_argument("--jump-step", type=int, default=0)
    args = parser.parse_args()
    transport = StudioHTTPTransport(
        timeout=120,
        curriculum_stage=args.curriculum_stage,
        action_repeat_ticks=args.action_repeat_ticks,
    )
    batch = RobloxObbyBatch(
        transport,
        8,
        jump_threshold=0,
        jump_cooldown_steps=args.jump_cooldown_steps,
        yaw_scale=0,
        terminate_on_hazard=True,
        scripted_forward=True,
    )
    finished = 0
    clean = 0
    hazards = 0
    next_seed = args.seed_start
    try:
        while finished < args.episodes:
            cohort_size = min(batch.num_envs, args.episodes - finished)
            batch.reset(list(range(next_seed, next_seed + batch.num_envs)))
            next_seed += batch.num_envs
            active = np.zeros(batch.num_envs, dtype=np.bool_)
            active[:cohort_size] = True
            decision_step = 0
            while np.any(active):
                actions = np.zeros((batch.num_envs, 4), dtype=np.float32)
                actions[:, 1] = 1
                actions[:, 3] = 1 if decision_step == args.jump_step else -1
                actions[~active] = 0
                _, _, terminated, truncated, infos = batch.step(actions)
                decision_step += 1
                dones = active & (terminated | truncated)
                for index in np.flatnonzero(dones):
                    hazard = bool(infos[index].get("hazard_recovered"))
                    hazards += int(hazard)
                    clean += int(bool(terminated[index]) and not hazard)
                    finished += 1
                    active[index] = False
    finally:
        batch.close()
    print(
        f"jump macro Stage {args.curriculum_stage} clean={clean}/{finished} "
        f"rate={clean / finished:.1%} hazards={hazards} "
        f"action_repeat_ticks={args.action_repeat_ticks} jump_step={args.jump_step}"
    )


if __name__ == "__main__":
    main()
