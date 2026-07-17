#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
from obby_rl.config import ROOT, load_json
from obby_rl.env import RobloxObbyEnv
from obby_rl.evaluation import evaluate_seeds
from obby_rl.transport import StudioHTTPTransport
from stable_baselines3 import PPO


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an M3 fixed-course PPO checkpoint")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "m3_fixed_ppo.json")
    parser.add_argument("--fixed-episodes", type=int)
    parser.add_argument("--validation-seeds", type=int)
    parser.add_argument("--curriculum-stage", type=int, choices=range(1, 5))
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = load_json(args.config)
    fixed_episodes = int(args.fixed_episodes or config["evaluation"]["fixed_episodes"])
    validation_count = int(args.validation_seeds or config["evaluation"]["validation_seed_count"])
    max_steps = int(args.max_steps or config["episode_max_steps"])
    if fixed_episodes < 1 or validation_count < 1 or max_steps < 1:
        raise ValueError("evaluation counts and max steps must be positive")

    mapping = config["action_mapping"]
    transport = StudioHTTPTransport(
        timeout=120,
        curriculum_stage=int(args.curriculum_stage or config["curriculum_stage"]),
    )
    base_env = RobloxObbyEnv(
        transport,
        jump_threshold=float(mapping["jump_threshold"]),
        jump_cooldown_steps=int(mapping["jump_cooldown_steps"]),
        yaw_scale=float(mapping["yaw_scale"]),
    )
    env = gym.wrappers.TimeLimit(base_env, max_episode_steps=max_steps)
    model = PPO.load(args.model, device="cpu")
    try:
        fixed = evaluate_seeds(model, env, [int(config["course_seed"])] * fixed_episodes)
        start = int(config["evaluation"]["validation_seed_start"])
        validation = evaluate_seeds(model, env, range(start, start + validation_count))
    finally:
        env.close()

    results = {"model": str(args.model), "fixed": fixed, "validation": validation}
    output = args.output or args.model.parent / "evaluation.json"
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "fixed": fixed | {"episodes": "omitted"},
                "validation": validation | {"episodes": "omitted"},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
