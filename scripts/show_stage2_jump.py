#!/usr/bin/env python3
from __future__ import annotations

import argparse

import gymnasium as gym
from obby_rl.env import RobloxObbyEnv
from obby_rl.evaluation import evaluate_seeds
from obby_rl.transport import StudioHTTPTransport
from stable_baselines3 import PPO


def main() -> None:
    parser = argparse.ArgumentParser(description="Show deterministic Stage 2 jump attempts")
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args()
    if args.episodes < 1:
        raise ValueError("episodes must be positive")
    transport = StudioHTTPTransport(timeout=120, curriculum_stage=2)
    env = gym.wrappers.TimeLimit(
        RobloxObbyEnv(transport, jump_threshold=0.0, jump_cooldown_steps=8, yaw_scale=0),
        max_episode_steps=120,
    )
    model = PPO.load("runs/m3-vector-stage2-isolated-jump-8192/final_model.zip", device="cpu")
    try:
        result = evaluate_seeds(model, env, [0] * args.episodes)
    finally:
        env.close()
    clean = sum(
        int(episode["completed"] and episode["hazards"] == 0) for episode in result["episodes"]
    )
    print(f"Stage 2 visible attempts clean={clean}/{args.episodes}")
    for index, episode in enumerate(result["episodes"], start=1):
        print(
            f"  attempt={index} complete={episode['completed']} "
            f"length={episode['length']} hazards={episode['hazards']}"
        )


if __name__ == "__main__":
    main()
