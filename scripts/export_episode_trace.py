#!/usr/bin/env python3
"""Export one deterministic evaluation episode as a per-step trace for the site.

Writes ``docs/media/episode-trace.json``, which ``docs/script.js`` renders as
the "Episode trace" figure in the Results section. The figure stays hidden
until this file exists, so the site never shows fabricated data.

Requires the same live setup as the evaluate scripts: Roblox Studio playing
the training place with the ObbyRLBridge plugin installed.

Example:
    .venv/bin/python scripts/export_episode_trace.py \
        --model runs/m4-stage20-two-segment-replay-v1/checkpoints/ppo_vector_4096_steps.zip \
        --curriculum-stage 20 --seed 0
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from obby_rl.config import ROOT, load_json
from obby_rl.env import RobloxObbyEnv
from obby_rl.transport import StudioHTTPTransport
from stable_baselines3 import PPO


def checkpoint_distance(observation: np.ndarray) -> float:
    return float(
        np.linalg.norm([observation[6] * 64, observation[7] * 32, observation[8] * 64])
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record one deterministic episode as a per-step trace for the project site"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "m3_fixed_ppo.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--curriculum-stage", type=int, choices=range(1, 24))
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--jump-threshold", type=float)
    parser.add_argument("--jump-cooldown-steps", type=int)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "docs" / "media" / "episode-trace.json"
    )
    args = parser.parse_args()

    config = load_json(args.config)
    mapping = config["action_mapping"]
    stage = int(args.curriculum_stage or config["curriculum_stage"])
    max_steps = int(args.max_steps or config["episode_max_steps"])
    jump_threshold = float(
        args.jump_threshold if args.jump_threshold is not None else mapping["jump_threshold"]
    )

    transport = StudioHTTPTransport(timeout=120, curriculum_stage=stage)
    base_env = RobloxObbyEnv(
        transport,
        jump_threshold=jump_threshold,
        jump_cooldown_steps=int(
            args.jump_cooldown_steps
            if args.jump_cooldown_steps is not None
            else mapping["jump_cooldown_steps"]
        ),
        yaw_scale=float(mapping["yaw_scale"]),
    )
    env = gym.wrappers.TimeLimit(base_env, max_episode_steps=max_steps)
    model = PPO.load(args.model, device="cpu")

    steps: list[dict[str, object]] = []
    terminated = False
    truncated = False
    try:
        observation, _ = env.reset(seed=int(args.seed))
        steps.append({"d": round(checkpoint_distance(observation), 3), "cp": 0, "jump": False})
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            checked_action = np.asarray(action, dtype=np.float32)
            observation, _, terminated, truncated, info = env.step(action)
            steps.append(
                {
                    "d": round(checkpoint_distance(observation), 3),
                    "cp": int(info.get("checkpoint_index", 0)),
                    "jump": bool(float(checked_action[3]) >= jump_threshold),
                }
            )
    finally:
        env.close()

    payload = {
        "meta": {
            "model": str(args.model),
            "seed": int(args.seed),
            "curriculum_stage": stage,
            "completed": bool(terminated),
            "steps": len(steps) - 1,
            "generated": datetime.date.today().isoformat(),
        },
        "steps": steps,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(
        f"wrote {args.output} — {len(steps) - 1} decisions, "
        f"{'completed' if terminated else 'not completed'}, "
        f"final checkpoint {steps[-1]['cp']}"
    )


if __name__ == "__main__":
    main()
