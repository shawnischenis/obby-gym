#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np
from obby_rl.env import RobloxObbyEnv
from obby_rl.transport import StudioHTTPTransport
from stable_baselines3 import PPO


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace policy and edge sensing on an approach")
    parser.add_argument(
        "--model",
        default="runs/m3-vector-stage1-continuous-hold-lease250-8192/final_model.zip",
    )
    parser.add_argument("--curriculum-stage", type=int, choices=range(2, 15), default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    model = PPO.load(args.model, device="cpu")
    transport = StudioHTTPTransport(timeout=120, curriculum_stage=args.curriculum_stage)
    try:
        response = transport.reset(seed=args.seed)
        observation = RobloxObbyEnv._observation(response)
        for step in range(args.steps):
            action, _ = model.predict(observation, deterministic=True)
            distance = float(np.linalg.norm(observation[6:9] * np.asarray([64.0, 32.0, 64.0])))
            print(
                f"step={step:02d} checkpoint_distance={distance:.3f} "
                f"grounded={observation[4]:.0f} raw_jump={float(action[3]):+.4f} "
                f"gap={observation[9] * 10:.3f} height={observation[10] * 3:.3f} "
                f"angle={observation[11] * 18:.3f} "
                f"forward_ray={observation[13]:.3f} edge_ray={observation[14]:.3f} "
                f"down_ray={observation[15]:.3f}"
            )
            response = transport.step({"strafe": 0.0, "forward": 1.0, "yaw": 0.0, "jump": False})
            observation = RobloxObbyEnv._observation(response)
            if response.get("info", {}).get("hazard_recovered"):
                break
    finally:
        transport.close()


if __name__ == "__main__":
    main()
