#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math

from obby_rl.transport import StudioHTTPTransport


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curriculum-stage", type=int, choices=range(1, 23), default=4)
    args = parser.parse_args()
    transport = StudioHTTPTransport(timeout=120, curriculum_stage=args.curriculum_stage)
    positive_checkpoint_steps = 0
    completed = False
    max_progress = 0.0
    minimum_checkpoint_distance = math.inf
    hazards = 0
    checkpoint_index = 0
    try:
        transport.reset(seed=0)
        for step in range(240):
            jump = args.curriculum_stage in {2, 3} and step % 12 == 0
            response = transport.step({"strafe": 0.0, "forward": 1.0, "yaw": 0.0, "jump": jump})
            components = response.get("info", {}).get("reward_components")
            values = response["observation"]["values"]
            max_progress = max(max_progress, float(values[12]))
            minimum_checkpoint_distance = min(
                minimum_checkpoint_distance,
                math.sqrt(
                    (float(values[6]) * 64) ** 2
                    + (float(values[7]) * 32) ** 2
                    + (float(values[8]) * 64) ** 2
                ),
            )
            hazards += int(bool(response.get("info", {}).get("hazard_recovered")))
            checkpoint_index = max(
                checkpoint_index, int(response.get("info", {}).get("checkpoint_index", 0))
            )
            if not isinstance(components, dict):
                raise RuntimeError("bridge did not return reward_components")
            component_sum = sum(float(value) for value in components.values())
            reward = float(response["reward"])
            if abs(component_sum - reward) > 1e-6:
                raise RuntimeError(
                    f"reward mismatch at step {step}: scalar={reward} components={components}"
                )
            if float(components.get("checkpoint", 0)) > 0:
                positive_checkpoint_steps += 1
            if response["terminated"]:
                completed = True
                break
        if positive_checkpoint_steps == 0:
            raise RuntimeError("forward movement never earned checkpoint-distance reward")
        if args.curriculum_stage in {1, 2} and not completed:
            raise RuntimeError(
                f"scripted controller did not complete stage {args.curriculum_stage}: "
                f"max_progress={max_progress:.3f} "
                f"min_checkpoint_distance={minimum_checkpoint_distance:.3f} "
                f"hazards={hazards} checkpoint_index={checkpoint_index}"
            )
    finally:
        transport.close()
    print(
        f"reward shaping passed stage={args.curriculum_stage} "
        f"positive_checkpoint_steps={positive_checkpoint_steps} completed={completed}"
    )


if __name__ == "__main__":
    main()
