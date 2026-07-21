#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np
from obby_rl.env import RobloxObbyEnv, privileged_observation
from obby_rl.transport import StudioHTTPTransport


def run_group(
    transport: StudioHTTPTransport, jump_steps: list[int], seed: int
) -> list[dict[str, object]]:
    results = transport.vector_reset(seeds=[seed] * len(jump_steps))
    observations = np.stack([RobloxObbyEnv._observation(dict(result)) for result in results])
    privileged = np.stack([privileged_observation(dict(result)) for result in results])
    gaps = observations[:, 9] * 10.0
    outcomes = ["timeout"] * len(jump_steps)
    takeoff_distances = [0.0] * len(jump_steps)
    takeoff_observations: list[list[float] | None] = [None] * len(jump_steps)
    active = [True] * len(jump_steps)
    for step in range(30):
        actions: list[dict[str, float | bool]] = []
        for index, jump_step in enumerate(jump_steps):
            jump = active[index] and step == jump_step
            if jump:
                takeoff_distances[index] = float(
                    np.linalg.norm(observations[index, 6:9] * np.asarray([64.0, 32.0, 64.0]))
                )
                takeoff_observations[index] = privileged[index].tolist()
            actions.append(
                {
                    "strafe": 0.0,
                    "forward": 1.0 if active[index] else 0.0,
                    "yaw": 0.0,
                    "jump": jump,
                }
            )
        step_results = transport.vector_step(actions)
        observations = np.stack(
            [RobloxObbyEnv._observation(dict(result)) for result in step_results]
        )
        privileged = np.stack([privileged_observation(dict(result)) for result in step_results])
        for index, result in enumerate(step_results):
            if not active[index]:
                continue
            if result["terminated"]:
                outcomes[index] = "complete"
                active[index] = False
            elif result.get("info", {}).get("hazard_recovered"):
                outcomes[index] = "hazard"
                active[index] = False
        if not any(active):
            break
    return [
        {
            "jump_step": step,
            "distance": distance,
            "gap": gap,
            "outcome": outcome,
            "observation": observation,
        }
        for step, distance, gap, outcome, observation in zip(
            jump_steps,
            takeoff_distances,
            gaps,
            outcomes,
            takeoff_observations,
            strict=True,
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep vector jump timing for one course seed")
    parser.add_argument("--curriculum-stage", type=int, choices=range(2, 18), default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-jump-step", type=int, default=15)
    parser.add_argument("--action-repeat-ticks", type=int, choices=range(1, 7), default=3)
    args = parser.parse_args()
    transport = StudioHTTPTransport(
        timeout=120,
        curriculum_stage=args.curriculum_stage,
        action_repeat_ticks=args.action_repeat_ticks,
    )
    try:
        results: list[dict[str, object]] = []
        for start in range(0, args.max_jump_step + 1, 8):
            results += run_group(
                transport,
                list(range(start, min(start + 8, args.max_jump_step + 1))),
                args.seed,
            )
    finally:
        transport.close()
    for result in results:
        print(
            f"gap={result['gap']:.3f} jump_step={result['jump_step']:02d} "
            f"distance={result['distance']:.3f} "
            f"outcome={result['outcome']}"
        )
    successful = [result for result in results if result["outcome"] == "complete"]
    if not successful:
        raise RuntimeError("no vector jump timing completed Stage 2")
    print(f"vector jump timing gate passed successful={successful}")


if __name__ == "__main__":
    main()
