#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any

from obby_rl.transport import StudioHTTPTransport


def course_identity(response: dict[str, Any]) -> tuple[str, int]:
    info = response.get("info", {})
    signature = info.get("course_signature")
    instance_count = info.get("course_instances")
    if not isinstance(signature, str) or not signature:
        raise RuntimeError("M2 response is missing course_signature")
    if not isinstance(instance_count, int) or instance_count <= 0:
        raise RuntimeError(f"invalid course_instances: {instance_count!r}")
    if info.get("checkpoint_index") != 0:
        raise RuntimeError(f"reset did not clear checkpoint progress: {info!r}")
    if info.get("checkpoint_count") != 8:
        raise RuntimeError(f"unexpected checkpoint count: {info!r}")
    return signature, instance_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate deterministic M2 Studio generation")
    parser.add_argument("--seeds", type=int, default=25)
    args = parser.parse_args()
    if args.seeds < 1:
        raise ValueError("--seeds must be positive")
    seeds = tuple(range(args.seeds))
    transport = StudioHTTPTransport(timeout=120)
    identities: dict[int, list[tuple[str, int]]] = defaultdict(list)
    try:
        for pass_index in range(2):
            for seed in seeds:
                try:
                    response = transport.reset(seed=seed)
                except Exception as error:
                    raise RuntimeError(
                        f"reset failed at pass={pass_index + 1} seed={seed}"
                    ) from error
                identities[seed].append(course_identity(response))
                if (seed + 1) % 50 == 0:
                    print(f"pass={pass_index + 1} resets={seed + 1}/{len(seeds)}", flush=True)
            print(f"completed deterministic reset pass {pass_index + 1}")

        transport.reset(seed=0)
        recovered = False
        velocity_x_range = [1.0, -1.0]
        checkpoint_x_range = [1.0, -1.0]
        down_ray_range = [1.0, -1.0]
        minimum_reward = float("inf")
        for _ in range(120):
            response = transport.step({"strafe": 1.0, "forward": 0.0, "yaw": 0.0, "jump": False})
            values = response["observation"]["values"]
            velocity_x_range = [
                min(velocity_x_range[0], float(values[0])),
                max(velocity_x_range[1], float(values[0])),
            ]
            checkpoint_x_range = [
                min(checkpoint_x_range[0], float(values[6])),
                max(checkpoint_x_range[1], float(values[6])),
            ]
            down_ray_range = [
                min(down_ray_range[0], float(values[15])),
                max(down_ray_range[1], float(values[15])),
            ]
            minimum_reward = min(minimum_reward, float(response["reward"]))
            if response.get("info", {}).get("hazard_recovered"):
                if float(response["reward"]) >= -0.5:
                    raise RuntimeError("hazard recovery did not apply the expected penalty")
                if response["terminated"]:
                    raise RuntimeError("hazard recovery incorrectly terminated the episode")
                recovered = True
                break
        if not recovered:
            raise RuntimeError(
                "failed to trigger hazard recovery: "
                f"velocity_x={velocity_x_range} checkpoint_x={checkpoint_x_range} "
                f"down_ray={down_ray_range} minimum_reward={minimum_reward:.4f}"
            )
        print("hazard penalty and non-terminal checkpoint recovery passed")
    finally:
        transport.close()

    for seed, samples in identities.items():
        if samples[0] != samples[1]:
            raise RuntimeError(f"seed {seed} did not replay identically: {samples}")
    unique_signatures = {samples[0][0] for samples in identities.values()}
    if len(unique_signatures) != len(seeds):
        raise RuntimeError("different seeds produced duplicate manifest signatures")
    instance_counts = sorted({samples[0][1] for samples in identities.values()})
    print(
        f"M2 live validation passed seeds={len(seeds)} "
        f"unique_signatures={len(unique_signatures)} instance_counts={instance_counts}"
    )


if __name__ == "__main__":
    main()
