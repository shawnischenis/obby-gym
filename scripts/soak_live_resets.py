#!/usr/bin/env python3
from __future__ import annotations

import statistics
import time

from obby_rl.transport import StudioHTTPTransport

RESET_COUNT = 1000


def main() -> None:
    transport = StudioHTTPTransport(timeout=120)
    latencies: list[float] = []
    instance_counts: set[int] = set()
    started = time.perf_counter()
    try:
        for seed in range(RESET_COUNT):
            before = time.perf_counter()
            response = transport.reset(seed=seed)
            latencies.append(time.perf_counter() - before)
            info = response.get("info", {})
            instance_counts.add(int(info.get("course_instances", -1)))
            if response["observation"]["schema"] != "obby-structured-v1":
                raise RuntimeError(f"schema mismatch at reset {seed}")
            if len(response["observation"]["values"]) != 22:
                raise RuntimeError(f"observation size mismatch at reset {seed}")
            if (seed + 1) % 100 == 0:
                print(f"resets={seed + 1} latest_ms={latencies[-1] * 1000:.1f}")
    finally:
        transport.close()

    elapsed = time.perf_counter() - started
    ordered = sorted(latencies)
    p95 = ordered[int(0.95 * (len(ordered) - 1))]
    print(
        f"completed={len(latencies)} elapsed={elapsed:.3f}s "
        f"resets_per_s={len(latencies) / elapsed:.2f} "
        f"median_ms={statistics.median(latencies) * 1000:.1f} "
        f"p95_ms={p95 * 1000:.1f} course_instance_counts={sorted(instance_counts)}"
    )
    if len(instance_counts) != 1 or -1 in instance_counts:
        raise RuntimeError(f"generated course instance count drifted: {instance_counts}")


if __name__ == "__main__":
    main()
