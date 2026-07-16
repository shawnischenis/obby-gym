#!/usr/bin/env python3
from __future__ import annotations

import time

from obby_rl.env import RobloxObbyEnv
from obby_rl.transport import StudioHTTPTransport


def main() -> None:
    transport = StudioHTTPTransport(timeout=120)
    env = RobloxObbyEnv(transport)
    print("Waiting for the ObbyRL Studio plugin at http://127.0.0.1:8765 ...")
    observation, info = env.reset(seed=0)
    print(f"connected seed={info['course_seed']} observation_size={observation.size}")
    started = time.perf_counter()
    steps = 0
    try:
        while steps < 100:
            observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
            steps += 1
            if terminated or truncated:
                observation, info = env.reset(seed=steps)
    finally:
        elapsed = time.perf_counter() - started
        env.close()
    print(f"steps={steps} elapsed={elapsed:.3f}s throughput={steps / elapsed:.2f} steps/s")


if __name__ == "__main__":
    main()
