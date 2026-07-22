#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

from obby_rl.transport import StudioHTTPTransport


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cycle seeded Roblox obbies at a recording-friendly cadence"
    )
    parser.add_argument("--count", type=int, default=12, help="number of courses to show")
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=2.0,
        help="time to leave each generated course visible after reset completes",
    )
    parser.add_argument(
        "--curriculum-stage",
        type=int,
        choices=range(1, 24),
        default=4,
        help="stage 4 is the full eight-segment procedural obby",
    )
    args = parser.parse_args()
    if args.count < 1:
        raise ValueError("--count must be positive")
    if args.hold_seconds < 0:
        raise ValueError("--hold-seconds must be non-negative")

    transport = StudioHTTPTransport(timeout=120, curriculum_stage=args.curriculum_stage)
    print("Waiting for the ObbyRL Studio plugin at http://127.0.0.1:8765 ...", flush=True)
    try:
        for index in range(args.count):
            seed = args.seed_start + index
            response = transport.reset(seed=seed)
            info = response.get("info", {})
            print(
                f"course={index + 1}/{args.count} seed={seed} "
                f"signature={info.get('course_signature', 'unknown')}",
                flush=True,
            )
            if index + 1 < args.count:
                time.sleep(args.hold_seconds)
        print("Finished cycling courses; the final seed remains visible.", flush=True)
    except KeyboardInterrupt:
        print("Stopped; the current course remains visible.", flush=True)
    finally:
        transport.close()


if __name__ == "__main__":
    main()
