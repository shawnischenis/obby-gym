#!/usr/bin/env python3
from __future__ import annotations

from obby_rl.transport import StudioHTTPTransport


def main() -> None:
    transport = StudioHTTPTransport(timeout=120, curriculum_stage=2)
    successes: list[int] = []
    trials: list[tuple[int, str, int]] = []
    try:
        for jump_step in range(14):
            transport.reset(seed=0)
            outcome = "timeout"
            length = 0
            for step in range(35):
                response = transport.step(
                    {
                        "strafe": 0.0,
                        "forward": 1.0,
                        "yaw": 0.0,
                        "jump": step == jump_step,
                    }
                )
                length = step + 1
                if response["terminated"]:
                    outcome = "complete"
                    successes.append(jump_step)
                    break
                if response.get("info", {}).get("hazard_recovered"):
                    outcome = "hazard"
                    break
            trials.append((jump_step, outcome, length))
    finally:
        transport.close()
    print("jump timing trials:")
    for jump_step, outcome, length in trials:
        print(f"  jump_step={jump_step:02d} outcome={outcome} length={length}")
    if not successes:
        raise RuntimeError("no scripted jump timing can complete the 7-stud Stage 2 course")
    print(f"jump timing gate passed successful_steps={successes}")


if __name__ == "__main__":
    main()
