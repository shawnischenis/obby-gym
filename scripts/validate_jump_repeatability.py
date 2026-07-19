#!/usr/bin/env python3
from __future__ import annotations

from obby_rl.transport import StudioHTTPTransport


def main() -> None:
    attempts = 20
    clean = 0
    transport = StudioHTTPTransport(timeout=120, curriculum_stage=2)
    try:
        for _ in range(attempts):
            transport.reset(seed=0)
            hazards = 0
            completed = False
            for step in range(25):
                response = transport.step(
                    {
                        "strafe": 0.0,
                        "forward": 1.0,
                        "yaw": 0.0,
                        "jump": step == 4,
                    }
                )
                hazards += int(bool(response.get("info", {}).get("hazard_recovered")))
                if response["terminated"]:
                    completed = True
                    break
                if hazards:
                    break
            clean += int(completed and hazards == 0)
    finally:
        transport.close()
    print(f"scripted jump repeatability clean={clean}/{attempts} jump_step=4")
    if clean < 18:
        raise RuntimeError("known-good scripted jump is not at least 90% repeatable")


if __name__ == "__main__":
    main()
