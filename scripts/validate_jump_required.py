#!/usr/bin/env python3
from __future__ import annotations

from obby_rl.transport import StudioHTTPTransport


def main() -> None:
    transport = StudioHTTPTransport(timeout=120, curriculum_stage=2)
    hazards = 0
    completed = False
    try:
        reset = transport.reset(seed=0)
        course_signature = reset.get("info", {}).get("course_signature")
        for _ in range(60):
            response = transport.step({"strafe": 0.0, "forward": 1.0, "yaw": 0.0, "jump": False})
            hazards += int(bool(response.get("info", {}).get("hazard_recovered")))
            if response["terminated"]:
                completed = True
                break
    finally:
        transport.close()
    if completed:
        raise RuntimeError("Stage 2 can still be completed without issuing jump")
    if hazards == 0:
        raise RuntimeError("no-jump controller never reached the Stage 2 hazard")
    print(
        f"jump-required gate passed completed={completed} hazards={hazards} "
        f"course_signature={course_signature}"
    )


if __name__ == "__main__":
    main()
