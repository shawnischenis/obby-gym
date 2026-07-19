#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from obby_rl.config import ROOT
from obby_rl.transport import StudioHTTPTransport


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Studio action/observation synchronization")
    parser.add_argument("--curriculum-stage", type=int, choices=range(1, 15), default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "dynamics-audit.jsonl")
    args = parser.parse_args()
    transport = StudioHTTPTransport(timeout=120, curriculum_stage=args.curriculum_stage)
    rows: list[dict[str, Any]] = []
    try:
        for label, forward in (("forward", 1.0), ("backward", -1.0), ("idle", 0.0)):
            reset = transport.reset(seed=0)
            reset_values = reset["observation"]["values"]
            if max(abs(float(value)) for value in reset_values[:4]) > 0.05:
                raise RuntimeError(f"reset velocity was not zero for {label}: {reset_values[:4]}")
            for index in range(12):
                action = {"strafe": 0.0, "forward": forward, "yaw": 0.0, "jump": False}
                response = transport.step(action)
                info = response["info"]
                transition = info.get("transition", {})
                values = response["observation"]["values"]
                row = {
                    "sequence": label,
                    "index": index,
                    "request_action": action,
                    "observed_previous_action": values[18:22],
                    "reward": response["reward"],
                    "reward_components": info.get("reward_components"),
                    "transition": transition,
                    "studio_command_seconds": info.get("studio_command_seconds"),
                }
                rows.append(row)
                if not info.get("default_controls_disabled"):
                    raise RuntimeError("Roblox default controls are still enabled")
                if info.get("humanoid_auto_rotate"):
                    raise RuntimeError("Humanoid.AutoRotate is still enabled")
                if float(transition.get("hold_seconds", 0)) < 0.045:
                    raise RuntimeError(f"action hold was too short: {transition}")
                observed = values[18:22]
                expected = [0.0, forward, 0.0, 0.0]
                if any(abs(float(a) - b) > 1e-5 for a, b in zip(observed, expected, strict=True)):
                    raise RuntimeError(f"previous-action observation lagged: {row}")
    finally:
        transport.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    forward_delta = sum(
        float(row["transition"]["distance_before"]) - float(row["transition"]["distance_after"])
        for row in rows
        if row["sequence"] == "forward"
    )
    backward_delta = sum(
        float(row["transition"]["distance_before"]) - float(row["transition"]["distance_after"])
        for row in rows
        if row["sequence"] == "backward"
    )
    if forward_delta <= 0 or backward_delta >= 0:
        raise RuntimeError(
            f"distance/reward direction failed: forward={forward_delta} backward={backward_delta}"
        )
    print(
        f"dynamics audit passed rows={len(rows)} "
        f"forward_distance_gain={forward_delta:.3f} backward_distance_gain={backward_delta:.3f} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
