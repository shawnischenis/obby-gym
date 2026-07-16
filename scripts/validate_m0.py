#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from obby_rl.config import config_sha256, validate_config  # noqa: E402
from obby_rl.protocol import validate_message  # noqa: E402


def main() -> None:
    config = validate_config()
    validate_message(
        {
            "protocol_version": "0.1.0",
            "message_type": "reset_request",
            "request_id": "m0-smoke-request",
            "episode_id": "m0-smoke-episode",
            "course_seed": config["generator"]["seed_partitions"]["validation"]["start"],
            "generator_version": config["generator"]["version"],
        }
    )
    print("M0 configuration and protocol schemas are valid")
    print(f"config_sha256={config_sha256(config)}")


if __name__ == "__main__":
    main()
