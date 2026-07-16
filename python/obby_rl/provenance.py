from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from obby_rl.config import ROOT, config_sha256

TRACKED_PACKAGES = (
    "gymnasium",
    "jsonschema",
    "numpy",
    "stable-baselines3",
    "sb3-contrib",
    "torch",
)


def _git(*args: str, root: Path = ROOT) -> str:
    result = subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def collect_provenance(config: dict[str, Any], root: Path = ROOT) -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for package in TRACKED_PACKAGES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None

    return {
        "git": {
            "commit": _git("rev-parse", "HEAD", root=root),
            "dirty": bool(_git("status", "--porcelain", root=root)),
        },
        "config": {
            "version": config["config_version"],
            "sha256": config_sha256(config),
        },
        "schemas": {
            "protocol": config["bridge"]["protocol_version"],
            "observation": config["schemas"]["observation"],
            "action": config["schemas"]["action"],
        },
        "generator": {
            "version": config["generator"]["version"],
            "seed_partitions": config["generator"]["seed_partitions"],
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": packages,
        },
        "roblox_studio": {"version": None},
    }
