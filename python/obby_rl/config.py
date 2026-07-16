from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def validate_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or ROOT / "configs" / "m0.json"
    config = load_json(config_path)
    schema = load_json(ROOT / "schemas" / "config.schema.json")
    Draft202012Validator(schema).validate(config)
    validate_seed_partitions(config)
    return config


def validate_seed_partitions(config: dict[str, Any]) -> None:
    partitions = config["generator"]["seed_partitions"]
    occupied: list[tuple[int, int, str]] = []
    for name, partition in partitions.items():
        start = int(partition["start"])
        end = start + int(partition["count"])
        for other_start, other_end, other_name in occupied:
            if start < other_end and other_start < end:
                raise ValueError(f"seed partitions overlap: {name} and {other_name}")
        occupied.append((start, end, name))


def config_sha256(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()
