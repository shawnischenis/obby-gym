from __future__ import annotations

from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from obby_rl.config import ROOT, load_json

PROTOCOL_VERSION = "0.1.0"
OBSERVATION_SCHEMA = "obby-structured-v1"
ACTION_SCHEMA = "obby-action-v1"


def validate_message(message: dict[str, Any], schema_path: Path | None = None) -> None:
    schema = load_json(schema_path or ROOT / "schemas" / "protocol.schema.json")
    Draft202012Validator(schema).validate(message)
