from __future__ import annotations

from copy import deepcopy

import pytest
from jsonschema import ValidationError

from obby_rl.config import config_sha256, validate_config, validate_seed_partitions
from obby_rl.protocol import validate_message


def test_m0_config_is_valid_and_stably_hashable() -> None:
    config = validate_config()
    assert config_sha256(config) == config_sha256(deepcopy(config))
    assert len(config_sha256(config)) == 64


def test_seed_partitions_are_disjoint() -> None:
    config = validate_config()
    config["generator"]["seed_partitions"]["test"]["start"] = 5
    with pytest.raises(ValueError, match="overlap"):
        validate_seed_partitions(config)


def test_reset_request_contract() -> None:
    validate_message(
        {
            "protocol_version": "0.1.0",
            "message_type": "reset_request",
            "request_id": "request-1",
            "episode_id": "episode-1",
            "course_seed": 1000000,
            "generator_version": "0.1.0",
        }
    )


def test_action_rejects_out_of_range_value() -> None:
    message = {
        "protocol_version": "0.1.0",
        "message_type": "action_request",
        "request_id": "request-2",
        "episode_id": "episode-1",
        "step_id": 0,
        "action": {"strafe": 0, "forward": 2, "yaw": 0, "jump": False},
    }
    with pytest.raises(ValidationError):
        validate_message(message)
