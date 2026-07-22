from __future__ import annotations

import json
import threading
import urllib.request
from typing import Any

import pytest
from obby_rl.transport import StudioHTTPTransport, TransportTimeout


def _post(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=2) as response:
        value = json.load(response)
    assert isinstance(value, dict)
    return value


def _studio_worker(transport: StudioHTTPTransport, commands: int) -> None:
    host, port = transport.address
    url = f"http://{host}:{port}/exchange"
    response: dict[str, Any] = {"protocol_version": "0.1.0"}
    handled = 0
    while handled < commands:
        command = _post(url, response)
        if command["message_type"] == "noop":
            continue
        response = {
            "protocol_version": "0.1.0",
            "message_type": "step_result",
            "ack_request_id": command["request_id"],
            "episode_id": command["episode_id"],
            "step_id": command.get("step_id", 0),
            "observation": {"schema": "obby-structured-v1", "values": [0.0] * 22},
            "reward": 0.0,
            "terminated": False,
            "truncated": False,
            "info": {
                "course_seed": command.get("course_seed"),
                "curriculum_stage": command.get("curriculum_stage"),
            },
        }
        handled += 1
    _post(url, response)


def _vector_worker(transport: StudioHTTPTransport, commands: int) -> None:
    host, port = transport.address
    url = f"http://{host}:{port}/exchange"
    response: dict[str, Any] = {"protocol_version": "0.1.0"}
    for _ in range(commands):
        command = _post(url, response)
        while command["message_type"] == "noop":
            command = _post(url, response)
        count = len(command.get("course_seeds", command.get("actions", [])))
        response = {
            "protocol_version": "0.1.0",
            "message_type": "vector_step_result",
            "ack_request_id": command["request_id"],
            "episode_id": command["episode_id"],
            "step_id": command.get("step_id", 0),
            "results": [
                {
                    "observation": {"schema": "obby-structured-v1", "values": [0.0] * 22},
                    "reward": 0.0,
                    "terminated": False,
                    "truncated": False,
                    "info": {"lane_index": index + 1},
                }
                for index in range(count)
            ],
        }
    _post(url, response)


def test_reset_and_step_round_trip() -> None:
    transport = StudioHTTPTransport(port=0, timeout=2)
    worker = threading.Thread(target=_studio_worker, args=(transport, 2), daemon=True)
    worker.start()
    reset = transport.reset(seed=42)
    assert reset["info"]["course_seed"] == 42
    step = transport.step({"strafe": 0.0, "forward": 1.0, "yaw": 0.0, "jump": False})
    assert step["step_id"] == 0
    transport.close()
    worker.join(timeout=2)


def test_timeout_is_explicit() -> None:
    transport = StudioHTTPTransport(port=0, timeout=0.02)
    with pytest.raises(TransportTimeout, match="did not acknowledge"):
        transport.reset(seed=1)
    transport.close()


def test_curriculum_stage_is_sent_with_reset() -> None:
    transport = StudioHTTPTransport(port=0, timeout=2, curriculum_stage=2)
    worker = threading.Thread(target=_studio_worker, args=(transport, 1), daemon=True)
    worker.start()
    reset = transport.reset(seed=5)
    assert reset["info"]["curriculum_stage"] == 2
    transport.close()
    worker.join(timeout=2)


def test_invalid_curriculum_stage_is_rejected() -> None:
    with pytest.raises(ValueError, match="1..24"):
        StudioHTTPTransport(port=0, curriculum_stage=25)


def test_invalid_action_repeat_ticks_is_rejected() -> None:
    with pytest.raises(ValueError, match="action_repeat_ticks must be 1..6"):
        StudioHTTPTransport(port=0, action_repeat_ticks=0)


def test_vector_reset_and_step_round_trip() -> None:
    transport = StudioHTTPTransport(port=0, timeout=2)
    worker = threading.Thread(target=_vector_worker, args=(transport, 2), daemon=True)
    worker.start()
    reset = transport.vector_reset(seeds=list(range(8)))
    assert [result["info"]["lane_index"] for result in reset] == list(range(1, 9))
    action = {"strafe": 0.0, "forward": 1.0, "yaw": 0.0, "jump": False}
    step = transport.vector_step([action] * 8)
    assert len(step) == 8
    transport.close()
    worker.join(timeout=2)
