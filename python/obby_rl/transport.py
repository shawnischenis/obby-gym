from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Mapping, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class TransportTimeout(TimeoutError):
    pass


class _Broker:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.pending: dict[str, Any] | None = None
        self.responses: dict[str, dict[str, Any]] = {}
        self.closed = False

    def exchange(self, message: dict[str, Any]) -> dict[str, Any]:
        with self.condition:
            ack = message.get("ack_request_id")
            if isinstance(ack, str):
                self.responses[ack] = message
                if self.pending and self.pending["request_id"] == ack:
                    self.pending = None
                self.condition.notify_all()
            return self.pending or {
                "protocol_version": "0.1.0",
                "message_type": "noop",
                "request_id": "noop",
            }

    def request(self, command: dict[str, Any], timeout: float) -> dict[str, Any]:
        request_id = command["request_id"]
        with self.condition:
            if self.closed:
                raise RuntimeError("transport is closed")
            if self.pending is not None:
                raise RuntimeError("only one in-flight Studio command is supported")
            self.pending = command
            received = self.condition.wait_for(
                lambda: request_id in self.responses or self.closed, timeout=timeout
            )
            if not received or request_id not in self.responses:
                if self.pending and self.pending["request_id"] == request_id:
                    self.pending = None
                raise TransportTimeout(
                    f"Studio did not acknowledge {command['message_type']} {request_id} "
                    f"within {timeout:.1f}s"
                )
            return self.responses.pop(request_id)


class _Handler(BaseHTTPRequestHandler):
    server: _BrokerServer

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/exchange":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            message = json.loads(self.rfile.read(length))
            if not isinstance(message, dict):
                raise ValueError("request body must be an object")
            response = self.server.broker.exchange(message)
            body = json.dumps(response, separators=(",", ":")).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (ValueError, json.JSONDecodeError) as error:
            self.send_error(400, str(error))

    def log_message(self, format: str, *args: object) -> None:
        return


class _BrokerServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], broker: _Broker):
        self.broker = broker
        super().__init__(address, _Handler)


class StudioHTTPTransport:
    """Synchronous Python side of the Studio-polls-Python protocol."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        timeout: float = 5.0,
        curriculum_stage: int = 4,
        action_repeat_ticks: int = 3,
        recording_view: bool = False,
        recording_camera: str = "auto",
        recording_visible_lane: int = 0,
    ) -> None:
        if host not in {"127.0.0.1", "localhost", "0.0.0.0"}:
            raise ValueError("M1 transport must bind to a local interface")
        self.timeout = timeout
        if curriculum_stage not in set(range(1, 25)):
            raise ValueError("curriculum_stage must be 1..24")
        self.curriculum_stage = curriculum_stage
        if action_repeat_ticks not in range(1, 7):
            raise ValueError("action_repeat_ticks must be 1..6")
        self.action_repeat_ticks = int(action_repeat_ticks)
        self.recording_view = bool(recording_view)
        if recording_camera not in {
            "auto",
            "parallel",
            "side",
            "behind",
            "completion",
            "completion-side",
            "completion-follow",
        }:
            raise ValueError(
                "recording_camera must be auto, parallel, side, behind, completion, "
                "completion-side, or completion-follow"
            )
        self.recording_camera = recording_camera
        if recording_visible_lane < 0:
            raise ValueError("recording_visible_lane must be non-negative")
        self.recording_visible_lane = int(recording_visible_lane)
        self.broker = _Broker()
        self.server = _BrokerServer((host, port), self.broker)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.episode_id = ""
        self.step_id = 0

    @property
    def address(self) -> tuple[str, int]:
        address = self.server.server_address
        return str(address[0]), int(address[1])

    def _request(self, message_type: str, **payload: Any) -> dict[str, Any]:
        command = {
            "protocol_version": "0.1.0",
            "message_type": message_type,
            "request_id": str(uuid.uuid4()),
            "episode_id": self.episode_id,
            **payload,
        }
        response = self.broker.request(command, self.timeout)
        if response.get("protocol_version") != "0.1.0":
            raise ValueError("Studio protocol version mismatch")
        if response.get("episode_id") != self.episode_id:
            raise ValueError("Studio episode ID mismatch")
        return response

    def reset(self, *, seed: int) -> Mapping[str, Any]:
        self.episode_id = str(uuid.uuid4())
        self.step_id = 0
        return self._request(
            "reset_command",
            course_seed=int(seed),
            generator_version="0.7.0",
            curriculum_stage=self.curriculum_stage,
            action_repeat_ticks=self.action_repeat_ticks,
            recording_view=self.recording_view,
            recording_camera=self.recording_camera,
            recording_visible_lane=self.recording_visible_lane,
        )

    def step(self, action: Mapping[str, float | bool]) -> Mapping[str, Any]:
        response = self._request("action_command", step_id=self.step_id, action=dict(action))
        self.step_id += 1
        return response

    def vector_reset(
        self, *, seeds: list[int], post_landing_mask: Sequence[bool] | None = None
    ) -> list[Mapping[str, Any]]:
        if not seeds:
            raise ValueError("vector reset requires at least one seed")
        landing_mask = [False] * len(seeds) if post_landing_mask is None else list(post_landing_mask)
        if len(landing_mask) != len(seeds):
            raise ValueError("post-landing mask must match vector seed count")
        self.episode_id = str(uuid.uuid4())
        self.step_id = 0
        response = self._request(
            "vector_reset_command",
            course_seeds=[int(seed) for seed in seeds],
            post_landing_mask=[bool(value) for value in landing_mask],
            generator_version="0.7.0",
            curriculum_stage=self.curriculum_stage,
            action_repeat_ticks=self.action_repeat_ticks,
            recording_view=self.recording_view,
            recording_camera=self.recording_camera,
            recording_visible_lane=self.recording_visible_lane,
        )
        results = response.get("results")
        if not isinstance(results, list) or len(results) != len(seeds):
            raise ValueError(f"expected {len(seeds)} vector reset results")
        return results

    def vector_step(self, actions: Sequence[Mapping[str, float | bool]]) -> list[Mapping[str, Any]]:
        if not actions:
            raise ValueError("vector step requires at least one action")
        response = self._request(
            "vector_action_command",
            step_id=self.step_id,
            actions=[dict(action) for action in actions],
        )
        self.step_id += 1
        results = response.get("results")
        if not isinstance(results, list) or len(results) != len(actions):
            raise ValueError(f"expected {len(actions)} vector step results")
        return results

    def vector_reset_lanes(
        self,
        *,
        seeds: Sequence[int],
        reset_mask: Sequence[bool],
        post_landing_mask: Sequence[bool] | None = None,
    ) -> list[Mapping[str, Any]]:
        if len(seeds) != len(reset_mask) or not seeds:
            raise ValueError("vector lane seeds and reset mask must have equal non-zero length")
        landing_mask = [False] * len(seeds) if post_landing_mask is None else list(post_landing_mask)
        if len(landing_mask) != len(seeds):
            raise ValueError("post-landing mask must match vector lane seed count")
        response = self._request(
            "vector_reset_lanes_command",
            step_id=self.step_id,
            course_seeds=[int(seed) for seed in seeds],
            reset_mask=[bool(value) for value in reset_mask],
            post_landing_mask=[bool(value) for value in landing_mask],
        )
        results = response.get("results")
        if not isinstance(results, list) or len(results) != len(seeds):
            raise ValueError(f"expected {len(seeds)} vector lane reset results")
        return results

    def close(self) -> None:
        with self.broker.condition:
            self.broker.closed = True
            self.broker.condition.notify_all()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
