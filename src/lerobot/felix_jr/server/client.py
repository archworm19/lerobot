"""Client for the robot server, used from the experiment computer.

  from lerobot.felix_jr.server.client import FelixClient

  robot = FelixClient("tcp://raspberrypi.local:5555")
  print(robot.hello())
  snap = robot.capture()                       # a Capture: .color, .raw_depth, .calibration, .state
  scene = snap.scene()                         # a felix_jr.scene.Scene, ready for pixel <-> xyz queries

Every method blocks until the reply arrives and raises `TimeoutError` if the server stays silent, or
`ServerError` if the server reports a failure.
"""

import itertools
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import zmq

from lerobot.felix_jr.server.protocol import (
    DEFAULT_PORT,
    PROTOCOL_VERSION,
    ServerError,
    decode_image,
    pack,
    unpack,
)

log = logging.getLogger("felix_client")


@dataclass
class Capture:
    """A snapshot from `FelixClient.capture`: same fields as the server's `Snapshot`."""

    color: np.ndarray  # RGB uint8 (H, W, 3)
    raw_depth: np.ndarray  # uint16 (H, W), raw units, not aligned to color
    calibration: dict[str, Any]
    state: dict[str, Any]
    timestamp: float  # server clock

    def scene(self):
        """The capture as a `Scene` (depth aligned to color, in meters)."""
        from lerobot.felix_jr.depth import with_matrices
        from lerobot.felix_jr.scene import Scene

        return Scene.from_capture(self.color, self.raw_depth, with_matrices(self.calibration))


class FelixClient:
    """A blocking connection to one robot server. Not thread-safe: use one client per thread."""

    def __init__(self, endpoint: str = f"tcp://localhost:{DEFAULT_PORT}", timeout_s: float = 10.0):
        """Connect to the server at `endpoint`.

        `timeout_s` is the default wait for a reply. The connection is checked with `hello`, and a warning is
        logged if the server's protocol differs from this client's.
        """
        self.timeout_s = timeout_s
        self._ids = itertools.count(1)
        self._context = zmq.Context()
        self._socket = self._make_socket(endpoint)
        self.endpoint = endpoint
        try:
            self.server_info = self.hello()
        except Exception:
            self.close()
            raise
        if self.server_info["protocol"] != PROTOCOL_VERSION:
            log.warning(
                "protocol mismatch: server %s, client %s", self.server_info["protocol"], PROTOCOL_VERSION
            )

    def _make_socket(self, endpoint: str):
        socket = self._context.socket(zmq.DEALER)
        socket.connect(endpoint)
        return socket

    def close(self) -> None:
        """Close the connection."""
        self._socket.close(linger=0)
        self._context.term()

    def __enter__(self) -> "FelixClient":
        """Use as a context manager: the connection is closed on exit."""
        return self

    def __exit__(self, *exc) -> None:
        """Close the connection."""
        self.close()

    def _call(
        self, cmd: str, timeout_s: float | None = None, **args: Any
    ) -> tuple[dict[str, Any], list[bytes], list[dict]]:
        req_id = next(self._ids)
        self._socket.send_multipart(pack({"id": req_id, "cmd": cmd, "args": args}))
        timeout_ms = int(1000 * (self.timeout_s if timeout_s is None else timeout_s))
        while True:
            if not self._socket.poll(timeout_ms):
                # A late reply to this request would be mistaken for the next one's; start a fresh socket.
                endpoint = self.endpoint
                self._socket.close(linger=0)
                self._socket = self._make_socket(endpoint)
                raise TimeoutError(f"no reply to {cmd!r} from {endpoint} within {timeout_ms / 1000:.0f} s")
            header, frames = unpack(self._socket.recv_multipart())
            if header.get("id") != req_id:
                continue  # a stale reply from an earlier request
            if not header["ok"]:
                raise ServerError(header["error"])
            return header["result"], frames, header.get("frames", [])

    def hello(self) -> dict[str, Any]:
        """Server info: `protocol`, `commit`, and which of `arms`, `head`, `base`, `camera` exist."""
        return self._call("hello")[0]

    def get_state(self) -> dict[str, Any]:
        """Joint positions: `{"left_arm": {"shoulder_pan.pos": deg, ...}, ..., "moving": bool}`."""
        return self._call("get_state")[0]["state"]

    def capture(self, color_format: str = "png", after: float | None = None) -> Capture:
        """The latest color + depth frames with the camera calibration and joint state.

        `color_format` "jpeg" trades some quality for a smaller message. With `after` (the `done_at` of a head
        move), the server waits for a frame taken after that time, so the image shows the result of the move.
        """
        result, frames, meta = self._call("capture", color_format=color_format, after=after)
        images = {m["name"]: decode_image(data) for m, data in zip(meta, frames, strict=True)}
        return Capture(
            color=images["color"],
            raw_depth=images["raw_depth"],
            calibration=result["calibration"],
            state=result["state"],
            timestamp=result["timestamp"],
        )

    def head_state(self) -> dict[str, dict[str, Any]]:
        """Per head servo: `ticks` now, `sign`, and the recorded `limit_pos`, `limit_neg`, `home`."""
        return self.get_state()["head"]

    def head_move(self, name: str, delta_deg: float | None = None, to: str | None = None) -> dict[str, Any]:
        """Move head servo `name` (`head_eyes` tilts, `head_neck` turns).

        Give `delta_deg` (positive = down / left) or `to="home"`. Blocks until it settles. Returns
        `{ticks, goal, clamped, reached, done_at}`.
        """
        return self._call("head_move", name=name, delta_deg=delta_deg, to=to)[0]

    def head_record(self, name: str, what: str) -> dict[str, Any]:
        """Save head servo `name`'s present position as `what` (`limit_pos`, `limit_neg`, `home`)."""
        return self._call("head_record", name=name, what=what)[0]

    def head_torque_limit(self, name: str, limit: int) -> dict[str, Any]:
        """Cap head servo `name`'s torque, 1-1000 (1000 = no cap), until it loses power."""
        return self._call("head_torque_limit", name=name, limit=limit)[0]
