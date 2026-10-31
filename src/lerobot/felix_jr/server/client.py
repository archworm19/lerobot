"""Client for the robot server, used from the experiment computer.

  from lerobot.felix_jr.server.client import FelixClient

  robot = FelixClient("tcp://raspberrypi.local:5555")
  print(robot.hello())
  snap = robot.capture()                       # a Capture: .color, .raw_depth, .calibration, .state
  scene = snap.scene()                         # a felix_jr.harness.scene.Scene, ready for pixel <-> xyz queries

Every method blocks until the reply arrives and raises `TimeoutError` if the server stays silent, or
`ServerError` if the server reports a failure.

Moving the robot: wrap the script in a session, so the robot goes to rest when the script ends (normally, with
Ctrl+C, or if the script dies and the heartbeats stop):

  with robot.session():                        # on_disconnect="home" by default
      robot.move_joints({"left_arm": {"shoulder_pan.pos": 10.0}})     # blocks until done
      status = robot.home()
      assert status["state"] == "done"         # "stopped" if something called stop() or the watchdog fired

Without a session the robot stays wherever the last motion left it.
"""

import contextlib
import itertools
import logging
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import zmq

from lerobot.felix_jr.server.protocol import (
    DEFAULT_PORT,
    PROTOCOL_VERSION,
    MotionBusyError,
    MotionFailedError,
    ServerError,
    decode_image,
    pack,
    unpack,
)

log = logging.getLogger("felix_client")


class _Heartbeat:
    """Sends a `heartbeat` request every `period_s` on its own socket.

    That way the server's watchdog hears from a client that is busy (or waiting on a long reply) in another
    thread.
    """

    def __init__(self, context: zmq.Context, endpoint: str, period_s: float):
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, args=(context, endpoint, period_s), daemon=True, name="felix-heartbeat"
        )
        self._thread.start()

    def _run(self, context: zmq.Context, endpoint: str, period_s: float) -> None:
        socket = context.socket(zmq.DEALER)  # sockets are not thread-safe: this one belongs to this thread
        socket.connect(endpoint)
        try:
            while not self._stop.wait(period_s):
                socket.send_multipart(pack({"id": 0, "cmd": "heartbeat", "args": {}}))
                while socket.poll(0):
                    socket.recv_multipart()  # drop the replies
        finally:
            socket.close(linger=0)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


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
        from lerobot.felix_jr.harness.scene import Scene

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
                if header["error"].startswith("MotionBusyError"):
                    raise MotionBusyError(header["error"])
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

    def head_wiggle(self, name: str, degrees: float = 8.0) -> dict[str, Any]:
        """Nudge head servo `name` `degrees` each way and back to where it started.

        Returns `to_pos` / `to_neg` (how far it went in each direction, in degrees), `limited` (a recorded limit
        shortened a move) and `blocked` (the servo stopped well short of a goal).
        """
        return self._call("head_wiggle", name=name, degrees=degrees)[0]

    def head_record(self, name: str, what: str) -> dict[str, Any]:
        """Save head servo `name`'s present position as `what` (`limit_pos`, `limit_neg`, `home`)."""
        return self._call("head_record", name=name, what=what)[0]

    def head_torque_limit(self, name: str, limit: int) -> dict[str, Any]:
        """Cap head servo `name`'s torque, 1-1000 (1000 = no cap), until it loses power."""
        return self._call("head_torque_limit", name=name, limit=limit)[0]

    # ---- motion ---------------------------------------------------------------------------------------------

    def motion_status(self) -> dict[str, Any]:
        """The running or most recent motion: `{moving, state, motion_id, kind, reason, result}`.

        `state` is idle, running, done, stopped or failed.
        """
        return self._call("motion_status")[0]

    def wait_motion(self, timeout_s: float = 120.0, poll_s: float = 0.1) -> dict[str, Any]:
        """Wait until no motion is running and return its final status.

        Raises `MotionFailedError` if it failed, and `TimeoutError` if it is still running after `timeout_s`
        (it is not stopped; call `stop()` if you want that). Polling also tells the watchdog you are alive.
        """
        deadline = time.monotonic() + timeout_s
        while True:
            status = self.motion_status()
            if not status["moving"]:
                if status["state"] == "failed":
                    raise MotionFailedError(f"{status['kind']} failed: {status['reason']}", status)
                return status
            if time.monotonic() > deadline:
                raise TimeoutError(f"{status['kind']} still running after {timeout_s:.0f} s")
            time.sleep(poll_s)

    def move_joints(
        self,
        targets: dict[str, dict[str, float]],
        max_vel: float | None = None,
        max_accel: float | None = None,
        tol: float = 1.5,
        timeout: float = 10.0,
        wait: bool = True,
    ) -> dict[str, Any]:
        """Move the arms to `targets`, all together: `{"left_arm": {"shoulder_pan.pos": deg, ...}, ...}`.

        This is one target pose, reached from wherever the arms are now; for a path or a replan, call it again
        with the next target (each call ends at rest). Joints are in degrees (the gripper 0-100); joints not
        listed stay put. `max_vel` / `max_accel` default
        to 60 deg/s and 120 deg/s^2; `tol` and `timeout` control the settling at the end. With `wait` (the
        default) it returns the final status (see `wait_motion`), else `{motion_id}` right away.
        """
        started = self._call(
            "move_joints", targets=targets, max_vel=max_vel, max_accel=max_accel, tol=tol, timeout=timeout
        )[0]
        return self.wait_motion() if wait else started

    def home(
        self,
        max_vel: float | None = None,
        max_accel: float | None = None,
        wait: bool = True,
        arms: list[str] | None = None,
    ) -> dict[str, Any]:
        """Go to rest: the arms to their recorded home poses (default: all), then the head to its home."""
        started = self._call("home", max_vel=max_vel, max_accel=max_accel, arms=arms)[0]
        return self.wait_motion() if wait else started

    def base_velocity(
        self, left_degps: float, right_degps: float, seconds: float, wait: bool = True
    ) -> dict[str, Any]:
        """Drive the wheels (deg/s, positive = forward) for `seconds` (at most 5), then stop them.

        The final status's `result["measured"]` holds the measured (left, right) speeds.
        """
        started = self._call(
            "base_velocity", left_degps=left_degps, right_degps=right_degps, seconds=seconds
        )[0]
        return self.wait_motion() if wait else started

    def stop(self) -> dict[str, Any]:
        """Abort the running motion. The arms hold where they are. Returns `{was_moving}`."""
        return self._call("stop")[0]

    def get_homes(self) -> dict[str, dict[str, float]]:
        """The recorded home pose of every arm (they are recorded on the robot, with `setup.record_home`)."""
        return self._call("get_homes")[0]["homes"]

    @contextlib.contextmanager
    def session(self, on_disconnect: str = "home", heartbeat_s: float = 1.0) -> Iterator["FelixClient"]:
        """Tell the server this client is in charge, and what to do when it ends or goes silent.

        With `on_disconnect="home"` (default) the robot goes to rest when the block ends, however it ends
        (normally, an exception, Ctrl+C): the motion is stopped and the arms go home. A background heartbeat
        keeps the server informed meanwhile; if the script dies or the network drops, the heartbeats stop and
        the server goes to rest by itself after a few seconds. "hold" leaves the robot where it is.

        If going to rest is itself interrupted, the session is left open on purpose, so the server finishes
        the job when the heartbeats stop.
        """
        self._call("session_open", on_disconnect=on_disconnect)
        heartbeat = _Heartbeat(self._context, self.endpoint, heartbeat_s)
        try:
            yield self
        finally:
            finished = False
            try:
                if on_disconnect == "home":
                    self._go_to_rest()
                finished = True
            except Exception:
                log.exception("could not go to rest from the client; the server will try when it loses us")
            finally:
                heartbeat.stop()
                if finished:
                    with contextlib.suppress(Exception):
                        self._call("session_close", timeout_s=2.0)

    def _go_to_rest(self) -> None:
        """Stop whatever is moving, wait for it to end, then go home."""
        self.stop()
        with contextlib.suppress(MotionFailedError):
            self.wait_motion(timeout_s=10.0)
        self.home()
