"""The robot server, run on the Raspberry Pi: answers requests from the experiment computer, one reply per request.

  uv run python -m lerobot.felix_jr.server.server --fake                       # no robot, for development
  uv run python -m lerobot.felix_jr.server.server --left-port P --right-port P  # real camera and arms
  uv run python -m lerobot.felix_jr.server.server --no-camera --left-port P     # arms only
  uv run python -m lerobot.felix_jr.server.server                               # camera only

Commands (see `protocol.py` for the message format):

- `hello`: protocol version, git commit and which parts of the robot exist.
- `get_state`: joint positions of every arm, and the head servos' ticks with their recorded range.
- `capture` (`color_format`: "png" | "jpeg", `after`: server time): the latest color + raw depth frames, the
  camera calibration and the joint state at that moment. With `after` (e.g. a head move's `done_at`) it waits for
  a frame taken after that time.
- `head_move` (`name`, and `delta_deg` or `to`): move a head servo (positive = down / left), clamped to its
  recorded limits; returns when it has settled, with `done_at` for `capture(after=...)`.
- `head_wiggle` (`name`, `degrees`): nudge a head servo both ways and back; returns how far it went each way.
- `head_record` (`name`, `what`): save the servo's present position as `limit_pos`, `limit_neg` or `home`.
- `head_torque_limit` (`name`, `limit`): cap a head servo's torque (1-1000) until it loses power.
- `move_joints` (`targets`: `{arm id: {"shoulder_pan.pos": deg, ...}}`, optional `max_vel`, `max_accel`, `tol`,
  `timeout`): start moving the arms, all together; replies at once with a `motion_id`.
- `home` (optional `arms`): start going to rest (the arms' recorded home poses, then the head's home).
- `base_velocity` (`left_degps`, `right_degps`, `seconds`): drive the wheels for a bounded time, then stop.
- `motion_status`: `{moving, state, motion_id, kind, reason, result}` of the running or latest motion.
- `stop`: abort the running motion; the arms hold where they are.
- `get_homes`: the recorded home pose of every arm.
- `heartbeat`, `session_open` (`on_disconnect`: "home" | "hold"), `session_close`: see `motion.py`.

Motions run on a worker thread (`motion.MotionController`), so `stop`, `motion_status`, `get_state` and `capture`
are answered while the arms move. Only one motion runs at a time (others get `MotionBusyError`). The server goes to
rest on exit (Ctrl+C or SIGTERM) before the servos are released, unless started with `--no-home-on-exit`.
"""

import argparse
import logging
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import zmq

from lerobot.felix_jr.server.hardware import DEFAULT_LIBREALSENSE_BUILD, FakeHardware, Hardware
from lerobot.felix_jr.server.motion import DEFAULT_SESSION_TIMEOUT_S, DEFAULT_WATCHDOG_S
from lerobot.felix_jr.server.protocol import DEFAULT_PORT, PROTOCOL_VERSION, encode_image, pack, unpack

log = logging.getLogger("felix_server")


def git_commit() -> str:
    """Short commit hash of the code this server runs, with `+dirty` if there are uncommitted changes."""
    repo = Path(__file__).resolve().parent
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=repo, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"
    return commit + ("+dirty" if dirty else "")


class RobotServer:
    """Serves one `Hardware` over a ZeroMQ ROUTER socket."""

    def __init__(self, hardware: Hardware):
        """Create the (unbound) server for `hardware`."""
        self.hardware = hardware
        self.commit = git_commit()
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.ROUTER)
        self._stopping = False

    def bind(self, endpoint: str) -> str:
        """Bind to `endpoint` (e.g. `tcp://*:5555`; a trailing `:*` picks a free port). Returns the endpoint."""
        self._socket.bind(endpoint)
        return self._socket.getsockopt_string(zmq.LAST_ENDPOINT)

    def serve_forever(self) -> None:
        """Answer requests until `shutdown` is called."""
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        while not self._stopping:
            if not dict(poller.poll(100)):
                continue
            identity, *parts = self._socket.recv_multipart()
            try:
                request, _ = unpack(parts)
            except Exception as e:  # not our protocol; there is no id to echo
                log.warning("dropping malformed message: %r", e)
                continue
            header, frames = self.handle(request)
            self._socket.send_multipart([identity, *pack(header, frames)])

    def shutdown(self) -> None:
        """Ask `serve_forever` to return (from another thread, or a signal handler)."""
        self._stopping = True

    def close(self) -> None:
        """Close the socket and the hardware."""
        self._socket.close(linger=0)
        self._context.term()
        self.hardware.close()

    def handle(self, request: dict[str, Any]) -> tuple[dict[str, Any], list[bytes]]:
        """Run one request and return the reply header and binary frames.

        Errors become `ok: false` replies, so a bad request never takes the server down.
        """
        req_id, cmd, args = request.get("id"), request.get("cmd"), request.get("args") or {}
        started = time.perf_counter()
        self.hardware.motion.touch()  # any request tells the watchdog the client is alive
        try:
            handler = getattr(self, f"cmd_{cmd}", None)
            if not isinstance(cmd, str) or handler is None:
                raise ValueError(f"unknown command {cmd!r}")
            result, frames = handler(**args)
            reply = {"id": req_id, "ok": True, "result": result}
            if frames:
                reply["frames"] = [{k: v for k, v in f.items() if k != "data"} for f in frames]
            log.info("%s ok in %.0f ms", cmd, 1000 * (time.perf_counter() - started))
            return reply, [f["data"] for f in frames]
        except Exception as e:
            log.exception("%s failed", cmd)
            return {"id": req_id, "ok": False, "error": f"{type(e).__name__}: {e}"}, []

    # ---- commands: each returns (result dict, list of binary frames) -------------------------------------

    def cmd_hello(self) -> tuple[dict[str, Any], list[dict]]:
        """Protocol version, git commit and what hardware exists."""
        return {"protocol": PROTOCOL_VERSION, "commit": self.commit, **self.hardware.info()}, []

    def cmd_get_state(self) -> tuple[dict[str, Any], list[dict]]:
        """Joint state of every arm."""
        return {"state": self.hardware.state(), "timestamp": time.time()}, []

    def cmd_capture(
        self, color_format: str = "png", after: float | None = None
    ) -> tuple[dict[str, Any], list[dict]]:
        """The latest color and raw depth images (binary frames), calibration and joint state."""
        snap = self.hardware.capture(after=after)
        frames = [
            {"name": "color", "format": color_format, "data": encode_image(snap.color, color_format)},
            {"name": "raw_depth", "format": "png", "data": encode_image(snap.raw_depth, "png")},
        ]
        result = {"timestamp": snap.timestamp, "calibration": snap.calibration, "state": snap.state}
        return result, frames

    def _head(self):
        if self.hardware.head is None:
            raise RuntimeError("this robot has no head servos")
        return self.hardware.head

    def cmd_head_move(
        self, name: str, delta_deg: float | None = None, to: str | None = None
    ) -> tuple[dict[str, Any], list[dict]]:
        """Move head servo `name` by `delta_deg` or to a recorded position `to`; see `HeadController.move`."""
        with self.hardware.motion.exclusive():
            return self._head().move(name, delta_deg=delta_deg, to=to), []

    def cmd_head_wiggle(self, name: str, degrees: float = 8.0) -> tuple[dict[str, Any], list[dict]]:
        """Nudge head servo `name` both ways and back; see `HeadController.wiggle`."""
        with self.hardware.motion.exclusive():
            return self._head().wiggle(name, float(degrees)), []

    def cmd_head_record(self, name: str, what: str) -> tuple[dict[str, Any], list[dict]]:
        """Record head servo `name`'s position as `what`; see `HeadController.record`."""
        with self.hardware.motion.exclusive():
            return self._head().record(name, what), []

    def cmd_head_torque_limit(self, name: str, limit: int) -> tuple[dict[str, Any], list[dict]]:
        """Cap head servo `name`'s torque; see `HeadController.torque_limit`."""
        with self.hardware.motion.exclusive():
            return self._head().torque_limit(name, limit), []

    def cmd_move_joints(
        self,
        targets: dict[str, dict[str, float]],
        max_vel: float | None = None,
        max_accel: float | None = None,
        tol: float = 1.5,
        timeout: float = 10.0,
    ) -> tuple[dict[str, Any], list[dict]]:
        """Start a synchronized arm motion; see `MotionController.move_joints`."""
        return self.hardware.motion.move_joints(targets, max_vel, max_accel, tol, timeout), []

    def cmd_home(
        self,
        max_vel: float | None = None,
        max_accel: float | None = None,
        arms: list[str] | None = None,
    ) -> tuple[dict[str, Any], list[dict]]:
        """Start going to rest; see `MotionController.home`."""
        return self.hardware.motion.home(max_vel, max_accel, arms=arms), []

    def cmd_base_velocity(
        self, left_degps: float, right_degps: float, seconds: float
    ) -> tuple[dict[str, Any], list[dict]]:
        """Drive the wheels for a bounded time; see `MotionController.base_velocity`."""
        return self.hardware.motion.base_velocity(left_degps, right_degps, seconds), []

    def cmd_motion_status(self) -> tuple[dict[str, Any], list[dict]]:
        """Status of the running or most recent motion."""
        return self.hardware.motion.status(), []

    def cmd_stop(self) -> tuple[dict[str, Any], list[dict]]:
        """Abort the running motion; the arms hold where they are."""
        return self.hardware.motion.stop(), []

    def cmd_get_homes(self) -> tuple[dict[str, Any], list[dict]]:
        """The recorded home pose of every arm."""
        return {"homes": self.hardware.motion.arms.homes()}, []

    def cmd_heartbeat(self) -> tuple[dict[str, Any], list[dict]]:
        """A sign of life from the client (every request counts; this one does nothing else)."""
        return {}, []

    def cmd_session_open(self, on_disconnect: str = "hold") -> tuple[dict[str, Any], list[dict]]:
        """Open a session; with `on_disconnect="home"` the server goes to rest if the client falls silent."""
        return self.hardware.motion.open_session(on_disconnect), []

    def cmd_session_close(self) -> tuple[dict[str, Any], list[dict]]:
        """End the session without going anywhere."""
        return self.hardware.motion.close_session(), []


def main() -> None:
    """Command line entry point; see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--bind", default=f"tcp://*:{DEFAULT_PORT}", help=f"endpoint to listen on (tcp://*:{DEFAULT_PORT})"
    )
    parser.add_argument("--fake", action="store_true", help="synthetic camera and arms; touches no hardware")
    parser.add_argument("--left-port", help="serial port of the left arm (id left_arm)")
    parser.add_argument("--right-port", help="serial port of the right arm (id right_arm)")
    parser.add_argument("--no-camera", action="store_true", help="do not open the RealSense")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--librealsense-build",
        default=DEFAULT_LIBREALSENSE_BUILD,
        help="build folder of the source-built librealsense that holds pyrealsense2 (default: %(default)s)",
    )
    parser.add_argument(
        "--no-home-on-exit", action="store_true", help="do not send the arms to rest when the server exits"
    )
    parser.add_argument(
        "--watchdog",
        type=float,
        default=DEFAULT_WATCHDOG_S,
        help="stop a motion after this many seconds of silence (%(default)s)",
    )
    parser.add_argument(
        "--session-timeout",
        type=float,
        default=DEFAULT_SESSION_TIMEOUT_S,
        help="a session that asked for it goes to rest after this many seconds of silence (%(default)s)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # lerobot's max_relative_target clamp logs a warning on every tick of a trajectory; it is expected here.
    logging.getLogger("lerobot.robots.utils").setLevel(logging.ERROR)
    motion_kwargs = {"watchdog_s": args.watchdog, "session_timeout_s": args.session_timeout}

    if args.fake:
        hardware: Hardware = FakeHardware(**motion_kwargs)
    else:
        from lerobot.felix_jr.server.hardware import RealHardware

        arms = {
            robot_id: port
            for robot_id, port in (("left_arm", args.left_port), ("right_arm", args.right_port))
            if port
        }
        hardware = RealHardware(
            arms=arms,
            camera=not args.no_camera,
            width=args.width,
            height=args.height,
            fps=args.fps,
            librealsense_build=args.librealsense_build,
            home_on_exit=not args.no_home_on_exit,
            **motion_kwargs,
        )
    server = RobotServer(hardware)
    signal.signal(signal.SIGTERM, lambda *_: server.shutdown())  # systemd stop: same clean exit as Ctrl+C
    log.info(
        "listening on %s (commit %s, protocol %d)", server.bind(args.bind), server.commit, PROTOCOL_VERSION
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")
    finally:
        server.close()


if __name__ == "__main__":
    main()
