"""The robot server, run on the Raspberry Pi: answers requests from the experiment computer, one reply per request.

  uv run python -m lerobot.felix_jr.server.server --fake                       # no robot, for development
  uv run python -m lerobot.felix_jr.server.server --left-port P --right-port P  # real camera and arms
  uv run python -m lerobot.felix_jr.server.server --no-camera --left-port P     # arms only
  uv run python -m lerobot.felix_jr.server.server                               # camera only

Commands (see `protocol.py` for the message format):

- `hello`: protocol version, git commit and which parts of the robot exist.
- `get_state`: joint positions of every arm.
- `capture` (`color_format`: "png" | "jpeg"): the latest color + raw depth frames, the camera calibration and the
  joint state at that moment.

Only the server's own thread touches the hardware. When motion commands arrive, they will run on a worker thread
so that `stop` and `get_state` can still be answered; the ROUTER socket already allows that.
"""

import argparse
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

import zmq

from lerobot.felix_jr.server.hardware import DEFAULT_LIBREALSENSE_BUILD, FakeHardware, Hardware
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

    def cmd_capture(self, color_format: str = "png") -> tuple[dict[str, Any], list[dict]]:
        """The latest color and raw depth images (binary frames), calibration and joint state."""
        snap = self.hardware.capture()
        frames = [
            {"name": "color", "format": color_format, "data": encode_image(snap.color, color_format)},
            {"name": "raw_depth", "format": "png", "data": encode_image(snap.raw_depth, "png")},
        ]
        result = {"timestamp": snap.timestamp, "calibration": snap.calibration, "state": snap.state}
        return result, frames


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
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.fake:
        hardware: Hardware = FakeHardware()
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
        )
    server = RobotServer(hardware)
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
