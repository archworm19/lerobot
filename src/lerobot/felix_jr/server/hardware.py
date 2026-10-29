"""What the server can ask of the robot: `Hardware`, a fake one for development, and the real one for the Pi.

`Hardware` is the only thing `server.py` talks to, so the server and client can be developed and tested on a
computer with no robot (`FakeHardware`). `RealHardware` drives the RealSense through `pyrealsense2` directly,
rather than through LeRobot's `RealSenseCamera`: a pyrealsense2 built from source has no pip metadata, so
LeRobot's availability check reports it as not installed and the camera class refuses to open.
"""

import contextlib
import glob
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from lerobot.felix_jr.server.head import HeadController, RangeStore

DEFAULT_LIBREALSENSE_BUILD = "/home/zcecere/CodeRepository/librealsense/build"
# A frame is only trusted to show the scene after a move once it arrived this long (seconds) after the move
# finished: the camera's exposure and the USB transfer both lag the frame's arrival time.
FRAME_LATENCY_S = 0.1


@dataclass
class Snapshot:
    """One synchronized capture.

    `color`: RGB uint8 (H, W, 3). `raw_depth`: uint16 (H, W), raw sensor units, NOT aligned to the color image.
    `calibration`: the `calibration.json` dict (see `diagnostics/realsense_check.py`) needed to align and
    deproject the depth. `state`: the joint state read at the same moment. `timestamp`: server clock, seconds
    since the epoch, when the frames were received.
    """

    color: np.ndarray
    raw_depth: np.ndarray
    calibration: dict[str, Any]
    state: dict[str, Any]
    timestamp: float


class Hardware(Protocol):
    """The interface `RobotServer` uses; implemented by `FakeHardware` and `RealHardware`."""

    def info(self) -> dict[str, Any]:
        """Which parts exist, e.g. `{"arms": ["left_arm"], "head": True, "base": True, "camera": True}`."""

    def state(self) -> dict[str, Any]:
        """Joint positions per arm (`{"left_arm": {"shoulder_pan.pos": deg, ...}}`) and `"moving": bool`."""

    def capture(self, after: float | None = None) -> Snapshot:
        """The latest camera frames together with the current joint state.

        With `after` (a server `time.time()` value, e.g. a move's `done_at`), waits for a frame that arrived
        more than `FRAME_LATENCY_S` later, so the image shows the scene after the move.
        """

    def close(self) -> None:
        """Release the camera and the serial ports."""

    # `head`: a `HeadController`, or None if the robot has no head servos. Set by the implementations.
    head: HeadController | None


class FakeHeadBackend:
    """Head servos that reach any goal instantly. They start at raw 2048 (mid-range)."""

    def __init__(self, names: tuple[str, ...] = ("head_neck", "head_eyes")):
        """Simulated servos called `names`."""
        self._ticks = dict.fromkeys(names, 2048)
        self.torque_limits = dict.fromkeys(names, 1000)

    def names(self) -> list[str]:
        """See `HeadBackend.names`."""
        return list(self._ticks)

    def read_ticks(self, name: str) -> int:
        """See `HeadBackend.read_ticks`."""
        return self._ticks[name]

    def write_goal(self, name: str, ticks: int) -> None:
        """See `HeadBackend.write_goal`."""
        self._ticks[name] = int(ticks)

    def write_torque_limit(self, name: str, limit: int) -> None:
        """See `HeadBackend.write_torque_limit`."""
        self.torque_limits[name] = int(limit)


class BusHeadBackend:
    """The head servos of a follower with `with_head=True`, in raw ticks through its motor bus."""

    def __init__(self, robot):
        """`robot`: a connected `SO101FollowerWithBase` with head servos."""
        self.robot = robot

    def names(self) -> list[str]:
        """See `HeadBackend.names`."""
        return list(self.robot.head_motors)

    def read_ticks(self, name: str) -> int:
        """See `HeadBackend.read_ticks`."""
        return int(self.robot.bus.read("Present_Position", name, normalize=False))

    def write_goal(self, name: str, ticks: int) -> None:
        """See `HeadBackend.write_goal`."""
        self.robot.bus.write("Goal_Position", name, int(ticks), normalize=False)

    def write_torque_limit(self, name: str, limit: int) -> None:
        """See `HeadBackend.write_torque_limit`."""
        self.robot.bus.write("Torque_Limit", name, int(limit), normalize=False)


class FakeHardware:
    """A robot with a synthetic scene: a flat wall 1 m away with a box 0.6 m away, and arms at rest.

    The eyes servo tilts the picture: each tick away from the middle shifts the image by `PIXELS_PER_TICK`
    rows, so a head move visibly changes what `capture` returns.
    """

    WIDTH, HEIGHT, FX = 320, 240, 200.0
    PIXELS_PER_TICK = 0.1

    def __init__(
        self, arms: tuple[str, ...] = ("left_arm", "right_arm"), head_range_path: Path | None = None
    ):
        """A fake robot with arms named `arms`.

        `head_range_path` is where the head's recorded range is kept (default: not saved).
        """
        self.arms = arms
        self.captures = 0
        self.head: HeadController | None = HeadController(
            FakeHeadBackend(), {"head_neck": -1, "head_eyes": 1}, RangeStore(head_range_path)
        )

    def info(self) -> dict[str, Any]:
        """See `Hardware.info`."""
        return {"arms": list(self.arms), "head": True, "base": True, "camera": True, "fake": True}

    def state(self) -> dict[str, Any]:
        """See `Hardware.state`."""
        joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        arm = {f"{j}.pos": 0.0 for j in joints}
        arm["gripper.pos"] = 50.0
        out = {**{name: dict(arm) for name in self.arms}, "moving": False}
        if self.head is not None:
            out["head"] = self.head.state()
        return out

    def capture(self, after: float | None = None) -> Snapshot:
        """See `Hardware.capture` (a fake frame is always fresh, so `after` changes nothing)."""
        self.captures += 1
        h, w = self.HEIGHT, self.WIDTH
        color = np.zeros((h, w, 3), dtype=np.uint8)
        color[..., 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
        color[..., 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
        depth = np.full((h, w), 1000, dtype=np.uint16)  # 1 raw unit = 1 mm
        depth[80:160, 120:200] = 600
        depth[20:40, 20:60] = 0  # a hole with no depth
        if self.head is not None:
            shift = round((self.head.backend.read_ticks("head_eyes") - 2048) * self.PIXELS_PER_TICK)
            color, depth = np.roll(color, shift, axis=0), np.roll(depth, shift, axis=0)
        intr = {"width": w, "height": h, "fx": self.FX, "fy": self.FX, "ppx": w / 2, "ppy": h / 2}
        calibration = {
            "depth_scale_m_per_unit": 0.001,
            "color_intrinsics": dict(intr),
            "depth_intrinsics": dict(intr),
            "depth_to_color_rotation_col_major": [1, 0, 0, 0, 1, 0, 0, 0, 1],
            "depth_to_color_translation_m": [0.0, 0.0, 0.0],
        }
        return Snapshot(color, depth, calibration, self.state(), time.time())

    def close(self) -> None:
        """Nothing to release."""


def add_librealsense_to_path(build_dir: str | Path) -> Path:
    """Make a source-built `pyrealsense2` importable.

    Puts the folder under `build_dir` that holds the `pyrealsense2*.so` module on `sys.path` and returns it.
    Must run before `import pyrealsense2`.
    """
    build_dir = Path(build_dir).expanduser()
    matches = sorted(glob.glob(str(build_dir / "**" / "pyrealsense2*.so*"), recursive=True))
    if not matches:
        raise FileNotFoundError(
            f"no pyrealsense2*.so under {build_dir}; is librealsense built with -DBUILD_PYTHON_BINDINGS=ON?"
        )
    folder = Path(matches[0]).parent
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    return folder


class _LatestFrames:
    """Polls a running RealSense pipeline on a thread and keeps only the newest color + raw depth."""

    def __init__(self, pipeline, timeout_ms: int = 2000):
        self._pipeline = pipeline
        self._timeout_ms = timeout_ms
        self._lock = threading.Lock()
        self._latest: tuple[np.ndarray, np.ndarray, float] | None = None
        self._error: Exception | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                frames = self._pipeline.wait_for_frames(self._timeout_ms)
                color = np.asanyarray(frames.get_color_frame().get_data()).copy()
                depth = np.asanyarray(frames.get_depth_frame().get_data()).copy()
                with self._lock:
                    self._latest, self._error = (color, depth, time.time()), None
            except Exception as e:  # keep polling; `get` reports the error if no frame ever arrives
                with self._lock:
                    self._error = e

    def get(self, after: float | None = None, wait_s: float = 5.0) -> tuple[np.ndarray, np.ndarray, float]:
        """The newest (color, depth, arrival time).

        With `after`, waits for one that arrived more than `FRAME_LATENCY_S` after that time. Raises
        RuntimeError after `wait_s` seconds without a usable frame.
        """
        deadline = time.time() + wait_s
        newer_than = 0.0 if after is None else after + FRAME_LATENCY_S
        while True:
            with self._lock:
                latest, error = self._latest, self._error
            if latest is not None and time.time() - latest[2] < 2.0 and latest[2] > newer_than:
                return latest
            if time.time() > deadline:
                raise RuntimeError(f"no recent camera frame: {error!r}")
            time.sleep(0.01)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)


class RealHardware:
    """The RealSense D415 plus any of the arms. Arms are optional so the camera can be tried alone first.

    `arms` maps a robot id to its serial port, e.g. `{"left_arm": "/dev/ttyACM0"}`. Each arm is connected with
    LeRobot's calibration for that id (`calibrate=False`: the saved calibration is used as is).
    """

    def __init__(
        self,
        arms: dict[str, str] | None = None,
        camera: bool = True,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        librealsense_build: str | Path | None = DEFAULT_LIBREALSENSE_BUILD,
    ):
        """Open the camera (unless `camera` is False) and connect each arm. Closes everything on failure."""
        from lerobot.felix_jr.home import (
            make_follower,
        )  # imported here so a camera-only run needs no serial libs

        self.robots = {}
        self.head: HeadController | None = None
        self._frames: _LatestFrames | None = None
        self._pipeline = None
        self._calibration: dict[str, Any] | None = None
        try:
            if camera:
                self._start_camera(width, height, fps, librealsense_build)
            for robot_id, port in (arms or {}).items():
                robot = make_follower(port, robot_id)
                robot.connect(calibrate=False)
                self.robots[robot_id] = robot
                if getattr(robot, "with_head", False):
                    self.head = HeadController(
                        BusHeadBackend(robot),
                        robot.head_signs,
                        RangeStore(robot.calibration_dir / f"{robot.id}_head_range.json"),
                    )
        except Exception:
            self.close()
            raise

    def _start_camera(self, width: int, height: int, fps: int, librealsense_build) -> None:
        if librealsense_build:
            add_librealsense_to_path(librealsense_build)
        import pyrealsense2 as rs

        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self._pipeline = rs.pipeline()
        profile = self._pipeline.start(config)
        streams = {
            name: profile.get_stream(kind).as_video_stream_profile()
            for name, kind in (("color", rs.stream.color), ("depth", rs.stream.depth))
        }

        def intrinsics(stream) -> dict[str, Any]:
            i = stream.get_intrinsics()
            return {
                "width": i.width, "height": i.height, "fx": i.fx, "fy": i.fy, "ppx": i.ppx, "ppy": i.ppy,
                "model": str(i.model), "coeffs": list(i.coeffs),
            }  # fmt: skip

        ext = streams["depth"].get_extrinsics_to(streams["color"])
        self._calibration = {
            "depth_scale_m_per_unit": profile.get_device().first_depth_sensor().get_depth_scale(),
            "color_intrinsics": intrinsics(streams["color"]),
            "depth_intrinsics": intrinsics(streams["depth"]),
            "depth_to_color_rotation_col_major": list(ext.rotation),
            "depth_to_color_translation_m": list(ext.translation),
        }
        self._frames = _LatestFrames(self._pipeline)
        self._frames.get(wait_s=10.0)  # warm up: the first frames come with poor exposure

    def info(self) -> dict[str, Any]:
        """See `Hardware.info`."""
        return {
            "arms": list(self.robots),
            "head": any(getattr(r, "with_head", False) for r in self.robots.values()),
            "base": any(getattr(r, "with_base", False) for r in self.robots.values()),
            "camera": self._frames is not None,
            "fake": False,
        }

    def state(self) -> dict[str, Any]:
        """See `Hardware.state`."""
        out: dict[str, Any] = {"moving": False}
        for robot_id, robot in self.robots.items():
            out[robot_id] = {k: float(v) for k, v in robot.get_observation().items()}
        if self.head is not None:
            out["head"] = self.head.state()
        return out

    def capture(self, after: float | None = None) -> Snapshot:
        """See `Hardware.capture`. Needs the camera."""
        if self._frames is None or self._calibration is None:
            raise RuntimeError("this server was started without a camera")
        color, depth, timestamp = self._frames.get(after=after)
        return Snapshot(color, depth, self._calibration, self.state(), timestamp)

    def close(self) -> None:
        """Stop the camera and disconnect the arms; safe to call twice."""
        if self._frames is not None:
            self._frames.stop()
            self._frames = None
        if self._pipeline is not None:
            with contextlib.suppress(Exception):
                self._pipeline.stop()
            self._pipeline = None
        for robot in self.robots.values():
            with contextlib.suppress(Exception):
                robot.disconnect()
        self.robots = {}
