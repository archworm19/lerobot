"""What the server can ask of the robot: `Hardware`, a fake one for development, and the real one for the Pi.

`Hardware` is the only thing `server.py` talks to, so the server and client can be developed and tested on a
computer with no robot (`FakeHardware`). `RealHardware` drives the RealSense through `pyrealsense2` directly,
rather than through LeRobot's `RealSenseCamera`: a pyrealsense2 built from source has no pip metadata, so
LeRobot's availability check reports it as not installed and the camera class refuses to open.
"""

import contextlib
import glob
import logging
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from lerobot.felix_jr.server.head import HeadController, RangeStore
from lerobot.felix_jr.server.motion import MotionController

log = logging.getLogger("felix_hardware")

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

    # Set by the implementations: `head` is a `HeadController`, or None if there are no head servos; `motion`
    # runs the arm and wheel motions (and answers `state()`).
    head: HeadController | None
    motion: MotionController


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


def make_head_controller(robot) -> HeadController | None:
    """The `HeadController` of a connected `SO101FollowerWithBase`, or None if it has no head servos.

    Its recorded limits and home are read from `<id>_head_range.json` next to the arm's calibration.
    """
    if not getattr(robot, "with_head", False):
        return None
    return HeadController(
        BusHeadBackend(robot),
        robot.head_signs,
        RangeStore(robot.calibration_dir / f"{robot.id}_head_range.json"),
    )


class FakeArms:
    """Simulated arms (and a wheel base on the last one) that follow a trajectory in a fraction of real time.

    A trajectory of `d` seconds takes `d * time_scale` seconds. The arms start at their home pose. Using the
    "bus" (`observe`, the wheels) from a second thread while a trajectory runs is counted in `violations`, as
    that would corrupt the packets on a real serial bus. `trajectories` records every request.
    """

    HOME = {
        "shoulder_pan.pos": 0.0,
        "shoulder_lift.pos": -100.0,
        "elbow_flex.pos": 95.0,
        "wrist_flex.pos": 70.0,
        "wrist_roll.pos": 0.0,
        "gripper.pos": 1.0,
    }

    def __init__(self, ids: tuple[str, ...] = ("left_arm", "right_arm"), time_scale: float = 0.05):
        """Arms called `ids`, each at its home pose."""
        self.ids, self.time_scale = list(ids), time_scale
        self.pos = {i: dict(self.HOME) for i in ids}
        self.violations = 0
        self.trajectories: list[tuple[dict, float, float]] = []
        self.wheels = (0.0, 0.0)
        self._active: int | None = None  # id of the thread running a trajectory

    def _use_bus(self) -> None:
        if self._active is not None and self._active != threading.get_ident():
            self.violations += 1

    def arm_ids(self) -> list[str]:
        """See `ArmBackend.arm_ids`."""
        return list(self.ids)

    def observe(self) -> dict[str, dict[str, float]]:
        """See `ArmBackend.observe`."""
        self._use_bus()
        return {i: dict(p) for i, p in self.pos.items()}

    def homes(self) -> dict[str, dict[str, float]]:
        """See `ArmBackend.homes`."""
        return {i: dict(self.HOME) for i in self.ids}

    def run_trajectory(self, targets, max_vel, max_accel, tol, timeout, should_stop, on_tick) -> bool:
        """See `ArmBackend.run_trajectory`: a straight-line move at `max_vel`, plus the time to speed up."""
        self.trajectories.append((targets, max_vel, max_accel))
        self._active = threading.get_ident()
        try:
            starts = {i: {k: self.pos[i][k] for k in joints} for i, joints in targets.items()}
            biggest = max(
                (abs(v - starts[i][k]) for i, j in targets.items() for k, v in j.items()), default=0.0
            )
            duration = max(0.02, (biggest / max_vel + max_vel / max_accel) * self.time_scale)
            started = time.monotonic()
            while (elapsed := time.monotonic() - started) < duration:
                if should_stop():
                    return False
                fraction = elapsed / duration
                for i, joints in targets.items():
                    for k, v in joints.items():
                        self.pos[i][k] = starts[i][k] + fraction * (v - starts[i][k])
                on_tick()
                time.sleep(0.005)
            for i, joints in targets.items():
                self.pos[i].update(joints)
            return True
        finally:
            self._active = None

    def has_base(self) -> bool:
        """See `ArmBackend.has_base`."""
        return True

    def set_base_velocity(self, left_degps: float, right_degps: float) -> None:
        """See `ArmBackend.set_base_velocity`."""
        self._use_bus()
        self.wheels = (left_degps, right_degps)

    def base_velocity(self) -> tuple[float, float]:
        """See `ArmBackend.base_velocity` (the wheels follow instantly)."""
        self._use_bus()
        return self.wheels

    def stop_base(self) -> None:
        """See `ArmBackend.stop_base`."""
        self._use_bus()
        self.wheels = (0.0, 0.0)


class RealArms:
    """The connected `SO101FollowerWithBase` arms, driven with `home.move_all`."""

    def __init__(self, robots: dict[str, Any]):
        """`robots` maps a robot id to its connected follower.

        Home poses are read from the saved files; an arm without one can be read but not moved.
        """
        from lerobot.felix_jr.home import load_home

        self.robots = robots
        self._homes: dict[str, dict[str, float]] = {}
        for robot_id, robot in robots.items():
            home = load_home(robot)
            if home is None:
                log.warning("%s has no recorded home (setup.record_home): it cannot be moved", robot_id)
            else:
                self._homes[robot_id] = home

    def arm_ids(self) -> list[str]:
        """See `ArmBackend.arm_ids`."""
        return list(self.robots)

    def observe(self) -> dict[str, dict[str, float]]:
        """See `ArmBackend.observe`."""
        return {
            robot_id: {k: float(v) for k, v in robot.get_observation().items()}
            for robot_id, robot in self.robots.items()
        }

    def homes(self) -> dict[str, dict[str, float]]:
        """See `ArmBackend.homes`."""
        return {robot_id: dict(home) for robot_id, home in self._homes.items()}

    def run_trajectory(self, targets, max_vel, max_accel, tol, timeout, should_stop, on_tick) -> bool:
        """See `ArmBackend.run_trajectory`."""
        from lerobot.felix_jr.home import move_all

        return move_all(
            {robot_id: self.robots[robot_id] for robot_id in targets},
            targets,
            tol=tol,
            max_vel=max_vel,
            max_accel=max_accel,
            timeout=timeout,
            should_stop=should_stop,
            on_tick=on_tick,
        )

    def _base_robot(self):
        return next((r for r in self.robots.values() if getattr(r, "with_base", False)), None)

    def has_base(self) -> bool:
        """See `ArmBackend.has_base`."""
        return self._base_robot() is not None

    def set_base_velocity(self, left_degps: float, right_degps: float) -> None:
        """See `ArmBackend.set_base_velocity`."""
        self._base_robot().set_base_velocity(left_degps, right_degps)

    def base_velocity(self) -> tuple[float, float]:
        """See `ArmBackend.base_velocity`."""
        speeds = self._base_robot().get_base_velocity()
        return float(speeds["base_left_wheel"]), float(speeds["base_right_wheel"])

    def stop_base(self) -> None:
        """See `ArmBackend.stop_base`."""
        robot = self._base_robot()
        if robot is not None:
            robot.stop_base()


class FakeHardware:
    """A robot with a synthetic scene: a flat wall 1 m away with a box 0.6 m away, and arms at rest.

    The eyes servo tilts the picture: each tick away from the middle shifts the image by `PIXELS_PER_TICK`
    rows, so a head move visibly changes what `capture` returns.
    """

    WIDTH, HEIGHT, FX = 320, 240, 200.0
    PIXELS_PER_TICK = 0.1

    def __init__(
        self,
        arms: tuple[str, ...] = ("left_arm", "right_arm"),
        head_range_path: Path | None = None,
        arm_time_scale: float = 0.25,
        **motion_kwargs: Any,
    ):
        """A fake robot with arms named `arms`.

        `head_range_path` is where the head's recorded range is kept (default: not saved). `arm_time_scale`
        shortens simulated arm motions (see `FakeArms`); `motion_kwargs` go to `MotionController` (e.g. a
        short `watchdog_s` for tests).
        """
        self.captures = 0
        self.arm_backend = FakeArms(arms, arm_time_scale)
        self.head: HeadController | None = HeadController(
            FakeHeadBackend(), {"head_neck": -1, "head_eyes": 1}, RangeStore(head_range_path)
        )
        self.motion = MotionController(self.arm_backend, head=self.head, **motion_kwargs)

    def info(self) -> dict[str, Any]:
        """See `Hardware.info`."""
        return {
            "arms": self.arm_backend.arm_ids(),
            "head": True,
            "base": True,
            "camera": True,
            "motion": True,
            "fake": True,
        }

    def state(self) -> dict[str, Any]:
        """See `Hardware.state`."""
        return self.motion.state()

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
        """Stop any motion and go to rest (there is nothing else to release)."""
        self.motion.shutdown(go_home=True)


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
        home_on_exit: bool = True,
        **motion_kwargs: Any,
    ):
        """Open the camera (unless `camera` is False) and connect each arm. Closes everything on failure.

        `home_on_exit`: on `close`, send the arms (and head) to their recorded home before the servos are
        released, since releasing switches torque off. `motion_kwargs` go to `MotionController`.
        """
        from lerobot.felix_jr.home import (
            make_follower,
        )  # imported here so a camera-only run needs no serial libs

        self.robots = {}
        self.head: HeadController | None = None
        self.motion: MotionController | None = None
        self._home_on_exit = home_on_exit
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
                self.head = make_head_controller(robot) or self.head
            self.motion = MotionController(RealArms(self.robots), head=self.head, **motion_kwargs)
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
            "motion": True,
            "fake": False,
        }

    def state(self) -> dict[str, Any]:
        """See `Hardware.state`."""
        return self.motion.state()

    def capture(self, after: float | None = None) -> Snapshot:
        """See `Hardware.capture`. Needs the camera."""
        if self._frames is None or self._calibration is None:
            raise RuntimeError("this server was started without a camera")
        color, depth, timestamp = self._frames.get(after=after)
        return Snapshot(color, depth, self._calibration, self.state(), timestamp)

    def close(self) -> None:
        """Go to rest, stop the camera and disconnect the arms; safe to call twice."""
        if self.motion is not None:
            self.motion.shutdown(go_home=self._home_on_exit)  # before the servos' torque is switched off
            self.motion = None
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
