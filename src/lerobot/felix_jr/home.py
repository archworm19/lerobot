"""Helpers for storing and checking an SO-101 follower's "home" (rest) pose.

The home pose is saved next to the arm's calibration file as `<id>_home.json`, in the same units that
`get_observation()` reports (degrees for body joints, 0-100 for the gripper).
"""

import json
import time
from collections.abc import Callable
from pathlib import Path

from lerobot.felix_jr.base_follower import SO101FollowerWithBase
from lerobot.felix_jr.trajectory import (  # (the trapezoids re-exported: `diagnostics/lift_test.py`)
    DWELL_S,
    WAYPOINT_HZ,
    MovePlan,
    Settler,
    move_starts,
    move_time_trap,  # noqa: F401
    schedule_trapezoid,  # noqa: F401
    trapezoid_offset,  # noqa: F401
)
from lerobot.lerobot_types import RobotAction  # aliases for dict[str, Any]
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
SETTLE_WINDOW = 0.5  # seconds a joint must stay still to count as settled
SETTLE_MOTION = 0.3  # degrees of movement over the window that still counts as "still"
DEFAULT_MAX_VEL = 60.0  # deg/s (gripper: units/s)
DEFAULT_MAX_ACCEL = 120.0  # deg/s^2
CENTER = {f"{j}.pos": 0.0 for j in JOINTS}  # 0 deg = middle of the calibrated range
CENTER["gripper.pos"] = 50.0  # gripper is 0-100
HOME_TOLERANCE = 15.0  # warn at startup if any joint is farther than this from the saved home


def make_follower(
    port: str,
    robot_id: str,
    max_relative_target: float = 5.0,
    with_base: bool | None = None,
    with_head: bool | None = None,
    calibration_dir: Path | None = None,
) -> SO101Follower:
    """Build a follower.

    `with_base` / `with_head` = None auto-detects the extra servos on this arm's bus: the
    wheels are present if setup.wheels has saved `<robot_id>_base.json`, the head if setup.head has saved
    `<robot_id>_head.json`. Pass True/False to override. `calibration_dir` overrides where the calibration,
    home and sign files live (default: LeRobot's calibration folder).
    """
    if max_relative_target is not None:
        print(
            f"[{robot_id}] max_relative_target={max_relative_target} deg (per-command cap) set by lerobot controller"
        )
    if max_relative_target is None:
        # safety check
        raise ValueError(
            "max_relative_target must be set to a positive number (degrees) to keep motion gentle"
        )
    config = SO101FollowerConfig(
        port=port,
        id=robot_id,
        use_degrees=True,
        max_relative_target=max_relative_target,  # per-command cap in degrees; keeps motion gentle
        calibration_dir=calibration_dir,
    )
    if with_base is None or with_head is None:
        probe = SO101FollowerWithBase(config, with_base=False)  # only used for its calibration dir
        if with_base is None:
            with_base = (probe.calibration_dir / f"{robot_id}_base.json").is_file()
            if with_base:
                print(f"[{robot_id}] wheel setup found: wheel servos (IDs 9, 10) must be on this arm's bus")
        if with_head is None:
            with_head = (probe.calibration_dir / f"{robot_id}_head.json").is_file()
            if with_head:
                print(f"[{robot_id}] head setup found: head servos (IDs 7, 8) must be on this arm's bus")
    return SO101FollowerWithBase(config, with_base=with_base, with_head=with_head)


def home_path(robot: SO101Follower) -> Path:
    """Where an arm's home pose is saved: next to its calibration, `<id>_home.json`."""
    return robot.calibration_dir / f"{robot.id}_home.json"


def save_home(robot: SO101Follower, pose: dict[str, float]) -> Path:
    """Save `pose` as the arm's home (`home_path`); returns the file."""
    path = home_path(robot)
    path.write_text(json.dumps(pose, indent=2))
    return path


def load_home(robot: SO101Follower) -> dict[str, float] | None:
    """The arm's saved home pose, or None if none was recorded."""
    path = home_path(robot)
    return json.loads(path.read_text()) if path.is_file() else None


def deviations(pose: dict[str, float], ref: dict[str, float]) -> dict[str, float]:
    """Each of `ref`'s joints: `pose`'s value less `ref`'s."""
    return {k: pose[k] - ref[k] for k in ref}


def stuck_report(robot: SO101Follower, joints: list[str]) -> dict[str, dict[str, int]]:
    """Raw servo status for `joints`: torque on?, load, current, temperature, error flags."""
    regs = ["Torque_Enable", "Present_Load", "Present_Current", "Present_Temperature", "Status"]
    data = {r: robot.bus.sync_read(r, joints, normalize=False) for r in regs}
    return {j: {r: int(data[r][j]) for r in regs} for j in joints}


# Last goal commanded to each arm (by robot id). A new trajectory starts from here instead of the measured
# pose: the arm rests a few degrees off its goal (gravity, P-only control), and starting from the measured
# pose would snap the servo's goal to that sagging position and make the arm jump.
_last_goal: dict[str, dict[str, float]] = {}


def remember_goal(robot_id: str, goal: dict[str, float]) -> None:
    """Tell `move_all` the last goal sent to an arm by code other than `move_all` (e.g. teleop).

    The next trajectory then starts from it.
    """
    _last_goal[robot_id] = dict(goal)


def update_goal(robot_id: str, joints: dict[str, float]) -> None:
    """Merge `joints` into the last goal `move_all` remembers for an arm (a goal written for some joints only)."""
    _last_goal[robot_id] = {**_last_goal.get(robot_id, {}), **joints}


def move_all(
    robots: dict[str, SO101Follower],  # mapping from robot_id to SO101Follower object
    targets: dict[str, RobotAction],  # mapping from robot_id to joint actions
    tol: float = 1.5,
    max_vel: float = DEFAULT_MAX_VEL,  # cruise speed limit, applied to every joint
    max_accel: float = DEFAULT_MAX_ACCEL,  # acceleration/deceleration limit, applied to every joint
    rate_hz: float = WAYPOINT_HZ,  # waypoint rate
    timeout: float = 10.0,  # seconds allowed for the settle phase after the trajectory ends
    min_dwell_time: float = DWELL_S,  # seconds all joints must stay within tolerance
    should_stop: Callable[[], bool]
    | None = None,  # polled every tick; True aborts and holds the last waypoint
    on_tick: Callable[[], None] | None = None,  # called every tick (e.g. to publish a progress snapshot)
) -> bool:
    """Move several arms at once along synchronized trapezoidal trajectories.

    Every joint follows its own trapezoid, scheduled so that all joints of all arms start and stop
    together (the joint with the longest move sets the duration). One waypoint per tick is sent; the
    lerobot `max_relative_target` clamp stays on as a safety net. Afterwards the final targets are held
    until every joint is within `tol` for `min_dwell_time`, or `timeout` expires.
    This is meant to run in a single thread ~ if you want parallel, spin this up as separate python process.

    `should_stop` lets another thread cancel: once it returns True, no further waypoints are sent (the servos
    keep holding the last one, and the next trajectory starts from it) and the function returns False.
    """
    period = 1.0 / rate_hz

    # plan: each joint from the last commanded goal (or the measured pose), one shared duration (`trajectory`, which
    # the simulator's replays share: `sim2real.shadow`)
    starts = {
        robot_id: move_starts(targets[robot_id], _last_goal.get(robot_id, {}), robot.get_observation())
        for robot_id, robot in robots.items()
    }
    plan = MovePlan.plan(starts, targets, max_vel, max_accel)
    total_time = plan.total_time

    # execute the trajectory at a fixed rate
    t0 = time.perf_counter()
    while (t := time.perf_counter() - t0) < total_time:
        if should_stop is not None and should_stop():
            return False
        if on_tick is not None:
            on_tick()
        waypoints = plan.waypoint(t)
        for robot_id, robot in robots.items():
            waypoint = waypoints[robot_id]
            robot.send_action(waypoint)
            _last_goal[robot_id] = waypoint  # so an interrupted move still hands off smoothly
        time.sleep(max(0.0, period - ((time.perf_counter() - t0) - t)))

    # hold the final targets until everything is within tolerance (or timeout)
    for robot_id in robots:
        _last_goal[robot_id] = dict(targets[robot_id])
    settler = Settler(timeout, min_dwell_time, began=time.perf_counter())
    while True:
        if should_stop is not None and should_stop():
            return False
        if on_tick is not None:
            on_tick()
        off_target: dict[str, float] = {}
        for robot_id, robot in robots.items():
            robot.send_action(targets[robot_id])
            obs = robot.get_observation()
            for k, v in targets[robot_id].items():
                if abs(obs[k] - v) >= tol:
                    off_target[f"{robot_id}.{k}"] = round(obs[k] - v, 1)
        verdict = settler.update(time.perf_counter(), within=not off_target)
        if verdict is True:
            return True
        if verdict is False:
            print(f"  (timeout after {timeout:.0f}s; still off target by deg: {off_target})")
            for robot_id, robot in robots.items():
                prefix = robot_id + "."
                joints = [k.removeprefix(prefix).removesuffix(".pos") for k in off_target]
                joints = [j for k, j in zip(off_target, joints, strict=True) if k.startswith(prefix)]
                if joints:
                    print(f"  [{robot_id}] servo status: {stuck_report(robot, joints)}")
            return False
        time.sleep(period)


def move_to(robot: SO101Follower, target: dict[str, float], tol: float = 1.5, timeout: float = 10.0) -> bool:
    """Move a single arm to `target`."""
    return move_all({robot.id: robot}, {robot.id: target}, tol=tol, timeout=timeout)
