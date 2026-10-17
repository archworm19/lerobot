"""Helpers for storing and checking an SO-101 follower's "home" (rest) pose.

The home pose is saved next to the arm's calibration file as `<id>_home.json`, in the same units that
`get_observation()` reports (degrees for body joints, 0-100 for the gripper).
"""

import json
import time
import numpy as np
from pathlib import Path

from lerobot.felix_jr.base_follower import SO101FollowerWithBase
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.lerobot_types import RobotAction, RobotObservation  # aliases for dict[str, Any]


JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
SETTLE_WINDOW = 0.5  # seconds a joint must stay still to count as settled
SETTLE_MOTION = 0.3  # degrees of movement over the window that still counts as "still"
DEFAULT_MAX_VEL = 60.0  # deg/s (gripper: units/s)
DEFAULT_MAX_ACCEL = 120.0  # deg/s^2
CENTER = {f"{j}.pos": 0.0 for j in JOINTS}  # 0 deg = middle of the calibrated range
CENTER["gripper.pos"] = 50.0  # gripper is 0-100
STALE_GOAL = 10.0  # deg: if an arm is farther than this from its last commanded goal, start from where it is
HOME_TOLERANCE = 15.0  # warn at startup if any joint is farther than this from the saved home


def make_follower(
    port: str, robot_id: str, max_relative_target: float = 5.0, with_base: bool | None = None
) -> SO101Follower:
    """Build a follower. `with_base=None` auto-detects wheels: they are present if setup_base has saved
    `<robot_id>_base.json` for this arm. Pass True/False to override."""
    if max_relative_target is not None:
        print(f"[{robot_id}] max_relative_target={max_relative_target} deg (per-command cap) set by lerobot controller")
    if max_relative_target is None:
        # safety check
        raise ValueError("max_relative_target must be set to a positive number (degrees) to keep motion gentle")
    config = SO101FollowerConfig(
        port=port,
        id=robot_id,
        use_degrees=True,
        max_relative_target=max_relative_target,  # per-command cap in degrees; keeps motion gentle
    )
    if with_base is None:
        probe = SO101FollowerWithBase(config, with_base=False)  # only used for its calibration dir
        with_base = (probe.calibration_dir / f"{robot_id}_base.json").is_file()
        if with_base:
            print(f"[{robot_id}] wheel setup found: wheel servos (IDs 9, 10) must be on this arm's bus")
    # with_base: wheel servos (IDs 9-10) share this arm's bus.
    return SO101FollowerWithBase(config, with_base=with_base)


def home_path(robot: SO101Follower) -> Path:
    return robot.calibration_dir / f"{robot.id}_home.json"


def save_home(robot: SO101Follower, pose: dict[str, float]) -> Path:
    path = home_path(robot)
    path.write_text(json.dumps(pose, indent=2))
    return path


def load_home(robot: SO101Follower) -> dict[str, float] | None:
    path = home_path(robot)
    return json.loads(path.read_text()) if path.is_file() else None


def deviations(pose: dict[str, float], ref: dict[str, float]) -> dict[str, float]:
    return {k: pose[k] - ref[k] for k in ref}


def stuck_report(robot: SO101Follower, joints: list[str]) -> dict[str, dict[str, int]]:
    """Raw servo status for `joints`: torque on?, load, current, temperature, error flags."""
    regs = ["Torque_Enable", "Present_Load", "Present_Current", "Present_Temperature", "Status"]
    data = {r: robot.bus.sync_read(r, joints, normalize=False) for r in regs}
    return {j: {r: int(data[r][j]) for r in regs} for j in joints}


def move_time_trap(displacement: float, max_vel: float, max_accel: float) -> float:
    """Minimum time for a trapezoidal velocity profile (accel = decel = `max_accel`) over `displacement`.

    If the joint can't reach `max_vel` before it has to brake, the profile is a triangle.
    With the same limits for every joint, only the largest displacement needs to be passed in.
    """
    d = abs(displacement)
    t_cruise = d / max_vel - max_vel / max_accel
    if t_cruise < 0:
        return float(2 * np.sqrt(d / max_accel))  # triangle: never reaches max_vel
    return float(t_cruise + 2 * max_vel / max_accel)


def schedule_trapezoid(displacement: float, total_time: float, max_accel: float) -> tuple[float, float]:
    """Stretch a joint's trapezoid so it takes exactly `total_time` (>= its own minimum time).

    Acceleration stays at `max_accel`; the cruise speed is lowered instead. Solving
    d = a*t1^2 + a*t1*(T - 2*t1) for t1 (the smaller root) gives the time spent accelerating.
    Returns (t_accel, v_peak). The triangle case falls out when the discriminant is 0 (t1 = T/2).
    """
    d = abs(displacement)
    if d == 0 or total_time <= 0:
        return 0.0, 0.0
    disc = max(0.0, total_time**2 - 4 * d / max_accel)  # clamp: float rounding at the limiting joint
    t_accel = (total_time - np.sqrt(disc)) / 2
    return float(t_accel), float(max_accel * t_accel)


def trapezoid_offset(
    t: float, displacement: float, total_time: float, t_accel: float, v_peak: float, max_accel: float
) -> float:
    """Signed distance travelled from the start after `t` seconds of a scheduled trapezoid."""
    d = abs(displacement)
    t = min(max(t, 0.0), total_time)
    if t < t_accel:
        s = 0.5 * max_accel * t**2
    elif t < total_time - t_accel:
        s = 0.5 * max_accel * t_accel**2 + v_peak * (t - t_accel)
    else:
        s = d - 0.5 * max_accel * (total_time - t) ** 2
    return float(np.sign(displacement) * s)


# Last goal commanded to each arm (by robot id). A new trajectory starts from here instead of the measured
# pose: the arm rests a few degrees off its goal (gravity, P-only control), and starting from the measured
# pose would snap the servo's goal to that sagging position and make the arm jump.
_last_goal: dict[str, dict[str, float]] = {}


def remember_goal(robot_id: str, goal: dict[str, float]) -> None:
    """Tell `move_all` the last goal sent to an arm by code other than `move_all` (e.g. teleop), so the
    next trajectory starts from it."""
    _last_goal[robot_id] = dict(goal)


def move_all(
    robots: dict[str, SO101Follower],  # mapping from robot_id to SO101Follower object
    targets: dict[str, RobotAction],  # mapping from robot_id to joint actions
    tol: float = 1.5,
    max_vel: float = DEFAULT_MAX_VEL,  # cruise speed limit, applied to every joint
    max_accel: float = DEFAULT_MAX_ACCEL,  # acceleration/deceleration limit, applied to every joint
    rate_hz: float = 30.0,  # waypoint rate
    timeout: float = 10.0,  # seconds allowed for the settle phase after the trajectory ends
    min_dwell_time: float = 0.2,  # seconds all joints must stay within tolerance
) -> bool:
    """Move several arms at once along synchronized trapezoidal trajectories.

    Every joint follows its own trapezoid, scheduled so that all joints of all arms start and stop
    together (the joint with the longest move sets the duration). One waypoint per tick is sent; the
    lerobot `max_relative_target` clamp stays on as a safety net. Afterwards the final targets are held
    until every joint is within `tol` for `min_dwell_time`, or `timeout` expires.
    This is meant to run in a single thread ~ if you want parallel, spin this up as separate python process.
    """
    period = 1.0 / rate_hz

    # plan: per-joint displacement from the last commanded goal (or measured pose), one shared duration
    starts: dict[str, dict[str, float]] = {}
    disp: dict[str, dict[str, float]] = {}
    for robot_id, robot in robots.items():
        obs = robot.get_observation()
        last = _last_goal.get(robot_id, {})
        starts[robot_id] = {}
        for k in targets[robot_id]:
            use_last = k in last and abs(last[k] - obs[k]) <= STALE_GOAL  # else moved by hand / first move
            starts[robot_id][k] = last[k] if use_last else obs[k]
        disp[robot_id] = {k: targets[robot_id][k] - starts[robot_id][k] for k in targets[robot_id]}
    biggest = max((abs(d) for per_robot in disp.values() for d in per_robot.values()), default=0.0)
    total_time = move_time_trap(biggest, max_vel, max_accel)
    plan = {
        robot_id: {k: schedule_trapezoid(d, total_time, max_accel) for k, d in per_robot.items()}
        for robot_id, per_robot in disp.items()
    }

    # execute the trajectory at a fixed rate
    t0 = time.perf_counter()
    while (t := time.perf_counter() - t0) < total_time:
        for robot_id, robot in robots.items():
            waypoint = {
                k: starts[robot_id][k]
                + trapezoid_offset(t, disp[robot_id][k], total_time, *plan[robot_id][k], max_accel)
                for k in disp[robot_id]
            }
            robot.send_action(waypoint)
            _last_goal[robot_id] = waypoint  # so an interrupted move still hands off smoothly
        time.sleep(max(0.0, period - ((time.perf_counter() - t0) - t)))

    # hold the final targets until everything is within tolerance (or timeout)
    for robot_id in robots:
        _last_goal[robot_id] = dict(targets[robot_id])
    settle_start = time.perf_counter()
    dwell_start: float | None = None
    while True:
        off_target: dict[str, float] = {}
        for robot_id, robot in robots.items():
            robot.send_action(targets[robot_id])
            obs = robot.get_observation()
            for k, v in targets[robot_id].items():
                if abs(obs[k] - v) >= tol:
                    off_target[f"{robot_id}.{k}"] = round(obs[k] - v, 1)
        now = time.perf_counter()
        if off_target:
            dwell_start = None
        elif dwell_start is None:
            dwell_start = now
        elif now - dwell_start >= min_dwell_time:
            return True
        if now - settle_start >= timeout:
            print(f"  (timeout after {timeout:.0f}s; still off target by deg: {off_target})")
            for robot_id, robot in robots.items():
                prefix = robot_id + "."
                joints = [k.removeprefix(prefix).removesuffix(".pos") for k in off_target]
                joints = [j for k, j in zip(off_target, joints) if k.startswith(prefix)]
                if joints:
                    print(f"  [{robot_id}] servo status: {stuck_report(robot, joints)}")
            return False
        time.sleep(period)


def move_to(
    robot: SO101Follower, target: dict[str, float], tol: float = 1.5, timeout: float = 10.0
) -> bool:
    """Move a single arm to `target`."""
    return move_all({robot.id: robot}, {robot.id: target}, tol=tol, timeout=timeout)
