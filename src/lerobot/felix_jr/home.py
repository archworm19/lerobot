"""Helpers for storing and checking an SO-101 follower's "home" (rest) pose.

The home pose is saved next to the arm's calibration file as `<id>_home.json`, in the same units that
`get_observation()` reports (degrees for body joints, 0-100 for the gripper).
"""

import json
import time
from pathlib import Path

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
SETTLE_WINDOW = 0.5  # seconds a joint must stay still to count as settled
SETTLE_MOTION = 0.3  # degrees of movement over the window that still counts as "still"
HOME_TOLERANCE = 15.0  # warn at startup if any joint is farther than this from the saved home


def make_follower(port: str, robot_id: str, max_relative_target: float | None = 5.0) -> SO101Follower:
    return SO101Follower(
        SO101FollowerConfig(
            port=port,
            id=robot_id,
            use_degrees=True,
            max_relative_target=max_relative_target,  # per-command cap in degrees; keeps motion gentle
        )
    )


def home_path(robot: SO101Follower) -> Path:
    return robot.calibration_dir / f"{robot.id}_home.json"


def read_pose(robot: SO101Follower) -> dict[str, float]:
    obs = robot.get_observation()
    return {f"{j}.pos": float(obs[f"{j}.pos"]) for j in JOINTS}


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


def move_all(
    robots: dict[str, SO101Follower],
    targets: dict[str, dict[str, float]],
    tol: float = 1.5,
    timeout: float = 10.0,
) -> bool:
    """Move several arms at once. Commands every arm's target each tick (max_relative_target makes each
    step gradual) until all arms are within `tol` of their targets, keeping arms that arrive in place."""
    t0 = time.time()
    history: list[tuple[float, dict[str, float]]] = []  # (time, {arm.joint: position}) for settle detection
    while True:
        errors = {}
        positions = {}
        for robot_id, robot in robots.items():
            robot.send_action(targets[robot_id])
            pose = read_pose(robot)
            for k, v in targets[robot_id].items():
                positions[f"{robot_id}.{k}"] = pose[k]
                if abs(pose[k] - v) >= tol:
                    errors[f"{robot_id}.{k}"] = round(pose[k] - v, 1)
        if not errors:
            return True
        now = time.time()
        history = [(t, p) for t, p in history if now - t <= SETTLE_WINDOW] + [(now, positions)]
        # Gravity leaves P-only joints a few degrees short of target. If everything still off target has
        # stopped moving, treat the move as done instead of waiting out the timeout.
        if now - t0 >= SETTLE_WINDOW and history[0][0] <= now - SETTLE_WINDOW + 0.1:
            moving = [k for k in errors if abs(positions[k] - history[0][1][k]) > SETTLE_MOTION]
            if not moving:
                return True
        if now - t0 >= timeout:
            print(f"  (timeout after {timeout:.0f}s; still off target by deg: {errors})")
            for robot_id, robot in robots.items():
                prefix = robot_id + "."
                joints = [k.removeprefix(prefix).removesuffix(".pos") for k in errors if k.startswith(prefix)]
                if joints:
                    print(f"  [{robot_id}] servo status: {stuck_report(robot, joints)}")
            return False
        time.sleep(1 / 30)


def move_to(
    robot: SO101Follower, target: dict[str, float], tol: float = 1.5, timeout: float = 10.0
) -> bool:
    """Move a single arm to `target`."""
    return move_all({robot.id: robot}, {robot.id: target}, tol, timeout)
