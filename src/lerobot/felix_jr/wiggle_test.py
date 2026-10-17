"""Sign-of-life test for SO-101 followers: go to center, nudge each joint +/- a few degrees, return home.

Pass --left and/or --right with the arm's serial port; at least one is required. If both are given, the
arms move at the same time. Run `record_home` for each arm first so it has a saved home pose.

Usage: uv run python -m lerobot.felix_jr.wiggle_test [--left PORT] [--right PORT] [--delta DEG]
       [--pause SEC] [--max-vel DEG/S] [--max-accel DEG/S2] [--tol DEG] [--timeout SEC]

Example command:
uv run python -m lerobot.felix_jr.wiggle_test --left /dev/tty.usbmodem5AE60864731 --right /dev/tty.usbmodem5AE60837301 --delta 20 --tol 5 --max-vel 60 --max-accel 25 --timeout 5       
"""

import argparse
import logging
import signal
import sys
import time

from lerobot.felix_jr.home import (
    DEFAULT_MAX_ACCEL,
    DEFAULT_MAX_VEL,
    HOME_TOLERANCE,
    JOINTS,
    deviations,
    home_path,
    load_home,
    make_follower,
    move_all
)
from lerobot.robots.so_follower import SO101Follower

CENTER = {f"{j}.pos": 0.0 for j in JOINTS}  # 0 deg = middle of the calibrated range
CENTER["gripper.pos"] = 50.0  # gripper is 0-100


def wiggle(
    robots: dict[str, SO101Follower],
    homes: dict[str, dict[str, float]],
    delta: float,
    pause: float,
    move_kw: dict[str, float],
) -> None:
    for robot_id, robot in robots.items():
        start = robot.get_observation()  # get the current pose in degrees
        print(f"[{robot_id}] Start pose:", {k: round(v, 1) for k, v in start.items()})
        devs = deviations(start, homes[robot_id])
        far = {k: round(d, 1) for k, d in devs.items() if abs(d) > HOME_TOLERANCE}
        if far:
            print(f"[{robot_id}] WARNING: not near saved home (off by more than {HOME_TOLERANCE} deg): {far}")
    input("Clear the workspace, hold the arm(s) if needed. ENTER to move to center...")

    def move(target: dict[str, float]) -> None:
        move_all(robots, dict.fromkeys(robots, target), **move_kw)

    try:
        move(CENTER)
        for j in JOINTS:
            key = f"{j}.pos"
            for sign in (+1, -1):
                target = {**CENTER, key: CENTER[key] + sign * delta}
                print(f"{key}: {target[key]:+.1f}")
                move(target)
                time.sleep(pause)
                print("   read back:", {i: f"{r.get_observation()[key]:+.1f}" for i, r in robots.items()})
            move(CENTER)
    except KeyboardInterrupt:
        print("\nCtrl+C: stopping the wiggle.")
    finally:
        # Ignore further Ctrl+C so a second press can't cut the trip home short.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        print("Returning to home (torque stays on until disconnect).")
        try:
            move_all(robots, homes, **move_kw)
        except Exception as e:  # e.g. an arm's USB serial port vanished
            print(f"Could not return to home: {e!r}")


parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--left", metavar="PORT", help="serial port of the left arm (id: left_arm)")
parser.add_argument("--right", metavar="PORT", help="serial port of the right arm (id: right_arm)")
parser.add_argument("--delta", type=float, default=10.0, help="nudge size in degrees (default: 10)")
parser.add_argument("--pause", type=float, default=0.3, help="seconds to hold each nudge (default: 0.3)")
parser.add_argument(
    "--max-vel", type=float, default=DEFAULT_MAX_VEL, help="cruise speed limit, deg/s (default: %(default)s)"
)
parser.add_argument(
    "--max-accel", type=float, default=DEFAULT_MAX_ACCEL, help="accel limit, deg/s^2 (default: %(default)s)"
)
parser.add_argument("--tol", type=float, default=1.5, help="deg from target that counts as arrived")
parser.add_argument("--timeout", type=float, default=10.0, help="max seconds to settle after each move")
args = parser.parse_args()
move_kw = {"max_vel": args.max_vel, "max_accel": args.max_accel, "tol": args.tol, "timeout": args.timeout}
# max_relative_target clamping logs a warning on every tick; it is expected here and drowns real output.
logging.getLogger("lerobot.robots.utils").setLevel(logging.ERROR)

arms = {"left_arm": args.left, "right_arm": args.right}
arms = {robot_id: port for robot_id, port in arms.items() if port}
if not arms:
    parser.error("supply at least one of --left PORT or --right PORT")

robots: dict[str, SO101Follower] = {}
try:
    # Connect everything and validate homes before any arm moves.
    for robot_id, port in arms.items():
        robot = make_follower(port, robot_id)
        robot.connect(calibrate=False)  # use the saved calibration, don't re-run it
        robots[robot_id] = robot
    homes = {}
    for robot_id, robot in robots.items():
        home = load_home(robot)
        if home is None:
            sys.exit(f"[{robot_id}] No home pose at {home_path(robot)}. Run lerobot.felix_jr.record_home.")
        homes[robot_id] = home

    wiggle(robots, homes, args.delta, args.pause, move_kw)
except KeyboardInterrupt:
    print("\nCtrl+C before the wiggle started; arms were not moved.")
finally:
    if robots:
        print("Disabling torque (arms will go limp - support them).")
    for robot_id, robot in robots.items():
        try:  # one dead port must not stop the other arms from disconnecting
            robot.disconnect()
        except Exception as e:
            print(f"[{robot_id}] disconnect failed ({e!r}); check power/USB, torque may still be on.")
