"""Sign-of-life test for SO-101 followers: go to center, nudge each joint +/- a few degrees, return home.
Then, for an arm that has them, the head servos (down/up, left/right) and the wheels (each alone, then both).

Pass --left and/or --right with the arm's serial port; at least one is required. If both are given, the
arms move at the same time. Run `setup/record_home` for each arm first so it has a saved home pose. Head and
wheel servos are included automatically once `setup.head` / `setup.wheels` have run for that arm (use
`--no-head` / `--no-base` to leave them out). The wheels get a prompt first: lift the robot so they can
spin freely.

Usage: uv run python -m lerobot.felix_jr.checks.wiggle_test [--left PORT] [--right PORT] [--delta DEG]
       [--pause SEC] [--max-vel DEG/S] [--max-accel DEG/S2] [--tol DEG] [--timeout SEC] [--no-base]
       [--no-head] [--head-delta DEG] [--wheel-speed DEG/S] [--wheel-seconds S]

Example command:
uv run python -m lerobot.felix_jr.checks.wiggle_test --left /dev/tty.usbmodem5AE60864731 --right /dev/tty.usbmodem5AE60837301 --delta 20 --tol 5 --max-vel 60 --max-accel 25 --timeout 5       
"""

import argparse
import logging
import signal
import sys
import time

from lerobot.felix_jr.home import (
    DEFAULT_MAX_ACCEL,
    DEFAULT_MAX_VEL,
    CENTER,
    HOME_TOLERANCE,
    JOINTS,
    deviations,
    home_path,
    load_home,
    make_follower,
    move_all
)
from lerobot.robots.so_follower import SO101Follower



def wiggle(
    robots: dict[str, SO101Follower],
    homes: dict[str, dict[str, float]],
    delta: float,
    pause: float,
    move_kw: dict[str, float],
) -> bool:
    """Wiggle the arm joints and return home. False if it was interrupted with Ctrl+C."""
    completed = True
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
        completed = False
        print("\nCtrl+C: stopping the wiggle.")
    finally:
        # Ignore further Ctrl+C so a second press can't cut the trip home short.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        print("Returning to home (torque stays on until disconnect).")
        try:
            move_all(robots, homes, **move_kw)
        except Exception as e:  # e.g. an arm's USB serial port vanished
            print(f"Could not return to home: {e!r}")
    return completed


def exercise_extras(
    robots: dict[str, SO101Follower], head_delta: float, wheel_speed: float, wheel_seconds: float
) -> None:
    """Head servos, then wheels, for every arm that has them."""
    for robot_id, robot in robots.items():
        for name in robot.head_motors:
            to_pos, to_neg = robot.wiggle_head(name, head_delta)
            positive, negative = ("down", "up") if name == "head_eyes" else ("left", "right")
            print(
                f"[{robot_id}] {name}: asked +/-{head_delta:.0f} deg, moved {to_pos:+.1f} deg {positive} "
                f"and {to_neg:+.1f} deg {negative}"
            )
            if min(to_pos, to_neg) < 0.3 * head_delta:
                print(f"[{robot_id}] WARNING: {name} moved much less than commanded (blocked or no power?)")
    for robot_id, robot in robots.items():
        if not robot.with_base:
            continue
        answer = input(f"[{robot_id}] Wheels: robot lifted? ENTER to spin them freely, 's' to skip: ")
        if answer.strip().lower() == "s":
            continue
        for step in robot.exercise_base(wheel_speed, wheel_seconds):
            (cl, cr), (ml, mr) = step["commanded"], step["measured"]
            print(
                f"[{robot_id}] {step['label']:22s} commanded L{cl:+.0f} R{cr:+.0f} deg/s, "
                f"measured L{ml:+.0f} R{mr:+.0f}"
            )
            for side, cmd, meas in (("left", cl, ml), ("right", cr, mr)):
                if cmd and meas * cmd < 0.3 * cmd * cmd:
                    print(f"[{robot_id}] WARNING: {side} wheel did not follow the command (direction/power?)")


parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--left", metavar="PORT", help="serial port of the left arm (id: left_arm)")
parser.add_argument("--right", metavar="PORT", help="serial port of the right arm (id: right_arm)")
parser.add_argument(
    "--no-base", action="store_true", help="ignore wheel servos even if an arm has a base set up"
)
parser.add_argument(
    "--no-head", action="store_true", help="ignore head servos even if an arm has a head set up"
)
parser.add_argument("--delta", type=float, default=10.0, help="nudge size in degrees (default: 10)")
parser.add_argument("--head-delta", type=float, default=8.0, help="head servo nudge in degrees (default: 8)")
parser.add_argument("--wheel-speed", type=float, default=40.0, help="wheel speed in deg/s (default: 40)")
parser.add_argument("--wheel-seconds", type=float, default=1.0, help="seconds per wheel step (default: 1)")
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
        robot = make_follower(
            port,
            robot_id,
            with_base=False if args.no_base else None,
            with_head=False if args.no_head else None,
        )
        robot.connect(calibrate=False)  # use the saved calibration, don't re-run it
        robots[robot_id] = robot
    homes = {}
    for robot_id, robot in robots.items():
        home = load_home(robot)
        if home is None:
            sys.exit(
                f"[{robot_id}] No home pose at {home_path(robot)}. Run lerobot.felix_jr.setup.record_home."
            )
        homes[robot_id] = home

    if wiggle(robots, homes, args.delta, args.pause, move_kw):
        exercise_extras(robots, args.head_delta, args.wheel_speed, args.wheel_seconds)
except KeyboardInterrupt:
    print("\nCtrl+C: stopping.")
    for robot in robots.values():
        try:
            robot.stop_base()  # a wheel step may have been running
        except Exception as e:
            print(f"Could not stop the wheels: {e!r}")
finally:
    if robots:
        print("Disabling torque (arms will go limp - support them).")
    for robot_id, robot in robots.items():
        try:  # one dead port must not stop the other arms from disconnecting
            robot.disconnect()
        except Exception as e:
            print(f"[{robot_id}] disconnect failed ({e!r}); check power/USB, torque may still be on.")
