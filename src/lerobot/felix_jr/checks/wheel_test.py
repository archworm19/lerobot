"""Sign-of-life test for the wheel base: each wheel alone, both together, then spinning in place.

Run `setup.wheels` first. Start with the robot LIFTED so the wheels spin freely; add --ground afterwards
for short low-speed moves on the floor. Ctrl+C stops the wheels. The arm joints hold their current pose.

Usage: uv run python -m lerobot.felix_jr.checks.wheel_test <PORT> [ROBOT_ID]
       [--speed DEG/S] [--seconds S] [--ground]
  PORT / ROBOT_ID: the arm whose board the wheels are plugged into (default id: right_arm).

For default setup where robot attached to right wheel:
uv run python -m lerobot.felix_jr.checks.wheel_test /dev/tty.usbmodem5AE60837301 right_arm  
"""

import argparse
import signal
import time

from lerobot.felix_jr.home import make_follower

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("port", help="serial port of the arm the wheels share a board with")
parser.add_argument("robot_id", nargs="?", default="right_arm", help="that arm's id (default: right_arm)")
parser.add_argument("--speed", type=float, default=60.0, help="wheel speed in deg/s (default: 60)")
parser.add_argument("--seconds", type=float, default=1.0, help="duration of each step (default: 1)")
parser.add_argument("--ground", action="store_true", help="also do short moves on the ground")
args = parser.parse_args()

S = args.speed
# (label, left deg/s, right deg/s); positive = forward for that wheel (the saved signs make it so)
LIFTED_STEPS = [
    ("left wheel forward", S, 0),
    ("left wheel backward", -S, 0),
    ("right wheel forward", 0, S),
    ("right wheel backward", 0, -S),
    ("both forward", S, S),
    ("both backward", -S, -S),
    ("spin in place, counter-clockwise (left back, right forward)", -S, S),
    ("spin in place, clockwise (left forward, right back)", S, -S),
]
GROUND_STEPS = [("drive forward", S, S), ("drive backward", -S, -S), ("rotate in place", -S, S)]

robot = make_follower(args.port, args.robot_id, with_base=True, with_head=False)
if args.speed <= 0 or args.speed > robot.max_base_degps:
    raise SystemExit(f"--speed must be in (0, {robot.max_base_degps}] deg/s")


def run_step(label: str, left: float, right: float) -> None:
    print(f"\n{label}: commanding left={left:+.0f}, right={right:+.0f} deg/s")
    robot.set_base_velocity(left, right)
    time.sleep(args.seconds)
    measured = robot.get_base_velocity()  # sampled while still moving
    robot.stop_base()
    m_left, m_right = measured["base_left_wheel"], measured["base_right_wheel"]
    print(f"   measured left={m_left:+.0f}, right={m_right:+.0f} deg/s")
    for side, cmd, meas in (("left", left, m_left), ("right", right, m_right)):
        if cmd and meas * cmd < 0.3 * cmd * cmd:  # wrong direction or under 30% of the commanded speed
            print(f"   WARNING: {side} wheel did not follow the command (check signs/cabling/power)")
    time.sleep(0.3)


def run_stages(steps: list[tuple[str, float, float]]) -> None:
    for label, left, right in steps:
        run_step(label, left, right)
        input("   ENTER for the next step (Ctrl+C to stop)...")


try:
    robot.connect(calibrate=False)
    robot.send_action(robot.get_observation())  # hold the arm where it is
    robot.stop_base()
    input("Robot LIFTED so the wheels spin freely? ENTER to start the wheel steps...")
    run_stages(LIFTED_STEPS)
    if args.ground:
        input("\nPut the robot on the floor with room to move. ENTER for the ground steps...")
        run_stages(GROUND_STEPS)
except KeyboardInterrupt:
    print("\nCtrl+C: stopping.")
finally:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if robot.is_connected:
        try:
            robot.stop_base()
            print("Wheels stopped; disconnecting (arm torque turns off - support it).")
        except Exception as e:
            print(f"Could not stop the base cleanly: {e!r}; power off the motors if wheels keep spinning.")
        try:
            robot.disconnect()
        except Exception as e:
            print(f"disconnect failed ({e!r}); check power/USB.")
