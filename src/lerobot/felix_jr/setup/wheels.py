"""One-time setup of the wheel servos: assign their IDs and record each wheel's forward direction.

Wiring assumed: controller board -> RIGHT wheel -> LEFT wheel, with the arm also on the same board.
Wheels leave the factory as ID 1, which collides with the arm (shoulder_pan is ID 1) and with each other,
so the first wheel has to be set up alone:
  1. Unplug the arm's cable from the board, and unplug the LEFT wheel from the RIGHT wheel.
     The RIGHT wheel (next to the board) is set up first, alone.
  2. Reconnect the LEFT wheel to the RIGHT wheel (the arm stays off). The right wheel now has ID 10, so
     the left one is found by scanning the chain for the one servo that isn't the right wheel.
  3. Plug the arm back in when the script finishes.
Do this with the robot lifted so the wheels spin freely.

Usage: uv run python -m lerobot.felix_jr.setup.wheels <PORT> <ROBOT_ID>
  PORT / ROBOT_ID: the arm whose board the wheels are plugged into (here the right arm: its port and
  right_arm); the wheel directions are saved with that arm's files.
"""

import sys

from lerobot.felix_jr.home import make_follower

# In daisy-chain order from the board: the right wheel is first and is set up alone.
WHEELS = {"base_right_wheel": "RIGHT", "base_left_wheel": "LEFT"}

port, robot_id = sys.argv[1], sys.argv[2]
robot = make_follower(port, robot_id, with_base=True)

signs: dict[str, int] = {}
try:
    for n, (name, side) in enumerate(WHEELS.items()):
        if n == 0:
            input(
                f"Unplug the arm from the board and the other wheel from the {side} wheel, "
                f"so the board talks to the {side} wheel ONLY, then ENTER..."
            )
        else:
            input(f"Plug the {side} wheel back in (the arm stays unplugged), then ENTER...")
        robot.setup_base_motor(name, configured=tuple(signs))
        while True:
            input(f"Robot lifted? ENTER to spin the {side} wheel for a second...")
            robot.spin_wheel(name)
            answer = input("Did it turn in the direction that drives the robot FORWARD? [y/n/r=repeat] ")
            if answer.strip().lower() == "r":
                continue
            signs[name] = 1 if answer.strip().lower() == "y" else -1
            break
    found = {name: robot.bus.ping(name) for name in WHEELS}
    print("Servos responding on the chain (model numbers):", found)
    print("Saved wheel directions to", robot.save_base_signs(signs), signs)
finally:
    if robot.bus.is_connected:
        robot.bus.disconnect(disable_torque=False)  # the arm's motors aren't on the chain here
