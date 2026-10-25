"""One-time setup of the head servos: assign their IDs and record which raw direction is down / left.

The head has two servos: the EYES (ID 8) tilt the RealSense up and down, and the NECK (ID 7) turns the head
left and right. Positive means DOWN for the eyes and LEFT for the neck, the same as the joint conventions
in `xlerobot.urdf` (`head_tilt_joint`: positive pitches the camera down; `head_pan_joint`: positive turns
it counter-clockwise seen from above).

Wiring assumed: controller board (the left arm's) -> EYES servo -> NECK servo. Servos leave the factory as
ID 1, which collides with the arm (shoulder_pan is ID 1) and with each other, so the first one is set up
alone:
  1. Unplug the arm's cable from the board, and unplug the NECK from the EYES. The EYES (next to the board)
     are set up first, alone.
  2. Reconnect the NECK to the EYES (the arm stays off). The eyes now have ID 8, so the neck is found by
     scanning the chain for the one servo that isn't the eyes.
  3. Plug the arm back in when the script finishes.
Each servo then makes a small move (about 7 degrees) and you say whether it went down / left, which fixes
the sign of its raw direction. The range of motion is not calibrated by this script.

Usage: uv run python -m lerobot.felix_jr.setup.head <PORT> <ROBOT_ID>
  PORT / ROBOT_ID: the arm whose board the head is plugged into (here the left arm: its port and left_arm);
  the head directions are saved with that arm's files (`<id>_head.json`).
"""

import sys
from collections.abc import Callable

from lerobot.felix_jr.home import make_follower

# In daisy-chain order from the board: the eyes are first and are set up alone.
# name -> (what you see move, the direction that counts as positive)
HEAD = {
    "head_eyes": ("camera", "DOWN"),
    "head_neck": ("head", "LEFT (counter-clockwise seen from above)"),
}


def run(robot, ask: Callable[[str], str] = input, say: Callable[..., None] = print) -> dict[str, int]:
    """Walk through the setup for `robot` (a follower made with `with_head=True`). Returns the signs: for
    each servo, +1 if its raw positive direction is down (eyes) / left (neck), otherwise -1."""
    signs: dict[str, int] = {}
    for n, (name, (mover, direction)) in enumerate(HEAD.items()):
        label = name.removeprefix("head_").upper()
        if n == 0:
            ask(
                f"Unplug the arm from the board and the NECK from the EYES, so the board talks to the "
                f"{label} servo ONLY, then ENTER..."
            )
        else:
            ask(f"Plug the {label} servo back in (the arm stays unplugged), then ENTER...")
        robot.setup_head_motor(name, configured=tuple(signs))
        while True:
            ask(f"Make sure the {mover} can move a few degrees freely, then ENTER to move the {label}...")
            used = robot.nudge_head(name)  # raw direction that was actually used, +1 or -1
            answer = ask(f"Did the {mover} move {direction}? [y/n/r=repeat] ").strip().lower()
            if answer == "r":
                continue
            signs[name] = used if answer == "y" else -used
            break
    found = {name: robot.bus.ping(name) for name in HEAD}
    say("Servos responding on the chain (model numbers):", found)
    say("Saved head directions to", robot.save_head_signs(signs), signs)
    return signs


def main() -> None:
    port, robot_id = sys.argv[1], sys.argv[2]
    robot = make_follower(port, robot_id, with_base=False, with_head=True)
    try:
        run(robot)
    finally:
        if robot.bus.is_connected:
            robot.bus.disconnect(disable_torque=False)  # the arm's motors aren't on the chain here


if __name__ == "__main__":
    main()
