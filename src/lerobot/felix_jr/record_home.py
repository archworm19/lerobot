"""Record the current pose of an SO-101 follower as its "home" pose.

Usage: uv run python -m lerobot.felix_jr.record_home <PORT> <ROBOT_ID>
"""

import sys

from lerobot.felix_jr.home import HOME_TOLERANCE, deviations, load_home, make_follower, save_home

port, robot_id = sys.argv[1], sys.argv[2]

robot = make_follower(port, robot_id)
robot.connect(calibrate=False)  # use the saved calibration
try:
    robot.bus.disable_torque()  # let the user pose the arm by hand
    input("Torque off. Put the arm in its home pose (support it), then press ENTER...")
    pose = robot.get_observation()  # get the current pose in degrees
    print("Home pose:", {k: round(v, 1) for k, v in pose.items()})

    old = load_home(robot)
    if old is not None:
        moved = {k: round(d, 1) for k, d in deviations(pose, old).items() if abs(d) > HOME_TOLERANCE}
        if moved:
            print(f"WARNING: differs from saved home by more than {HOME_TOLERANCE} deg: {moved}")
        if input("Overwrite existing home? [y/N] ").strip().lower() != "y":
            sys.exit("Not saved.")
    print("Saved to", save_home(robot, pose))
finally:
    robot.disconnect()
