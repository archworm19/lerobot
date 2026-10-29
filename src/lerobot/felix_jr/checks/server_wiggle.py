"""The wiggle test, driven from your computer through the robot server (the arms are on the Pi).

  uv run python -m lerobot.felix_jr.checks.server_wiggle tcp://<pi>:5555 [--arms left_arm right_arm] [--delta DEG]
      [--pause SEC] [--max-vel DEG/S] [--max-accel DEG/S2] [--tol DEG] [--timeout SEC] [--no-head] [--no-base]
      [--head-delta DEG] [--wheel-speed DEG/S] [--wheel-seconds S] [--capture-dir DIR]

The same sequence as `checks.wiggle_test`: every chosen arm (default: all) moves to center together, each joint
is nudged +/- DEG, the arms return home; then the head servos (home, nudge, home) and the wheels (each alone, then
both). Start the server on the Pi first, with the arms' ports, and make sure each arm has a recorded home
(`setup.record_home`). Stop other scripts that use the arms' serial ports; the server must be their only user.

It runs inside a *session*: when the script ends in any way (normally, Ctrl+C, a crash) the robot stops and goes
to rest, and if this computer dies or loses the network the server sends the robot to rest by itself a few seconds
after the heartbeats stop. The wheels get a prompt first: lift the robot so they can spin freely.

With --capture-dir it also saves a camera frame after every step (taken after the move finished) with the joint
state in `log.jsonl`, which shows the camera and the arm motion working together.
"""

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

import cv2

from lerobot.felix_jr.server.client import FelixClient
from lerobot.felix_jr.server.protocol import DEFAULT_PORT

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CENTER = {f"{j}.pos": 0.0 for j in JOINTS}  # 0 deg = middle of the calibrated range
CENTER["gripper.pos"] = 50.0  # the gripper is 0-100
HOME_TOLERANCE = 15.0  # warn if an arm starts farther than this from its home


class Recorder:
    """Saves a camera frame and the joint state after each step (does nothing without a directory)."""

    def __init__(self, client: FelixClient, directory: Path | None):
        """Frames go to `directory`; None switches recording off."""
        self.client, self.directory, self.count = client, directory, 0
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)

    def snap(self, label: str, finished_at: float | None) -> None:
        """Capture a frame taken after server time `finished_at` and save it as `NN_<label>.jpg`."""
        if self.directory is None:
            return
        self.count += 1
        capture = self.client.capture(color_format="jpeg", after=finished_at)
        name = f"{self.count:02d}_{label.replace(' ', '_').replace('.', '').replace('+', 'p')}.jpg"
        cv2.imwrite(str(self.directory / name), cv2.cvtColor(capture.color, cv2.COLOR_RGB2BGR))
        with (self.directory / "log.jsonl").open("a") as f:
            f.write(
                json.dumps({"step": self.count, "label": label, "image": name, "state": capture.state}) + "\n"
            )


def wiggle_arms(
    client: FelixClient,
    arms: list[str],
    delta: float,
    pause: float,
    move_kw: dict[str, float],
    recorder: Recorder,
    say: Callable[[str], None],
) -> bool:
    """Center, then each joint +/- `delta`, on all `arms` together. False if a move did not complete."""

    def move(label: str, pose: dict[str, float]) -> bool:
        status = client.move_joints(dict.fromkeys(arms, pose), **move_kw)
        if status["state"] != "done":
            say(f"{label}: the motion ended as {status['state']} ({status['reason']})")
            return False
        if not status["result"]["reached"]:
            say(f"{label}: WARNING: the arms did not settle within the tolerance")
        recorder.snap(label, status["finished_at"])
        return True

    if not move("center", CENTER):
        return False
    for joint in JOINTS:
        key = f"{joint}.pos"
        for sign in (+1, -1):
            target = {**CENTER, key: CENTER[key] + sign * delta}
            say(f"{key}: {target[key]:+.1f}")
            if not move(f"{key} {target[key]:+.0f}", target):
                return False
            time.sleep(pause)
            state = client.get_state()
            say("   read back: " + str({arm: f"{state[arm][key]:+.1f}" for arm in arms}))
        if not move("center", CENTER):
            return False
    return True


def wiggle_head(
    client: FelixClient, head_delta: float, recorder: Recorder, say: Callable[[str], None]
) -> None:
    """Each head servo: go to its recorded home (if any), nudge both ways, and end at home."""
    for name, servo in client.head_state().items():
        home = servo["home"]
        try:
            if home is not None:
                say(f"{name}: going to home")
                client.head_move(name, to="home")
            else:
                say(
                    f"WARNING: no home recorded for {name} (run setup.head_range); nudging it from where it is"
                )
            result = client.head_wiggle(name, head_delta)
            positive, negative = ("down", "up") if name == "head_eyes" else ("left", "right")
            say(
                f"{name}: asked +/-{head_delta:.0f} deg, moved {result['to_pos']:+.1f} deg {positive} "
                f"and {result['to_neg']:+.1f} deg {negative}"
            )
            if result["limited"]:
                say(f"{name}: a move was shortened by the recorded limit")
            if result["blocked"] or (
                not result["limited"] and min(result["to_pos"], result["to_neg"]) < 0.3 * head_delta
            ):
                say(f"WARNING: {name} moved much less than commanded (blocked or no power?)")
        finally:
            if home is not None:
                client.head_move(name, to="home")
                say(f"{name}: back at home")
    recorder.snap("head", None)


def exercise_wheels(
    client: FelixClient, speed: float, seconds: float, say: Callable[[str], None], ask: Callable[[str], str]
) -> None:
    """Each wheel alone, then both, forward and back; compares commanded and measured speeds."""
    if ask("Wheels: robot lifted? ENTER to spin them freely, 's' to skip: ").strip().lower() == "s":
        return
    steps = [
        ("left wheel forward", speed, 0.0),
        ("left wheel backward", -speed, 0.0),
        ("right wheel forward", 0.0, speed),
        ("right wheel backward", 0.0, -speed),
        ("both forward", speed, speed),
        ("both backward", -speed, -speed),
    ]
    for label, left, right in steps:
        status = client.base_velocity(left, right, seconds)
        if status["state"] != "done":
            say(f"{label}: the motion ended as {status['state']} ({status['reason']})")
            return
        measured_left, measured_right = status["result"]["measured"]
        say(
            f"{label:22s} commanded L{left:+.0f} R{right:+.0f} deg/s, "
            f"measured L{measured_left:+.0f} R{measured_right:+.0f}"
        )
        for side, cmd, meas in (("left", left, measured_left), ("right", right, measured_right)):
            if cmd and meas * cmd < 0.3 * cmd * cmd:
                say(f"WARNING: {side} wheel did not follow the command (direction/power?)")


def run(
    client: FelixClient,
    args: argparse.Namespace,
    ask: Callable[[str], str] = input,
    say: Callable[[str], None] = print,
) -> bool:
    """Run the whole sequence in a session. Returns False if it was cut short."""
    info = client.server_info
    if not info.get("motion"):
        raise SystemExit("this server has no motion commands; update and restart it")
    arms = args.arms or info["arms"]
    unknown = [arm for arm in arms if arm not in info["arms"]]
    if unknown or not arms:
        raise SystemExit(f"arms {unknown or arms} not on this server (it has {info['arms']})")
    homes = client.get_homes()
    missing = [arm for arm in arms if arm not in homes]
    if missing:
        raise SystemExit(f"no recorded home for {missing}: run setup.record_home on the robot first")

    state = client.get_state()
    for arm in arms:
        say(f"[{arm}] Start pose: " + str({k: round(v, 1) for k, v in state[arm].items()}))
        far = {
            k: round(state[arm][k] - v, 1)
            for k, v in homes[arm].items()
            if abs(state[arm][k] - v) > HOME_TOLERANCE
        }
        if far:
            say(f"[{arm}] WARNING: not near saved home (off by more than {HOME_TOLERANCE} deg): {far}")
    ask("Clear the workspace, hold the arm(s) if needed. ENTER to move to center...")

    move_kw = {"max_vel": args.max_vel, "max_accel": args.max_accel, "tol": args.tol, "timeout": args.timeout}
    recorder = Recorder(client, args.capture_dir)
    completed = False
    with client.session():  # on any exit: stop, then go to rest
        completed = wiggle_arms(client, arms, args.delta, args.pause, move_kw, recorder, say)
        if completed:
            say("Returning to home.")
            status = client.home(arms=arms)
            recorder.snap("home", status["finished_at"])
            if info["head"] and not args.no_head:
                wiggle_head(client, args.head_delta, recorder, say)
            if info["base"] and not args.no_base:
                exercise_wheels(client, args.wheel_speed, args.wheel_seconds, say, ask)
    say("Done; the robot is at rest." if completed else "Stopped early; the robot was sent to rest.")
    return completed


def main() -> None:
    """Command line entry point; see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("endpoint", nargs="?", default=f"tcp://localhost:{DEFAULT_PORT}")
    parser.add_argument(
        "--arms", nargs="+", metavar="ID", help="arm ids to wiggle (default: all on the server)"
    )
    parser.add_argument("--delta", type=float, default=10.0, help="nudge size in degrees (10)")
    parser.add_argument("--pause", type=float, default=0.3, help="seconds to hold each nudge (0.3)")
    parser.add_argument("--max-vel", type=float, default=60.0, help="cruise speed limit, deg/s (60)")
    parser.add_argument("--max-accel", type=float, default=120.0, help="accel limit, deg/s^2 (120)")
    parser.add_argument("--tol", type=float, default=1.5, help="deg from target that counts as arrived (1.5)")
    parser.add_argument(
        "--timeout", type=float, default=10.0, help="max seconds to settle after each move (10)"
    )
    parser.add_argument("--no-head", action="store_true", help="skip the head servos")
    parser.add_argument("--no-base", action="store_true", help="skip the wheels")
    parser.add_argument("--head-delta", type=float, default=8.0, help="head nudge in degrees (8)")
    parser.add_argument("--wheel-speed", type=float, default=40.0, help="wheel speed in deg/s (40)")
    parser.add_argument("--wheel-seconds", type=float, default=1.0, help="seconds per wheel step (1)")
    parser.add_argument("--capture-dir", type=Path, help="save a camera frame after each step here")
    args = parser.parse_args()
    try:
        with FelixClient(args.endpoint, timeout_s=15.0) as client:
            run(client, args)
    except KeyboardInterrupt:
        print("\nCtrl+C: stopped; the robot was sent to rest.")
        sys.exit(130)


if __name__ == "__main__":
    main()
