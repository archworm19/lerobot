"""Teleoperate an SO-101 follower with a 3Dconnexion SpaceMouse.

The puck deflection is a velocity: each tick it is integrated into a target end-effector pose (position +
orientation, in the robot base frame), which inverse kinematics turns into joint targets. The SO-101 has only
five joints before the gripper, so it can't follow every combination of the 6 puck axes: x/y/z move the tip,
pitch tilts the tool in the arm's plane, roll twists the tool about its axis and yaw swings the whole arm
about the shoulder pan. The arm first moves to its center pose (ENTER to confirm). The gripper opens and
closes continuously (0-100): hold the right/left puck button, or press the Up/Down arrow keys in the terminal.
Ctrl+C returns the arm to its saved home pose and disconnects.

Needs: pyspacemouse + the hidapi system library (macOS: `brew install hidapi`), placo (the `placo-dep` extra),
and the SO-101 URDF with its meshes (git clone of TheRobotStudio/SO-ARM100, Simulation/SO101).

Usage: uv run python -m lerobot.felix_jr.teleop.spacemouse_teleop <PORT> [ROBOT_ID] [options]
  PORT                    serial port of the arm
  ROBOT_ID                arm id (default: left_arm)
  --urdf PATH             SO-101 URDF, with its assets/ folder next to it
  --lin-speed M/S         tip speed at full puck push (default: 0.05)
  --ang-speed RAD/S       tool rotation speed at full puck twist (default: 0.6)
  --deadband F            ignore puck deflection below this, 0-1 (default: 0.08)
  --rate HZ               control loop rate (default: 30)
  --reach M               tip may move at most this far from its start point (default: 0.15)
  --orientation-weight W  IK weight on tool orientation vs position, 0-1 (default: 0.5)
  --max-step DEG          lerobot per-command cap: goal stays within this of the measured position (5)
  --max-joint-speed DEG/S limit on every joint's speed (default: 60)
  --tol DEG               how close to center counts as arrived for the initial move (default: 4)
  --gripper-step N        gripper units (0-100) per Up/Down arrow press (default: 5)
  --invert AXES           comma-separated puck axes to flip, from x,y,z,roll,pitch,yaw (e.g. y,yaw)
  --log CSV               write per-tick commanded vs measured joints/tip to this CSV
  --sim                   no hardware: run a scripted push through the IK and print the result

Tips for linking to underlying libraries on macOS and killing 3xconnexion's background process:
export DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH"
launchctl unload /Library/LaunchAgents/com.3dconnexion.nlupdate.plist 2>/dev/null
pkill -f 3Dconnexion

"""

import argparse
import csv
import ctypes
import ctypes.util
import logging
import os
import select
import signal
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np

from lerobot.felix_jr.home import CENTER, JOINTS, load_home, make_follower, move_all, remember_goal
from lerobot.model.kinematics import RobotKinematics

ARM_JOINTS = [j for j in JOINTS if j != "gripper"]
DEFAULT_URDF = "~/.cache/felix_jr/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
SIM_START_DEG = np.array([0.0, -30.0, 40.0, 50.0, 0.0])  # --sim only: a comfortable bent-arm pose
AXES = ["x", "y", "z", "roll", "pitch", "yaw"]
UNREACHABLE_M = 0.02  # if IK misses the position target by more than this, drop the target to the arm
MAX_IK_JUMP_DEG = 20.0  # reject a tick whose IK solution moves any joint further than this (singularity)
GRIPPER_RATE = 40.0  # gripper units/s while a button is held


def preload_hidapi() -> None:
    """Load the hidapi system library into the process before pyspacemouse imports it.

    pyspacemouse's HID layer (easyhid) calls `dlopen` by name, which on macOS misses Homebrew's
    /opt/homebrew/lib and then silently falls back to the global namespace, giving a misleading
    "HID API is probably not installed". Loading it first (globally) makes its symbols visible.
    """
    candidates = [ctypes.util.find_library("hidapi"), "/opt/homebrew/lib/libhidapi.dylib"]
    candidates += ["/usr/local/lib/libhidapi.dylib", "libhidapi-hidraw.so.0"]
    for path in filter(None, candidates):
        try:
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            return
        except OSError:
            continue


def deadband(value: float, band: float) -> float:
    """Zero inside +/-band, then rescale so the output still reaches +/-1."""
    if abs(value) <= band:
        return 0.0
    return float(np.sign(value) * (abs(value) - band) / (1.0 - band))


def exp_so3(w: np.ndarray) -> np.ndarray:
    """Rotation matrix for the rotation vector `w` (axis * angle, radians)."""
    theta = float(np.linalg.norm(w))
    if theta < 1e-9:
        return np.eye(3)
    k = w / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


def pan_axis_xy(kin: RobotKinematics, q: np.ndarray) -> np.ndarray:
    """(x, y) of the shoulder-pan axis in the base frame, from the circle the tip sweeps as the pan turns.

    The URDF base frame isn't centered on the pan axis (it is ~4 cm off in x), so swinging the tip about
    the base origin would not match what the pan joint does.
    """
    pts = []
    for pan in (-25.0, 0.0, 25.0):
        q_pan = q.copy()
        q_pan[0] = pan
        pts.append(kin.forward_kinematics(q_pan)[:2, 3])
    a, b, c = pts
    return np.linalg.solve(2 * np.array([b - a, c - a]), np.array([b @ b - a @ a, c @ c - a @ a]))


def integrate_target(
    pos_target: np.ndarray,
    rot_current: np.ndarray,
    pan_deg: float,
    pan_axis: np.ndarray,
    origin: np.ndarray,
    axes: dict[str, float],
    dt: float,
    lin_speed: float,
    ang_speed: float,
    reach: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance the target by one tick of the (already deadbanded) puck deflection.

    Returns (position target, 3x3 orientation target), both in the robot base frame.
    - Translation: x/y/z push moves the position target, kept inside a sphere of radius `reach`.
    - Orientation: only the rotations a 5-joint arm can follow are used. Puck pitch rotates about the
      horizontal axis perpendicular to the arm plane (the wrist_flex axis, which turns with the shoulder
      pan); puck roll twists about the tool's own axis (wrist_roll); puck yaw swings the whole arm about the
      shoulder-pan axis (the position target and the tool heading turn together).
    """
    pos = pos_target + lin_speed * dt * np.array([axes["x"], axes["y"], axes["z"]])
    # yaw: swing about the vertical line through the shoulder-pan axis
    swing = exp_so3(np.array([0.0, 0.0, ang_speed * dt * axes["yaw"]]))
    pivot = np.array([pan_axis[0], pan_axis[1], 0.0])
    pos = pivot + swing @ (pos - pivot)
    rot_current = swing @ rot_current
    offset = pos - origin
    if np.linalg.norm(offset) > reach:  # stay inside a sphere around where we started
        pos = origin + offset * reach / np.linalg.norm(offset)
    pan = np.deg2rad(pan_deg)
    pitch_axis = np.array([-np.sin(pan), np.cos(pan), 0.0])
    rot = exp_so3(pitch_axis * ang_speed * dt * axes["pitch"]) @ rot_current
    rot = rot @ exp_so3(np.array([0.0, 0.0, ang_speed * dt * axes["roll"]]))
    return pos, rot


def step_joints(
    kin: RobotKinematics,
    q_cmd: np.ndarray,
    pos_target: np.ndarray,
    pan_axis: np.ndarray,
    origin: np.ndarray,
    axes: dict[str, float],
    dt: float,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """One control tick: returns (new joint command, new position target, whether IK was rejected).

    The position target accumulates, so wrist moves (which shift the tip) are compensated by the other
    joints. The orientation target is re-anchored on the commanded pose each tick and only ever asks for
    reachable rotations, so it can't wind up. If the position target turns out unreachable, it is pulled
    back to where the arm is.
    """
    t_cur = kin.forward_kinematics(q_cmd)
    pos, rot = integrate_target(
        pos_target, t_cur[:3, :3], q_cmd[0], pan_axis, origin, axes, dt,
        args.lin_speed, args.ang_speed, args.reach,
    )
    target = np.eye(4)
    target[:3, :3], target[:3, 3] = rot, pos
    q_new = kin.inverse_kinematics(q_cmd, target, orientation_weight=args.orientation_weight)
    if np.max(np.abs(q_new - q_cmd)) > MAX_IK_JUMP_DEG:  # near a singularity: ignore this tick
        return q_cmd, t_cur[:3, 3].copy(), True
    # Joint speed limit. Near a folded-up pose a small tip motion needs a big joint motion; shrink the
    # whole step (all joints together, so the path keeps its shape) and drop the part of the target that
    # we couldn't follow, so it can't pile up and lurch later.
    max_delta = args.max_joint_speed * dt
    biggest = float(np.max(np.abs(q_new - q_cmd)))
    if biggest > max_delta:
        q_new = q_cmd + (q_new - q_cmd) * (max_delta / biggest)
        return q_new, kin.forward_kinematics(q_new)[:3, 3].copy(), False
    reached = kin.forward_kinematics(q_new)[:3, 3]
    if np.linalg.norm(reached - pos) > UNREACHABLE_M:
        pos = reached.copy()  # target is outside the workspace: don't let it run away
    return q_new, pos, False


def read_axes(state, band: float, invert: set[str]) -> dict[str, float]:
    return {a: (-1 if a in invert else 1) * deadband(float(getattr(state, a)), band) for a in AXES}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("port", help="serial port of the arm")
    p.add_argument("robot_id", nargs="?", default="left_arm", help="arm id (default: left_arm)")
    p.add_argument("--urdf", default=DEFAULT_URDF, help="SO-101 URDF (with its assets/ folder next to it)")
    p.add_argument("--lin-speed", type=float, default=0.05, help="m/s at full puck push (default: 0.05)")
    p.add_argument("--ang-speed", type=float, default=0.6, help="rad/s at full puck twist (default: 0.6)")
    p.add_argument("--deadband", type=float, default=0.08, help="ignore puck deflection below this")
    p.add_argument("--rate", type=float, default=30.0, help="control loop Hz (default: 30)")
    p.add_argument("--reach", type=float, default=0.15, help="max distance from the start point, m")
    p.add_argument("--orientation-weight", type=float, default=0.5, help="IK orientation weight (0-1)")
    p.add_argument("--tol", type=float, default=4.0, help="deg from center that counts as arrived")
    p.add_argument("--log", metavar="CSV", help="write per-tick commanded vs measured joints/tip to this CSV")
    p.add_argument(
        "--max-step", type=float, default=5.0,
        help="lerobot's per-command cap: the goal is kept within this many deg of the measured position",
    )
    p.add_argument("--max-joint-speed", type=float, default=60.0, help="deg/s limit on every joint")
    p.add_argument("--gripper-step", type=float, default=5.0, help="gripper units per Up/Down key press")
    p.add_argument("--invert", default="", help="comma-separated axes to flip, e.g. y,yaw")
    p.add_argument("--sim", action="store_true", help="no hardware: run a scripted push through IK")
    args = p.parse_args()
    logging.getLogger("lerobot.robots.utils").setLevel(logging.ERROR)  # clamp warning fires every tick
    invert = {a for a in args.invert.split(",") if a}
    if invert - set(AXES):
        p.error(f"--invert axes must be among {AXES}")
    urdf = str(Path(args.urdf).expanduser())
    kin = RobotKinematics(urdf, "gripper_frame_link", ARM_JOINTS)
    dt = 1.0 / args.rate

    if args.sim:
        # Drive a scripted push along +x and +z to exercise FK/IK without hardware.
        q_cmd = SIM_START_DEG.copy()
        origin = kin.forward_kinematics(q_cmd)[:3, 3].copy()
        pan_axis = pan_axis_xy(kin, q_cmd)
        pos_target = origin.copy()
        rejected = 0
        for i in range(int(2 * args.rate)):
            axes = dict.fromkeys(AXES, 0.0) | {"x": 1.0, "z": 0.5 if i > args.rate else 0.0}
            q_cmd, pos_target, was_rejected = step_joints(
                kin, q_cmd, pos_target, pan_axis, origin, axes, dt, args
            )
            rejected += was_rejected
        moved = np.round(kin.forward_kinematics(q_cmd)[:3, 3] - origin, 3)
        print(f"sim: joints {np.round(q_cmd, 1)}, tip moved {moved} m, rejected ticks: {rejected}")
        return

    preload_hidapi()
    import pyspacemouse  # imported late: needs the hidapi system library

    # Open the mouse before touching the arm, so a missing mouse/driver can't leave the arm moving.
    with pyspacemouse.open(nonblocking=True, axis_convention=pyspacemouse.AxisConvention.ROS) as mouse:
        run_teleop(mouse, kin, args, invert, dt)


class KeyReader:
    """Non-blocking up/down arrow reader for the terminal running the script.

    Puts the terminal in cbreak mode (no echo, no line buffering) while active. Ctrl+C still raises
    KeyboardInterrupt, and the original terminal settings are restored on exit. If stdin is not a
    terminal, polling just returns no keys. Only works while this terminal window has focus.
    """

    def __enter__(self) -> "KeyReader":
        self.fd = sys.stdin.fileno()
        self.active = sys.stdin.isatty()
        if self.active:
            self.saved = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc) -> None:
        if self.active:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)

    def poll(self) -> int:
        """Net number of Up minus Down presses since the last call."""
        net = 0
        while self.active and select.select([self.fd], [], [], 0)[0]:
            data = os.read(self.fd, 256)
            net += data.count(b"\x1b[A") + data.count(b"\x1bOA")  # Up (normal / application mode)
            net -= data.count(b"\x1b[B") + data.count(b"\x1bOB")  # Down
        return net


LOG_COLUMNS = (
    ["t", "tick_s"]
    + [f"puck_{a}" for a in AXES]
    + [f"q_cmd_{j}" for j in ARM_JOINTS]
    + [f"q_meas_{j}" for j in ARM_JOINTS]
    + [f"tip_cmd_{c}" for c in "xyz"]
    + [f"tip_meas_{c}" for c in "xyz"]
    + [f"tip_target_{c}" for c in "xyz"]
    + ["gripper_cmd"]
)


def save_log(rows: list[list[float]], path: str, dt: float) -> None:
    """Write the per-tick log as CSV and print a short tracking summary."""
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(LOG_COLUMNS)
        writer.writerows(rows)
    data = np.array(rows)
    col = {name: i for i, name in enumerate(LOG_COLUMNS)}
    err = np.stack([data[:, col[f"tip_meas_{c}"]] - data[:, col[f"tip_cmd_{c}"]] for c in "xyz"], axis=1)
    print(f"Saved {len(rows)} ticks to {path}")
    tick_ms = 1000 * data[:, col["tick_s"]]
    print(f"  tick period: mean {tick_ms.mean():.1f} ms, max {tick_ms.max():.1f} ms (goal {1000 * dt:.1f})")
    for i, c in enumerate("xyz"):
        mean_mm, max_mm = 1000 * err[:, i].mean(), 1000 * np.abs(err[:, i]).max()
        print(f"  tip {c}: measured - commanded  mean {mean_mm:+.1f} mm, max |err| {max_mm:.1f} mm")


def run_teleop(mouse, kin: RobotKinematics, args: argparse.Namespace, invert: set[str], dt: float) -> None:
    robot = make_follower(
        args.port, args.robot_id, max_relative_target=args.max_step, with_base=False, with_head=False
    )
    robot.connect(calibrate=False)
    home = load_home(robot)
    if home is None:
        robot.disconnect()
        raise SystemExit(f"No home pose for {args.robot_id}. Run lerobot.felix_jr.setup.record_home first.")

    log_rows: list[list[float]] = []
    try:
        input("Clear the workspace, hold the arm if needed. ENTER to move to center...")
        move_all({robot.id: robot}, {robot.id: CENTER}, tol=args.tol)
        # Start from the commanded center, not the measured pose: the arm rests a few degrees off its
        # goal (gravity), and starting from the measured pose would snap the setpoint and jerk the arm.
        q_cmd = np.array([CENTER[f"{j}.pos"] for j in ARM_JOINTS], dtype=float)
        gripper = CENTER["gripper.pos"]
        origin = kin.forward_kinematics(q_cmd)[:3, 3].copy()
        pan_axis = pan_axis_xy(kin, q_cmd)
        pos_target = origin.copy()
        print(f"Start tip position (m): {np.round(origin, 3)}. Push/twist the puck; Ctrl+C returns home.")
        print("Gripper: puck buttons (left closes, right opens) or the Up/Down arrow keys in this terminal.")

        t_start = last_tick = next_tick = time.perf_counter()
        with KeyReader() as keys:
            while True:
                state = mouse.read()
                axes = read_axes(state, args.deadband, invert)
                buttons = list(state.buttons)
                close, open_ = (buttons[0] if buttons else 0), (buttons[1] if len(buttons) > 1 else 0)
                gripper += GRIPPER_RATE * dt * (open_ - close) + args.gripper_step * keys.poll()
                gripper = float(np.clip(gripper, 0.0, 100.0))

                q_cmd, pos_target, _ = step_joints(kin, q_cmd, pos_target, pan_axis, origin, axes, dt, args)
                action = {f"{j}.pos": float(q) for j, q in zip(ARM_JOINTS, q_cmd, strict=True)}
                action["gripper.pos"] = gripper
                robot.send_action(action)
                remember_goal(robot.id, action)

                if args.log:  # measured joints after this command, and the tip they put the gripper at
                    obs = robot.get_observation()
                    q_meas = np.array([obs[f"{j}.pos"] for j in ARM_JOINTS], dtype=float)
                    now = time.perf_counter()
                    log_rows.append(
                        [now - t_start, now - last_tick]
                        + [axes[a] for a in AXES]
                        + list(q_cmd)
                        + list(q_meas)
                        + list(kin.forward_kinematics(q_cmd)[:3, 3])
                        + list(kin.forward_kinematics(q_meas)[:3, 3])
                        + list(pos_target)
                        + [gripper]
                    )
                    last_tick = now

                next_tick += dt
                time.sleep(max(0.0, next_tick - time.perf_counter()))
    except KeyboardInterrupt:
        print("\nCtrl+C: returning to home.")
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if args.log:
            save_log(log_rows, args.log, dt)
        try:
            move_all({robot.id: robot}, {robot.id: home})
        except Exception as e:  # e.g. the arm's USB serial port vanished
            print(f"Could not return to home: {e!r}")
        try:
            robot.disconnect()
        except Exception as e:
            print(f"disconnect failed ({e!r}); check power/USB, torque may still be on.")


if __name__ == "__main__":
    main()
