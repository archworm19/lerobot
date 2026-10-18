"""Load test for the shoulder_lift servo: does it hold up when lifting the arm against gravity?

Moves the arm to center, then drives only shoulder_lift: down by --down degrees (with gravity), up past
center by --up degrees (against gravity), and back to 0. All other joints stay at their center command.
While it moves, the lift servo's position, load, current, voltage and temperature are sampled each tick.
Per phase it prints the peak load/current, the lowest voltage and how far the joint trailed its command, so
you can tell a torque-limited or under-powered shoulder (load pinned high, voltage sagging, position
stalled) from one that just needs a stiffer position gain.

Usage: uv run python -m lerobot.felix_jr.diagnostics.lift_test <PORT> [ROBOT_ID] [--down DEG] [--up DEG]
       [--p-lift N] [--max-step DEG] [--log CSV]
"""

import argparse
import csv
import signal
import time

import numpy as np

from lerobot.felix_jr.home import (
    CENTER,
    DEFAULT_MAX_ACCEL,
    DEFAULT_MAX_VEL,
    load_home,
    make_follower,
    move_all,
    move_time_trap,
    remember_goal,
    schedule_trapezoid,
    trapezoid_offset,
)

JOINT = "shoulder_lift"
REGISTERS = [
    "Present_Load", "Present_Current", "Present_Voltage", "Present_Temperature", "Torque_Limit", "Status"
]
COLUMNS = [
    "t", "phase", "lift_cmd", "lift_meas", "load_pct", "current_ma",
    "voltage_v", "temp_c", "torque_limit", "status",
]
RATE_HZ = 30.0
SETTLE_DEG = 3.0  # "arrived" for the recovery-time measurement

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("port", help="serial port of the arm")
parser.add_argument("robot_id", nargs="?", default="left_arm", help="arm id (default: left_arm)")
parser.add_argument("--down", type=float, default=18.0, help="deg to move shoulder_lift positive (18)")
parser.add_argument("--up", type=float, default=14.0, help="deg to move it negative, against gravity (14)")
parser.add_argument("--hold", type=float, default=2.0, help="seconds to hold after each move (default: 2)")
parser.add_argument("--max-vel", type=float, default=DEFAULT_MAX_VEL, help="deg/s (default: %(default)s)")
parser.add_argument("--max-accel", type=float, default=DEFAULT_MAX_ACCEL, help="deg/s^2 (%(default)s)")
parser.add_argument("--tol", type=float, default=4.0, help="deg from center that counts as arrived")
parser.add_argument("--p-lift", type=int, help="shoulder_lift P gain for this run (default 16)")
parser.add_argument("--max-step", type=float, default=5.0, help="lerobot per-command cap in deg (default: 5)")
parser.add_argument("--log", metavar="CSV", help="write every sample to this CSV")
args = parser.parse_args()

robot = make_follower(args.port, args.robot_id, max_relative_target=args.max_step, with_base=False)
robot.connect(calibrate=False)
home = load_home(robot)
if home is None:
    robot.disconnect()
    raise SystemExit(f"No home pose for {args.robot_id}. Run lerobot.felix_jr.setup.record_home first.")

if args.p_lift is not None:
    # P_Coefficient is stored in EEPROM, so torque has to be off to write it; the next normal connect
    # writes the configured value (16) again.
    with robot.bus.torque_disabled([JOINT]):
        robot.bus.write("P_Coefficient", JOINT, args.p_lift)
    print(f"{JOINT} P_Coefficient set to {robot.bus.read('P_Coefficient', JOINT)} for this run")

samples: list[list] = []
t_start = 0.0


def sample(phase: str, lift_cmd: float) -> None:
    """Read the lift servo's status registers and record one row."""
    raw = {r: float(robot.bus.sync_read(r, [JOINT], normalize=False)[JOINT]) for r in REGISTERS}
    lift_meas = float(robot.get_observation()[f"{JOINT}.pos"])
    samples.append(
        [
            time.perf_counter() - t_start,
            phase,
            lift_cmd,
            lift_meas,
            raw["Present_Load"] / 10.0,  # 0.1 % units, signed by direction
            raw["Present_Current"] * 6.5,  # 6.5 mA per step
            raw["Present_Voltage"] / 10.0,  # 0.1 V units
            raw["Present_Temperature"],
            raw["Torque_Limit"],  # RAM limit (1000 = 100%); firmware lowers it when overload protection trips
            raw["Status"],  # error flags; bit 0x20 = overload
        ]
    )


def run_phase(label: str, start: float, end: float) -> float:
    """Drive shoulder_lift from `start` to `end` on a trapezoid, then hold, sampling every tick."""
    disp = end - start
    total = move_time_trap(abs(disp), args.max_vel, args.max_accel)
    t_accel, v_peak = schedule_trapezoid(disp, total, args.max_accel)
    t0 = time.perf_counter()
    next_tick = t0
    while (t := time.perf_counter() - t0) < total + args.hold:
        lift_cmd = start + trapezoid_offset(t, disp, total, t_accel, v_peak, args.max_accel)
        action = {**CENTER, f"{JOINT}.pos": lift_cmd}
        robot.send_action(action)
        remember_goal(robot.id, action)
        sample(label, lift_cmd)
        next_tick += 1.0 / RATE_HZ
        time.sleep(max(0.0, next_tick - time.perf_counter()))
    return end


def summarize(label: str, rows: list[list]) -> None:
    data = np.array([r[2:] for r in rows], dtype=float)  # cmd meas load cur volt temp limit status
    err = data[:, 1] - data[:, 0]
    cmd_end = data[-1, 0]
    # recovery: after the command stops changing, how long until the joint is within SETTLE_DEG?
    moving = np.abs(data[:, 0] - cmd_end) > 1e-3
    last_move = int(np.max(np.where(moving))) if moving.any() else 0
    after = np.abs(err[last_move:]) <= SETTLE_DEG
    settle = f"{(np.argmax(after)) / RATE_HZ:.2f} s" if after.any() else f"not within {SETTLE_DEG:.0f} deg"
    print(
        f"{label:22s} peak |load| {np.abs(data[:, 2]).max():5.1f} %  "
        f"peak current {data[:, 3].max():5.0f} mA  "
        f"min V {data[:, 4].min():4.1f}  max err {np.abs(err).max():5.1f} deg  "
        f"final err {err[-1]:+5.1f} deg  settle after move: {settle}"
    )
    flags = sorted({int(x) for x in data[:, 7] if x})
    limit = f"{data[:, 6].min():.0f}..{data[:, 6].max():.0f}"
    print(f"{'':22s} torque limit {limit} (of 1000)  status flags seen: {flags or 'none'}")


try:
    input("Clear the workspace, hold the arm if needed. ENTER to move to center...")
    move_all({robot.id: robot}, {robot.id: CENTER}, tol=args.tol)
    t_start = time.perf_counter()
    lift = CENTER[f"{JOINT}.pos"]
    print("Driving shoulder_lift: down (with gravity), up (against gravity), back to 0. Ctrl+C to stop.")
    for label, end in (
        (f"down +{args.down:g} deg", lift + args.down),
        (f"UP -{args.up:g} deg (vs gravity)", lift - args.up),
        ("back to 0", lift),
    ):
        lift = run_phase(label, lift, end)
    print()
    print("Phase                  (load is % of max torque; current/voltage from the servo's own sensors)")
    for label in dict.fromkeys(s[1] for s in samples):
        summarize(label, [s for s in samples if s[1] == label])
    v = np.array([s[6] for s in samples])
    print(f"Supply voltage seen by the servo: rest {v[0]:.1f} V, lowest {v.min():.1f} V "
          f"(expect ~12 V; a big dip under load points at the supply or wiring)")
except KeyboardInterrupt:
    print("\nCtrl+C: stopping.")
finally:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if args.log and samples:
        with open(args.log, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(COLUMNS)
            writer.writerows(samples)
        print(f"Saved {len(samples)} samples to {args.log}")
    try:
        move_all({robot.id: robot}, {robot.id: home})
    except Exception as e:  # e.g. the arm's USB serial port vanished
        print(f"Could not return to home: {e!r}")
    try:
        robot.disconnect()
    except Exception as e:
        print(f"disconnect failed ({e!r}); check power/USB, torque may still be on.")
