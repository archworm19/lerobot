"""Measure how far `gripper.pos` opens an arm's jaws, and write it into a calibrated URDF. Runs on your computer.

  uv run python -m lerobot.felix_jr.calibration.gripper_step tcp://<pi>:5555 --arm right --start-from <calibration.urdf>
      --out <new calibration.urdf> [--positions 10,25,40,55,70,85]
  uv run python -m lerobot.felix_jr.calibration.gripper_step --refit --arm right --start-from <urdf> --out <urdf>

Start the server on the Pi first, with the arm's port (`server.server --right-port <PORT>`). The script steps the
gripper to each of `--positions` (LeRobot's `gripper.pos`, 0-100) and asks for the gap between the fingertips: the
inner faces, at the very tips, with calipers. Enter skips a position, q stops early. The gripper goes back to where it
started, and the session sends the arm to rest when the script ends.

The SO-101's gripper is a claw: the moving jaw turns about a hinge, and its tip meets the fixed jaw's at the pinch
point. So a gap is a chord of the circle the moving tip turns on (`XleRobotModel.jaw_radius`, from the calibration),
and gives the jaws' opening angle. A straight line through (`gripper.pos`, opening) gives the map: where the tips
meet (`pos_closed`) and degrees per unit. Gaps under half a millimeter count as touching and are left out of the fit
(closing further only squeezes). The fit should come out near the servo's own scale, the calibrated tick range
over 100 (about 1.27 deg per unit); the script warns when it does not.

The map goes into `--out`: the `--start-from` calibration plus a `GripperMap` for the arm (`gripper_maps` in its
`<felix_calibration>` element), replacing an earlier one. `--start-from` is never overwritten. The map also records
the gripper's LeRobot calibration, as the robot reports it (`FelixClient.arm_calibration`): `lerobot-calibrate`
changes it, and then the map no longer holds (`XleRobotModel.check_robot` refuses it). For the same reason the script
refuses a `--start-from` whose camera calibration was made under a different calibration of the arm.

`--refit` fits the measurements already in `--start-from`'s map again, with no robot: after a new camera calibration
changes the jaw radius, say.
"""

import argparse
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np

from lerobot.felix_jr.calibration.xle_robot_model import GripperMap, XleRobotModel, arm_record, motor_record
from lerobot.felix_jr.server.client import FelixClient
from lerobot.felix_jr.server.protocol import DEFAULT_PORT

DEFAULT_POSITIONS = (10.0, 25.0, 40.0, 55.0, 70.0, 85.0)
TOUCHING_M = 0.0005  # a smaller gap: the tips touch
TICKS_PER_TURN = 4096
SCALE_WARNING = 0.1  # warn when the fitted scale is this far (fraction) from the servo's own


def opening_from_gap(gap_m: float, radius_m: float) -> float:
    """The jaws' opening angle (degrees) for a gap `gap_m` between the tips: a chord of the tip's circle."""
    if not 0.0 <= gap_m <= 2 * radius_m:
        raise ValueError(
            f"a gap of {gap_m * 1000:.1f} mm does not fit a jaw of radius {radius_m * 1000:.1f} mm"
        )
    return float(np.degrees(2 * np.arcsin(gap_m / (2 * radius_m))))


def servo_deg_per_unit(calibration: Mapping[str, int]) -> float:
    """Degrees the servo turns per unit of `gripper.pos`: the gripper's calibrated tick range over 100."""
    return (calibration["range_max"] - calibration["range_min"]) * 360.0 / TICKS_PER_TURN / 100.0


def fit_map(
    measurements: Sequence[tuple[float, float]], radius_m: float, calibration: Mapping[str, int]
) -> GripperMap:
    """The map through `(gripper.pos, gap m)` pairs, leaving out the touching ones (see the module docstring)."""
    open_ = [(pos, gap) for pos, gap in measurements if gap >= TOUCHING_M]
    if len({pos for pos, _ in open_}) < 2:
        raise ValueError("need gaps at two or more gripper positions with the jaws apart")
    pos = np.array([p for p, _ in open_])
    opening = np.array([opening_from_gap(gap, radius_m) for _, gap in open_])
    slope, intercept = np.polyfit(pos, opening, 1)
    return GripperMap(
        pos_closed=float(-intercept / slope),
        deg_per_unit=float(slope),
        calibration=motor_record(calibration),
        measurements=tuple((float(p), float(g)) for p, g in measurements),
    )


def arm_calibration(client, arm: str) -> dict[str, dict[str, int]]:
    """`arm`'s LeRobot calibration as the robot reports it (`arm_record`); its gripper's is `["gripper"]`."""
    return arm_record(client.arm_calibration()[f"{arm}_arm"])


def measure(
    client,
    arm: str,
    positions: Sequence[float],
    ask: Callable[[str], str] = input,
    say: Callable[..., None] = print,
) -> list[tuple[float, float]]:
    """Step `arm`'s gripper through `positions` and ask for the gap at each; `(gripper.pos read back, gap m)` pairs.

    The gripper returns to where it started at the end, however it ends.
    """
    arm_id = f"{arm}_arm"
    start = client.get_state()[arm_id]["gripper.pos"]
    measurements: list[tuple[float, float]] = []
    try:
        for target in positions:
            client.move_joints({arm_id: {"gripper.pos": float(target)}})
            pos = float(client.get_state()[arm_id]["gripper.pos"])
            while True:
                answer = ask(
                    f"gripper.pos {pos:.1f}: gap between the tips in mm (Enter skips, q stops): "
                ).strip()
                if answer.lower() == "q":
                    return measurements
                if not answer:
                    break
                try:
                    gap_mm = float(answer)
                except ValueError:
                    say(f"not a number: {answer!r}")
                    continue
                if gap_mm < 0:
                    say("a gap cannot be negative")
                    continue
                measurements.append((pos, gap_mm / 1000.0))
                break
        return measurements
    finally:
        client.move_joints({arm_id: {"gripper.pos": float(start)}})


def report(gripper: GripperMap, radius_m: float, say: Callable[..., None] = print) -> list[str]:
    """Print the fit against each measurement; returns the warnings it printed."""
    say(f"jaw radius {radius_m * 1000:.1f} mm")
    say("  gripper.pos   gap mm   opening deg   fit deg")
    for pos, gap in gripper.measurements:
        opening = opening_from_gap(gap, radius_m)
        touching = "  (touching: not fit)" if gap < TOUCHING_M else ""
        say(
            f"  {pos:11.1f}   {gap * 1000:6.1f}   {opening:11.1f}   {gripper.opening_deg(pos):7.1f}{touching}"
        )
    servo = servo_deg_per_unit(gripper.calibration)
    say(
        f"tips meet at gripper.pos {gripper.pos_closed:.1f}; {gripper.deg_per_unit:.3f} deg per unit "
        f"(the servo's range gives {servo:.3f})"
    )
    warnings = []
    if abs(abs(gripper.deg_per_unit) / servo - 1.0) > SCALE_WARNING:
        warnings.append(
            f"the fitted scale is more than {SCALE_WARNING:.0%} off the servo's: re-measure, or check the gaps"
        )
    for warning in warnings:
        say(f"WARNING: {warning}")
    return warnings


def main() -> None:
    """Command line entry point; see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("endpoint", nargs="?", default=f"tcp://localhost:{DEFAULT_PORT}")
    parser.add_argument("--arm", default="right", choices=["left", "right"])
    parser.add_argument(
        "--start-from", type=Path, required=True, help="the calibrated URDF to add the map to"
    )
    parser.add_argument("--out", type=Path, required=True, help="where to write the result (a new file)")
    parser.add_argument(
        "--positions",
        default=",".join(f"{p:g}" for p in DEFAULT_POSITIONS),
        help="gripper.pos values to measure at, comma-separated",
    )
    parser.add_argument("--refit", action="store_true", help="refit --start-from's measurements; no robot")
    args = parser.parse_args()

    if args.out.resolve() == args.start_from.resolve():
        raise SystemExit("--out must be a new file: --start-from is never overwritten")
    model = XleRobotModel.load(args.start_from)
    radius = model.jaw_radius(args.arm)
    old = model.gripper_maps.get(args.arm)

    if args.refit:
        if old is None:
            raise SystemExit(f"{args.start_from} has no gripper map for the {args.arm} arm to refit")
        gripper = fit_map(old.measurements, radius, old.calibration)
    else:
        positions = [float(p) for p in args.positions.split(",")]
        with FelixClient(args.endpoint, timeout_s=15.0) as client:
            if f"{args.arm}_arm" not in client.server_info["arms"]:
                raise SystemExit(f"the server has no {args.arm} arm (start it with --{args.arm}-port)")
            record = arm_calibration(client, args.arm)
            camera = model.arm_calibration.get(args.arm)
            if camera is not None and camera != record:
                raise SystemExit(
                    f"the {args.arm} arm was recalibrated since {args.start_from}'s camera calibration: recalibrate "
                    "the camera first (docs/README.md section 6)"
                )
            if old is not None and old.calibration != record["gripper"]:
                print(f"the {args.arm} arm was recalibrated since its gripper map was measured: replacing it")
            print("Keep the jaws clear: they open and close by themselves.")
            with client.session():
                measurements = measure(client, args.arm, positions)
        gripper = fit_map(measurements, radius, record["gripper"])

    report(gripper, radius)
    model.gripper_maps[args.arm] = gripper
    model.save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
