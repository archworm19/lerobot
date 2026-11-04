"""Measure how far `gripper.pos` opens an arm's jaws, and write it into a calibrated URDF. Runs on your computer.

  uv run python -m lerobot.felix_jr.calibration.gripper_step tcp://<pi>:5555 --arm right --start-from <calibration.urdf>
      --out <new calibration.urdf> [--positions 10,25,40,55,70,85] [--no-plot]
  uv run python -m lerobot.felix_jr.calibration.gripper_step --refit --arm right --start-from <urdf> --out <urdf>

Start the server on the Pi first, with the arm's port (`server.server --right-port <PORT>`). The script steps the
gripper to each of `--positions` (LeRobot's `gripper.pos`, 0-100) and asks for the gap between the fingertips: the
inner faces, at the very tips, with calipers. Enter skips a position, q stops early. The gripper goes back to where it
started, and the session sends the arm to rest when the script ends. A window plots the measurements and the fit as
you go (`--no-plot` leaves it out); the final plot is saved next to `--out`, as a PNG.

The SO-101's gripper is a claw: the moving jaw turns about a hinge, and its tip meets the fixed jaw's at the pinch
point. So a gap is a chord of the circle the tip turns on, `gap = 2 r sin(theta / 2)`, and the opening angle theta is
the servo's own: `theta = k (gripper.pos - pos_closed)`, where k, degrees per unit, is the gripper's calibrated tick
range over 100 (the servo drives the jaw directly). The fit finds where the tips meet (`pos_closed`) and r, the radius
where the calipers touch. Fitting k as well does not work: over the jaw's range, a larger k and a smaller r fit the
gaps equally well. r should come out near the model's distance from the pinch point to the hinge
(`XleRobotModel.jaw_radius`), a few millimeters more if the calipers touch beyond the pinch point; the script warns
when it does not, or when the gaps scatter around the fit. Gaps under half a millimeter count as touching and are
left out (closing further only squeezes).

The map goes into `--out`: the `--start-from` calibration plus a `GripperMap` for the arm (`gripper_maps` in its
`<felix_calibration>` element), replacing an earlier one. `--start-from` is never overwritten. The map also records
the gripper's LeRobot calibration, as the robot reports it (`FelixClient.arm_calibration`): `lerobot-calibrate`
changes it, and then the map no longer holds (`XleRobotModel.check_robot` refuses it). For the same reason the script
refuses a `--start-from` whose camera calibration was made under a different calibration of the arm.

`--refit` fits the measurements already in `--start-from`'s map again, with no robot.
"""

import argparse
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from lerobot.felix_jr.calibration.xle_robot_model import GripperMap, XleRobotModel, arm_record, motor_record
from lerobot.felix_jr.server.client import FelixClient
from lerobot.felix_jr.server.protocol import DEFAULT_PORT
from lerobot.utils.import_utils import _matplotlib_available, _scipy_available, require_package

if TYPE_CHECKING or _scipy_available:
    from scipy.optimize import least_squares
if TYPE_CHECKING or _matplotlib_available:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

DEFAULT_POSITIONS = (10.0, 25.0, 40.0, 55.0, 70.0, 85.0)
TOUCHING_M = 0.0005  # a smaller gap: the tips touch
TICKS_PER_TURN = 4096
RADIUS_WARNING_M = 0.010  # warn when the fitted radius is this far from the model's
SCATTER_WARNING_M = 0.0025  # warn when the gaps scatter this much (RMS) around the fit


def opening_from_gap(gap_m: float, radius_m: float) -> float:
    """The jaws' opening angle (degrees) for a gap `gap_m` between the tips: a chord of the tip's circle."""
    if not 0.0 <= gap_m <= 2 * radius_m:
        raise ValueError(
            f"a gap of {gap_m * 1000:.1f} mm does not fit a jaw of radius {radius_m * 1000:.1f} mm"
        )
    return float(np.degrees(2 * np.arcsin(gap_m / (2 * radius_m))))


def chord(radius_m: float, opening_deg: np.ndarray) -> np.ndarray:
    """The gap between the tips (m) at each opening angle; `opening_from_gap`'s inverse, 0 once they touch."""
    return 2 * radius_m * np.sin(np.radians(np.clip(opening_deg, 0.0, 180.0)) / 2)


def servo_deg_per_unit(calibration: Mapping[str, int]) -> float:
    """Degrees the servo turns per unit of `gripper.pos`: the gripper's calibrated tick range over 100."""
    return (calibration["range_max"] - calibration["range_min"]) * 360.0 / TICKS_PER_TURN / 100.0


def open_measurements(measurements: Sequence[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    """The `(gripper.pos, gap m)` pairs with the jaws apart, as two arrays."""
    open_ = [(pos, gap) for pos, gap in measurements if gap >= TOUCHING_M]
    return np.array([p for p, _ in open_], dtype=float), np.array([g for _, g in open_], dtype=float)


def fit_map(
    measurements: Sequence[tuple[float, float]], calibration: Mapping[str, int], radius_guess_m: float
) -> GripperMap:
    """The map through `(gripper.pos, gap m)` pairs, at the servo's own scale (see the module docstring).

    Fits where the tips meet and the radius, starting from `radius_guess_m` (the model's), leaving out touching tips.
    """
    require_package("scipy", extra="felix_jr")
    pos, gap = open_measurements(measurements)
    if len(set(pos.tolist())) < 2:
        raise ValueError("need gaps at two or more gripper positions with the jaws apart")
    k = servo_deg_per_unit(calibration)
    openings = [opening_from_gap(min(g, 2 * radius_guess_m), radius_guess_m) for g in gap]
    closed_guess = float(np.mean(pos - np.array(openings) / k))

    def residuals(x: np.ndarray) -> np.ndarray:
        return chord(float(x[0]), k * (pos - x[1])) - gap

    solution = least_squares(residuals, [radius_guess_m, closed_guess])
    return GripperMap(
        pos_closed=float(solution.x[1]),
        deg_per_unit=k,
        calibration=motor_record(calibration),
        measurements=tuple((float(p), float(g)) for p, g in measurements),
        radius_m=float(solution.x[0]),
    )


def scatter_m(gripper: GripperMap) -> float:
    """RMS (m) of the measured gaps around the fit, over the jaws apart."""
    assert gripper.radius_m is not None
    pos, gap = open_measurements(gripper.measurements)
    fitted = chord(gripper.radius_m, gripper.deg_per_unit * (pos - gripper.pos_closed))
    return float(np.sqrt(np.mean((fitted - gap) ** 2)))


def arm_calibration(client, arm: str) -> dict[str, dict[str, int]]:
    """`arm`'s LeRobot calibration as the robot reports it (`arm_record`); its gripper's is `["gripper"]`."""
    return arm_record(client.arm_calibration()[f"{arm}_arm"])


def measure(
    client,
    arm: str,
    positions: Sequence[float],
    ask: Callable[[str], str] = input,
    say: Callable[..., None] = print,
    on_measure: Callable[[list[tuple[float, float]]], None] | None = None,
) -> list[tuple[float, float]]:
    """Step `arm`'s gripper through `positions` and ask for the gap at each; `(gripper.pos read back, gap m)` pairs.

    `on_measure` gets the measurements so far after each one (to plot them). The gripper returns to where it started
    at the end, however it ends.
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
                if on_measure is not None:
                    on_measure(list(measurements))
                break
        return measurements
    finally:
        client.move_joints({arm_id: {"gripper.pos": float(start)}})


def report(gripper: GripperMap, model_radius_m: float, say: Callable[..., None] = print) -> list[str]:
    """Print the fit against each measurement; returns the warnings it printed."""
    assert gripper.radius_m is not None
    r, k, closed = gripper.radius_m, gripper.deg_per_unit, gripper.pos_closed
    say("  gripper.pos   gap mm   fit mm   opening deg")
    for pos, gap in gripper.measurements:
        fitted = float(chord(r, np.array(k * (pos - closed))))
        opening = opening_from_gap(min(gap, 2 * r), r)
        touching = "  (touching: not fit)" if gap < TOUCHING_M else ""
        say(f"  {pos:11.1f}   {gap * 1000:6.1f}   {fitted * 1000:6.1f}   {opening:11.1f}{touching}")
    scatter = scatter_m(gripper)
    say(
        f"tips meet at gripper.pos {closed:.1f}; {k:.3f} deg per unit (the servo's); radius {r * 1000:.1f} mm "
        f"(the model's pinch point: {model_radius_m * 1000:.1f} mm); gaps within {scatter * 1000:.1f} mm RMS"
    )
    warnings = []
    if abs(r - model_radius_m) > RADIUS_WARNING_M:
        warnings.append(
            f"the radius is {(r - model_radius_m) * 1000:+.0f} mm from the model's: check where the calipers touch "
            "the tips, and the gaps"
        )
    if scatter > SCATTER_WARNING_M:
        warnings.append(f"the gaps scatter {scatter * 1000:.1f} mm around the fit: re-measure")
    for warning in warnings:
        say(f"WARNING: {warning}")
    return warnings


def _pos_minus(value: float) -> str:
    """`pos − value` for an equation, without a double sign."""
    return f"pos − {value:.1f}" if value >= 0 else f"pos + {-value:.1f}"


def plot(
    fig: "Figure",
    measurements: Sequence[tuple[float, float]],
    gripper: GripperMap | None,
    model_radius_m: float,
) -> None:
    """Draw the measurements, and the fit if there is one, into `fig`: the gaps, and the opening angles they give."""
    fig.clear()
    gaps_ax, angles_ax = fig.subplots(1, 2)
    pos = np.array([p for p, _ in measurements], dtype=float)
    gap_mm = np.array([g for _, g in measurements], dtype=float) * 1000
    apart = gap_mm >= TOUCHING_M * 1000
    gaps_ax.plot(pos[apart], gap_mm[apart], "o", color="C0", label="measured")
    if (~apart).any():
        gaps_ax.plot(pos[~apart], gap_mm[~apart], "o", mfc="none", color="C0", label="touching (not fit)")
    title = f"{len(measurements)} measurement(s)"
    if gripper is not None and gripper.radius_m is not None:
        r, k, closed = gripper.radius_m, gripper.deg_per_unit, gripper.pos_closed
        lo = min(closed, float(pos.min())) - 3.0
        grid = np.linspace(lo, max(float(pos.max()), 100.0) + 2.0, 200)
        gaps_ax.plot(
            grid,
            chord(r, k * (grid - closed)) * 1000,
            color="C1",
            label=f"gap = 2 · {r * 1000:.1f} mm · sin({k:.3f}° · ({_pos_minus(closed)}) / 2)",
        )
        gaps_ax.plot(
            grid,
            chord(model_radius_m, k * (grid - closed)) * 1000,
            "--",
            color="C7",
            label=f"same angles, the model's radius {model_radius_m * 1000:.1f} mm",
        )
        openings = [opening_from_gap(min(g / 1000, 2 * r), r) for g in gap_mm[apart]]
        angles_ax.plot(pos[apart], openings, "o", color="C0", label="from each gap")
        angles_ax.plot(
            grid, k * (grid - closed), color="C1", label=f"θ = {k:.3f}°/unit · ({_pos_minus(closed)})"
        )
        angles_ax.legend(loc="upper left", fontsize="small")
        title += f": tips meet at {closed:.1f}, radius {r * 1000:.1f} mm, gaps within {scatter_m(gripper) * 1000:.1f} mm RMS"
    gaps_ax.set(xlabel="gripper.pos", ylabel="gap between the tips (mm)")
    angles_ax.set(xlabel="gripper.pos", ylabel="opening θ (deg)")
    for ax in (gaps_ax, angles_ax):
        ax.grid(alpha=0.3)
    gaps_ax.legend(loc="upper left", fontsize="small")
    fig.suptitle(title)
    fig.tight_layout()


def try_fit(
    measurements: Sequence[tuple[float, float]], calibration: Mapping[str, int], radius_guess_m: float
) -> GripperMap | None:
    """`fit_map`, or None while there are too few measurements."""
    try:
        return fit_map(measurements, calibration, radius_guess_m)
    except ValueError:
        return None


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
    parser.add_argument("--no-plot", action="store_true", help="no plot window (the PNG is still saved)")
    args = parser.parse_args()

    if args.out.resolve() == args.start_from.resolve():
        raise SystemExit("--out must be a new file: --start-from is never overwritten")
    require_package("matplotlib", extra="felix_jr")
    model = XleRobotModel.load(args.start_from)
    radius = model.jaw_radius(args.arm)
    old = model.gripper_maps.get(args.arm)
    fig = Figure(figsize=(12, 5)) if args.no_plot else plt.figure(figsize=(12, 5))

    def show(measurements: list[tuple[float, float]], calibration: Mapping[str, int]) -> None:
        plot(fig, measurements, try_fit(measurements, calibration, radius), radius)
        if not args.no_plot:
            plt.pause(0.05)  # draw now: the prompt that follows blocks the window's event loop

    if args.refit:
        if old is None:
            raise SystemExit(f"{args.start_from} has no gripper map for the {args.arm} arm to refit")
        gripper = fit_map(old.measurements, old.calibration, radius)
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
            if not args.no_plot:
                plt.ion()
                show([], record["gripper"])
            with client.session():
                measurements = measure(
                    client, args.arm, positions, on_measure=lambda m: show(m, record["gripper"])
                )
        gripper = fit_map(measurements, record["gripper"], radius)

    report(gripper, radius)
    model.gripper_maps[args.arm] = gripper
    model.save(args.out)
    plot(fig, gripper.measurements, gripper, radius)
    png = args.out.with_suffix(".png")
    fig.savefig(png, dpi=120)
    print(f"wrote {args.out} and {png}")
    if not args.no_plot:
        print("close the plot window to finish")
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
