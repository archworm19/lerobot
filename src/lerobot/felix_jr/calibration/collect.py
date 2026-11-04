"""Collect calibration clicks: you click the pinch point of the closed gripper in the camera image, view after view.

  uv run python -m lerobot.felix_jr.calibration.collect tcp://<pi>:5555 --arm left --out-dir ~/cal_left
      [--start-from calibration.urdf] [--with ~/cal_right ...] [--n 40] [--step 2] [--max-vel 30]
      [--seed-pose "shoulder_lift=-10"] [--delta-scale 1] [--side positive] [--floor-z Z] [--resume]

Start the server on the Pi with the arm (and the head, which plugs into the left arm's board). Clear the workspace
in front of the robot; the script commands the gripper closed in every pose. A window shows the head camera. For each
view the arm moves (slowly) and the head turns so that, as the current model predicts, the pinch point is in the
middle of the picture and seen from the side the wrist camera does not block (`views.find_view`); a cyan circle
marks the predicted pinch point. Then:

  left mouse click    record the pinch point: click where the closed fingertips meet
  n                   go to the next view
  arrow keys          turn the head (up / down tilt, left / right pan) by --step degrees, e.g. to find the gripper
  u                   undo the last click
  r                   refit now and print how well the model explains the clicks
  q or Esc            finish: fit everything and write `calibration.urdf` (the robot is sent to rest)

**Starting point.** `--start-from` an earlier calibration (`calibration.urdf`) and the first view already shows the
gripper. On a fresh robot the head's zero angles are unknown (tilt by about 20 degrees), so the first views may miss:
nudge the head with the arrow keys until the gripper is in the picture and click; after a few clicks the model
refits and the head lands on the pinch point by itself. Vary the head too: after a few views, nudge it a few degrees
before clicking, so the head's parameters are not confounded with the arm's. Aim for 30-60 clicks.

**One calibration.** Every click refits the whole model (`xle_fit.fit`): this session's clicks together with the
clicks of the sessions given with `--with` (e.g. the other arm's), so the head and camera have one estimate. At the
end the same joint fit, with held-out validation, writes `calibration.urdf` and its report `calibration.md` to
`--out-dir`. All sessions must share the head's servo map (its recorded home); see `xle_fit`.

**Safety.** Views keep the pinch point a few centimeters above the floor: `--floor-z` (root frame, meters), else the
calibration's, else the arm base's height (conservative: the table is usually a little below it). Nothing checks
for other collisions: the region is small, the arm moves slowly, and you watch it.

Clicks are saved as you go in `--out-dir`: `observations.json` and `images/` (the color and raw depth of every clicked
frame). `--resume` continues an earlier folder. Every click uses the *measured* joint angles reported with the
frame. The script runs inside a client session, so if you press Ctrl+C or it dies, the robot goes to rest.
"""

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lerobot.felix_jr.calibration.live_view import (
    INTRINSICS_KEYS,
    NECK_RANGE_TICKS,  # noqa: F401  (re-exported: head_limits' range without recorded limits)
    LiveView,
    decode_key,
    head_limits,  # noqa: F401  (re-exported)
    parse_seed,
    set_floor,
)
from lerobot.felix_jr.calibration.views import MAX_ROLL_DEG
from lerobot.felix_jr.calibration.xle_fit import (
    SESSION_FILE,
    Click,
    FitConfig,
    Session,
    arm_calibration_of,
    calibrate,
    check_compatible,
    fit,
    load_session,
    pixel_errors,
    report,
    rms,
    save_session,
)
from lerobot.felix_jr.calibration.xle_robot_model import (
    HEAD_JOINTS,
    SIDES,
    XleRobotModel,
    arm_record,
    default_parameters,
)
from lerobot.felix_jr.depth import with_matrices
from lerobot.felix_jr.harness.scene import Scene
from lerobot.felix_jr.server.client import FelixClient
from lerobot.felix_jr.server.protocol import DEFAULT_PORT

MIN_CLICKS_TO_FIT = 4
NECK_PRIOR_NO_HOME = np.radians(45.0)  # head pan zero prior sigma when the neck has no recorded home
REFIT_EVALUATIONS = (
    20  # optimizer evaluations per refit after a click (`r` and the final fit run to convergence)
)
CALIBRATION_FILE = "calibration.urdf"


def annotate(
    image: np.ndarray, lines: list[str], predicted: np.ndarray | None, clicks: list[tuple[float, float]]
) -> np.ndarray:
    """Copy of the RGB `image` with the predicted pinch point (cyan circle), this frame's clicks (green) and text."""
    out = image.copy()
    if predicted is not None and np.isfinite(predicted).all():
        centre = (int(round(float(predicted[0]))), int(round(float(predicted[1]))))
        cv2.circle(out, centre, 14, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.circle(out, centre, 14, (0, 255, 255), 1, cv2.LINE_AA)
    for u, v in clicks:
        cv2.drawMarker(out, (int(round(u)), int(round(v))), (0, 255, 0), cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
    for row, text in enumerate(lines):
        org = (8, 20 + 20 * row)
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


class CollectSession(LiveView):
    """One calibration run: moves the robot, shows frames, records clicks, refits after each. See the module docstring."""

    no_more_views = "no more views: click what is left, then q"

    def __init__(
        self,
        client: FelixClient,
        arm: str,
        out_dir: Path,
        show: Callable[[np.ndarray], None],
        start_from: Path | None = None,
        other_sessions: Sequence[Path] = (),
        n_targets: int = 40,
        max_vel: float = 30.0,
        max_accel: float = 60.0,
        head_step_deg: float = 2.0,
        resume: bool = False,
        color_format: str = "png",
        seed_pose: dict[str, float] | None = None,
        delta_scale: float = 1.0,
        side: str | None = None,
        floor_z: float | None = None,
        max_roll: float = MAX_ROLL_DEG,
        refit_evaluations: int | None = REFIT_EVALUATIONS,
    ):
        """Build the model, read the earlier clicks, plan the targets, and show a first frame.

        `start_from`: a calibrated URDF whose estimates start the model (its head servo directions must match the
        robot's). `other_sessions`: session folders whose clicks join every fit. `seed_pose`: LeRobot degrees of the
        arm pose the targets spread around (`{"shoulder_pan.pos": ...}`; default the center pose). `side` overrides
        which side of the gripper must face the camera; `floor_z` the floor (see the module docstring).
        """
        self.out_dir, self.refit_evaluations = out_dir, refit_evaluations
        self.notes: list[str] = []
        self.last_rms: float | None = None
        self.frame_clicks: list[tuple[float, float]] = []
        (out_dir / "images").mkdir(parents=True, exist_ok=True)

        capture = client.capture(color_format=color_format)
        intrinsics = {k: capture.calibration["color_intrinsics"][k] for k in INTRINSICS_KEYS}
        model = self._build_model(start_from, client.head_state(), intrinsics)
        record = arm_record(client.arm_calibration()[f"{arm}_arm"])  # what the clicks' joint angles mean
        if side is not None:
            if side not in SIDES:
                raise ValueError(f"side must be one of {SIDES}")
            model.visible_side[arm] = side
        set_floor(model, arm, floor_z, self._note)
        super().__init__(
            client,
            model,
            arm,
            show,
            capture,
            n_targets,
            max_vel,
            max_accel,
            head_step_deg,
            color_format,
            seed_pose,
            delta_scale,
            max_roll,
        )
        self.message = (
            self.notes[-1] if self.notes else "click the pinch point of the closed gripper; n = next view"
        )

        if resume and (out_dir / SESSION_FILE).is_file():
            self.session = load_session(out_dir)
            if self.session.arm != arm:
                raise ValueError(f"{out_dir} holds {self.session.arm} arm clicks, not {arm}")
            self.message = f"resumed with {len(self.session.clicks)} clicks"
        else:
            self.session = Session(
                out_dir, arm, [], self.model.head_signs, self.model.head_refs, intrinsics, record
            )
        self.others = [load_session(path) for path in other_sessions]
        here = Session(out_dir, arm, [], self.model.head_signs, self.model.head_refs, intrinsics, record)
        check_compatible([here, self.session, *self.others])
        if self.others:
            self._note(
                f"refitting with {sum(len(s.clicks) for s in self.others)} clicks from the other sessions"
            )

        self._refit()
        self.render()

    # ---- the start of a session ----------------------------------------------------------------------------

    def _note(self, text: str) -> None:
        self.notes.append(text)
        self.message = text

    def _build_model(
        self, start_from: Path | None, head: dict[str, dict[str, Any]], intrinsics: dict[str, float]
    ) -> XleRobotModel:
        """The model to start from: an earlier calibration, or the URDF with the head's servo map as it is now."""
        signs = {n: head[n]["sign"] for n in HEAD_JOINTS}
        if start_from is not None:
            model = XleRobotModel.load(start_from)
            if model.head_signs != signs:
                raise ValueError(
                    f"the head's saved directions changed since that calibration ({model.head_signs} then, {signs} "
                    "now): its estimates do not apply; run without --start-from"
                )
            if model.intrinsics != intrinsics:
                self._note("the camera's intrinsics differ from the calibration's: using the camera's")
                model.intrinsics = intrinsics
            self._note(f"starting from {start_from}")
            return model
        refs = {n: head[n]["home"] if head[n]["home"] is not None else head[n]["ticks"] for n in HEAD_JOINTS}
        overrides = {}
        if head["head_neck"]["home"] is None:
            # its reference is then wherever the neck happens to be, so its zero can be anything
            overrides["zero:head_pan_joint"] = (0.0, NECK_PRIOR_NO_HOME)
            self._note(
                "the neck has no recorded home, so the head's pan is unknown at first (record it with "
                "setup.head_range --name head_neck); expect the circle to be off sideways until a few clicks"
            )
        return XleRobotModel.from_urdf(
            head_signs=signs,
            head_refs=refs,
            intrinsics=intrinsics,
            parameters=default_parameters(overrides=overrides),
        )

    # ---- showing --------------------------------------------------------------------------------------------

    def status_lines(self, predicted: np.ndarray | None) -> list[str]:
        """Text for the image: counts, the model's current accuracy, and the last message."""
        fit_text = f"fit RMS {self.last_rms:.1f} px" if self.last_rms is not None else "no fit yet"
        lines = [
            f"{self.arm} arm: {len(self.session.clicks)} clicks, view {self.target_index}/{len(self.targets)}, "
            f"{fit_text}",
            "click the pinch point | n next view | arrows head | u undo | r refit | q quit",
        ]
        if predicted is None or not self.model.in_view(predicted):
            lines.append("predicted pinch point is off the picture: nudge the head, or press n")
        if self.message:
            lines.append(self.message)
        return lines

    def render(self) -> None:
        """Show the current frame with the predicted pinch point, this frame's clicks and the status text."""
        pred = self.model.predict(self.arm, self.current_q())
        uv = pred.uv if pred.z > 0 else None
        self.show(annotate(self.capture.color, self.status_lines(uv), uv, self.frame_clicks))

    def refresh(self, after: float | None = None) -> None:
        """Take a new frame (taken after server time `after`, if given) and show it, with no clicks on it yet."""
        self.frame_clicks = []
        super().refresh(after)

    # ---- clicks ---------------------------------------------------------------------------------------------

    def click(self, u: float, v: float) -> Click:
        """Record a click at pixel (u, v) of the current frame, save it, and refit."""
        state = self.capture.state
        index = len(self.session.clicks)
        scene = Scene.from_capture(
            self.capture.color, self.capture.raw_depth, with_matrices(self.capture.calibration)
        )
        depth = float(scene.pixel_to_xyz(np.array([[u, v]]), window=7, max_spread_m=0.06)[0, 2])
        image = f"images/{index:03d}.png"
        click = Click(
            arm=self.arm,
            arm_pose={k: float(value) for k, value in state[self.arm_id].items()},
            head_ticks={n: int(state["head"][n]["ticks"]) for n in HEAD_JOINTS},
            pixel=(float(u), float(v)),
            depth_m=depth if np.isfinite(depth) else None,
            image=image,
            session=self.out_dir.name,
            index=index,
            timestamp=self.capture.timestamp,
        )
        cv2.imwrite(str(self.out_dir / image), cv2.cvtColor(self.capture.color, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(self.out_dir / f"images/{index:03d}_depth.png"), self.capture.raw_depth)
        self.session.clicks.append(click)
        self.frame_clicks.append(click.pixel)
        save_session(self.session)
        self.message = f"click {len(self.session.clicks)} saved"
        self._refit()
        self.render()
        return click

    def undo(self) -> None:
        """Drop the last click (its images stay in the folder) and refit."""
        if self.session.clicks:
            self.session.clicks.pop()
            if self.frame_clicks:
                self.frame_clicks.pop()
            save_session(self.session)
            self.message = f"undid the last click ({len(self.session.clicks)} left)"
            self._refit()
        self.render()

    def all_clicks(self) -> list[Click]:
        """This session's clicks and those of the other sessions: everything the fit uses."""
        return [*self.session.clicks, *(c for s in self.others for c in s.clicks)]

    def _refit(self, evaluations: int | None = None) -> None:
        """Fit the model to every click, starting from its current estimates, and keep the result."""
        clicks = self.all_clicks()
        if len(clicks) < MIN_CLICKS_TO_FIT:
            self.last_rms = None
            return
        result = fit(self.model, clicks, FitConfig(max_nfev=evaluations or self.refit_evaluations))
        self.model.set_estimates(result.theta)
        self.model.covariance = result.covariance
        own = len(self.session.clicks)
        self.last_rms = rms(result.pixel_errors[:own]) if own else None

    def refit(self) -> None:
        """Refit to convergence and say how well the model explains this session's clicks."""
        previous, self.refit_evaluations = self.refit_evaluations, None
        try:
            self._refit()
        finally:
            self.refit_evaluations = previous
        if self.last_rms is not None:
            before = rms(pixel_errors(self.model, self.session.clicks, np.zeros(len(self.model.parameters))))
            self.message = f"refit: {before:.0f} px uncalibrated -> {self.last_rms:.1f} px"
        else:
            self.message = f"need {MIN_CLICKS_TO_FIT} clicks to fit"
        self.render()

    def finish(self, holdout_every: int = 5) -> Path | None:
        """Fit every click with held-out validation and write `calibration.urdf` and `calibration.md`.

        Returns the calibration's path, or None if there are too few clicks to fit.
        """
        clicks = self.all_clicks()
        if len(clicks) < MIN_CLICKS_TO_FIT:
            return None
        cfg = FitConfig()
        final, holdout = calibrate(self.model, clicks, cfg, holdout_every)
        # the estimates hold under the arm calibrations the clicks were taken with, not a --start-from's
        self.model.arm_calibration = arm_calibration_of([self.session, *self.others])
        path = self.out_dir / CALIBRATION_FILE
        self.model.save(path)
        path.with_suffix(".md").write_text(report(self.model, clicks, final, holdout, cfg))
        return path

    # ---- key handling ---------------------------------------------------------------------------------------

    def handle(self, action: str) -> bool:
        """Do one action from `decode_key`. Returns False when it is time to quit."""
        if action == "undo":
            self.undo()
        elif action == "refit":
            self.refit()
        else:
            return super().handle(action)
        return True


def main() -> None:
    """Command line entry point; see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("endpoint", nargs="?", default=f"tcp://localhost:{DEFAULT_PORT}")
    parser.add_argument("--arm", choices=["left", "right"], default="left")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--start-from", type=Path, help="an earlier calibration.urdf to start from")
    parser.add_argument(
        "--with",
        dest="other_sessions",
        nargs="*",
        type=Path,
        default=[],
        help="other session folders to fit with",
    )
    parser.add_argument("--n", type=int, default=40, help="number of target views (40)")
    parser.add_argument("--step", type=float, default=2.0, help="head jog per arrow key, degrees (2)")
    parser.add_argument("--max-vel", type=float, default=30.0, help="arm speed limit, deg/s (30)")
    parser.add_argument(
        "--delta-scale", type=float, default=1.0, help="scale of the target region (1 = +/-20 deg per joint)"
    )
    parser.add_argument(
        "--seed-pose", help='center of the target region, e.g. "shoulder_lift=-10,elbow_flex=10" (default: 0)'
    )
    parser.add_argument(
        "--side",
        choices=SIDES,
        help="side of the gripper the camera must see (default: the calibration's, positive)",
    )
    parser.add_argument("--floor-z", type=float, help="floor height in the robot's root frame, meters")
    parser.add_argument(
        "--max-roll",
        type=float,
        default=MAX_ROLL_DEG,
        help="largest wrist roll used to show the pinch point, deg (130)",
    )
    parser.add_argument("--resume", action="store_true", help="continue an earlier --out-dir")
    args = parser.parse_args()

    window = "calibration (click the pinch point; n = next view; q = quit)"
    clicks: list[tuple[int, int]] = []

    def on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))

    def show(image: np.ndarray) -> None:
        cv2.imshow(window, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with FelixClient(args.endpoint, timeout_s=20.0) as client, client.session():
        session = CollectSession(
            client,
            args.arm,
            args.out_dir,
            show,
            start_from=args.start_from,
            other_sessions=args.other_sessions,
            n_targets=args.n,
            max_vel=args.max_vel,
            head_step_deg=args.step,
            resume=args.resume,
            seed_pose=parse_seed(args.seed_pose),
            delta_scale=args.delta_scale,
            side=args.side,
            floor_z=args.floor_z,
            max_roll=args.max_roll,
        )
        for note in session.notes:
            print(note)
        cv2.setMouseCallback(window, on_mouse)
        try:
            while True:
                key = cv2.waitKeyEx(30)
                while clicks:
                    session.click(*clicks.pop(0))
                action = decode_key(key)
                if action is not None and not session.handle(action):
                    break
        finally:
            cv2.destroyAllWindows()
    print(f"{len(session.session.clicks)} clicks saved to {args.out_dir / SESSION_FILE}")
    path = session.finish()
    if path is not None:
        print(f"Wrote {path} and {path.with_suffix('.md')}")


if __name__ == "__main__":
    main()
