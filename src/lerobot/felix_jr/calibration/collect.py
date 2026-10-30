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

from lerobot.felix_jr.calibration.views import MAX_ROLL_DEG, Target, find_view, plan_targets
from lerobot.felix_jr.calibration.xle_fit import (
    SESSION_FILE,
    Click,
    FitConfig,
    Session,
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
    TICKS_PER_TURN,
    XleRobotModel,
    default_parameters,
)
from lerobot.felix_jr.depth import with_matrices
from lerobot.felix_jr.scene import Scene
from lerobot.felix_jr.server.client import Capture, FelixClient
from lerobot.felix_jr.server.protocol import DEFAULT_PORT, ServerError

TICKS_PER_DEG = TICKS_PER_TURN / 360.0
MIN_CLICKS_TO_FIT = 4
NECK_RANGE_TICKS = (
    450  # +/- ticks (about 40 deg) the neck may be turned from where it is, if no limit is recorded
)
NECK_PRIOR_NO_HOME = np.radians(45.0)  # head pan zero prior sigma when the neck has no recorded home
REFIT_EVALUATIONS = (
    20  # optimizer evaluations per refit after a click (`r` and the final fit run to convergence)
)
CALIBRATION_FILE = "calibration.urdf"
INTRINSICS_KEYS = ("fx", "fy", "ppx", "ppy", "width", "height")
# cv2.waitKeyEx codes differ by platform (macOS, Linux GTK, Windows, Qt).
UP_KEYS = {63232, 65362, 2490368, 0, ord("i")}
DOWN_KEYS = {63233, 65364, 2621440, 1, ord("k")}
LEFT_KEYS = {63234, 65361, 2424832, 2, ord("j")}
RIGHT_KEYS = {63235, 65363, 2555904, 3, ord("l")}
KEY_ACTIONS = {ord("n"): "next", ord("u"): "undo", ord("r"): "refit", ord("q"): "quit", 27: "quit"}


def decode_key(key: int) -> str | None:
    """The action for a `cv2.waitKeyEx` code: up, down, left, right, next, undo, refit, quit; None if none."""
    for keys, action in ((UP_KEYS, "up"), (DOWN_KEYS, "down"), (LEFT_KEYS, "left"), (RIGHT_KEYS, "right")):
        if key in keys:
            return action
    return KEY_ACTIONS.get(key)


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


class CollectSession:
    """One calibration run: moves the robot, shows frames, records clicks, refits after each. See the module docstring."""

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
        self.client, self.arm, self.out_dir, self.show = client, arm, out_dir, show
        self.arm_id = f"{arm}_arm"
        self.notes: list[str] = []
        self.max_vel, self.max_accel, self.head_step_deg = max_vel, max_accel, head_step_deg
        self.color_format, self.max_roll, self.refit_evaluations = color_format, max_roll, refit_evaluations
        self.message = "click the pinch point of the closed gripper; n = next view"
        self.last_rms: float | None = None
        self.frame_clicks: list[tuple[float, float]] = []
        (out_dir / "images").mkdir(parents=True, exist_ok=True)

        self.capture: Capture = client.capture(color_format=color_format)
        intrinsics = {k: self.capture.calibration["color_intrinsics"][k] for k in INTRINSICS_KEYS}
        self.model = self._build_model(start_from, client.head_state(), intrinsics)
        if side is not None:
            if side not in SIDES:
                raise ValueError(f"side must be one of {SIDES}")
            self.model.visible_side[arm] = side
        self._set_floor(floor_z)

        if resume and (out_dir / SESSION_FILE).is_file():
            self.session = load_session(out_dir)
            if self.session.arm != arm:
                raise ValueError(f"{out_dir} holds {self.session.arm} arm clicks, not {arm}")
            self.message = f"resumed with {len(self.session.clicks)} clicks"
        else:
            self.session = Session(out_dir, arm, [], self.model.head_signs, self.model.head_refs, intrinsics)
        self.others = [load_session(path) for path in other_sessions]
        here = Session(out_dir, arm, [], self.model.head_signs, self.model.head_refs, intrinsics)
        check_compatible([here, self.session, *self.others])
        if self.others:
            self._note(
                f"refitting with {sum(len(s.clicks) for s in self.others)} clicks from the other sessions"
            )

        self.targets: list[Target] = plan_targets(self.model, arm, n_targets, seed_pose, delta_scale)
        self.target_index = 0
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

    def _set_floor(self, floor_z: float | None) -> None:
        if floor_z is not None:
            self.model.floor_z = floor_z
        elif self.model.floor_z is None:
            base = self.model.frame_pose(f"{self.arm}_base_link", self.model.joint_vector())
            self.model.floor_z = float(base[2, 3])
            self._note(
                "no floor height known: keeping the pinch point above the arm base's height (--floor-z)"
            )

    # ---- showing --------------------------------------------------------------------------------------------

    def current_q(self) -> np.ndarray:
        """The robot's measured joints in the current frame, as the model's `q`."""
        state = self.capture.state
        return self.model.joint_vector(
            {self.arm: state[self.arm_id]}, {n: int(state["head"][n]["ticks"]) for n in HEAD_JOINTS}
        )

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
        """Take a new frame (taken after server time `after`, if given) and show it."""
        self.capture = self.client.capture(color_format=self.color_format, after=after)
        self.frame_clicks = []
        self.render()

    # ---- moving ---------------------------------------------------------------------------------------------

    def _move_head_to(self, targets: dict[str, int]) -> float | None:
        """Move each head servo to raw ticks `targets`. Returns the finish time of the last move."""
        head = self.client.head_state()
        done = None
        for name, target in targets.items():
            delta = head[name]["sign"] * (target - head[name]["ticks"]) / TICKS_PER_DEG
            if abs(delta) * TICKS_PER_DEG >= 2:
                done = self.client.head_move(name, delta_deg=delta)["done_at"]
        return done

    def next_pose(self) -> None:
        """Find a view of the next target that shows the pinch point, move the arm and the head there, show it.

        Targets whose best view does not show the pinch point (`View.ok`) are skipped.
        """
        head = self.client.head_state()
        limits = head_limits(head)
        ticks = {n: int(head[n]["ticks"]) for n in HEAD_JOINTS}
        skipped = 0
        while self.target_index < len(self.targets):
            target = self.targets[self.target_index]
            self.target_index += 1
            seed = self.model.joint_vector({self.arm: target.arm_pose}, ticks)
            view = find_view(self.model, self.arm, seed, target.point, limits, max_roll_deg=self.max_roll)
            if view.ok:
                break
            skipped += 1
        else:
            self.message = "no more views: click what is left, then q"
            self.render()
            return
        status = self.client.move_joints(
            {self.arm_id: view.arm_pose}, max_vel=self.max_vel, max_accel=self.max_accel
        )
        if status["state"] != "done":
            self.message = f"the arm motion ended as {status['state']}: {status['reason']}"
            self.refresh(status["finished_at"])
            return
        done = self._move_head_to(view.head_ticks) or status["finished_at"]
        skip_text = f" ({skipped} targets skipped: no view showed them)" if skipped else ""
        self.message = (
            f"view {self.target_index}: gripper facing the camera within {view.facing_deg:.0f} deg{skip_text}"
        )
        self.refresh(done)

    def jog(self, neck_deg: float = 0.0, eyes_deg: float = 0.0) -> None:
        """Turn the head by the given degrees (positive = left / down) and show a new frame."""
        done = None
        try:
            for name, delta in (("head_neck", neck_deg), ("head_eyes", eyes_deg)):
                if delta:
                    result = self.client.head_move(name, delta_deg=delta)
                    done = result["done_at"]
                    self.message = "" if not result["clamped"] else f"{name} is at its recorded limit"
        except ServerError as e:
            self.message = str(e)
        self.refresh(done)

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
        """Fit every click with held-out validation and write `calibration.urdf` and `calibration.md`; None if
        there are too few clicks."""
        clicks = self.all_clicks()
        if len(clicks) < MIN_CLICKS_TO_FIT:
            return None
        cfg = FitConfig()
        final, holdout = calibrate(self.model, clicks, cfg, holdout_every)
        path = self.out_dir / CALIBRATION_FILE
        self.model.save(path)
        path.with_suffix(".md").write_text(report(self.model, clicks, final, holdout, cfg))
        return path

    # ---- key handling ---------------------------------------------------------------------------------------

    def handle(self, action: str) -> bool:
        """Do one action from `decode_key`. Returns False when it is time to quit."""
        step = self.head_step_deg
        if action == "quit":
            return False
        if action == "next":
            self.next_pose()
        elif action == "undo":
            self.undo()
        elif action == "refit":
            self.refit()
        elif action in ("up", "down"):
            self.jog(eyes_deg=step if action == "down" else -step)  # the eyes' positive direction is down
        elif action in ("left", "right"):
            self.jog(neck_deg=step if action == "left" else -step)  # the neck's positive direction is left
        return True


def head_limits(head: dict[str, dict[str, Any]]) -> dict[str, tuple[int, int]]:
    """Allowed raw ticks per head servo from `client.head_state()`: the recorded limits, else a range around now.

    Limits are recorded as the servo's positive / negative direction (down / left, up / right), in raw ticks, so
    the lower and upper bound are whichever of the two is smaller. Without recorded limits the servo may move
    `NECK_RANGE_TICKS` either side of where it is now (nothing is known about the cable or the mount).
    """
    limits = {}
    for name, servo in head.items():
        if servo["limit_neg"] is not None and servo["limit_pos"] is not None:
            low, high = sorted((servo["limit_neg"], servo["limit_pos"]))
        else:
            low, high = servo["ticks"] - NECK_RANGE_TICKS, servo["ticks"] + NECK_RANGE_TICKS
        limits[name] = (max(low, 0), min(high, TICKS_PER_TURN - 1))
    return limits


def parse_seed(text: str | None) -> dict[str, float] | None:
    """`"shoulder_pan=10,elbow_flex=-5"` -> `{"shoulder_pan.pos": 10.0, "elbow_flex.pos": -5.0}` (None for no text)."""
    if not text:
        return None
    return {f"{k.strip()}.pos": float(v) for k, v in (part.split("=") for part in text.split(","))}


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
