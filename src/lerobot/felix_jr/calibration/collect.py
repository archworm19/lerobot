"""Collect calibration clicks: you click the tip of the closed gripper in the camera image, pose after pose.

  uv run python -m lerobot.felix_jr.calibration.collect tcp://<pi>:5555 --arm left --out-dir ~/cal_left
      [--n 40] [--step 2] [--max-vel 30] [--delta-scale 1] [--resume]

Start the server on the Pi with the arm (and the head, which plugs into the left arm's board). Close the gripper
(the script commands it closed in every pose) and clear the workspace in front of the robot. A window shows the
head camera. For each pose the arm moves (slowly) and the head turns to where the current model predicts the tip
is; a cyan circle marks the predicted tip. Then:

  left mouse click    record the tip: click the very end of the closed gripper, where the fingertips meet
  n                   move the arm to the next planned pose (the first `n` is the seed pose, the wiggle's center)
  arrow keys          turn the head (up / down tilt, left / right pan) by --step degrees, e.g. to find the tip
  u                   undo the last click
  r                   refit now and print the report line
  q or Esc            finish (the robot is sent to rest)

At the start the script works out where the camera is pointing, so the first frames already show the gripper:
it reads the camera's pitch from the table plane in the depth image (override with `--head-tilt-deg`, how far
below horizontal the camera looks now), uses it as the head tilt's starting value, and searches for a first arm
pose whose tip lands in the middle of the picture at the head's current pose. Every planned pose keeps the tip above
the table and within what the head can bring into view. It prints what it decided (`--no-bootstrap` skips all of
this). If the tip is not visible after nudging the head, press `n` for another pose. The head's pan zero is still
unknown to about 20 degrees, so the circle can be off sideways at first; each click refits the model (`fit.py`),
so the circle gets closer and the head lands on the tip by itself. Vary the head too:
after a few poses, nudge the head a few degrees before clicking, so the head's parameters are not confounded with
the arm's. Aim for 30-60 clicks; the fit report says which parameters they pinned down.

Everything is saved as you go in `--out-dir`: `observations.json` (the clicks, the measured joint angles, the model
settings) and `images/` (the color and raw depth of every clicked frame). `--resume` continues an earlier folder.
Then run `calibration.fit observations.json --out-dir ...`.

Every click uses the *measured* joint angles reported with the frame, not the commanded ones. The script runs
inside a client session, so if you press Ctrl+C or the script dies, the robot goes to rest.
"""

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lerobot.felix_jr.calibration.bootstrap import (
    TILT_PRIOR_SIGMA,
    estimate_tilt_zero,
    estimate_up_vector,
    seed_pose_for_view,
    table_height,
    up_from_pitch,
)
from lerobot.felix_jr.calibration.design import MAX_ROLL_DEG, best_roll, center_head, plan_poses
from lerobot.felix_jr.calibration.fit import FitConfig, fit, pixel_errors, rms
from lerobot.felix_jr.calibration.model import (
    ARM_JOINTS,
    HEAD_JOINTS,
    TICKS_PER_TURN,
    CalibrationModel,
    Observation,
    head_priors,
    load_observations,
    save_observations,
)
from lerobot.felix_jr.depth import with_matrices
from lerobot.felix_jr.scene import Scene
from lerobot.felix_jr.server.client import Capture, FelixClient
from lerobot.felix_jr.server.protocol import DEFAULT_PORT, ServerError

TICKS_PER_DEG = TICKS_PER_TURN / 360.0
MIN_CLICKS_TO_FIT = 4
VIEW_MARGIN = 60  # pixels from the image border a planned tip must stay
NECK_RANGE_TICKS = (
    450  # +/- ticks (about 40 deg) the neck may be turned from where it is, if no limit is recorded
)
FACING_MAX_DEG = (
    50.0  # the gripper's hinge axis must be within this of the line of sight (pinch point visible)
)
NECK_PRIOR_NO_HOME = np.radians(45.0)  # head pan zero prior sigma when the neck has no recorded home
TABLE_CLEARANCE = 0.04  # meters the tip must stay above the table (the model is good to a centimeter or two)
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
    """Copy of the RGB `image` with the predicted tip (cyan circle), this frame's clicks (green) and text."""
    out = image.copy()
    if predicted is not None and np.isfinite(predicted).all():
        centre = (int(round(predicted[0])), int(round(predicted[1])))
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
        poses: list[dict[str, float]] | None,
        max_vel: float = 30.0,
        max_accel: float = 60.0,
        head_step_deg: float = 2.0,
        resume: bool = False,
        color_format: str = "png",
        fit_iterations: int = 10,
        n_poses: int = 40,
        delta_scale: float = 1.0,
        seed_pose: dict[str, float] | None = None,
        head_tilt_deg: float | None = None,
        bootstrap: bool = True,
        max_roll: float = MAX_ROLL_DEG,
        head_from: Path | None = None,
    ):
        """Connect the pieces and take a first frame.

        `poses` are the planned arm poses (`design.plan_poses`). With `poses=None` they are planned here, after the
        head tilt has been estimated: around `seed_pose` if given, else around a pose found by
        `bootstrap.seed_pose_for_view`, which shows the tip near the middle of the picture at the head's current
        pose. `head_tilt_deg` says how far below horizontal the camera looks now (default: estimated from the table
        plane in the depth image; `bootstrap=False` skips both). `notes` collects what was decided, for printing.

        `head_from` is the output folder of an earlier `fit` (usually the other arm's): its head and camera
        parameters become this session's priors, with its head reference ticks, so the head is known from the first
        frame and only this arm's parameters are left to learn (see `model.head_priors`).
        """
        self.client, self.arm, self.out_dir, self.show = client, arm, out_dir, show
        self.notes: list[str] = []
        self.table_z: float | None = (
            None  # the table's height in the root frame, if the depth image showed it
        )
        self.max_roll = max_roll
        self.arm_id = f"{arm}_arm"
        self.poses, self.pose_index = poses or [], 0
        self.max_vel, self.max_accel, self.head_step_deg = max_vel, max_accel, head_step_deg
        self.color_format, self.fit_iterations = color_format, fit_iterations
        self.message = "click the tip of the closed gripper; n = next pose"
        self.theta: dict[str, float] = {}
        self.last_rms: float | None = None
        self.frame_clicks: list[tuple[float, float]] = []
        (out_dir / "images").mkdir(parents=True, exist_ok=True)
        self.observations_path = out_dir / "observations.json"

        self.capture: Capture = client.capture(color_format=color_format)
        if resume and self.observations_path.is_file():
            self.model, self.observations = load_observations(self.observations_path)
            self.message = f"resumed with {len(self.observations)} clicks"
            self.theta.update({name: mean for name, (mean, _) in self.model.prior_overrides.items()})
            self._refit()
        elif head_from is not None:
            self.observations = []
            self._load_head(Path(head_from), client.head_state())
            if bootstrap:
                self._estimate_table()
        else:
            head = client.head_state()
            self.model = CalibrationModel(
                arm,
                self.capture.calibration["color_intrinsics"],
                {n: head[n]["sign"] for n in HEAD_JOINTS},
                {
                    n: head[n]["home"] if head[n]["home"] is not None else head[n]["ticks"]
                    for n in HEAD_JOINTS
                },
            )
            self.observations: list[Observation] = []
            if head["head_neck"]["home"] is None:
                # its reference is then wherever the neck happens to be, so its zero can be anything
                self.model.prior_overrides["zero:head_pan_joint"] = (0.0, NECK_PRIOR_NO_HOME)
                self._note(
                    "the neck has no recorded home, so the head's pan is unknown at first (record it with "
                    "setup.head_range --name head_neck); expect the circle to be off sideways until a few clicks"
                )
            if bootstrap:
                self._estimate_head_tilt(head_tilt_deg)
        if poses is None:
            self.poses = self._plan(n_poses, delta_scale, seed_pose, bootstrap)
        self.render()

    # ---- the start of a session ----------------------------------------------------------------------------

    def _note(self, text: str) -> None:
        self.notes.append(text)
        self.message = text

    def _load_head(self, folder: Path, head: dict[str, dict[str, Any]]) -> None:
        """Build the model with the head and camera of an earlier fit (see `__init__`, `head_from`)."""
        calibration = json.loads((folder / "kinematic_calibration.json").read_text())
        earlier = calibration["settings"]
        signs_now = {n: head[n]["sign"] for n in HEAD_JOINTS}
        if signs_now != earlier["head_signs"]:
            raise ValueError(
                f"the head's saved directions changed since that fit ({earlier['head_signs']} then, {signs_now} now): "
                "its head parameters do not apply; run without --head-from"
            )
        priors = head_priors(calibration)
        self.model = CalibrationModel(
            self.arm,
            self.capture.calibration["color_intrinsics"],
            earlier["head_signs"],
            earlier["head_refs"],  # the same reference ticks, so its zero offsets mean the same thing here
            prior_overrides=priors,
        )
        self.theta.update({name: mean for name, (mean, _) in priors.items()})
        tilt_mean, tilt_sigma = priors["zero:head_tilt_joint"]
        pan_mean, pan_sigma = priors["zero:head_pan_joint"]
        lateral = priors["pos:fixed_head_camera_link:y"]
        self._note(
            f"head and camera from {folder} ({earlier['arm']} arm fit): tilt zero {np.degrees(tilt_mean):+.1f} "
            f"+/- {np.degrees(tilt_sigma):.1f} deg, pan zero {np.degrees(pan_mean):+.1f} +/- {np.degrees(pan_sigma):.1f} deg, "
            f"camera lateral offset {1000 * lateral[0]:+.1f} +/- {1000 * lateral[1]:.1f} mm"
        )

    def _estimate_table(self) -> None:
        """Find the table's height from the depth image (when the head is already known)."""
        plane = estimate_up_vector(self.capture.scene())
        if plane is None:
            self._note("no table plane in the depth image: poses are not checked against the table")
            return
        ticks = {n: int(self.capture.state["head"][n]["ticks"]) for n in HEAD_JOINTS}
        self.table_z = table_height(self.model, ticks, self.theta, plane)
        base_z = self.model.head_tree.link_pose(self.model.base_link)[2, 3]
        self.notes.append(f"the table is {100 * (base_z - self.table_z):.0f} cm below the arm base")

    def _estimate_head_tilt(self, pitch_deg: float | None) -> None:
        """Set the head tilt zero (as a prior with a 5 degree sigma) from the camera's pitch.

        The pitch is given (`pitch_deg`, degrees below horizontal) or measured from the table plane in the depth
        image. Leaves the zero at the model's default if neither is available or the estimate does not fit.
        """
        ticks = {n: int(self.capture.state["head"][n]["ticks"]) for n in HEAD_JOINTS}
        plane = None
        if pitch_deg is not None:
            up, source = (
                up_from_pitch(np.radians(pitch_deg)),
                f"given as {pitch_deg:.1f} deg below horizontal",
            )
        else:
            found = estimate_up_vector(self.capture.scene())
            if found is None:
                self._note(
                    "no table plane in the depth image: the head tilt starts at the model's zero (about 20 deg "
                    "unsure); give --head-tilt-deg if you can estimate how far down the camera looks"
                )
                return
            plane = found
            up, fraction = plane.normal, plane.fraction
            source = f"table plane ({100 * fraction:.0f}% of the depth points)"
        tilt, miss = estimate_tilt_zero(self.model, ticks, up)
        if miss > 6.0:
            self._note(
                f"the {source} does not match the model's camera (off by {miss:.1f} deg): head tilt left at zero"
            )
            return
        self.model.prior_overrides["zero:head_tilt_joint"] = (tilt, TILT_PRIOR_SIGMA)
        self.theta["zero:head_tilt_joint"] = tilt
        if plane is not None:
            self.table_z = table_height(self.model, ticks, self.theta, plane)
            base_z = self.model.head_tree.link_pose(self.model.base_link)[2, 3]
            self.notes.append(f"the table is {100 * (base_z - self.table_z):.0f} cm below the arm base")
        pitch = float(np.degrees(np.arcsin(np.clip(-up[2], -1.0, 1.0))))
        self._note(
            f"camera looks {pitch:.1f} deg below horizontal ({source}); head tilt zero set to {np.degrees(tilt):+.1f} deg"
        )

    def _plan(
        self, n: int, delta_scale: float, seed_pose: dict[str, float] | None, bootstrap: bool
    ) -> list[dict[str, float]]:
        """The arm poses to visit: around `seed_pose`, or around a pose that shows the tip at the current head pose."""
        ticks = {name: int(self.capture.state["head"][name]["ticks"]) for name in HEAD_JOINTS}
        if seed_pose is None and bootstrap:
            seed_pose = seed_pose_for_view(self.model, self.theta, ticks, table_z=self.table_z)
            if seed_pose is None:
                self._note("no arm pose shows the tip at this head pose: starting from the center pose")
            else:
                shown = ", ".join(
                    f"{k.removesuffix('.pos')} {v:+.0f}" for k, v in seed_pose.items() if k != "gripper.pos"
                )
                self._note(f"first pose: {shown}")
        if not bootstrap:
            return plan_poses(n, seed_pose, delta_scale)
        limits = head_limits(self.client.head_state())
        floor = None if self.table_z is None else self.table_z + TABLE_CLEARANCE

        def adjust(
            pose: dict[str, float], head_ticks: dict[str, int] | None = None
        ) -> dict[str, float] | None:
            """The pose with the wrist rolled to show the pinch point, or None if it cannot be shown.

            It must stay above the table, the head must be able to bring the tip into the picture within its
            limits, and the gripper's hinge axis must then face the camera within `FACING_MAX_DEG`.
            """
            if floor is not None and self.model.tip_in_root(pose, self.theta)[2] < floor:
                return None
            target = head_ticks or center_head(self.model, self.theta, pose, ticks, limits)
            if target is None:
                return None
            roll, angle = best_roll(self.model, self.theta, pose, target, self.max_roll)
            if angle > FACING_MAX_DEG:
                return None
            pose = {**pose, "wrist_roll.pos": roll}
            tip = self.model.predict(Observation(arm=pose, head_ticks=target, pixel=(0.0, 0.0)), self.theta)
            return pose if self.model.in_view(tip.uv, margin=VIEW_MARGIN) else None

        if seed_pose is not None:  # roll the seed too, for the head where it is now
            full = {**{f"{j}.pos": 0.0 for j in ARM_JOINTS}, **seed_pose, "gripper.pos": 1.0}
            seed_pose = adjust(full, ticks) or {
                **full,
                "wrist_roll.pos": best_roll(self.model, self.theta, full, ticks, self.max_roll)[0],
            }
        poses = plan_poses(n, seed_pose, delta_scale, adjust=adjust)
        if len(poses) < n:
            self._note(
                f"only {len(poses)} of {n} poses show the pinch point within the head's range and above the table"
            )
        return poses

    # ---- showing --------------------------------------------------------------------------------------------

    def current_observation(self) -> Observation:
        """The robot's measured joints in the current frame, as an Observation with no click yet."""
        state = self.capture.state
        return Observation(
            arm=dict(state[self.arm_id]),
            head_ticks={n: int(state["head"][n]["ticks"]) for n in HEAD_JOINTS},
            pixel=(0.0, 0.0),
            timestamp=self.capture.timestamp,
        )

    def status_lines(self, predicted: np.ndarray | None) -> list[str]:
        """Text for the image: counts, the model's current accuracy, and the last message."""
        fit_text = f"fit RMS {self.last_rms:.1f} px" if self.last_rms is not None else "no fit yet"
        lines = [
            f"{self.arm} arm: {len(self.observations)} clicks, pose {self.pose_index}/{len(self.poses)}, {fit_text}",
            "click the tip | n next pose | arrows head | u undo | r refit | q quit",
        ]
        if predicted is None or not self.model.in_view(predicted):
            lines.append("predicted tip is off the picture: nudge the head, or press n")
        if self.message:
            lines.append(self.message)
        return lines

    def render(self) -> None:
        """Show the current frame with the predicted tip, this frame's clicks and the status text."""
        pred = self.model.predict(self.current_observation(), self.theta)
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
        """Move the arm to the next planned pose, turn the head towards the predicted tip, and show a new frame."""
        if self.pose_index >= len(self.poses):
            self.message = "no more planned poses: click what is left, then q"
            self.render()
            return
        pose = self.poses[self.pose_index]
        self.pose_index += 1
        status = self.client.move_joints({self.arm_id: pose}, max_vel=self.max_vel, max_accel=self.max_accel)
        if status["state"] != "done":
            self.message = f"the arm motion ended as {status['state']}: {status['reason']}"
            self.refresh(status["finished_at"])
            return
        state = self.client.get_state()
        head = self.client.head_state()
        targets = center_head(
            self.model,
            self.theta,
            state[self.arm_id],
            {n: head[n]["ticks"] for n in HEAD_JOINTS},
            head_limits(head),
        )
        done = status["finished_at"]
        if targets is not None:
            done = self._move_head_to(targets) or done
            self.message = f"pose {self.pose_index}: head turned towards the predicted tip"
        else:
            self.message = f"pose {self.pose_index}: the tip is predicted behind the camera; steer the head"
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

    def click(self, u: float, v: float) -> Observation:
        """Record a click at pixel (u, v) of the current frame, save it, and refit."""
        obs = self.current_observation()
        obs.pixel = (float(u), float(v))
        obs.index = len(self.observations)
        obs.image = f"images/{obs.index:03d}.png"
        scene = Scene.from_capture(
            self.capture.color, self.capture.raw_depth, with_matrices(self.capture.calibration)
        )
        depth = float(scene.pixel_to_xyz(np.array([[u, v]]), window=7, max_spread_m=0.06)[0, 2])
        obs.depth_m = depth if np.isfinite(depth) else None
        cv2.imwrite(str(self.out_dir / obs.image), cv2.cvtColor(self.capture.color, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(self.out_dir / f"images/{obs.index:03d}_depth.png"), self.capture.raw_depth)
        self.observations.append(obs)
        self.frame_clicks.append(obs.pixel)
        save_observations(self.observations_path, self.model, self.observations)
        self.message = f"click {len(self.observations)} saved"
        self._refit()
        self.render()
        return obs

    def undo(self) -> None:
        """Drop the last click (its images stay in the folder) and refit."""
        if self.observations:
            self.observations.pop()
            if self.frame_clicks:
                self.frame_clicks.pop()
            save_observations(self.observations_path, self.model, self.observations)
            self.message = f"undid the last click ({len(self.observations)} left)"
            self._refit()
        self.render()

    def _refit(self) -> None:
        if len(self.observations) < MIN_CLICKS_TO_FIT:
            self.last_rms = None
            return
        result = fit(
            self.model,
            self.observations,
            cfg=FitConfig(max_iter=self.fit_iterations),
            theta0=self.theta,
        )
        self.theta = result.theta
        self.last_rms = rms(result.pixel_errors)

    def refit(self) -> None:
        """Refit with more iterations and say how well the model explains the clicks."""
        self.fit_iterations, previous = 40, self.fit_iterations
        try:
            self._refit()
        finally:
            self.fit_iterations = previous
        if self.last_rms is not None:
            before = rms(pixel_errors(self.model, self.observations, {}))
            self.message = f"refit: {before:.0f} px uncalibrated -> {self.last_rms:.1f} px"
        else:
            self.message = f"need {MIN_CLICKS_TO_FIT} clicks to fit"
        self.render()

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
    """`"shoulder_pan=10,elbow_flex=-5"` -> `{"shoulder_pan": 10.0, "elbow_flex": -5.0}` (None for no text)."""
    if not text:
        return None
    return {k.strip(): float(v) for k, v in (part.split("=") for part in text.split(","))}


def main() -> None:
    """Command line entry point; see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("endpoint", nargs="?", default=f"tcp://localhost:{DEFAULT_PORT}")
    parser.add_argument("--arm", choices=["left", "right"], default="left")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=40, help="number of planned arm poses (40)")
    parser.add_argument("--step", type=float, default=2.0, help="head jog per arrow key, degrees (2)")
    parser.add_argument("--max-vel", type=float, default=30.0, help="arm speed limit, deg/s (30)")
    parser.add_argument(
        "--delta-scale", type=float, default=1.0, help="scale of the pose region (1 = +/-20 deg)"
    )
    parser.add_argument(
        "--seed-pose", help='region center, e.g. "shoulder_lift=-10,elbow_flex=10" (default: searched for)'
    )
    parser.add_argument(
        "--head-tilt-deg",
        type=float,
        help="how far below horizontal the camera looks now (default: from the table plane)",
    )
    parser.add_argument(
        "--no-bootstrap", action="store_true", help="skip the head tilt estimate and the first-pose search"
    )
    parser.add_argument(
        "--max-roll",
        type=float,
        default=MAX_ROLL_DEG,
        help="largest wrist roll used to show the pinch point, deg (100)",
    )
    parser.add_argument("--resume", action="store_true", help="continue an earlier --out-dir")
    parser.add_argument(
        "--head-from",
        type=Path,
        help="fit output folder of the other arm: reuse its head and camera parameters (as priors)",
    )
    args = parser.parse_args()

    window = "calibration (click the tip; n = next pose; q = quit)"
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
            None,
            max_vel=args.max_vel,
            head_step_deg=args.step,
            resume=args.resume,
            n_poses=args.n,
            delta_scale=args.delta_scale,
            seed_pose=parse_seed(args.seed_pose),
            head_tilt_deg=args.head_tilt_deg,
            bootstrap=not args.no_bootstrap,
            max_roll=args.max_roll,
            head_from=args.head_from,
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
    print(f"{len(session.observations)} clicks saved to {session.observations_path}")
    print(
        f"Next: uv run python -m lerobot.felix_jr.calibration.fit {session.observations_path} --out-dir {args.out_dir}"
    )


if __name__ == "__main__":
    main()
