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

If the tip is not visible after nudging the head, press `n` for another pose. The first few clicks are the hard
ones: the head's zero offsets are unknown to about 20 degrees, so the tip may be far from the circle. Each click
refits the model (`fit.py`), so the circle gets closer and the head lands on the tip by itself. Vary the head too:
after a few poses, nudge the head a few degrees before clicking, so the head's parameters are not confounded with
the arm's. Aim for 30-60 clicks; the fit report says which parameters they pinned down.

Everything is saved as you go in `--out-dir`: `observations.json` (the clicks, the measured joint angles, the model
settings) and `images/` (the color and raw depth of every clicked frame). `--resume` continues an earlier folder.
Then run `calibration.fit observations.json --out-dir ...`.

Every click uses the *measured* joint angles reported with the frame, not the commanded ones. The script runs
inside a client session, so if you press Ctrl+C or the script dies, the robot goes to rest.
"""

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lerobot.felix_jr.calibration.design import center_head, plan_poses
from lerobot.felix_jr.calibration.fit import FitConfig, fit, pixel_errors, rms
from lerobot.felix_jr.calibration.model import (
    HEAD_JOINTS,
    TICKS_PER_TURN,
    CalibrationModel,
    Observation,
    load_observations,
    save_observations,
)
from lerobot.felix_jr.depth import with_matrices
from lerobot.felix_jr.scene import Scene
from lerobot.felix_jr.server.client import Capture, FelixClient
from lerobot.felix_jr.server.protocol import DEFAULT_PORT, ServerError

TICKS_PER_DEG = TICKS_PER_TURN / 360.0
MIN_CLICKS_TO_FIT = 4
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
        poses: list[dict[str, float]],
        max_vel: float = 30.0,
        max_accel: float = 60.0,
        head_step_deg: float = 2.0,
        resume: bool = False,
        color_format: str = "png",
        fit_iterations: int = 10,
    ):
        """Connect the pieces and take a first frame. `poses` are the planned arm poses (`design.plan_poses`)."""
        self.client, self.arm, self.out_dir, self.show = client, arm, out_dir, show
        self.arm_id = f"{arm}_arm"
        self.poses, self.pose_index = poses, 0
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
            self._refit()
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
        self.render()

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
        limits = {
            n: (head[n]["limit_neg"], head[n]["limit_pos"])
            if head[n]["limit_neg"] is not None and head[n]["limit_pos"] is not None
            else (0, TICKS_PER_TURN - 1)
            for n in head
        }
        # head limits are recorded as "pos" = the servo's positive direction (down / left), in raw ticks
        limits = {n: (min(lo, hi), max(lo, hi)) for n, (lo, hi) in limits.items()}
        targets = center_head(
            self.model, self.theta, state[self.arm_id], {n: head[n]["ticks"] for n in HEAD_JOINTS}, limits
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
        "--seed-pose", help='region center, e.g. "shoulder_lift=-10,elbow_flex=10" (default: all 0)'
    )
    parser.add_argument("--resume", action="store_true", help="continue an earlier --out-dir")
    args = parser.parse_args()

    window = "calibration (click the tip; n = next pose; q = quit)"
    clicks: list[tuple[int, int]] = []

    def on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))

    def show(image: np.ndarray) -> None:
        cv2.imshow(window, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    poses = plan_poses(args.n, parse_seed(args.seed_pose), args.delta_scale)
    with FelixClient(args.endpoint, timeout_s=20.0) as client, client.session():
        session = CollectSession(
            client,
            args.arm,
            args.out_dir,
            show,
            poses,
            max_vel=args.max_vel,
            head_step_deg=args.step,
            resume=args.resume,
        )
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
