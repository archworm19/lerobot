"""Check a calibration on the robot: the model's two fingertips drawn over the head camera, view after view.

  uv run python -m lerobot.felix_jr.calibration.check tcp://<pi>:5555 --arm right --model <calibration.urdf>
      [--openings 0,15,30,50,70] [--gripper-step 10] [--out-dir DIR] [--n 20] [--step 2] [--max-vel 30]
      [--seed-pose "shoulder_lift=-10"] [--delta-scale 1] [--floor-z Z] [--max-roll 130]

Start the server on the Pi with the arm (and the head, which plugs into the left arm's board), and clear the
workspace, as for `collect`. Nothing is clicked or fitted. `n` moves the arm and the head to the next view (the same
views as `collect`: the pinch point in the middle of the picture, the jaws in profile) with the gripper opened to the
next of `--openings` (LeRobot's `gripper.pos`), and the frame shows where the model puts:

  cyan circle       the pinch point: the fixed jaw's tip (placed by the camera calibration)
  magenta circle    the moving jaw's tip (placed by the gripper map and the hinge too), with a line between them
  yellow outlines   the wrist cameras' plates the calibration has, both arms' (`calibration.arm_shapes`)
  orange outlines   the arms' capsules (fingers, gripper body, wrist, forearm) it has (`calibration.arm_shapes`)

computed from the joint angles and the `gripper.pos` measured with the frame. If the calibration is right, the two
circles sit on the two fingertips at every view and every opening; how far off they are, and whether it changes
with the pose, the head or the opening, says which part is wrong.

  n                   next view, with the next opening
  [ / ]               close / open the gripper by --gripper-step (gripper.pos), at this view
  arrow keys          turn the head by --step degrees (the circles should follow the fingertips)
  q or Esc            quit (the robot is sent to rest)

The model must match the robot (`XleRobotModel.check_robot`) and have a gripper map for the arm
(`calibration.gripper_step`). With `--out-dir`, every frame is saved as it is shown: `NNN.png` (as drawn) and
`NNN_color.png`, and `frames.json` lists, per frame, the measured state, the opening and both points' predicted
pixels and positions. The script runs inside a client session, so on Ctrl+C or a crash the robot goes to rest.
"""

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lerobot.felix_jr.calibration.arm_shapes import CAPSULE_COLOR, capsule_outline
from lerobot.felix_jr.calibration.live_view import (
    INTRINSICS_KEYS,
    LiveView,
    decode_key,
    parse_seed,
    set_floor,
)
from lerobot.felix_jr.calibration.views import MAX_ROLL_DEG
from lerobot.felix_jr.calibration.xle_robot_model import HEAD_JOINTS, Prediction, XleRobotModel
from lerobot.felix_jr.server.client import FelixClient
from lerobot.felix_jr.server.protocol import DEFAULT_PORT

DEFAULT_OPENINGS = (0.0, 15.0, 30.0, 50.0, 70.0)  # gripper.pos per view, in turn
PINCH_COLOR, MOVING_COLOR, BOARD_COLOR = (
    (0, 255, 255),
    (255, 0, 255),
    (255, 220, 0),
)  # RGB: cyan, magenta, yellow
KEY_ACTIONS = {ord("n"): "next", ord("["): "close", ord("]"): "open", ord("q"): "quit", 27: "quit"}
FRAMES_FILE = "frames.json"


def annotate(
    image: np.ndarray,
    lines: list[str],
    points: Sequence[tuple[np.ndarray | None, tuple, str]],
    outlines: Sequence[np.ndarray] = (),
    capsules: Sequence[np.ndarray] = (),
) -> np.ndarray:
    """Copy of the RGB `image` with each `(pixel, color, label)` as a labeled circle, and text at the top left.

    A line joins the first two drawn points; a point with no pixel (None, or behind the camera) is left out. Each of
    `outlines` (N, 2 pixels) is drawn as a closed yellow polygon: a wrist camera's plate; each of `capsules` as an
    orange one: a part of an arm.
    """
    out = image.copy()
    for shapes, color in ((outlines, BOARD_COLOR), (capsules, CAPSULE_COLOR)):
        for outline in shapes:
            if np.isfinite(outline).all():
                polygon = np.round(outline).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(out, [polygon], True, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.polylines(out, [polygon], True, color, 1, cv2.LINE_AA)
    drawn = [(uv, color, label) for uv, color, label in points if uv is not None and np.isfinite(uv).all()]
    if len(drawn) >= 2:
        a, b = (tuple(int(round(float(c))) for c in uv) for uv, _, _ in drawn[:2])
        cv2.line(out, a, b, (255, 255, 255), 1, cv2.LINE_AA)
    for uv, color, label in drawn:
        centre = (int(round(float(uv[0]))), int(round(float(uv[1]))))
        cv2.circle(out, centre, 12, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.circle(out, centre, 12, color, 1, cv2.LINE_AA)
        org = (centre[0] + 15, centre[1] - 8)
        cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    for row, text in enumerate(lines):
        org = (8, 20 + 20 * row)
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


class CheckSession(LiveView):
    """One check of a calibration: moves the robot, opens the gripper, draws the model's fingertips.

    See the module docstring.
    """

    def __init__(
        self,
        client: FelixClient,
        model: XleRobotModel,
        arm: str,
        show: Callable[[np.ndarray], None],
        openings: Sequence[float] = DEFAULT_OPENINGS,
        gripper_step: float = 10.0,
        out_dir: Path | None = None,
        n_targets: int = 20,
        max_vel: float = 30.0,
        max_accel: float = 60.0,
        head_step_deg: float = 2.0,
        color_format: str = "png",
        seed_pose: dict[str, float] | None = None,
        delta_scale: float = 1.0,
        floor_z: float | None = None,
        max_roll: float = MAX_ROLL_DEG,
    ):
        """Check that `model` is for this robot and has the arm's gripper map, plan the views, show a first frame.

        Raises `RobotMismatchError` (from `check_robot`) or `ValueError` (no gripper map) before anything moves.
        """
        if arm not in model.gripper_maps:
            raise ValueError(
                f"the calibration has no gripper map for the {arm} arm: run calibration.gripper_step"
            )
        model.check_robot(client)
        self.notes: list[str] = []
        capture = client.capture(color_format=color_format)
        intrinsics = {k: capture.calibration["color_intrinsics"][k] for k in INTRINSICS_KEYS}
        if model.intrinsics != intrinsics:
            self.notes.append("the camera's intrinsics differ from the calibration's: using the camera's")
            model.intrinsics = intrinsics
        set_floor(model, arm, floor_z, self.notes.append)
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
        self.openings, self.gripper_step, self.out_dir = [float(o) for o in openings], gripper_step, out_dir
        self.opening_index = 0
        self.frames: list[dict[str, Any]] = []
        self.message = self.notes[-1] if self.notes else "n = next view"
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
        self.refresh()

    # ---- what the model says about this frame ---------------------------------------------------------------

    def gripper_pos(self) -> float:
        """The gripper's `gripper.pos` measured with the current frame."""
        return float(self.capture.state[self.arm_id]["gripper.pos"])

    def predictions(self) -> tuple[float, Prediction, Prediction]:
        """The jaws' opening (degrees) at the measured `gripper.pos`, and the model's pinch point and moving tip.

        A gripper squeezed past where the tips meet (a negative opening) has its moving tip at the pinch point.
        """
        q = self.current_q()
        opening = self.model.opening_deg(self.arm, self.gripper_pos())
        pinch = self.model.predict(self.arm, q)
        moving = self.model.predict(self.arm, q, point="moving_tip", opening_deg=max(opening, 0.0))
        return opening, pinch, moving

    def render(self) -> None:
        """Show the current frame with the predicted fingertips and the status text."""
        self.image = self._annotated()
        self.show(self.image)

    def _annotated(self) -> np.ndarray:
        opening, pinch, moving = self.predictions()
        apart = float(np.linalg.norm(moving.point - pinch.point))
        lines = [
            f"{self.arm} arm: view {self.target_index}/{len(self.targets)}",
            f"gripper.pos {self.gripper_pos():.1f}: opening {opening:.0f} deg, tips {apart * 1000:.0f} mm apart",
            "n next view | [ ] close / open | arrows head | q quit",
        ]
        if not all(p.z > 0 and self.model.in_view(p.uv) for p in (pinch, moving)):
            lines.append("a predicted fingertip is off the picture: turn the head, or press n")
        if self.message:
            lines.append(self.message)
        points = [
            (pinch.uv if pinch.z > 0 else None, PINCH_COLOR, "pinch"),
            (moving.uv if moving.z > 0 else None, MOVING_COLOR, "moving tip"),
        ]
        return annotate(self.capture.color, lines, points, self.board_outlines(), self.capsule_outlines())

    def board_outlines(self) -> list[np.ndarray]:
        """Each wrist camera plate the calibration has (both arms'), as its seen face in the current frame.

        Each is (4, 2) pixels; a plate behind the camera is left out. The other arm's plate is placed at the joints this frame measured for it, if any, else at its rest in the URDF.
        """
        state = self.capture.state
        arms = {arm: state[f"{arm}_arm"] for arm in self.model.wrist_boards if f"{arm}_arm" in state}
        q = self.model.joint_vector(arms, {n: int(state["head"][n]["ticks"]) for n in HEAD_JOINTS})
        outlines = []
        for arm in self.model.wrist_boards:
            uv, z = self.model.project(self.model.board_corners(arm, q), q)
            if (z > 0).all():
                outlines.append(uv)
        return outlines

    def capsule_outlines(self) -> list[np.ndarray]:
        """Each capsule the calibration has (both arms'), as its outline in the current frame (`capsule_outline`).

        Placed like `board_outlines`; a moving finger at its arm's measured opening (closed without a gripper map).
        """
        state = self.capture.state
        arms = {arm: state[f"{arm}_arm"] for arm in self.model.capsules if f"{arm}_arm" in state}
        q = self.model.joint_vector(arms, {n: int(state["head"][n]["ticks"]) for n in HEAD_JOINTS})
        outlines = []
        for arm in self.model.capsules:
            opening = 0.0
            if arm in arms and arm in self.model.gripper_maps:
                opening = max(self.model.opening_deg(arm, float(arms[arm]["gripper.pos"])), 0.0)
            for a_root, b_root, radius in self.model.capsules_root(arm, q, opening).values():
                polygon = capsule_outline(self.model, q, a_root, b_root, radius)
                if polygon is not None:
                    outlines.append(polygon)
        return outlines

    # ---- frames ---------------------------------------------------------------------------------------------

    def refresh(self, after: float | None = None) -> None:
        """Take a new frame, show it, and save it (with `out_dir`)."""
        super().refresh(after)
        if self.out_dir is not None:
            self._save()

    def _save(self) -> None:
        assert self.out_dir is not None
        index = len(self.frames)
        opening, pinch, moving = self.predictions()
        name = f"{index:03d}"
        cv2.imwrite(str(self.out_dir / f"{name}.png"), cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(
            str(self.out_dir / f"{name}_color.png"), cv2.cvtColor(self.capture.color, cv2.COLOR_RGB2BGR)
        )
        self.frames.append(
            {
                "index": index,
                "image": f"{name}.png",
                "timestamp": self.capture.timestamp,
                "view": self.target_index,
                "state": {self.arm_id: self.capture.state[self.arm_id], "head": self.capture.state["head"]},
                "opening_deg": opening,
                **{
                    key: {
                        "uv": p.uv.tolist(),
                        "z": p.z,
                        "point": p.point.tolist(),
                        "visibility": p.visibility,
                    }
                    for key, p in (("pinch", pinch), ("moving_tip", moving))
                },
            }
        )
        (self.out_dir / FRAMES_FILE).write_text(json.dumps(self.frames, indent=1) + "\n")

    # ---- moving ---------------------------------------------------------------------------------------------

    def next_view(self) -> bool:
        """The next view, with the gripper at the next of `openings`."""
        opening = self.openings[self.opening_index % len(self.openings)]
        self.opening_index += 1
        return self.next_pose(gripper_pos=opening)

    def move_gripper(self, delta: float) -> None:
        """Open (positive) or close the gripper by `delta` (gripper.pos, kept within 0-100) and show a new frame."""
        target = float(np.clip(self.gripper_pos() + delta, 0.0, 100.0))
        status = self.client.move_joints({self.arm_id: {"gripper.pos": target}})
        self.message = "" if status["state"] == "done" else f"the gripper ended as {status['state']}"
        self.refresh(status["finished_at"])

    def handle(self, action: str) -> bool:
        """Do one action from `decode_key`. Returns False when it is time to quit."""
        if action == "next":
            self.next_view()
        elif action == "open":
            self.move_gripper(self.gripper_step)
        elif action == "close":
            self.move_gripper(-self.gripper_step)
        else:
            return super().handle(action)
        return True


def main() -> None:
    """Command line entry point; see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("endpoint", nargs="?", default=f"tcp://localhost:{DEFAULT_PORT}")
    parser.add_argument("--arm", choices=["left", "right"], default="right")
    parser.add_argument("--model", type=Path, required=True, help="the calibrated URDF to check")
    parser.add_argument(
        "--openings",
        default=",".join(f"{o:g}" for o in DEFAULT_OPENINGS),
        help="gripper.pos per view, in turn, comma-separated",
    )
    parser.add_argument("--gripper-step", type=float, default=10.0, help="gripper.pos per [ or ] (10)")
    parser.add_argument("--out-dir", type=Path, help="save every frame here (e.g. under a scratch/ folder)")
    parser.add_argument("--n", type=int, default=20, help="number of target views (20)")
    parser.add_argument("--step", type=float, default=2.0, help="head jog per arrow key, degrees (2)")
    parser.add_argument("--max-vel", type=float, default=30.0, help="arm speed limit, deg/s (30)")
    parser.add_argument(
        "--delta-scale", type=float, default=1.0, help="scale of the target region (1 = +/-20 deg per joint)"
    )
    parser.add_argument(
        "--seed-pose", help='center of the target region, e.g. "shoulder_lift=-10,elbow_flex=10" (default: 0)'
    )
    parser.add_argument("--floor-z", type=float, help="floor height in the robot's root frame, meters")
    parser.add_argument(
        "--max-roll", type=float, default=MAX_ROLL_DEG, help="largest wrist roll used, deg (130)"
    )
    args = parser.parse_args()

    window = "calibration check (n = next view; [ ] gripper; q = quit)"

    def show(image: np.ndarray) -> None:
        cv2.imshow(window, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    model = XleRobotModel.load(args.model)
    with FelixClient(args.endpoint, timeout_s=20.0) as client, client.session():
        session = CheckSession(
            client,
            model,
            args.arm,
            show,
            openings=[float(o) for o in args.openings.split(",")],
            gripper_step=args.gripper_step,
            out_dir=args.out_dir,
            n_targets=args.n,
            max_vel=args.max_vel,
            head_step_deg=args.step,
            seed_pose=parse_seed(args.seed_pose),
            delta_scale=args.delta_scale,
            floor_z=args.floor_z,
            max_roll=args.max_roll,
        )
        for note in session.notes:
            print(note)
        try:
            while True:
                action = decode_key(cv2.waitKeyEx(30), KEY_ACTIONS)
                if action is not None and not session.handle(action):
                    break
        finally:
            cv2.destroyAllWindows()
    if args.out_dir is not None:
        print(f"{len(session.frames)} frames saved to {args.out_dir}")


if __name__ == "__main__":
    main()
