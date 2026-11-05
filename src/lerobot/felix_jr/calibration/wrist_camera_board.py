"""Fit an arm's wrist camera mount plate into the calibration: you click three of its corners, view after view.

  uv run python -m lerobot.felix_jr.calibration.wrist_camera_board tcp://<pi>:5555 --arm right
      --start-from <calibration.urdf> --out-dir ~/board_right [--n 12] [--step 2] [--max-vel 30]
      [--seed-pose "shoulder_lift=-10"] [--delta-scale 1] [--floor-z Z] [--max-roll 130] [--resume]

The wrist camera sits on a square plate fixed to the gripper. Seen from the head camera, that plate is what hides
the jaws in some grasps (`docs/sessions.md`, pen2), so the occlusion checks need it in the model. Before starting,
put small numbered marks (1, 2, 3) on three corners of the plate's back (the side the head camera sees), going
around it, so that corner 2 is between 1 and 3. The plate is assumed 1 cm thick (`BOARD_THICKNESS_M`).

Start the server on the Pi, clear the workspace, as for `collect`. A window shows the head camera. `n` moves the arm
and the head to the next view, chosen like `collect`'s but from the plate's side of the gripper, so the head camera
sees its back. In each view click the marked corners in order:

  left mouse click    record the corner asked for (1, then 2, then 3)
  s                   skip that corner (hidden in this view)
  n                   go to the next view (back to corner 1)
  arrow keys          turn the head by --step degrees
  u                   undo the last click
  q or Esc            finish: fit and write `calibration.urdf` (the robot is sent to rest)

**The fit.** Each corner is a point fixed in the arm's `<arm>_gripper_link` frame, 9 numbers for the three; the rest
of the calibration (`--start-from`) stays as it is. Every click asks the corner's predicted pixel (from the measured
joints) to land on it (the depth there is saved, but not fitted: at a corner it is often what is behind). After a few views
the fitted plate is drawn as a yellow outline, which should sit on the real one. 8-12 views are plenty. The report
(`wrist_board.md`) gives the pixel RMS and how square the fitted plate is: two edges of equal length at 90 deg.

**The output.** `--out-dir/calibration.urdf` is `--start-from` with the plate added (`XleRobotModel.with_wrist_board`:
a box link on a fixed joint to the gripper, and the fitted corners in `<felix_calibration>`); for the other arm, run
again with `--start-from` that file. The clicks and their images are kept in `--out-dir` (`corners.json`, `images/`);
`--resume` continues. The script runs inside a client session, so on Ctrl+C or a crash the robot goes to rest.
"""

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from lerobot.felix_jr.calibration.live_view import (
    INTRINSICS_KEYS,
    LiveView,
    decode_key,
    parse_seed,
    set_floor,
)
from lerobot.felix_jr.calibration.views import MAX_ROLL_DEG
from lerobot.felix_jr.calibration.xle_robot_model import (
    BOARD_THICKNESS_M,
    HEAD_JOINTS,
    SIDES,
    WristBoard,
    XleRobotModel,
    arm_record,
)
from lerobot.felix_jr.depth import with_matrices
from lerobot.felix_jr.harness.scene import Scene
from lerobot.felix_jr.server.client import FelixClient
from lerobot.felix_jr.server.protocol import DEFAULT_PORT
from lerobot.utils.import_utils import _scipy_available, require_package

if TYPE_CHECKING or _scipy_available:
    from scipy.optimize import least_squares

CORNERS = (1, 2, 3)
CLICKS_FILE = "corners.json"
CALIBRATION_FILE = "calibration.urdf"
REPORT_FILE = "wrist_board.md"
PIXEL_SIGMA = 2.0  # px: a click's expected error
MIN_CLICKS_PER_CORNER = 2  # before the plate is fitted and drawn
BOARD_COLOR = (255, 220, 0)  # RGB: yellow
KEY_ACTIONS = {ord("n"): "next", ord("s"): "skip", ord("u"): "undo", ord("q"): "quit", 27: "quit"}


def board_parent(arm: str) -> str:
    """The link the plate is fixed to: the gripper's body (not the pinch frame, whose placement is calibrated)."""
    return f"{arm}_gripper_link"


@dataclass
class CornerClick:
    """One clicked corner: which, the pixel, the depth there, and the robot's measured joints with the frame."""

    arm: str
    corner: int
    arm_pose: dict[str, float]  # LeRobot's degrees, {"shoulder_pan.pos": ..., ...}
    head_ticks: dict[str, int]
    pixel: tuple[float, float]
    depth_m: float | None = None
    image: str | None = None  # relative to the session folder
    index: int = 0
    timestamp: float = 0.0


def save_clicks(directory: Path, arm: str, record: dict[str, Any], clicks: Sequence[CornerClick]) -> None:
    """Write `corners.json`: the arm, its LeRobot calibration and the clicks."""
    data = {"arm": arm, "arm_calibration": record, "clicks": [asdict(c) for c in clicks]}
    (directory / CLICKS_FILE).write_text(json.dumps(data, indent=1) + "\n")


def load_clicks(directory: Path) -> tuple[str, dict[str, Any], list[CornerClick]]:
    """Read `corners.json` back: the arm, its LeRobot calibration and the clicks."""
    data = json.loads((directory / CLICKS_FILE).read_text())
    clicks = [CornerClick(**c | {"pixel": tuple(c["pixel"])}) for c in data["clicks"]]
    return data["arm"], data["arm_calibration"], clicks


# ---- the fit ------------------------------------------------------------------------------------------------


def _q(model: XleRobotModel, click: CornerClick) -> np.ndarray:
    return model.joint_vector({click.arm: click.arm_pose}, click.head_ticks or None)


def parent_in_camera(model: XleRobotModel, clicks: Sequence[CornerClick]) -> np.ndarray:
    """Per click, the parent link's pose in the camera's optical frame (N, 4, 4): all the fit needs of the robot."""
    poses = []
    for click in clicks:
        q = _q(model, click)
        poses.append(np.linalg.inv(model.camera_pose(q)) @ model.frame_pose(board_parent(click.arm), q))
    return np.array(poses)


def predicted(
    model: XleRobotModel, clicks: Sequence[CornerClick], corners: np.ndarray, frames: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Each click's corner (`corners` (3, 3), the parent link's frame) projected: pixels (N, 2) and depths (N,).

    `frames` are `parent_in_camera(model, clicks)`, if computed already.
    """
    frames = parent_in_camera(model, clicks) if frames is None else frames
    points = corners[[c.corner - 1 for c in clicks]]  # (N, 3)
    p = np.einsum("nij,nj->ni", frames[:, :3, :3], points) + frames[:, :3, 3]
    z = p[:, 2]
    safe_z = np.where(z > 1e-6, z, np.nan)
    k = model.intrinsics
    uv = np.stack([k["fx"] * p[:, 0] / safe_z + k["ppx"], k["fy"] * p[:, 1] / safe_z + k["ppy"]], axis=1)
    return uv, z


def initial_corners(model: XleRobotModel, clicks: Sequence[CornerClick]) -> np.ndarray:
    """A start for the fit: the pinch point (the plate is a few cm from it), a centimeter apart per corner."""
    frames = [
        (model.frame_pose(board_parent(c.arm), _q(model, c)), model.pinch_point(c.arm, _q(model, c)))
        for c in clicks
    ]
    pinch = np.mean([f[:3, :3].T @ (p - f[:3, 3]) for f, p in frames], axis=0)
    return pinch + np.array([[0.01, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.01, 0.0]])


def fit_corners(
    model: XleRobotModel, clicks: Sequence[CornerClick], start: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """The three corners (3, 3) in the parent link's frame that best explain `clicks`, and each click's pixel error.

    Only the pixels are fitted (`PIXEL_SIGMA` each, a Huber loss so a misclick does not pull the rest): the depth at
    a corner is often the background behind the plate's edge, so it is kept with the click but not used.
    """
    require_package("scipy", extra="felix_jr")
    pixels = np.array([c.pixel for c in clicks], dtype=float)
    frames = parent_in_camera(model, clicks)

    def residuals(x: np.ndarray) -> np.ndarray:
        uv, _ = predicted(model, clicks, x.reshape(3, 3), frames)
        return np.nan_to_num(((uv - pixels) / PIXEL_SIGMA).ravel(), nan=1e3)

    x0 = (initial_corners(model, clicks) if start is None else start).ravel()
    result = least_squares(residuals, x0, loss="huber", x_scale=0.01)
    corners = result.x.reshape(3, 3)
    uv, _ = predicted(model, clicks, corners, frames)
    return corners, np.linalg.norm(uv - pixels, axis=1)


def board_from_fit(
    model: XleRobotModel, clicks: Sequence[CornerClick], corners: np.ndarray, errors: np.ndarray
) -> WristBoard:
    """The plate from fitted corners: thick away from where the head camera saw it from (on average)."""
    arm = clicks[0].arm
    center = (corners[0] + corners[2]) / 2
    away = np.zeros(3)
    for click in clicks:
        q = _q(model, click)
        frame = model.frame_pose(board_parent(arm), q)
        camera = frame[:3, :3].T @ (model.camera_pose(q)[:3, 3] - frame[:3, 3])
        away += (center - camera) / np.linalg.norm(center - camera)
    away /= np.linalg.norm(away)
    return WristBoard(
        parent=board_parent(arm),
        corners=tuple((float(x), float(y), float(z)) for x, y, z in corners),
        away=(float(away[0]), float(away[1]), float(away[2])),
        thickness=BOARD_THICKNESS_M,
        rms_px=float(np.sqrt(np.mean(errors**2))),
        n_clicks=len(clicks),
    )


def squareness(board: WristBoard) -> tuple[float, float, float]:
    """The two fitted edges' lengths (m) and the angle between them (deg): a square plate has them equal, at 90."""
    c1, c2, c3 = board.face_corners()[:3]
    e1, e2 = c1 - c2, c3 - c2
    angle = np.degrees(np.arccos(np.clip(e1 @ e2 / (np.linalg.norm(e1) * np.linalg.norm(e2)), -1.0, 1.0)))
    return float(np.linalg.norm(e1)), float(np.linalg.norm(e2)), float(angle)


def report(board: WristBoard, clicks: Sequence[CornerClick], errors: np.ndarray) -> str:
    """The fit's report (markdown): the plate, how square it came out, and the pixel errors per corner."""
    edge1, edge2, angle = squareness(board)
    pose, size = board.box()
    lines = [
        f"# Wrist camera plate: {clicks[0].arm} arm",
        "",
        f"{len(clicks)} clicks; pixel RMS {board.rms_px:.1f} px.",
        "",
        f"- edges 1-2 {edge1 * 1000:.1f} mm, 2-3 {edge2 * 1000:.1f} mm, at {angle:.1f} deg (a square: equal, 90)",
        f"- box center in `{board.parent}` ({', '.join(f'{v * 1000:.1f}' for v in pose[:3, 3])}) mm, size "
        f"{' x '.join(f'{v * 1000:.0f}' for v in size)} mm (thickness assumed)",
        "",
        "| corner | clicks | RMS (px) | max (px) |",
        "|---|---|---|---|",
    ]
    for corner in CORNERS:
        mine = errors[[c.corner == corner for c in clicks]]
        if len(mine):
            lines.append(f"| {corner} | {len(mine)} | {np.sqrt(np.mean(mine**2)):.1f} | {mine.max():.1f} |")
    return "\n".join(lines) + "\n"


# ---- the session --------------------------------------------------------------------------------------------


def annotate(
    image: np.ndarray,
    lines: list[str],
    outline: np.ndarray | None,
    clicks: Sequence[tuple[tuple[float, float], int]],
) -> np.ndarray:
    """Copy of the RGB `image` with the fitted plate, this frame's clicks and text.

    The plate is a yellow outline with its corners numbered; the clicks are green crosses, numbered.
    """
    out = image.copy()
    if outline is not None and np.isfinite(outline).all():
        points = np.round(outline).astype(np.int32)
        cv2.polylines(out, [points.reshape(-1, 1, 2)], True, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.polylines(out, [points.reshape(-1, 1, 2)], True, BOARD_COLOR, 1, cv2.LINE_AA)
        for number, (u, v) in zip(CORNERS, points[:3], strict=True):
            cv2.putText(
                out,
                str(number),
                (int(u) + 6, int(v) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                BOARD_COLOR,
                1,
                cv2.LINE_AA,
            )
    for (u, v), number in clicks:
        centre = (int(round(u)), int(round(v)))
        cv2.drawMarker(out, centre, (0, 255, 0), cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
        cv2.putText(
            out,
            str(number),
            (centre[0] + 6, centre[1] + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    for row, text in enumerate(lines):
        org = (8, 20 + 20 * row)
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def flip_side(side: str) -> str:
    """The other side of the gripper (`SIDES`)."""
    return SIDES[1 - SIDES.index(side)]


class BoardSession(LiveView):
    """One plate fit: moves the robot, shows frames, records corner clicks, refits. See the module docstring."""

    no_more_views = "no more views: click what is left, then q"

    def __init__(
        self,
        client: FelixClient,
        start_from: Path,
        arm: str,
        out_dir: Path,
        show: Callable[[np.ndarray], None],
        n_targets: int = 12,
        max_vel: float = 30.0,
        max_accel: float = 60.0,
        head_step_deg: float = 2.0,
        resume: bool = False,
        color_format: str = "png",
        seed_pose: dict[str, float] | None = None,
        delta_scale: float = 1.0,
        floor_z: float | None = None,
        max_roll: float = MAX_ROLL_DEG,
    ):
        """Load the calibration, check it is for this robot, plan the views from the plate's side, show a frame.

        Raises `RobotMismatchError` (from `check_robot`) before anything moves.
        """
        self.start_from, self.out_dir = start_from, out_dir
        self.calibration = XleRobotModel.load(start_from)  # what is written out, with the plate added
        self.calibration.check_robot(client)
        model = XleRobotModel.load(start_from)  # the one the views are planned with: from the plate's side
        model.visible_side[arm] = flip_side(model.visible_side[arm])
        self.notes: list[str] = []
        capture = client.capture(color_format=color_format)
        intrinsics = {k: capture.calibration["color_intrinsics"][k] for k in INTRINSICS_KEYS}
        if model.intrinsics != intrinsics:
            self.notes.append("the camera's intrinsics differ from the calibration's: using the camera's")
            model.intrinsics = self.calibration.intrinsics = intrinsics
        set_floor(model, arm, floor_z, self.notes.append)
        self.record = arm_record(client.arm_calibration()[f"{arm}_arm"])
        (out_dir / "images").mkdir(parents=True, exist_ok=True)
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
        self.clicks: list[CornerClick] = []
        if resume and (out_dir / CLICKS_FILE).is_file():
            saved_arm, record, self.clicks = load_clicks(out_dir)
            if saved_arm != arm or record != self.record:
                raise ValueError(f"{out_dir} holds clicks of another arm or arm calibration")
        self.corner_index = 0  # into CORNERS: the corner the next click is
        self.frame_clicks: list[tuple[tuple[float, float], int]] = []
        self.corners: np.ndarray | None = None
        self.errors: np.ndarray | None = None
        self.message = self.notes[-1] if self.notes else "n = first view"
        self._refit()
        self.render()

    # ---- showing --------------------------------------------------------------------------------------------

    def asking(self) -> int | None:
        """The corner the next click is (None: all three done in this view)."""
        return CORNERS[self.corner_index] if self.corner_index < len(CORNERS) else None

    def outline(self) -> np.ndarray | None:
        """The fitted plate's four corners in the current frame (4, 2), if fitted and in front of the camera."""
        if self.corners is None:
            return None
        q = self.current_q()
        frame = self.model.frame_pose(board_parent(self.arm), q)
        c1, c2, c3 = self.corners
        points = np.array([c1, c2, c3, c1 + c3 - c2]) @ frame[:3, :3].T + frame[:3, 3]
        uv, z = self.model.project(points, q)
        return uv if (z > 0).all() else None

    def render(self) -> None:
        """Show the current frame with the fitted plate, this frame's clicks and the status text."""
        corner = self.asking()
        fit_text = (
            f"fit RMS {np.sqrt(np.mean(self.errors**2)):.1f} px"
            if self.errors is not None
            else "not fitted yet"
        )
        lines = [
            f"{self.arm} arm: {len(self.clicks)} clicks, view {self.target_index}/{len(self.targets)}, {fit_text}",
            f"click corner {corner} | s skip it | n next view | arrows head | u undo | q quit"
            if corner is not None
            else "all three corners done here: n next view | u undo | q quit",
        ]
        if self.message:
            lines.append(self.message)
        self.show(annotate(self.capture.color, lines, self.outline(), self.frame_clicks))

    def refresh(self, after: float | None = None) -> None:
        """Take a new frame and show it, asking for corner 1 again."""
        self.frame_clicks, self.corner_index = [], 0
        super().refresh(after)

    # ---- clicks ---------------------------------------------------------------------------------------------

    def click(self, u: float, v: float) -> CornerClick | None:
        """Record pixel (u, v) as the corner asked for, save it with its images, refit. None if all are done here."""
        corner = self.asking()
        if corner is None:
            self.message = "all three corners are done in this view: n for the next"
            self.render()
            return None
        state = self.capture.state
        index = len(self.clicks)
        scene = Scene.from_capture(
            self.capture.color, self.capture.raw_depth, with_matrices(self.capture.calibration)
        )
        depth = float(scene.pixel_to_xyz(np.array([[u, v]]), window=5, max_spread_m=0.03)[0, 2])
        image = f"images/{index:03d}.png"
        click = CornerClick(
            arm=self.arm,
            corner=corner,
            arm_pose={k: float(value) for k, value in state[self.arm_id].items()},
            head_ticks={n: int(state["head"][n]["ticks"]) for n in HEAD_JOINTS},
            pixel=(float(u), float(v)),
            depth_m=depth if np.isfinite(depth) else None,
            image=image,
            index=index,
            timestamp=self.capture.timestamp,
        )
        cv2.imwrite(str(self.out_dir / image), cv2.cvtColor(self.capture.color, cv2.COLOR_RGB2BGR))
        self.clicks.append(click)
        self.frame_clicks.append((click.pixel, corner))
        self.corner_index += 1
        save_clicks(self.out_dir, self.arm, self.record, self.clicks)
        self.message = f"corner {corner} saved"
        self._refit()
        self.render()
        return click

    def skip(self) -> None:
        """Skip the corner asked for (hidden in this view)."""
        if self.asking() is not None:
            self.message = f"skipped corner {self.asking()}"
            self.corner_index += 1
        self.render()

    def undo(self) -> None:
        """Drop the last click (its image stays in the folder) and refit."""
        if self.clicks:
            self.clicks.pop()
            if self.frame_clicks:
                self.frame_clicks.pop()
                self.corner_index = max(self.corner_index - 1, 0)
            save_clicks(self.out_dir, self.arm, self.record, self.clicks)
            self.message = f"undid the last click ({len(self.clicks)} left)"
            self._refit()
        self.render()

    def _refit(self) -> None:
        """Fit the corners once every corner has `MIN_CLICKS_PER_CORNER` clicks (from the last fit, if any)."""
        counts = [sum(c.corner == corner for c in self.clicks) for corner in CORNERS]
        if min(counts) < MIN_CLICKS_PER_CORNER:
            self.corners = self.errors = None
            return
        self.corners, self.errors = fit_corners(self.calibration, self.clicks, self.corners)

    def finish(self) -> Path | None:
        """Fit every click and write the calibration with the plate (`calibration.urdf`) and the report.

        Returns the calibration's path, or None if there are too few clicks to fit.
        """
        self._refit()
        if self.corners is None or self.errors is None:
            return None
        board = board_from_fit(self.calibration, self.clicks, self.corners, self.errors)
        path = self.out_dir / CALIBRATION_FILE
        self.calibration.with_wrist_board(self.arm, board).save(path)
        (self.out_dir / REPORT_FILE).write_text(report(board, self.clicks, self.errors))
        return path

    # ---- key handling ---------------------------------------------------------------------------------------

    def handle(self, action: str) -> bool:
        """Do one action from `decode_key`. Returns False when it is time to quit."""
        if action == "skip":
            self.skip()
        elif action == "undo":
            self.undo()
        else:
            return super().handle(action)
        return True


def main() -> None:
    """Command line entry point; see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("endpoint", nargs="?", default=f"tcp://localhost:{DEFAULT_PORT}")
    parser.add_argument("--arm", choices=["left", "right"], required=True)
    parser.add_argument(
        "--start-from", type=Path, required=True, help="the calibration.urdf to add the plate to"
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=12, help="number of target views (12)")
    parser.add_argument("--step", type=float, default=2.0, help="head jog per arrow key, degrees (2)")
    parser.add_argument("--max-vel", type=float, default=30.0, help="arm speed limit, deg/s (30)")
    parser.add_argument("--delta-scale", type=float, default=1.0, help="scale of the target region")
    parser.add_argument("--seed-pose", help='center of the target region, e.g. "shoulder_lift=-10"')
    parser.add_argument("--floor-z", type=float, help="floor height in the robot's root frame, meters")
    parser.add_argument("--max-roll", type=float, default=MAX_ROLL_DEG, help="largest wrist roll, deg (130)")
    parser.add_argument("--resume", action="store_true", help="continue an earlier --out-dir")
    args = parser.parse_args()
    if (args.out_dir / CALIBRATION_FILE).resolve() == args.start_from.resolve():
        parser.error("--out-dir would overwrite --start-from: pick another folder")

    window = "wrist camera plate (click the corner asked for; n = next view; q = quit)"
    pending: list[tuple[int, int]] = []

    def on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            pending.append((x, y))

    def show(image: np.ndarray) -> None:
        cv2.imshow(window, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with FelixClient(args.endpoint, timeout_s=20.0) as client, client.session():
        session = BoardSession(
            client,
            args.start_from,
            args.arm,
            args.out_dir,
            show,
            n_targets=args.n,
            max_vel=args.max_vel,
            head_step_deg=args.step,
            resume=args.resume,
            seed_pose=parse_seed(args.seed_pose),
            delta_scale=args.delta_scale,
            floor_z=args.floor_z,
            max_roll=args.max_roll,
        )
        for note in session.notes:
            print(note)
        cv2.setMouseCallback(window, on_mouse)
        try:
            while True:
                key = cv2.waitKeyEx(30)
                while pending:
                    session.click(*pending.pop(0))
                action = decode_key(key, KEY_ACTIONS)
                if action is not None and not session.handle(action):
                    break
        finally:
            cv2.destroyAllWindows()
    print(f"{len(session.clicks)} clicks saved to {args.out_dir / CLICKS_FILE}")
    path = session.finish()
    if path is None:
        print(f"too few clicks to fit: each corner needs {MIN_CLICKS_PER_CORNER}")
    else:
        print(f"Wrote {path} and {args.out_dir / REPORT_FILE}")
        print((args.out_dir / REPORT_FILE).read_text())


if __name__ == "__main__":
    main()
