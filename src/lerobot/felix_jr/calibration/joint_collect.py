"""Collect a joint calibration of the head, an arm and its wrist camera (`joint_fit`): one arm per session.

  uv run --extra felix_jr --extra felix_jr_sim --extra pyzmq-dep python -m lerobot.felix_jr.calibration.joint_collect \
      tcp://<pi>:5555 --arm right --out-dir <scratch folder> --start-from <calibration.urdf>
      [--board checkerboard:8x6:0.025] [--placements 2] [--arm-views 10] [--head-views 6] [--low-views 10]
      [--max-vel 20] [--wrist-tip sweep|click] [--resume]

`docs/camera_calibration.md`, "The rework". Start the server on the Pi with the arm and the head. `--start-from` must
have the arm's wrist camera (its lens, from `calibration.wrist_camera`, and a first pose on the gripper). The window
shows three panes: the head's picture twice, once per fingertip face (the pinch face, magenta; the wrist camera's
side, a finger's thickness away, cyan), and the wrist camera's (cyan: it sees only its own side), each with that
face's tip as the model predicts it circled. A click is on the face of the pane it is in. Per board placement:

1. **Place the board.** The arms go to rest and the head looks where the board should go: lay it flat on the table
   on the drawn outline (roughly: anywhere the head sees it works). The head's picture is live, the board's corners
   dotted green where it finds them. `n`: the head finds it, then looks at it from `--head-views` directions around
   it (the arms stay folded).
2. **Arm views.** Each `n` moves the arm (slowly: watch it) to the next planned view of the board, the head looking at
   the board: a moment both cameras see it at ties the head to the arm. Click the fixed fingertip in the wrist
   camera's picture (the face toward it), and in the head's if it shows (the face the head sees).

Then **low poses**: each `n` moves the gripper low over the table, as for a grasp, and the head looks at the fingertip:
click it in both pictures. The head's clicks matter: a wrist click fixes only the direction from its camera to the
fingertip, the head's the distance along it. Leave the board where it is: a picture that shows it is one more sighting.

The head sees one face of the fingertip: click it in that face's head pane (the one marked "expected" is the face the
model expects it to see; trust the picture over it).

**The wrist camera's fingertip is not clicked** (`--wrist-tip sweep`, the default): it is at one pixel in every wrist
frame (the camera and the finger both on the gripper), and by eye, with the jaws closed, it was clicked in the notch
between the tips, ~10 px off (`calibration.auto_tips`). After `q`, before the fit, a gripper sweep measures it with
the jaws open against the table (`gripper_sweep`: the sim's checks, the person asked, the sweep pose, the live
preview, one sweep), and each open frame's fingertip goes into the session as a wrist click (`auto_tips.sweep_clicks`;
its frames in `wrist_tip_sweep/`). Clicks in the wrist pane are declined; the head's are as before. `--wrist-tip
click`: clicked by eye, as before.

Keys: `n` next, arrows turn the head (if the board is cut off), `u` undo the last click, `q` stop and
fit. Everything is saved as you go (`joint_session.json`, `images/`), `--resume` continues. At the end the session
is fitted (`joint_fit.calibrate`) and `calibration.urdf` and `calibration.md` written to `--out-dir`. The script runs
in a client session: Ctrl+C or a crash sends the robot to rest.
"""

import argparse
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lerobot.felix_jr.calibration import joint_views as jv
from lerobot.felix_jr.calibration.boards import Board, Detection, board_pose, parse_board
from lerobot.felix_jr.calibration.joint_fit import (
    HEAD,
    SESSION_FILE,
    BoardSighting,
    JointSession,
    Look,
    TipClick,
    calibrate,
    load_session,
    report,
    save_session,
)
from lerobot.felix_jr.calibration.live_view import INTRINSICS_KEYS, decode_key, head_limits, move_head_to
from lerobot.felix_jr.calibration.xle_robot_model import HEAD_JOINTS, XleRobotModel, arm_record
from lerobot.felix_jr.harness.look import aim_head
from lerobot.felix_jr.server.client import Capture, FelixClient, WristCapture
from lerobot.felix_jr.server.protocol import DEFAULT_PORT

PANE = (480, 360)  # each picture is shown at this size (three side by side)
# the window's panes: the head's picture twice, one per fingertip face, and the wrist camera's (which sees only its own
# side). A click is on the face of the pane it is in.
PANES = ("head:pinch", "head:wrist", "wrist")
WRIST_TIPS = ("sweep", "click")  # how the wrist camera's fingertip is measured (the module docstring)
WRIST_TIP_SWEEP = "wrist_tip_sweep"  # the sweep's folder in the session's, and its clicks' `image`
PANE_TITLES = {
    "head:pinch": "head: pinch face",
    "head:wrist": "head: wrist-camera face",
    "wrist": "wrist camera",
}
FACE_COLORS = {"pinch": (255, 64, 255), "wrist": (0, 255, 255)}  # RGB: magenta, cyan
STATUS_HEIGHT = 90
LIVE_PERIOD_S = 0.2  # the head picture's refresh while placing the board
CALIBRATION_FILE = "calibration.urdf"
KEYS = {ord("n"): "next", ord("u"): "undo", ord("q"): "quit", 27: "quit"}
STILL_DEG, STILL_WINDOW_S, STILL_POLL_S, STILL_TIMEOUT_S = 0.15, 0.4, 0.1, 3.0  # `_wait_still`
BOARD_MOVED_M = (
    0.02  # a head sighting this far from the placement's board: it was moved (or the head's model is off)
)
EDGE_ON_DEG = (75.0, 105.0)  # the fingers this near edge-on to the head: which face shows is unclear


@dataclass
class Region:
    """Where a picture is drawn in the window: a click there maps back to the picture's pixel."""

    pane: str  # "head" or "wrist"
    x0: int
    y0: int
    w: int
    h: int
    scale: float  # window pixels per picture pixel
    origin: tuple[float, float]  # the picture pixel at (x0, y0)

    def contains(self, x: int, y: int) -> bool:
        """Whether window pixel (x, y) is in this region."""
        return self.x0 <= x < self.x0 + self.w and self.y0 <= y < self.y0 + self.h

    def to_picture(self, x: int, y: int) -> tuple[float, float]:
        """The picture pixel under window pixel (x, y) (pixel centers at integers)."""
        return (
            self.origin[0] + (x - self.x0 + 0.5) / self.scale - 0.5,
            self.origin[1] + (y - self.y0 + 0.5) / self.scale - 0.5,
        )


def _draw_text(image: np.ndarray, lines: list[str], org: tuple[int, int]) -> None:
    for row, text in enumerate(lines):
        at = (org[0], org[1] + 20 * row)
        cv2.putText(image, text, at, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, text, at, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def pane_camera(pane: str) -> str:
    """Whose picture a pane shows: "head" or "wrist"."""
    return pane.split(":")[0]


def pane_face(pane: str) -> str:
    """The fingertip face a pane's clicks are on: "pinch" or "wrist"."""
    return pane.split(":")[1] if ":" in pane else "wrist"


def compose(
    pictures: dict[str, np.ndarray],
    predicted: dict[str, np.ndarray | None],
    clicks: dict[str, list[tuple[float, float]]],
    outlines: dict[str, np.ndarray],
    lines: list[str],
    found: dict[str, np.ndarray] | None = None,
    titles: dict[str, str] | None = None,
    expected: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, list[Region]]:
    """The window: `PANES` side by side, each with its face's predicted tip circled in the face's color.

    `pictures`, `outlines` (the board's, while placing it) and `found` (board corners detected, green dots) are per
    camera ("head", "wrist"); `predicted` and `clicks` per pane. `titles`: each pane's (default `PANE_TITLES`).
    `expected`: per camera, the board's corners where the model has them (small orange circles).
    """
    window = np.zeros((PANE[1] + STATUS_HEIGHT, len(PANES) * PANE[0], 3), np.uint8)
    regions: list[Region] = []
    for i, pane in enumerate(PANES):
        camera = pane_camera(pane)
        picture = pictures[camera]
        scale = min(PANE[0] / picture.shape[1], PANE[1] / picture.shape[0])
        shown = cv2.resize(picture, (int(picture.shape[1] * scale), int(picture.shape[0] * scale)))
        x0 = i * PANE[0]

        def at(uv: np.ndarray, s: float = scale) -> tuple[int, int]:
            return int(round((uv[0] + 0.5) * s - 0.5)), int(round((uv[1] + 0.5) * s - 0.5))

        if camera in outlines and np.isfinite(outlines[camera]).all():
            cv2.polylines(
                shown,
                [np.array([at(p) for p in outlines[camera]], np.int32)],
                True,
                (255, 160, 0),
                2,
                cv2.LINE_AA,
            )
        for p in (expected or {}).get(camera, []):
            if np.isfinite(p).all():
                cv2.circle(shown, at(p), 4, (255, 160, 0), 1, cv2.LINE_AA)
        for p in (found or {}).get(camera, []):
            cv2.circle(shown, at(p), 3, (0, 255, 0), -1, cv2.LINE_AA)
        for p in clicks.get(pane, []):
            cv2.drawMarker(shown, at(np.array(p)), (0, 255, 0), cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
        tip = predicted.get(pane)
        if tip is not None and np.isfinite(tip).all():
            cv2.circle(shown, at(tip), 12, FACE_COLORS[pane_face(pane)], 2, cv2.LINE_AA)
        _draw_text(shown, [(titles or PANE_TITLES)[pane]], (8, 20))
        cv2.rectangle(
            shown, (0, 0), (shown.shape[1] - 1, shown.shape[0] - 1), FACE_COLORS[pane_face(pane)], 1
        )
        window[: shown.shape[0], x0 : x0 + shown.shape[1]] = shown
        regions.append(Region(pane, x0, 0, shown.shape[1], shown.shape[0], scale, (0.0, 0.0)))
    _draw_text(window, lines, (8, PANE[1] + 18))
    return window, regions


class JointCollect:
    """One arm's joint-calibration session on the robot (the module docstring)."""

    def __init__(
        self,
        client: FelixClient,
        arm: str,
        out_dir: Path,
        show: Callable[[np.ndarray], None],
        start_from: Path,
        board: Board,
        placements: int = 2,
        arm_views: int = 10,
        head_views: int = 6,
        low_views: int = 10,
        max_vel: float = 20.0,
        head_step_deg: float = 2.0,
        resume: bool = False,
        seed: int = 0,
        wrist_tip: str = "sweep",
    ):
        """Load the start, check it against the robot, and send the arms to rest for the first placement."""
        self.client, self.arm, self.arm_id, self.out_dir, self.show = client, arm, f"{arm}_arm", out_dir, show
        self.board, self.n_placements, self.n_arm_views = board, placements, arm_views
        self.n_head_views, self.n_low_views, self.max_vel = head_views, low_views, max_vel
        self.head_step_deg = head_step_deg
        self.rng = np.random.default_rng(seed)
        if wrist_tip not in WRIST_TIPS:
            raise ValueError(f"wrist_tip: one of {WRIST_TIPS}, not {wrist_tip!r}")
        self.wrist_tip = wrist_tip
        (out_dir / "images").mkdir(parents=True, exist_ok=True)
        self.model = XleRobotModel.load(start_from)
        head = client.head_state()
        signs = {n: head[n]["sign"] for n in HEAD_JOINTS}
        if self.model.head_signs != signs:
            raise ValueError(
                f"the head's directions changed since {start_from} ({self.model.head_signs} -> {signs})"
            )
        camera = self.model.wrist_cameras.get(arm)
        if camera is None or camera.gripper_from_camera is None:
            raise ValueError(
                f"{start_from} has no {arm} wrist camera: run calibration.wrist_camera's lens step first"
            )
        capture = client.capture()
        self.model.intrinsics = {k: capture.calibration["color_intrinsics"][k] for k in INTRINSICS_KEYS}
        self.rest = {k: float(v) for k, v in client.get_homes()[self.arm_id].items()}
        base = self.model.frame_pose(f"{arm}_base_link", self.model.joint_vector())[:3, 3]
        self.table_z = float(self.model.floor_z if self.model.floor_z is not None else base[2] - 0.06)
        if resume and (out_dir / SESSION_FILE).is_file():
            self.session = load_session(out_dir)
            if self.session.arm != arm:
                raise ValueError(f"{out_dir} is the {self.session.arm} arm's session")
        else:
            record = arm_record(client.arm_calibration()[self.arm_id])
            self.session = JointSession(
                out_dir,
                arm,
                board,
                dict(self.model.head_signs),
                dict(self.model.head_refs),
                dict(self.model.intrinsics),
                record,
            )
        self.placement = len(self.session.placements)  # the next placement's index
        self.phase = "place" if self.placement < placements else "low"
        self.views: list[dict[str, float]] = []
        self.view_index = 0
        self.root_board: np.ndarray | None = None
        self.board_known = False  # the placement's pose measured (not just suggested)
        self.head: Capture | None = None
        self.wrist: WristCapture | None = None
        self.look: Look | None = None
        # the current pictures' files (held, not keyed by id(): a freed capture's id is reused by the next)
        self.saved: list[tuple[Capture | WristCapture, str]] = []
        self.n_images = len(list((out_dir / "images").glob("*.png")))  # --resume numbers on from there
        self.frame_clicks: dict[str, list[tuple[float, float]]] = {pane: [] for pane in PANES}
        self.regions: list[Region] = []
        self.head_face, self.face_angle = "pinch", 0.0  # set by each capture (`_capture`)
        self.live: Detection | None = None  # the board in the live head picture while placing it
        self.detected: dict[
            str, np.ndarray
        ] = {}  # per camera ("head", "wrist"): the corners found in its picture
        self.board_warning = ""
        self.live_at = 0.0
        self.message = ""
        if self.phase == "place":
            self._start_placement()
        else:
            self._start_low()

    # ---- the robot ---------------------------------------------------------------------------------------------

    def _ticks(self) -> dict[str, int]:
        return {n: int(self.client.head_state()[n]["ticks"]) for n in HEAD_JOINTS}

    def _aim(self, target: np.ndarray) -> float | None:
        """Turn the head to look at root-frame `target` (within its limits)."""
        ticks, _ = aim_head(self.model, self._ticks(), target)
        limits = head_limits(self.client.head_state())
        ticks = {n: int(np.clip(t, *limits[n])) for n, t in ticks.items()}
        return move_head_to(self.client, ticks)

    def _move_arm(self, pose: dict[str, float]) -> bool:
        status = self.client.move_joints({self.arm_id: {**pose, "gripper.pos": 0.0}}, max_vel=self.max_vel)
        if status["state"] != "done":
            self.message = f"the arm stopped ({status['state']}: {status['reason']}); n tries the next view"
            return False
        self._wait_still()
        return True

    def _wait_still(self) -> bool:
        """Wait for the arm to be still, not just settled within the move's tolerance (False: it never was).

        A move is done once every joint is within 1.5 deg of its goal for 0.2 s, while the arm can still vibrate below
        that (on the robot, visibly, right after a move): a picture then is blurred and its joints read a moment
        apart from it. Still: no joint moving more than `STILL_DEG` over `STILL_WINDOW_S`, read every `STILL_POLL_S`,
        for `STILL_TIMEOUT_S` at most.
        """
        history: list[tuple[float, np.ndarray]] = []
        deadline = time.monotonic() + STILL_TIMEOUT_S
        while time.monotonic() < deadline:
            now = time.monotonic()
            arm = self.client.get_state()[self.arm_id]
            history.append(
                (now, np.array([arm[k] for k in sorted(arm) if k != "gripper.pos"], dtype=np.float64))
            )
            history = [(t, j) for t, j in history if now - t <= STILL_WINDOW_S]
            if now - history[0][0] >= 0.9 * STILL_WINDOW_S:
                spread = np.ptp(np.array([j for _, j in history]), axis=0)
                if spread.max() <= STILL_DEG:
                    return True
            time.sleep(STILL_POLL_S)
        return False

    def _capture(self, after: float | None = None) -> None:
        """Take both pictures and the moment they show (the arm's joints from the wrist's, the head's ticks)."""
        self.head = self.client.capture(after=after)
        self.wrist = self.client.wrist_capture(self.arm, color_format="png", after=after)
        arm = {k: float(v) for k, v in self.wrist.state[self.arm_id].items()}
        ticks = {n: int(self.head.state["head"][n]["ticks"]) for n in HEAD_JOINTS}
        self.look = Look({self.arm: arm}, ticks)
        self.frame_clicks = {pane: [] for pane in PANES}
        self.saved = []
        self.detected = {}
        # the face of the fingertip the head shows, as the model has it: its pane is marked "expected"
        self.face_angle = jv.head_face_angle(
            self.model, self.arm, self.model.joint_vector(self.look.arms, ticks)
        )
        self.head_face = "pinch" if self.face_angle < 90.0 else "wrist"

    def _image(self, capture: Capture | WristCapture, name: str) -> str:
        """The capture's picture saved once in `images/` (its file name)."""
        for saved, path in self.saved:
            if saved is capture:
                return path
        path = f"images/{self.n_images:04d}_{name}.png"
        self.n_images += 1
        cv2.imwrite(str(self.out_dir / path), cv2.cvtColor(capture.color, cv2.COLOR_RGB2BGR))
        self.saved.append((capture, path))
        return path

    def _sight(self, camera: str) -> Detection | None:
        """Find the board in `camera`'s current picture ("head" or the arm) and save the sighting.

        A symmetric board reads two ways (`Board.readings`): once the placement is known (`board_known`), the reading
        whose pose, through the model, agrees with it is kept (the other is turned ~180 deg).
        """
        capture = self.head if camera == HEAD else self.wrist
        assert capture is not None and self.look is not None
        found = self.board.detect(capture.color)
        if found is None:
            return None
        readings = self.board.readings(found)
        if len(readings) > 1 and self.board_known and self.root_board is not None:
            found = min(readings, key=lambda r: self._turn_from_placement(camera, r))
        name = (
            f"p{self.placement - (self.phase == 'low')}"  # low poses: the board still at the last placement
        )
        image = self._image(capture, "head" if camera == HEAD else "wrist")
        self.detected["head" if camera == HEAD else "wrist"] = found.pixels
        if camera == HEAD and self.board_known and self.root_board is not None:
            center = self.board.points().mean(axis=0)
            seen = self._root_from_board(HEAD, found)
            off = np.linalg.norm(
                seen[:3, :3] @ center
                + seen[:3, 3]
                - (self.root_board[:3, :3] @ center + self.root_board[:3, 3])
            )
            self.board_warning = (
                f"the head sees the board {1000 * off:.0f} mm from where it was placed: was it moved?"
                if off > BOARD_MOVED_M
                else ""
            )
        self.session.sightings.append(BoardSighting(camera, name, self.look, found, image))
        save_session(self.session)
        return found

    def _root_from_board(self, camera: str, reading: Detection) -> np.ndarray:
        """The board's pose in the root frame from one reading in `camera`'s current picture, through the model."""
        assert self.look is not None
        q = self.model.joint_vector(self.look.arms, self.look.head_ticks)
        if camera == HEAD:
            intr = self.model.intrinsics
            matrix = np.array([[intr["fx"], 0, intr["ppx"]], [0, intr["fy"], intr["ppy"]], [0, 0, 1.0]])
            return self.model.camera_pose(q) @ board_pose(self.board, reading, matrix)
        lens = self.model.wrist_cameras[self.arm]
        camera_board = board_pose(self.board, reading, lens.matrix(), lens.coeffs, lens.model == "fisheye")
        return jv.wrist_pose(self.model, self.arm, q) @ camera_board

    def _turn_from_placement(self, camera: str, reading: Detection) -> float:
        """How far (rad) a reading's board is turned from the placement's."""
        assert self.root_board is not None
        r = self._root_from_board(camera, reading)[:3, :3] @ self.root_board[:3, :3].T
        return float(np.arccos(np.clip((np.trace(r) - 1) / 2, -1.0, 1.0)))

    # ---- the phases --------------------------------------------------------------------------------------------

    def _start_placement(self) -> None:
        self.client.home(max_vel=self.max_vel, arms=[self.arm_id])
        suggested = jv.suggest_placements(self.model, self.arm, self.board, self.n_placements, self.table_z)[
            self.placement
        ]
        self.root_board, self.board_known = suggested, False
        self._capture(self._aim(jv.board_points(self.board, suggested).mean(axis=0)))
        self.phase = "place"
        self.message = (
            f"placement {self.placement + 1}/{self.n_placements}: lay the board flat on the outline, then n"
        )
        self.render()

    def refresh(self) -> None:
        """While placing the board: a new head picture every `LIVE_PERIOD_S`, with the board found in it marked."""
        if self.phase != "place" or self.look is None or time.monotonic() - self.live_at < LIVE_PERIOD_S:
            return
        self.head = self.client.capture(color_format="jpeg")
        ticks = {n: int(self.head.state["head"][n]["ticks"]) for n in HEAD_JOINTS}
        self.look = Look(self.look.arms, ticks)
        self.live = self.board.detect(self.head.color)
        self.live_at = time.monotonic()
        self.message = f"placement {self.placement + 1}/{self.n_placements}: " + (
            "the head sees the board whole: n" if self.live else "lay the board flat on the outline"
        )
        self.render()

    def _confirm_placement(self) -> None:
        self._capture()  # full quality, not the live view's
        self.live = None
        found = self._sight(HEAD)
        if found is None:
            self.message = (
                "the head does not see the whole board: move it onto the outline (or the head), then n"
            )
            self.render()
            return
        self.root_board = self._root_from_board(HEAD, found)  # the first reading sets the board's frame
        self.board_known = True
        self.table_z = float(self.root_board[2, 3])
        self.session.placements[f"p{self.placement}"] = self.root_board.tolist()
        points = jv.board_points(self.board, self.root_board)
        seen = 1
        for target in jv.head_aims(points.mean(axis=0), self.n_head_views):
            self._capture(self._aim(target))
            seen += self._sight(HEAD) is not None
        region = jv.low_poses(self.model, self.arm, self.rest, 40, self.table_z, self.rng)
        self.views = jv.arm_views(
            self.model,
            self.arm,
            self.rest,
            points,
            self.n_arm_views,
            self.rng,
            min_fraction=self.board.min_view_fraction,
            region=region,
        )
        self.view_index = 0
        self.phase = "arm"
        save_session(self.session)
        self.message = (
            f"the head saw the board {seen} times; {len(self.views)} arm views planned: n moves the arm"
        )
        self.render()

    def _next_arm_view(self) -> None:
        if self.view_index >= len(self.views):
            self.placement += 1
            if self.placement < self.n_placements:
                self._start_placement()
            else:
                self._start_low()
            return
        pose = self.views[self.view_index]
        self.view_index += 1
        if not self._move_arm(pose):
            self.render()
            return
        assert self.root_board is not None
        self._capture(self._aim(jv.board_points(self.board, self.root_board).mean(axis=0)))
        wrist = self._sight(self.arm) is not None
        head = self._sight(HEAD) is not None
        self.message = (
            f"view {self.view_index}/{len(self.views)}: the board {'seen' if wrist else 'NOT seen'} by the wrist, "
            f"{'seen' if head else 'NOT seen'} by the head; click the fingertips, then n"
        )
        self.render()

    def _start_low(self) -> None:
        self.views = jv.low_poses(self.model, self.arm, self.rest, self.n_low_views, self.table_z, self.rng)
        self.view_index = 0
        self.phase = "low"
        self.message = f"{len(self.views)} poses low over the table: n moves the arm there (watch it)"
        if self.head is None:
            self._capture()
        self.render()

    def _next_low(self) -> None:
        if self.view_index >= len(self.views):
            self.phase = "done"
            self.message = "all views done: q to fit"
            self.render()
            return
        pose = self.views[self.view_index]
        self.view_index += 1
        if not self._move_arm(pose):
            self.render()
            return
        q = self.model.joint_vector({self.arm: pose}, self._ticks())
        self._capture(self._aim(self.model.pinch_point(self.arm, q)))  # the head on the fingertip: its click
        seen = ""
        if self.board_known:  # the board still lies where the arm views saw it: sightings, if it shows
            wrist, head = self._sight(self.arm) is not None, self._sight(HEAD) is not None
            seen = f" (the board seen: wrist {'yes' if wrist else 'no'}, head {'yes' if head else 'no'})"
        self.message = f"low pose {self.view_index}/{len(self.views)}{seen}: click the fingertips, then n"
        self.render()

    # ---- actions -----------------------------------------------------------------------------------------------

    def next(self) -> None:
        """`n`: the next step of the current phase."""
        if self.phase == "place":
            self._confirm_placement()
        elif self.phase == "arm":
            self._next_arm_view()
        elif self.phase == "low":
            self._next_low()
        else:
            self.message = "all views done: q to fit"
            self.render()

    def click_at(self, x: int, y: int) -> TipClick | None:
        """A click in the window at (x, y): a fingertip in the picture under it (none in the status bar)."""
        region = next((r for r in self.regions if r.contains(x, y)), None)
        if region is None or self.look is None or self.phase == "place":
            return None
        return self.click(region.pane, *region.to_picture(x, y))

    def click(self, pane: str, u: float, v: float) -> TipClick | None:
        """Record the fingertip at pixel (u, v) of `pane`'s picture (one of `PANES`), on the pane's face.

        None (declined) in the wrist pane when a sweep measures the wrist's fingertip (`wrist_tip`, the module docstring).
        """
        if pane not in PANES:
            raise ValueError(f"no pane {pane!r}: one of {PANES}")
        if pane_camera(pane) != "head" and self.wrist_tip == "sweep":
            self.message = (
                "the wrist camera's fingertip is measured by a sweep after q: click only the head's panes"
            )
            self.render()
            return None
        head = pane_camera(pane) == "head"
        capture = self.head if head else self.wrist
        assert capture is not None and self.look is not None
        camera = HEAD if head else self.arm
        click = TipClick(
            camera,
            self.arm,
            self.look,
            np.array([u, v]),
            self._image(capture, pane_camera(pane)),
            pane_face(pane),
        )
        self.session.clicks.append(click)
        self.frame_clicks[pane].append((u, v))
        save_session(self.session)
        self.message = f"{len(self.session.clicks)} clicks saved"
        self.render()
        return click

    def undo(self) -> None:
        """`u`: drop the last click."""
        if self.session.clicks:
            last = self.session.clicks.pop()
            pane = f"head:{last.on_face}" if last.camera == HEAD else "wrist"
            if self.frame_clicks[pane]:
                self.frame_clicks[pane].pop()
            save_session(self.session)
            self.message = f"undid the last click ({len(self.session.clicks)} left)"
        self.render()

    def nudge(self, direction: str) -> None:
        """Arrows: turn the head a step and look again (the board's sighting is saved again from there)."""
        name, sign = {
            "up": ("head_eyes", -1),
            "down": ("head_eyes", 1),
            "left": ("head_neck", 1),
            "right": ("head_neck", -1),
        }[direction]
        done = self.client.head_move(name, delta_deg=sign * self.head_step_deg)["done_at"]
        self._capture(done)
        if self.phase == "arm":
            self._sight(HEAD)
        self.render()

    def measure_wrist_tip(self, window: bool = True, ask: Callable[[str], str] = input) -> int:
        """The wrist camera's fingertip by a gripper sweep (the module docstring), into the session as wrist clicks.

        Replaces the wrist clicks an earlier sweep made (`--resume`); the number added (0: refused, declined, stopped
        at the preview, or no fingertip seen).
        """
        from .auto_tips import sweep_clicks
        from .gripper_sweep import DEFAULT_POSITIONS, Fitter, check_pose, run_sweeps

        folder = self.out_dir / WRIST_TIP_SWEEP
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        fitter = Fitter(self.model, self.arm, folder, window=window)
        positions = list(DEFAULT_POSITIONS)
        start = check_pose(self.client, self.model, self.arm, fitter, positions, None, False, 1, ask)
        done = start is not None and run_sweeps(self.client, self.arm, fitter, positions, 1, False, start)
        fitter.fingers.close()
        if not done:
            return 0
        found = sweep_clicks(self.model, self.arm, [frame for frame, _, _ in fitter.seen])
        if not found:
            return 0
        ticks = self._ticks()
        self.session.clicks = [
            c
            for c in self.session.clicks
            if not (c.camera == self.arm and c.image.startswith(WRIST_TIP_SWEEP))
        ]
        for i, (frame, pixel, _) in enumerate(found):
            look = Look({self.arm: dict(frame.pose)}, ticks)
            image = f"{WRIST_TIP_SWEEP}/frame_{i:02d}.npz"
            self.session.clicks.append(TipClick(self.arm, self.arm, look, pixel, image, "wrist"))
        save_session(self.session)
        return len(found)

    def finish(self, holdout_every: int = 5, head_axes: bool = False) -> Path | None:
        """Fit the session and write `calibration.urdf` and `calibration.md` (None if there is too little).

        `head_axes`: fit the head joints' axes' tilts too (`XleRobotModel.with_head_axes`).
        """
        save_session(self.session)
        if not self.session.sightings or not self.session.clicks:
            return None
        start = self.model.with_head_axes() if head_axes else self.model
        calibrated, _, numbers = calibrate(start, [self.session], holdout_every)
        path = self.out_dir / CALIBRATION_FILE
        calibrated.save(path)
        path.with_suffix(".md").write_text(report(calibrated, start, numbers))
        return path

    # ---- showing -----------------------------------------------------------------------------------------------

    def render(self) -> None:
        """Show both pictures, the predicted fingertips, the clicks, the board's outline when placing it."""
        if self.head is None or self.wrist is None or self.look is None:
            return
        q = self.model.joint_vector(self.look.arms, self.look.head_ticks)
        head_tip, wrist_tip = jv.fingertips(self.model, self.arm, q)
        in_head = self.model.project(np.array([head_tip, wrist_tip]), q)[
            0
        ]  # both faces, in the head's picture
        predicted = {
            "head:pinch": in_head[0],
            "head:wrist": in_head[1],
            "wrist": jv.wrist_pixels(self.model, self.arm, q, wrist_tip[None])[0],
        }
        expected_face = f"head:{self.head_face}"  # the face the model expects the head to see
        titles = {
            pane: PANE_TITLES[pane] + (" (expected)" if pane == expected_face else "") for pane in PANES
        }
        # the board where the model has it: its outline (the suggested placement, then the measured one), and once
        # measured its corners, in every pane: their gap to the corners found is the model's error
        outlines: dict[str, np.ndarray] = {}
        expected: dict[str, np.ndarray] = {}
        if self.root_board is not None:
            points = self.board.points()  # any board: the rectangle its points span
            (x0, y0), (x1, y1) = points[:, :2].min(axis=0), points[:, :2].max(axis=0)
            corners = np.array([[x0, y0, 0.0], [x1, y0, 0.0], [x1, y1, 0.0], [x0, y1, 0.0]])
            root = corners @ self.root_board[:3, :3].T + self.root_board[:3, 3]
            outlines["head"] = self.model.project(root, q)[0]
            if self.board_known:
                every = jv.board_points(self.board, self.root_board)
                outlines["wrist"] = jv.wrist_pixels(self.model, self.arm, q, root)
                expected = {
                    "head": self.model.project(every, q)[0],
                    "wrist": jv.wrist_pixels(self.model, self.arm, q, every),
                }
        edge_on = EDGE_ON_DEG[0] <= self.face_angle <= EDGE_ON_DEG[1]
        lines = [
            f"{self.arm} arm | {self.phase} | placement {min(self.placement + 1, self.n_placements)}/{self.n_placements} | "
            f"{len(self.session.sightings)} board sightings, {len(self.session.clicks)} clicks"
            + (" | NEARLY EDGE-ON: click the head only if you can tell the face" if edge_on else ""),
            "orange: the board where the model has it, green: the corners found"
            + (f" | {self.board_warning.upper()}" if self.board_warning else ""),
            "n next | a fingertip in the pane of its face: magenta the pinch face, cyan the wrist camera's side | "
            "arrows head | u undo | q fit",
            self.message,
        ]
        window, self.regions = compose(
            {"head": self.head.color, "wrist": self.wrist.color},
            predicted,
            self.frame_clicks,
            outlines,
            lines,
            {"head": self.live.pixels} if self.phase == "place" and self.live is not None else self.detected,
            titles,
            expected,
        )
        self.show(window)

    def handle(self, action: str) -> bool:
        """Do one key's action. False when it is time to quit."""
        if action == "quit":
            return False
        if action == "next":
            self.next()
        elif action == "undo":
            self.undo()
        elif action in ("up", "down", "left", "right"):
            self.nudge(action)
        return True


def main() -> None:
    """Command line entry point; see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("endpoint", nargs="?", default=f"tcp://localhost:{DEFAULT_PORT}")
    parser.add_argument("--arm", choices=["left", "right"], required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--start-from", type=Path, required=True, help="a calibration.urdf with the arm's wrist camera"
    )
    parser.add_argument("--board", default="checkerboard:8x6:0.025", help="the board (boards.parse_board)")
    parser.add_argument("--placements", type=int, default=2)
    parser.add_argument("--arm-views", type=int, default=10, help="arm views per placement")
    parser.add_argument("--head-views", type=int, default=6, help="head views per placement, the arms folded")
    parser.add_argument("--low-views", type=int, default=10, help="poses low over the table")
    parser.add_argument("--max-vel", type=float, default=20.0, help="arm speed limit, deg/s (20)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--wrist-tip",
        choices=WRIST_TIPS,
        default="sweep",
        help="the wrist camera's fingertip: measured by a gripper sweep after q (default), or clicked",
    )
    parser.add_argument(
        "--head-axes",
        action="store_true",
        help="fit the head's pan and tilt axes' tilts too (4 more parameters)",
    )
    args = parser.parse_args()

    window = "joint calibration (n next; click each fingertip in its face's pane; q fit and quit)"
    pending: list[tuple[int, int]] = []

    def on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            pending.append((x, y))

    def show(image: np.ndarray) -> None:
        cv2.imshow(window, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with FelixClient(args.endpoint, timeout_s=20.0) as client, client.session():
        session = JointCollect(
            client,
            args.arm,
            args.out_dir,
            show,
            args.start_from,
            parse_board(args.board),
            args.placements,
            args.arm_views,
            args.head_views,
            args.low_views,
            args.max_vel,
            resume=args.resume,
            wrist_tip=args.wrist_tip,
        )
        cv2.setMouseCallback(window, on_mouse)
        try:
            while True:
                key = cv2.waitKeyEx(30)
                while pending:
                    session.click_at(*pending.pop(0))
                session.refresh()
                action = decode_key(key, KEYS)
                if action is not None and not session.handle(action):
                    break
        finally:
            cv2.destroyAllWindows()
        if args.wrist_tip == "sweep":
            added = session.measure_wrist_tip()
            print(
                f"the wrist camera's fingertip: {added} frames of the sweep into the session"
                if added
                else "the wrist camera's fingertip was not measured: the fit has the head's clicks only for it"
            )
        client.home(max_vel=args.max_vel)
    print(f"saved to {args.out_dir / SESSION_FILE}")
    path = session.finish(head_axes=args.head_axes)
    if path is not None:
        print(path.with_suffix(".md").read_text())
        print(f"Wrote {path} and {path.with_suffix('.md')}")


if __name__ == "__main__":
    main()
