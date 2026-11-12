"""Collect a joint calibration of the head, an arm and its wrist camera (`joint_fit`): one arm per session.

  uv run python -m lerobot.felix_jr.calibration.joint_collect tcp://<pi>:5555 --arm right --out-dir <scratch folder>
      --start-from <calibration.urdf> [--board checkerboard:8x6:0.025] [--placements 2] [--arm-views 10]
      [--head-views 6] [--low-views 10] [--max-vel 20] [--resume]

`docs/camera_calibration.md`, "The rework". Start the server on the Pi with the arm and the head. `--start-from` must
have the arm's wrist camera (its lens, from `calibration.wrist_camera`, and a first pose on the gripper). The window
shows the head's picture (left) and the wrist camera's (right), each with the fingertip the model predicts circled.
Per board placement:

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

In the head's picture both faces' tips are circled, P (the pinch point) and W (the wrist camera's side, one finger's
thickness away); the bold one is the face the next head click is on, the one the model expects the head to see. If
the picture shows the other face, `f` switches.

Keys: `n` next, `f` the head click's face, arrows turn the head (if the board is cut off), `u` undo the last click, `q` stop and
fit. Everything is saved as you go (`joint_session.json`, `images/`), `--resume` continues. At the end the session
is fitted (`joint_fit.calibrate`) and `calibration.urdf` and `calibration.md` written to `--out-dir`. The script runs
in a client session: Ctrl+C or a crash sends the robot to rest.
"""

import argparse
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

PANE = (640, 480)  # each picture is shown at this size
STATUS_HEIGHT = 70
LIVE_PERIOD_S = 0.2  # the head picture's refresh while placing the board
CALIBRATION_FILE = "calibration.urdf"
KEYS = {ord("n"): "next", ord("u"): "undo", ord("f"): "face", ord("q"): "quit", 27: "quit"}
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


def compose(
    pictures: dict[str, np.ndarray],
    predicted: dict[str, np.ndarray | None],
    clicks: dict[str, list[tuple[float, float]]],
    outlines: dict[str, np.ndarray],
    lines: list[str],
    found: dict[str, np.ndarray] | None = None,
    labeled: dict[str, list[tuple[np.ndarray, str, bool]]] | None = None,
) -> tuple[np.ndarray, list[Region]]:
    """The window: the head's picture and the wrist's side by side, each with its predicted tip circled.

    `found`: board corners detected in a pane, drawn as green dots. `labeled`: predicted tips with a label each (the
    fingertip's two faces), the ones flagged True drawn bold (the face the next click is on).
    """
    window = np.zeros((PANE[1] + STATUS_HEIGHT, 2 * PANE[0], 3), np.uint8)
    regions: list[Region] = []
    for i, pane in enumerate(("head", "wrist")):
        picture = pictures[pane]
        scale = min(PANE[0] / picture.shape[1], PANE[1] / picture.shape[0])
        shown = cv2.resize(picture, (int(picture.shape[1] * scale), int(picture.shape[0] * scale)))
        x0 = i * PANE[0]

        def at(uv: np.ndarray, s: float = scale) -> tuple[int, int]:
            return int(round((uv[0] + 0.5) * s - 0.5)), int(round((uv[1] + 0.5) * s - 0.5))

        if pane in outlines and np.isfinite(outlines[pane]).all():
            cv2.polylines(
                shown,
                [np.array([at(p) for p in outlines[pane]], np.int32)],
                True,
                (255, 160, 0),
                2,
                cv2.LINE_AA,
            )
        for p in (found or {}).get(pane, []):
            cv2.circle(shown, at(p), 3, (0, 255, 0), -1, cv2.LINE_AA)
        for p in clicks.get(pane, []):
            cv2.drawMarker(shown, at(np.array(p)), (0, 255, 0), cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
        tip = predicted.get(pane)
        if tip is not None and np.isfinite(tip).all():
            cv2.circle(shown, at(tip), 12, (0, 255, 255), 1, cv2.LINE_AA)
        for uv, label, bold in (labeled or {}).get(pane, []):
            if np.isfinite(uv).all():
                x, y = at(uv)
                cv2.circle(shown, (x, y), 12, (0, 255, 255), 2 if bold else 1, cv2.LINE_AA)
                _draw_text(shown, [label], (x + 14, y - 8))
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
    ):
        """Load the start, check it against the robot, and send the arms to rest for the first placement."""
        self.client, self.arm, self.arm_id, self.out_dir, self.show = client, arm, f"{arm}_arm", out_dir, show
        self.board, self.n_placements, self.n_arm_views = board, placements, arm_views
        self.n_head_views, self.n_low_views, self.max_vel = head_views, low_views, max_vel
        self.head_step_deg = head_step_deg
        self.rng = np.random.default_rng(seed)
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
        self.frame_clicks: dict[str, list[tuple[float, float]]] = {"head": [], "wrist": []}
        self.regions: list[Region] = []
        self.head_face, self.face_angle = "pinch", 0.0  # set by each capture (`_capture`)
        self.live: Detection | None = None  # the board in the live head picture while placing it
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
        return True

    def _capture(self, after: float | None = None) -> None:
        """Take both pictures and the moment they show (the arm's joints from the wrist's, the head's ticks)."""
        self.head = self.client.capture(after=after)
        self.wrist = self.client.wrist_capture(self.arm, color_format="png", after=after)
        arm = {k: float(v) for k, v in self.wrist.state[self.arm_id].items()}
        ticks = {n: int(self.head.state["head"][n]["ticks"]) for n in HEAD_JOINTS}
        self.look = Look({self.arm: arm}, ticks)
        self.frame_clicks = {"head": [], "wrist": []}
        self.saved = []
        # the face of the fingertip the head shows, as the model has it: the next head click's (`f` flips it)
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
            f"{'seen' if head else 'NOT seen'} by the head; click the fingertip in each picture it shows in, then n"
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
        self.message = (
            f"low pose {self.view_index}/{len(self.views)}{seen}: click the fingertip in each picture, then n"
        )
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

    def click(self, pane: str, u: float, v: float) -> TipClick:
        """Record the fingertip at pixel (u, v) of `pane`'s current picture ("head" or "wrist")."""
        capture = self.head if pane == "head" else self.wrist
        assert capture is not None and self.look is not None
        camera = HEAD if pane == "head" else self.arm
        face = self.head_face if pane == "head" else "wrist"  # the wrist camera sees only its own side
        click = TipClick(camera, self.arm, self.look, np.array([u, v]), self._image(capture, pane), face)
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
            pane = "head" if last.camera == HEAD else "wrist"
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
        predicted = {"wrist": jv.wrist_pixels(self.model, self.arm, q, wrist_tip[None])[0]}
        in_head = self.model.project(np.array([head_tip, wrist_tip]), q)[
            0
        ]  # both faces, in the head's picture
        labeled = {
            "head": [
                (in_head[0], "P", self.head_face == "pinch"),
                (in_head[1], "W", self.head_face == "wrist"),
            ]
        }
        outlines = {}
        if self.phase == "place" and self.root_board is not None:
            points = self.board.points()  # any board: the rectangle its points span
            (x0, y0), (x1, y1) = points[:, :2].min(axis=0), points[:, :2].max(axis=0)
            corners = np.array([[x0, y0, 0.0], [x1, y0, 0.0], [x1, y1, 0.0], [x0, y1, 0.0]])
            root = corners @ self.root_board[:3, :3].T + self.root_board[:3, 3]
            outlines["head"] = self.model.project(root, q)[0]
        lines = [
            f"{self.arm} arm | {self.phase} | placement {min(self.placement + 1, self.n_placements)}/{self.n_placements} | "
            f"{len(self.session.sightings)} board sightings, {len(self.session.clicks)} clicks",
            "n next | click the fingertip in each picture | f the head's face | arrows head | u undo | q fit and quit",
            f"head clicks on the {'P (pinch)' if self.head_face == 'pinch' else 'W (wrist-camera)'} face"
            + (
                " | nearly edge-on: click only if you can tell the face"
                if EDGE_ON_DEG[0] <= self.face_angle <= EDGE_ON_DEG[1]
                else ""
            )
            + (f" | {self.message}" if self.message else ""),
        ]
        window, self.regions = compose(
            {"head": self.head.color, "wrist": self.wrist.color},
            predicted,
            self.frame_clicks,
            outlines,
            lines,
            {"head": self.live.pixels} if self.phase == "place" and self.live is not None else None,
            labeled,
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
        elif action == "face":
            self.head_face = "wrist" if self.head_face == "pinch" else "pinch"
            self.render()
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
        "--head-axes",
        action="store_true",
        help="fit the head's pan and tilt axes' tilts too (4 more parameters)",
    )
    args = parser.parse_args()

    window = "joint calibration (n next; click the fingertips; q fit and quit)"
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
        client.home(max_vel=args.max_vel)
    print(f"saved to {args.out_dir / SESSION_FILE}")
    path = session.finish(head_axes=args.head_axes)
    if path is not None:
        print(path.with_suffix(".md").read_text())
        print(f"Wrote {path} and {path.with_suffix('.md')}")


if __name__ == "__main__":
    main()
