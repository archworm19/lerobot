"""Calibrate the wrist cameras' lenses (intrinsics) from a ChArUco board held in front of them.

  uv run python -m lerobot.felix_jr.calibration.wrist_camera tcp://<pi>:5555 --start-from <calibration.urdf>
      --out-dir <folder> [--arms left,right] [--board charuco:9x7:0.025:0.018:DICT_4X4_50] [--lens rational|fisheye]
      [--no-focus] [--resume] [--step-cm 2] [--max-vel 20]
  uv run python -m lerobot.felix_jr.calibration.wrist_camera --refit --start-from <calibration.urdf> --out-dir <folder>

Once per lens: redo it when a lens is refocused or a camera replaced, not when a mount moves. The camera's pose on the
gripper is the joint session's (`joint_collect`), which ties it to the head's view of the fingers; this step keeps
`--start-from`'s pose as it was (`docs/calibration_plan.md`, items 1-2: the camera-pose step was retired 2026-11-16).

The server must run with the wrist cameras (`server.server --left-wrist-camera ... --right-wrist-camera ...`). The
board is the printed ChArUco (`boards.PRINTED_BOARD`; measure a square: printers scale).

After one approval in the terminal, both arms go to their start poses (the arms not calibrated too: out of the way):
the gripper tilted down, about 18 cm above the table and 22 cm in front of the arm. They stay there until the end, when
the robot goes to rest. Then for each arm in `--arms` in turn, a window shows its wrist camera:

0. **Focus.** The window streams the camera with a sharpness score (`sharpness`: of the board where it is found, else
   of the middle of the picture) and the best so far. Hold the board about 20 cm in front of the camera (where the
   objects will be when grasping) and turn the lens until the score peaks. Enter, a click in the window or `c`
   accepts (`--no-focus` skips it).
1. **The lens.** Hold the board in front of the camera and move it slowly: tilted every way, near and far, and into
   the corners of the picture (the board may run off the picture's edge: its markers name the corners that show).
   Views are kept by themselves (`LensViews`): a board held still, with enough of its corners, that shows a new pose or
   covers picture cells no kept view does; at most `MAX_LENS_VIEWS`. The cells covered so far are drawn. `c` fits the
   lens (OpenCV's rational model, or `--lens fisheye` for a lens wider than about 120 deg) once there are
   `MIN_INTRINSIC_VIEWS` and the views settle it (`lens_settled`).

Keys: arrows, `w` / `s`, `[` / `]`, `,` / `.` move the arm a little (the tip as the reference); space keeps this view;
`u` drops the last view; `c` fits; `x x` skips the arm's camera; `q` / Esc stops (what is fitted so far is written).

The output, `--out-dir/calibration.urdf`, is `--start-from` with the new lenses (`XleRobotModel.with_wrist_camera`,
the camera's pose on the gripper kept), and `wrist_cameras.md` reports the fit. The views (`<arm>_views.json`,
`images/`) are kept: `--refit` fits them again with no robot. The script runs inside a client session, so on Ctrl+C or
a crash the robot goes to rest.
"""

import argparse
import json
import os
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# OpenCL off for the whole process (not only this thread): on macOS its kernels fail with alarming but harmless errors
os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from lerobot.felix_jr.calibration.boards import (  # noqa: E402
    PRINTED_BOARD,
    Board,
    Detection,
    board_from_dict,
    parse_board,
)
from lerobot.felix_jr.calibration.controls import decode_key  # noqa: E402
from lerobot.felix_jr.calibration.merge_urdf import wrist_camera_frame  # noqa: E402
from lerobot.felix_jr.calibration.xle_robot_model import (  # noqa: E402
    HEAD_JOINTS,
    WRIST_CAMERA_MODELS,
    WristCamera,
    XleRobotModel,
    arm_record,
    pinch_frame,
)
from lerobot.felix_jr.server.client import FelixClient, WristCapture  # noqa: E402
from lerobot.felix_jr.server.protocol import DEFAULT_PORT, ServerError  # noqa: E402

ARMS = ("left", "right")
CALIBRATION_FILE = "calibration.urdf"
REPORT_FILE = "wrist_cameras.md"
MIN_INTRINSIC_VIEWS = 12
# a view's board shows at least this part of its corners (ChArUco names any it finds: a few in a corner of the
# picture would be kept for their cells and say little about the lens)
LENS_MIN_FRACTION = 0.3
MAX_LENS_VIEWS = 100  # at most: past it, nothing more is kept (the fit is slow and gains nothing)
# the board held still: against the frame before, the same corners within this (median) - a moving board's corners
# blur and lag
STILL_PX = 1.5
# a picture's cells (columns x rows) for the coverage: a view showing corners in a cell no kept view does is kept
COVERAGE_CELLS = (8, 6)
# a cell a view covers has this many of its corners at least (one could be a corner jittering across a cell's edge)
CELL_MIN_CORNERS = 2
# the lens views should include the board tilted toward each side (its far side shrunk by this much) and
# at different distances (`LENS_SCALE_RATIO`), shown as guidance; what decides is `lens_settled`. Flat views at one
# distance leave the focal length trading off with the distance (wrist_cams5, 2026-11-07: 47 flat views, fx anywhere
# from 314 to 3733)
LENS_TILT = 0.07  # ~20-25 deg for a 25 x 17 cm board at 25-35 cm (flat views read under 0.05)
LENS_SCALE_RATIO = 1.3  # the nearest view's board at least this much bigger than the farthest's (flat: 1.09)
LENS_SETTLED = (
    0.03  # the lens is accepted when its focal length, fitted on random halves of the views, agrees this well
)
LENS_SPLITS = 12  # how many random halves (`lens_settled`)
FOCAL_RANGE = (0.2, 3.0)  # a focal length (px) outside this range times the picture's width is refused
# the start pose keeps every joint this far from its bounds, so the keys can still tilt and turn the gripper from it
START_JOINT_MARGIN_DEG = 15.0
# the start pose: the gripper's tip this far in front of the arm's base, the first of these heights above the base
# (about 6 cm above the table on this robot) the arm reaches with the gripper tilted down the steepest of these pitches
START_AHEAD_M = 0.22
START_HEIGHTS_M = (0.12, 0.08, 0.04)
START_PITCHES_DEG = (-90.0, -80.0, -70.0, -60.0, -50.0, -40.0)
PITCH_STEP_DEG = 10.0
MOVE_TOL_DEG = 3.0  # a move ends once every joint is this close
ROLL_STEP_DEG = 15.0
# a board counts as a new pose when, against every view kept, it moved or resized this much (of the picture's size),
# or its apparent tilt changed this much, or it turned this far in the picture (`view_signature`)
NEW_VIEW_PX = 0.05
NEW_VIEW_TILT = 0.08
NEW_VIEW_TURN_DEG = 20.0
KEY_ACTIONS = {
    ord("w"): "up_z",
    ord("s"): "down_z",
    ord("["): "pitch_down",
    ord("]"): "pitch_up",
    ord(","): "roll_left",
    ord("."): "roll_right",
    ord(" "): "take",
    ord("u"): "undo",
    ord("c"): "calibrate",
    ord("x"): "skip",
    ord("q"): "quit",
    27: "quit",
    13: "accept",  # Enter
    10: "accept",
}
PHASE_TITLES = {"focus": "1/2 FOCUS", "intrinsics": "2/2 LENS"}
CORNER_COLOR, COVERED_COLOR = (0, 255, 0), (0, 160, 0)  # RGB
LAST_FRAME_FILE = (
    "last_frame.png"  # the latest frame the board was not found in (to see why), every few seconds
)
LAST_FRAME_EVERY_S = 2.0
SHARP_ENOUGH = (
    100.0  # a sharpness (`sharpness`) above this reads as in focus; the out-of-focus lenses read ~2
)


# ---- the board in a picture ---------------------------------------------------------------------------------


def sharpness(image: np.ndarray, pixels: np.ndarray | None = None) -> float:
    """How sharp an RGB image is: the variance of its Laplacian (higher is sharper).

    Over the board (`pixels`' hull, its corners found), else the middle half of the picture. It depends on the scene,
    so compare it while turning the lens.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    if pixels is not None and len(pixels) >= 3:
        mask = np.zeros(gray.shape, np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(pixels.astype(np.int32)), 1)
        return float(laplacian[mask > 0].var())
    height, width = gray.shape
    return float(laplacian[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4].var())


def board_outline(board: Board) -> np.ndarray:
    """The ids of the board's four outermost corners (4,): its first row's ends, its last row's ends."""
    cols, rows = board.cols - 1, board.rows - 1  # inner corners
    return np.array([0, cols - 1, cols * rows - 1, cols * (rows - 1)])


def view_signature(detection: Detection, size: tuple[int, int], board: Board) -> np.ndarray:
    """Where the whole board is in the picture, how big, how tilted and turned, from the corners found.

    (center x, center y, scale, tilt x, tilt y, turn), from the board's outline (`board_outline`) put in the picture
    by the homography of the corners found: the same for any part of the board found (a board half off the picture
    has its center near or past the edge). Positions and scale as fractions of the picture; the tilts are how much
    the board's far side shrinks against its near one (0 for a board square to the camera); the turn is the direction
    of its first row (deg). The lens's distortion bends the homography a little: fine for telling poses apart.
    """
    width, height = size
    plane = board.points(detection.ids)[:, :2]
    homography, _ = cv2.findHomography(plane, detection.pixels)
    if homography is None:
        raise ValueError("the corners found do not give the board's homography (in a line?)")
    outline = board.points(board_outline(board))[:, :2]
    quad = cv2.perspectiveTransform(outline.reshape(-1, 1, 2), homography).reshape(-1, 2)
    first, last, far_last, far_first = quad
    center = quad.mean(axis=0) / [width, height]
    scale = float(np.sqrt(abs(cv2.contourArea(quad.astype(np.float32))) / (width * height)))
    top, bottom = np.linalg.norm(last - first), np.linalg.norm(far_last - far_first)
    left, right = np.linalg.norm(far_first - first), np.linalg.norm(far_last - last)
    tilt_x = (top - bottom) / max(top + bottom, 1e-9)
    tilt_y = (left - right) / max(left + right, 1e-9)
    row = last - first
    return np.array(
        [center[0], center[1], scale, tilt_x, tilt_y, float(np.degrees(np.arctan2(row[1], row[0])))]
    )


def is_new_pose(signature: np.ndarray, kept: Sequence[np.ndarray]) -> bool:
    """Whether a view's board differs from every kept one's.

    It does when it moved or resized by `NEW_VIEW_PX`, tilted by `NEW_VIEW_TILT` or turned by `NEW_VIEW_TURN_DEG`.
    """
    for other in kept:
        moved = np.linalg.norm(signature[:3] - other[:3])
        tilted = np.abs(signature[3:5] - other[3:5]).max()
        turned = abs((signature[5] - other[5] + 180.0) % 360.0 - 180.0)
        if moved < NEW_VIEW_PX and tilted < NEW_VIEW_TILT and turned < NEW_VIEW_TURN_DEG:
            return False
    return True


def cells(pixels: np.ndarray, size: tuple[int, int]) -> set[tuple[int, int]]:
    """The picture's cells (`COVERAGE_CELLS`: column, row) with `CELL_MIN_CORNERS` of the pixels in them."""
    width, height = size
    cols, rows = COVERAGE_CELLS
    u = np.clip((pixels[:, 0] / width * cols).astype(int), 0, cols - 1)
    v = np.clip((pixels[:, 1] / height * rows).astype(int), 0, rows - 1)
    found, counts = np.unique(np.c_[u, v], axis=0, return_counts=True)
    return {(int(a), int(b)) for (a, b), n in zip(found, counts, strict=True) if n >= CELL_MIN_CORNERS}


@dataclass
class LensViews:
    """Which of the frames to keep for the lens: few and varied, out of the many a ChArUco board is found in.

    A frame's board is kept (`offer`) when it is held still (its corners within `STILL_PX` of the frame before's), shows
    `LENS_MIN_FRACTION` of the board's corners, and either is a new pose (`is_new_pose`) or puts corners in a picture
    cell no kept view has (`cells`: the picture's corners, where the distortion is decided, are reached by partial
    boards); at most `MAX_LENS_VIEWS`. Each `offer` says why, in words, for the status line.
    """

    board: Board
    size: tuple[int, int]
    views: list[Detection] = field(default_factory=list)
    signatures: list[np.ndarray] = field(default_factory=list)
    covered: set[tuple[int, int]] = field(default_factory=set)
    last: Detection | None = None  # the frame before's board (`still`)

    def still(self, detection: Detection) -> bool:
        """Whether the board is where the frame before had it (the corners both found, within `STILL_PX`)."""
        last, self.last = self.last, detection
        if last is None:
            return False
        common, here, there = np.intersect1d(detection.ids, last.ids, return_indices=True)
        if len(common) < len(detection) // 2:
            return False
        moved = np.linalg.norm(detection.pixels[here] - last.pixels[there], axis=1)
        return bool(np.median(moved) <= STILL_PX)

    def enough(self, detection: Detection) -> bool:
        """Whether the view shows `LENS_MIN_FRACTION` of the board's corners."""
        return len(detection) >= LENS_MIN_FRACTION * len(self.board.points())

    def offer(self, detection: Detection | None) -> tuple[bool, str]:
        """Keep the frame's board if it is worth it (the class docstring): (kept, what became of it, in words)."""
        if detection is None:
            self.last = None
            return False, "board NOT found"
        still = self.still(detection)
        total = len(self.board.points())
        if len(self.views) >= MAX_LENS_VIEWS:
            return False, f"{MAX_LENS_VIEWS} views kept: enough (c fits)"
        if not self.enough(detection):
            return False, f"{len(detection)} of {total} corners: too few (show more of the board)"
        if not still:
            return False, "hold the board still"
        signature = view_signature(detection, self.size, self.board)
        new_cells = cells(detection.pixels, self.size) - self.covered
        if not new_cells and not is_new_pose(signature, self.signatures):
            return False, "like a kept view: tilt, turn or move it"
        self.keep(detection, signature)
        return True, f"kept ({len(detection)} corners" + (
            f", {len(new_cells)} new cells)" if new_cells else ")"
        )

    def keep(self, detection: Detection, signature: np.ndarray | None = None) -> None:
        """Keep a view whatever it is like (space; resuming)."""
        self.views.append(detection)
        self.signatures.append(
            view_signature(detection, self.size, self.board) if signature is None else signature
        )
        self.covered |= cells(detection.pixels, self.size)

    def drop_last(self) -> None:
        """Forget the last view kept (`u`)."""
        self.views.pop()
        self.signatures.pop()
        self.covered = set().union(*(cells(v.pixels, self.size) for v in self.views))


# ---- the fits ------------------------------------------------------------------------------------------------


def lens_coverage_missing(signatures: Sequence[np.ndarray]) -> list[str]:
    """What the lens views still lack (`LENS_TILT`, `LENS_SCALE_RATIO`), in words; empty when they have it all."""
    if not signatures:
        return ["views"]
    sigs = np.asarray(signatures)
    missing = [
        name
        for name, column, sign in (
            # the board's edge nearer the camera (it looks longer), as seen in the window
            ("with its top edge nearer", 3, 1.0),
            ("with its bottom edge nearer", 3, -1.0),
            ("with its left edge nearer", 4, 1.0),
            ("with its right edge nearer", 4, -1.0),
        )
        if not (sign * sigs[:, column] > LENS_TILT).any()
    ]
    if sigs[:, 2].max() < LENS_SCALE_RATIO * sigs[:, 2].min():
        missing.append("nearer and farther")
    return missing


def _objects_images(views: Sequence[Detection], board: Board) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Each view's board points (N, 1, 3) and pixels (N, 1, 2), float64, for OpenCV's calibrations."""
    objects = [board.points(v.ids).reshape(-1, 1, 3) for v in views]
    images = [np.asarray(v.pixels, dtype=np.float64).reshape(-1, 1, 2) for v in views]
    return objects, images


def lens_settled(
    views: Sequence[Detection], board: Board, size: tuple[int, int], seed: int = 0
) -> tuple[float, float]:
    """How well the views pin the focal length down: its mean and its spread (relative) over fits on random halves.

    Views that do not constrain it (wrist_cams5: flat, at one distance) give halves that disagree; good views agree
    within a few percent (wrist_cams6, held up in a cramped workspace: 342 +- 2.4%). Plain 5-coefficient fits, quick.
    """
    rng = np.random.default_rng(seed)
    objects, images = _objects_images(views, board)
    focal = []
    for _ in range(LENS_SPLITS):
        half = rng.choice(len(images), size=max(len(images) // 2, 3), replace=False)
        _, matrix, *_ = cv2.calibrateCamera(
            [objects[i].astype(np.float32) for i in half],
            [images[i].astype(np.float32) for i in half],
            size,
            None,
            None,
        )
        focal.append(0.5 * (matrix[0, 0] + matrix[1, 1]))
    return float(np.mean(focal)), float(np.std(focal) / np.mean(focal))


def calibrate_intrinsics(
    views: Sequence[Detection], board: Board, size: tuple[int, int], lens: str = "rational"
) -> tuple[WristCamera, np.ndarray]:
    """The lens from views of the board (any corners of it each): the camera and each view's RMS error (px)."""
    if lens not in WRIST_CAMERA_MODELS:
        raise ValueError(f"unknown lens model {lens!r}: one of {WRIST_CAMERA_MODELS}")
    if len(views) < 3:
        raise ValueError(f"{len(views)} views: too few to fit a lens")
    objects, images = _objects_images(views, board)
    if lens == "fisheye":
        flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
        rms, matrix, coeffs, rvecs, tvecs = cv2.fisheye.calibrate(
            objects, images, size, None, None, flags=flags
        )
    else:
        rms, matrix, coeffs, rvecs, tvecs = cv2.calibrateCamera(
            [o.astype(np.float32) for o in objects],
            [i.astype(np.float32) for i in images],
            size,
            None,
            None,
            flags=cv2.CALIB_RATIONAL_MODEL,
        )
    camera = WristCamera(
        width=int(size[0]),
        height=int(size[1]),
        fx=float(matrix[0, 0]),
        fy=float(matrix[1, 1]),
        ppx=float(matrix[0, 2]),
        ppy=float(matrix[1, 2]),
        model=lens,
        coeffs=tuple(float(c) for c in np.ravel(coeffs)),
        rms_px=float(rms),
        n_views=len(views),
    )
    errors = []
    for view, rvec, tvec in zip(views, rvecs, tvecs, strict=True):
        pose = np.eye(4)
        pose[:3, :3] = cv2.Rodrigues(np.ravel(rvec))[0]
        pose[:3, 3] = np.ravel(tvec)
        predicted = camera.project(board.points(view.ids) @ pose[:3, :3].T + pose[:3, 3])
        errors.append(float(np.sqrt(np.mean(np.sum((predicted - view.pixels) ** 2, axis=1)))))
    return camera, np.array(errors)


def fingertips_in_wrist_image(
    model: XleRobotModel, arm: str, camera: WristCamera, arm_pose: dict[str, float]
) -> np.ndarray:
    """Where the model puts the two fingertips (the pinch point, the moving tip) in the wrist image (2, 2) px."""
    q = model.joint_vector({arm: arm_pose})
    opening = (
        max(model.opening_deg(arm, float(arm_pose["gripper.pos"])), 0.0) if arm in model.gripper_maps else 0.0
    )
    pinch = model.frame_pose(pinch_frame(arm), q)
    tips = np.array([pinch[:3, 3], model.moving_tip(arm, q, opening)])
    tips = tips + pinch[:3, :3] @ model.wrist_face(arm)  # the face the wrist camera sees
    camera_from_root = np.linalg.inv(model.frame_pose(wrist_camera_frame(arm), q))
    return camera.project(tips @ camera_from_root[:3, :3].T + camera_from_root[:3, 3])


# ---- saving -------------------------------------------------------------------------------------------------


@dataclass
class ArmRecord:
    """What a session collected for one arm (`<arm>_views.json`): the board, the lens's views, the arm's calibration."""

    arm: str
    board: Board
    size: tuple[int, int]
    lens: str
    arm_calibration: dict[str, Any]
    intrinsic_views: list[Detection] = field(default_factory=list)
    focus_sharpness: float | None = None  # the sharpness when the focus was accepted

    def save(self, directory: Path) -> None:
        """Write `<arm>_views.json` in `directory`."""
        data = {
            "arm": self.arm,
            "board": self.board.to_dict(),
            "size": list(self.size),
            "lens": self.lens,
            "arm_calibration": self.arm_calibration,
            "intrinsic_views": [
                {"ids": v.ids.tolist(), "pixels": np.asarray(v.pixels).tolist()} for v in self.intrinsic_views
            ],
            "focus_sharpness": self.focus_sharpness,
        }
        (directory / f"{self.arm}_views.json").write_text(json.dumps(data) + "\n")

    @classmethod
    def load(cls, path: Path) -> "ArmRecord":
        """Read a `<arm>_views.json` (a checkerboard's, from before 2026-11-16, is refused: `boards`)."""
        data = json.loads(path.read_text())
        return cls(
            arm=data["arm"],
            board=board_from_dict(data["board"]),
            size=tuple(data["size"]),
            lens=data["lens"],
            arm_calibration=data["arm_calibration"],
            intrinsic_views=[
                Detection(np.asarray(v["ids"], dtype=int), np.asarray(v["pixels"], dtype=np.float64))
                for v in data["intrinsic_views"]
            ],
            focus_sharpness=data.get("focus_sharpness"),
        )


def with_lens(model: XleRobotModel, arm: str, lens: WristCamera) -> XleRobotModel:
    """`model` with `arm`'s wrist camera's lens replaced by `lens`.

    Its pose on the gripper (the model's wrist camera parameters: the design's, or a joint session's corrections) is
    untouched.
    """
    return model.with_wrist_camera(arm, lens)


def fit_record(model: XleRobotModel, record: ArmRecord) -> tuple[WristCamera | None, list[str]]:
    """Fit the lens from `record`'s views. Returns the lens (None: too few views) and the report's lines."""
    lines = [f"## {record.arm} wrist camera", ""]
    if len(record.intrinsic_views) < 3:
        return None, [*lines, "Too few views for the lens: not calibrated.", ""]
    camera, errors = calibrate_intrinsics(record.intrinsic_views, record.board, record.size, record.lens)
    if record.focus_sharpness is not None:
        lines.append(
            f"- focus accepted at sharpness {record.focus_sharpness:.0f} (in focus above {SHARP_ENOUGH:.0f})"
        )
    corners = [len(v) for v in record.intrinsic_views]
    moved = [
        abs(v) for k, v in model.estimate_dict().items() if f"{record.arm}_wrist_camera_joint" in k and v
    ]
    lines += [
        f"- lens: {camera.model}, {camera.width} x {camera.height}, fx {camera.fx:.1f} fy {camera.fy:.1f}, "
        f"center ({camera.ppx:.1f}, {camera.ppy:.1f}); RMS {camera.rms_px or 0:.2f} px over {camera.n_views} views "
        f"(worst {errors.max():.2f} px; {min(corners)}-{max(corners)} corners each)",
        "- pose on the gripper: "
        + (
            "the start's, a joint session's corrections to its design (kept)"
            if moved
            else "its design (`design_xlerobot_so101.urdf`): a joint session corrects it"
        ),
    ]
    return camera, [*lines, ""]


def write(model: XleRobotModel, records: Sequence[ArmRecord], out_dir: Path) -> Path | None:
    """Fit every arm's views and write the calibration (`calibration.urdf`) and the report; None if nothing fits."""
    report = ["# Wrist cameras: the lenses", ""]
    fitted = False
    for record in records:
        camera, lines = fit_record(model, record)
        report += lines
        if camera is not None:
            model, fitted = with_lens(model, record.arm, camera), True
    if not fitted:
        return None
    path = out_dir / CALIBRATION_FILE
    model.save(path)
    (out_dir / REPORT_FILE).write_text("\n".join(report))
    return path


def refit(start_from: Path, out_dir: Path) -> Path | None:
    """Fit every `<arm>_views.json` in `out_dir` again, with no robot; write the calibration and the report."""
    model = XleRobotModel.load(start_from)
    records = []
    for path in sorted(out_dir.glob("*_views.json")):
        record = ArmRecord.load(path)
        if (
            record.arm in model.arm_calibration
            and model.arm_calibration[record.arm] != record.arm_calibration
        ):
            raise ValueError(f"{path} was taken under another arm calibration than {start_from}'s")
        records.append(record)
    if (out_dir / CALIBRATION_FILE).resolve() == start_from.resolve():
        raise ValueError(f"{out_dir / CALIBRATION_FILE} would overwrite --start-from: pick another folder")
    return write(model, records, out_dir)


# ---- the session --------------------------------------------------------------------------------------------


def wrap(lines: Sequence[str], width_px: int, scale: float = 0.5) -> list[str]:
    """`lines` broken where they would run wider than `width_px` on the picture: at " | ", else at spaces."""

    def width(text: str) -> int:
        return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]

    def greedy(parts: list[str], sep: str) -> list[str]:
        out: list[str] = []
        for part in parts:
            if out and width(out[-1] + sep + part) <= width_px:
                out[-1] += sep + part
            elif width(part) > width_px and sep != " ":
                out += greedy(part.split(" "), " ")
            else:
                out.append(part)
        return out

    return [piece for line in lines for piece in greedy(line.split(" | "), " | ")]


def annotate(
    image: np.ndarray,
    lines: Sequence[str],
    detection: Detection | None,
    covered: Iterable[tuple[int, int]] = (),
) -> np.ndarray:
    """A copy of the RGB `image` with the cells covered so far, the corners found, and the status lines."""
    out = image.copy()
    height, width = out.shape[:2]
    cols, rows = COVERAGE_CELLS
    for u, v in covered:  # a covered cell's frame
        x0, y0 = int(u * width / cols), int(v * height / rows)
        x1, y1 = int((u + 1) * width / cols) - 1, int((v + 1) * height / rows) - 1
        cv2.rectangle(out, (x0 + 2, y0 + 2), (x1 - 2, y1 - 2), COVERED_COLOR, 1, cv2.LINE_AA)
    if detection is not None:
        for u, v in detection.pixels:
            cv2.circle(out, (int(round(u)), int(round(v))), 3, CORNER_COLOR, -1, cv2.LINE_AA)
    for row, text in enumerate(wrap(lines, width - 16)):
        org = (8, 20 + 20 * row)
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


class WristCameraSession:
    """Calibrate one arm's wrist camera's lens. See the module docstring.

    `show` displays an RGB image. Nothing moves until `start` (which the caller approves first).
    """

    def __init__(
        self,
        client: FelixClient,
        model: XleRobotModel,
        arm: str,
        board: Board,
        out_dir: Path,
        show: Callable[[np.ndarray], None],
        lens: str = "rational",
        step_m: float = 0.02,
        max_vel: float = 20.0,
        focus: bool = True,
        resume: bool = False,
        next_arm: str | None = None,
        say: Callable[[str], None] = print,
    ):
        """Set up the session.

        With `focus`, it starts by streaming the camera for the lens to be focused (accepted with `accept`). With
        `resume`, the views saved in `out_dir` for this arm (`<arm>_views.json`) are kept and added to. `next_arm` is
        the arm whose camera comes after this one (None: this is the last), for the status line. `say` prints each
        step's keys as it starts (to the terminal).
        """
        self.next_arm = next_arm
        self.skipped = False  # `x x`: this arm's camera left out
        self.confirming: str | None = None  # "skip": the action waiting for its second key
        self.say = say
        self.announced: str | None = None  # the phase whose keys were last printed
        self.client, self.model, self.arm, self.board, self.out_dir, self.show = (
            client,
            model,
            arm,
            board,
            out_dir,
            show,
        )
        self.step_m, self.max_vel = step_m, max_vel
        self.arm_id = f"{arm}_arm"
        self.record = ArmRecord(arm, board, (0, 0), lens, arm_record(client.arm_calibration()[self.arm_id]))
        self.views: LensViews | None = None  # once the picture's size is known (the first frame)
        self.camera: WristCamera | None = None  # the fitted lens
        self.phase = "focus" if focus else "intrinsics"
        self.sharpness = 0.0  # of the current frame
        self.seen = ""  # what became of the board in the current frame (the status line says)
        self.last_frame_saved = 0.0
        self.best_sharpness = 0.0  # in the focus phase
        self.target: tuple[np.ndarray, float, float] | None = (
            None  # the tip's position, pitch and roll asked for
        )
        self.solved: dict[str, float] = {}  # the joints `move_to` last solved
        self.frame: WristCapture | None = None
        self.detection: Detection | None = None
        self.message = ""
        self.done = False
        (out_dir / "images").mkdir(parents=True, exist_ok=True)
        saved = out_dir / f"{arm}_views.json"
        if resume and saved.is_file():
            old = ArmRecord.load(saved)
            if old.board != board or old.arm_calibration != self.record.arm_calibration:
                raise ValueError(f"{saved} was taken with another board or arm calibration: a new --out-dir")
            self.record = old
            self.views = LensViews(board, old.size)
            for view in old.intrinsic_views:
                self.views.keep(view)
            self.message = f"resumed: {len(old.intrinsic_views)} lens views (c fits)"

    # ---- moving -----------------------------------------------------------------------------------------------

    def start_poses(self) -> list[tuple[np.ndarray, float, float]]:
        """The start poses to try, best first, as (tip position, pitch, roll).

        The tip `START_AHEAD_M` in front of the arm's base, each of `START_HEIGHTS_M` above it, at each of
        `START_PITCHES_DEG`, roll 0.
        """
        base = self.model.frame_pose(f"{self.arm}_base_link", self.model.joint_vector())[:3, 3]
        return [
            (np.array([base[0] - START_AHEAD_M, base[1], base[2] + up]), pitch, 0.0)
            for up in START_HEIGHTS_M
            for pitch in START_PITCHES_DEG
        ]

    def move_to(self, position: np.ndarray, pitch: float, roll: float, dry_run: bool = False) -> bool:
        """Solve and move the arm so its tip is at `position` with `pitch` and `roll`; False (and why) if it cannot.

        With `dry_run`, only solve.
        """
        from lerobot.felix_jr.harness.reach import solve_pose

        state = self.client.get_state()
        current = {k: float(v) for k, v in state[self.arm_id].items()}
        head = {n: int(state["head"][n]["ticks"]) for n in HEAD_JOINTS} if "head" in state else None
        capture = SimpleNamespace(arms={self.arm: current}, head_ticks=head)
        poses = solve_pose(self.model, self.arm, capture, position, pitch, None, roll, init=current)  # type: ignore[arg-type]
        if not poses.ok[0]:
            self.message = (
                f"out of reach there ({poses.position_error_m[0] * 100:.1f} cm off): try another way"
            )
            return False
        self.solved = dict(poses.joints[0])
        if dry_run:
            return True
        target = dict(poses.joints[0]) | {"gripper.pos": current.get("gripper.pos", 0.0)}
        status = self.client.move_joints({self.arm_id: target}, max_vel=self.max_vel, tol=MOVE_TOL_DEG)
        if status["state"] != "done":
            self.message = f"the move ended as {status['state']}: {status.get('reason', '')}"
            return False
        self.target = (np.asarray(position, dtype=np.float64), pitch, roll)
        return True

    def joint_margin_deg(self, joints: dict[str, float]) -> float:
        """How close (deg) the closest of the arm's joints at `joints` (LeRobot degrees) comes to its bound."""
        from lerobot.felix_jr.harness.reach import ArmIK

        ik = ArmIK(self.model, self.arm, {self.arm: joints}, None, 0.0, 0.0)
        x = ik.radians(joints)
        return float(np.degrees(np.minimum(x - ik.lower, ik.upper - x)).min())

    def go_to_start(self) -> bool:
        """Move to a start pose (the caller has the person approve it first); no frame taken.

        The first of `start_poses` the arm reaches with every joint `START_JOINT_MARGIN_DEG` from its bounds, so the
        gripper can still tilt and turn from it; else the reachable one with the most margin.
        """
        reachable = []
        for position, pitch, roll in self.start_poses():
            if self.move_to(position, pitch, roll, dry_run=True):
                margin = self.joint_margin_deg(self.solved)
                reachable.append((margin, (position, pitch, roll)))
                if margin >= START_JOINT_MARGIN_DEG:
                    break
        if not reachable:
            self.message = "no start pose in reach"
            return False
        _, (position, pitch, roll) = next(
            (r for r in reachable if r[0] >= START_JOINT_MARGIN_DEG), max(reachable, key=lambda r: r[0])
        )
        return self.move_to(position, pitch, roll)

    def start(self) -> bool:
        """Move to the start pose (`go_to_start`; nothing moves if the arm is there already) and show the camera."""
        ok = self.go_to_start()
        self.refresh()
        return ok

    def jog(self, action: str) -> None:
        """One key's small move from the pose last asked for."""
        if self.target is None:
            return
        position, pitch, roll = self.target
        # the arrow keys as seen from behind the robot: away from it (-x), toward it, to its left (-y), its right
        step = {"up": (-1, 0, 0), "down": (1, 0, 0), "left": (0, -1, 0), "right": (0, 1, 0)}
        if action in step:
            position = position + self.step_m * np.array(step[action], dtype=float)
        elif action in ("up_z", "down_z"):
            position = position + [0.0, 0.0, self.step_m if action == "up_z" else -self.step_m]
        elif action in ("pitch_up", "pitch_down"):
            pitch = float(
                np.clip(pitch + (PITCH_STEP_DEG if action == "pitch_up" else -PITCH_STEP_DEG), -90.0, 0.0)
            )
        elif action in ("roll_left", "roll_right"):
            roll += ROLL_STEP_DEG if action == "roll_left" else -ROLL_STEP_DEG
        if self.move_to(position, pitch, roll):
            self.message = f"tip at ({', '.join(f'{v * 100:.0f}' for v in position)}) cm, pitch {pitch:.0f}, roll {roll:.0f}"
        self.refresh()

    # ---- seeing -----------------------------------------------------------------------------------------------

    def refresh(self) -> None:
        """Take a new wrist frame, find the board in it, offer it to the lens's views (`LensViews`), show it."""
        try:
            self.frame = self.client.wrist_capture(self.arm)
        except ServerError as e:  # e.g. the camera dropped for a moment: the server reopens it
            self.message = f"no frame from the camera ({e}); retrying"
            self.detection = None
            if self.frame is not None:
                self.render()
            return
        height, width = self.frame.color.shape[:2]
        if self.record.size == (0, 0):
            self.record.size = (width, height)
        if self.views is None:
            self.views = LensViews(self.board, self.record.size)
        self.detection = self.board.detect(self.frame.color)
        self.sharpness = sharpness(
            self.frame.color, None if self.detection is None else self.detection.pixels
        )
        if self.detection is None and time.time() - self.last_frame_saved > LAST_FRAME_EVERY_S:
            cv2.imwrite(
                str(self.out_dir / LAST_FRAME_FILE), cv2.cvtColor(self.frame.color, cv2.COLOR_RGB2BGR)
            )
            self.last_frame_saved = time.time()
        if self.phase == "focus":
            self.best_sharpness = max(self.best_sharpness, self.sharpness)
            self.seen = "board found" if self.detection is not None else "board NOT found"
        elif self.phase == "intrinsics":
            kept, self.seen = self.views.offer(self.detection)
            if kept:
                self._saved_view()
        self.render()

    def _saved_view(self) -> None:
        """The view just kept: its picture saved, the record too."""
        assert self.frame is not None and self.views is not None
        index = len(self.views.views) - 1
        cv2.imwrite(
            str(self.out_dir / "images" / f"{self.arm}_intrinsics_{index:03d}.png"),
            cv2.cvtColor(self.frame.color, cv2.COLOR_RGB2BGR),
        )
        self.record.intrinsic_views = list(self.views.views)
        self.record.save(self.out_dir)
        self.message = f"view {index + 1} kept"

    def status(self) -> list[str]:
        """The lines drawn on the picture: where things stand, the key to press next, the last message."""
        title = f"{self.arm.upper()} {PHASE_TITLES[self.phase]}"
        if self.phase == "focus":
            verdict = "sharp" if self.sharpness >= SHARP_ENOUGH else "blurred"
            lines = [
                f"{title}: sharpness {self.sharpness:.0f} ({verdict}, best {self.best_sharpness:.0f})",
                "turn the lens until the number peaks | Enter: accept",
            ]
        else:
            n = len(self.record.intrinsic_views)
            missing = lens_coverage_missing(self.views.signatures if self.views else [])
            lines = [
                f"{title}: {n} views (of {MIN_INTRINSIC_VIEWS}) | {self.seen}",
                "c: fit the lens"
                if n >= MIN_INTRINSIC_VIEWS
                else f"move the board: {missing[0]}"
                if missing
                else "tilt, turn, raise the board; into the picture's corners",
            ]
        if self.message:
            lines.append(self.message)
        return lines

    def keys(self) -> str:
        """This phase's keys, in words (printed to the terminal as it starts)."""
        rest = f"x x: skip the {self.arm} camera ({self.after()}), q: quit"
        if self.phase == "focus":
            return f"{self.arm} FOCUS: hold the board ~20 cm from the camera, turn the lens; Enter or click: accept"
        return (
            f"{self.arm} LENS: hold the board in front of the camera and move it slowly, tilted every way, near and "
            f"far, into the corners of the picture (it may run off the edge); views are kept by themselves when it "
            f"is held still. space: keep this one, u: drop the last, c: fit the lens. {rest}"
        )

    def render(self) -> None:
        """Show the frame with the cells covered, the corners found and the status."""
        if self.phase != self.announced:
            self.announced = self.phase
            self.say(self.keys())
        if self.frame is None:
            return
        covered = self.views.covered if self.views is not None and self.phase == "intrinsics" else set()
        self.show(annotate(self.frame.color, self.status(), self.detection, covered))

    # ---- views and fits ---------------------------------------------------------------------------------------

    def take(self) -> None:
        """Keep this frame's board as a view whatever it is like (space), in the lens phase."""
        if self.phase != "intrinsics" or self.views is None:
            self.message = "views are taken in the lens step (Enter accepts the focus first)"
        elif self.frame is None or self.detection is None:
            self.message = "the board is not found here: no view taken"
        elif not self.views.enough(self.detection):
            self.message = f"too few corners ({len(self.detection)}): no view taken"
        else:
            self.views.keep(self.detection)
            self._saved_view()
        self.render()

    def undo(self) -> None:
        """Drop the last view (`u`)."""
        if self.views is not None and self.views.views:
            self.views.drop_last()
            self.record.intrinsic_views = list(self.views.views)
            self.record.save(self.out_dir)
            self.message = f"dropped the last view ({len(self.views.views)} left)"
        self.render()

    def after(self) -> str:
        """What comes after this arm's camera, in words."""
        return (
            f"next the {self.next_arm} wrist camera" if self.next_arm else "then the calibration is written"
        )

    def calibrate(self) -> None:
        """`c`: fit the lens; once it is settled and plausible, the arm is done."""
        if self.phase != "intrinsics":
            self.render()
            return
        views = self.record.intrinsic_views
        if len(views) < MIN_INTRINSIC_VIEWS:
            self.message = f"{MIN_INTRINSIC_VIEWS} views first"
            self.render()
            return
        focal, spread = lens_settled(views, self.board, self.record.size)
        if spread > LENS_SETTLED:
            missing = lens_coverage_missing(self.views.signatures if self.views else [])
            self.message = (
                f"not settled: the focal length ({focal:.0f} px) varies {spread * 100:.0f}% across the views (under "
                f"{LENS_SETTLED * 100:.0f}%): more views"
                + (f", the board {'; '.join(missing)}" if missing else ", tilted and nearer / farther")
            )
            self.render()
            return
        camera, errors = calibrate_intrinsics(views, self.board, self.record.size, self.record.lens)
        low, high = (f * self.record.size[0] for f in FOCAL_RANGE)
        if not (low <= camera.fx <= high and low <= camera.fy <= high):
            self.message = (
                f"the lens fit is implausible (fx {camera.fx:.0f} px): take more views, tilted and at different "
                "distances (u drops bad ones), then c"
            )
            self.render()
            return
        self.camera = camera
        self.model = with_lens(self.model, self.arm, camera)
        self.done = True
        self.message = (
            f"lens fitted: fx {camera.fx:.0f} px (halves agree within {spread * 100:.1f}%), RMS "
            f"{camera.rms_px:.2f} px (worst view {errors.max():.2f})"
        )
        self.say(f"the {self.arm} wrist camera's {self.message}")
        self.render()

    def accept_focus(self) -> None:
        """Enter / a click in the focus phase: keep the lens as it is and go on to the views."""
        if self.phase == "focus":
            self.record.focus_sharpness = self.sharpness
            self.record.save(self.out_dir)
            self.phase = "intrinsics"
            self.message = f"focus accepted at sharpness {self.sharpness:.0f}"
            self.refresh()
            return
        self.render()

    def handle(self, action: str) -> bool:
        """Do one action from `decode_key`. Returns False when the person quits."""
        if action == "quit":
            return False
        confirmed, self.confirming = self.confirming == action, None
        if action == "accept" or (action == "calibrate" and self.phase == "focus"):
            self.accept_focus()
        elif action == "take":
            self.take()
        elif action == "undo":
            self.undo()
        elif action == "calibrate":
            self.calibrate()
        elif action == "skip" and confirmed:
            self.skipped = self.done = True
        elif action == "skip":
            self.confirming = "skip"
            self.message = f"x again skips the {self.arm} camera ({self.after()}); any other key: no"
            self.render()
        else:
            self.jog(action)
        return True


def main() -> None:
    """Command line entry point; see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("endpoint", nargs="?", default=f"tcp://localhost:{DEFAULT_PORT}")
    parser.add_argument(
        "--start-from", type=Path, required=True, help="the calibration.urdf to put the lenses in"
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--arms", default="left,right", help="which wrist cameras, in order (left,right)")
    parser.add_argument(
        "--board", default=PRINTED_BOARD, help=f"the board (boards.parse_board; {PRINTED_BOARD})"
    )
    parser.add_argument(
        "--lens", choices=WRIST_CAMERA_MODELS, default="rational", help="the lens model (rational)"
    )
    parser.add_argument(
        "--no-focus", action="store_true", help="skip the focus step (the lenses are focused)"
    )
    parser.add_argument(
        "--resume", action="store_true", help="keep the views saved in --out-dir and add to them"
    )
    parser.add_argument("--step-cm", type=float, default=2.0, help="each key's move, cm (2)")
    parser.add_argument("--max-vel", type=float, default=20.0, help="arm speed limit, deg/s (20)")
    parser.add_argument("--refit", action="store_true", help="no robot: fit --out-dir's saved views again")
    args = parser.parse_args()
    if (args.out_dir / CALIBRATION_FILE).resolve() == args.start_from.resolve():
        parser.error("--out-dir would overwrite --start-from: pick another folder")
    if args.refit:
        path = refit(args.start_from, args.out_dir)
        if path is None:
            parser.error(f"no saved views to fit in {args.out_dir}")
        print(f"Wrote {path}\n{(args.out_dir / REPORT_FILE).read_text()}")
        return
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    if not arms or set(arms) - set(ARMS):
        parser.error(f"--arms must be some of {','.join(ARMS)}")
    board = parse_board(args.board)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = XleRobotModel.load(args.start_from)
    window = "wrist camera (keys: see the status line)"

    def show(image: np.ndarray) -> None:
        cv2.imshow(window, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    clicks: list[int] = []

    def on_mouse(event: int, *_: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append(1)

    records: list[ArmRecord] = []
    with FelixClient(args.endpoint, timeout_s=20.0) as client, client.session():
        model.check_robot(client)
        missing = set(arms) - set(client.hello().get("wrist_cameras", []))
        if missing:
            raise SystemExit(
                f"the server has no {', '.join(sorted(missing))} wrist camera: start it with --<arm>-wrist-camera"
            )
        try:
            # every arm goes to its start pose first, the ones not calibrated too: out of the way of the board
            sessions = {
                arm: WristCameraSession(
                    client,
                    model,
                    arm,
                    board,
                    args.out_dir,
                    show,
                    args.lens,
                    args.step_cm / 100,
                    args.max_vel,
                    focus=not args.no_focus,
                    resume=args.resume,
                )
                for arm in ARMS
            }
            answer = input(
                "both arms go to their start poses (grippers tilted down, ~18 cm over the table, in front of each "
                "arm): go? [y/N] "
            )
            if answer.strip().lower() == "y":
                for arm, session in sessions.items():
                    if not session.go_to_start():
                        raise SystemExit(f"the {arm} arm did not reach its start pose: {session.message}")
            else:
                arms = []
            for index, arm in enumerate(arms):
                session = sessions[arm]
                session.next_arm = arms[index + 1] if index + 1 < len(arms) else None
                print(
                    f"the {arm} wrist camera"
                    + (f" (then the {session.next_arm})" if session.next_arm else "")
                )
                session.start()
                cv2.setMouseCallback(window, on_mouse)
                keep_going = True
                while keep_going and not session.done:
                    key = cv2.waitKeyEx(30)
                    action = "accept" if clicks and session.phase == "focus" else decode_key(key, KEY_ACTIONS)
                    clicks.clear()
                    if action is None:
                        session.refresh()
                        continue
                    keep_going = session.handle(action)
                if session.skipped:
                    print(f"the {arm} wrist camera: skipped")
                else:
                    records.append(session.record)
                    print(f"the {arm} wrist camera: done")
                if not keep_going:
                    break
        finally:
            cv2.destroyAllWindows()
    path = write(XleRobotModel.load(args.start_from), records, args.out_dir)
    if path is None:
        print("nothing calibrated: nothing written (the views are kept)")
    else:
        print(f"Wrote {path}\n{(args.out_dir / REPORT_FILE).read_text()}")


if __name__ == "__main__":
    main()
