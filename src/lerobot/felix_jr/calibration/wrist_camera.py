"""Calibrate the wrist cameras: their lens (intrinsics) and their pose on the gripper (extrinsics), from a checkerboard.

  uv run python -m lerobot.felix_jr.calibration.wrist_camera tcp://<pi>:5555 --start-from <calibration.urdf>
      --out-dir <folder> [--arms left,right] [--pattern 9x6] [--square-mm 25] [--lens rational|fisheye]
      [--only intrinsics|extrinsics] [--no-focus] [--step-cm 2] [--max-vel 20]
  uv run python -m lerobot.felix_jr.calibration.wrist_camera --refit --start-from <calibration.urdf> --out-dir <folder>

The server must run with the wrist cameras (`server.server --left-wrist-camera ... --right-wrist-camera ...`). Without
`--pattern` and `--square-mm` the terminal asks for the board: its inner corners (columns x rows: the corners where
four squares meet, one fewer each way than the squares: 9 x 7 squares are 8 x 6) and the size of one square in mm as
printed (measure it: printers scale). A wrong count can still be "found", with a row on the paper's edge: the status
line warns when the board looks like another count (`pattern_hint`).

After one approval in the terminal, both arms go to their start poses (the arms not calibrated too: out of the way of
the board): the gripper tilted down as steeply as it reaches (70 deg on this robot: straight down is out of reach),
about 18 cm above the table and 22 cm in front of the arm. They stay there until the end, when the robot goes to
rest. Then for each arm in `--arms` in turn, a window shows its wrist
camera; the keys move the arm (each a small move, the gripper's tip as the reference):

  arrow keys    move it over the table (root frame x and y) by --step-cm
  w / s         up / down
  [ / ]         tilt the gripper (pitch) by 10 deg
  , / .         turn it about its axis (roll) by 15 deg
  n             camera pose: move the arm to the next suggested view (`POSE_VIEWS`) and take it
  space         take this view (the board must be found: its corners are drawn)
  u             drop the last view
  c             calibrate with the views so far, and go on
  q or Esc      stop (what is calibrated so far is written)

0. **Focus.** First the window streams the camera with a sharpness score (`sharpness`: of the board where it is
   found, else of the middle of the picture) and the best so far. Hold the board about 20 cm in front of the camera
   (where the objects will be when grasping) and turn the lens until the score peaks: the board's corners are drawn
   once it is sharp enough to be found. Enter, a click in the window or `c` accepts (`--no-focus` skips it).
1. **Intrinsics.** Hold the board under the camera and move it: tilted every way, near and far, into the corners of
   the picture (the arm may move too). A view is taken by itself when the board is found in a pose unlike the ones
   kept (or with space); the corners of the picture covered so far are drawn. `c` fits the lens (OpenCV's rational
   model, or `--lens fisheye` for a lens wider than about 120 deg) once there are `MIN_INTRINSIC_VIEWS`.
2. **Extrinsics.** Put the board flat on the table, where it stays, and press `c`. Then `n` again and again: the arm
   moves to the next suggested view around the start pose (`POSE_VIEWS`: tilted, turned, shifted; views from one
   direction only, however many, leave the camera's pose undetermined) and takes it if the board is in view; or move
   the arm by keys and take a view with space, as many as you like. From `MIN_EXTRINSIC_VIEWS` the camera's pose in
   `<arm>_gripper_link` and the board's on the table are fitted together after every new view (`fit_extrinsics`:
   `cv2.calibrateRobotWorldHandEye`, then refined on the corners' pixels), with how well the views pin it down (each
   view left out in turn), and the model's fingertips are drawn on the live picture from it. `c` accepts it once
   settled (`ExtrinsicsFit.settled`: turned `MIN_SPREAD_DEG` about two axes, known to `POSE_SETTLED_M` and
   `POSE_SETTLED_DEG`); until then the status line says what would help.

The fit uses the arm's measured joints and the model (`--start-from`), so the arm's own error (its sag, its
calibration) averages into the camera's pose; many views across the workspace keep it small. The output,
`--out-dir/calibration.urdf`, is `--start-from` with the wrist cameras (`XleRobotModel.with_wrist_camera`), and
`wrist_cameras.md` reports the errors. The views (`<arm>_views.json`, `images/`) are kept: `--refit` fits them again
with no robot. The script runs inside a client session, so on Ctrl+C or a crash the robot goes to rest.
"""

import argparse
import json
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

# OpenCL off for the whole process (not only this thread): on macOS its kernels fail with alarming but harmless errors
os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from lerobot.felix_jr.calibration.live_view import decode_key
from lerobot.felix_jr.calibration.xle_robot_model import (
    HEAD_JOINTS,
    WRIST_CAMERA_MODELS,
    WristCamera,
    XleRobotModel,
    arm_record,
    pinch_frame,
    wrist_camera_link,
)
from lerobot.felix_jr.server.client import FelixClient, WristCapture
from lerobot.felix_jr.server.protocol import DEFAULT_PORT, ServerError
from lerobot.utils.import_utils import _scipy_available, require_package

if TYPE_CHECKING or _scipy_available:
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

ARMS = ("left", "right")
CALIBRATION_FILE = "calibration.urdf"
REPORT_FILE = "wrist_cameras.md"
MIN_INTRINSIC_VIEWS = 12
# the lens views should include the board tilted toward each side (its far side shrunk by this much) and
# at different distances (`LENS_SCALE_RATIO`), shown as guidance; what decides is `lens_settled`. Flat views at one
# distance leave the focal length
# trading off with the distance (wrist_cams5, 2026-11-07: 47 flat views, fx anywhere from 314 to 3733)
LENS_TILT = 0.07  # ~20-25 deg for a 25 x 17 cm board at 25-35 cm (flat views read under 0.05)
LENS_SCALE_RATIO = 1.3  # the nearest view's board at least this much bigger than the farthest's (flat: 1.09)
LENS_SETTLED = (
    0.03  # the lens is accepted when its focal length, fitted on random halves of the views, agrees this well
)
LENS_SPLITS = 12  # how many random halves (`lens_settled`)
FOCAL_RANGE = (0.2, 3.0)  # a focal length (px) outside this range times the picture's width is refused
MIN_EXTRINSIC_VIEWS = 8
# the views must turn this far about two axes: about one only, the camera's offset along it is free
MIN_SPREAD_DEG = 15.0
# the camera's pose is accepted when known this well: the spread of the fits with each view left out (a jackknife).
# The arm's own error (its joints, its calibration) dominates, and the fit's pixel residuals do not show it: simulated
# with 0.3 deg of joint noise, the jackknife matched the true error (8 views: 18 vs 19 mm; 13: 13 vs 9) where the fit's
# covariance said 3 mm. More views shrink it: keep adding them (fit as you go) until it settles
POSE_SETTLED_M = 0.005
POSE_SETTLED_DEG = 1.0
# the start pose keeps every joint this far from its bounds, so the keys and the suggested views can tilt and turn the
# gripper from it (calib_full3, 2026-11-07: the steepest start put the wrist flex at its 90 deg bound: no tilt left)
START_JOINT_MARGIN_DEG = 15.0
# the start pose: the gripper's tip this far in front of the arm's base, the first of these heights above the base
# (about 6 cm above the table on this robot) the arm reaches with the gripper tilted down the steepest of these
# pitches (straight down is out of reach: the wrist flex is at its limit)
START_AHEAD_M = 0.22
START_HEIGHTS_M = (0.12, 0.08, 0.04)
START_PITCHES_DEG = (-90.0, -80.0, -70.0, -60.0, -50.0, -40.0)
# the suggested views for the camera's pose (`n`), from the start pose: (dx, dy, dz) m in the root frame (+x back
# toward the robot), then the pitch's change (deg, + less steep; the start keeps the joints' margin for both ways) and
# the roll (deg): turns about two axes (the pitch, the roll), the board under the start pose kept in view
POSE_VIEWS = (
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (0.02, 0.0, 0.0, 15.0, 0.0),
    (-0.02, 0.0, 0.01, -15.0, 0.0),
    (0.0, 0.03, 0.0, 0.0, 35.0),
    (0.0, -0.03, 0.0, 0.0, -35.0),
    (0.02, 0.02, 0.0, 10.0, 50.0),
    (-0.02, -0.02, 0.01, -10.0, -50.0),
    (0.02, -0.02, 0.0, 15.0, -25.0),
    (-0.02, 0.02, 0.01, -15.0, 25.0),
    (0.0, 0.0, 0.03, 0.0, 70.0),
    (0.0, 0.0, 0.03, 0.0, -70.0),
    (0.03, 0.0, 0.01, 20.0, 0.0),
    (-0.02, 0.0, 0.02, -20.0, 0.0),
    (0.01, 0.03, 0.0, 10.0, -60.0),
    (-0.01, -0.03, 0.0, -10.0, 60.0),
)
PITCH_STEP_DEG = 10.0
MOVE_TOL_DEG = (
    3.0  # a move ends once every joint is this close (the views use the measured joints, not the targets)
)
ROLL_STEP_DEG = 15.0
# a found board counts as a new view when, against every view kept, it moved or resized this much (of the picture's
# size), or its apparent tilt changed this much, or it turned this far in the picture
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
    ord("n"): "next",
    ord("q"): "quit",
    27: "quit",
    13: "accept",  # Enter
    10: "accept",
}
FINGERTIP_COLORS = ((0, 255, 255), (255, 0, 255))  # RGB: the pinch point (fixed tip), the moving tip
LAST_FRAME_FILE = (
    "last_frame.png"  # the latest frame the board was not found in (to see why), every few seconds
)
LAST_FRAME_EVERY_S = 2.0
SHARP_ENOUGH = (
    100.0  # a sharpness (`sharpness`) above this reads as in focus; the out-of-focus lenses read ~2
)


# ---- the board ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Board:
    """A checkerboard: `cols` x `rows` inner corners, squares `square_m` wide."""

    cols: int
    rows: int
    square_m: float

    @property
    def symmetric(self) -> bool:
        """Whether the board looks the same turned around: both counts even, or both odd.

        Then which end its corners start from cannot be told from the picture (`find_board`), only from the arm's
        motion (`orient_views`).
        """
        return self.cols % 2 == self.rows % 2

    def points(self) -> np.ndarray:
        """The inner corners in the board's frame (N, 3), row by row, z = 0: OpenCV's order."""
        grid = np.mgrid[0 : self.cols, 0 : self.rows].T.reshape(-1, 2)
        return np.c_[grid * self.square_m, np.zeros(len(grid))].astype(np.float64)


def parse_pattern(text: str) -> tuple[int, int]:
    """`"9x6"` -> (9, 6): inner corners, columns x rows (one fewer each way than the squares)."""
    cols, rows = (int(v) for v in text.lower().replace(" ", "").split("x"))
    if cols < 3 or rows < 3:
        raise ValueError("a board needs at least 3 x 3 inner corners")
    return cols, rows


def find_board(image: np.ndarray, board: Board) -> np.ndarray | None:
    """The board's inner corners in an RGB image (N, 2), or None if it is not found whole.

    In OpenCV's order (row by row), started from the end whose outer square, diagonally past the first corner, is
    dark: the detector may start from either end, and a fixed board must keep one frame across the views (the
    extrinsics). With one odd and one even count the two ends' outer squares differ in color; a symmetric board
    (`Board.symmetric`) is left as found, and `orient_views` tells its ends apart from the arm's motion.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    found, corners = cv2.findChessboardCornersSB(
        gray, (board.cols, board.rows), flags=cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if not found:
        return None
    corners = corners.reshape(-1, 2).astype(np.float64)
    if board.symmetric:  # both ends alike: left as found (`calibrate_extrinsics` sorts it out)
        return corners
    first, last = _outer_brightness(gray, corners, board, 0), _outer_brightness(gray, corners, board, -1)
    if first is not None and last is not None:
        flip = first > last
    elif first is not None:
        flip = first > 127
    elif last is not None:
        flip = last < 127
    else:
        flip = False
    return corners[::-1].copy() if flip else corners


def pattern_hint(image: np.ndarray, board: Board) -> str | None:
    """A warning when the board in `image` looks like another count of inner corners than `board`'s, else None.

    Asked for one row or column too many, the detector can still "find" the board, with the extra row on the paper's
    edge (wrist_cams2, 2026-11-07: an 8 x 6 board found as 9 x 6, every view 30-40 px off): so a board also found one
    row or column smaller is suspect, and a board not found at all is tried one larger and smaller.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    def found(cols: int, rows: int) -> bool:
        return bool(cv2.findChessboardCornersSB(gray, (cols, rows), flags=cv2.CALIB_CB_NORMALIZE_IMAGE)[0])

    smaller = [(board.cols - 1, board.rows), (board.cols, board.rows - 1)]
    larger = [(board.cols + 1, board.rows), (board.cols, board.rows + 1)]
    candidates = smaller if found(board.cols, board.rows) else smaller + larger
    for cols, rows in candidates:
        if min(cols, rows) >= 3 and found(cols, rows):
            return (
                f"the board looks like {cols} x {rows} inner corners, not {board.cols} x {board.rows} "
                f"(count the squares, less one each way): restart with --pattern {cols}x{rows}"
            )
    return None


def _outer_brightness(gray: np.ndarray, corners: np.ndarray, board: Board, end: int) -> float | None:
    """The brightness of the square diagonally outside the grid at its first (`end` 0) or last (-1) corner."""
    grid = corners.reshape(board.rows, board.cols, 2)
    corner, inner = (grid[0, 0], grid[1, 1]) if end == 0 else (grid[-1, -1], grid[-2, -2])
    u, v = corner + 0.5 * (corner - inner)
    height, width = gray.shape
    if not (2 <= u < width - 2 and 2 <= v < height - 2):
        return None
    return float(gray[int(v) - 1 : int(v) + 2, int(u) - 1 : int(u) + 2].mean())


def sharpness(image: np.ndarray, corners: np.ndarray | None = None) -> float:
    """How sharp an RGB image is: the variance of its Laplacian (higher is sharper).

    Over the board (`corners`' hull), else the middle half of the picture. It depends on the scene, so compare it
    while turning the lens.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    if corners is not None:
        mask = np.zeros(gray.shape, np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(corners.astype(np.int32)), 1)
        return float(laplacian[mask > 0].var())
    height, width = gray.shape
    return float(laplacian[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4].var())


def view_signature(corners: np.ndarray, size: tuple[int, int], board: Board) -> np.ndarray:
    """Where the board is in the picture, how big, how tilted and turned.

    (center x, center y, scale, tilt x, tilt y, turn): positions and scale as fractions of the picture; the tilts are
    how much the board's far rows / columns shrink against its near ones (0 for a board square to the camera); the turn
    is the direction of its first row in the picture (deg; the corners' order fixes it, `find_board`).
    """
    width, height = size
    grid = corners.reshape(board.rows, board.cols, 2)
    center = corners.mean(axis=0) / [width, height]
    scale = float(np.sqrt(cv2.contourArea(cv2.convexHull(corners.astype(np.float32))) / (width * height)))
    top, bottom = np.linalg.norm(grid[0, -1] - grid[0, 0]), np.linalg.norm(grid[-1, -1] - grid[-1, 0])
    left, right = np.linalg.norm(grid[-1, 0] - grid[0, 0]), np.linalg.norm(grid[-1, -1] - grid[0, -1])
    tilt_x = (top - bottom) / max(top + bottom, 1e-9)
    tilt_y = (left - right) / max(left + right, 1e-9)
    row = grid[0, -1] - grid[0, 0]
    turn = float(np.degrees(np.arctan2(row[1], row[0])))
    return np.array([center[0], center[1], scale, tilt_x, tilt_y, turn])


def is_new_view(signature: np.ndarray, kept: Sequence[np.ndarray]) -> bool:
    """Whether a view differs from every kept one.

    It does when it moved or resized by `NEW_VIEW_PX`, tilted by `NEW_VIEW_TILT` or turned by `NEW_VIEW_TURN_DEG`.
    """
    for other in kept:
        moved = np.linalg.norm(signature[:3] - other[:3])
        tilted = np.abs(signature[3:5] - other[3:5]).max()
        turned = abs((signature[5] - other[5] + 180.0) % 360.0 - 180.0)
        if moved < NEW_VIEW_PX and tilted < NEW_VIEW_TILT and turned < NEW_VIEW_TURN_DEG:
            return False
    return True


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


def lens_settled(
    views: Sequence[np.ndarray], board: Board, size: tuple[int, int], seed: int = 0
) -> tuple[float, float]:
    """How well the views pin the focal length down: its mean and its spread (relative) over fits on random halves.

    Views that do not constrain it (wrist_cams5: flat, at one distance) give halves that disagree; good views agree
    within a few percent (wrist_cams6, held up in a cramped workspace: 342 +- 2.4%). Plain 5-coefficient fits, quick.
    """
    rng = np.random.default_rng(seed)
    objects = board.points().astype(np.float32).reshape(-1, 1, 3)
    images = [np.asarray(v, np.float32).reshape(-1, 1, 2) for v in views]
    focal = []
    for _ in range(LENS_SPLITS):
        half = rng.choice(len(images), size=max(len(images) // 2, 3), replace=False)
        _, matrix, *_ = cv2.calibrateCamera(
            [objects] * len(half), [images[i] for i in half], size, None, None
        )
        focal.append(0.5 * (matrix[0, 0] + matrix[1, 1]))
    return float(np.mean(focal)), float(np.std(focal) / np.mean(focal))


def calibrate_intrinsics(
    views: Sequence[np.ndarray], board: Board, size: tuple[int, int], lens: str = "rational"
) -> tuple[WristCamera, np.ndarray]:
    """The lens from views of the board (each its corners, (N, 2)): the camera and each view's RMS error (px)."""
    if lens not in WRIST_CAMERA_MODELS:
        raise ValueError(f"unknown lens model {lens!r}: one of {WRIST_CAMERA_MODELS}")
    if len(views) < 3:
        raise ValueError(f"{len(views)} views: too few to fit a lens")
    objects = [board.points().reshape(-1, 1, 3) for _ in views]
    images = [v.reshape(-1, 1, 2).astype(np.float64) for v in views]
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
        pose = _pose(np.ravel(rvec), np.ravel(tvec))
        predicted = camera.project(_apply(pose, board.points()))
        errors.append(float(np.sqrt(np.mean(np.sum((predicted - view) ** 2, axis=1)))))
    return camera, np.array(errors)


def board_pose(camera: WristCamera, corners: np.ndarray, board: Board) -> np.ndarray:
    """camera_from_board (4x4) from one view's corners, through the lens."""
    if camera.model == "fisheye":
        normalized = cv2.fisheye.undistortPoints(
            corners.reshape(-1, 1, 2), camera.matrix(), np.asarray(camera.coeffs)
        )
        ok, rvec, tvec = cv2.solvePnP(board.points(), normalized, np.eye(3), None)
    else:
        ok, rvec, tvec = cv2.solvePnP(board.points(), corners, camera.matrix(), np.asarray(camera.coeffs))
    if not ok:
        raise ValueError("the board's pose could not be solved from this view")
    return _pose(np.ravel(rvec), np.ravel(tvec))


@dataclass
class ArmView:
    """One extrinsics view: the arm's measured joints (LeRobot degrees) and the board's corners in the wrist image."""

    arm_pose: dict[str, float]
    corners: list[list[float]]
    image: str | None = None
    timestamp: float = 0.0


def rotation_spread(rotations: Sequence[np.ndarray]) -> np.ndarray:
    """How far the views' rotations spread about their three main axes (deg, largest first).

    The spread of the rotation vectors (each relative to the first view) along their principal directions: a
    hand-eye fit needs turns about at least two axes.
    """
    if len(rotations) < 2:
        return np.zeros(3)
    first = Rotation.from_matrix(rotations[0])
    vectors = np.array([(Rotation.from_matrix(r) * first.inv()).as_rotvec() for r in rotations])
    centered = vectors - vectors.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    return np.degrees(2.0 * singular / np.sqrt(len(rotations)))


@dataclass(frozen=True)
class ExtrinsicsFit:
    """A fit of the camera's pose on the gripper (`fit_extrinsics`), with how well the views pin it down.

    `position_sd_m` and `rotation_sd_deg` are how far it moves along / about `<arm>_gripper_link`'s x, y, z with each
    view left out in turn (a jackknife: it carries the arm's own error, which the pixels' residuals do not), NaN with
    too few views. `spread_deg` is how far the views turn about their main axes (`rotation_spread`).
    """

    gripper_from_camera: np.ndarray  # (4, 4)
    root_from_board: np.ndarray  # (4, 4)
    errors_px: np.ndarray  # each view's RMS error
    position_sd_m: np.ndarray  # (3,)
    rotation_sd_deg: np.ndarray  # (3,)
    spread_deg: np.ndarray  # (3,)

    @property
    def settled(self) -> bool:
        """Whether the pose is known well enough: turned about two axes, the jackknife within `POSE_SETTLED_*`."""
        return bool(
            self.spread_deg[1] >= MIN_SPREAD_DEG
            and np.nanmax(self.position_sd_m) <= POSE_SETTLED_M
            and np.nanmax(self.rotation_sd_deg) <= POSE_SETTLED_DEG
        )

    def needs(self) -> str:
        """What would pin the pose down, in words ("" when settled).

        Views turned about one axis only leave the camera's offset along it free: tilting the gripper (`[ ]`) and
        turning it about its own axis (`, .`) are the two turns (moving it about only pans it, much the same turn).
        Past that, more views of any new pose shrink the error.
        """
        if self.settled:
            return ""
        if self.spread_deg[1] < MIN_SPREAD_DEG:
            return "tilt ([ ]) and turn (, .) the gripper between views: they all turn about one axis"
        return "more views, from new poses (n, or keys and space)"

    def describe(self) -> str:
        """The pose and how well it is known, in a line."""
        t = self.gripper_from_camera[:3, 3] * 1000
        sd = self.position_sd_m * 1000
        return (
            f"camera at ({t[0]:.0f}, {t[1]:.0f}, {t[2]:.0f}) mm +- ({sd[0]:.1f}, {sd[1]:.1f}, {sd[2]:.1f}), turn +- "
            f"{np.nanmax(self.rotation_sd_deg):.2f} deg, RMS {float(np.sqrt(np.mean(self.errors_px**2))):.2f} px"
        )


def calibrate_extrinsics(
    model: XleRobotModel, arm: str, camera: WristCamera, views: Sequence[ArmView], board: Board
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`fit_extrinsics`' gripper_from_camera, root_from_board and each view's RMS error (px)."""
    fit = fit_extrinsics(model, arm, camera, views, board)
    return fit.gripper_from_camera, fit.root_from_board, fit.errors_px


def fit_extrinsics(
    model: XleRobotModel,
    arm: str,
    camera: WristCamera,
    views: Sequence[ArmView],
    board: Board,
    jackknife: bool = True,
) -> ExtrinsicsFit:
    """The camera's pose on the gripper and the board's on the table, from views of the board at several poses.

    The pose is gripper_from_camera (4x4, in `<arm>_gripper_link`). Started from `cv2.calibrateRobotWorldHandEye` on
    the board's pose in each view (`board_pose`) and the gripper's (the model at the measured joints), then refined on
    all the corners' pixels (a Huber loss: a bad view does not pull the rest).
    """
    require_package("scipy", extra="felix_jr")
    if len(views) < 3:
        raise ValueError(f"{len(views)} views: too few for the extrinsics")
    link = wrist_camera_link(arm)
    root_from_gripper = [model.frame_pose(link, model.joint_vector({arm: v.arm_pose})) for v in views]
    observed = [np.asarray(v.corners) for v in views]
    if board.symmetric:
        observed = orient_views(camera, observed, root_from_gripper, board)
    camera_from_board = [board_pose(camera, corners, board) for corners in observed]
    gripper_from_root = [np.linalg.inv(p) for p in root_from_gripper]
    try:
        r_base2world, t_base2world, r_gripper2cam, t_gripper2cam = cv2.calibrateRobotWorldHandEye(
            [p[:3, :3] for p in camera_from_board],
            [p[:3, 3] for p in camera_from_board],
            [p[:3, :3] for p in gripper_from_root],
            [p[:3, 3] for p in gripper_from_root],
        )
        camera_from_gripper = np.eye(4)
        camera_from_gripper[:3, :3], camera_from_gripper[:3, 3] = r_gripper2cam, np.ravel(t_gripper2cam)
        board_from_root = np.eye(4)
        board_from_root[:3, :3], board_from_root[:3, 3] = r_base2world, np.ravel(t_base2world)
        start_gc, start_rb = np.linalg.inv(camera_from_gripper), np.linalg.inv(board_from_root)
    # views turned about one axis: OpenCV's solver fails; start from the camera at the gripper
    except cv2.error:
        start_gc = np.eye(4)
        start_rb = root_from_gripper[0] @ camera_from_board[0]

    points = board.points()

    def unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return _pose(x[:3], x[3:6]), _pose(x[6:9], x[9:12])

    def per_view(x: np.ndarray) -> list[np.ndarray]:
        gripper_from_camera, root_from_board = unpack(x)
        out = []
        for root_gripper, seen in zip(root_from_gripper, observed, strict=True):
            camera_from_root = np.linalg.inv(root_gripper @ gripper_from_camera)
            predicted = camera.project(_apply(camera_from_root @ root_from_board, points))
            out.append(np.nan_to_num(predicted - seen, nan=1e3))
        return out

    def residuals(x: np.ndarray) -> np.ndarray:
        return np.concatenate([r.ravel() for r in per_view(x)])

    x0 = np.r_[_rotvec(start_gc), start_gc[:3, 3], _rotvec(start_rb), start_rb[:3, 3]]
    result = least_squares(residuals, x0, loss="huber", f_scale=2.0)
    gripper_from_camera, root_from_board = unpack(result.x)
    errors = np.array([float(np.sqrt(np.mean(np.sum(r**2, axis=1)))) for r in per_view(result.x)])
    spread = rotation_spread([p[:3, :3] for p in root_from_gripper])
    position_sd, rotation_sd = np.full(3, np.nan), np.full(3, np.nan)
    if jackknife and len(views) >= 4:
        left_out = [
            fit_extrinsics(model, arm, camera, [*views[:i], *views[i + 1 :]], board, jackknife=False)
            for i in range(len(views))
        ]
        n = len(left_out)
        positions = np.array([f.gripper_from_camera[:3, 3] for f in left_out])
        turns = np.array(
            [_rotvec(f.gripper_from_camera[:3, :3] @ gripper_from_camera[:3, :3].T) for f in left_out]
        )
        position_sd = np.sqrt((n - 1) / n * ((positions - positions.mean(axis=0)) ** 2).sum(axis=0))
        rotation_sd = np.degrees(np.sqrt((n - 1) / n * ((turns - turns.mean(axis=0)) ** 2).sum(axis=0)))
    return ExtrinsicsFit(gripper_from_camera, root_from_board, errors, position_sd, rotation_sd, spread)


def orient_views(
    camera: WristCamera, corners: Sequence[np.ndarray], root_from_gripper: Sequence[np.ndarray], board: Board
) -> list[np.ndarray]:
    """Each view's corners started from the same end of a symmetric board as the first view's.

    The camera turns between two views as the gripper does (hand-eye: the same angle, about axes the camera's pose
    relates), so of a view's two readings, as found and reversed, the right one turns from the first view by the
    gripper's angle; the other is off by about 180 deg.
    """

    def angle(rotation: np.ndarray) -> float:
        return float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))))

    first = board_pose(camera, corners[0], board)
    out = [np.asarray(corners[0])]
    for view, root_gripper in zip(corners[1:], root_from_gripper[1:], strict=True):
        arm_turn = angle(root_from_gripper[0][:3, :3].T @ root_gripper[:3, :3])
        readings = [np.asarray(view), np.asarray(view)[::-1].copy()]
        errors = [
            abs(angle(first[:3, :3] @ board_pose(camera, reading, board)[:3, :3].T) - arm_turn)
            for reading in readings
        ]
        out.append(readings[int(np.argmin(errors))])
    return out


def _pose(rotvec: np.ndarray, translation: np.ndarray) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = cv2.Rodrigues(np.asarray(rotvec, dtype=np.float64))[0]
    pose[:3, 3] = translation
    return pose


def _rotvec(pose: np.ndarray) -> np.ndarray:
    return cv2.Rodrigues(pose[:3, :3])[0].ravel()


def _apply(pose: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ pose[:3, :3].T + pose[:3, 3]


def fingertips_in_wrist_image(
    model: XleRobotModel, arm: str, camera: WristCamera, arm_pose: dict[str, float]
) -> np.ndarray:
    """Where the model puts the two fingertips (the pinch point, the moving tip) in the wrist image (2, 2) px."""
    q = model.joint_vector({arm: arm_pose})
    opening = (
        max(model.opening_deg(arm, float(arm_pose["gripper.pos"])), 0.0) if arm in model.gripper_maps else 0.0
    )
    tips = np.array([model.frame_pose(pinch_frame(arm), q)[:3, 3], model.moving_tip(arm, q, opening)])
    if camera.gripper_from_camera is None:
        raise ValueError("the camera's pose on the gripper is not known yet")
    root_from_camera = model.frame_pose(wrist_camera_link(arm), q) @ np.array(camera.gripper_from_camera)
    return camera.project(_apply(np.linalg.inv(root_from_camera), tips))


# ---- saving -------------------------------------------------------------------------------------------------


@dataclass
class ArmRecord:
    """What a session collected for one arm (`<arm>_views.json`): the board, the views, the arm's calibration."""

    arm: str
    board: Board
    size: tuple[int, int]
    lens: str
    arm_calibration: dict[str, Any]
    intrinsic_views: list[list[list[float]]] = field(default_factory=list)
    extrinsic_views: list[ArmView] = field(default_factory=list)
    focus_sharpness: float | None = None  # the sharpness when the focus was accepted

    def save(self, directory: Path) -> None:
        """Write `<arm>_views.json` in `directory`."""
        data = asdict(self) | {"board": asdict(self.board), "size": list(self.size)}
        (directory / f"{self.arm}_views.json").write_text(json.dumps(data) + "\n")

    @classmethod
    def load(cls, path: Path) -> "ArmRecord":
        """Read a `<arm>_views.json`."""
        data = json.loads(path.read_text())
        return cls(
            arm=data["arm"],
            board=Board(**data["board"]),
            size=tuple(data["size"]),
            lens=data["lens"],
            arm_calibration=data["arm_calibration"],
            intrinsic_views=data["intrinsic_views"],
            extrinsic_views=[ArmView(**v) for v in data["extrinsic_views"]],
            focus_sharpness=data.get("focus_sharpness"),
        )


def fit_record(model: XleRobotModel, record: ArmRecord) -> tuple[WristCamera | None, list[str]]:
    """Fit what `record` has views for: the lens, then the camera's pose. Returns the camera and the report's lines.

    With no views for the lens (`--only extrinsics`), the lens is `model`'s.
    """
    lines = [f"## {record.arm} wrist camera", ""]
    camera: WristCamera | None
    if len(record.intrinsic_views) >= 3:
        views = [np.asarray(v) for v in record.intrinsic_views]
        camera, errors = calibrate_intrinsics(views, record.board, record.size, record.lens)
        lens_line = f"(worst {errors.max():.2f} px)"
    else:
        camera, lens_line = model.wrist_cameras.get(record.arm), "(the calibration's, not refitted)"
    if camera is None:
        return None, [*lines, "Too few views for the lens: not calibrated.", ""]
    if record.focus_sharpness is not None:
        lines.append(
            f"- focus accepted at sharpness {record.focus_sharpness:.0f} (in focus above {SHARP_ENOUGH:.0f})"
        )
    lines += [
        f"- lens: {camera.model}, {camera.width} x {camera.height}, fx {camera.fx:.1f} fy {camera.fy:.1f}, "
        f"center ({camera.ppx:.1f}, {camera.ppy:.1f}); RMS {camera.rms_px or 0:.2f} px over {camera.n_views} views "
        f"{lens_line}",
    ]
    if len(record.extrinsic_views) >= 3:
        fit = fit_extrinsics(model, record.arm, camera, record.extrinsic_views, record.board)
        gripper_from_camera, root_from_board, ext_errors = (
            fit.gripper_from_camera,
            fit.root_from_board,
            fit.errors_px,
        )
        camera = replace(
            camera,
            gripper_from_camera=tuple(tuple(float(v) for v in row) for row in gripper_from_camera),
            extrinsics_rms_px=float(np.sqrt(np.mean(ext_errors**2))),
            extrinsics_views=len(record.extrinsic_views),
        )
        euler = Rotation.from_matrix(gripper_from_camera[:3, :3]).as_euler("xyz", degrees=True)
        sd = fit.position_sd_m * 1000
        lines += [
            f"- pose in `{wrist_camera_link(record.arm)}`: at ({', '.join(f'{v * 1000:.1f}' for v in gripper_from_camera[:3, 3])}) mm, "
            f"turned ({', '.join(f'{v:.1f}' for v in euler)}) deg (xyz); RMS {camera.extrinsics_rms_px:.2f} px over "
            f"{camera.extrinsics_views} views (worst {ext_errors.max():.2f} px)",
            f"- known to +- ({', '.join(f'{v:.1f}' for v in sd)}) mm and {np.nanmax(fit.rotation_sd_deg):.2f} deg "
            f"(each view left out in turn); the views' turns spread {', '.join(f'{v:.0f}' for v in fit.spread_deg)} deg"
            + ("" if fit.settled else f": NOT settled ({fit.needs()})"),
            f"- the board on the table at ({', '.join(f'{v * 100:.1f}' for v in root_from_board[:3, 3])}) cm (root frame)",
        ]
    else:
        lines.append("- pose on the gripper: too few views, not calibrated")
    return camera, [*lines, ""]


def refit(start_from: Path, out_dir: Path, square_m: float | None = None) -> Path | None:
    """Fit every `<arm>_views.json` in `out_dir` again, with no robot; write the calibration and the report.

    `square_m` replaces the square size the views were taken with (measured again: a printer may scale the page). The
    lens does not depend on it; the camera's distance to the board, and so its pose, scale with it.
    """
    model = XleRobotModel.load(start_from)
    report = ["# Wrist cameras (refitted from the saved views)", ""]
    fitted = False
    for path in sorted(out_dir.glob("*_views.json")):
        record = ArmRecord.load(path)
        if square_m is not None:
            record.board = replace(record.board, square_m=square_m)
        if (
            record.arm in model.arm_calibration
            and model.arm_calibration[record.arm] != record.arm_calibration
        ):
            raise ValueError(f"{path} was taken under another arm calibration than {start_from}'s")
        camera, lines = fit_record(model, record)
        report += lines
        if camera is not None:
            model, fitted = model.with_wrist_camera(record.arm, camera), True
    if not fitted:
        return None
    path = out_dir / CALIBRATION_FILE
    if path.resolve() == start_from.resolve():
        raise ValueError(f"{path} would overwrite --start-from: pick another folder")
    model.save(path)
    (out_dir / REPORT_FILE).write_text("\n".join(report))
    return path


# ---- the session --------------------------------------------------------------------------------------------


def annotate(image: np.ndarray, lines: Sequence[str], corners: np.ndarray | None, board: Board) -> np.ndarray:
    """A copy of the RGB `image` with the found corners and the status lines."""
    out = image.copy()
    if corners is not None:
        cv2.drawChessboardCorners(
            out, (board.cols, board.rows), corners.reshape(-1, 1, 2).astype(np.float32), True
        )
    for row, text in enumerate(lines):
        org = (8, 20 + 20 * row)
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


class WristCameraSession:
    """Calibrate one arm's wrist camera, moved by keys. See the module docstring.

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
        only: str | None = None,
        focus: bool = True,
        resume: bool = False,
    ):
        """Set up the session; `only` is "intrinsics" or "extrinsics" (the latter needs the lens in `model`).

        With `focus`, it starts by streaming the camera for the lens to be focused (accepted with `accept`). With
        `resume`, the views saved in `out_dir` for this arm (`<arm>_views.json`) are kept and added to.
        """
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
        self.camera: WristCamera | None = model.wrist_cameras.get(arm) if only == "extrinsics" else None
        if only == "extrinsics" and self.camera is None:
            raise ValueError(f"no {arm} wrist camera lens in the calibration: calibrate the intrinsics first")
        self.first_phase = "extrinsics" if only == "extrinsics" else "intrinsics"
        self.phase = "focus" if focus else self.first_phase
        self.only = only
        self.sharpness = 0.0  # of the current frame
        self.seen = ""  # what became of the board in the current frame (the status line says)
        self.hint: str | None = None  # the board looks like another pattern (`pattern_hint`)
        self.last_frame_saved = 0.0
        self.best_sharpness = 0.0  # in the focus phase
        self.signatures: list[np.ndarray] = []
        self.board_fixed = False  # extrinsics: the person put the board down and pressed c
        self.start_target: tuple[np.ndarray, float, float] | None = None  # the start pose (tip, pitch, roll)
        self.next_view = 0  # into `POSE_VIEWS`
        self.solved: dict[str, float] = {}  # the joints `move_to` last solved
        self.pose_fit: ExtrinsicsFit | None = None  # the camera's pose so far (fitted as the views come)
        self.target: tuple[np.ndarray, float, float] | None = (
            None  # the tip's position, pitch and roll asked for
        )
        self.frame: WristCapture | None = None
        self.corners: np.ndarray | None = None
        self.message = ""
        self.done = False
        (out_dir / "images").mkdir(parents=True, exist_ok=True)
        saved = out_dir / f"{arm}_views.json"
        if resume and saved.is_file():
            old = ArmRecord.load(saved)
            if old.board != board or old.arm_calibration != self.record.arm_calibration:
                raise ValueError(f"{saved} was taken with another board or arm calibration: a new --out-dir")
            self.record = old
            self.signatures = [view_signature(np.asarray(v), old.size, board) for v in old.intrinsic_views]
            self.message = f"resumed: {len(old.intrinsic_views)} lens views, {len(old.extrinsic_views)} pose views (c fits)"

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
        # a few degrees short is fine: every view keeps the joints measured with it
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
        margin, (position, pitch, roll) = next(
            (r for r in reachable if r[0] >= START_JOINT_MARGIN_DEG), max(reachable, key=lambda r: r[0])
        )
        self.start_target = (position, pitch, roll)
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
        """Take a new wrist frame, find the board in it, keep it as a view if it is a new one (intrinsics), show it."""
        try:
            self.frame = self.client.wrist_capture(self.arm)
        except ServerError as e:  # e.g. the camera dropped for a moment: the server reopens it
            self.message = f"no frame from the camera ({e}); retrying"
            self.corners = None
            if self.frame is not None:
                self.render()
            return
        height, width = self.frame.color.shape[:2]
        if self.record.size == (0, 0):
            self.record.size = (width, height)
        self.corners = find_board(self.frame.color, self.board)
        self.sharpness = sharpness(self.frame.color, self.corners)
        if self.phase == "focus":
            self.best_sharpness = max(self.best_sharpness, self.sharpness)
            self.hint = pattern_hint(self.frame.color, self.board)
        if self.corners is None:
            n = self.board.cols * self.board.rows
            self.seen = f"board NOT found (all {n} inner corners must show; frame saved as {LAST_FRAME_FILE})"
            if time.time() - self.last_frame_saved > LAST_FRAME_EVERY_S:
                cv2.imwrite(
                    str(self.out_dir / LAST_FRAME_FILE), cv2.cvtColor(self.frame.color, cv2.COLOR_RGB2BGR)
                )
                self.last_frame_saved = time.time()
        else:
            self.seen = "board found"
        if self.phase == "intrinsics" and self.corners is not None:
            signature = view_signature(self.corners, self.record.size, self.board)
            if is_new_view(signature, self.signatures):
                self.hint = pattern_hint(self.frame.color, self.board)
                if self.hint is None:
                    self._keep_intrinsic(signature)
                else:
                    self.seen = "not kept: the pattern looks wrong"
            else:
                self.seen = "board found, like a kept view: tilt, turn or move it (space keeps it)"
        self.render()

    def _keep_intrinsic(self, signature: np.ndarray) -> None:
        assert self.frame is not None and self.corners is not None
        index = len(self.record.intrinsic_views)
        cv2.imwrite(
            str(self.out_dir / "images" / f"{self.arm}_intrinsics_{index:03d}.png"),
            cv2.cvtColor(self.frame.color, cv2.COLOR_RGB2BGR),
        )
        self.record.intrinsic_views.append(self.corners.tolist())
        self.signatures.append(signature)
        self.record.save(self.out_dir)
        self.message = f"view {index + 1} kept"

    def status(self) -> list[str]:
        """The status lines drawn on the picture."""
        if self.phase == "focus":
            verdict = "in focus" if self.sharpness >= SHARP_ENOUGH else "blurred"
            lines = [
                f"{self.arm} FOCUS: hold the board ~20 cm away, turn the lens until the number peaks",
                f"sharpness {self.sharpness:.0f} ({verdict}; best {self.best_sharpness:.0f})"
                + (" | board found" if self.corners is not None else "")
                + " | Enter or click: accept",
            ]
        elif self.phase == "intrinsics":
            n = len(self.record.intrinsic_views)
            lines = [
                f"{self.arm} LENS: tilt, turn, raise the board for new views (space: keep this one): "
                f"{n} of {MIN_INTRINSIC_VIEWS} kept",
            ]
            missing = lens_coverage_missing(self.signatures)
            if missing:
                lines.append(f"would help: the board {'; '.join(missing)} (held up, tilted)")
            if n >= MIN_INTRINSIC_VIEWS:
                lines.append(
                    "c: fit the lens (if the views pin it down; then the board goes flat on the table)"
                )
        elif self.phase == "extrinsics" and not self.board_fixed:
            lines = [
                f"{self.arm} CAMERA POSE: lay the board flat on the table under the camera (its corners drawn here),",
                "where it stays from now on; then c.",
            ]
        elif self.phase == "extrinsics":
            n = len(self.record.extrinsic_views)
            spread = self.spread()
            lines = [
                f"{self.arm} CAMERA POSE: the board stays put; the arm looks at it from several poses",
                f"n: the arm moves to suggested view {min(self.next_view + 1, len(POSE_VIEWS))} of {len(POSE_VIEWS)} "
                "and takes it | space: take this view (after moving the arm by keys)",
                f"{n} views (fitted from {MIN_EXTRINSIC_VIEWS}); turns {spread[0]:.0f} and {spread[1]:.0f} deg "
                f"(need {MIN_SPREAD_DEG:.0f} each)",
            ]
            if self.pose_fit is not None:
                lines.append(self.pose_fit.describe())
                lines.append(
                    "SETTLED: c accepts (more views first: fine)"
                    if self.pose_fit.settled
                    else f"not settled (under {POSE_SETTLED_M * 1000:.0f} mm, {POSE_SETTLED_DEG:.0f} deg): "
                    + self.pose_fit.needs()
                )
        else:
            lines = [
                f"{self.arm} CHECK: the drawn fingertips should sit on the real ones | c: done, u: more views"
            ]
        if self.hint:
            lines.append(self.hint)
        if self.seen and self.phase != "focus":
            lines.append(self.seen)
        lines.append("arrows / w s: move | [ ]: pitch | , .: roll | u: undo | q: quit")
        if self.message:
            lines.append(self.message)
        return lines

    def render(self) -> None:
        """Show the frame with the found corners, the fingertips (once the pose is known) and the status."""
        if self.frame is None:
            return
        out = annotate(self.frame.color, self.status(), self.corners, self.board)
        if self.camera is not None and self.camera.gripper_from_camera is not None:
            tips = fingertips_in_wrist_image(self.model, self.arm, self.camera, self.frame.state[self.arm_id])
            for (u, v), color in zip(tips, FINGERTIP_COLORS, strict=True):
                if np.isfinite([u, v]).all():
                    cv2.circle(out, (int(round(u)), int(round(v))), 9, color, 2, cv2.LINE_AA)
        if self.phase == "intrinsics":
            for view in self.record.intrinsic_views:  # where the picture is covered so far
                hull = cv2.convexHull(np.asarray(view, dtype=np.float32))
                cv2.polylines(out, [hull.astype(np.int32)], True, (0, 160, 0), 1, cv2.LINE_AA)
        self.show(out)

    # ---- views and fits ---------------------------------------------------------------------------------------

    def spread(self) -> np.ndarray:
        """The extrinsics views' turns about their two main axes (deg)."""
        rotations = [
            self.model.frame_pose(
                wrist_camera_link(self.arm), self.model.joint_vector({self.arm: v.arm_pose})
            )[:3, :3]
            for v in self.record.extrinsic_views
        ]
        return rotation_spread(rotations)

    def can_fit_extrinsics(self) -> bool:
        """Enough views, turned about two axes."""
        return len(self.record.extrinsic_views) >= MIN_EXTRINSIC_VIEWS and self.spread()[1] >= MIN_SPREAD_DEG

    def refit_pose(self) -> None:
        """Fit the camera's pose to the views so far (from `MIN_EXTRINSIC_VIEWS`), with its jackknife; draw it."""
        views = self.record.extrinsic_views
        if self.camera is None or len(views) < MIN_EXTRINSIC_VIEWS:
            self.pose_fit = None
            return
        self.pose_fit = fit_extrinsics(self.model, self.arm, self.camera, views, self.board)
        self.camera = replace(  # the fingertips drawn live from it
            self.camera,
            gripper_from_camera=tuple(
                tuple(float(v) for v in row) for row in self.pose_fit.gripper_from_camera
            ),
        )

    def take(self) -> None:
        """Keep this frame as a view (space): any view in intrinsics, the arm's pose too in extrinsics."""
        if self.frame is None or self.corners is None:
            self.message = "the board is not found here: no view taken"
        elif self.phase == "intrinsics":
            self._keep_intrinsic(view_signature(self.corners, self.record.size, self.board))
        elif self.phase == "extrinsics" and self.board_fixed:
            index = len(self.record.extrinsic_views)
            image = f"images/{self.arm}_extrinsics_{index:03d}.png"
            cv2.imwrite(str(self.out_dir / image), cv2.cvtColor(self.frame.color, cv2.COLOR_RGB2BGR))
            arm_pose = {k: float(v) for k, v in self.frame.state[self.arm_id].items()}
            self.record.extrinsic_views.append(
                ArmView(arm_pose, self.corners.tolist(), image, self.frame.timestamp)
            )
            self.record.save(self.out_dir)
            self.message = f"view {index + 1} taken"
            self.refit_pose()
        self.render()

    def undo(self) -> None:
        """Drop the last view of this phase."""
        views: list[Any] = (
            self.record.intrinsic_views if self.phase == "intrinsics" else self.record.extrinsic_views
        )
        if views:
            views.pop()
            if self.phase == "intrinsics" and self.signatures:
                self.signatures.pop()
            self.record.save(self.out_dir)
            self.message = f"dropped the last view ({len(views)} left)"
            if self.phase in ("extrinsics", "check"):
                self.refit_pose()
        if self.phase == "check":
            self.phase, self.message = "extrinsics", "take more views, then c"
        self.render()

    def calibrate(self) -> None:
        """`c`: fit this phase and go on (intrinsics -> extrinsics -> check -> done)."""
        if self.phase == "intrinsics":
            views = [np.asarray(v) for v in self.record.intrinsic_views]
            if len(views) < MIN_INTRINSIC_VIEWS:
                self.message = f"{MIN_INTRINSIC_VIEWS} views first"
                self.render()
                return
            focal, spread = lens_settled(views, self.board, self.record.size)
            if spread > LENS_SETTLED:
                missing = lens_coverage_missing(self.signatures)
                self.message = (
                    f"not settled: the focal length ({focal:.0f} px) varies {spread * 100:.0f}% across the views (under "
                    f"{LENS_SETTLED * 100:.0f}%): more views"
                    + (f", the board {'; '.join(missing)}" if missing else ", tilted and nearer / farther")
                )
            else:
                camera, errors = calibrate_intrinsics(views, self.board, self.record.size, self.record.lens)
                low, high = (f * self.record.size[0] for f in FOCAL_RANGE)
                if not (low <= camera.fx <= high and low <= camera.fy <= high):
                    self.message = (
                        f"the lens fit is implausible (fx {camera.fx:.0f} px): take more views, tilted and at "
                        "different distances (u drops bad ones), then c"
                    )
                    self.render()
                    return
                self.camera = camera
                self.message = f"lens fitted: fx {camera.fx:.0f} px (halves agree within {spread * 100:.1f}%), RMS {camera.rms_px:.2f} px (worst view {errors.max():.2f})"
                if self.only == "intrinsics":
                    self.done = True
                else:
                    self.phase = "extrinsics"
        elif self.phase == "extrinsics" and not self.board_fixed:
            self.board_fixed, self.message = True, "board down: n for the first suggested view"
        elif self.phase == "extrinsics":
            fit = self.pose_fit
            if fit is None:
                self.message = f"{MIN_EXTRINSIC_VIEWS} views first"
            elif not fit.settled:
                self.message = f"not settled: {fit.needs()}"
            else:
                assert self.camera is not None
                self.camera = replace(
                    self.camera,
                    gripper_from_camera=tuple(
                        tuple(float(v) for v in row) for row in fit.gripper_from_camera
                    ),
                    extrinsics_rms_px=float(np.sqrt(np.mean(fit.errors_px**2))),
                    extrinsics_views=len(fit.errors_px),
                )
                self.model = self.model.with_wrist_camera(self.arm, self.camera)
                self.phase = "check"
                self.message = f"pose accepted: {fit.describe()}"
        else:
            self.done = True
        self.render()

    def accept_focus(self) -> None:
        """Enter / a click in the focus phase: keep the lens as it is and go on to the views."""
        if self.phase == "focus":
            self.record.focus_sharpness = self.sharpness
            self.record.save(self.out_dir)
            self.phase = self.first_phase
            self.message = f"focus accepted at sharpness {self.sharpness:.0f}"
            self.refresh()  # the board in view now: the first view
            return
        self.render()

    def suggested_view(self) -> None:
        """`n`: move the arm to the next of `POSE_VIEWS` and take the view if the board is found there."""
        if self.phase != "extrinsics" or not self.board_fixed or self.start_target is None:
            self.message = "n moves the arm to suggested views once the board lies on the table (c)"
            self.render()
            return
        while self.next_view < len(POSE_VIEWS):
            dx, dy, dz, dpitch, roll = POSE_VIEWS[self.next_view]
            self.next_view += 1
            position, pitch, _ = self.start_target
            if self.move_to(position + [dx, dy, dz], pitch + dpitch, roll):
                break
            # out of reach: try the next one
        else:
            self.message = "no more suggested views: move the arm by keys and take views with space, or c"
            self.render()
            return
        self.refresh()
        if self.corners is None:
            self.message = (
                "the board is not (all) in view here: move the arm by keys, then space; or n to skip"
            )
            self.render()
            return
        self.take()

    def handle(self, action: str) -> bool:
        """Do one action from `decode_key`. Returns False when the person quits."""
        if action == "quit":
            return False
        if action == "accept" or (action == "calibrate" and self.phase == "focus"):
            self.accept_focus()
        elif action == "next":
            self.suggested_view()
        elif action == "take":
            self.take()
        elif action == "undo":
            self.undo()
        elif action == "calibrate":
            self.calibrate()
        else:
            self.jog(action)
        return True


def write(model: XleRobotModel, records: Sequence[ArmRecord], out_dir: Path) -> Path | None:
    """Fit every arm's views and write the calibration (`calibration.urdf`) and the report; None if nothing fits."""
    report = ["# Wrist cameras", ""]
    fitted = False
    for record in records:
        camera, lines = fit_record(model, record)
        report += lines
        if camera is not None:
            model, fitted = model.with_wrist_camera(record.arm, camera), True
    if not fitted:
        return None
    path = out_dir / CALIBRATION_FILE
    model.save(path)
    (out_dir / REPORT_FILE).write_text("\n".join(report))
    return path


def ask_board(pattern: str | None, square_mm: float | None) -> Board:
    """The board from the arguments, or asked for in the terminal."""
    while pattern is None:
        text = input("the board's inner corners, columns x rows (e.g. 9x6): ").strip()
        try:
            parse_pattern(text)
            pattern = text
        except ValueError:
            print("two whole numbers, e.g. 9x6 (the corners where four squares meet)")
    while square_mm is None:
        text = input("one square's size as printed, in mm (measure it): ").strip()
        try:
            square_mm = float(text)
        except ValueError:
            print("a number of mm, e.g. 25")
    cols, rows = parse_pattern(pattern)
    return Board(cols, rows, square_mm / 1000.0)


def main() -> None:
    """Command line entry point; see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("endpoint", nargs="?", default=f"tcp://localhost:{DEFAULT_PORT}")
    parser.add_argument(
        "--start-from", type=Path, required=True, help="the calibration.urdf to add the cameras to"
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--arms", default="left,right", help="which wrist cameras, in order (left,right)")
    parser.add_argument(
        "--pattern", help="the board's inner corners, columns x rows, e.g. 9x6 (asked if not given)"
    )
    parser.add_argument(
        "--square-mm", type=float, help="one square's size as printed, mm (asked if not given)"
    )
    parser.add_argument(
        "--lens", choices=WRIST_CAMERA_MODELS, default="rational", help="the lens model (rational)"
    )
    parser.add_argument(
        "--only",
        choices=("intrinsics", "extrinsics"),
        help="one step (extrinsics: the lens from --start-from)",
    )
    parser.add_argument(
        "--no-focus", action="store_true", help="skip the focus step (the lenses are focused)"
    )
    parser.add_argument(
        "--resume", action="store_true", help="keep the views saved in --out-dir and add to them"
    )
    parser.add_argument("--step-cm", type=float, default=2.0, help="each key's move, cm (2)")
    parser.add_argument("--max-vel", type=float, default=20.0, help="arm speed limit, deg/s (20)")
    parser.add_argument(
        "--refit",
        action="store_true",
        help="no robot: fit --out-dir's saved views again (with --square-mm: a square size measured again)",
    )
    args = parser.parse_args()
    if (args.out_dir / CALIBRATION_FILE).resolve() == args.start_from.resolve():
        parser.error("--out-dir would overwrite --start-from: pick another folder")
    if args.refit:
        path = refit(
            args.start_from, args.out_dir, None if args.square_mm is None else args.square_mm / 1000.0
        )
        if path is None:
            parser.error(f"no saved views to fit in {args.out_dir}")
        print(f"Wrote {path}\n{(args.out_dir / REPORT_FILE).read_text()}")
        return
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    if not arms or set(arms) - set(ARMS):
        parser.error(f"--arms must be some of {','.join(ARMS)}")
    board = ask_board(args.pattern, args.square_mm)
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
                    args.only,
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
            for arm in arms:
                session = sessions[arm]
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
                records.append(session.record)
                for other in sessions.values():  # the next arm's model has this arm's camera
                    other.model = session.model
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
