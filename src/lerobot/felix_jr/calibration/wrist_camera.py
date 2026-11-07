"""Calibrate the wrist cameras: their lens (intrinsics) and their pose on the gripper (extrinsics), from a checkerboard.

  uv run python -m lerobot.felix_jr.calibration.wrist_camera tcp://<pi>:5555 --start-from <calibration.urdf>
      --out-dir <folder> [--arms left,right] [--pattern 9x6] [--square-mm 25] [--lens rational|fisheye]
      [--only intrinsics|extrinsics] [--step-cm 2] [--max-vel 20]
  uv run python -m lerobot.felix_jr.calibration.wrist_camera --refit --start-from <calibration.urdf> --out-dir <folder>

The server must run with the wrist cameras (`server.server --left-wrist-camera ... --right-wrist-camera ...`). Without
`--pattern` and `--square-mm` the terminal asks for the board: its inner corners (columns x rows: the corners where
four squares meet, not the squares) and the size of one square in mm as printed (measure it: printers scale).

For each arm in `--arms` in turn (the other arm stays at rest), after one approval in the terminal, the arm goes to a
start pose: the gripper tilted down as steeply as it reaches (70 deg on this robot: straight down is out of reach),
about 18 cm above the table and 22 cm in front of the arm. A window shows its wrist
camera; the keys move the arm (each a small move, the gripper's tip as the reference):

  arrow keys    move it over the table (root frame x and y) by --step-cm
  w / s         up / down
  [ / ]         tilt the gripper (pitch) by 10 deg
  , / .         turn it about its axis (roll) by 15 deg
  space         take this view (the board must be found: its corners are drawn)
  u             drop the last view
  c             calibrate with the views so far, and go on
  q or Esc      stop (what is calibrated so far is written)

1. **Intrinsics.** Hold the board under the camera and move it: tilted every way, near and far, into the corners of
   the picture (the arm may move too). A view is taken by itself when the board is found in a pose unlike the ones
   kept (or with space); the corners of the picture covered so far are drawn. `c` fits the lens (OpenCV's rational
   model, or `--lens fisheye` for a lens wider than about 120 deg) once there are `MIN_INTRINSIC_VIEWS`.
2. **Extrinsics.** Put the board flat on the table, where it stays, and press `c`. Then move the arm to see it from
   different places and tilts (tilt and roll it: a view from straight above only, however many, leaves the camera's
   pose undetermined) and take each with space. `c` fits, once there are `MIN_EXTRINSIC_VIEWS` spread over at least
   `MIN_SPREAD_DEG` about two axes, the camera's pose in `<arm>_gripper_link` and the board's on the table together
   (`cv2.calibrateRobotWorldHandEye`, then refined on the corners' pixels). The model's fingertips are then drawn on
   the live picture: they should sit on the real ones.

The fit uses the arm's measured joints and the model (`--start-from`), so the arm's own error (its sag, its
calibration) averages into the camera's pose; many views across the workspace keep it small. The output,
`--out-dir/calibration.urdf`, is `--start-from` with the wrist cameras (`XleRobotModel.with_wrist_camera`), and
`wrist_cameras.md` reports the errors. The views (`<arm>_views.json`, `images/`) are kept: `--refit` fits them again
with no robot. The script runs inside a client session, so on Ctrl+C or a crash the robot goes to rest.
"""

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

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
from lerobot.felix_jr.server.protocol import DEFAULT_PORT
from lerobot.utils.import_utils import _scipy_available, require_package

if TYPE_CHECKING or _scipy_available:
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

ARMS = ("left", "right")
CALIBRATION_FILE = "calibration.urdf"
REPORT_FILE = "wrist_cameras.md"
MIN_INTRINSIC_VIEWS = 12
MIN_EXTRINSIC_VIEWS = 8
MIN_SPREAD_DEG = 15.0  # the views' turns must spread this far about two axes for the extrinsics
# the start pose: the gripper's tip this far in front of the arm's base, the first of these heights above the base
# (about 6 cm above the table on this robot) the arm reaches with the gripper tilted down the steepest of these
# pitches (straight down is out of reach: the wrist flex is at its limit)
START_AHEAD_M = 0.22
START_HEIGHTS_M = (0.12, 0.08, 0.04)
START_PITCHES_DEG = (-90.0, -80.0, -70.0, -60.0, -50.0, -40.0)
PITCH_STEP_DEG = 10.0
ROLL_STEP_DEG = 15.0
NEW_VIEW_PX = (
    0.08  # a found board counts as a new view when it moved this much (of the picture's size) or ...
)
NEW_VIEW_TILT = 0.15  # ... its apparent tilt changed this much from every view kept
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
    ord("q"): "quit",
    27: "quit",
}
FINGERTIP_COLORS = ((0, 255, 255), (255, 0, 255))  # RGB: the pinch point (fixed tip), the moving tip


# ---- the board ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Board:
    """A checkerboard: `cols` x `rows` inner corners, squares `square_m` wide."""

    cols: int
    rows: int
    square_m: float

    def points(self) -> np.ndarray:
        """The inner corners in the board's frame (N, 3), row by row, z = 0: OpenCV's order."""
        grid = np.mgrid[0 : self.cols, 0 : self.rows].T.reshape(-1, 2)
        return np.c_[grid * self.square_m, np.zeros(len(grid))].astype(np.float64)


def parse_pattern(text: str) -> tuple[int, int]:
    """`"9x6"` -> (9, 6): inner corners, columns x rows; one odd and one even (`find_board`)."""
    cols, rows = (int(v) for v in text.lower().replace(" ", "").split("x"))
    if cols < 3 or rows < 3:
        raise ValueError("a board needs at least 3 x 3 inner corners")
    if cols % 2 == rows % 2:
        raise ValueError(
            f"{cols} x {rows}: one count odd and one even (e.g. 9 x 6), or the board looks the same turned around"
        )
    return cols, rows


def find_board(image: np.ndarray, board: Board) -> np.ndarray | None:
    """The board's inner corners in an RGB image (N, 2), or None if it is not found whole.

    In OpenCV's order (row by row), started from the end whose outer square, diagonally past the first corner, is
    dark: the detector may start from either end, and a fixed board must keep one frame across the views (the
    extrinsics). With one odd and one even count the two ends' outer squares differ in color (`parse_pattern`).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    found, corners = cv2.findChessboardCornersSB(
        gray, (board.cols, board.rows), flags=cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if not found:
        return None
    corners = corners.reshape(-1, 2).astype(np.float64)
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


def _outer_brightness(gray: np.ndarray, corners: np.ndarray, board: Board, end: int) -> float | None:
    """The brightness of the square diagonally outside the grid at its first (`end` 0) or last (-1) corner."""
    grid = corners.reshape(board.rows, board.cols, 2)
    corner, inner = (grid[0, 0], grid[1, 1]) if end == 0 else (grid[-1, -1], grid[-2, -2])
    u, v = corner + 0.5 * (corner - inner)
    height, width = gray.shape
    if not (2 <= u < width - 2 and 2 <= v < height - 2):
        return None
    return float(gray[int(v) - 1 : int(v) + 2, int(u) - 1 : int(u) + 2].mean())


def view_signature(corners: np.ndarray, size: tuple[int, int], board: Board) -> np.ndarray:
    """Where the board is in the picture, how big and how tilted: (center x, center y, scale, tilt x, tilt y).

    Positions and scale as fractions of the picture; the tilts are how much the board's far rows / columns shrink
    against its near ones (0 for a board square to the camera).
    """
    width, height = size
    grid = corners.reshape(board.rows, board.cols, 2)
    center = corners.mean(axis=0) / [width, height]
    scale = float(np.sqrt(cv2.contourArea(cv2.convexHull(corners.astype(np.float32))) / (width * height)))
    top, bottom = np.linalg.norm(grid[0, -1] - grid[0, 0]), np.linalg.norm(grid[-1, -1] - grid[-1, 0])
    left, right = np.linalg.norm(grid[-1, 0] - grid[0, 0]), np.linalg.norm(grid[-1, -1] - grid[0, -1])
    tilt_x = (top - bottom) / max(top + bottom, 1e-9)
    tilt_y = (left - right) / max(left + right, 1e-9)
    return np.array([center[0], center[1], scale, tilt_x, tilt_y])


def is_new_view(signature: np.ndarray, kept: Sequence[np.ndarray]) -> bool:
    """Whether a view differs from every kept one: moved or resized by `NEW_VIEW_PX`, or tilted by `NEW_VIEW_TILT`."""
    for other in kept:
        moved = np.linalg.norm(signature[:3] - other[:3])
        tilted = np.abs(signature[3:] - other[3:]).max()
        if moved < NEW_VIEW_PX and tilted < NEW_VIEW_TILT:
            return False
    return True


# ---- the fits ------------------------------------------------------------------------------------------------


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


def calibrate_extrinsics(
    model: XleRobotModel, arm: str, camera: WristCamera, views: Sequence[ArmView], board: Board
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The camera's pose on the gripper and the board's on the table, from views of the board at several poses.

    Returns gripper_from_camera (4x4, in `<arm>_gripper_link`), root_from_board (4x4) and each view's RMS error (px).
    Started from `cv2.calibrateRobotWorldHandEye` on the board's pose in each view (`board_pose`) and the gripper's
    (the model at the measured joints), then refined on all the corners' pixels (a Huber loss: a bad view does not
    pull the rest).
    """
    require_package("scipy", extra="felix_jr")
    if len(views) < 3:
        raise ValueError(f"{len(views)} views: too few for the extrinsics")
    link = wrist_camera_link(arm)
    root_from_gripper = [model.frame_pose(link, model.joint_vector({arm: v.arm_pose})) for v in views]
    camera_from_board = [board_pose(camera, np.asarray(v.corners), board) for v in views]
    gripper_from_root = [np.linalg.inv(p) for p in root_from_gripper]
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

    points = board.points()
    observed = [np.asarray(v.corners) for v in views]

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
    return gripper_from_camera, root_from_board, errors


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
    camera_from_root = np.linalg.inv(model.wrist_camera_pose(arm, q))
    return camera.project(_apply(camera_from_root, tips))


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
    lines += [
        f"- lens: {camera.model}, {camera.width} x {camera.height}, fx {camera.fx:.1f} fy {camera.fy:.1f}, "
        f"center ({camera.ppx:.1f}, {camera.ppy:.1f}); RMS {camera.rms_px or 0:.2f} px over {camera.n_views} views "
        f"{lens_line}",
    ]
    if len(record.extrinsic_views) >= 3:
        gripper_from_camera, root_from_board, ext_errors = calibrate_extrinsics(
            model, record.arm, camera, record.extrinsic_views, record.board
        )
        rotations = [
            model.frame_pose(wrist_camera_link(record.arm), model.joint_vector({record.arm: v.arm_pose}))[
                :3, :3
            ]
            for v in record.extrinsic_views
        ]
        spread = rotation_spread(rotations)
        camera = replace(
            camera,
            gripper_from_camera=tuple(tuple(float(v) for v in row) for row in gripper_from_camera),
            extrinsics_rms_px=float(np.sqrt(np.mean(ext_errors**2))),
            extrinsics_views=len(record.extrinsic_views),
        )
        euler = Rotation.from_matrix(gripper_from_camera[:3, :3]).as_euler("xyz", degrees=True)
        lines += [
            f"- pose in `{wrist_camera_link(record.arm)}`: at ({', '.join(f'{v * 1000:.1f}' for v in gripper_from_camera[:3, 3])}) mm, "
            f"turned ({', '.join(f'{v:.1f}' for v in euler)}) deg (xyz); RMS {camera.extrinsics_rms_px:.2f} px over "
            f"{camera.extrinsics_views} views (worst {ext_errors.max():.2f} px); the views' turns spread "
            f"{', '.join(f'{v:.0f}' for v in spread)} deg",
            f"- the board on the table at ({', '.join(f'{v * 100:.1f}' for v in root_from_board[:3, 3])}) cm (root frame)",
        ]
    else:
        lines.append("- pose on the gripper: too few views, not calibrated")
    return camera, [*lines, ""]


def refit(start_from: Path, out_dir: Path) -> Path | None:
    """Fit every `<arm>_views.json` in `out_dir` again, with no robot; write the calibration and the report."""
    model = XleRobotModel.load(start_from)
    report = ["# Wrist cameras (refitted from the saved views)", ""]
    fitted = False
    for path in sorted(out_dir.glob("*_views.json")):
        record = ArmRecord.load(path)
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
    ):
        """Set up the session; `only` is "intrinsics" or "extrinsics" (the latter needs the lens in `model`)."""
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
        self.phase = "extrinsics" if only == "extrinsics" else "intrinsics"
        self.only = only
        self.signatures: list[np.ndarray] = []
        self.board_fixed = False  # extrinsics: the person put the board down and pressed c
        self.target: tuple[np.ndarray, float, float] | None = (
            None  # the tip's position, pitch and roll asked for
        )
        self.frame: WristCapture | None = None
        self.corners: np.ndarray | None = None
        self.message = ""
        self.done = False
        (out_dir / "images").mkdir(parents=True, exist_ok=True)

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
        if dry_run:
            return True
        target = dict(poses.joints[0]) | {"gripper.pos": current.get("gripper.pos", 0.0)}
        status = self.client.move_joints({self.arm_id: target}, max_vel=self.max_vel)
        if status["state"] != "done":
            self.message = f"the move ended as {status['state']}: {status.get('reason', '')}"
            return False
        self.target = (np.asarray(position, dtype=np.float64), pitch, roll)
        return True

    def start(self) -> bool:
        """Move to the first start pose the arm reaches (the caller has the person approve it first)."""
        for position, pitch, roll in self.start_poses():
            if self.move_to(position, pitch, roll, dry_run=True):
                ok = self.move_to(position, pitch, roll)
                self.refresh()
                return ok
        self.message = "no start pose in reach"
        self.refresh()
        return False

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
        self.frame = self.client.wrist_capture(self.arm)
        height, width = self.frame.color.shape[:2]
        if self.record.size == (0, 0):
            self.record.size = (width, height)
        self.corners = find_board(self.frame.color, self.board)
        if self.phase == "intrinsics" and self.corners is not None:
            signature = view_signature(self.corners, self.record.size, self.board)
            if is_new_view(signature, self.signatures):
                self._keep_intrinsic(signature)
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
        if self.phase == "intrinsics":
            n = len(self.record.intrinsic_views)
            lines = [
                f"{self.arm} INTRINSICS: move the board (or the arm) for new views: {n} kept"
                + (" | c: fit" if n >= MIN_INTRINSIC_VIEWS else f" (c from {MIN_INTRINSIC_VIEWS})"),
            ]
        elif self.phase == "extrinsics" and not self.board_fixed:
            lines = [f"{self.arm} EXTRINSICS: put the board flat on the table, where it stays; then c"]
        elif self.phase == "extrinsics":
            n = len(self.record.extrinsic_views)
            spread = self.spread()
            lines = [
                f"{self.arm} EXTRINSICS: space takes a view: {n} | turns {spread[0]:.0f} / {spread[1]:.0f} deg"
                + (
                    " | c: fit"
                    if self.can_fit_extrinsics()
                    else f" (c from {MIN_EXTRINSIC_VIEWS}, {MIN_SPREAD_DEG:.0f} deg x2)"
                ),
            ]
        else:
            lines = [
                f"{self.arm} CHECK: the drawn fingertips should sit on the real ones | c: done, u: more views"
            ]
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
        if self.phase == "check":
            self.phase, self.message = "extrinsics", "take more views, then c"
        self.render()

    def calibrate(self) -> None:
        """`c`: fit this phase and go on (intrinsics -> extrinsics -> check -> done)."""
        if self.phase == "intrinsics":
            if len(self.record.intrinsic_views) < MIN_INTRINSIC_VIEWS:
                self.message = f"{MIN_INTRINSIC_VIEWS} views first"
            else:
                views = [np.asarray(v) for v in self.record.intrinsic_views]
                self.camera, errors = calibrate_intrinsics(
                    views, self.board, self.record.size, self.record.lens
                )
                self.message = f"lens fitted: RMS {self.camera.rms_px:.2f} px (worst view {errors.max():.2f})"
                if self.only == "intrinsics":
                    self.done = True
                else:
                    self.phase = "extrinsics"
        elif self.phase == "extrinsics" and not self.board_fixed:
            self.board_fixed, self.message = True, "board down: move the arm, space for each view"
        elif self.phase == "extrinsics":
            if not self.can_fit_extrinsics():
                self.message = (
                    f"{MIN_EXTRINSIC_VIEWS} views turned {MIN_SPREAD_DEG:.0f} deg about two axes first"
                )
            else:
                assert self.camera is not None
                gripper_from_camera, _, errors = calibrate_extrinsics(
                    self.model, self.arm, self.camera, self.record.extrinsic_views, self.board
                )
                self.camera = replace(
                    self.camera,
                    gripper_from_camera=tuple(tuple(float(v) for v in row) for row in gripper_from_camera),
                    extrinsics_rms_px=float(np.sqrt(np.mean(errors**2))),
                    extrinsics_views=len(errors),
                )
                self.model = self.model.with_wrist_camera(self.arm, self.camera)
                self.phase = "check"
                self.message = (
                    f"pose fitted: RMS {self.camera.extrinsics_rms_px:.2f} px (worst view {errors.max():.2f})"
                )
        else:
            self.done = True
        self.render()

    def handle(self, action: str) -> bool:
        """Do one action from `decode_key`. Returns False when the person quits."""
        if action == "quit":
            return False
        if action == "take":
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
    board = ask_board(args.pattern, args.square_mm)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = XleRobotModel.load(args.start_from)
    window = "wrist camera (keys: see the status line)"

    def show(image: np.ndarray) -> None:
        cv2.imshow(window, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    records: list[ArmRecord] = []
    with FelixClient(args.endpoint, timeout_s=20.0) as client, client.session():
        model.check_robot(client)
        missing = set(arms) - set(client.hello().get("wrist_cameras", []))
        if missing:
            raise SystemExit(
                f"the server has no {', '.join(sorted(missing))} wrist camera: start it with --<arm>-wrist-camera"
            )
        try:
            for arm in arms:
                session = WristCameraSession(
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
                )
                answer = input(
                    f"the {arm} arm goes to its start pose (gripper down, ~30 cm over the table): go? [y/N] "
                )
                if answer.strip().lower() != "y":
                    break
                session.start()
                keep_going = True
                while keep_going and not session.done:
                    key = cv2.waitKeyEx(30)
                    action = decode_key(key, KEY_ACTIONS)
                    if action is None:
                        session.refresh()
                        continue
                    keep_going = session.handle(action)
                records.append(session.record)
                model = session.model
                client.home(arms=[f"{arm}_arm"])
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
