"""Calibration boards: what the board looks like, where its points are on it, and finding them in a picture.

`docs/camera_calibration.md`, "The rework" (2b). A board gives, per picture, the points it found as ids and pixels
(`Detection`), and each id's position on the board (`points`). Fits take whatever each view found: a partial board,
or a different number of points in each view and camera, needs nothing special.

Every calibration uses a `CharucoBoard`: a checkerboard with an ArUco marker in each white square (OpenCV's
`cv2.aruco`). Every corner is identified by the markers around it, so a board half hidden (by the arm) or half out of
the picture still gives the corners it shows. The plain checkerboard (found whole or not at all, and read two ways
when symmetric) was dropped on 2026-11-16 (`docs/calibration_plan.md`, item 1): one kind of detection everywhere.
Sessions recorded with it (before 2026-11-11) are no longer read.

`parse_board` reads a board from text ("charuco:9x7:0.025:0.018:DICT_4X4_50"; `PRINTED_BOARD` is the one printed),
and `board_from_dict` / `Board.to_dict` save and load one.
"""

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, ClassVar

import cv2
import numpy as np

MIN_POINTS = 6  # fewer points in a picture is no detection (a pose needs 4, and they could be in a row)
REFINE_WINDOW_PX = (
    2,
    2,
)  # ChArUco corners refined in a 5 x 5 window: smaller than a corner's gap to its markers
REFINE_STOP = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)


@dataclass(frozen=True)
class Detection:
    """The board's points found in one picture: their ids (N,) and pixels (N, 2), in the board's id order."""

    ids: np.ndarray
    pixels: np.ndarray

    def __len__(self) -> int:
        """The number of points found."""
        return len(self.ids)


# ---- ChArUco ------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CharucoBoard:
    """A ChArUco board: `cols` x `rows` squares (`square_m`), an ArUco marker (`marker_m`) in each white one.

    Its points are the inner corners, (cols - 1) x (rows - 1), ids row by row (OpenCV's); the board's frame has its
    origin at the board's outer top-left corner (as OpenCV's), z = 0 on the board.
    """

    cols: int
    rows: int
    square_m: float
    marker_m: float
    dictionary: str = "DICT_5X5_100"
    kind: ClassVar[str] = "charuco"
    min_view_fraction: ClassVar[float] = 0.5  # the markers name any corners found: part of the board will do

    def __post_init__(self) -> None:
        """Check the dictionary and that its markers are enough for the board's white squares."""
        if not hasattr(cv2.aruco, self.dictionary):
            raise ValueError(f"no ArUco dictionary {self.dictionary!r}")
        if self.marker_m >= self.square_m:
            raise ValueError("the markers must be smaller than the squares")
        markers = (self.cols * self.rows) // 2
        if markers > self._dictionary().bytesList.shape[0]:
            raise ValueError(f"{self.dictionary} has too few markers for {markers} white squares")

    def _dictionary(self) -> "cv2.aruco.Dictionary":
        return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dictionary))

    def _board(self) -> "cv2.aruco.CharucoBoard":
        return cv2.aruco.CharucoBoard(
            (self.cols, self.rows), self.square_m, self.marker_m, self._dictionary()
        )

    def points(self, ids: Sequence[int] | np.ndarray | None = None) -> np.ndarray:
        """The inner corners of `ids` (all, by default) in the board's frame (N, 3), z = 0."""
        everything = np.asarray(self._board().getChessboardCorners(), dtype=np.float64).reshape(-1, 3)
        return everything if ids is None else everything[np.asarray(ids, dtype=int)]

    def detect(self, image: np.ndarray) -> Detection | None:
        """The inner corners the markers identify (any number of them), or None if fewer than `MIN_POINTS`."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        corners, ids, _, _ = cv2.aruco.CharucoDetector(self._board()).detectBoard(gray)
        if ids is None or len(ids) < MIN_POINTS:
            return None
        # OpenCV's ChArUco corners sit ~0.44 px off in x and y on small squares (a synthetic test, `test_boards`:
        # the same at any square size and distance, as if half a pixel): refined again on the picture, they do not
        corners = cv2.cornerSubPix(
            gray,
            np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2),
            REFINE_WINDOW_PX,
            (-1, -1),
            REFINE_STOP,
        )
        ids = np.asarray(ids, dtype=int).ravel()
        order = np.argsort(ids)
        return Detection(ids[order], np.asarray(corners, dtype=np.float64).reshape(-1, 2)[order])

    def image(self, px_per_m: float, margin_px: int = 0) -> np.ndarray:
        """The board; its frame's origin (the outer top-left corner) at (`margin_px`, `margin_px`), x to the right."""
        size = (
            int(round(self.cols * self.square_m * px_per_m)) + 2 * margin_px,
            int(round(self.rows * self.square_m * px_per_m)) + 2 * margin_px,
        )
        return self._board().generateImage(size, marginSize=margin_px)

    def to_dict(self) -> dict[str, Any]:
        """{"kind": "charuco", "cols", "rows", "square_m", "marker_m", "dictionary"}."""
        return {"kind": self.kind, **asdict(self)}


Board = CharucoBoard  # the one kind of board (the module docstring); the name the fits and the scripts use


# ---- reading, saving, the pose --------------------------------------------------------------------------------

# the board printed for this robot (2026-11-11, on A4): 9 x 7 squares of 25 mm,
# 18 mm markers; every script's default
PRINTED_BOARD = "charuco:9x7:0.025:0.018:DICT_4X4_50"


def board_from_dict(data: dict[str, Any]) -> Board:
    """A board from `to_dict`'s dict. A checkerboard's (a dict without "kind": the older sessions') is refused."""
    data = dict(data)
    kind = data.pop("kind", "checkerboard")
    if kind != CharucoBoard.kind:
        raise ValueError(
            f"a {kind} board: no longer read (ChArUco only since 2026-11-16, `boards`); record the session again"
        )
    return CharucoBoard(**data)


def parse_board(text: str) -> Board:
    """A board from text: "charuco:COLSxROWS:SQUARE_M:MARKER_M[:DICTIONARY]" (squares; `PRINTED_BOARD`)."""
    kind, *fields = text.strip().split(":")
    try:
        cols, rows = (int(v) for v in fields[0].lower().split("x"))
        if kind == "charuco" and len(fields) in (3, 4):
            return CharucoBoard(cols, rows, float(fields[1]), float(fields[2]), *fields[3:])
    except (IndexError, ValueError) as e:
        raise ValueError(f"cannot read the board {text!r}: {e}") from e
    raise ValueError(f"cannot read the board {text!r}: e.g. {PRINTED_BOARD} (ChArUco only)")


def board_pose(
    board: Board,
    detection: Detection,
    matrix: np.ndarray,
    coeffs: Sequence[float] = (),
    fisheye: bool = False,
) -> np.ndarray:
    """camera_from_board (4x4) from one detection, through the lens (`matrix`, `coeffs`; OpenCV's models)."""
    objects = board.points(detection.ids)
    if fisheye:
        normalized = cv2.fisheye.undistortPoints(
            detection.pixels.reshape(-1, 1, 2), matrix, np.asarray(coeffs, dtype=np.float64)
        )
        ok, rvec, tvec = cv2.solvePnP(objects, normalized, np.eye(3), None)
    else:
        ok, rvec, tvec = cv2.solvePnP(
            objects, detection.pixels, matrix, np.asarray(coeffs, dtype=np.float64) if len(coeffs) else None
        )
    if not ok:
        raise ValueError("the board's pose could not be solved from this detection")
    pose = np.eye(4)
    pose[:3, :3] = cv2.Rodrigues(rvec)[0]
    pose[:3, 3] = np.ravel(tvec)
    return pose
