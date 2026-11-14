"""Calibration boards: what each kind looks like, where its points are on it, and finding them in a picture.

`docs/camera_calibration.md`, "The rework" (2b). A board kind gives, per picture, the points it found as ids and
pixels (`Detection`), and each id's position on the board (`points`). Fits take whatever each view found: a partial
board, or a different number of points in each view and camera, needs nothing special.

- `Checkerboard`: the inner corners of a plain checkerboard, found whole or not at all (OpenCV's
  `findChessboardCornersSB`). A symmetric one (both counts even, or both odd) looks the same turned around, so a
  detection has two readings (`readings`): the fit, or the arm's motion, picks one.
- `CharucoBoard`: a checkerboard with an ArUco marker in each white square (OpenCV's `cv2.aruco`). Every corner is
  identified by the markers around it: a board half hidden (by the arm) still gives the corners it shows, with one
  reading.

`parse_board` reads a board from text ("checkerboard:8x6:0.025", "charuco:12x9:0.02:0.015:DICT_5X5_100"), and
`board_from_dict` / `Board.to_dict` save and load one.
"""

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, ClassVar, Protocol

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


class Board(Protocol):
    """What every board kind provides (the module docstring)."""

    kind: ClassVar[str]
    # the least part of its points a view must show to count (a checkerboard is found whole or not at all)
    min_view_fraction: ClassVar[float]

    def points(self, ids: Sequence[int] | np.ndarray | None = None) -> np.ndarray:
        """The points of `ids` (all, by default) in the board's frame (N, 3), m, z = 0."""
        ...

    def detect(self, image: np.ndarray) -> Detection | None:
        """The board's points in an RGB image, or None if fewer than `MIN_POINTS` are found."""
        ...

    def readings(self, detection: Detection) -> list[Detection]:
        """Every assignment of ids the picture allows: one, or two for a symmetric board (turned around)."""
        ...

    def image(self, px_per_m: float, margin_px: int = 0) -> np.ndarray:
        """The board to print (grayscale, uint8), and the pixel of its frame's origin: see each kind."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """The board as a JSON-able dict (`board_from_dict` reads it back)."""
        ...


# ---- a plain checkerboard -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Checkerboard:
    """A checkerboard: `cols` x `rows` inner corners, squares `square_m` wide.

    The ids run row by row (OpenCV's order), from the corner whose outer square, diagonally past it, is dark; the
    board's frame has its origin at id 0, x along the row, z = 0 on the board.
    """

    cols: int
    rows: int
    square_m: float
    kind: ClassVar[str] = "checkerboard"
    min_view_fraction: ClassVar[float] = 1.0

    @property
    def symmetric(self) -> bool:
        """Whether the board looks the same turned around: both counts even, or both odd.

        Then which end its corners start from cannot be told from the picture (`detect`), only from elsewhere (the
        arm's motion, the fit): `readings` gives both.
        """
        return self.cols % 2 == self.rows % 2

    def points(self, ids: Sequence[int] | np.ndarray | None = None) -> np.ndarray:
        """The inner corners of `ids` (all, by default) in the board's frame (N, 3), z = 0."""
        grid = np.mgrid[0 : self.cols, 0 : self.rows].T.reshape(-1, 2)
        everything = np.c_[grid * self.square_m, np.zeros(len(grid))].astype(np.float64)
        return everything if ids is None else everything[np.asarray(ids, dtype=int)]

    def corners(self, image: np.ndarray) -> np.ndarray | None:
        """The inner corners in an RGB image (N, 2) in id order, or None if the board is not found whole."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Plain first: normalizing (histogram equalization) lifts the table's wood grain and lost the board in 4 of
        # 6 head views on the robot (2026-11-11). The wrist cameras' glare needs it, and 2 of their 6 views with
        # the board whole needed the exhaustive search too (the same corners where both find them, ~50 ms).
        for flags in (
            0,
            cv2.CALIB_CB_NORMALIZE_IMAGE,
            cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE,
        ):
            found, corners = cv2.findChessboardCornersSB(gray, (self.cols, self.rows), flags=flags)
            if found:
                break
        else:
            return None
        corners = corners.reshape(-1, 2).astype(np.float64)
        if self.symmetric:  # both ends alike: left as found (`readings` gives the other end)
            return corners
        first, last = self._outer_brightness(gray, corners, 0), self._outer_brightness(gray, corners, -1)
        if first is not None and last is not None:
            flip = first > last
        elif first is not None:
            flip = first > 127
        elif last is not None:
            flip = last < 127
        else:
            flip = False
        return corners[::-1].copy() if flip else corners

    def detect(self, image: np.ndarray) -> Detection | None:
        """The board's inner corners, all of them (`corners`), or None."""
        corners = self.corners(image)
        return None if corners is None else Detection(np.arange(len(corners)), corners)

    def readings(self, detection: Detection) -> list[Detection]:
        """As found and, for a symmetric board, turned around (id k is then the last but k)."""
        if not self.symmetric:
            return [detection]
        return [detection, Detection(self.cols * self.rows - 1 - detection.ids, detection.pixels)]

    def image(self, px_per_m: float, margin_px: int = 0) -> np.ndarray:
        """The board, (cols + 1) x (rows + 1) squares, the one diagonally past id 0 dark.

        Id 0 is one square (and `margin_px`) from the top left, x to the right.
        """
        square = px_per_m * self.square_m
        width = int(round((self.cols + 1) * square)) + 2 * margin_px
        height = int(round((self.rows + 1) * square)) + 2 * margin_px
        v, u = np.mgrid[0:height, 0:width].astype(np.float64)
        i = np.floor((u - margin_px) / square)
        j = np.floor((v - margin_px) / square)
        inside = (i >= 0) & (i <= self.cols) & (j >= 0) & (j <= self.rows)
        out = np.full((height, width), 255, np.uint8)
        out[inside & ((i + j) % 2 == 0)] = 0
        return out

    def to_dict(self) -> dict[str, Any]:
        """{"kind": "checkerboard", "cols", "rows", "square_m"}."""
        return {"kind": self.kind, **asdict(self)}

    def _outer_brightness(self, gray: np.ndarray, corners: np.ndarray, end: int) -> float | None:
        """The brightness of the square diagonally outside the grid at its first (`end` 0) or last (-1) corner."""
        grid = corners.reshape(self.rows, self.cols, 2)
        corner, inner = (grid[0, 0], grid[1, 1]) if end == 0 else (grid[-1, -1], grid[-2, -2])
        u, v = corner + 0.5 * (corner - inner)
        height, width = gray.shape
        if not (2 <= u < width - 2 and 2 <= v < height - 2):
            return None
        return float(gray[int(v) - 1 : int(v) + 2, int(u) - 1 : int(u) + 2].mean())


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

    def readings(self, detection: Detection) -> list[Detection]:
        """One: the markers say which corner is which."""
        return [detection]

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


# ---- reading, saving, the pose --------------------------------------------------------------------------------

KINDS: dict[str, type] = {"checkerboard": Checkerboard, "charuco": CharucoBoard}


def board_from_dict(data: dict[str, Any]) -> Board:
    """A board from `to_dict`'s dict (a dict without "kind" is a checkerboard: the older sessions')."""
    data = dict(data)
    kind = data.pop("kind", "checkerboard")
    if kind not in KINDS:
        raise ValueError(f"unknown board kind {kind!r}; known: {', '.join(KINDS)}")
    return KINDS[kind](**data)


def parse_board(text: str) -> Board:
    """A board from text.

    "checkerboard:COLSxROWS:SQUARE_M" (inner corners) or "charuco:COLSxROWS:SQUARE_M:MARKER_M[:DICTIONARY]" (squares).
    """
    kind, *fields = text.strip().split(":")
    try:
        cols, rows = (int(v) for v in fields[0].lower().split("x"))
        if kind == "checkerboard" and len(fields) == 2:
            if cols < 3 or rows < 3:
                raise ValueError("a checkerboard needs at least 3 x 3 inner corners")
            return Checkerboard(cols, rows, float(fields[1]))
        if kind == "charuco" and len(fields) in (3, 4):
            return CharucoBoard(cols, rows, float(fields[1]), float(fields[2]), *fields[3:])
    except (IndexError, ValueError) as e:
        raise ValueError(f"cannot read the board {text!r}: {e}") from e
    raise ValueError(
        f"cannot read the board {text!r}: checkerboard:8x6:0.025 or charuco:12x9:0.02:0.015[:DICT_5X5_100]"
    )


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
