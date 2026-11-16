"""`calibration.boards`: the ChArUco board found in a synthetic view, its points where the truth projects them.

A board's printable image, rendered finely (10 px per mm, so the pixel conventions of the printout do not matter), is
warped into a 640 x 480 pinhole camera looking at it from 30 cm, tilted; the points found are compared with the board's
points projected through the same camera.
"""

import cv2
import numpy as np
import pytest

from lerobot.felix_jr.calibration.boards import (
    MIN_POINTS,
    PRINTED_BOARD,
    CharucoBoard,
    Detection,
    board_from_dict,
    board_pose,
    parse_board,
)

K = np.array([[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]])
SIZE = (640, 480)
PX_PER_M = 10_000.0
MARGIN_PX = 400


def camera_from_board(board, distance=0.30, tilt_deg=(20.0, -15.0, 10.0)):
    """The board's center 30 cm ahead, the board turned by `tilt_deg` (about x, y, z)."""
    center = board.points().mean(axis=0)
    rotation = cv2.Rodrigues(np.radians(np.array(tilt_deg)))[0]
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = np.array([0.0, 0.0, distance]) - rotation @ center
    return pose


def image_from_board_frame(board):
    """The printable image's pixel of a board-frame point (x, y): a similarity, found from the image itself.

    ChArUco's frame starts at the board's outer corner, at the margin.
    """
    return np.array([[PX_PER_M, 0.0, MARGIN_PX], [0.0, PX_PER_M, MARGIN_PX], [0.0, 0.0, 1.0]])


def render(board, pose, occlude=None):
    """The board seen by the camera at `pose` (camera_from_board): an RGB image on gray; `occlude` a pixel box."""
    printed = board.image(PX_PER_M, MARGIN_PX)
    # board frame (x, y, 1) -> camera pixels: K [r1 r2 t]; the printout's pixels -> board frame: inverse similarity
    board_to_pixels = K @ np.column_stack([pose[:3, 0], pose[:3, 1], pose[:3, 3]])
    homography = board_to_pixels @ np.linalg.inv(image_from_board_frame(board))
    view = cv2.warpPerspective(  # the printout is blurred first (`blur_printouts`): no aliasing
        printed, homography, SIZE, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=128
    )
    if view.std() < 1:
        raise AssertionError("the board is not in view")
    if occlude is not None:
        u0, v0, u1, v1 = occlude
        view[v0:v1, u0:u1] = 128
    return cv2.cvtColor(view, cv2.COLOR_GRAY2RGB)


def projected(board, pose, ids):
    points = board.points(ids)
    camera = points @ pose[:3, :3].T + pose[:3, 3]
    uv = camera @ K.T
    return uv[:, :2] / uv[:, 2:]


@pytest.fixture(autouse=True)
def blur_printouts(monkeypatch):
    """Blur the fine printout before the warp so its edges come out anti-aliased (as a real camera's would)."""
    original = CharucoBoard.image

    def blurred(self, px_per_m, margin_px=0):
        return cv2.GaussianBlur(original(self, px_per_m, margin_px), (0, 0), PX_PER_M / 2000.0)

    monkeypatch.setattr(CharucoBoard, "image", blurred)


def test_a_charuco_board_half_hidden_gives_the_corners_it_shows():
    board = CharucoBoard(12, 9, 0.02, 0.015)
    pose = camera_from_board(board)
    whole = board.detect(render(board, pose))
    assert whole is not None and len(whole) == 88
    half = board.detect(render(board, pose, occlude=(320, 0, 640, 480)))  # the right half covered (an arm)
    assert half is not None and MIN_POINTS <= len(half) < 70
    assert set(half.ids) <= set(whole.ids)
    errors = np.linalg.norm(half.pixels - projected(board, pose, half.ids), axis=1)
    assert errors.max() < 0.3, errors.max()
    assert (half.pixels[:, 0] < 330).all()  # only what shows


def test_too_little_of_a_board_is_no_detection():
    board = CharucoBoard(12, 9, 0.02, 0.015)
    pose = camera_from_board(board)
    assert board.detect(render(board, pose, occlude=(0, 0, 600, 480))) is None
    assert board.detect(np.full((480, 640, 3), 128, np.uint8)) is None


def test_the_pose_from_a_detection():
    board = CharucoBoard(12, 9, 0.02, 0.015)
    pose = camera_from_board(board)
    found = board.detect(render(board, pose, occlude=(400, 0, 640, 480)))
    solved = board_pose(board, found, K)
    assert np.linalg.norm(solved[:3, 3] - pose[:3, 3]) < 0.001
    turn = cv2.Rodrigues(solved[:3, :3] @ pose[:3, :3].T)[0]
    assert np.degrees(np.linalg.norm(turn)) < 0.2


def test_boards_are_read_from_text_and_saved():
    assert parse_board("charuco:12x9:0.02:0.015") == CharucoBoard(12, 9, 0.02, 0.015)
    assert parse_board("charuco:5x4:0.03:0.02:DICT_4X4_50") == CharucoBoard(5, 4, 0.03, 0.02, "DICT_4X4_50")
    assert parse_board(PRINTED_BOARD) == CharucoBoard(9, 7, 0.025, 0.018, "DICT_4X4_50")
    board = CharucoBoard(12, 9, 0.02, 0.015)
    assert board_from_dict(board.to_dict()) == board
    for old in (
        {"cols": 8, "rows": 6, "square_m": 0.025},
        {"kind": "checkerboard", "cols": 8, "rows": 6, "square_m": 0.025},
    ):
        with pytest.raises(
            ValueError, match="no longer read"
        ):  # the checkerboard's sessions (before 2026-11-16)
            board_from_dict(old)
    for bad in ("checkerboard:8x6:0.025", "charuco:12x9:0.02", "hexagons:3x3:0.1"):
        with pytest.raises(ValueError, match="cannot read the board"):
            parse_board(bad)
    with pytest.raises(ValueError, match="smaller than the squares"):
        CharucoBoard(12, 9, 0.02, 0.025)
    with pytest.raises(ValueError, match="too few markers"):
        CharucoBoard(12, 9, 0.02, 0.015, "DICT_4X4_50")


def test_a_detection_is_its_points():
    found = Detection(np.array([3, 7]), np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert len(found) == 2
    board = CharucoBoard(5, 4, 0.01, 0.007, "DICT_4X4_50")  # 4 x 3 inner corners, from (0.01, 0.01)
    np.testing.assert_allclose(board.points(found.ids)[:, :2], [[0.04, 0.01], [0.04, 0.02]])
