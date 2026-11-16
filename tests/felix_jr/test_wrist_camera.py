"""`calibration.wrist_camera`: the ChArUco board found in a picture (whole or in part), the lens's views chosen out of
a stream of frames, and the lens fitted, on synthetic views (the truth is known). The session against the fake robot:
`smoke_tests/test_wrist_camera_session_smoke.py`."""

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("scipy")

from lerobot.felix_jr.calibration.boards import CharucoBoard, Detection  # noqa: E402
from lerobot.felix_jr.calibration.wrist_camera import (  # noqa: E402
    COVERAGE_CELLS,
    ArmRecord,
    LensViews,
    calibrate_intrinsics,
    cells,
    fingertips_in_wrist_image,
    is_new_pose,
    lens_coverage_missing,
    lens_settled,
    sharpness,
    view_signature,
    with_lens,
    wrap,
)
from lerobot.felix_jr.calibration.xle_robot_model import (  # noqa: E402
    WristCamera,
    XleRobotModel,
    wrist_camera_link,
)

URDF = Path(__file__).parent / "reference" / "pen1_calibration_10312026.urdf"
BOARD = CharucoBoard(9, 7, 0.025, 0.018, "DICT_4X4_50")  # the printed one: 22.5 x 17.5 cm, 48 inner corners
CENTER = np.array([4.5 * 0.025, 3.5 * 0.025, 0.0])  # the board's middle, in its frame
SIZE = (640, 480)
# the true camera: 640 x 480, about 77 deg across, no distortion; looking along the gripper's axis (the gripper link's
# -z), 4.5 cm to its side and 3 cm toward the tips
TRUE_CAMERA = WristCamera(640, 480, 400.0, 402.0, 322.0, 238.0, "rational", (0.0,) * 8)
GRIPPER_FROM_CAMERA = np.array(
    [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.045], [0.0, 0.0, -1.0, -0.03], [0.0, 0.0, 0.0, 1.0]]
)
PX_PER_M = 10_000.0  # the rendered board's texture: 250 px a square (as `test_boards`)
MARGIN_PX = 300  # its white border (the markers need a quiet zone)


def _pose(rotvec, translation) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = cv2.Rodrigues(np.asarray(rotvec, dtype=float))[0]
    pose[:3, 3] = translation
    return pose


def held(rotvec, middle) -> np.ndarray:
    """camera_from_board: the board turned by `rotvec` about its middle, its middle at `middle` (camera frame)."""
    pose = _pose(rotvec, middle)
    pose[:3, 3] -= pose[:3, :3] @ CENTER
    return pose


def render_board(
    camera: WristCamera, camera_from_board: np.ndarray, board: CharucoBoard = BOARD
) -> np.ndarray:
    """An RGB picture of the board (its white border around it) on a gray table, from `camera` (no distortion)."""
    # blurred before the warp, so its edges come out anti-aliased as a real camera's would (`test_boards`)
    texture = cv2.GaussianBlur(board.image(PX_PER_M, MARGIN_PX), (0, 0), PX_PER_M / 2000.0)
    board_from_texture = np.array(
        [[1 / PX_PER_M, 0, -MARGIN_PX / PX_PER_M], [0, 1 / PX_PER_M, -MARGIN_PX / PX_PER_M], [0, 0, 1]]
    )
    plane = camera_from_board[:3, [0, 1, 3]]  # the board's z = 0 plane, into the camera
    homography = camera.matrix() @ plane @ board_from_texture
    gray = cv2.warpPerspective(
        texture, homography, (camera.width, camera.height), flags=cv2.INTER_LINEAR, borderValue=90
    )
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def projected(
    camera_from_board: np.ndarray,
    rng: np.random.Generator | None = None,
    noise_px: float = 0.2,
    margin: float = 5.0,
) -> Detection | None:
    """The corners the camera sees of a board (those in the picture), with noise: a detection without rendering."""
    points = BOARD.points()
    uv = TRUE_CAMERA.project(points @ camera_from_board[:3, :3].T + camera_from_board[:3, 3])
    inside = np.isfinite(uv).all(axis=1) & (uv[:, 0] > margin) & (uv[:, 0] < SIZE[0] - margin)
    inside &= (uv[:, 1] > margin) & (uv[:, 1] < SIZE[1] - margin)
    ids = np.flatnonzero(inside)
    if len(ids) < 6:
        return None
    noise = 0.0 if rng is None else rng.normal(0.0, noise_px, (len(ids), 2))
    return Detection(ids, uv[ids] + noise)


def synthetic_views(n: int, seed: int = 0, partial: bool = True) -> list[Detection]:
    """`n` views of the board at varied poses, some running off the picture (with `partial`)."""
    rng = np.random.default_rng(seed)
    views = []
    while len(views) < n:
        spread = [0.12, 0.09] if partial else [0.04, 0.03]
        middle = [
            rng.uniform(-spread[0], spread[0]),
            rng.uniform(-spread[1], spread[1]),
            rng.uniform(0.25, 0.45),
        ]
        view = projected(held(rng.uniform(-0.5, 0.5, 3), middle), rng)
        if view is not None and len(view) >= (15 if partial else 48):
            views.append(view)
    return views


@pytest.fixture(scope="module")
def model():
    return XleRobotModel.load(URDF)


TABLE_BELOW_BASE_M = 0.06  # the table under the arms' bases, as on the robot
REST = {
    "shoulder_pan.pos": 0.0,
    "shoulder_lift.pos": -20.0,
    "elbow_flex.pos": 40.0,
    "wrist_flex.pos": 60.0,
    "wrist_roll.pos": 0.0,
}


def start_joints(model: XleRobotModel, arm: str) -> tuple[dict[str, float], np.ndarray, float]:
    """The session's start pose, chosen as `WristCameraSession.go_to_start` does: the joints, the tip, the pitch.

    (For the wrist smoke tests, `test_wrist_align_smoke` and `test_wrist_scan_smoke`, which start from it.)
    """
    from types import SimpleNamespace

    from lerobot.felix_jr.calibration.wrist_camera import (
        START_AHEAD_M,
        START_HEIGHTS_M,
        START_JOINT_MARGIN_DEG,
        START_PITCHES_DEG,
    )
    from lerobot.felix_jr.harness.reach import ArmIK, solve_pose

    base = model.frame_pose(f"{arm}_base_link", model.joint_vector())[:3, 3]
    capture = SimpleNamespace(arms={arm: dict(REST)}, head_ticks=None)
    reachable = []
    for up in START_HEIGHTS_M:
        for pitch in START_PITCHES_DEG:
            target = np.array([base[0] - START_AHEAD_M, base[1], base[2] + up])
            poses = solve_pose(model, arm, capture, target, pitch, None, 0.0)
            if not poses.ok[0]:
                continue
            joints = dict(poses.joints[0])
            ik = ArmIK(model, arm, {arm: joints}, None, 0.0, 0.0)
            x = ik.radians(joints)
            margin = float(np.degrees(np.minimum(x - ik.lower, ik.upper - x)).min())
            reachable.append((margin, joints, target, pitch))
            if margin >= START_JOINT_MARGIN_DEG:
                return joints | {"gripper.pos": 1.0}, target, pitch
    assert reachable, "no start pose"
    _, joints, target, pitch = max(reachable, key=lambda r: r[0])
    return joints | {"gripper.pos": 1.0}, target, pitch


# ---- the board in a picture ---------------------------------------------------------------------------------


def test_the_board_is_found_whole_and_in_part():
    """Whole, every corner where it projects; half off the picture's edge, the corners that show, named right."""
    whole = held([0.2, -0.15, 0.1], [0.0, 0.0, 0.32])
    found = BOARD.detect(render_board(TRUE_CAMERA, whole))
    truth = projected(whole)
    assert found is not None and truth is not None and len(found) == 48
    np.testing.assert_array_equal(found.ids, truth.ids)
    assert np.linalg.norm(found.pixels - truth.pixels, axis=1).max() < 0.3
    off = held([0.1, 0.2, -0.3], [0.2, -0.1, 0.3])  # its middle near the picture's top right corner
    found, truth = BOARD.detect(render_board(TRUE_CAMERA, off)), projected(off)
    assert found is not None and truth is not None and 6 <= len(found) < 48
    assert set(found.ids) <= set(truth.ids)
    full = TRUE_CAMERA.project(BOARD.points(found.ids) @ off[:3, :3].T + off[:3, 3])
    assert np.linalg.norm(found.pixels - full, axis=1).max() < 0.3


def test_the_signature_is_the_whole_boards_from_any_part_of_it():
    pose = held([0.3, -0.2, 0.4], [0.02, 0.01, 0.33])
    full = projected(pose)
    assert full is not None and len(full) == 48
    part = Detection(full.ids[:20], full.pixels[:20])  # the first rows only
    a, b = view_signature(full, SIZE, BOARD), view_signature(part, SIZE, BOARD)
    assert np.abs(a[:5] - b[:5]).max() < 1e-6 and abs(a[5] - b[5]) < 1e-4


def test_new_poses_are_told_apart():
    a = view_signature(projected(held([0, 0, 0], [0.0, 0.0, 0.3])), SIZE, BOARD)
    nudged = view_signature(projected(held([0, 0, 0], [0.001, 0.0, 0.3])), SIZE, BOARD)
    moved = view_signature(projected(held([0, 0, 0], [0.05, 0.0, 0.3])), SIZE, BOARD)
    turned = view_signature(projected(held([0, 0, np.radians(30)], [0.0, 0.0, 0.3])), SIZE, BOARD)
    tilted = view_signature(projected(held([0.5, 0, 0], [0.0, 0.0, 0.3])), SIZE, BOARD)
    assert not is_new_pose(nudged, [a])
    assert is_new_pose(moved, [a]) and is_new_pose(turned, [a]) and is_new_pose(tilted, [a])
    assert abs(turned[5] - a[5]) > 25


# ---- choosing the lens's views ----------------------------------------------------------------------------------


def stream(poses, seed: int = 0) -> LensViews:
    """Offer one frame per pose (the board's corners, with noise) to the lens's views."""
    rng = np.random.default_rng(seed)
    views = LensViews(BOARD, SIZE)
    for pose in poses:
        views.offer(projected(pose, rng))
    return views


def test_a_board_held_still_is_kept_once():
    pose = held([0.2, 0.1, 0.0], [0.0, 0.0, 0.3])
    views = stream([pose] * 200)
    assert len(views.views) == 1


def test_a_moving_board_is_not_kept():
    """Moved ~6 px between frames (a hand sweeping it): its corners blur and lag, nothing kept."""
    views = stream([held([0.2, 0.1, 0.0], [-0.1 + 0.005 * i, 0.0, 0.3]) for i in range(40)])
    assert views.views == []
    kept, why = views.offer(projected(held([0.2, 0.1, 0.0], [0.2, 0.0, 0.3])))
    assert not kept and "still" in why


def test_a_slow_pass_keeps_a_handful_not_hundreds():
    """The board moved slowly (under a pixel a frame) across the picture and tilted as it goes, 600 frames: a few
    dozen views at most, the picture's width covered."""
    n = 600
    poses = [
        held(
            [0.5 * np.sin(i / 90), 0.4 * np.cos(i / 120), 0.2 * np.sin(i / 200)],
            [-0.14 + 0.28 * i / n, 0.0, 0.32],
        )
        for i in range(n)
    ]
    views = stream(poses)
    assert 5 <= len(views.views) <= 40, len(views.views)
    columns = {u for u, _ in views.covered}
    assert columns == set(range(COVERAGE_CELLS[0]))


def test_partial_boards_reach_the_picture_corners():
    """A board pushed into each corner of the picture, running off its edges: kept for the cells only it covers."""
    views = LensViews(BOARD, SIZE)
    first = projected(held([0, 0, 0], [0.0, 0.0, 0.35]))
    views.offer(first)
    views.offer(first)  # held still: kept
    assert len(views.views) == 1
    for sx, sy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
        corner = projected(held([0, 0, 0], [sx * 0.2, sy * 0.15, 0.35]))
        assert corner is not None and 15 <= len(corner) < 48
        views.offer(corner)
        kept, why = views.offer(corner)
        assert kept and "new cells" in why, why
    cols, rows = COVERAGE_CELLS
    assert {(0, 0), (cols - 1, 0), (0, rows - 1), (cols - 1, rows - 1)} <= views.covered


def test_too_few_corners_are_not_kept_and_the_views_stop_at_the_cap(monkeypatch):
    views = LensViews(BOARD, SIZE)
    sliver = projected(held([0, 0, 0], [0.27, 0.2, 0.35]))
    assert sliver is not None and len(sliver) < 15
    views.offer(sliver)
    kept, why = views.offer(sliver)
    assert not kept and "too few" in why
    monkeypatch.setattr("lerobot.felix_jr.calibration.wrist_camera.MAX_LENS_VIEWS", 3)
    rng = np.random.default_rng(4)
    for i in range(10):
        pose = held(rng.uniform(-0.5, 0.5, 3), [0.0, 0.0, 0.25 + 0.02 * i])
        views.offer(projected(pose))
        views.offer(projected(pose))
    assert len(views.views) == 3
    kept, why = views.offer(projected(held([0, 0, 0], [0.0, 0.0, 0.3])))
    assert not kept and "enough" in why


def test_the_last_view_is_dropped_with_its_cells():
    views = LensViews(BOARD, SIZE)
    for middle in ([0.0, 0.0, 0.35], [0.2, 0.15, 0.35]):
        view = projected(held([0, 0, 0], middle))
        views.offer(view)
        views.offer(view)
    assert len(views.views) == 2
    before = set(views.covered)
    views.drop_last()
    assert len(views.views) == 1 and views.covered == cells(views.views[0].pixels, SIZE) < before


# ---- the lens --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("lens", ["rational", "fisheye"])
def test_the_lens_is_fitted_from_whole_and_partial_boards(lens):
    views = synthetic_views(20)
    assert any(len(v) < 48 for v in views) and any(len(v) == 48 for v in views)
    camera, errors = calibrate_intrinsics(views, BOARD, SIZE, lens)
    assert camera.model == lens and camera.n_views == 20
    assert camera.fx == pytest.approx(400.0, rel=0.01) and camera.fy == pytest.approx(402.0, rel=0.01)
    assert camera.ppx == pytest.approx(322.0, abs=3) and camera.ppy == pytest.approx(238.0, abs=3)
    assert camera.rms_px < 0.5 and errors.max() < 0.6


def test_flat_views_at_one_distance_are_not_enough_for_the_lens():
    """wrist_cams5 (2026-11-07): 47 views of the board flat on the table left the focal length free."""

    def signature(rotvec, middle):
        return view_signature(projected(held(rotvec, middle)), SIZE, BOARD)

    flat = [signature([0, 0, a], [x, 0.0, 0.3]) for a, x in [(0, 0), (0.5, 0.02), (1.0, -0.02), (2.0, 0.01)]]
    assert set(lens_coverage_missing(flat)) == {
        "with its top edge nearer",
        "with its bottom edge nearer",
        "with its left edge nearer",
        "with its right edge nearer",
        "nearer and farther",
    }
    tilted = flat + [
        signature(r, c)
        for r, c in [
            ([0.7, 0, 0], [0, 0, 0.25]),
            ([-0.7, 0, 0], [0, 0, 0.25]),
            ([0, 0.7, 0], [0, 0, 0.42]),
            ([0, -0.7, 0], [0, 0, 0.42]),
        ]
    ]
    assert lens_coverage_missing(tilted) == []


def test_the_lens_is_settled_by_varied_views_and_not_by_flat_ones():
    focal, spread = lens_settled(synthetic_views(16), BOARD, SIZE)
    assert focal == pytest.approx(401.0, rel=0.02) and spread < 0.03
    rng = np.random.default_rng(2)
    flat = []
    while (
        len(flat) < 16
    ):  # the board flat to the camera at one distance, moved and turned about (wrist_cams5)
        view = projected(
            held([0.0, 0.0, rng.uniform(-3, 3)], [rng.uniform(-0.03, 0.03), rng.uniform(-0.02, 0.02), 0.32]),
            rng,
        )
        if view is not None and len(view) >= 30:
            flat.append(view)
    _, flat_spread = lens_settled(flat, BOARD, SIZE)
    assert flat_spread > 0.03


def test_sharpness_drops_with_blur():
    sharp = render_board(TRUE_CAMERA, held([0.2, -0.15, 0.05], [0.0, 0.0, 0.3]))
    blurred = cv2.GaussianBlur(sharp, (0, 0), 6)
    found = BOARD.detect(sharp)
    assert found is not None
    assert sharpness(sharp, found.pixels) > 10 * sharpness(blurred, found.pixels)
    assert sharpness(sharp) > 10 * sharpness(blurred)  # the middle of the picture, without the board


# ---- saving -------------------------------------------------------------------------------------------------


def _with_pose(camera: WristCamera) -> WristCamera:
    return replace(camera, gripper_from_camera=tuple(map(tuple, GRIPPER_FROM_CAMERA)))


def test_a_record_is_saved_and_read_back_and_a_checkerboards_is_refused(tmp_path):
    views = synthetic_views(3)
    record = ArmRecord("right", BOARD, SIZE, "rational", {"gripper": {"range_min": 1}}, views, 123.0)
    record.save(tmp_path)
    back = ArmRecord.load(tmp_path / "right_views.json")
    assert back.board == BOARD and back.size == SIZE and back.focus_sharpness == 123.0
    for a, b in zip(back.intrinsic_views, views, strict=True):
        np.testing.assert_array_equal(a.ids, b.ids)
        np.testing.assert_allclose(a.pixels, b.pixels)
    old = (tmp_path / "right_views.json").read_text().replace('"kind": "charuco", ', "")
    (tmp_path / "left_views.json").write_text(
        old
    )  # a board without a kind: a checkerboard, before 2026-11-16
    with pytest.raises(ValueError, match="no longer read"):
        ArmRecord.load(tmp_path / "left_views.json")


def test_a_new_lens_keeps_the_cameras_pose_on_the_gripper(model):
    posed = model.with_wrist_camera("right", _with_pose(TRUE_CAMERA))
    lens = replace(TRUE_CAMERA, fx=410.0)
    kept = with_lens(posed, "right", lens).wrist_cameras["right"]
    assert kept.fx == 410.0 and np.allclose(kept.gripper_from_camera, GRIPPER_FROM_CAMERA)
    fresh = with_lens(model, "left", lens).wrist_cameras["left"]
    assert fresh.gripper_from_camera is None  # none to keep: the joint session needs a start pose


def test_the_model_carries_the_wrist_camera(model, tmp_path):
    camera = _with_pose(TRUE_CAMERA)
    with_camera = model.with_wrist_camera("right", camera)
    assert model.wrist_cameras == {}  # a copy
    with_camera.save(tmp_path / "c.urdf")
    loaded = XleRobotModel.load(tmp_path / "c.urdf")
    assert loaded.wrist_cameras["right"] == camera
    q = model.joint_vector()
    np.testing.assert_allclose(
        loaded.wrist_camera_pose("right", q),
        model.frame_pose(wrist_camera_link("right"), q) @ GRIPPER_FROM_CAMERA,
    )
    with pytest.raises(ValueError, match="no calibrated left"):
        loaded.wrist_camera_pose("left", q)
    with pytest.raises(ValueError, match="unknown lens"):
        model.with_wrist_camera("left", WristCamera(1, 1, 1, 1, 0, 0, "pinhole", ()))
    arm_pose = {"gripper.pos": 1.0} | dict.fromkeys(
        ("shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos"), 0.0
    )
    tips = fingertips_in_wrist_image(loaded, "right", camera, arm_pose)
    assert (
        tips.shape == (2, 2) and np.isfinite(tips).all()
    )  # the camera looks along the jaws: it sees the tips


def test_long_status_lines_wrap_inside_the_picture():
    line = "c: fit the lens | like a kept view: tilt, turn or move it | x x: skip the right camera, next the left"
    pieces = wrap([line, "short"], 300)
    assert len(pieces) > 2 and pieces[-1] == "short"
    assert all(cv2.getTextSize(p, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] <= 300 for p in pieces)
    assert " | ".join(pieces[:-1]).replace(" | ", " ").split() == line.replace(" | ", " ").split()
