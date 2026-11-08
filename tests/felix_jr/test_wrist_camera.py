"""`calibration.wrist_camera`: the board, the lens and hand-eye fits on synthetic views, and a session against the
fake robot whose wrist cameras render a checkerboard on the table from the arm's pose (the truth is known)."""

from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("scipy")

from lerobot.felix_jr.calibration.wrist_camera import (  # noqa: E402
    ArmView,
    Board,
    board_pose,
    calibrate_extrinsics,
    calibrate_intrinsics,
    find_board,
    fingertips_in_wrist_image,
    fit_extrinsics,
    is_new_view,
    lens_coverage_missing,
    lens_settled,
    parse_pattern,
    pattern_hint,
    rotation_spread,
    sharpness,
    view_signature,
)
from lerobot.felix_jr.calibration.xle_robot_model import (  # noqa: E402
    WristCamera,
    XleRobotModel,
    wrist_camera_link,
)

URDF = Path(__file__).parent / "reference" / "pen1_calibration_10312026.urdf"
BOARD = Board(9, 6, 0.018)  # 14 cm x 9 cm of corners: in view from ~20 cm
# the true camera: 640 x 480, about 77 deg across, no distortion; looking along the gripper's axis (the gripper link's
# -z), 4.5 cm to its side and 3 cm toward the tips
TRUE_CAMERA = WristCamera(640, 480, 400.0, 402.0, 322.0, 238.0, "rational", (0.0,) * 8)
GRIPPER_FROM_CAMERA = np.array(
    [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.045], [0.0, 0.0, -1.0, -0.03], [0.0, 0.0, 0.0, 1.0]]
)


def _pose(rotvec, translation) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = cv2.Rodrigues(np.asarray(rotvec, dtype=float))[0]
    pose[:3, 3] = translation
    return pose


def render_board(camera: WristCamera, camera_from_board: np.ndarray, board: Board = BOARD) -> np.ndarray:
    """An RGB picture of the board (a white margin of one square around the squares) seen from `camera`."""
    image = np.full((camera.height, camera.width, 3), 90, np.uint8)  # the table
    s = board.square_m

    def pixels(points: np.ndarray) -> np.ndarray:
        cam = points @ camera_from_board[:3, :3].T + camera_from_board[:3, 3]
        return np.round(camera.project(cam) * 16).astype(np.int32)  # 4 bits of subpixel

    margin = np.array(
        [
            [-2 * s, -2 * s, 0],
            [(board.cols + 1) * s, -2 * s, 0],
            [(board.cols + 1) * s, (board.rows + 1) * s, 0],
            [-2 * s, (board.rows + 1) * s, 0],
        ]
    )
    cv2.fillConvexPoly(image, pixels(margin), (255, 255, 255), cv2.LINE_AA, 4)
    for i in range(-1, board.cols):
        for j in range(-1, board.rows):
            if (i + j) % 2 == 0:
                square = np.array([[i, j, 0], [i + 1, j, 0], [i + 1, j + 1, 0], [i, j + 1, 0]], float) * s
                cv2.fillConvexPoly(image, pixels(square), (0, 0, 0), cv2.LINE_AA, 4)
    return image


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
    """The session's start pose, chosen as `WristCameraSession.go_to_start` does: the joints, the tip, the pitch."""
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


def board_under(model: XleRobotModel, arm: str) -> np.ndarray:
    """root_from_board: the board flat on the table, centered where the true camera looks from the start pose."""
    joints, _, _ = start_joints(model, arm)
    truth = model.with_wrist_camera(arm, _with_pose(TRUE_CAMERA))
    camera = truth.wrist_camera_pose(arm, model.joint_vector({arm: joints}))
    table = model.frame_pose(f"{arm}_base_link", model.joint_vector())[2, 3] - TABLE_BELOW_BASE_M
    along = (table - camera[2, 3]) / camera[2, 2]  # the optical axis down to the table
    center = camera[:3, 3] + along * camera[:3, 2]
    return _pose([0.0, 0.0, 0.3], center - [4 * BOARD.square_m, 2.5 * BOARD.square_m, 0.0])


# ---- pure parts ---------------------------------------------------------------------------------------------


def test_the_board_is_found_in_opencvs_order():
    camera_from_board = _pose([0.1, -0.15, 0.05], [-0.1, -0.06, 0.3])
    image = render_board(TRUE_CAMERA, camera_from_board)
    corners = find_board(image, BOARD)
    assert corners is not None and corners.shape == (54, 2)
    truth = TRUE_CAMERA.project(BOARD.points() @ camera_from_board[:3, :3].T + camera_from_board[:3, 3])
    assert np.abs(corners - truth).max() < 0.5  # the same corners, in the same order
    np.testing.assert_allclose(board_pose(TRUE_CAMERA, corners, BOARD), camera_from_board, atol=5e-3)
    assert find_board(np.full((480, 640, 3), 128, np.uint8), BOARD) is None
    assert parse_pattern("9x6") == (9, 6)
    with pytest.raises(ValueError):
        parse_pattern("2x6")
    assert parse_pattern("8x6") == (
        8,
        6,
    )  # both even: turned around it looks the same (`orient_views` handles it)
    turned = _pose(
        [0.1, -0.15, 0.05 + np.pi], [0.1, 0.06, 0.3]
    )  # the board turned around: the same first corner
    corners = find_board(render_board(TRUE_CAMERA, turned), BOARD)
    truth = TRUE_CAMERA.project(BOARD.points() @ turned[:3, :3].T + turned[:3, 3])
    assert corners is not None and np.abs(corners - truth).max() < 0.5


def test_new_views_are_told_apart():
    a = view_signature(TRUE_CAMERA.project(BOARD.points() + [-0.1, -0.06, 0.3]), (640, 480), BOARD)
    nudged = view_signature(TRUE_CAMERA.project(BOARD.points() + [-0.099, -0.06, 0.3]), (640, 480), BOARD)
    moved = view_signature(TRUE_CAMERA.project(BOARD.points() + [-0.02, -0.06, 0.3]), (640, 480), BOARD)
    assert not is_new_view(nudged, [a]) and is_new_view(moved, [a])
    turn = cv2.Rodrigues(np.array([0.0, 0.0, np.radians(30)]))[0]
    turned = view_signature(
        TRUE_CAMERA.project((BOARD.points() - [0.08, 0.05, 0]) @ turn.T + [-0.02, -0.01, 0.3]),
        (640, 480),
        BOARD,
    )
    assert abs(turned[5] - a[5]) > 25 and is_new_view(turned, [a])  # turned in the picture: a new view


def synthetic_views(n: int, seed: int = 0, noise_px: float = 0.2) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    views = []
    while len(views) < n:
        pose = _pose(
            rng.uniform(-0.5, 0.5, 3),
            [rng.uniform(-0.15, 0.0), rng.uniform(-0.1, 0.0), rng.uniform(0.25, 0.45)],
        )
        uv = TRUE_CAMERA.project(BOARD.points() @ pose[:3, :3].T + pose[:3, 3])
        if np.isfinite(uv).all() and (uv > 5).all() and (uv[:, 0] < 635).all() and (uv[:, 1] < 475).all():
            views.append(uv + rng.normal(0, noise_px, uv.shape))
    return views


@pytest.mark.parametrize("lens", ["rational", "fisheye"])
def test_the_lens_is_fitted(lens):
    camera, errors = calibrate_intrinsics(synthetic_views(15), BOARD, (640, 480), lens)
    assert camera.model == lens and camera.n_views == 15
    assert camera.fx == pytest.approx(400.0, rel=0.01) and camera.fy == pytest.approx(402.0, rel=0.01)
    assert camera.ppx == pytest.approx(322.0, abs=3) and camera.ppy == pytest.approx(238.0, abs=3)
    assert camera.rms_px < 0.5 and errors.max() < 0.6


def test_rotation_spread():
    turns = [
        cv2.Rodrigues(np.array(v, float))[0] for v in ([0, 0, 0], [0.4, 0, 0], [-0.4, 0, 0], [0, 0.4, 0])
    ]
    spread = rotation_spread(turns)
    assert spread[0] > 15 and spread[1] > 10 and spread[2] < 1e-6
    assert rotation_spread(turns[:1]).tolist() == [0.0, 0.0, 0.0]


def arm_views(
    model: XleRobotModel, arm: str, root_from_board: np.ndarray, noise_px: float = 0.2, board: Board = BOARD
) -> list[ArmView]:
    """The board's corners as the true camera sees them, from arm poses around the start pose, tilted and rolled."""
    from types import SimpleNamespace

    from lerobot.felix_jr.harness.reach import solve_pose

    rng = np.random.default_rng(1)
    with_camera = model.with_wrist_camera(arm, _with_pose(TRUE_CAMERA))
    joints, start, pitch0 = start_joints(model, arm)
    capture = SimpleNamespace(arms={arm: dict(REST)}, head_ticks=None)
    views = []
    offsets = zip(
        [0, 0.02, -0.02, 0.01, -0.01, 0.02, -0.02, 0, 0.01, -0.01, 0.02, -0.02, 0, 0.01],
        [0, 0.01, -0.01, 0.02, -0.02, -0.01, 0.01, 0.02, -0.02, 0, 0.02, -0.02, -0.01, 0.01],
        [0, 0, -0.01, 0.01, 0, -0.01, 0.01, 0, -0.01, 0.01, 0, 0, -0.01, 0.01],
        [0, 10, 0, 15, -10, 5, 20, 10, -5, 15, 0, 10, 20, -10],
        [0, 20, -20, 35, -35, 10, -10, 45, -45, 25, -25, 30, -30, 0],
        strict=True,
    )
    for dx, dy, dz, dpitch, roll in offsets:
        poses = solve_pose(
            model, arm, capture, start + [dx, dy, dz], pitch0 + dpitch, None, roll, init=joints
        )
        if not poses.ok[0]:
            continue
        arm_pose = dict(poses.joints[0]) | {"gripper.pos": 1.0}
        q = model.joint_vector({arm: arm_pose})
        camera_from_board = np.linalg.inv(with_camera.wrist_camera_pose(arm, q)) @ root_from_board
        uv = TRUE_CAMERA.project(board.points() @ camera_from_board[:3, :3].T + camera_from_board[:3, 3])
        if np.isfinite(uv).all() and (uv > 0).all() and (uv[:, 0] < 640).all() and (uv[:, 1] < 480).all():
            views.append(ArmView(arm_pose, (uv + rng.normal(0, noise_px, uv.shape)).tolist()))
    return views


def _with_pose(camera: WristCamera) -> WristCamera:
    from dataclasses import replace

    return replace(camera, gripper_from_camera=tuple(map(tuple, GRIPPER_FROM_CAMERA)))


def test_the_camera_pose_on_the_gripper_is_fitted(model):
    root_from_board = board_under(model, "right")
    views = arm_views(model, "right", root_from_board)
    assert len(views) >= 8
    gripper_from_camera, fitted_board, errors = calibrate_extrinsics(
        model, "right", TRUE_CAMERA, views, BOARD
    )
    np.testing.assert_allclose(gripper_from_camera[:3, 3], GRIPPER_FROM_CAMERA[:3, 3], atol=0.002)
    turn = np.degrees(
        np.linalg.norm(cv2.Rodrigues(gripper_from_camera[:3, :3] @ GRIPPER_FROM_CAMERA[:3, :3].T)[0])
    )
    assert turn < 0.5
    np.testing.assert_allclose(fitted_board[:3, 3], root_from_board[:3, 3], atol=0.003)
    assert errors.max() < 0.6


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
    tips = fingertips_in_wrist_image(
        loaded,
        "right",
        camera,
        {"gripper.pos": 1.0}
        | dict.fromkeys(
            ("shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos"),
            0.0,
        ),
    )
    assert (
        tips.shape == (2, 2) and np.isfinite(tips).all()
    )  # the camera looks along the jaws: it sees the tips


def test_sharpness_drops_with_blur():
    camera_from_board = _pose([0.1, -0.15, 0.05], [-0.1, -0.06, 0.3])
    sharp = render_board(TRUE_CAMERA, camera_from_board)
    blurred = cv2.GaussianBlur(sharp, (0, 0), 6)
    corners = find_board(sharp, BOARD)
    assert sharpness(sharp, corners) > 10 * sharpness(blurred, corners)
    assert sharpness(sharp) > 10 * sharpness(blurred)  # the middle of the picture, without the board


BOARD_8X6 = Board(8, 6, 0.018)  # both counts even: it looks the same turned around


def test_a_symmetric_board_read_from_either_end_still_gives_the_camera_pose(model):
    """The detector may start a symmetric board from either end; the arm's motion tells which (`orient_views`)."""
    root_from_board = board_under(model, "right")
    views = arm_views(model, "right", root_from_board, board=BOARD_8X6)
    assert len(views) >= 8
    rng = np.random.default_rng(5)
    flipped = [
        ArmView(v.arm_pose, np.asarray(v.corners)[::-1].tolist()) if i > 0 and rng.random() < 0.5 else v
        for i, v in enumerate(views)
    ]
    assert flipped != views
    gripper_from_camera, _, errors = calibrate_extrinsics(model, "right", TRUE_CAMERA, flipped, BOARD_8X6)
    np.testing.assert_allclose(gripper_from_camera[:3, 3], GRIPPER_FROM_CAMERA[:3, 3], atol=0.002)
    assert errors.max() < 0.6


def test_a_wrong_pattern_is_named():
    """An 8 x 6 board asked for as 9 x 6 (wrist_cams2): the status line says what it looks like."""
    camera_from_board = _pose([0.1, -0.15, 0.05], [-0.08, -0.05, 0.3])
    image = render_board(TRUE_CAMERA, camera_from_board, BOARD_8X6)
    hint = pattern_hint(image, Board(9, 6, 0.018))
    assert hint is not None and "8 x 6" in hint and "--pattern 8x6" in hint
    assert pattern_hint(image, BOARD_8X6) is None


def test_flat_views_at_one_distance_are_not_enough_for_the_lens():
    """wrist_cams5 (2026-11-07): 47 views of the board flat on the table left the focal length free."""

    def signature(rotvec, center):
        pose = _pose(rotvec, center)
        corners = TRUE_CAMERA.project((BOARD.points() - [0.07, 0.045, 0]) @ pose[:3, :3].T + pose[:3, 3])
        return view_signature(corners, (640, 480), BOARD)

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
            ([0.7, 0, 0], [0, 0, 0.2]),
            ([-0.7, 0, 0], [0, 0, 0.2]),
            ([0, 0.7, 0], [0, 0, 0.4]),
            ([0, -0.7, 0], [0, 0, 0.4]),
        ]
    ]
    assert lens_coverage_missing(tilted) == []


def test_the_lens_is_settled_by_varied_views_and_not_by_flat_ones():
    good = synthetic_views(16)
    focal, spread = lens_settled(good, BOARD, (640, 480))
    assert focal == pytest.approx(401.0, rel=0.02) and spread < 0.03
    rng = np.random.default_rng(2)
    flat = []
    for _ in range(16):  # the board flat to the camera at one distance, moved and turned about (wrist_cams5)
        pose = _pose([0.0, 0.0, rng.uniform(-3, 3)], [rng.uniform(-0.06, 0.0), rng.uniform(-0.05, 0.0), 0.3])
        flat.append(
            TRUE_CAMERA.project(BOARD.points() @ pose[:3, :3].T + pose[:3, 3]) + rng.normal(0, 0.2, (54, 2))
        )
    _, flat_spread = lens_settled(flat, BOARD, (640, 480))
    assert flat_spread > 0.03


def test_views_turned_about_one_axis_do_not_settle_the_pose(model):
    """calib_full3 (2026-11-07): the arm only panned between views. The views leave the camera's offset along the pan
    axis free: not settled, and it says to tilt and turn the gripper. Views tilted and turned settle it."""
    from types import SimpleNamespace

    from lerobot.felix_jr.harness.reach import solve_pose

    root_from_board = board_under(model, "right")
    good = fit_extrinsics(model, "right", TRUE_CAMERA, arm_views(model, "right", root_from_board), BOARD)
    assert good.settled and good.needs() == "" and np.nanmax(good.position_sd_m) < 0.002
    assert "camera at" in good.describe()
    joints, start, pitch = start_joints(model, "right")
    truth = model.with_wrist_camera("right", _with_pose(TRUE_CAMERA))
    capture = SimpleNamespace(arms={"right": dict(REST)}, head_ticks=None)
    panned = []
    for dx, dy in [
        (0, 0),
        (0, 0.03),
        (0, -0.03),
        (0.02, 0.02),
        (0.02, -0.02),
        (-0.02, 0.03),
        (-0.02, -0.03),
        (0.03, 0),
    ]:
        poses = solve_pose(model, "right", capture, start + [dx, dy, 0.0], pitch, None, 0.0, init=joints)
        arm_pose = dict(poses.joints[0]) | {"gripper.pos": 1.0}
        root_from_camera = truth.wrist_camera_pose("right", model.joint_vector({"right": arm_pose}))
        camera_from_board = np.linalg.inv(root_from_camera) @ root_from_board
        uv = TRUE_CAMERA.project(BOARD.points() @ camera_from_board[:3, :3].T + camera_from_board[:3, 3])
        panned.append(ArmView(arm_pose, uv.tolist()))
    one_axis = fit_extrinsics(model, "right", TRUE_CAMERA, panned, BOARD)
    assert one_axis.spread_deg[1] < 5.0 and not one_axis.settled
    assert "tilt" in one_axis.needs() and "turn" in one_axis.needs()


def test_long_status_lines_wrap_inside_the_picture():
    import cv2

    from lerobot.felix_jr.calibration.wrist_camera import wrap

    line = "n: next view (9/15) | tilt [ ] and turn , . the gripper | c: accept anyway, next the right wrist camera"
    pieces = wrap([line, "short"], 300)
    assert len(pieces) > 2 and pieces[-1] == "short"
    assert all(cv2.getTextSize(p, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] <= 300 for p in pieces)
    assert " | ".join(pieces[:-1]).replace(" | ", " ").split() == line.replace(" | ", " ").split()
