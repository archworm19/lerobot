"""`calibration.wrist_camera_board`: the plate in the model and the URDF, the corner fit, and a session against the
fake robot."""

import threading
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("scipy")

from lerobot.felix_jr.calibration.wrist_camera_board import (  # noqa: E402
    CORNERS,
    BoardSession,
    CornerClick,
    board_from_fit,
    board_parent,
    fit_corners,
    squareness,
)
from lerobot.felix_jr.calibration.xle_robot_model import (  # noqa: E402
    WristBoard,
    XleRobotModel,
    arm_record,
    wrist_board_link,
)
from lerobot.felix_jr.server.client import FelixClient  # noqa: E402
from lerobot.felix_jr.server.hardware import FakeHardware  # noqa: E402
from lerobot.felix_jr.server.server import RobotServer  # noqa: E402

URDF = Path(__file__).parent / "reference" / "pen1_calibration_10312026.urdf"
# a 4 cm square plate beside the gripper, in the gripper link's frame (m): corners 1, 2, 3 in turn
TRUE = np.array([[0.02, -0.03, 0.06], [0.02, -0.03, 0.02], [0.02, 0.01, 0.02]])
REST = {
    "shoulder_pan.pos": 2.154,
    "shoulder_lift.pos": -102.989,
    "elbow_flex.pos": 96.527,
    "wrist_flex.pos": 71.429,
    "wrist_roll.pos": -0.22,
}


@pytest.fixture(scope="module")
def model():
    return XleRobotModel.load(URDF)


def board(arm: str = "right") -> WristBoard:
    return WristBoard(board_parent(arm), tuple(map(tuple, TRUE)), away=(1.0, 0.0, 0.0))


def test_the_plate_is_a_box_on_its_corners():
    plate = board()
    corners = plate.face_corners()
    np.testing.assert_allclose(corners[3], TRUE[0] + TRUE[2] - TRUE[1])  # the fourth, opposite corner 2
    pose, size = plate.box()
    np.testing.assert_allclose(size, [0.04, 0.04, 0.01])
    np.testing.assert_allclose(pose[:3, :3] @ pose[:3, :3].T, np.eye(3), atol=1e-12)
    assert np.linalg.det(pose[:3, :3]) == pytest.approx(1.0)
    np.testing.assert_allclose(pose[:3, 2], [1.0, 0.0, 0.0], atol=1e-12)  # thick toward `away`
    np.testing.assert_allclose(pose[:3, 3], corners.mean(axis=0) + [0.005, 0.0, 0.0])
    assert squareness(plate) == pytest.approx((0.04, 0.04, 90.0))


def test_the_plate_goes_into_the_urdf_and_survives_save_and_load(model, tmp_path):
    with_plate = model.with_wrist_board("right", board())
    assert "right" not in model.wrist_boards  # a copy
    q = with_plate.joint_vector({"right": REST})
    np.testing.assert_allclose(with_plate.pinch_point("right", q), model.pinch_point("right", q))  # unchanged
    # the URDF's link and the record agree: the box's center is the link's origin
    link = with_plate.frame_pose(wrist_board_link("right"), q)
    corners = with_plate.board_corners("right", q)
    pose, _ = board().box()
    parent = with_plate.frame_pose(board_parent("right"), q)
    np.testing.assert_allclose(link[:3, 3], parent[:3, :3] @ pose[:3, 3] + parent[:3, 3], atol=1e-9)
    np.testing.assert_allclose(corners.mean(axis=0) + link[:3, 2] * 0.005, link[:3, 3], atol=1e-9)

    path = tmp_path / "calibration.urdf"
    with_plate.save(path)
    loaded = XleRobotModel.load(path)
    assert loaded.wrist_boards["right"] == with_plate.wrist_boards["right"]
    np.testing.assert_allclose(loaded.frame_pose(wrist_board_link("right"), q), link, atol=1e-9)
    again = loaded.with_wrist_board("right", board())  # replaces it, does not add a second
    assert again.urdf.count(f'name="{wrist_board_link("right")}"') == 1
    with pytest.raises(ValueError, match="no wrist camera plate for the left arm"):
        loaded.board_corners("left", q)


def synthetic_clicks(model: XleRobotModel, n_views: int, noise_px: float, seed: int = 0) -> list[CornerClick]:
    """Clicks where the true corners land, over random poses around rest, with depth."""
    rng = np.random.default_rng(seed)
    clicks = []
    for _ in range(n_views):
        pose = {k: v + rng.uniform(-15, 15) for k, v in REST.items()}
        q = model.joint_vector({"right": pose})
        frame = model.frame_pose(board_parent("right"), q)
        for corner in CORNERS:
            point = frame[:3, :3] @ TRUE[corner - 1] + frame[:3, 3]
            uv, z = model.project(point, q)
            pixel = uv[0] + rng.normal(0, noise_px, 2)
            clicks.append(
                CornerClick("right", corner, pose, {}, tuple(pixel), float(z[0]), index=len(clicks))
            )
    return clicks


def test_the_fit_finds_the_corners_from_clicks(model):
    clicks = synthetic_clicks(model, 8, noise_px=0.5)
    corners, errors = fit_corners(model, clicks)
    np.testing.assert_allclose(corners, TRUE, atol=0.002)
    assert np.sqrt(np.mean(errors**2)) < 1.5
    plate = board_from_fit(model, clicks, corners, errors)
    edge1, edge2, angle = squareness(plate)
    assert edge1 == pytest.approx(0.04, abs=0.002) and edge2 == pytest.approx(0.04, abs=0.002)
    assert angle == pytest.approx(90.0, abs=3.0)
    assert plate.n_clicks == 24 and plate.rms_px == pytest.approx(np.sqrt(np.mean(errors**2)))
    # thick away from the camera: the camera is on the other side of the face
    q = model.joint_vector({"right": REST})
    frame = model.frame_pose(board_parent("right"), q)
    camera = frame[:3, :3].T @ (model.camera_pose(q)[:3, 3] - frame[:3, 3])
    assert (camera - TRUE.mean(axis=0)) @ np.asarray(plate.away) < 0


def test_the_fit_works_without_depth(model):
    clicks = synthetic_clicks(model, 10, noise_px=0.3, seed=1)
    for click in clicks:
        click.depth_m = None
    corners, _ = fit_corners(model, clicks, start=TRUE + 0.01)  # without depth, from a rough start
    np.testing.assert_allclose(corners, TRUE, atol=0.003)


# ---- a session against the fake robot -------------------------------------------------------------------------

FAKE_INTRINSICS = {"fx": 200.0, "fy": 200.0, "ppx": 160.0, "ppy": 120.0, "width": 320, "height": 240}
SIGNS = {"head_neck": -1, "head_eyes": 1}


@pytest.fixture
def robot(tmp_path):
    fake = FakeHardware(arm_time_scale=0.02, head_range_path=tmp_path / "range.json", session_timeout_s=5.0)
    server = RobotServer(fake)
    address = server.bind("tcp://127.0.0.1:*")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    with FelixClient(address, timeout_s=5.0) as client:
        for name in SIGNS:
            client.head_record(name, "home")
        yield fake, client
    server.shutdown()
    thread.join(timeout=3.0)
    server.close()


def calibration_for(client, path: Path) -> Path:
    """A calibration made for the fake robot (its head homes, looking down at the arms), saved at `path`."""
    refs = {name: servo["home"] for name, servo in client.head_state().items()}
    model = XleRobotModel.from_urdf(
        head_signs=SIGNS,
        head_refs=refs,
        intrinsics=FAKE_INTRINSICS,
        arm_calibration={"left": arm_record(client.arm_calibration()["left_arm"])},
    )
    model.set_estimates({"zero:head_tilt_joint": np.radians(33.0)})
    model.save(path)
    return path


def click_true_corners(session: BoardSession, plate: np.ndarray) -> None:
    """Click each corner where the plate `plate` (the parent's frame) puts it in the current frame."""
    q = session.current_q()
    frame = session.model.frame_pose(board_parent(session.arm), q)
    for corner in CORNERS:
        uv, _ = session.model.project(frame[:3, :3] @ plate[corner - 1] + frame[:3, 3], q)
        session.click(*uv[0])


def test_a_session_clicks_the_corners_view_by_view_and_writes_the_plate(robot, tmp_path):
    fake, client = robot
    start = calibration_for(client, tmp_path / "start.urdf")
    out = tmp_path / "board"
    with client.session():
        shown: list[np.ndarray] = []
        session = BoardSession(client, start, "left", out, shown.append, n_targets=8, floor_z=0.5)
        assert session.model.visible_side["left"] == "negative"  # views from the plate's side
        assert session.calibration.visible_side["left"] == "positive"  # what is written keeps its own
        plate = TRUE * [1, 1, 1]
        views = 0
        while views < 3 and session.next_pose():
            click_true_corners(session, plate)
            views += 1
        assert views == 3 and len(session.clicks) == 9
        assert session.click(10.0, 10.0) is None  # all three done in this view
        assert session.corners is not None  # fitted once every corner had two clicks
        np.testing.assert_allclose(session.corners, plate, atol=0.002)
        assert session.outline() is not None and len(shown) > 9

        session.undo()  # the last corner of this view, asked for again
        assert len(session.clicks) == 8 and session.asking() == 3
        session.skip()
        assert session.asking() is None
        path = session.finish()
    assert path == out / "calibration.urdf"
    written = XleRobotModel.load(path)
    np.testing.assert_allclose(np.array(written.wrist_boards["left"].corners), plate, atol=0.002)
    assert written.wrist_boards["left"].n_clicks == 8
    written.check_robot(client)  # still the same robot's calibration
    assert (out / "corners.json").is_file() and "Wrist camera plate: left arm" in (
        out / "wrist_board.md"
    ).read_text()


def test_a_session_is_refused_for_another_robots_calibration(robot, tmp_path):
    fake, client = robot
    start = calibration_for(client, tmp_path / "start.urdf")
    model = XleRobotModel.load(start)
    model.arm_calibration["left"]["elbow_flex"]["homing_offset"] += 40
    model.save(start)
    with pytest.raises(Exception, match="elbow_flex recalibrated"):
        BoardSession(client, start, "left", tmp_path / "board", lambda image: None)
    assert fake.arm_backend.trajectories == []


# ---- the plates in the camera's way ---------------------------------------------------------------------------


def test_a_segment_hits_a_box_only_through_it():
    from lerobot.felix_jr.calibration.xle_robot_model import segment_hits_box

    box, half = np.eye(4), np.array([0.02, 0.02, 0.005])
    assert segment_hits_box(np.array([0, 0, 1.0]), np.array([0, 0, -1.0]), box, half)  # straight through
    assert not segment_hits_box(np.array([0.05, 0, 1.0]), np.array([0.05, 0, -1.0]), box, half)  # beside it
    assert not segment_hits_box(np.array([0, 0, 1.0]), np.array([0, 0, 0.1]), box, half)  # stops short
    assert segment_hits_box(np.array([-1.0, 0.01, 0]), np.array([1.0, 0.01, 0]), box, half)  # edge-on, inside
    turned = np.eye(4)
    turned[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]  # the thin axis now along y
    assert not segment_hits_box(np.array([0, 0.03, 1.0]), np.array([0, 0.03, -1.0]), box, half)
    assert segment_hits_box(np.array([0, 0.003, 1.0]), np.array([0, 0.003, -1.0]), turned, half)


def plate_in_front_of_pinch(
    model: XleRobotModel, q: np.ndarray, ahead: float, aside: float = 0.0
) -> WristBoard:
    """A 4 cm plate `ahead` m in front of the right pinch point toward the camera, `aside` m off the line of sight,
    facing the camera."""
    pinch = model.pinch_point("right", q)
    camera = model.camera_pose(q)[:3, 3]
    sight = (camera - pinch) / np.linalg.norm(camera - pinch)
    u = np.cross(sight, [0.0, 0.0, 1.0])
    u /= np.linalg.norm(u)
    v = np.cross(sight, u)
    center = pinch + ahead * sight + aside * u
    corners_root = [center + 0.02 * (u + v), center + 0.02 * (u - v), center + 0.02 * (-u - v)]
    frame = model.frame_pose(board_parent("right"), q)
    corners = tuple(tuple(frame[:3, :3].T @ (c - frame[:3, 3])) for c in corners_root)
    return WristBoard(board_parent("right"), corners, tuple(frame[:3, :3].T @ -sight))


def test_a_plate_between_the_camera_and_the_pinch_point_hides_it(model):
    q = model.joint_vector({"right": REST})
    assert model.hidden_by("right", q) == ()  # no plates fitted: nothing hides
    hiding = model.with_wrist_board("right", plate_in_front_of_pinch(model, q, ahead=0.03))
    assert hiding.hidden_by("right", q) == ("right",)
    assert hiding.predict("right", q).hidden_by == ("right",)
    beside = model.with_wrist_board("right", plate_in_front_of_pinch(model, q, ahead=0.03, aside=0.06))
    assert beside.hidden_by("right", q) == () and beside.predict("right", q).hidden_by == ()
    behind = model.with_wrist_board("right", plate_in_front_of_pinch(model, q, ahead=-0.03))
    assert behind.hidden_by("right", q) == ()  # behind the point: not in the way


def test_solved_poses_say_what_the_camera_sees_of_the_grasp(model):
    from dataclasses import replace
    from types import SimpleNamespace

    from lerobot.felix_jr.harness.reach import join_poses, select_poses, solve_pose

    capture = SimpleNamespace(arms={"right": REST}, head_ticks={})
    pick = np.array([-0.4538, 0.1421, 0.8632])
    pose = solve_pose(model, "right", capture, pick, -15.0, roll_deg=0.0, width=0.01, label="pick")
    q = model.joint_vector({"right": pose.joints[0]})
    visibility, hidden = model.view_of(
        "right", q, pose.frames[0][:3, :3] @ pose.grasp + pose.frames[0][:3, 3]
    )
    assert pose.visibility[0] == pytest.approx(visibility) and pose.hidden_by == ((),) and hidden == ()
    assert "camera: jaws" in pose.describe() and f"{visibility:+.2f}" in pose.describe()
    hidden_pose = replace(pose, hidden_by=(("right",),))
    assert "camera: hidden by the right wrist camera plate" in hidden_pose.describe()
    edge_on = replace(pose, visibility=np.array([0.03]))
    assert "camera: jaws edge-on (+0.03)" in edge_on.describe()
    up = solve_pose(model, "right", capture, pick + [0, 0, 0.05], -15.0, width=0.01, init=pose, label="up")
    path = join_poses(hidden_pose, up)
    assert path.hidden_by == (("right",), ()) and len(path.visibility) == 2
    assert select_poses(path, "pick").hidden_by == (("right",),)
