"""The arms' shapes in the model: the wrist cameras' plates and the parts' boxes, in the URDF and in the camera's
line of sight (`XleRobotModel`; the boxes are measured with `calibration.arm_shapes`)."""

import json
import xml.etree.ElementTree as ET  # nosec B405: our own files
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("scipy")

from lerobot.felix_jr.calibration.xle_robot_model import (  # noqa: E402
    CALIBRATION_TAG,
    ArmBox,
    WristBoard,
    XleRobotModel,
    wrist_board_link,
)

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


def board_parent(arm: str) -> str:
    return f"{arm}_gripper_link"


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


# ---- the boxes ------------------------------------------------------------------------------------------------


def test_boxes_in_the_model_and_the_urdf(model, tmp_path):
    forearm = ArmBox("right_lower_arm_link", (-0.06, 0.0, 0.0), (0.13, 0.05, 0.04), rms_px=1.2, n_clicks=30)
    finger = ArmBox("right_gripper_frame_link", (-0.005, 0.0, -0.03), (0.01, 0.025, 0.06), moving=True)
    with_parts = model.with_boxes("right", {"forearm": forearm, "moving_finger": finger})
    assert model.boxes == {}  # a copy
    with_parts.save(tmp_path / "c.urdf")
    loaded = XleRobotModel.load(tmp_path / "c.urdf")
    assert loaded.boxes["right"] == {"forearm": forearm, "moving_finger": finger}
    again = loaded.with_boxes(
        "right", {"forearm": ArmBox("right_lower_arm_link", (0, 0, 0), (0.2, 0.05, 0.05))}
    )
    assert (
        again.boxes["right"]["forearm"].size == (0.2, 0.05, 0.05) and "moving_finger" in again.boxes["right"]
    )
    with pytest.raises(ValueError, match="unknown parts"):
        model.with_boxes("right", {"elbow": forearm})
    # the moving finger turns with the jaw, about the hinge: its center moves, by the jaw's own turn
    q = model.joint_vector({"right": REST})
    closed, _ = again.boxes_root("right", q, 0.0)["moving_finger"]
    opened, half = again.boxes_root("right", q, 30.0)["moving_finger"]
    turn = model.moving_jaw("right", 30.0)
    pinch = model.frame_pose("right_gripper_frame_link", q)
    expected = pinch[:3, :3] @ (turn[:3, :3] @ np.asarray(finger.center) + turn[:3, 3]) + pinch[:3, 3]
    np.testing.assert_allclose(opened[:3, 3], expected, atol=1e-12)
    np.testing.assert_allclose(opened[:3, :3], pinch[:3, :3] @ turn[:3, :3], atol=1e-12)  # square to the jaw
    np.testing.assert_allclose(half, 0.5 * np.asarray(finger.size))
    assert np.linalg.norm(opened[:3, 3] - closed[:3, 3]) > 0.01


def test_an_old_calibrations_capsules_are_left_out(model, tmp_path, caplog):
    """Before the boxes (2026-11-06) the parts were capsules: loading such a file says so, and keeps the rest."""
    path = tmp_path / "old.urdf"
    model.save(path)
    root = ET.fromstring(path.read_text())
    element = root.find(CALIBRATION_TAG)
    assert element is not None and element.text
    data = json.loads(element.text)
    data["capsules"] = {
        "right": {
            "forearm": {"parent": "right_lower_arm_link", "a": [0, 0, 0], "b": [-0.1, 0, 0], "radius": 0.03}
        }
    }
    element.text = json.dumps(data)
    path.write_text(ET.tostring(root, encoding="unicode"))
    with caplog.at_level("WARNING"):
        old = XleRobotModel.load(path)
    assert old.boxes == {} and "capsules are left out" in caplog.text
    assert old.intrinsics == model.intrinsics  # the rest kept


# the true forearm: about 1 cm off its URDF start (the elbow to the wrist flex joint, along -x), a little bigger,


def test_a_plate_comes_off_the_model(model, tmp_path):
    both = model.with_wrist_board("right", board("right")).with_wrist_board("left", board("left"))
    right_off = both.without_wrist_board("right")
    assert set(right_off.wrist_boards) == {"left"}
    assert wrist_board_link("right") not in right_off.urdf and wrist_board_link("left") in right_off.urdf
    right_off.save(tmp_path / "off.urdf")
    assert set(XleRobotModel.load(tmp_path / "off.urdf").wrist_boards) == {"left"}
    assert right_off.without_wrist_board("right") is right_off  # none there: the same model
