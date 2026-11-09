"""The claw grasp: the model's grasp geometry, `solve_pose` putting the grasp point on the target, the open jaw it
keeps for drawing, and the gripper commands `prepare_execution` derives from the object's width."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("scipy")

from lerobot.felix_jr.calibration.xle_robot_model import GripperMap, XleRobotModel, pinch_frame  # noqa: E402
from lerobot.felix_jr.harness.execute import gripper_pos, held_object, prepare_execution  # noqa: E402
from lerobot.felix_jr.harness.reach import OPEN_CLEARANCE_M, SQUEEZE_M, join_poses, solve_pose  # noqa: E402

URDF = Path(__file__).parent / "reference" / "pen1_calibration_10312026.urdf"
GRIPPER = GripperMap(
    -1.37, 1.2718, {"drive_mode": 0, "homing_offset": -860, "range_min": 2046, "range_max": 3493}
)
REST = {
    "right": {
        "shoulder_pan.pos": 2.154,
        "shoulder_lift.pos": -102.989,
        "elbow_flex.pos": 96.527,
        "wrist_flex.pos": 71.429,
        "wrist_roll.pos": -0.22,
    }
}
CAPTURE = SimpleNamespace(arms=REST, head_ticks={})
PICK = np.array([-0.4538, 0.1421, 0.8632])  # pen1's pick


@pytest.fixture(scope="module")
def model():
    m = XleRobotModel.load(URDF)
    m.gripper_maps["right"] = GRIPPER
    return m


def test_the_grasp_point_is_midway_between_the_tips(model):
    radius = model.jaw_radius("right")
    for width in (0.0, 0.01, 0.04, 0.1):
        opening = model.opening_for_width("right", width)
        jaw = model.moving_jaw("right", opening)
        assert np.linalg.norm(jaw[:3, 3]) == pytest.approx(
            width
        )  # the moving tip, `width` from the pinch point
        assert opening == pytest.approx(np.degrees(2 * np.arcsin(width / (2 * radius))))
        # midway between the tips, in the middle of the fingers' thickness
        np.testing.assert_allclose(
            model.grasp_offset("right", width), jaw[:3, 3] / 2 + model.wrist_face("right") / 2
        )
        np.testing.assert_allclose(model.grasp_offset("right", width, mid_plane=False), jaw[:3, 3] / 2)
        np.testing.assert_allclose(jaw[:3, :3] @ jaw[:3, :3].T, np.eye(3), atol=1e-12)  # a rigid turn
    q = model.joint_vector()
    frame = model.frame_pose(pinch_frame("right"), q)
    tip = (frame @ model.moving_jaw("right", 30.0))[:3, 3]
    np.testing.assert_allclose(tip, model.moving_tip("right", q, 30.0))  # the same tip as `moving_tip`
    # a claw: the grasp point of a 4 cm object is not on the jaw axis, the tip's arc brings it forward
    grasp = model.grasp_offset("right", 0.04)
    assert grasp[0] == pytest.approx(-0.02, abs=0.001) and grasp[2] > 0.002
    with pytest.raises(ValueError, match="cannot be"):
        model.opening_for_width("right", 2 * radius + 0.001)


def test_solve_pose_puts_the_grasp_point_on_the_target_and_keeps_the_open_jaw(model):
    width = 0.03
    pose = solve_pose(model, "right", CAPTURE, PICK, -15.0, roll_deg=0.0, width=width, label="pick")
    assert pose.ok[0], pose.describe()
    frame = pose.frames[0]
    np.testing.assert_allclose(frame[:3, :3] @ pose.grasp + frame[:3, 3], PICK, atol=0.005)
    np.testing.assert_allclose(pose.grasp, model.grasp_offset("right", width))
    # drawn open for the approach: the tips the clearance wider apart than the object
    assert np.linalg.norm(pose.open_jaw[:3, 3]) == pytest.approx(width + OPEN_CLEARANCE_M)
    assert "for an object 3.0 cm wide" in pose.describe()
    up = solve_pose(model, "right", CAPTURE, PICK + [0, 0, 0.05], -15.0, width=width, init=pose, label="up")
    joined = join_poses(pose, up)
    assert joined.grasp is pose.grasp and joined.open_jaw is pose.open_jaw
    with pytest.raises(ValueError, match="object width"):
        join_poses(pose, solve_pose(model, "right", CAPTURE, PICK, -15.0, width=0.02, label="other"))


def test_the_gripper_opens_and_closes_by_the_objects_width(model):
    pose = solve_pose(model, "right", CAPTURE, PICK, -15.0, roll_deg=0.0, width=0.03, label="pick")
    up = solve_pose(model, "right", CAPTURE, PICK + [0, 0, 0.05], -15.0, width=0.03, init=pose, label="up")
    execution = prepare_execution(model, join_poses(pose, up), {"start": "open", "pick": "close"})
    grips = {m.label: m for m in execution.moves if "gripper" in m.label}
    assert set(grips) == {
        "start: gripper open (tips 5.0 cm apart)",
        "pick: gripper close (tips 2.7 cm apart)",
    }
    opened, closed = (grips[k].targets["gripper.pos"] for k in sorted(grips, reverse=True))
    assert opened == pytest.approx(GRIPPER.pos_for(model.opening_for_width("right", 0.03 + OPEN_CLEARANCE_M)))
    assert closed == pytest.approx(GRIPPER.pos_for(model.opening_for_width("right", 0.03 - SQUEEZE_M)))
    assert not grips["pick: gripper close (tips 2.7 cm apart)"].settle  # it stops on the object
    # no object: closed is where the tips meet (not below 0); open no wider than the jaws go
    assert gripper_pos(model, "right", 0.0) == 0.0
    assert gripper_pos(model, "right", 1.0) == pytest.approx(100.0)


def test_gripper_actions_need_the_arms_gripper_map(model):
    bare = XleRobotModel.load(URDF)
    pose = solve_pose(bare, "right", CAPTURE, PICK, -15.0, roll_deg=0.0, width=0.03, label="pick")
    prepare_execution(bare, pose)  # no gripper actions: no map needed
    with pytest.raises(ValueError, match="no gripper map for the right arm"):
        prepare_execution(bare, pose, {"pick": "close"})


def test_the_held_object_is_centered_on_the_grasp_point(model):
    pose = solve_pose(model, "right", CAPTURE, PICK, -15.0, roll_deg=0.0, width=0.03, label="pick")
    held = held_object(pose, 0.1, "pick", "pick")
    np.testing.assert_allclose(held.centers[0], pose.frames[0][:3, :3] @ pose.grasp + pose.frames[0][:3, 3])


def test_a_margin_aims_toward_the_moving_jaw_and_the_object_ends_on_the_fixed_one(model):
    width, margin = 0.01, 0.01
    plain = model.grasp_offset("right", width, mid_plane=False)
    aimed = model.grasp_offset("right", width, margin, mid_plane=False)
    assert np.linalg.norm(aimed) == pytest.approx(
        width / 2 + margin
    )  # the fixed tip that far from the center (in the plane the jaws open in)
    np.testing.assert_allclose(
        aimed / np.linalg.norm(aimed), plain / np.linalg.norm(plain)
    )  # along the tips' line
    assert np.linalg.norm(model.grasp_offset("right", 0.0, margin, mid_plane=False)) == pytest.approx(
        margin
    )  # no width: still aimed

    pose = solve_pose(
        model, "right", CAPTURE, PICK, -15.0, roll_deg=0.0, width=width, margin=margin, label="pick"
    )
    assert pose.ok[0], pose.describe()
    frame = pose.frames[0]
    np.testing.assert_allclose(
        frame[:3, :3] @ pose.grasp + frame[:3, 3], PICK, atol=0.005
    )  # aimed at the target
    np.testing.assert_allclose(pose.held, model.grasp_offset("right", width))  # held at the plain grasp point
    fixed_tip_to_center = np.linalg.norm(PICK - frame[:3, 3])
    assert fixed_tip_to_center == pytest.approx(width / 2 + margin, abs=0.005)
    assert np.linalg.norm(pose.open_jaw[:3, 3]) == pytest.approx(width + margin + OPEN_CLEARANCE_M)
    assert "the fixed jaw 1.0 cm clear of it" in pose.describe()

    held = held_object(pose, 0.1, "pick", "pick")
    np.testing.assert_allclose(
        held.centers[0], frame[:3, :3] @ pose.held + frame[:3, 3]
    )  # swept onto the fixed jaw
    up = solve_pose(
        model, "right", CAPTURE, PICK + [0, 0, 0.05], -15.0, width=width, margin=margin, init=pose, label="up"
    )
    execution = prepare_execution(model, join_poses(pose, up), {"start": "open", "pick": "close"})
    labels = [m.label for m in execution.moves if "gripper" in m.label]
    assert labels == ["start: gripper open (tips 4.0 cm apart)", "pick: gripper close (tips 0.7 cm apart)"]
    with pytest.raises(ValueError, match="margin"):
        join_poses(pose, solve_pose(model, "right", CAPTURE, PICK, -15.0, width=width, label="other"))


def test_prepare_execution_tool_passes_the_pose_tolerance(model):
    from lerobot.felix_jr.harness import api

    h = api.Harness(model=model, endpoint="tcp://127.0.0.1:1")  # never connects: nothing here moves
    capture = CAPTURE
    pose = h.call("solve_pose", "right", capture, PICK, -15.0, roll_deg=0.0, width=0.03, label="pick")
    up = h.call("solve_pose", "right", capture, PICK + [0, 0, 0.05], -15.0, width=0.03, init=pose, label="up")
    path = h.call("join_poses", pose, up)
    default = h.call("prepare_execution", path)
    loose = h.call("prepare_execution", path, pose_tol_deg=4.0)
    labeled = [m.label for m in default.moves if not m.label.startswith("step")]
    assert [m.tol for m in default.moves if m.label in labeled] == [2.5] * len(labeled)
    assert [m.tol for m in loose.moves if m.label in labeled] == [4.0] * len(labeled)
    assert [m.tol for m in loose.moves if m.label.startswith("step")] == [
        m.tol for m in default.moves if m.label.startswith("step")
    ]  # the steps keep their own
