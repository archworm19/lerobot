"""`joint_views`' planners: the jaws open at the views, the fingertips clear of the table, no fingers edge-on."""

import numpy as np
import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("scipy")

from lerobot.felix_jr.calibration import joint_views as jv  # noqa: E402
from lerobot.felix_jr.calibration.joint_sim import REST  # noqa: E402
from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel  # noqa: E402
from lerobot.felix_jr.harness import api  # noqa: E402

ARM = "right"


@pytest.fixture(scope="module")
def model():
    return XleRobotModel.load(api.MODEL)  # it has both arms' gripper maps


@pytest.fixture(scope="module")
def table_z(model):
    return float(model.frame_pose(f"{ARM}_base_link", model.joint_vector())[2, 3] - 0.06)


def test_the_views_take_the_openings_in_turn_and_stay_closed_without_a_map(model):
    poses = [dict(REST) for _ in range(5)]
    opened = jv.with_openings(model, ARM, poses)
    assert [p["gripper.pos"] for p in opened] == [20.0, 35.0, 50.0, 20.0, 35.0]
    assert jv.with_openings(model, ARM, poses, ())[0]["gripper.pos"] == 0.0
    no_map = XleRobotModel.load(api.MODEL)
    no_map.gripper_maps = {}
    assert {p["gripper.pos"] for p in jv.with_openings(no_map, ARM, poses)} == {0.0}


def test_low_poses_keep_both_tips_off_the_table_at_every_opening_and_the_fingers_off_edge_on(model, table_z):
    """The moving tip turns down past the fixed one as the jaws open: a low pose clears the table with it at 50, and
    shows the head the face it clicks, short of edge-on (`joint_collect.EDGE_ON_DEG`)."""
    from lerobot.felix_jr.calibration.joint_collect import EDGE_ON_DEG

    poses = jv.low_poses(model, ARM, REST, 8, table_z, np.random.default_rng(0))
    assert len(poses) >= 6
    for pose in poses:
        q = model.joint_vector({ARM: pose})
        for pos in jv.VIEW_OPENINGS:
            assert model.moving_tip(ARM, q, model.opening_deg(ARM, pos))[2] > table_z + jv.TIP_CLEARANCE_M
        assert model.pinch_point(ARM, q)[2] > table_z + jv.TIP_CLEARANCE_M
        assert jv.head_face_angle(model, ARM, q) < EDGE_ON_DEG[0]


def test_a_pose_whose_open_jaw_would_touch_the_table_is_refused(model, table_z):
    """Found by lowering a planned low pose until its fixed tip alone is just clear: the open moving tip is not."""
    pose = jv.low_poses(model, ARM, REST, 4, table_z, np.random.default_rng(1))[0]
    q = model.joint_vector({ARM: pose})
    pinch = model.pinch_point(ARM, q)[2]
    lowest = min(model.moving_tip(ARM, q, model.opening_deg(ARM, pos))[2] for pos in jv.VIEW_OPENINGS)
    if lowest >= pinch:
        pytest.skip("this pose's moving tip turns up as the jaws open")
    table = (pinch + lowest) / 2 - jv.TIP_CLEARANCE_M  # the fixed tip clear, the moving one not
    assert not jv.tips_clear(model, ARM, pose, table)
    no_map = XleRobotModel.load(api.MODEL)
    no_map.gripper_maps = {}
    assert jv.tips_clear(no_map, ARM, pose, table)  # the jaws then stay closed: the fixed tip is all there is


def test_the_tip_views_show_the_head_the_tips_at_heights(model, table_z):
    """`head_views`: the fingers about level, the pinch face toward the head (`head_sees`), both tips over the table
    at every opening, from low to well above it; the old low poses never showed it face-on (`HEAD_SEES_DEG`)."""
    from lerobot.felix_jr.calibration.xle_robot_model import pinch_frame

    poses = jv.head_views(model, ARM, REST, 10, table_z, np.random.default_rng(0))
    assert len(poses) >= 6
    heights = []
    for pose in poses:
        q = model.joint_vector({ARM: pose})
        assert jv.head_sees(model, ARM, q) and jv.tips_clear(model, ARM, pose, table_z)
        pinch = model.frame_pose(pinch_frame(ARM), q)
        assert abs(np.degrees(np.arcsin(-pinch[2, 2]))) < 35.0  # about level
        heights.append(pinch[2, 3] - table_z)
    assert max(heights) - min(heights) > 0.08  # a spread of heights
    low = jv.low_poses(model, ARM, REST, 8, table_z, np.random.default_rng(0))
    assert not any(jv.head_sees(model, ARM, model.joint_vector({ARM: p})) for p in low)


def test_a_path_may_end_as_near_the_table_as_its_view_but_not_dip_below_both_ends(monkeypatch):
    """The margin along a move stubbed as a function of the pan (0 at the start, 10 at the end): a view lower than the
    start is reached the straight way (the left arm's session went around through rest to each one); a path that dips
    below both ends, or goes in from clear ones, is not clear."""
    start = {f"{j}.pos": 0.0 for j in jv.SO101_JOINTS}
    end = {**start, "shoulder_pan.pos": 10.0}

    def clear_with(margin_of_t):
        monkeypatch.setattr(
            jv, "table_margin", lambda model, arm, pose, table_z: margin_of_t(pose["shoulder_pan.pos"] / 10)
        )
        return jv.path_clear(None, "left", start, end, 0.0)

    assert clear_with(lambda t: -0.001 - 0.02 * t)  # down to a view 2 cm within the clearances: its own depth
    assert clear_with(lambda t: -0.02 + 0.02 * t)  # up out of one
    assert not clear_with(lambda t: -0.02 - 0.02 * np.sin(np.pi * t))  # a dip below both ends
    assert not clear_with(lambda t: 0.01 - 0.03 * np.sin(np.pi * t))  # into the clearances from clear ends
