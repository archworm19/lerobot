"""Choosing views: `find_view` (head and arm joints that show a pinch point) and `plan_targets`."""

import numpy as np
import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("scipy")

from lerobot.felix_jr.calibration import views  # noqa: E402
from lerobot.felix_jr.calibration.views import find_view, head_bounds, plan_targets  # noqa: E402
from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel  # noqa: E402

INTRINSICS = {"fx": 610.97, "fy": 610.01, "ppx": 315.95, "ppy": 247.03, "width": 640, "height": 480}
SIGNS = {"head_neck": -1, "head_eyes": 1}
REFS = {"head_neck": 1605, "head_eyes": 1324}
LIMITS = {name: (ref - 450, ref + 450) for name, ref in REFS.items()}


@pytest.fixture(scope="module")
def model():
    """A model with the head about as the first real calibration found it (tilted 33 degrees down at home)."""
    m = XleRobotModel.from_urdf(head_signs=SIGNS, head_refs=REFS, intrinsics=INTRINSICS)
    m.set_estimates({"zero:head_tilt_joint": np.radians(33.0), "zero:head_pan_joint": np.radians(-9.0)})
    return m


@pytest.mark.parametrize("arm", ["left", "right"])
def test_a_view_shows_the_pinch_point_in_the_middle_from_the_visible_side(model, arm):
    target = plan_targets(model, arm, 5)[0]
    seed = model.joint_vector({arm: target.arm_pose}, REFS)
    view = find_view(model, arm, seed, target.point, LIMITS)
    assert view.ok, view.reason
    assert np.linalg.norm(view.prediction.uv - [320, 240]) < 20  # in the middle
    assert view.facing_deg < views.FACING_MAX_DEG and view.prediction.visibility > 0  # the positive side
    assert np.linalg.norm(view.prediction.point - target.point) < 0.01
    assert view.arm_pose["gripper.pos"] == views.CLOSED
    assert all(low <= view.head_ticks[n] <= high for n, (low, high) in LIMITS.items())
    # what the view commands is what the model was solved for
    again = model.predict(arm, model.joint_vector({arm: view.arm_pose}, view.head_ticks))
    np.testing.assert_allclose(again.uv, view.prediction.uv, atol=2.0)  # the head is rounded to whole ticks


def test_a_target_out_of_the_heads_reach_still_gets_a_view_that_shows_the_pinch_point(model):
    targets = plan_targets(model, "left", 5)
    target = targets[3]  # far to the left: the neck reaches its limit before it would be in the middle
    view = find_view(model, "left", model.joint_vector({"left": target.arm_pose}, REFS), target.point, LIMITS)
    assert view.ok, (
        view.reason
    )  # in the picture, with a margin: showing the pinch point beats reaching the target
    assert view.head_ticks["head_neck"] == LIMITS["head_neck"][0]


def test_a_view_is_found_even_when_the_head_starts_looking_away(model):
    away = {"head_neck": REFS["head_neck"] + 400, "head_eyes": REFS["head_eyes"] - 300}
    assert model.predict("left", model.joint_vector({"left": {}}, away)).z < 0 or not model.in_view(
        model.predict("left", model.joint_vector({"left": {}}, away)).uv
    )
    view = find_view(model, "left", model.joint_vector({"left": {}}, away), head_limits=LIMITS)
    assert view.ok, view.reason


def test_the_other_side_is_shown_when_asked(model):
    flipped = XleRobotModel(model.urdf, SIGNS, REFS, INTRINSICS, visible_side={"left": "negative"})
    flipped.set_estimates(model.estimates)
    view = find_view(flipped, "left", flipped.joint_vector({"left": {}}, REFS), head_limits=LIMITS)
    assert view.prediction.visibility < 0 and view.facing_deg < views.FACING_MAX_DEG


def test_a_view_keeps_the_pinch_point_above_the_floor(model):
    floored = XleRobotModel(model.urdf, SIGNS, REFS, INTRINSICS)
    floored.set_estimates(model.estimates)
    low = {"shoulder_lift.pos": 30.0, "elbow_flex.pos": 40.0}
    seed = floored.joint_vector({"left": low}, REFS)
    floored.floor_z = float(floored.pinch_point("left", seed)[2])  # the seed's pinch point is on the floor
    view = find_view(floored, "left", seed, head_limits=LIMITS)
    assert view.prediction.point[2] >= floored.floor_z + views.FLOOR_CLEARANCE - 0.005


def test_targets_spread_around_the_seed_and_stay_above_the_floor(model):
    targets = plan_targets(model, "left", 20)
    assert len(targets) == 20 and targets[0].arm_pose["shoulder_pan.pos"] == 0.0  # the seed first
    points = np.array([t.point for t in targets])
    gaps = np.linalg.norm(points[:, None] - points[None], axis=2) + np.eye(len(points))
    assert gaps.min() > 0.02  # spread out, not clustered
    for t in targets:  # each target comes with the pose that reaches it
        np.testing.assert_allclose(
            model.pinch_point("left", model.joint_vector({"left": t.arm_pose})), t.point
        )
    floored = XleRobotModel(model.urdf, SIGNS, REFS, INTRINSICS, floor_z=float(points[:, 2].mean()))
    floored.set_estimates(model.estimates)
    high = plan_targets(floored, "left", 10)
    assert all(t.point[2] >= floored.floor_z + views.FLOOR_CLEARANCE for t in high[1:])


def test_head_bounds_come_from_tick_limits_whatever_the_servo_direction(model):
    bounds = head_bounds(model, LIMITS)
    step = 2 * np.pi / 4096
    assert bounds["head_pan_joint"] == pytest.approx((-450 * step, 450 * step))  # sign -1: ends swap
    assert bounds["head_tilt_joint"] == pytest.approx((-450 * step, 450 * step))
    assert head_bounds(model, None)["head_tilt_joint"] == model.limits("head_tilt_joint")
