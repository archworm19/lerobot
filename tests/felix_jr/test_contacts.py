"""`harness.contacts.patches_from_points`: the contact patches and the move, on exact synthetic cylinders.

The move must be self-consistent: with the jaws moved by it (the points by its opposite, in the pinch frame), the same
object has its widest chord `into` the fingers and its fixed-side contact `gap` from the fixed finger, whichever side
of the fixed finger it started on (bowl8: the shift's sign, composed by hand, went the wrong way once).
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("scipy")

from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel  # noqa: E402
from lerobot.felix_jr.harness.contacts import _jaw_axes, patches_from_points  # noqa: E402

URDF = Path(__file__).parents[2] / "src" / "lerobot" / "felix_jr" / "urdfs" / "xle_calib_1008_wrist2.urdf"
ARM = "right"
RADIUS = 0.009
TABLE = -0.03  # the table, along `up`, from the pinch point


@pytest.fixture(scope="module")
def model():
    return XleRobotModel.load(URDF)


def up_vector() -> np.ndarray:
    """Vertical in the pinch frame for a gripper pitched ~30 deg down (the approach, pinch +z, tilted below level)."""
    return cv2.Rodrigues(np.array([np.radians(-30.0), 0.0, 0.0]))[0] @ np.array([0.0, -1.0, 0.0])


def cylinder(model, axis_along: float, axis_ahead: float, rng) -> np.ndarray:
    """Points on an upright cylinder whose axis is `axis_along` the closing line and `axis_ahead` along the approach
    from the fingers' middle (pinch frame), from the table up 4 cm."""
    up = up_vector()
    closing, approach, middle = _jaw_axes(model, ARM, 2 * RADIUS)
    a_h = approach - (approach @ up) * up
    c_h = closing - (closing @ up) * up
    foot = middle + axis_along * c_h / (c_h @ c_h) + axis_ahead * a_h / (a_h @ a_h)
    foot = foot - (foot @ up - TABLE) * up
    e1 = np.cross(up, a_h)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    ang = rng.uniform(0, 2 * np.pi, 3000)
    h = rng.uniform(0.005, 0.04, 3000)
    return foot + np.outer(h, up) + RADIUS * (np.outer(np.cos(ang), e1) + np.outer(np.sin(ang), e2))


@pytest.mark.parametrize(("along", "ahead"), [(0.03, 0.03), (-0.01, 0.025), (0.012, -0.005)])
def test_the_move_puts_the_chord_into_the_fingers_and_the_fixed_finger_a_gap_away(model, along, ahead):
    rng = np.random.default_rng(1)
    pts = cylinder(model, along, ahead, rng)
    first = patches_from_points(model, ARM, pts, up_vector(), TABLE, gap=0.003, into=0.01)
    assert first.radius == pytest.approx(RADIUS, abs=2e-4)
    assert first.chord == pytest.approx(2 * RADIUS, abs=5e-4)
    # beyond the fixed finger's side (along < radius): the shift is negative, toward the fixed side
    assert np.sign(first.shift) == np.sign(first.fixed_contact - 0.003)
    after = patches_from_points(model, ARM, pts - first.move_pinch, up_vector(), TABLE, gap=0.003, into=0.01)
    assert after.ahead == pytest.approx(-0.01, abs=5e-4)
    assert after.fixed_contact == pytest.approx(0.003, abs=5e-4)
    assert np.linalg.norm(after.move_pinch) < 1e-3


def test_a_held_radius_and_too_few_points(model):
    rng = np.random.default_rng(2)
    pts = cylinder(model, 0.02, 0.02, rng)
    held = patches_from_points(model, ARM, pts, up_vector(), TABLE, radius=RADIUS)
    assert held.radius == RADIUS and held.fit_rms_m < 1e-4
    with pytest.raises(ValueError, match="points on the graspable part"):
        patches_from_points(model, ARM, pts[:5], up_vector(), TABLE)
