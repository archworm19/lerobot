"""`calibration.auto_tips`: the wrist fingertip a gripper sweep measured, as a session's wrist clicks (synthetic frames:
the sim's fingers painted on wood; skipped without mujoco, pinocchio or the meshes)."""

import copy
from dataclasses import replace

import cv2
import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("pinocchio")

from lerobot.felix_jr.calibration import auto_tips, gripper_sweep as sweep  # noqa: E402
from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel, pinch_frame  # noqa: E402
from lerobot.felix_jr.harness import api  # noqa: E402
from lerobot.felix_jr.sim2real import robot  # noqa: E402
from lerobot.felix_jr.sim2real.assets import is_cached  # noqa: E402
from lerobot.felix_jr.sim2real.view import RULER2_POSE  # noqa: E402

pytestmark = pytest.mark.skipif(
    not is_cached() or robot.FINGER_STL.read_bytes()[: len(robot.LFS_POINTER)] == robot.LFS_POINTER,
    reason="the SO-101 model is not cached, or the finger mesh not pulled",
)


@pytest.fixture(scope="module")
def model():
    return XleRobotModel.load(api.MODEL)


def test_to_raw_undoes_the_undistortion(model):
    camera = model.wrist_cameras["left"]
    raw = np.array([[120.0, 80.0], [320.0, 240.0], [500.0, 400.0]])
    pinhole = cv2.undistortPoints(
        raw.reshape(-1, 1, 2), camera.matrix(), np.asarray(camera.coeffs), P=camera.matrix()
    )
    back = np.array([auto_tips.to_raw(model, "left", p) for p in pinhole.reshape(-1, 2)])
    np.testing.assert_allclose(back, raw, atol=0.05)


def test_a_sweeps_fingertip_is_where_the_model_puts_the_wrist_faces_tip(model, tmp_path):
    """A sweep drawn by the model (its lens without distortion, the frames a pinhole's): the click it gives is where the
    model puts the wrist face's tip, the point the joint calibration fits the wrist clicks to, to a pixel or two; the
    ends found agree across the frames."""
    camera = model.wrist_cameras["left"]
    pinhole = copy.copy(model)
    pinhole.wrist_cameras = dict(model.wrist_cameras) | {
        "left": replace(camera, coeffs=(0.0,) * len(camera.coeffs))
    }
    drawn = sweep.FingerRenderer(pinhole, "left")
    for i, pos in enumerate((10.0, 25.0, 40.0)):
        pose = dict(RULER2_POSE, **{"gripper.pos": pos})
        drawn.set_arm(pose)
        fixed, moving = drawn.masks(pinhole.opening_deg("left", pos))
        image = np.empty(fixed.shape + (3,), np.uint8)
        image[:] = (205, 160, 120)
        image[fixed | moving] = (170, 190, 230)
        image = np.clip(image + np.random.default_rng(i).normal(0, 3, image.shape), 0, 255).astype(np.uint8)
        sweep.save_frame(tmp_path, i, sweep.SweepFrame("left", pose, image))
    drawn.close()
    click, spread = auto_tips.sweep_tip(pinhole, "left", tmp_path)
    q = pinhole.joint_vector({"left": RULER2_POSE})
    pinch = pinhole.frame_pose(pinch_frame("left"), q)
    face = sweep._project(pinhole, "left", q, (pinch @ np.r_[pinhole.wrist_face("left"), 1.0])[:3])
    assert np.linalg.norm(click - face) < 2.0, (click, face)
    assert spread.max() < 1.0
