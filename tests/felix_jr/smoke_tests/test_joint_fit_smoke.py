"""The joint calibration's fit (`calibration.joint_fit`) on a simulated session (`calibration.joint_sim`).

A true robot drawn at 30% of the priors, two board placements, each seen by the wrist camera from 8 arm poses and by
the head from 4 with the arms folded, the fingertips clicked: the fit brings the held-out errors near the noise
(`joint_sim`'s grid: two placements are what it takes).
"""

import numpy as np
import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("scipy")

from lerobot.felix_jr.calibration.boards import Checkerboard  # noqa: E402
from lerobot.felix_jr.calibration.joint_fit import JointProblem, fit_staged  # noqa: E402
from lerobot.felix_jr.calibration.joint_sim import ARM, Scenario, World, draw_truth  # noqa: E402
from lerobot.felix_jr.calibration.xle_robot_model import JAW_THICKNESS_M, XleRobotModel  # noqa: E402
from lerobot.felix_jr.harness import api  # noqa: E402


def test_a_simulated_session_is_fitted():
    model = XleRobotModel.load(api.MODEL)
    rng = np.random.default_rng(3)
    world = World(model, draw_truth(model, rng, scale=0.3), Checkerboard(8, 6, 0.025), rng)
    sightings, clicks = world.collect(Scenario(2, 8, 4, True))
    fit = fit_staged(JointProblem(model, world.board, {ARM: world.camera}, sightings, clicks))
    assert fit.rms_px[f"{ARM} board"] < 1.0 and fit.rms_px["head board"] < 1.0, fit.rms_px
    poses = world.grasp_poses(8)
    before = world.score(model.estimates, world.gc0, JAW_THICKNESS_M, poses)
    after = world.score(fit.theta, fit.gripper_from_camera[ARM], fit.thickness, poses)
    assert after["wrist px"] < 2.0 and after["camera mm"] < 3.0 and after["camera deg"] < 1.5, after
    assert after["head px"] < 4.0 < before["head px"], (before, after)
