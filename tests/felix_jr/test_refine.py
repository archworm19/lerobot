"""`sim2real.refine`: the parameters searched, the grasp's camera clearance left to the simulator, the camera's board
colliding in the simulator (the rollouts themselves: `smoke_tests/test_refine_smoke.py`)."""

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("pinocchio")

from lerobot.felix_jr.calibration.joint_sim import REST  # noqa: E402
from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel  # noqa: E402
from lerobot.felix_jr.harness import api  # noqa: E402
from lerobot.felix_jr.harness.primitives import Cylinder  # noqa: E402
from lerobot.felix_jr.sim2real import bank, refine  # noqa: E402
from lerobot.felix_jr.sim2real.assets import is_cached  # noqa: E402

# seed 0's placement 4 (as `test_bank_smoke.py`): its greedy grasp (side 1), 10 mm deeper (the camera's board then
# within its clearance of the bottle at the grasp, not at the attack)
DEEP = {"t": 0.0986, "phi": 0.0, "psi": 30.0, "depth": 0.01, "offset": 0.0}


@pytest.fixture(scope="module")
def model():
    return XleRobotModel.load(api.MODEL)


@pytest.fixture(scope="module")
def placement(model):
    return bank.placements(model, bank.BankConfig(), 5)[4]


def test_the_search_is_the_familys_parameters_and_the_grippers_clipped_or_wrapped():
    cyl = Cylinder.upright(np.zeros(3), 0.15, 0.025)
    space = refine.space(cyl, {"psi": (0.0, 45.0)})
    assert list(space) == ["t", "phi", "psi", "depth", "offset"]
    assert (space["depth"].low, space["depth"].high) == (
        0.0,
        pytest.approx(0.025 + refine.DEPTH_PAST_RADIUS_M),
    )
    assert (space["offset"].low, space["offset"].high) == (pytest.approx(-0.0125), pytest.approx(0.0125))
    clipped = refine._clip({"t": 0.2, "phi": -3.0, "psi": 50.0, "depth": -0.01, "offset": 0.0}, space)
    assert clipped == {"t": 0.15, "phi": pytest.approx(357.0), "psi": 45.0, "depth": 0.0, "offset": 0.0}


def test_a_deep_grasp_the_cameras_clearance_refuses_at_the_grasp_is_planned_when_checked_at_the_attack_only(
    model, placement
):
    settings = bank.BankConfig().settings()
    start = {**REST, "gripper.pos": 2.0}
    target = placement.primitive()
    refused = refine.plan_for(model, "right", target, DEEP, 1, settings, start)
    assert isinstance(refused, str) and refused.startswith("wrist camera")
    planned = refine.plan_for(
        model, "right", target, DEEP, 1, replace(settings, camera_clearance_at_grasp=False), start
    )
    assert not isinstance(planned, str), planned
    candidate, execution = planned
    assert candidate.claw.depth == pytest.approx(0.01) and len(execution.moves) > 5


@pytest.mark.skipif(not is_cached(), reason="the SO-101 model is not cached")
def test_the_wrist_cameras_real_board_collides_and_the_modules_mesh_does_not(model):
    from lerobot.felix_jr.sim2real import robot
    from lerobot.felix_jr.sim2real.scene import build

    scene = build(model, "right", table_z=-1.0)
    m = scene.m
    board = m.geom("wrist_camera_board").id
    assert m.geom_contype[board] == 1 and np.allclose(2 * m.geom_size[board], robot.CAMERA_BOARD_M)
    module = m.body(robot.CAMERA_MODULE).id
    assert all(m.geom_contype[i] == 0 for i in range(m.ngeom) if m.geom_bodyid[i] == module)
    for pos in (0.0, 50.0, 100.0):  # (the board clear of the gripper's own parts, the jaws anywhere)
        scene.set_pose({**{f"{j}.pos": 0.0 for j in robot.JOINTS}, "gripper.pos": pos})
        assert not [c for c in scene.d.contact[: scene.d.ncon] if board in (c.geom1, c.geom2)]
