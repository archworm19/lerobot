"""Object-centric grasping, phase A's replay: bowl1's jar from its hand-fit cylinder (`docs/object_grasping.md`).

bowl1 (2026-11-05, `docs/sessions.md`): the right arm took the jar (4.0 cm wide, z 71.4..79.3 cm) by its lid. The
first plan, at roll -60 to keep the jaws visible, put the fixed jaw inside the jar (a tilted closing line crosses a
vertical cylinder over width / cos(roll)); the re-plan that worked was level: pitch -40, heading -129, roll 0, the
fixed jaw 0.8 cm clear. Checked, with no robot and no capture, only the cylinder:

1. The best candidates close level across the jar: the roll -60 grasp is not in a cylinder's family, and the
   ranking puts the most antipodal first (generous bounds accept more: they rank lower).
2. The grasp that worked is found as it was planned, with nothing said about pitch, heading or roll.
3. The view filters: pitch -30 from the robot's front right is the grasp bowl1 turned down because the right wrist
   camera's plate hid it; it is rejected for the plate, though the arm reaches it.
4. Ranked by the antipodal error (5 deg steps), then the joint margin, the best is the grasp that worked (pitch -44,
   antipodal within 2 deg, every joint >= 50 deg from its bound), and the head camera sees it by the old check too
   (`XleRobotModel.view_of`).
5. Every stage is timed.
6. The best, planned (`plan_pick`), runs through the harness's execution as it is (`prepare_execution`): stage,
   attack point, grasp, lift, every step reached with no joint jump; its attack point and grasp within 1 cm of
   bowl1's.

The numbers are bowl1's recorded objects (`harness/scratch/sessions/bowl1`, gitignored), copied here.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("scipy")

from lerobot.felix_jr.calibration.xle_robot_model import WristBoard, XleRobotModel  # noqa: E402
from lerobot.felix_jr.harness.execute import prepare_execution  # noqa: E402
from lerobot.felix_jr.harness.grasping import (  # noqa: E402
    PICK_GRIP,
    STAGES,
    GraspSettings,
    grasp_candidates,
    plan_pick,
)
from lerobot.felix_jr.harness.objects import angles_of  # noqa: E402
from lerobot.felix_jr.harness.primitives import Cylinder  # noqa: E402
from lerobot.felix_jr.harness.reach import ArmIK  # noqa: E402

URDF = Path(__file__).parents[3] / "src" / "lerobot" / "felix_jr" / "urdfs" / "xle_calib_11062026.urdf"
# bowl1 had the wrist cameras' plates on; the URDF no longer has them (they are off the robot since bowl2): the plates
# as fitted on 2026-11-06 (`calibration.arm_shapes`), copied here
PLATES = {
    "right": WristBoard(
        "right_gripper_link",
        ((-0.011794, 0.098130, -0.012091), (0.025274, 0.093440, -0.010682), (0.025162, 0.066445, 0.011865)),
        (-0.277419, -0.867940, -0.411969),
        0.01,
    ),
    "left": WristBoard(
        "left_gripper_link",
        ((-0.003598, 0.092922, -0.016610), (0.035652, 0.089523, -0.019473), (0.030896, 0.048261, -0.003595)),
        (-0.388869, -0.862821, -0.322987),
        0.01,
    ),
}
JAR = Cylinder.upright(np.array([-0.344, -0.107, 0.714]), 0.079, 0.020)
CAPTURE = SimpleNamespace(  # bowl1's capture1: the arms at rest
    arms={
        "right": {
            "shoulder_pan.pos": -0.132,
            "shoulder_lift.pos": -102.989,
            "elbow_flex.pos": 96.527,
            "wrist_flex.pos": 71.341,
            "wrist_roll.pos": -0.220,
            "gripper.pos": 1.037,
        },
        "left": {
            "shoulder_pan.pos": -8.088,
            "shoulder_lift.pos": -102.725,
            "elbow_flex.pos": 96.484,
            "wrist_flex.pos": 67.165,
            "wrist_roll.pos": -2.330,
            "gripper.pos": 0.975,
        },
    },
    head_ticks={"head_neck": 1256, "head_eyes": 1329},
)
WORKED = (-40.0, -129.0, 0.0)  # pitch, heading, roll of the pick that landed
EDGE_ON = 0.2  # |visibility| below it: the jaws are seen edge-on (`objects.EDGE_ON_VISIBILITY`)


def _wrap(deg: float) -> float:
    return abs((deg + 180.0) % 360.0 - 180.0)


@pytest.fixture(scope="module")
def model():
    model = XleRobotModel.load(URDF)
    for arm, board in PLATES.items():
        model = model.with_wrist_board(arm, board)
    return model


@pytest.fixture(scope="module")
def found(model):
    # the lid (the top 2 cm), from the side to straight down: as the session would ask
    return grasp_candidates(
        JAR,
        model,
        CAPTURE,
        arms=("right",),
        where={"t": (0.059, 0.079), "psi": (0.0, 90.0)},
        n=200,
    )


def test_the_best_close_level(found):
    assert found.passed > 0
    for c in found.candidates[:5]:
        assert abs(c.claw.grasp.closing_root[2]) < 1e-9  # the family: square to the vertical axis
        assert c.antipodal_error_deg < found.settings.rank_step_deg, c.describe()
        # the solved tip-to-tip line: within the first step of level
        d_root = c.grasp.root_from_pinch[:3, :3] @ (c.claw.aim_pinch / np.linalg.norm(c.claw.aim_pinch))
        assert np.degrees(np.arcsin(abs(d_root[2]))) <= found.settings.rank_step_deg


def test_the_grasp_that_worked_is_found(found):
    def distance(c) -> float:
        pitch, heading, roll = angles_of(c.grasp.root_from_pinch[:3, :3], WORKED[1])
        return max(abs(pitch - WORKED[0]), _wrap(heading - WORKED[1]), _wrap(roll - WORKED[2]))

    nearest = min(found.candidates, key=distance)
    assert distance(nearest) <= 6.0, nearest.describe()
    assert nearest.claw.grasp.params["psi"] == 45.0 and nearest.claw.grasp.params["phi"] == 60.0


def seen(model, candidate) -> tuple[float, tuple[str, ...]]:
    """How the head camera (at the capture's pose) sees the candidate's grasp: the jaws' visibility, hiding plates."""
    ik = ArmIK(
        model, "right", CAPTURE.arms, CAPTURE.head_ticks, candidate.claw.grasp.opening, candidate.claw.margin
    )
    q = ik.q.copy()
    q[ik.index] = candidate.grasp.x
    return model.view_of("right", q, ik.tool(candidate.grasp.root_from_pinch))


def visible(model, candidate) -> bool:
    visibility, hidden_by = seen(model, candidate)
    return not hidden_by and abs(visibility) >= EDGE_ON


def test_the_hidden_grasp_is_rejected_with_its_reason(found):
    """Pitch -30 from the robot's front right, the grasp bowl1 turned down: the right plate hides it. It is rejected
    at the attack point, and says so."""
    (hidden,) = (
        r
        for r in found.rejected
        if dict(r.grasp.params) == {"t": 0.069, "phi": 60.0, "psi": 30.0} and r.fixed_side == 0
    )
    assert hidden.stage in ("attack view", "grasp view") and "the plate hides the grasp" in hidden.reason
    assert "right arm's" in hidden.detail
    assert hidden.attack is not None and hidden.attack.ok  # the arm reaches it: only the view rules it out


def test_the_view_filters_can_be_turned_off(model):
    """The same grasp with the view filters off (the one at the hold judges by the picture): it passes, and the
    settings say nothing checked the view."""
    settings = GraspSettings(see_at_attack=False, see_at_grasp=False)
    where = {"t": (0.069, 0.069), "phi": (60.0, 60.0), "psi": (30.0, 30.0)}
    unseen = grasp_candidates(JAR, model, CAPTURE, arms=("right",), where=where, settings=settings)
    assert any(c.claw.fixed_side == 0 for c in unseen.candidates)
    assert not any(r.stage in ("attack view", "grasp view") for r in unseen.rejected)
    assert "the head camera's view not checked" in unseen.describe()


def test_the_best_is_seen(model, found):
    best = found.candidates[0]
    assert visible(model, best), best.describe()  # by the old check (plates, edge-on) too
    assert best.limit_margin_deg >= 40.0


def test_the_best_is_like_the_grasp_that_worked(found):
    best = found.candidates[0]
    pitch, heading, roll = angles_of(best.grasp.root_from_pinch[:3, :3], WORKED[1])
    level = min(_wrap(roll - WORKED[2]), _wrap(roll - WORKED[2] - 180.0))  # either jaw on either side
    assert max(abs(pitch - WORKED[0]), _wrap(heading - WORKED[1]), level) <= 6.0, best.describe()
    low = [c for c in found.candidates if c.grasp.root_from_pinch[2, 2] > -0.3]  # pitch above about -17
    assert all(c.limit_margin_deg < best.limit_margin_deg for c in low)


def test_the_stages_are_timed(found):
    assert set(found.timing) <= set(STAGES)
    total, runs = found.timing["attack point"]
    assert runs > 0 and total / runs < 0.2  # an IK solve: tens of ms
    for stage in ("attack view", "grasp view"):  # the view filters: a few ms each
        total, runs = found.timing[stage]
        assert runs > 0 and total / runs < 0.05, (stage, total / runs)
    assert "attack point" in found.describe() and "ms x" in found.describe()


def test_the_best_is_planned_and_runs_through_the_execution(model, found):
    pick = plan_pick(model, found.candidates[0], CAPTURE)
    assert pick.ok.all(), pick.describe()
    execution = prepare_execution(model, pick, PICK_GRIP)
    assert len(execution.moves) > 4
    # bowl1's own plan, as executed: the attack point (pick.back) and the grasp aim (pick), root frame
    np.testing.assert_allclose(pick.points[1], [-0.329, -0.083, 0.806], atol=0.01)
    np.testing.assert_allclose(pick.points[2], [-0.348, -0.107, 0.780], atol=0.01)
