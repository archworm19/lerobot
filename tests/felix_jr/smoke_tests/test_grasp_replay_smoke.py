"""Object-centric grasping, phase A's replay: bowl1's jar from its hand-fit cylinder (`docs/object_grasping.md`).

bowl1 (2026-11-05, `docs/sessions.md`): the right arm took the jar (4.0 cm wide, z 71.4..79.3 cm) by its lid. The
first plan, at roll -60 to keep the jaws visible, put the fixed jaw inside the jar (a tilted closing line crosses a
vertical cylinder over width / cos(roll)); the re-plan that worked was level: pitch -40, heading -129, roll 0, the
fixed jaw 0.8 cm clear. Checked, with no robot and no capture, only the cylinder:

1. Every candidate closes level across the jar: the roll -60 grasp is not in a cylinder's family, and the solved
   gripper keeps the closing line within the approach tolerance of level.
2. The grasp that worked is found as it was planned, with nothing said about pitch, heading or roll.
3. The camera checks: pitch -30 from the robot's front right, the best by the IK alone, is the grasp bowl1 turned
   down because the right wrist camera's plate hid it; it is rejected for that, with the reason.
4. Ranked (jaws not edge-on, no joint near a bound, then how squarely the camera sees the jaws), the best is seen and
   clear of the joints' bounds. It is not yet the grasp that worked (an expected failure): the best come low (pitch
   -16) from the front right, the fixed jaw on the far side, seen at +0.69; the grasp that worked (-0.35) ranks
   third. They cross where the jar's neighbours stood, which the clearance checks (step 3) will judge.

The numbers are bowl1's recorded objects (`harness/scratch/sessions/bowl1`, gitignored), copied here.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("scipy")

from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel  # noqa: E402
from lerobot.felix_jr.harness.grasping import ROTATION_TOLS_DEG, grasp_candidates  # noqa: E402
from lerobot.felix_jr.harness.objects import angles_of  # noqa: E402
from lerobot.felix_jr.harness.primitives import Cylinder  # noqa: E402
from lerobot.felix_jr.harness.reach import ArmIK  # noqa: E402

URDF = Path(__file__).parents[3] / "src" / "lerobot" / "felix_jr" / "urdfs" / "xle_calib_11052026.urdf"
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
    return XleRobotModel.load(URDF)


@pytest.fixture(scope="module")
def found(model):
    # the lid (the top 2 cm), from the side to straight down: as the session would ask
    return grasp_candidates(
        JAR, model, CAPTURE, arms=("right",), where={"t": (0.059, 0.079), "psi": (0.0, 90.0)}, n=200
    )


def test_every_candidate_closes_level(found):
    assert found.passed > 0
    for c in found.candidates:
        assert abs(c.claw.grasp.closing_root[2]) < 1e-9  # the family: square to the vertical axis
        # the solved tip-to-tip line: no more than the approach tolerance off level
        d_root = c.grasp.root_from_pinch[:3, :3] @ (c.claw.aim_pinch / np.linalg.norm(c.claw.aim_pinch))
        assert np.degrees(np.arcsin(abs(d_root[2]))) <= ROTATION_TOLS_DEG[1]  # 4 cm reads at most 4.015 cm


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
    """The grasp ranked best before the camera checks (pitch -30 from the robot's front right) is the one bowl1 turned
    down: the right plate hides it. It is rejected, and says so."""
    (hidden,) = [
        r
        for r in found.rejected
        if dict(r.grasp.params) == {"t": 0.069, "phi": 60.0, "psi": 30.0} and r.fixed_side == 0
    ]
    assert hidden.stage == "camera" and "right wrist camera plate" in hidden.reason
    assert (
        hidden.at_grasp is not None and hidden.at_grasp.ok
    )  # the arm reaches it: only the camera rules it out


def test_the_best_is_seen(model, found):
    best = found.candidates[0]
    assert visible(model, best), best.describe()
    assert not best.near_limit


@pytest.mark.xfail(
    strict=True,
    reason="phase B step 3: the best approach low (pitch -16) from the front right, across the jar's neighbours; "
    "the clearance checks are not in, and whether they or the visibility tier should decide is open",
)
def test_the_best_is_like_the_grasp_that_worked(found):
    best = found.candidates[0]
    pitch, heading, roll = angles_of(best.grasp.root_from_pinch[:3, :3], WORKED[1])
    level = min(_wrap(roll - WORKED[2]), _wrap(roll - WORKED[2] - 180.0))  # either jaw on either side
    assert max(abs(pitch - WORKED[0]), _wrap(heading - WORKED[1]), level) <= 6.0, best.describe()
