"""The claw mapping: `to_claw` puts the fixed tip on its contact and the moving tip, at the grasp's opening, on the
other, agreeing with `XleRobotModel.grasp_offset` (`docs/object_grasping.md`)."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("scipy")

from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel  # noqa: E402
from lerobot.felix_jr.harness.grasping import (  # noqa: E402
    STAGES,
    GraspCandidate,
    grasp_candidates,
    off_arm_plane_deg,
    to_claw,
)
from lerobot.felix_jr.harness.objects import angles_of  # noqa: E402
from lerobot.felix_jr.harness.primitives import Cylinder  # noqa: E402
from lerobot.felix_jr.harness.reach import ArmIK  # noqa: E402

URDF = Path(__file__).parent / "reference" / "pen1_calibration_10312026.urdf"
JAR = Cylinder.upright(np.array([-0.344, -0.107, 0.714]), 0.079, 0.020)  # bowl1's jar, the hand fit


@pytest.fixture(scope="module")
def model():
    return XleRobotModel.load(URDF)


def jar_grasp(phi: float, psi: float):
    (g,) = JAR.antipodal_grasps(where={"t": (0.07, 0.07), "phi": (phi, phi), "psi": (psi, psi)})
    return g


def at(root_from_frame: np.ndarray, p_frame: np.ndarray) -> np.ndarray:
    return root_from_frame[:3, :3] @ p_frame + root_from_frame[:3, 3]


@pytest.mark.parametrize("arm", ["right", "left"])
@pytest.mark.parametrize("fixed_side", [0, 1])
@pytest.mark.parametrize(("phi", "psi"), [(0.0, 0.0), (30.0, 40.0), (90.0, 90.0), (210.0, -20.0)])
def test_the_tips_land_on_the_contacts(model, arm, fixed_side, phi, psi):
    g = jar_grasp(phi, psi)
    claw = to_claw(g, model, arm, fixed_side)
    rot = claw.root_from_pinch[:3, :3]
    np.testing.assert_allclose(rot.T @ rot, np.eye(3), atol=1e-12)
    assert np.linalg.det(rot) == pytest.approx(1.0)
    fixed, moving = g.contacts_root[fixed_side], g.contacts_root[1 - fixed_side]
    np.testing.assert_allclose(
        claw.root_from_pinch[:3, 3], fixed, atol=1e-9
    )  # the pinch point is the fixed tip
    moving_tip_pinch = model.moving_jaw(arm, model.opening_for_width(arm, g.opening))[:3, 3]
    np.testing.assert_allclose(at(claw.root_from_pinch, moving_tip_pinch), moving, atol=1e-9)
    # the hinge (pinch y) is square to the closing line and the approach: the jaws close in the grasp's plane
    assert rot[:, 1] @ g.closing_root == pytest.approx(0.0, abs=1e-5)
    assert rot[:, 1] @ g.approach_root == pytest.approx(0.0, abs=1e-5)
    # the gripper axis leans off the approach by the tip line's tilt from the pinch -x, toward the fingertips
    tilt = np.degrees(np.arctan2(moving_tip_pinch[2], -moving_tip_pinch[0]))
    assert claw.axis_lean_deg == pytest.approx(abs(tilt), abs=1e-6)


@pytest.mark.parametrize("margin", [0.0, 0.005])
def test_agrees_with_grasp_offset(model, margin):
    g = jar_grasp(0.0, 0.0)
    claw = to_claw(g, model, "right", 0, margin=margin)
    np.testing.assert_allclose(claw.aim_pinch, model.grasp_offset("right", g.opening, margin), atol=1e-12)
    np.testing.assert_allclose(claw.held_pinch, model.grasp_offset("right", g.opening), atol=1e-12)
    np.testing.assert_allclose(at(claw.root_from_pinch, claw.aim_pinch), g.center_root, atol=1e-9)
    # with a margin the fixed tip stands that far off its contact, back along the closing line
    outward = g.contacts_root[0] - g.center_root
    outward /= np.linalg.norm(outward)
    np.testing.assert_allclose(claw.root_from_pinch[:3, 3], g.contacts_root[0] + margin * outward, atol=1e-9)
    assert claw.open_apart == pytest.approx(min(g.opening + margin + 0.02, 2 * model.jaw_radius("right")))


def test_level_grasp_from_the_robots_side(model):
    """bowl1's jar from the robot's side, square to the axis: the jaws close level, the gripper axis points away
    from the robot (heading 180), leaning off it only by the claw's tilt; the two sides swap the jaws."""
    g = jar_grasp(0.0, 0.0)
    rolls = []
    for side in (0, 1):
        claw = to_claw(g, model, "right", side)
        pitch, heading, roll = angles_of(claw.root_from_pinch[:3, :3], 180.0)
        hinge_up = claw.root_from_pinch[:3, 1] @ np.array([0.0, 0.0, 1.0])
        assert abs(hinge_up) == pytest.approx(1.0, abs=1e-9)
        assert pitch == pytest.approx(0.0, abs=1e-3)
        assert abs((heading - 180.0 + 180.0) % 360.0 - 180.0) == pytest.approx(claw.axis_lean_deg, abs=1e-3)
        rolls.append(roll)
    assert abs((rolls[0] - rolls[1]) % 360.0 - 180.0) == pytest.approx(0.0, abs=1e-3)


def test_from_above_points_down(model):
    claw = to_claw(jar_grasp(90.0, 90.0), model, "right", 0)
    pitch, _, _ = angles_of(claw.root_from_pinch[:3, :3])
    assert pitch == pytest.approx(-90.0 + claw.axis_lean_deg, abs=1e-3)
    hinge_up = claw.root_from_pinch[:3, 1] @ np.array([0.0, 0.0, 1.0])
    assert hinge_up == pytest.approx(0.0, abs=1e-5)  # the hinge level: the jaws close horizontally


def test_attack_point_is_back_along_the_approach(model):
    g = jar_grasp(30.0, 40.0)
    claw = to_claw(g, model, "right", 1, attack=0.04)
    np.testing.assert_allclose(claw.root_from_attack[:3, :3], claw.root_from_pinch[:3, :3])
    np.testing.assert_allclose(
        claw.root_from_attack[:3, 3], claw.root_from_pinch[:3, 3] - 0.04 * g.approach_root, atol=1e-12
    )


def test_refuses_a_grasp_wider_than_the_jaws(model):
    wide = Cylinder.upright(np.array([-0.3, 0.0, 0.714]), 0.05, 0.09)
    (g,) = wide.antipodal_grasps(where={"t": (0.03, 0.03), "phi": (0.0, 0.0), "psi": (0.0, 0.0)})
    with pytest.raises(ValueError, match="cannot be"):
        to_claw(g, model, "right", 0)
    with pytest.raises(ValueError, match="fixed_side"):
        to_claw(jar_grasp(0.0, 0.0), model, "right", 2)


# bowl1's robot at its first scan (`harness/scratch/sessions/bowl1/capture1`, gitignored), copied here
BOWL1_ARMS = {
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
}
BOWL1_CAPTURE = SimpleNamespace(arms=BOWL1_ARMS, head_ticks={"head_neck": 1256, "head_eyes": 1329})


def test_candidates_are_ranked_and_named(model):
    found = grasp_candidates(
        JAR,
        model,
        BOWL1_CAPTURE,
        arms=("right",),
        where={"t": (0.069, 0.069), "phi": (30.0, 60.0), "psi": (45.0, 45.0)},
    )
    assert found.tried == 4  # two grasps x the two fixed sides
    assert found.passed + len(found.rejected) == found.tried
    assert all(r.stage in STAGES and r.reason for r in found.rejected)
    assert found.passed >= 1
    assert [c.name for c in found.candidates] == ["A", "B", "C", "D"][: len(found.candidates)]
    costs = [c.cost for c in found.candidates]
    assert costs == sorted(costs)
    for c in found.candidates:
        assert c.attack.ok and c.grasp.ok
    assert found["A"] is found.candidates[0]
    with pytest.raises(KeyError):
        found["Z"]
    text = found.describe()
    assert "A: right arm" in text and "rejected:" in text


def test_too_wide_is_rejected_before_solving(model):
    wide = Cylinder.upright(np.array([-0.3, 0.0, 0.714]), 0.05, 0.09)
    found = grasp_candidates(
        wide, model, BOWL1_CAPTURE, arms=("right",), where={"t": (0.03, 0.03), "psi": (0.0, 0.0)}
    )
    assert found.passed == 0
    assert found.rejected_by_reason() == {"too wide for the jaws": found.tried}
    assert all(r.claw is None and r.stage == "jaws" for r in found.rejected)


def test_off_arm_plane(model):
    ik = ArmIK(model, "right", BOWL1_ARMS)
    pan = model.frame_pose("right_shoulder_link", ik.q)[:3, 3]
    target = np.array([-0.35, -0.10, 0.75])
    radial = (target - pan)[:2] / np.linalg.norm((target - pan)[:2])
    for angle in (0.0, 30.0, 90.0):  # the gripper axis turned out of the plane, horizontal
        a = np.radians(angle)
        axis = np.array([*(np.cos(a) * radial + np.sin(a) * np.array([-radial[1], radial[0]])), 0.0])
        frame = np.eye(4)
        frame[:3, 2], frame[:3, 3] = axis, target
        assert off_arm_plane_deg(model, ik, frame) == pytest.approx(angle, abs=1e-6)
    frame[:3, 2] = [0.0, 0.0, -1.0]  # straight down: any heading will do
    assert off_arm_plane_deg(model, ik, frame) == pytest.approx(0.0, abs=1e-6)


def test_rank_tiers():
    def candidate(visibility: float, limit: float, cost: float) -> GraspCandidate:
        return GraspCandidate("", None, None, None, visibility, limit, cost)  # type: ignore[arg-type]

    best = candidate(-0.35, 20.0, 1.0)
    order = [
        best,
        candidate(-0.36, 20.0, 2.0),  # seen as squarely: the cost decides
        candidate(-0.25, 20.0, 0.0),  # seen less squarely: lower, whatever the cost
        candidate(-0.90, 2.0, 0.0),  # a joint near its bound: below every one that is not
        candidate(+0.05, 30.0, 0.0),  # edge-on: last
    ]
    shuffled = [order[i] for i in (3, 0, 4, 2, 1)]
    assert sorted(shuffled, key=GraspCandidate.rank_key) == order
