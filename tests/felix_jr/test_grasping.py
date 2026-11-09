"""The claw mapping: `to_claw` puts the fixed tip on its contact and the moving tip, at the grasp's opening, on the
other, agreeing with `XleRobotModel.grasp_offset` (`docs/object_grasping.md`)."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("scipy")

from lerobot.felix_jr.calibration.xle_robot_model import (  # noqa: E402
    GripperMap,
    XleRobotModel,
)
from lerobot.felix_jr.harness.execute import prepare_execution  # noqa: E402
from lerobot.felix_jr.harness.grasping import (  # noqa: E402
    ATTACK_M,
    LIFT_M,
    PICK_GRIP,
    SAG_LIFT_M,
    STAGE_UP_M,
    STAGES,
    GraspCandidate,
    GraspSettings,
    grasp_candidates,
    plan_pick,
    to_claw,
)
from lerobot.felix_jr.harness.objects import Object, angles_of  # noqa: E402
from lerobot.felix_jr.harness.primitives import Cylinder  # noqa: E402
from lerobot.felix_jr.harness.reach import select_poses  # noqa: E402

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
    # the tips' middle (the pinch point is on one face of the fingers, `wrist_face` / 2 from their middle)
    middle = model.wrist_face(arm) / 2
    np.testing.assert_allclose(
        at(claw.root_from_pinch, middle), fixed, atol=1e-9
    )  # the fixed tip on its contact
    moving_tip_pinch = model.moving_jaw(arm, model.opening_for_width(arm, g.opening))[:3, 3]
    np.testing.assert_allclose(at(claw.root_from_pinch, moving_tip_pinch + middle), moving, atol=1e-9)
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
    middle = at(claw.root_from_pinch, model.wrist_face("right") / 2)  # the fixed tip, the fingers' middle
    np.testing.assert_allclose(middle, g.contacts_root[0] + margin * outward, atol=1e-9)
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
    keys = [c.rank_key() for c in found.candidates]
    assert keys == sorted(keys)
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


def test_rank_antipodal_first_then_joint_margin():
    def solved(about_approach: float, about_hinge: float):
        return SimpleNamespace(rotation_error_deg=np.array([10.0, about_approach, about_hinge]))

    def candidate(
        about_approach: float, about_hinge: float, limit: float, cost: float = 0.0
    ) -> GraspCandidate:
        return GraspCandidate("", None, None, solved(about_approach, about_hinge), limit, cost)  # type: ignore[arg-type]

    order = [
        candidate(1.0, 2.0, 30.0),  # within the first 5 deg step: the joint margin decides
        candidate(4.0, 0.0, 20.0),
        candidate(0.0, 6.0, 50.0),  # the second step (about the hinge counts too), whatever its joints
        candidate(12.0, 0.0, 60.0),
    ]
    shuffled = [order[i] for i in (3, 0, 2, 1)]
    assert sorted(shuffled, key=GraspCandidate.rank_key) == order
    assert order[2].antipodal_error_deg == 6.0
    coarse = GraspSettings(
        rank_step_deg=10.0
    )  # the caller's step: now the first three tie, the joints decide
    regraded = [GraspCandidate("", None, None, c.grasp, c.limit_margin_deg, 0.0, coarse) for c in order]  # type: ignore[arg-type]
    assert [c.limit_margin_deg for c in sorted(regraded, key=GraspCandidate.rank_key)] == [
        50.0,
        30.0,
        20.0,
        60.0,
    ]


# ---- objects and the pick plan ----------------------------------------------------------------------------------


def test_an_object_names_its_parts():
    jar = Object("jar", {"body": JAR})
    assert jar.part() is JAR and jar.part("body") is JAR
    pan = Object("pan", {"handle": Cylinder.between(np.zeros(3), np.array([0.1, 0, 0]), 0.01), "body": JAR})
    with pytest.raises(ValueError, match="say which"):
        pan.part()
    with pytest.raises(ValueError, match="no part"):
        pan.part("lid")
    with pytest.raises(ValueError, match="no parts"):
        Object("nothing", {})
    assert jar.describe().startswith("jar: body: cylinder")


@pytest.fixture(scope="module")
def gripped():
    with_map = XleRobotModel.load(URDF)
    with_map.gripper_maps["right"] = GripperMap(
        -1.37, 1.2718, {"drive_mode": 0, "homing_offset": -860, "range_min": 2046, "range_max": 3493}
    )
    return with_map


def test_plan_pick_runs_through_the_existing_execution(gripped):
    jar = Object("jar", {"body": JAR})
    one = {"t": (0.069, 0.069), "phi": (60.0, 60.0), "psi": (45.0, 45.0)}  # bowl1's grasp
    found = grasp_candidates(jar, gripped, BOWL1_CAPTURE, arms=("right",), where=one)
    best = found["A"]
    pick = plan_pick(gripped, best, BOWL1_CAPTURE)
    assert pick.labels == ("stage", "pick.back", "pick", "lift") and pick.ok.all()
    assert pick.width == pytest.approx(JAR.radius * 2) and pick.margin == best.claw.margin
    stage, back, at, lift = pick.points
    lifted = best.claw.grasp.center_root + [0.0, 0.0, SAG_LIFT_M]  # aimed high against the arm's sag
    np.testing.assert_allclose(at, lifted, atol=1e-9)  # the aim point on the grasp's center
    np.testing.assert_allclose(back, at - ATTACK_M * best.claw.grasp.approach_root, atol=1e-9)
    np.testing.assert_allclose(stage - back, [0, 0, STAGE_UP_M], atol=1e-9)
    np.testing.assert_allclose(lift - at, [0, 0, LIFT_M], atol=1e-9)
    np.testing.assert_allclose(pick.frames[2], best.grasp.root_from_pinch)  # as grasp_candidates solved it
    whole = prepare_execution(gripped, pick, PICK_GRIP)
    assert whole.moves[0].label.startswith("start: gripper open")
    assert any("pick: gripper close" in move.label for move in whole.moves)
    hold = prepare_execution(gripped, select_poses(pick, "stage", "pick.back"))  # rule 1 stops here
    assert hold.moves[-1].label.startswith("pick.back")


def test_settings_are_the_callers(model):
    """A narrow bound about the approach (a slippery object) rejects what the generous default accepts, and the
    result says what was assumed."""
    one = {"t": (0.069, 0.069), "phi": (30.0, 60.0), "psi": (30.0, 45.0)}
    loose = grasp_candidates(JAR, model, BOWL1_CAPTURE, arms=("right",), where=one)
    narrow = grasp_candidates(
        JAR,
        model,
        BOWL1_CAPTURE,
        arms=("right",),
        where=one,
        settings=GraspSettings(rotation_bounds_deg=(45, 0.01, 45)),
    )
    assert loose.passed > narrow.passed
    assert "approach 0.01" in narrow.describe() and "approach 45" in loose.describe()
    for c in loose.candidates:
        assert c.grasp.ok and max(c.grasp.rotation_error_deg) <= 45.0
