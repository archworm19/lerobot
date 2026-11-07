"""The claw mapping: `to_claw` puts the fixed tip on its contact and the moving tip, at the grasp's opening, on the
other, agreeing with `XleRobotModel.grasp_offset` (`docs/object_grasping.md`)."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("scipy")

from lerobot.felix_jr.calibration.xle_robot_model import (  # noqa: E402
    ArmBox,
    GripperMap,
    XleRobotModel,
)
from lerobot.felix_jr.harness.execute import prepare_execution  # noqa: E402
from lerobot.felix_jr.harness.grasping import (  # noqa: E402
    ATTACK_M,
    FINGER_LENGTH_M,
    FINGER_THICKNESS_M,
    FINGER_WIDTH_M,
    LIFT_M,
    PICK_GRIP,
    STAGE_UP_M,
    STAGES,
    Body,
    GraspCandidate,
    GraspSettings,
    arm_boxes,
    finger_boxes,
    grasp_candidates,
    plan_pick,
    refine,
    sight_blockers,
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


def test_a_box_in_the_line_of_sight_hides_a_point(model):
    q = model.joint_vector({"right": BOWL1_ARMS["right"]})
    camera = model.camera_pose(q)[:3, 3]
    point = camera + np.array([-0.3, 0.0, -0.3])
    middle = np.eye(4)
    middle[:3, 3] = 0.5 * (camera + point)
    half = np.array([0.02, 0.05, 0.02])
    beside = middle.copy()
    beside[:3, 3] += [0.0, 0.0, 0.05]
    in_the_way, off = Body("right forearm", middle, half), Body("right forearm", beside, half)
    assert sight_blockers(model, q, [off, in_the_way], {"center": point}) == {"center": "right forearm"}
    assert sight_blockers(model, q, [off], {"center": point}) == {}
    # a plate hides what passes within its margin too
    assert sight_blockers(model, q, [Body("right plate", beside, half, 0.04)], {"center": point}) == {
        "center": "right plate"
    }


# boxes about where `urdfs/xle_calib_11062026.urdf`'s capsules had the right arm's parts (`calibration.arm_shapes`)
RIGHT_BOXES = {
    "gripper_body": ArmBox("right_gripper_link", (-0.002, -0.0017, -0.029), (0.077, 0.045, 0.045)),
    "forearm": ArmBox("right_lower_arm_link", (-0.054, -0.020, 0.020), (0.20, 0.067, 0.067)),
}
LID = {"t": (0.069, 0.069), "psi": (0.0, 90.0)}


@pytest.fixture(scope="module")
def shaped(model):
    return model.with_boxes("right", RIGHT_BOXES)


def test_stand_in_fingers_run_from_the_tips_to_the_body(shaped, model):
    fingers = finger_boxes(shaped, "right")
    fixed, moving = fingers["fixed_finger"], fingers["moving_finger"]
    assert fixed.size[:2] == moving.size[:2] == pytest.approx((FINGER_THICKNESS_M, FINGER_WIDTH_M))
    assert not fixed.moving and moving.moving
    # the inner faces are on the tips' plane (the pinch frame's x = 0), on either side of it, from the tips back
    for box, side in ((fixed, 1.0), (moving, -1.0)):
        assert box.center[0] - side * box.size[0] / 2 == pytest.approx(0.0, abs=1e-12)
        assert box.center[2] + box.size[2] / 2 == pytest.approx(0.0, abs=1e-12)
    assert 0.02 < fixed.size[2] < 0.12  # back to the gripper body's box
    q = shaped.joint_vector()
    root_from_body, half = shaped.boxes_root("right", q)["gripper_body"]
    back = shaped.box_frame(fixed, "right", q) @ np.array([0.0, 0.0, -fixed.size[2] / 2, 1.0])
    gap = np.abs(np.linalg.inv(root_from_body)[:3] @ back) - half  # the fingers' back: on the body's surface
    assert gap.max() == pytest.approx(0.0, abs=0.012)
    assert set(arm_boxes(shaped, "right")) == {"fixed_finger", "moving_finger", "gripper_body", "forearm"}
    assert arm_boxes(model, "right") == {}  # none fitted: none at all
    assert finger_boxes(model, "right")["fixed_finger"].size[2] == FINGER_LENGTH_M


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
def gripped(model):
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
    np.testing.assert_allclose(
        at, best.claw.grasp.center_root, atol=1e-9
    )  # the aim point on the grasp's center
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


def test_refine_aims_to_cancel_the_arms_error(model):
    """At the hold the arm was measured 1 cm low: the same grasp, re-solved from there, is aimed 1 cm higher; the
    object (moved 5 mm) is where the new fit says."""
    one = {"t": (0.069, 0.069), "phi": (60.0, 60.0), "psi": (45.0, 45.0)}  # bowl1's grasp
    found = grasp_candidates(JAR, model, BOWL1_CAPTURE, arms=("right",), where=one)
    chosen = found["A"]
    moved = Cylinder.upright(JAR.bottom_root + np.array([0.005, 0.0, 0.0]), JAR.length, JAR.radius)
    offset = np.array([0.0, 0.0, -0.01])  # measured minus planned: low
    again = refine(model, chosen, moved, BOWL1_CAPTURE, offset)
    assert again.passed == 1 and again["A"].claw.fixed_side == chosen.claw.fixed_side
    expected = chosen.claw.root_from_pinch[:3, 3] + np.array([0.005, 0.0, 0.0]) - offset
    np.testing.assert_allclose(again["A"].claw.root_from_pinch[:3, 3], expected, atol=1e-6)
    np.testing.assert_allclose(
        again["A"].claw.grasp.center_root, chosen.claw.grasp.center_root + [0.005, 0, 0], atol=1e-9
    )
    with pytest.raises(ValueError, match="cylinder"):
        refine(model, chosen, object(), BOWL1_CAPTURE)  # type: ignore[arg-type]
