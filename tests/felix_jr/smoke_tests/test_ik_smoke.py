"""IK smoke test: `harness.reach` on the calibrated URDF, around the pen1 session (`docs/sessions.md`).

pen1 (2026-11-01): the right arm picks a pen leaning in one glass cup and puts it in the other, from the side, the
roll held at 0. `solve_pose` is one local optimization (pitch held, position / heading / roll weighted by tolerances,
from `init`; `docs/harness.md` 1.J). Orchestration (sweeping a pitch, choosing a start) is the caller's, as a model
would do it in a snippet. Checked:

1. Round trip: a full pose read off a random configuration is solved back, starting near it (both arms).
2. Straight down, pen1's waypoints are all out of reach: the wrist flex is at its limit.
3. A chained pitch sweep finds the steepest reachable pitch of each waypoint (pinned); placo confirms a solution 5 deg
   shallower, along the heading ours found.
4. The session's pitch window: perpendicular to the pen (-36) reaches the pick but not the lift or the place; the
   pitches it used (-15, then -12) reach all three.
5. The executed plan (attempt 1) is solved again as recorded and prepares into an execution with no joint jumps.
6. The pen turns with the gripper: carried rigidly, its bottom lands on the empty cup's center (where the session
   aimed it), clears the rim on the way, and would miss by centimeters had the place been aimed as if it did not turn.
7. A miss says why.

The numbers are pen1's recorded objects (`harness/scratch/sessions/pen1`, gitignored), copied here.

Placo mishandles the URDF's continuous joints (the wheels: two coordinates but one velocity): its joint limits land on
the wrong dofs and the QP goes infeasible after a step or two. The copy placo loads has the wheels fixed.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("scipy")
placo = pytest.importorskip("placo")

from lerobot.felix_jr.calibration.merge_urdf import SO101_JOINTS  # noqa: E402
from lerobot.felix_jr.calibration.xle_robot_model import (  # noqa: E402
    GripperMap,
    XleRobotModel,
    arm_joint,
    pinch_frame,
)
from lerobot.felix_jr.harness import reach as reach_module  # noqa: E402
from lerobot.felix_jr.harness.execute import held_object, prepare_execution  # noqa: E402
from lerobot.felix_jr.harness.objects import angles_of  # noqa: E402
from lerobot.felix_jr.harness.reach import approach, join_poses, solve_pose  # noqa: E402

# the calibration pen1 was planned with (frozen: its numbers only hold under it)
URDF = Path(__file__).parents[1] / "reference" / "pen1_calibration_10312026.urdf"
# the right gripper's map (calibration.gripper_step, 2026-11-04)
GRIPPER = GripperMap(
    pos_closed=-1.37,
    deg_per_unit=1.2718,
    calibration={"drive_mode": 0, "homing_offset": -860, "range_min": 2046, "range_max": 3493},
    radius_m=0.0855,
)

# the arms at rest, as pen1's capture4 saw them (the default start)
REST = {
    "left": {
        "shoulder_pan.pos": -7.385,
        "shoulder_lift.pos": -102.725,
        "elbow_flex.pos": 96.571,
        "wrist_flex.pos": 67.253,
        "wrist_roll.pos": -1.626,
    },
    "right": {
        "shoulder_pan.pos": 2.154,
        "shoulder_lift.pos": -102.989,
        "elbow_flex.pos": 96.527,
        "wrist_flex.pos": 71.429,
        "wrist_roll.pos": -0.22,
    },
}
CAPTURE = SimpleNamespace(arms=REST, head_ticks={})  # what the solver reads from a capture

# attempt 1 (root frame, cm): the grasp on the pen, a 14.2 cm lift (the pen's bottom over the higher rim), and the
# place, aimed so the pen's bottom lands in the empty cup; jaws 1 cm, pitch -15, roll 0
PICK_CM, PLACE_CM = (-45.38, 14.21, 86.32), (-45.98, -1.3, 97.22)
LIFT_M, PLACE_DROP_M = 0.142, 0.033
PITCH_DEG, WIDTH = -15.0, 0.01
WAYPOINTS_CM = {
    "pick": PICK_CM,
    "pick.up": (PICK_CM[0], PICK_CM[1], PICK_CM[2] + LIFT_M * 100),
    "place": PLACE_CM,
}

# the plan as executed (poses146)
PLAN_CM = {
    "pick.back": (-40.55, 14.02, 87.62),
    "pick": PICK_CM,
    "pick.up": (-45.38, 14.21, 100.52),
    "place.up": (-45.98, -1.3, 100.52),
    "place": PLACE_CM,
    "place.back": (-41.53, 0.56, 98.52),
}
# a branch switch jumps tens of degrees; at place.up (the edge of reach) the elbow turns ~14 deg / cm of descent
MAX_STEP_JUMP_DEG = 15.0

# the steepest pitch a sweep from -90 toward 0 (5 deg steps, each solve from the last) reaches, roll free
PITCHES_DEG = np.arange(-90.0, 0.1, 5.0)
STEEPEST_PITCH_DEG = {"pick": -65, "pick.up": -30, "place": -30}

# the pen: its axis from a line fit through depth samples along it (the end up first, leaning ~35 deg away from the
# robot), its bottom 15 cm below the grasp; the empty cup's center at the height the bottom was aimed, and its rim
PEN_AXIS = (-0.551690610388469, 0.20228655714243862, 0.8091462285697545)
PEN_BELOW_GRASP = 0.15
CUP_CM = (-37.2, -0.57, 85.08)
EMPTY_RIM_Z = 0.872


@pytest.fixture(scope="module")
def model():
    m = XleRobotModel.load(URDF)
    m.gripper_maps["right"] = GRIPPER  # measured after pen1; the gripper's servo has not changed since
    return m


@pytest.fixture(scope="module")
def placo_robot(tmp_path_factory):
    root = ET.parse(URDF).getroot()
    for joint in root.findall("joint"):
        if joint.get("type") == "continuous":
            joint.set("type", "fixed")
    path = tmp_path_factory.mktemp("placo") / "robot.urdf"
    ET.ElementTree(root).write(path)
    return placo.RobotWrapper(str(path))


@pytest.fixture(scope="module")
def plan(model):
    """pen1's attempt 1, solved as the session did: the pick, the place from it, and their approaches."""
    pick = solve_pose(
        model, "right", CAPTURE, cm(PICK_CM), PITCH_DEG, roll_deg=0.0, width=WIDTH, label="pick"
    )
    place = solve_pose(
        model,
        "right",
        CAPTURE,
        cm(PLACE_CM),
        PITCH_DEG,
        roll_deg=0.0,
        width=WIDTH,
        init=pick,
        label="place",
    )
    return join_poses(
        approach(model, pick, up=LIFT_M),
        approach(model, place, up=PLACE_DROP_M, order=("up", "at", "back")),
    )


def cm(xyz) -> np.ndarray:
    return np.asarray(xyz, dtype=np.float64) / 100.0


def placo_ik(robot, arm: str, target: np.ndarray, axis: np.ndarray, starts: int = 8) -> tuple[float, float]:
    """Placo's best (position error m, axis error deg) for the pinch point at `target`, its +z along `axis`."""
    joints = [arm_joint(arm, j) for j in SO101_JOINTS]
    limits = np.array([robot.get_joint_limits(j) for j in joints])
    rng = np.random.default_rng(0)
    best = None
    for start in range(starts):
        seed = rng.uniform(limits[:, 0], limits[:, 1]) if start else np.zeros(len(joints))
        solver = placo.KinematicsSolver(robot)
        solver.mask_fbase(True)
        for name in robot.joint_names():
            if name not in joints:
                solver.mask_dof(name)
        solver.enable_joint_limits(True)
        for name, value in zip(joints, seed, strict=True):
            robot.set_joint(name, value)
        robot.update_kinematics()
        solver.add_position_task(pinch_frame(arm), target).configure("position", "soft", 1.0)
        solver.add_axisalign_task(pinch_frame(arm), np.array([0.0, 0.0, 1.0]), axis).configure(
            "axis", "soft", 0.1
        )
        for _ in range(150):
            solver.solve(True)
            robot.update_kinematics()
        pose = robot.get_T_world_frame(pinch_frame(arm))
        result = (
            float(np.linalg.norm(pose[:3, 3] - target)),
            float(np.degrees(np.arccos(np.clip(pose[:3, 2] @ axis, -1.0, 1.0)))),
        )
        if best is None or result[0] + result[1] / 1000 < best[0] + best[1] / 1000:
            best = result
    assert best is not None
    return best


def sweep(model, target: np.ndarray) -> list[bool]:
    """Whether each of PITCHES_DEG reaches `target` (roll free), each solve starting from the last: a model's loop."""
    reached, last = [], None
    for pitch in PITCHES_DEG:
        last = solve_pose(model, "right", CAPTURE, target, pitch, roll_deg=None, init=last)
        reached.append(bool(last.ok[0]))
    return reached


def pen_bottom(plan, label: str) -> np.ndarray:
    """Where the pen's bottom is at pose `label`, carried rigidly from the pick."""
    pen = held_object(plan, PEN_BELOW_GRASP, "pick", "place", axis=PEN_AXIS)
    i = plan.labels.index(label)
    return pen.centers[i] - PEN_BELOW_GRASP * pen.axes[i]


# ---- 1. round trip ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("arm", ["right", "left"])
def test_a_full_pose_is_solved_back_from_nearby(model, arm):
    rng = np.random.default_rng(9)
    index = [model.q_index(arm_joint(arm, j)) for j in SO101_JOINTS]
    limits = np.array([model.limits(arm_joint(arm, j)) for j in SO101_JOINTS]) * reach_module.LIMIT_MARGIN
    limits[4] = np.clip(limits[4], -np.radians(125), np.radians(125))
    width = 0.03
    for _ in range(15):
        x = rng.uniform(limits[:, 0], limits[:, 1])
        q = model.joint_vector()
        q[index] = x
        frame = model.frame_pose(pinch_frame(arm), q)
        tool = frame[:3, 3] - width / 2 * frame[:3, 0]
        pitch, heading, roll = angles_of(frame[:3, :3])
        init = {
            f"{j}.pos": float(np.degrees(v))
            for j, v in zip(SO101_JOINTS, x + rng.normal(0, 0.15, 5), strict=True)
        }
        pose = solve_pose(
            model, arm, CAPTURE, tool, pitch, heading, roll, width, heading_tol_deg=2.0, init=init
        )
        assert pose.ok[0], pose.describe()


# ---- 2. straight down ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("label", list(WAYPOINTS_CM))
def test_straight_down_is_out_of_reach(model, label):
    pose = solve_pose(model, "right", CAPTURE, cm(WAYPOINTS_CM[label]), -90.0, roll_deg=None)
    assert not pose.ok[0], pose.describe()
    assert "wrist_flex" in pose.at_limit[0], pose.describe()  # the wrist cannot bend far enough


# ---- 3. steepest pitch ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("label", list(STEEPEST_PITCH_DEG))
def test_steepest_pitch_of_a_chained_sweep(model, placo_robot, label):
    target = cm(WAYPOINTS_CM[label])
    reached = sweep(model, target)
    steepest = float(PITCHES_DEG[reached.index(True)])
    assert steepest == STEEPEST_PITCH_DEG[label], reached
    # placo agrees, 5 deg shallower, along the heading ours found there
    pose = solve_pose(model, "right", CAPTURE, target, steepest + 5.0, roll_deg=None)
    assert pose.ok[0], pose.describe()
    p, h = np.radians(steepest + 5.0), np.radians(pose.angles()[0][1])
    axis = np.array([np.cos(p) * np.cos(h), np.cos(p) * np.sin(h), np.sin(p)])
    error, off = placo_ik(placo_robot, "right", pose.frames[0][:3, 3], axis)
    assert error <= reach_module.POSITION_TOL_M and off <= 5.0, (error, off)


# ---- 4. the pitch window, roll 0 ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pitch_deg", "reached"),
    [(-36.0, {"pick"}), (-15.0, set(WAYPOINTS_CM)), (-12.0, set(WAYPOINTS_CM))],
)
def test_the_pitch_window_of_a_side_grasp(model, pitch_deg, reached):
    found = {
        label
        for label, waypoint in WAYPOINTS_CM.items()
        if solve_pose(model, "right", CAPTURE, cm(waypoint), pitch_deg, roll_deg=0.0, width=WIDTH).ok[0]
    }
    assert found == reached


# ---- 5. the executed plan -------------------------------------------------------------------------------------------


def test_the_plan_is_solved_as_recorded_and_prepares_without_jumps(model, plan):
    assert plan.labels == tuple(PLAN_CM)
    assert plan.ok.all(), plan.describe()
    np.testing.assert_allclose(plan.points, cm(list(PLAN_CM.values())), atol=0.001)
    execution = prepare_execution(model, plan, {"start": "open", "pick": "close", "place": "open"})
    tols = {m.label.split(" ")[0]: m.tol for m in execution.moves}
    assert tols["pick"] == 2.5 and tols["step"] == 4.0  # the labeled poses tighter than the steps between
    jumps = [
        max(abs(b[k] - a[k]) for k in a)
        for a, b in zip(execution.steps.joints[:-1], execution.steps.joints[1:], strict=True)
    ]
    assert max(jumps) < MAX_STEP_JUMP_DEG, f"step {int(np.argmax(jumps)) + 2}: {max(jumps):.1f} deg"
    grips = [m for m in execution.moves if "gripper" in m.label]
    assert [m.label for m in grips] == [
        "start: gripper open (tips 3.0 cm apart)",  # the 1 cm pen, plus the clearance
        "pick: gripper close (tips 0.7 cm apart)",  # the pen, less the squeeze
        "place: gripper open (tips 3.0 cm apart)",
    ]
    opening, closing = grips[0].targets["gripper.pos"], grips[1].targets["gripper.pos"]
    assert opening == pytest.approx(GRIPPER.pos_for(model.opening_for_width("right", 0.03)))
    assert 0.0 < closing < opening < 25.0  # not pen1's guessed 60-80


# ---- 6. carrying the pen --------------------------------------------------------------------------------------------


def test_the_pen_turns_with_the_gripper_and_lands_in_the_cup(plan):
    rolls = [roll for _, _, roll in plan.angles()]
    assert max(abs(r) for r in rolls) < 2.0, rolls  # the same side of the jaws throughout: no flip
    headings = dict(zip(plan.labels, (heading for _, heading, _ in plan.angles()), strict=True))
    turn = (headings["place"] - headings["pick"] + 180.0) % 360.0 - 180.0
    assert 20.0 < abs(turn) < 30.0, headings  # the heading follows the shoulder pan from cup to cup
    # carried rigidly, the bottom lands on the cup's center; aimed as if the pen did not turn, it would miss
    np.testing.assert_allclose(pen_bottom(plan, "place"), cm(CUP_CM), atol=0.005)
    unturned = cm(PLACE_CM) - PEN_BELOW_GRASP * np.asarray(PEN_AXIS)
    assert np.linalg.norm(unturned[:2] - cm(CUP_CM)[:2]) > 0.03
    # the carry keeps the bottom over the empty cup's rim
    for label in ("pick.up", "place.up"):
        assert pen_bottom(plan, label)[2] > EMPTY_RIM_Z + 0.01, label


# ---- 7. a miss ------------------------------------------------------------------------------------------------------


def test_a_miss_says_why(model):
    pose = solve_pose(model, "right", CAPTURE, cm((-75.0, 0.0, 78.0)), -45.0)
    assert not pose.ok[0]
    assert "cm off" in pose.describe() and "at limit" in pose.describe()
