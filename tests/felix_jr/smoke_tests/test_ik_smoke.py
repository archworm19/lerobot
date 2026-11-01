"""IK smoke test: `harness.reach` on the calibrated URDF, around the cable / chapstick experiments.

`solve_pose` is one local optimization (pitch held, position / heading / roll weighted by tolerances, from `init`;
`docs/harness.md` 1.J). Orchestration (sweeping a pitch, choosing a start) is the caller's, as a model would do it in
a snippet. Checked:

1. Round trip: a full pose read off a random configuration is solved back, starting near it (both arms).
2. Straight down on the experiments' waypoints (2026-10-31, `harness/scratch/sessions/{cable1,chapstick1}`): only the
   lowest are reachable with the pitch held at -90; the others miss with the wrist flex at its limit.
3. A chained pitch sweep finds the steepest reachable pitch of each waypoint (pinned); placo confirms a solution 5 deg
   shallower, along the heading ours found.
4. The cable plan, at a pitch every waypoint allows, prepares into an execution with no joint jumps.
5. A pick and place from the side, the roll held at 0 at both, keeps the carried object upright (the flip found on
   the cable plan: with the roll free, the place had landed on the other side of the jaws).
6. A miss says why.

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
from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel, arm_joint, pinch_frame  # noqa: E402
from lerobot.felix_jr.harness import reach as reach_module  # noqa: E402
from lerobot.felix_jr.harness.execute import held_object, prepare_execution  # noqa: E402
from lerobot.felix_jr.harness.objects import angles_of  # noqa: E402
from lerobot.felix_jr.harness.reach import approach, join_poses, solve_pose  # noqa: E402

URDF = Path(reach_module.__file__).parents[1] / "urdfs" / "xlerobot_calib10312026.urdf"

# the arms at rest, as both sessions' captures saw them (the default start)
REST = {
    "left": {
        "shoulder_pan.pos": -7.385,
        "shoulder_lift.pos": -102.725,
        "elbow_flex.pos": 96.484,
        "wrist_flex.pos": 67.253,
        "wrist_roll.pos": -1.626,
    },
    "right": {
        "shoulder_pan.pos": -0.044,
        "shoulder_lift.pos": -102.989,
        "elbow_flex.pos": 96.527,
        "wrist_flex.pos": 71.341,
        "wrist_roll.pos": -0.22,
    },
}
CAPTURE = SimpleNamespace(arms=REST, head_ticks={})  # what the solver reads from a capture

# every waypoint the experiments proposed (root frame, cm) -> reachable by the right arm pointing straight down?
DOWN_CM = {
    # cable1: the chapstick at (-37.5, 9.8), over the cable to (-37.2, -6.3), then moved closer to (-35.5, -5)
    (-37.5, 9.8, 90.0): False,
    (-37.5, 9.8, 84.0): False,
    (-37.5, 9.8, 82.0): False,
    (-37.5, 9.8, 78.0): True,
    (-37.2, -6.3, 90.0): False,
    (-37.2, -6.3, 84.0): False,
    (-37.2, -6.3, 78.5): False,
    (-35.5, -5.0, 84.0): False,
    (-35.5, -5.0, 82.0): False,
    (-35.5, -5.0, 80.0): False,
    (-35.5, -5.0, 78.5): False,
    # chapstick1: the chapstick at (-38.9, 8.7), past a journal to (-38.5, -17)
    (-38.9, 8.7, 92.0): False,
    (-38.9, 8.7, 78.0): True,
    (-38.5, -17.0, 92.0): False,
    (-38.5, -17.0, 78.0): False,
}

# the steepest pitch a sweep from -90 toward 0 (5 deg steps, each solve from the last) reaches, roll free
PITCHES_DEG = np.arange(-90.0, 0.1, 5.0)
STEEPEST_PITCH_DEG = {
    (-37.5, 9.8, 90.0): -70,
    (-37.5, 9.8, 84.0): -85,
    (-37.5, 9.8, 82.0): -85,
    (-37.5, 9.8, 78.0): -90,
    (-37.2, -6.3, 90.0): -60,
    (-37.2, -6.3, 84.0): -70,
    (-37.2, -6.3, 78.5): -75,
    (-35.5, -5.0, 84.0): -75,
    (-35.5, -5.0, 82.0): -75,
    (-35.5, -5.0, 80.0): -80,
    (-35.5, -5.0, 78.5): -80,
    (-38.9, 8.7, 92.0): -65,
    (-38.9, 8.7, 78.0): -90,
    (-38.5, -17.0, 92.0): -25,
    (-38.5, -17.0, 78.0): -40,
}

# the executed cable plan (plan5), and a pitch all of it allows
CABLE_PLAN_CM = [
    (-37.5, 9.8, 82.0),
    (-37.5, 9.8, 78.0),
    (-37.5, 9.8, 82.0),
    (-35.5, -5.0, 80.0),
    (-35.5, -5.0, 78.5),
    (-35.5, -5.0, 80.0),
]
CABLE_PITCH_DEG = -80.0
MAX_STEP_JUMP_DEG = 8.0  # between 1 cm steps (near the edge of reach the elbow speeds up: ~5.4 deg/cm)

# the chapstick (located in the cable1 session: the middle of the stick) and where it was to go, at the same height
CHAPSTICK_WIDTH, CHAPSTICK_LENGTH = 0.017, 0.065
PICK_CM, PLACE_CM = (-37.27, 9.6, 77.96), (-37.22, -6.35, 77.91)


@pytest.fixture(scope="module")
def model():
    return XleRobotModel.load(URDF)


@pytest.fixture(scope="module")
def placo_robot(tmp_path_factory):
    root = ET.parse(URDF).getroot()
    for joint in root.findall("joint"):
        if joint.get("type") == "continuous":
            joint.set("type", "fixed")
    path = tmp_path_factory.mktemp("placo") / "robot.urdf"
    ET.ElementTree(root).write(path)
    return placo.RobotWrapper(str(path))


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


@pytest.mark.parametrize("waypoint_cm", list(DOWN_CM), ids=lambda w: ",".join(f"{v:g}" for v in w))
def test_straight_down(model, waypoint_cm):
    pose = solve_pose(model, "right", CAPTURE, cm(waypoint_cm), -90.0, roll_deg=None)
    assert bool(pose.ok[0]) == DOWN_CM[waypoint_cm], pose.describe()
    if not pose.ok[0]:
        assert "wrist_flex" in pose.at_limit[0], pose.describe()  # the wrist cannot bend far enough


# ---- 3. steepest pitch ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("waypoint_cm", list(STEEPEST_PITCH_DEG), ids=lambda w: ",".join(f"{v:g}" for v in w))
def test_steepest_pitch_of_a_chained_sweep(model, placo_robot, waypoint_cm):
    target = cm(waypoint_cm)
    reached = sweep(model, target)
    steepest = float(PITCHES_DEG[reached.index(True)])
    assert steepest == STEEPEST_PITCH_DEG[waypoint_cm], reached
    # placo agrees, 5 deg shallower, along the heading ours found there
    pose = solve_pose(model, "right", CAPTURE, target, steepest + 5.0, roll_deg=None)
    assert pose.ok[0], pose.describe()
    p, h = np.radians(steepest + 5.0), np.radians(pose.angles()[0][1])
    axis = np.array([np.cos(p) * np.cos(h), np.cos(p) * np.sin(h), np.sin(p)])
    error, off = placo_ik(placo_robot, "right", pose.frames[0][:3, 3], axis)
    assert error <= reach_module.POSITION_TOL_M and off <= 5.0, (error, off)


# ---- 4. the cable plan --------------------------------------------------------------------------------------------


def test_the_cable_plan_prepares_without_jumps(model):
    # the first pose with the roll free; the rest held at the roll it got (left free, the roll goes anywhere along
    # its free direction from pose to pose, and the steps between swing the wrist)
    first = solve_pose(
        model, "right", CAPTURE, cm(CABLE_PLAN_CM[0]), CABLE_PITCH_DEG, roll_deg=None, label="1"
    )
    roll = first.angles()[0][2]
    parts, last = [first], first
    for i, waypoint in enumerate(CABLE_PLAN_CM[1:], start=2):
        last = solve_pose(
            model, "right", CAPTURE, cm(waypoint), CABLE_PITCH_DEG, roll_deg=roll, init=last, label=str(i)
        )
        parts.append(last)
    plan = join_poses(*parts)
    assert plan.ok.all(), plan.describe()
    execution = prepare_execution(model, plan)
    jumps = [
        max(abs(b[k] - a[k]) for k in a)
        for a, b in zip(execution.steps.joints[:-1], execution.steps.joints[1:], strict=True)
    ]
    assert max(jumps) < MAX_STEP_JUMP_DEG, f"step {int(np.argmax(jumps)) + 2}: {max(jumps):.1f} deg"


# ---- 5. pick and place ----------------------------------------------------------------------------------------------


def test_a_side_pick_and_place_keeps_the_object_upright(model):
    pick = solve_pose(
        model, "right", CAPTURE, cm(PICK_CM), -20.0, roll_deg=0.0, width=CHAPSTICK_WIDTH, label="pick"
    )
    place = solve_pose(
        model,
        "right",
        CAPTURE,
        cm(PLACE_CM),
        -20.0,
        roll_deg=0.0,
        width=CHAPSTICK_WIDTH,
        init=pick,
        label="place",
    )
    path = join_poses(approach(model, pick), approach(model, place, order=("up", "at", "back")))
    assert path.labels == ("pick.back", "pick", "pick.up", "place.up", "place", "place.back")
    assert path.ok.all(), path.describe()
    rolls = [roll for _, _, roll in path.angles()]
    assert max(abs(r) for r in rolls) < 2.0, rolls  # the same side of the jaws throughout: no flip
    tilt = held_object(path, CHAPSTICK_LENGTH, "pick", "place").tilt_deg
    assert np.nanmax(tilt) < 2.0, tilt
    # the stick's middle is between the jaws: half its width from the fixed jaw, toward the moving one
    frame = path.frames[path.labels.index("pick")]
    np.testing.assert_allclose(
        frame[:3, 3] - CHAPSTICK_WIDTH / 2 * frame[:3, 0], cm(PICK_CM), atol=reach_module.POSITION_TOL_M
    )
    execution = prepare_execution(model, path, {"start": "open", "pick": "close", "place": "open"})
    assert [m.label for m in execution.moves if "gripper" in m.label] == [
        "start: gripper open",
        "pick: gripper close",
        "place: gripper open",
    ]


# ---- 6. a miss ------------------------------------------------------------------------------------------------------


def test_a_miss_says_why(model):
    pose = solve_pose(model, "right", CAPTURE, cm((-75.0, 0.0, 78.0)), -45.0)
    assert not pose.ok[0]
    assert "cm off" in pose.describe() and "at limit" in pose.describe()
