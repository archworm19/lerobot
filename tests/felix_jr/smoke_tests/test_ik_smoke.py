"""IK smoke test: `harness.reach` on the calibrated URDF, around the cable / chapstick experiments.

`check_poses` / `approach_poses` are our own IK (scipy least squares on the pinocchio model, analytic Jacobian).
They are checked five ways:

1. Completeness: configurations sampled inside the limits are solved back (FK -> IK round trip): pointing down, at
   whatever pitch the sample has, and with a tool point between open jaws and the jaws across a direction.
2. Against placo on the experiments' waypoints (2026-10-31, `harness/scratch/sessions/{cable1,chapstick1}`): where
   placo finds a down-pointing solution we must too, and our verdicts are pinned. Most waypoints the experiments
   found unreachable are: pointing down at those heights needs the wrist flex past its +-95 deg URDF limit (the
   real servo range is only ~+-98 deg), in both solvers.
3. Pitch: every waypoint is reachable once the gripper may tilt. The steepest reachable pitch of each is pinned, and
   placo confirms a solution 5 deg shallower, along the heading ours found.
4. Continuity: along the densified cable plan with the wrist roll pinned, no joint jumps between steps.
5. `approach_poses` on the experiments: one sweep finds a pitch for the pick and the place together, with the jaws
   closing horizontally across the chapstick, or says which pitches each target would need.

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
from lerobot.felix_jr.harness.reach import (  # noqa: E402
    VERTICAL,
    approach_poses,
    check_poses,
    heading_of,
    pitch_of,
    pitch_ranges,
)

URDF = Path(reach_module.__file__).parents[1] / "urdfs" / "xlerobot_calib10312026.urdf"

# the arms at rest, as both sessions' captures saw them (IK seeds from these)
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
CAPTURE = SimpleNamespace(arms=REST, head_ticks={})  # what the solvers read from a capture

# every waypoint the experiments proposed (root frame, cm) -> does the right arm reach it pointing down?
WAYPOINTS_CM = {
    # cable1: the chapstick at (-37.5, 9.8), over the cable to (-37.2, -6.3), then moved closer to (-35.5, -5)
    (-37.5, 9.8, 90.0): False,
    (-37.5, 9.8, 84.0): True,
    (-37.5, 9.8, 82.0): True,
    (-37.5, 9.8, 78.0): True,
    (-37.2, -6.3, 90.0): False,
    (-37.2, -6.3, 84.0): False,
    (-37.2, -6.3, 78.5): False,
    (-35.5, -5.0, 84.0): False,
    (-35.5, -5.0, 82.0): False,
    (-35.5, -5.0, 80.0): True,
    (-35.5, -5.0, 78.5): True,
    # chapstick1: the chapstick at (-38.9, 8.7), past a journal to (-38.5, -17)
    (-38.9, 8.7, 92.0): False,
    (-38.9, 8.7, 78.0): True,
    (-38.5, -17.0, 92.0): False,
    (-38.5, -17.0, 78.0): False,
}

# the executed cable plan (plan5)
CABLE_PLAN_CM = [
    (-37.5, 9.8, 82.0),
    (-37.5, 9.8, 78.0),
    (-37.5, 9.8, 82.0),
    (-35.5, -5.0, 80.0),
    (-35.5, -5.0, 78.5),
    (-35.5, -5.0, 80.0),
]
CABLE_ROLL_DEG = 130.0  # the wrist roll that keeps all of it reachable
MAX_STEP_JUMP_DEG = 5.0  # between 1 cm steps

# the right arm's steepest reachable pitch at each waypoint, on a 5 deg grid from -90 (down) to 0 (horizontal)
PITCHES_DEG = np.arange(-90.0, 0.1, 5.0)
STEEPEST_PITCH_DEG = {
    (-37.5, 9.8, 90.0): -75,
    (-37.5, 9.8, 84.0): -90,
    (-37.5, 9.8, 82.0): -90,
    (-37.5, 9.8, 78.0): -90,
    (-37.2, -6.3, 90.0): -65,
    (-37.2, -6.3, 84.0): -75,
    (-37.2, -6.3, 78.5): -85,
    (-35.5, -5.0, 84.0): -80,
    (-35.5, -5.0, 82.0): -85,
    (-35.5, -5.0, 80.0): -90,
    (-35.5, -5.0, 78.5): -90,
    (-38.9, 8.7, 92.0): -70,
    (-38.9, 8.7, 78.0): -90,
    (-38.5, -17.0, 92.0): -35,
    (-38.5, -17.0, 78.0): -50,
}

# the chapsticks (located in the sessions: the middle of the stick) and where they were to go, at the same height
CHAPSTICK_WIDTH = 0.017
CABLE_PICK_CM, CABLE_PLACE_CM = (-37.27, 9.6, 77.96), (-37.22, -6.35, 77.91)
CHAPSTICK_PICK_CM, CHAPSTICK_PLACE_CM = (-38.61, 8.64, 77.95), (-38.5, -17.0, 77.95)
DOWN = np.array([0.0, 0.0, -1.0])


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


def tilt_deg(pose: np.ndarray) -> float:
    """How far the pinch frame's +z is from straight down."""
    return float(np.degrees(np.arccos(np.clip(-pose[2, 2], -1.0, 1.0))))


def reaches(error_m: float, off_deg: float) -> bool:
    return error_m <= reach_module.POSITION_TOL_M and off_deg <= reach_module.PITCH_TOL_DEG


def placo_ik(
    robot, arm: str, target: np.ndarray, axis: np.ndarray = DOWN, starts: int = 8
) -> tuple[float, float, np.ndarray]:
    """Placo's best (position error m, axis error deg, joints rad) for the pinch point at `target`, its +z along `axis`."""
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
            np.array([robot.get_joint(j) for j in joints]),
        )
        if best is None or (reaches(*result[:2]), -result[0]) > (reaches(*best[:2]), -best[0]):
            best = result
    assert best is not None
    return best


def densify(waypoints: np.ndarray, step: float) -> np.ndarray:
    points = [waypoints[0]]
    for a, b in zip(waypoints[:-1], waypoints[1:], strict=True):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        points += [a + (b - a) * k / n for k in range(1, n + 1)]
    return np.array(points)


def random_configs(model, arm: str, n: int, rng) -> list[np.ndarray]:
    """`n` configurations inside the solver's limits."""
    index = [model.q_index(arm_joint(arm, j)) for j in SO101_JOINTS]
    limits = np.array([model.limits(arm_joint(arm, j)) for j in SO101_JOINTS]) * reach_module.LIMIT_MARGIN
    roll_cap = np.radians(reach_module.MAX_ROLL_DEG)
    limits[4] = np.clip(limits[4], -roll_cap, roll_cap)
    configs = []
    for _ in range(n):
        q = model.joint_vector()
        q[index] = rng.uniform(limits[:, 0], limits[:, 1])
        configs.append(q)
    return configs


def down_pointing_configs(model, arm: str, n: int, rng) -> list[np.ndarray]:
    """`n` configurations inside the solver's limits whose gripper points down (wrist flex chosen to make it so)."""
    flex = model.q_index(arm_joint(arm, "wrist_flex"))
    lower, upper = np.array(model.limits(arm_joint(arm, "wrist_flex"))) * reach_module.LIMIT_MARGIN
    candidates = np.linspace(lower, upper, 721)
    configs: list[np.ndarray] = []
    while len(configs) < n:
        q = random_configs(model, arm, 1, rng)[0]
        tilts = []
        for value in candidates:
            q[flex] = value
            tilts.append(tilt_deg(model.frame_pose(pinch_frame(arm), q)))
        if min(tilts) < 0.5:
            q[flex] = candidates[int(np.argmin(tilts))]
            configs.append(q)
    return configs


def max_jump_deg(joints: tuple[dict[str, float], ...]) -> tuple[float, int]:
    """The largest joint change between consecutive poses, and the 1-based index of the pose it lands on."""
    jumps = [max(abs(b[k] - a[k]) for k in a) for a, b in zip(joints[:-1], joints[1:], strict=True)]
    return max(jumps), int(np.argmax(jumps)) + 2


# ---- 1. completeness -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("arm", ["right", "left"])
def test_down_pointing_configurations_are_solved_back(model, arm):
    for q in down_pointing_configs(model, arm, 15, np.random.default_rng(7)):
        target = model.frame_pose(pinch_frame(arm), q)[:3, 3]
        poses = check_poses(model, arm, CAPTURE, target, across=None)
        assert poses.ok[0], f"missed {np.round(100 * target, 1)} cm: {poses.describe()}"


@pytest.mark.parametrize("arm", ["right", "left"])
def test_configurations_are_solved_back_at_their_own_pitch(model, arm):
    for q in random_configs(model, arm, 15, np.random.default_rng(8)):
        pose = model.frame_pose(pinch_frame(arm), q)
        poses = check_poses(model, arm, CAPTURE, pose[:3, 3], pitch_deg=pitch_of(pose), across=None)
        assert poses.ok[0], (
            f"missed {np.round(100 * pose[:3, 3], 1)} cm at pitch {pitch_of(pose):.0f}: {poses.describe()}"
        )


@pytest.mark.parametrize("arm", ["right", "left"])
def test_tool_point_between_open_jaws_and_jaws_across_are_solved_back(model, arm):
    rng = np.random.default_rng(9)
    width = 0.03
    for q in random_configs(model, arm, 15, rng):
        pose = model.frame_pose(pinch_frame(arm), q)
        tool = pose[:3, 3] - width / 2 * pose[:3, 0]  # the moving jaw opens along -x
        across = np.cross(pose[:3, 0], rng.normal(size=3))  # some direction the sample's jaws are across
        poses = check_poses(model, arm, CAPTURE, tool, pitch_of(pose), width=width, across=across)
        assert poses.ok[0], poses.describe()
        solved = poses.frames[0]
        np.testing.assert_allclose(
            solved[:3, 3] - width / 2 * solved[:3, 0], tool, atol=reach_module.POSITION_TOL_M
        )


# ---- 2. against placo, pointing down ----------------------------------------------------------------------------


@pytest.mark.parametrize("waypoint_cm", list(WAYPOINTS_CM), ids=lambda w: ",".join(f"{v:g}" for v in w))
def test_experiment_waypoints_agree_with_placo(model, placo_robot, waypoint_cm):
    target = cm(waypoint_cm)
    poses = check_poses(model, "right", CAPTURE, target, across=None)
    ours = bool(poses.ok[0])
    error, tilt, joints = placo_ik(placo_robot, "right", target)
    # placo's answer, re-checked on our model (both load the same calibrated URDF)
    q = model.joint_vector(
        {"right": {f"{j}.pos": np.degrees(v) for j, v in zip(SO101_JOINTS, joints, strict=True)}}
    )
    np.testing.assert_allclose(model.frame_pose(pinch_frame("right"), q)[:3, 3], target, atol=error + 1e-6)
    if reaches(error, tilt):
        assert ours, f"placo reaches it (error {100 * error:.2f} cm, tilt {tilt:.1f} deg) but we do not"
    assert ours == WAYPOINTS_CM[waypoint_cm], poses.describe()


def test_unreachable_waypoints_hit_the_wrist_flex_limit(model, placo_robot):
    flex = SO101_JOINTS.index("wrist_flex")
    lower, upper = model.limits(arm_joint("right", "wrist_flex"))
    for waypoint_cm, expected in WAYPOINTS_CM.items():
        error, tilt, joints = placo_ik(placo_robot, "right", cm(waypoint_cm))
        if not expected and not reaches(error, tilt):
            assert min(joints[flex] - lower, upper - joints[flex]) < np.radians(0.5), waypoint_cm


# ---- 3. pitch -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("waypoint_cm", list(STEEPEST_PITCH_DEG), ids=lambda w: ",".join(f"{v:g}" for v in w))
def test_steepest_reachable_pitch(model, placo_robot, waypoint_cm):
    target = cm(waypoint_cm)
    sweep = [check_poses(model, "right", CAPTURE, target, pitch_deg=p, across=None) for p in PITCHES_DEG]
    ranges = pitch_ranges(PITCHES_DEG, [bool(s.ok[0]) for s in sweep])
    assert len(ranges) == 1, f"the reachable pitches should be one range: {ranges}"
    assert ranges[0][0] == STEEPEST_PITCH_DEG[waypoint_cm], ranges
    # placo agrees, 5 deg shallower than our steepest, along the heading ours found there
    pitch = ranges[0][0] + 5.0
    frame = sweep[list(PITCHES_DEG).index(pitch)].frames[0]
    p, h = np.radians(pitch), np.radians(heading_of(frame))
    axis = np.array([np.cos(p) * np.cos(h), np.cos(p) * np.sin(h), np.sin(p)])
    error, off, _ = placo_ik(placo_robot, "right", target, axis)
    assert reaches(error, off), (
        f"placo at pitch {pitch:.0f}: error {100 * error:.2f} cm, axis off {off:.1f} deg"
    )


# ---- 4. continuity ------------------------------------------------------------------------------------------------


def test_cable_plan_is_continuous_with_the_roll_pinned(model):
    # A grasp from above, as planned: any jaw direction. Pointing down, the roll is nearly free, and solving one
    # pose at a time cannot look ahead: left free, it relaxes on the way down and has to swing back ~50 deg in the
    # last centimeter up, where the off-axis pinch point needs it near its cap. So the plan pins it (`execute
    # --roll-deg`). With the jaws held horizontal instead, two steps at that edge miss by 0.5 cm.
    steps = densify(cm(CABLE_PLAN_CM), 0.01)
    poses = check_poses(model, "right", CAPTURE, steps, across=None, roll_deg=CABLE_ROLL_DEG)
    assert poses.ok.all(), poses.describe()
    jump, at = max_jump_deg(poses.joints)
    assert jump < MAX_STEP_JUMP_DEG, f"step {at}: {jump:.1f} deg"


# ---- 5. approach_poses --------------------------------------------------------------------------------------------


def test_side_pick_and_place_of_the_cable_chapstick(model):
    poses = approach_poses(
        model, "right", CAPTURE, [cm(CABLE_PICK_CM), cm(CABLE_PLACE_CM)], (0, -90), width=CHAPSTICK_WIDTH
    )
    assert poses.pitch_deg is not None and poses.ok.all(), poses.describe()
    assert poses.labels == ("back1", "at1", "up1", "up2", "at2", "back2")
    assert poses.pitch_deg == -20.0  # from the side: -15 works too, but -10 does not, so -20 keeps a margin
    for frame in poses.frames:
        assert abs(frame[:3, 0] @ VERTICAL) < np.sin(np.radians(reach_module.JAW_TOL_DEG))  # jaws horizontal
    at1 = poses.frames[poses.labels.index("at1")]
    # the stick's middle is between the jaws: half its width from the fixed jaw, toward the moving one
    np.testing.assert_allclose(
        at1[:3, 3] - CHAPSTICK_WIDTH / 2 * at1[:3, 0], cm(CABLE_PICK_CM), atol=reach_module.POSITION_TOL_M
    )
    tool = at1[:3, 3] - CHAPSTICK_WIDTH / 2 * at1[:3, 0]  # where the solved pose put the tool point
    np.testing.assert_allclose((tool - poses.points[0]) / 0.05, at1[:3, 2], atol=1e-9)  # back along the axis
    np.testing.assert_allclose(poses.points[2] - tool, [0, 0, 0.05], atol=1e-12)  # then straight up
    assert "-20 deg" in poses.describe()


def test_one_sweep_finds_a_pitch_for_pick_and_place_together(model):
    # chapstick1 failed pointing down; a place-only sweep would need -45 or shallower, the pick prefers steep
    poses = approach_poses(
        model,
        "right",
        CAPTURE,
        [cm(CHAPSTICK_PICK_CM), cm(CHAPSTICK_PLACE_CM)],
        (-90, 0),
        width=CHAPSTICK_WIDTH,
    )
    assert poses.pitch_deg is not None and poses.ok.all(), poses.describe()
    pick, place = poses.target_reachable_deg
    assert pick[0][0] == -90.0  # the pick alone could go straight down
    assert place[0][0] > -90.0  # the place cannot: it limits the shared range
    assert poses.reachable_deg[0][0] == place[0][0]
    assert poses.pitch_deg == place[0][0] + 5.0  # the steepest shared pitch, one step in from the edge


def test_no_shared_pitch_says_what_each_target_needs(model):
    too_far = cm((-75.0, 0.0, 78.0))
    poses = approach_poses(
        model, "right", CAPTURE, [cm(CABLE_PICK_CM), too_far], (-90, -60), width=CHAPSTICK_WIDTH
    )
    assert poses.pitch_deg is None and len(poses.points) == 0
    pick, far = poses.target_reachable_deg
    assert pick and not far
    assert (
        "no pitch in -90..-60" in poses.describe()
        and "target 2 (-75.0, 0.0, 78.0) cm: none" in poses.describe()
    )
