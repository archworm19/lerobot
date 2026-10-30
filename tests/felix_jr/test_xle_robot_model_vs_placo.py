"""Cross-check of `XleRobotModel` (pinocchio) against placo, which LeRobot's `RobotKinematics` uses.

The merged URDF loads in placo as it is: it has no meshes, and its root joint is fixed (placo adds its own free
base, and a second floating joint in the URDF scrambles its joint indexing). Corrections reach placo by an
independent route: `to_urdf` writes them into the joint origins, and placo loads that text.
"""

import numpy as np
import pytest

pytest.importorskip("pinocchio")
placo = pytest.importorskip("placo")

from lerobot.felix_jr.calibration.merge_urdf import MERGED_URDF, OPTICAL_FRAME, SO101_JOINTS  # noqa: E402
from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel, pinch_frame  # noqa: E402

SIGNS = {"head_neck": -1, "head_eyes": 1}
REFS = {"head_neck": 1520, "head_eyes": 1324}
FRAMES = [
    pinch_frame("left"),
    pinch_frame("right"),
    "left_wrist_link",
    "right_moving_jaw_so101_v1_link",
    "head_tilt_link",
    OPTICAL_FRAME,
]
MOVING = [
    "head_pan_joint",
    "head_tilt_joint",
    *(f"{a}_{j}" for a in ("left", "right") for j in (*SO101_JOINTS, "gripper")),
]
TOLERANCE = 1e-9

# one representative correction of every kind and chain: head joints, camera mount, arm mounts, arm joints (zero
# offsets, origin positions) and the pinch point offset; fixed joints (which pinocchio folds into frames) included
CORRECTIONS = [
    {"zero:head_pan_joint": np.radians(2.0)},
    {"zero:head_tilt_joint": np.radians(-20.0)},
    {"pos:head_tilt_joint:z": -0.003},
    {"pos:fixed_head_camera_link:x": 0.005, "pos:fixed_head_camera_link:y": 0.02},
    {"rot:fixed_head_camera_link:x": np.radians(2.0), "rot:fixed_head_camera_link:z": np.radians(-3.0)},
    {"pos:fixed_Base:x": 0.003, "rot:fixed_Base:y": np.radians(1.0)},
    {"pos:fixed_Base_2:z": -0.01, "rot:fixed_Base_2:z": np.radians(-4.0)},
    {"zero:left_shoulder_pan": np.radians(2.0), "zero:right_elbow_flex": np.radians(-3.0)},
    {"pos:left_elbow_flex:y": 0.002, "pos:right_wrist_flex:z": -0.002},
    {"pos:left_gripper_frame_joint:z": -0.009, "pos:right_gripper_frame_joint:x": 0.004},
]


@pytest.fixture(scope="module")
def model():
    return XleRobotModel.from_urdf(head_signs=SIGNS, head_refs=REFS)


def placo_robot(text: str, path):
    path.write_text(text)
    return placo.RobotWrapper(str(path))


def random_q(model, rng) -> tuple[np.ndarray, dict[str, float]]:
    """A random configuration within the joint limits, as pinocchio's `q` and as `{joint: angle}` for placo."""
    angles = {j: float(rng.uniform(*model.limits(j))) for j in MOVING}
    q = model.joint_vector()
    for joint, angle in angles.items():
        q[model.q_index(joint)] = angle
    return q, angles


def placo_pose(robot, frame: str, angles: dict[str, float]) -> np.ndarray:
    for name, value in angles.items():
        robot.set_joint(name, value)
    robot.update_kinematics()
    return np.array(robot.get_T_world_frame(frame))


def test_every_moving_joint_at_random_positions(model, tmp_path):
    robot = placo_robot(MERGED_URDF.read_text(), tmp_path / "robot.urdf")
    rng = np.random.default_rng(1)
    for _ in range(10):
        q, angles = random_q(model, rng)
        for frame in FRAMES:
            np.testing.assert_allclose(
                placo_pose(robot, frame, angles), model.frame_pose(frame, q, {}), atol=TOLERANCE
            )


@pytest.mark.parametrize("theta", CORRECTIONS, ids=lambda t: "+".join(t))
def test_corrections_equal_placo_on_the_exported_urdf(model, tmp_path, theta):
    robot = placo_robot(model.to_urdf(theta), tmp_path / "corrected.urdf")
    rng = np.random.default_rng(2)
    for _ in range(4):
        q, angles = random_q(model, rng)
        for frame in FRAMES:
            np.testing.assert_allclose(
                placo_pose(robot, frame, angles), model.frame_pose(frame, q, theta), atol=TOLERANCE
            )


def test_all_corrections_at_once_and_each_one_moves_something(model, tmp_path):
    every = {k: v for t in CORRECTIONS for k, v in t.items()}
    robot = placo_robot(model.to_urdf(every), tmp_path / "all.urdf")
    q, angles = random_q(model, np.random.default_rng(3))
    for frame in FRAMES:
        np.testing.assert_allclose(
            placo_pose(robot, frame, angles), model.frame_pose(frame, q, every), atol=TOLERANCE
        )
    # guard against a correction that silently does nothing: each moves some frame by more than 0.1 mm
    for theta in CORRECTIONS:
        moved = max(np.abs(model.frame_pose(f, q, theta) - model.frame_pose(f, q, {})).max() for f in FRAMES)
        assert moved > 1e-4, theta


def test_lerobots_robot_kinematics_agrees_for_each_arm(model, tmp_path):
    from lerobot.model.kinematics import RobotKinematics

    path = tmp_path / "robot.urdf"
    path.write_text(MERGED_URDF.read_text())
    rng = np.random.default_rng(4)
    for arm in ("left", "right"):
        joints = [f"{arm}_{j}" for j in SO101_JOINTS]
        kinematics = RobotKinematics(str(path), pinch_frame(arm), joints)
        for _ in range(3):
            degrees = {j: float(rng.uniform(-60, 60)) for j in SO101_JOINTS}
            q = model.joint_vector({arm: {f"{j}.pos": v for j, v in degrees.items()}})
            np.testing.assert_allclose(
                kinematics.forward_kinematics(np.array([degrees[j] for j in SO101_JOINTS])),
                model.frame_pose(pinch_frame(arm), q, {}),
                atol=TOLERANCE,
            )
