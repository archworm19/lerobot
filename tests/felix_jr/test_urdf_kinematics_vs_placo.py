"""Cross-check of our numpy forward kinematics against placo (what LeRobot's RobotKinematics uses).

placo needs a URDF it can load, so these tests build a copy of `xlerobot.urdf` with the meshes stripped
(the URDF references meshes we do not have) and the root `floating` joint made `fixed`: placo adds its own free
base, and a second floating joint in the URDF scrambles its joint indexing (`set_joint` then moves nothing).
Our numpy code treats the floating joint as fixed too, so the two describe the same tree.

Parameter errors are applied by patching the URDF text (joint origin xyz / rpy) and loading the patched copy
in placo, which is an independent route to the same perturbed geometry.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from lerobot.felix_jr.calibration.sensitivity import ARMS, make_configurations, parameters, project_targets
from lerobot.felix_jr.calibration.urdf_kinematics import (
    PinholeCamera,
    UrdfTree,
    invert,
    rotation_about_axis,
    rotation_from_rpy,
    rpy_from_rotation,
)

placo = pytest.importorskip("placo")

XLEROBOT_URDF = Path(__file__).resolve().parents[2] / "src/lerobot/felix_jr/xlerobot.urdf"
FRAMES = ["Wrist_Pitch_Roll", "Moving_Jaw", "Moving_Jaw_2", "head_tilt_link", "head_camera_link"]
TOLERANCE = 1e-9


def patched_urdf(tmp_path: Path, patch=None, name: str = "robot.urdf") -> Path:
    """A placo-loadable copy of xlerobot.urdf; `patch(joint_elements_by_name)` may edit it in place."""
    root = ET.parse(XLEROBOT_URDF).getroot()
    for link in root.findall("link"):
        for tag in ("visual", "collision"):
            for element in link.findall(tag):
                link.remove(element)
    joints = {j.get("name"): j for j in root.findall("joint")}
    for joint in joints.values():
        if joint.get("type") == "floating":
            joint.set("type", "fixed")
    if patch:
        patch(joints)
    path = tmp_path / name
    ET.ElementTree(root).write(path)
    return path


def set_origin(joint: ET.Element, xyz=None, rpy=None) -> None:
    origin = joint.find("origin")
    if xyz is not None:
        origin.set("xyz", " ".join(repr(float(v)) for v in xyz))
    if rpy is not None:
        origin.set("rpy", " ".join(repr(float(v)) for v in rpy))


def origin_values(joint: ET.Element) -> tuple[np.ndarray, np.ndarray]:
    origin = joint.find("origin")
    return (
        np.array([float(v) for v in origin.get("xyz").split()]),
        np.array([float(v) for v in origin.get("rpy").split()]),
    )


def apply_perturbation(joints: dict[str, ET.Element], perturb: dict[str, float], tree: UrdfTree) -> None:
    """Edit joint origins in the URDF text so that placo sees the same perturbed geometry as `perturb`.

    Rotation errors on one joint are combined into a single roll-pitch-yaw rotation, as `UrdfTree` applies
    them (applying them one after another would give a different result, since rotations do not commute).
    """
    extra_rpy: dict[str, np.ndarray] = {}
    for key, value in perturb.items():
        kind, name, *rest = key.split(":")
        xyz, rpy = origin_values(joints[name])
        if kind == "pos":
            set_origin(joints[name], xyz=xyz + value * np.eye(3)[{"x": 0, "y": 1, "z": 2}[rest[0]]])
        elif kind == "rot":
            extra_rpy.setdefault(name, np.zeros(3))[{"roll": 0, "pitch": 1, "yaw": 2}[rest[0]]] = value
        elif kind == "zero":  # a joint zero offset is a fixed extra rotation about the joint's own axis
            rotation = rotation_from_rpy(*rpy) @ rotation_about_axis(tree.joints[name].axis, value)
            set_origin(joints[name], rpy=rpy_from_rotation(rotation))
        else:
            raise ValueError(key)
    for name, extra in extra_rpy.items():
        _, rpy = origin_values(joints[name])
        set_origin(joints[name], rpy=rpy_from_rotation(rotation_from_rpy(*rpy) @ rotation_from_rpy(*extra)))


def random_positions(tree: UrdfTree, rng: np.random.Generator) -> dict[str, float]:
    return {
        name: rng.uniform(j.lower if j.lower is not None else -3.0, j.upper if j.upper is not None else 3.0)
        for name, j in tree.joints.items()
        if j.moves
    }


def placo_pose(robot, frame: str, q: dict[str, float]) -> np.ndarray:
    for name, value in q.items():
        robot.set_joint(name, value)
    robot.update_kinematics()
    return np.array(robot.get_T_world_frame(frame))


@pytest.fixture(scope="module")
def tree() -> UrdfTree:
    return UrdfTree(XLEROBOT_URDF)


@pytest.fixture(scope="module")
def stock_robot(tmp_path_factory):
    return placo.RobotWrapper(str(patched_urdf(tmp_path_factory.mktemp("stock"))))


class TestRpyConversion:
    def test_round_trip_through_a_rotation_matrix(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            rpy = (rng.uniform(-np.pi, np.pi), rng.uniform(-1.5, 1.5), rng.uniform(-np.pi, np.pi))
            np.testing.assert_allclose(rpy_from_rotation(rotation_from_rpy(*rpy)), rpy, atol=1e-9)

    def test_gimbal_lock_still_reproduces_the_rotation(self):
        for pitch in (np.pi / 2, -np.pi / 2):
            rotation = rotation_from_rpy(0.3, pitch, 0.5)
            np.testing.assert_allclose(rotation_from_rpy(*rpy_from_rotation(rotation)), rotation, atol=1e-9)


class TestAgainstPlaco:
    def test_every_moving_joint_at_random_positions(self, tree, stock_robot):
        rng = np.random.default_rng(1)
        for _ in range(10):
            q = random_positions(tree, rng)
            for frame in FRAMES:
                np.testing.assert_allclose(
                    placo_pose(stock_robot, frame, q), tree.link_pose(frame, q), atol=TOLERANCE
                )

    def test_nominal_pose_matches_before_any_joint_is_moved(self, tree, stock_robot):
        zero = dict.fromkeys((n for n, j in tree.joints.items() if j.moves), 0.0)
        for frame in FRAMES:
            np.testing.assert_allclose(
                placo_pose(stock_robot, frame, zero), tree.link_pose(frame, zero), atol=TOLERANCE
            )

    def test_lerobots_robot_kinematics_agrees_for_the_left_arm(self, tree, tmp_path):
        from lerobot.model.kinematics import RobotKinematics

        joints = ["Rotation_L", "Pitch_L", "Elbow_L", "Wrist_Pitch_L", "Wrist_Roll_L"]
        kinematics = RobotKinematics(str(patched_urdf(tmp_path)), "Moving_Jaw", joints)
        rng = np.random.default_rng(2)
        for _ in range(5):
            q = {j: rng.uniform(tree.joints[j].lower, tree.joints[j].upper) for j in joints}
            degrees = np.degrees([q[j] for j in joints])
            np.testing.assert_allclose(
                kinematics.forward_kinematics(degrees), tree.link_pose("Moving_Jaw", q), atol=TOLERANCE
            )


# one representative perturbation of every kind and chain: the head tower, head joints, camera mount, arm mount,
# arm joints (zero offsets, origin positions) and the tool joint
PERTURBATIONS = [
    {"pos:fixed_top_base_link:y": 0.004},
    {"rot:fixed_top_base_link:yaw": np.radians(1.5)},
    {"zero:head_pan_joint": np.radians(2.0)},
    {"zero:head_tilt_joint": np.radians(-2.0)},
    {"pos:head_tilt_joint:z": -0.003},
    {"pos:fixed_head_camera_link:x": 0.005, "pos:fixed_head_camera_link:z": -0.004},
    {"rot:fixed_head_camera_link:roll": np.radians(2.0)},
    {"rot:fixed_head_camera_link:pitch": np.radians(-2.0), "rot:fixed_head_camera_link:yaw": np.radians(1.0)},
    {"pos:fixed_Base:x": 0.003},
    {"rot:fixed_Base:pitch": np.radians(1.0)},
    {"zero:Rotation_L": np.radians(2.0)},
    {"zero:Elbow_L": np.radians(-3.0), "zero:Wrist_Pitch_L": np.radians(2.0)},
    {"pos:Elbow_L:y": 0.002, "pos:Wrist_Pitch_L:z": -0.002},
    {"pos:Jaw_L:x": 0.003},
]


class TestPerturbationsAgainstPatchedUrdfs:
    @pytest.mark.parametrize("perturb", PERTURBATIONS, ids=lambda p: "+".join(p))
    def test_numpy_perturbation_equals_patching_the_urdf(self, tree, tmp_path, perturb):
        robot = placo.RobotWrapper(
            str(patched_urdf(tmp_path, lambda j: apply_perturbation(j, perturb, tree)))
        )
        rng = np.random.default_rng(3)
        for _ in range(4):
            q = random_positions(tree, rng)
            for frame in FRAMES:
                np.testing.assert_allclose(
                    placo_pose(robot, frame, q), tree.link_pose(frame, q, perturb), atol=TOLERANCE
                )

    def test_the_perturbations_actually_change_the_pose(self, tree):
        # guard against a patch that silently does nothing: every case moves at least one frame by > 0.1 mm
        q = random_positions(tree, np.random.default_rng(4))
        for perturb in PERTURBATIONS:
            moved = max(np.abs(tree.link_pose(f, q, perturb) - tree.link_pose(f, q)).max() for f in FRAMES)
            assert moved > 1e-4, perturb

    def test_projected_pixels_with_all_perturbations_at_once(self, tree, tmp_path):
        """End to end: pixels from placo's perturbed poses equal the pixels our sensitivity code computes."""
        camera = PinholeCamera("head_camera_link", 611.0, 610.0, 316.0, 247.0, 640, 480)
        perturb = {k: v for p in PERTURBATIONS for k, v in p.items()}
        robot = placo.RobotWrapper(
            str(patched_urdf(tmp_path, lambda j: apply_perturbation(j, perturb, tree)))
        )
        for config in make_configurations(tree, camera, "left", per_head_pose=1, seed=5)[:6]:
            expected = project_targets(tree, camera, "left", config, perturb)
            x_root_cam = placo_pose(robot, "head_camera_link", config) @ PinholeCamera.LINK_TO_OPTICAL
            points = np.array(
                [placo_pose(robot, link, config)[:3, 3] for link in ARMS["left"]["targets"].values()]
            )
            np.testing.assert_allclose(camera.project(x_root_cam, points), expected, atol=1e-6)

    def test_every_sensitivity_parameter_name_is_one_the_patcher_understands(self, tree, tmp_path):
        for arm in ARMS:
            for param in parameters(tree, arm):
                robot_xml = patched_urdf(
                    tmp_path, lambda j, p=param: apply_perturbation(j, {p.name: 0.001}, tree)
                )
                assert robot_xml.exists()
        assert invert(np.eye(4)).shape == (4, 4)
