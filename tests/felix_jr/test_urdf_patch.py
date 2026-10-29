"""Patched URDFs give the same poses as perturbing the original in numpy (no placo needed)."""

from pathlib import Path

import numpy as np
import pytest

from lerobot.felix_jr.calibration.urdf_kinematics import UrdfTree
from lerobot.felix_jr.calibration.urdf_patch import patch_urdf

FELIX = Path(__file__).resolve().parents[2] / "src" / "lerobot" / "felix_jr"
URDFS = {"xlerobot": FELIX / "xlerobot.urdf", "so101": FELIX / "so101_new_calib.urdf"}
LINKS = {"xlerobot": ["head_camera_link", "Moving_Jaw", "Moving_Jaw_2"], "so101": ["gripper_frame_link"]}


def random_perturbation(tree, link, rng, kinds=("zero", "pos", "rot")):
    perturb = {}
    for joint in tree.chain(link):
        if "zero" in kinds and joint.moves:
            perturb[f"zero:{joint.name}"] = rng.normal(0, 0.05)
        if "pos" in kinds:
            for a in "xyz":
                perturb[f"pos:{joint.name}:{a}"] = rng.normal(0, 0.004)
        if "rot" in kinds:
            for a in ("roll", "pitch", "yaw"):
                perturb[f"rot:{joint.name}:{a}"] = rng.normal(0, 0.04)
    return perturb


@pytest.mark.parametrize("name", ["xlerobot", "so101"])
def test_patched_urdf_gives_the_same_poses_as_the_perturbed_original(name):
    original = UrdfTree(URDFS[name])
    rng = np.random.default_rng(0)
    for link in LINKS[name]:
        perturb = random_perturbation(original, link, rng)
        patched = UrdfTree(xml=patch_urdf(URDFS[name].read_text(), perturb))
        for _ in range(5):
            q = {j: rng.uniform(-0.8, 0.8) for j in original.moving_joints(link)}
            np.testing.assert_allclose(
                patched.link_pose(link, q), original.link_pose(link, q, perturb), atol=1e-9
            )


def test_each_kind_of_parameter_on_its_own():
    xml = URDFS["so101"].read_text()
    original = UrdfTree(xml=xml)
    q = {"shoulder_pan": 0.3, "shoulder_lift": -0.5, "elbow_flex": 0.7, "wrist_flex": 0.2, "wrist_roll": -0.4}
    for perturb in (
        {"zero:elbow_flex": 0.1},
        {"pos:wrist_flex:y": 0.003},
        {"rot:shoulder_lift:pitch": 0.02, "rot:shoulder_lift:yaw": -0.03},
        {"zero:shoulder_pan": -0.2, "pos:gripper_frame_joint:z": 0.005},
    ):
        patched = UrdfTree(xml=patch_urdf(xml, perturb))
        np.testing.assert_allclose(
            patched.link_pose("gripper_frame_link", q),
            original.link_pose("gripper_frame_link", q, perturb),
            atol=1e-9,
        )
    assert not np.allclose(
        original.link_pose("gripper_frame_link", q),
        original.link_pose("gripper_frame_link", q, {"zero:elbow_flex": 0.1}),
    )  # the perturbations above really change something


def test_a_prismatic_joint_and_the_limits_and_everything_else_are_kept():
    xml = """<robot name="r"><link name="a"/><link name="b"/>
      <joint name="slide" type="prismatic"><parent link="a"/><child link="b"/>
        <origin xyz="0 0 0.1" rpy="0 1.0 0"/><axis xyz="1 0 0"/><limit lower="-0.1" upper="0.2"/></joint></robot>"""
    perturb = {"zero:slide": 0.03}
    patched_xml = patch_urdf(xml, perturb)
    np.testing.assert_allclose(
        UrdfTree(xml=patched_xml).link_pose("b", {"slide": 0.05}),
        UrdfTree(xml=xml).link_pose("b", {"slide": 0.05}, perturb),
        atol=1e-12,
    )
    joint = UrdfTree(xml=patched_xml).joints["slide"]
    assert (joint.lower, joint.upper) == (-0.1, 0.2)  # limits are on the measured position: unchanged
    assert patch_urdf(xml, {}).count("<link") == 2


def test_unknown_joints_are_refused():
    with pytest.raises(KeyError, match="no joint 'nose'"):
        patch_urdf(URDFS["so101"].read_text(), {"zero:nose": 0.1})
