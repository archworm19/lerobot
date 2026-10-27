from pathlib import Path

import numpy as np
import pytest

from lerobot.felix_jr.calibration.urdf_kinematics import (
    PinholeCamera,
    UrdfTree,
    invert,
    pose,
    rotation_about_axis,
    rotation_from_rpy,
)

XLEROBOT_URDF = Path(__file__).resolve().parents[2] / "src/lerobot/felix_jr/xlerobot.urdf"

# A planar two-link arm on a 1 m post: base -> (j1, about z) -> l1 -> (j2 at x=1, about z) -> l2 -> tip at x=1.
PLANAR = """
<robot name="planar">
  <link name="world"/><link name="base"/><link name="l1"/><link name="l2"/><link name="tip"/><link name="slide"/>
  <joint name="mount" type="floating"><parent link="world"/><child link="base"/>
    <origin xyz="0 0 1" rpy="0 0 0"/></joint>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-3" upper="3"/></joint>
  <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/>
    <origin xyz="1 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-3" upper="3"/></joint>
  <joint name="tool" type="fixed"><parent link="l2"/><child link="tip"/><origin xyz="1 0 0" rpy="0 0 0"/></joint>
  <joint name="rail" type="prismatic"><parent link="base"/><child link="slide"/>
    <origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="0" upper="1"/></joint>
</robot>
"""


@pytest.fixture
def planar() -> UrdfTree:
    return UrdfTree(xml=PLANAR)


class TestRotations:
    def test_rpy_is_yaw_pitch_roll_order(self):
        r = rotation_from_rpy(0.3, -0.2, 1.1)
        expected = (
            rotation_about_axis([0, 0, 1], 1.1)
            @ rotation_about_axis([0, 1, 0], -0.2)
            @ rotation_about_axis([1, 0, 0], 0.3)
        )
        np.testing.assert_allclose(r, expected, atol=1e-12)

    def test_axis_rotation_turns_x_into_y(self):
        np.testing.assert_allclose(
            rotation_about_axis([0, 0, 2.0], np.pi / 2) @ [1, 0, 0], [0, 1, 0], atol=1e-12
        )

    def test_invert(self):
        x = pose(rotation_from_rpy(0.1, 0.2, 0.3), np.array([1.0, -2.0, 0.5]))
        np.testing.assert_allclose(x @ invert(x), np.eye(4), atol=1e-12)


class TestTree:
    def test_root_and_chain(self, planar):
        assert planar.root == "world"
        assert [j.name for j in planar.chain("tip")] == ["mount", "j1", "j2", "tool"]
        assert planar.moving_joints("tip") == ["j1", "j2"]
        with pytest.raises(KeyError):
            planar.chain("nope")

    def test_forward_kinematics_of_the_planar_arm(self, planar):
        np.testing.assert_allclose(planar.point("tip", np.zeros(3)), [2, 0, 1], atol=1e-12)
        np.testing.assert_allclose(planar.point("tip", np.zeros(3), {"j1": np.pi / 2}), [0, 2, 1], atol=1e-12)
        np.testing.assert_allclose(planar.point("tip", np.zeros(3), {"j2": np.pi / 2}), [1, 1, 1], atol=1e-12)

    def test_prismatic_joint_slides_along_its_axis(self, planar):
        np.testing.assert_allclose(planar.point("slide", np.zeros(3), {"rail": 0.4}), [0, 0.4, 1], atol=1e-12)

    def test_joint_limits_are_read(self, planar):
        assert (planar.joints["j1"].lower, planar.joints["j1"].upper) == (-3.0, 3.0)

    def test_zero_offset_acts_like_extra_joint_angle(self, planar):
        offset = planar.point("tip", np.zeros(3), {"j1": 0.2}, {"zero:j1": 0.3})
        np.testing.assert_allclose(offset, planar.point("tip", np.zeros(3), {"j1": 0.5}), atol=1e-12)

    def test_position_offset_is_in_the_parent_frame(self, planar):
        # j1 at 90 deg turns l1, so an x offset of j2's origin (in l1's frame) moves the tip along world y
        moved = planar.point("tip", np.zeros(3), {"j1": np.pi / 2}, {"pos:j2:x": 0.1})
        np.testing.assert_allclose(moved, [0, 2.1, 1], atol=1e-12)

    def test_rotation_offset_turns_everything_downstream(self, planar):
        # a 90 deg yaw of the fixed tool joint turns the tip frame: a point 1 m along its x now lies along +y
        p = planar.point("tip", np.array([1.0, 0, 0]), None, {"rot:tool:yaw": np.pi / 2})
        np.testing.assert_allclose(p, [2, 1, 1], atol=1e-12)


class TestPinholeCamera:
    camera = PinholeCamera("base", fx=500.0, fy=400.0, ppx=320.0, ppy=240.0, width=640, height=480)

    def x_root_cam(self, planar):
        return self.camera.pose_in_root(planar)  # camera on `base`, looking along +x from (0, 0, 1)

    def test_point_straight_ahead_is_the_principal_point(self, planar):
        uv = self.camera.project(self.x_root_cam(planar), np.array([[2.0, 0, 1]]))
        np.testing.assert_allclose(uv, [[320, 240]], atol=1e-9)

    def test_right_is_minus_y_and_down_is_minus_z(self, planar):
        x = self.x_root_cam(planar)
        right = self.camera.project(x, np.array([[2.0, -0.2, 1]]))[0]
        below = self.camera.project(x, np.array([[2.0, 0, 0.8]]))[0]
        np.testing.assert_allclose(right, [320 + 500 * 0.2 / 2, 240], atol=1e-9)
        np.testing.assert_allclose(below, [320, 240 + 400 * 0.2 / 2], atol=1e-9)

    def test_points_behind_are_nan_and_out_of_view(self, planar):
        uv = self.camera.project(self.x_root_cam(planar), np.array([[-1.0, 0, 1], [2.0, 5.0, 1]]))
        assert np.isnan(uv[0]).all()
        assert self.camera.in_view(uv).tolist() == [False, False]


class TestXlerobotUrdf:
    def test_parses_and_has_the_head_and_arm_joints(self):
        tree = UrdfTree(XLEROBOT_URDF)
        for name in ("head_pan_joint", "head_tilt_joint", "Rotation_L", "Jaw_R", "fixed_head_camera_link"):
            assert name in tree.joints

    def test_head_conventions_positive_tilt_looks_down_positive_pan_turns_left(self):
        tree = UrdfTree(XLEROBOT_URDF)
        cam = PinholeCamera("head_camera_link", 611.0, 610.0, 316.0, 247.0, 640, 480)

        def view(q):
            return cam.pose_in_root(tree, q)[:3, 2]  # optical z axis in the root frame

        rest = view({})
        np.testing.assert_allclose(rest, [-1, 0, 0], atol=1e-6)  # forward is -x in this URDF
        assert view({"head_tilt_joint": 0.3})[2] < 0  # looks down
        left = np.cross([0, 0, 1], rest)  # robot's left when facing `rest`
        assert view({"head_pan_joint": 0.3}) @ left > 0  # turns left
