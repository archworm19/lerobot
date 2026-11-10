"""`felix_jr.sim`: the SO-101 scene in MuJoCo (skipped without mujoco or without the model cached: tests never
download it; `sim.assets.so101_model()` does, once)."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from lerobot.felix_jr.sim.assets import MESHES, cache_dir, is_cached  # noqa: E402
from lerobot.felix_jr.sim.scene import Box, Cylinder, build, scene_xml  # noqa: E402

pytestmark = pytest.mark.skipif(not is_cached(), reason="the SO-101 model is not cached")

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def test_the_cache_holds_the_model_and_its_meshes():
    assert (cache_dir() / "assets").is_dir() and len(MESHES) == 15


def test_the_arm_has_lerobots_joints_as_position_servos():
    m = build()
    assert tuple(m.actuator(i).name for i in range(m.nu)) == JOINTS
    assert all(m.actuator(i).trntype == mujoco.mjtTrn.mjTRN_JOINT for i in range(m.nu))
    assert "table" in scene_xml()


def test_objects_rest_on_the_table_and_the_arm_holds_its_pose():
    m = build((Cylinder("bottle", 0.25, 0.0, 0.018, 0.1), Box("cube", 0.2, 0.1, (0.02, 0.02, 0.02))))
    d = mujoco.MjData(m)
    target = np.array([0.0, -0.5, 0.5, 0.8, 0.0, 0.5])
    d.ctrl[:] = target
    for _ in range(1500):  # 3 s
        mujoco.mj_step(m, d)
    assert np.isfinite(d.qpos).all()
    assert d.xpos[m.body("bottle").id][2] == pytest.approx(0.05, abs=0.003)  # standing, its middle 5 cm up
    assert d.xpos[m.body("cube").id][2] == pytest.approx(0.02, abs=0.003)
    joints = np.array([d.qpos[m.joint(j).qposadr[0]] for j in JOINTS])
    assert np.abs(joints - target).max() < np.radians(
        5
    )  # the servos got there (gravity, backlash: a few deg)
