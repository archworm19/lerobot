"""The mjlab environment builds, and its compiled model is the CPU simulator's (`sim2real/mjlab_env.py`)."""

import pytest

pytest.importorskip("mjlab")
pytest.importorskip("pinocchio")

from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel  # noqa: E402
from lerobot.felix_jr.harness import api  # noqa: E402
from lerobot.felix_jr.sim2real import mjlab_env as me  # noqa: E402


def test_the_environments_model_is_the_cpu_simulators():
    model = XleRobotModel.load(api.MODEL)
    env = me.make_env(me.env_cfg(model, randomize=False, give_scale=1.0))  # (the CPU model as is)
    cpu = me.cpu_model(model)
    assert me.model_differences(cpu, env.sim.mj_model) == []
    import mujoco

    d = mujoco.MjData(cpu)
    mujoco.mj_forward(cpu, d)
    assert me.pose_differences(cpu, env.sim.mj_model, d.qpos) == []


def test_mjlabs_give_correction_is_the_one_difference():
    model = XleRobotModel.load(api.MODEL)
    env = me.make_env(me.env_cfg(model, randomize=False))
    diffs = me.model_differences(me.cpu_model(model), env.sim.mj_model)
    assert diffs and all("_give.jnt_stiffness" in d for d in diffs)
