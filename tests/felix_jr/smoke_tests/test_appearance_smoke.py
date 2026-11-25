"""The randomized look in the mjlab environment (`sim2real/appearance.py`): each world its own table, floor, walls and
object colour after a reset; the physics untouched (the same contacts as without)."""

import numpy as np
import pytest

pytest.importorskip("mjlab")
pytest.importorskip("pinocchio")
torch = pytest.importorskip("torch")

from test_start_pool_smoke import _pool  # noqa: E402  (the same folder: pytest puts it on the path)

from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel  # noqa: E402
from lerobot.felix_jr.harness import api  # noqa: E402
from lerobot.felix_jr.sim2real import mjlab_env as me  # noqa: E402
from lerobot.felix_jr.sim2real.render_bench import SENSOR, camera_cfg  # noqa: E402


def test_each_world_looks_different_and_moves_the_same(tmp_path):
    """Eight worlds with the look randomized: the table's materials and the object's colours vary between worlds and
    resets, the frames differ; stepped alike, the robot moves as it does without the look."""
    model = XleRobotModel.load(api.MODEL)
    path = tmp_path / "pool.npz"
    _pool(model).save(path)
    joints = {}
    for look in (False, True):
        cfg = me.env_cfg(model, num_envs=8, start_pool=path, randomize=False, appearance=look)
        cfg.scene.sensors = (*cfg.scene.sensors, camera_cfg(224, 168))
        cfg.seed = 0  # (the same starts both times: the pool's reset draws before the look's)
        torch.manual_seed(0)
        env = me.make_env(cfg, "cpu")
        env.reset()
        if look:
            table = env.scene["table"].indexing.geom_ids
            matid = env.sim.model.geom_matid[:, table].cpu().numpy()
            assert len({tuple(r) for r in matid}) > 1  # (the worlds' tables differ)
            obj_ids = env.scene["object"].indexing.geom_ids
            rgba = env.sim.model.geom_rgba[:, obj_ids].cpu().numpy()
            assert rgba[:, 0, :3].std(axis=0).max() > 0.05  # (the objects' colours differ)
            frames = env.scene.sensors[SENSOR].data.rgb.cpu().numpy().astype(float)
            assert np.abs(frames[0] - frames[1]).mean() > 5.0
        for _ in range(3):
            env.step(0.3 * torch.ones(8, env.action_manager.total_action_dim))
        joints[look] = env.scene["robot"].data.joint_pos.cpu().numpy().copy()
    assert np.allclose(joints[False], joints[True], atol=1e-6)
