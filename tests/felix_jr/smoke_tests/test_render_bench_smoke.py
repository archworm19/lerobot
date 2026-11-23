"""The wrist camera rendered in mjlab (`sim2real/render_bench.py`): frames of the asked size, the object in them, and
frames that follow the arm."""

import numpy as np
import pytest

pytest.importorskip("mjlab")
pytest.importorskip("pinocchio")
torch = pytest.importorskip("torch")

from test_start_pool_smoke import _pool  # noqa: E402  (the same folder: pytest puts it on the path)

from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel  # noqa: E402
from lerobot.felix_jr.harness import api  # noqa: E402
from lerobot.felix_jr.sim2real import mjlab_env as me  # noqa: E402
from lerobot.felix_jr.sim2real.render_bench import SENSOR, VitS14, camera_cfg  # noqa: E402


def test_the_wrist_camera_renders_the_object_and_follows_the_arm(tmp_path):
    """Two worlds: 224 x 168 RGB, the object's blue in each, the frames changed after the arm moves; the ViT's tokens."""
    model = XleRobotModel.load(api.MODEL)
    path = tmp_path / "pool.npz"
    _pool(model).save(path)
    cfg = me.env_cfg(model, num_envs=2, start_pool=path, randomize=False)
    cfg.scene.sensors = (*cfg.scene.sensors, camera_cfg(224, 168))
    env = me.make_env(cfg, "cpu")
    env.reset()
    sensor = env.scene.sensors[SENSOR]
    before = sensor.data.rgb.cpu().numpy().copy()  # (on the CPU device, a view of the buffer)
    assert before.shape == (2, 168, 224, 3) and before.dtype == np.uint8
    # the object (blue, `scene.Cylinder`'s colour) is in view
    blue = (before[..., 2] > 150) & (before[..., 0] < 100)
    assert blue.reshape(2, -1).mean(axis=1).min() > 0.005
    for _ in range(5):
        env.step(torch.ones(2, env.action_manager.total_action_dim))
    after = sensor.data.rgb.cpu().numpy()
    assert np.abs(after.astype(int) - before.astype(int)).mean() > 5.0
    tokens = VitS14()(sensor.data.rgb)
    assert tokens.shape == (2, 1 + 16 * 12, 384)
