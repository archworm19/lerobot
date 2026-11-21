"""The start pool: saved by name, read back, laid out as the mjlab environment's model, written into its worlds at a
reset (`sim2real/start_pool.py`)."""

import numpy as np
import pytest

pytest.importorskip("mjlab")
pytest.importorskip("pinocchio")

from lerobot.felix_jr.calibration.joint_sim import REST  # noqa: E402
from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel  # noqa: E402
from lerobot.felix_jr.harness import api  # noqa: E402
from lerobot.felix_jr.sim2real import (
    mjlab_env as me,  # noqa: E402
    scene as sc,  # noqa: E402
)
from lerobot.felix_jr.sim2real.start_pool import POOLS, StartPool, _layout  # noqa: E402

TABLE_Z = 0.7779
PYRI = (0.0225, 0.13, 0.08)


def _pool(model: XleRobotModel) -> StartPool:
    """Four starts, one a pool: the arm at rest moved a joint further each time, the object 1 cm further along y."""
    rows = []
    for k in range(len(POOLS)):
        s = sc.build(model, "right", [sc.Cylinder("object", -0.35, 0.10 + 0.01 * k, *PYRI)], table_z=TABLE_Z)
        s.set_pose({**REST, "shoulder_pan.pos": REST["shoulder_pan.pos"] + 5 * k, "gripper.pos": 20.0 + k})
        lay = _layout(s.m)
        rows.append(
            {
                "qpos": {n: s.d.qpos[q : q + w].copy() for n, q, w, _, _ in lay["joints"]},
                "qvel": {n: s.d.qvel[v : v + w].copy() for n, _, _, v, w in lay["joints"]},
                "ctrl": {n: s.d.ctrl[a : a + 1].copy() for a, n in enumerate(lay["actuators"])},
            }
        )
    stack = {
        kind: {n: np.stack([r[kind][n] for r in rows]) for n in rows[0][kind]}
        for kind in ("qpos", "qvel", "ctrl")
    }
    return StartPool(
        stack["qpos"], stack["qvel"], stack["ctrl"], np.array(POOLS), np.arange(4), PYRI, TABLE_Z
    )


def test_a_reset_writes_a_start_of_the_pool_into_each_world(tmp_path):
    model = XleRobotModel.load(api.MODEL)
    pool = _pool(model)
    path = tmp_path / "pool.npz"
    pool.save(path)
    back = StartPool.load(path)
    assert back.object_size == PYRI and len(back) == 4
    for kind in ("qpos", "qvel", "ctrl"):
        assert all(np.allclose(getattr(back, kind)[n], getattr(pool, kind)[n]) for n in getattr(pool, kind))

    env = me.make_env(
        me.env_cfg(
            model, num_envs=8, randomize=False, start_pool=path, pool_weights=dict.fromkeys(POOLS, 1.0)
        )
    )
    env.reset()
    qpos, _, ctrl = back.layout(env.sim.mj_model)
    now = env.sim.data.qpos.cpu().numpy()
    for w in range(env.num_envs):
        gaps = np.abs(qpos - now[w]).max(axis=1)
        assert gaps.min() < 1e-5, f"world {w} holds no start of the pool"
        row = int(gaps.argmin())
        assert np.allclose(env.sim.data.ctrl[w].cpu().numpy(), ctrl[row], atol=1e-6)
        z = me._joint_slices(env.sim.mj_model)["floating_base_joint"].start + 2
        assert float(env.felix_object_z0[w]) == pytest.approx(qpos[row, z], abs=1e-6)
