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


def test_the_task_steps_from_pool_starts(tmp_path):
    """The task's terms (`mjlab_task`) on pool starts: the observations' shape, a delta action, the rewards finite."""
    import torch

    model = XleRobotModel.load(api.MODEL)
    path = tmp_path / "pool.npz"
    _pool(model).save(path)
    env = me.make_env(
        me.env_cfg(
            model, num_envs=4, randomize=False, start_pool=path, pool_weights=dict.fromkeys(POOLS, 1.0)
        )
    )
    obs, _ = env.reset()
    assert (
        obs["actor"].shape == (4, 42) and env.action_manager.total_action_dim == 6
    )  # (8+8 joints, 6+6 pose, 6, 2, 6)
    term = env.action_manager.get_term("joint_targets")
    before = term.targets.clone()
    action = torch.zeros(4, 6)
    action[:, 0] = 1.0  # the pan a full step
    _, reward, _, _, _ = env.step(action)
    assert torch.isfinite(reward).all()
    assert torch.allclose(
        term.targets[:, 0] - before[:, 0], torch.full((4,), float(np.radians(1.0))), atol=1e-5
    )


SIZES = [(0.020, 0.10, 0.05), (0.0225, 0.13, 0.08), (0.026, 0.16, 0.10), (0.030, 0.19, 0.12)]


def _sized_pool(model: XleRobotModel) -> StartPool:
    """Four starts, each its own object (`SIZES`), standing on the table; the arm at rest, out of the way."""
    rows = []
    for k, size in enumerate(SIZES):
        s = sc.build(model, "right", [sc.Cylinder("object", -0.35, 0.10 + 0.01 * k, *size)], table_z=TABLE_Z)
        s.set_pose({**REST, "gripper.pos": 20.0})
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
        stack["qpos"], stack["qvel"], stack["ctrl"], np.array(POOLS), np.arange(4), np.array(SIZES), TABLE_Z
    )


def test_a_sized_pool_resizes_each_worlds_object(tmp_path):
    """Objects of several sizes: each reset world's cylinder, mass and inertia its start's, standing still on the table
    (a wrong size or stale collision bounds would sink it or pop it up), its size observed."""
    import torch

    from lerobot.felix_jr.sim2real.start_pool import _object_ids

    model = XleRobotModel.load(api.MODEL)
    path = tmp_path / "sized.npz"
    pool = _sized_pool(model)
    pool.save(path)
    back = StartPool.load(path)
    assert back.sizes_vary and np.allclose(back.objects, SIZES)
    with pytest.raises(ValueError):
        _ = back.object_size
    env = me.make_env(
        me.env_cfg(
            model, num_envs=8, randomize=False, start_pool=path, pool_weights=dict.fromkeys(POOLS, 1.0)
        )
    )
    obs, _ = env.reset()
    assert obs["actor"].shape == (8, 44)  # (the one object's 42, and its radius and height)
    body, geom = _object_ids(env.sim.mj_model)
    qpos, _, _ = back.layout(env.sim.mj_model)
    z = me._joint_slices(env.sim.mj_model)["floating_base_joint"].start + 2
    now = env.sim.data.qpos.cpu().numpy()
    rows = [int(np.abs(qpos - now[w]).max(axis=1).argmin()) for w in range(env.num_envs)]
    assert len(set(rows)) > 1, "every world drew the same start: nothing to tell apart"
    m = env.sim.model
    for w, row in enumerate(rows):
        r, h, mass = SIZES[row]
        assert m.geom_size[w, geom, :2].cpu().numpy() == pytest.approx([r, h / 2], abs=1e-6)
        assert float(m.geom_rbound[w, geom]) == pytest.approx(np.hypot(r, h / 2), abs=1e-6)
        assert float(m.body_mass[w, body]) == pytest.approx(mass, abs=1e-6)
        assert float(m.body_inertia[w, body, 2]) == pytest.approx(mass * r * r / 2, rel=1e-5)
        assert float(env.felix_object_rest_z[w]) == pytest.approx(TABLE_Z + h / 2, abs=1e-6)
        assert obs["actor"][w, -2:].cpu().numpy() == pytest.approx([r, h], abs=1e-6)
    for _ in range(10):
        env.step(torch.zeros(8, 6))
    after = env.sim.data.qpos[:, z].cpu().numpy()
    for w, row in enumerate(rows):
        assert after[w] == pytest.approx(TABLE_Z + SIZES[row][1] / 2, abs=2e-3), (
            f"world {w}: the object moved"
        )
