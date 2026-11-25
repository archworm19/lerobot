"""The closed-loop test's plumbing (`sim2real/probe_loop.py`): the object inputs it writes land in the actor's
observation where the environment puts them, and an estimate in the pinch frame survives the trip through the root."""

import pytest

pytest.importorskip("mjlab")
pytest.importorskip("pinocchio")
torch = pytest.importorskip("torch")

from test_start_pool_smoke import _pool  # noqa: E402  (the same folder: pytest puts it on the path)

from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel  # noqa: E402
from lerobot.felix_jr.harness import api  # noqa: E402
from lerobot.felix_jr.sim2real import mjlab_env as me, mjlab_object as obj  # noqa: E402
from lerobot.felix_jr.sim2real.probe_loop import Inputs, term_slices  # noqa: E402
from lerobot.felix_jr.sim2real.render_bench import camera_cfg  # noqa: E402


class TruthProbe:
    """A probe that always tells the truth (the object's pose in the pinch frame and its size), in view."""

    def __init__(self, env) -> None:
        self.env, self.cyl = env, obj.of(env)

    def __call__(self, rgb):
        p = self.cyl.in_pinch(self.env)
        n = len(p)
        return {
            "center": p[:, :3],
            "axis": p[:, 3:],
            "size": self.cyl.size(self.env),
            "in_view": torch.ones(n, dtype=bool),
        }


def _env(tmp_path):
    model = XleRobotModel.load(api.MODEL)
    path = tmp_path / "pool.npz"
    _pool(model).save(path)
    cfg = me.env_cfg(model, num_envs=2, start_pool=path, randomize=False, object_noise_m=0.0)
    cfg.scene.sensors = (*cfg.scene.sensors, camera_cfg(224, 168))
    env = me.make_env(cfg, "cpu")
    env.reset()
    return env


def test_the_inputs_land_where_the_environment_puts_them(tmp_path):
    """Without noise, `truth_path` writes exactly the actor's own object terms; after steps too."""
    env = _env(tmp_path)
    cols = term_slices(env)
    inputs = Inputs("truth_path", env, None, noise_m=0.0)
    for k in range(3):
        if k:
            env.step(0.3 * torch.ones(2, env.action_manager.total_action_dim))
        actor = env.observation_manager.compute()["actor"]
        new = inputs(torch.zeros(2, dtype=torch.bool))
        for name in ("object_in_root", "object_in_pinch"):
            assert torch.allclose(actor[:, cols[name]], new[name], atol=1e-6), (k, name)


def test_a_true_estimate_comes_back_true_through_the_root_frame(tmp_path):
    """The probe's conditions keep the estimate in the root frame and give the actor its pinch-frame pose back from the
    gripper's live pose: a probe telling the truth gives the truth, live and frozen before the arm moves."""
    env = _env(tmp_path)
    cyl = obj.of(env)
    for condition in ("probe", "probe_smoothed", "probe_frozen"):
        inputs = Inputs(condition, env, TruthProbe(env), noise_m=0.0)
        fresh = torch.ones(2, dtype=torch.bool)
        for _ in range(2):
            new = inputs(fresh)
            assert torch.allclose(new["object_in_pinch"], cyl.in_pinch(env), atol=1e-5), condition
            assert torch.allclose(new["object_in_root"], cyl.in_root(env), atol=1e-5), condition
            fresh = torch.zeros(2, dtype=torch.bool)
