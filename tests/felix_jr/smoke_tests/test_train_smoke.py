"""The training script end to end on the CPU: a few worlds, two PPO iterations, the checkpoints and `eval.json`
(`sim2real/train.py`)."""

import json

import pytest

pytest.importorskip("mjlab")
pytest.importorskip("pinocchio")

from test_start_pool_smoke import _pool  # noqa: E402  (the same folder: pytest puts it on the path)

from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel  # noqa: E402
from lerobot.felix_jr.harness import api  # noqa: E402
from lerobot.felix_jr.sim2real import train  # noqa: E402
from lerobot.felix_jr.sim2real.start_pool import POOLS  # noqa: E402


def test_training_runs_and_evaluates(tmp_path):
    pool = tmp_path / "pool_train.npz"
    _pool(XleRobotModel.load(api.MODEL)).save(pool)
    _pool(XleRobotModel.load(api.MODEL)).save(
        tmp_path / "pool_test.npz"
    )  # (found beside it: the held-out pool)
    out = tmp_path / "train"
    train.main(
        ["--pool", str(pool), "--device", "cpu", "--num-envs", "4", "--iterations", "2", "--steps-per-env", "4",
         "--save-interval", "1", "--log-dir", str(out), "--video-every", "0", "--eval-worlds", "4", "--pool-weights", *(f"{p}=1" for p in POOLS)]
    )  # fmt: skip
    assert (out / "model_final.pt").is_file() and (out / "settings.json").is_file()
    result = json.loads((out / "eval.json").read_text())
    assert result["train_pool"]["worlds"] == 4 and 0.0 <= result["train_pool"]["all"]["held_ever"] <= 1.0
    assert result["held_out"]["worlds"] == 4  # (the held-out pool, its own environment)
    assert json.loads((out / "settings.json").read_text())["eval_pool"].endswith("pool_test.npz")


def test_watcher_evaluates_every_checkpoint(tmp_path):
    """`--watcher`: a process beside the trainer evaluates each checkpoint on the held-out pool (`sim2real/watch.py`)."""
    from lerobot.felix_jr.sim2real.watch import checkpoints

    pool = tmp_path / "pool_train.npz"
    _pool(XleRobotModel.load(api.MODEL)).save(pool)
    _pool(XleRobotModel.load(api.MODEL)).save(tmp_path / "pool_test.npz")
    out = tmp_path / "train"
    train.main(
        ["--pool", str(pool), "--device", "cpu", "--num-envs", "4", "--iterations", "2", "--steps-per-env", "4",
         "--save-interval", "1", "--log-dir", str(out), "--video-every", "0", "--eval-worlds", "4", "--watcher",
         "--pool-weights", *(f"{p}=1" for p in POOLS)]
    )  # fmt: skip
    saved = checkpoints(out)
    assert saved, "no numbered checkpoints"
    for checkpoint in saved:
        result = json.loads((out / "evals" / f"{checkpoint.stem}.json").read_text())
        assert result["pool"] == "held_out" and result["worlds"] == 4
    final = json.loads((out / "eval.json").read_text())["held_out"]
    assert final["iteration"] == int(saved[-1].stem.removeprefix("model_"))  # (the watcher's, of the last)
    assert list(out.glob("*.watch")), "no tensorboard events from the watcher"
