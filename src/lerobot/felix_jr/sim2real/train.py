"""Train the grasp policy in the mjlab environment with PPO (rsl_rl, mjlab's runner): the first experiment's script.

  uv run --extra felix_jr_sim --extra felix_jr_gpu python -m lerobot.felix_jr.sim2real.train \
      --pool src/lerobot/felix_jr/sim2real/scratch/pools/pyri1.npz [--num-envs 4096] [--iterations 2000] \
      [--pool-weights attack=0.2 at_object=0.5 in_jaws=0.2 lifted=0.1] [--object-noise-mm 1] [--seed 42] \
      [--logger tensorboard|wandb] [--log-dir DIR] [--device cuda:0|cpu]

The environment: `mjlab_env.env_cfg` from a start pool (`start_pool`), the task's terms (`mjlab_task`). PPO's
settings: mjlab's lift-cube task's (its YAM arm: actor and critic 512-256-128, ELU, observations normalized, an
adaptive learning rate from 1e-3, 24 steps a world an update). The logs and checkpoints in `--log-dir`: `$RUN_DIR/train`
on a pod (the run script's), else `sim2real/scratch/train/<time>/`. At the end the policy is played one episode in
every world, deterministically: `eval.json`, the share of worlds holding the object at the end and at any time, by
start pool (the positive control's measure: from the easy starts it should be high); on the held-out pool too
(`--eval-pool`, default `<pool>_test.npz` beside a `<pool>_train.npz`: placements never trained on, in an environment
PPO never stepped). Each checkpoint, as it is saved, is evaluated on the held-out pool and filmed by a process of its
own beside the trainer (`--watcher`, on by default on a GPU: `sim2real/watch.py`, its output in `watch.log`):
`evals/model_<iteration>.json`, tensorboard's `held_out/...` curves, `videos/model_<iteration>.mp4` (every
`--video-every`-th); the final held-out result is its. Without it (`--no-watcher`), checkpoints are filmed in the trainer
(it waits) and the held-out pool evaluated at the end only.

On a pod, through `cloud/train.py` (the pool sent along, the run in the background). On the Mac's CPU (`--device cpu`,
a few worlds and iterations): to catch what is miswired before paying for a GPU.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel
from lerobot.utils.import_utils import require_package

from . import mjlab_env as me, mjlab_task as task
from .start_pool import POOLS

LOCAL_TRAIN = Path(__file__).parent / "scratch" / "train"


def runner_cfg(iterations: int, seed: int, logger: str, save_interval: int, steps_per_env: int) -> Any:
    """PPO's settings: mjlab's lift-cube task's (`mjlab.tasks.manipulation.config.yam.rl_cfg`)."""
    from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
        ),
        critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="felix_grasp",
        save_interval=save_interval,
        num_steps_per_env=steps_per_env,
        max_iterations=iterations,
        logger=logger,
        seed=seed,
    )


def filming_runner(base: Any) -> Any:
    """`base` (mjlab's runner class) filming the policy at its checkpoints (`videos.record`), if given a video env."""

    class FilmingRunner(base):
        video: tuple[Any, Any, Path, int] | None = None  # (env, wrapper, folder, every n-th checkpoint)
        _saved = 0

        def save(self, path: str, infos: Any = None) -> None:
            super().save(path, infos)
            if self.video is None:
                return
            env, wrapped, folder, every = self.video
            self._saved += 1
            name = Path(path).stem
            if every <= 0 or (name != "model_final" and (self._saved - 1) % every):
                return
            from .videos import record

            try:  # (a failed film never stops the training)
                from .videos import checkpoint_step, log_to_tensorboard

                frames: list[Any] = []
                out = record(
                    env,
                    wrapped,
                    self.get_inference_policy(device=str(env.device)),
                    folder / f"{name}.mp4",
                    name,
                    frames,
                )
                writer = getattr(self.logger, "writer", None)
                if writer is not None and hasattr(writer, "add_image"):  # (tensorboard's: its Images tab)
                    step = checkpoint_step(name, self.current_learning_iteration)
                    log_to_tensorboard(writer, "policy", frames, step, 1.0 / env.step_dt)
                print(f"filmed {name}: {out}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"filming {name} failed: {e!r}", flush=True)
            finally:
                self.alg.train_mode()

    return FilmingRunner


def evaluate(env: Any, wrapped: Any, policy: Any) -> dict[str, Any]:
    """The policy played one episode in every world from a fresh reset: held at the end and ever, by start pool."""
    import torch

    from .start_pool import StartPool

    obs, _ = wrapped.reset()
    sim = env.sim
    pool = env._felix_pool
    layout = StartPool.load(Path(pool["path"])).layout(sim.mj_model)[0]
    now = sim.data.qpos.cpu().numpy()
    rows = [int(np.abs(layout - now[w]).max(axis=1).argmin()) for w in range(env.num_envs)]
    which = StartPool.load(Path(pool["path"])).pool[rows]
    ever = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    done = torch.zeros_like(ever)
    held_at_end = torch.zeros_like(ever)
    holding = task.held(env) > 0
    with torch.inference_mode():
        for _ in range(int(env.max_episode_length)):
            before = holding  # (a world whose episode ends in this step is reset within it: its end is the step before)
            obs, _, dones, _ = wrapped.step(policy(obs))
            holding = task.held(env) > 0
            ended = dones.bool() & ~done
            held_at_end |= ended & before
            ever |= (holding & ~done & ~ended) | (ended & before)
            done |= ended
        held_at_end |= ~done & holding
    out: dict[str, Any] = {"worlds": env.num_envs}
    for name in ("all", *POOLS):
        mask = np.ones(env.num_envs, bool) if name == "all" else which == name
        if mask.any():
            m = torch.as_tensor(mask, device=env.device)
            out[name] = {
                "n": int(mask.sum()),
                "held_at_end": float(held_at_end[m].float().mean()),
                "held_ever": float(ever[m].float().mean()),
            }
    return out


def main(argv: list[str] | None = None) -> None:
    """Command line entry point; see the module docstring."""
    require_package("mjlab", extra="felix_jr_gpu")
    import torch
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

    from lerobot.felix_jr.harness import api

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pool", type=Path, required=True, help="a start pool (`start_pool`)")
    parser.add_argument(
        "--eval-pool",
        type=Path,
        default=None,
        help="the held-out starts (default: <pool>_test.npz beside a <pool>_train.npz, if there); never trained on",
    )
    parser.add_argument("--eval-worlds", type=int, default=2048, help="the held-out evaluation's worlds")
    parser.add_argument("--num-envs", type=int, default=None, help="default 4096 on a GPU, 8 on the CPU")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--steps-per-env", type=int, default=24)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--pool-weights", nargs="+", default=None, metavar="POOL=W")
    parser.add_argument("--object-noise-mm", type=float, default=1000 * me.OBJECT_NOISE_M)
    parser.add_argument(
        "--size-error-mm",
        type=float,
        default=1000 * me.SIZE_ERROR_M,
        help="objects of several sizes: the error in the size the actor is told, an episode's (std)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logger", choices=("tensorboard", "wandb"), default="tensorboard")
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--device", default=None, help="default cuda:0 if there is a GPU, else cpu")
    parser.add_argument("--model", type=Path, default=api.MODEL)
    parser.add_argument(
        "--video-every", type=int, default=1, help="film every n-th checkpoint (and the final one); 0: none"
    )
    parser.add_argument(
        "--video-worlds", type=int, default=6, help="the worlds filmed (every pool equally likely)"
    )
    parser.add_argument(
        "--watcher",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="checkpoints evaluated and filmed by a process beside the trainer (`watch.py`; default: on a GPU)",
    )
    args = parser.parse_args(argv)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    num_envs = args.num_envs or (4096 if device.startswith("cuda") else 8)
    weights = (
        {k: float(v) for k, v in (w.split("=") for w in args.pool_weights)} if args.pool_weights else None
    )
    if args.log_dir is not None:
        log_dir = args.log_dir
    elif os.environ.get("RUN_DIR"):
        log_dir = Path(os.environ["RUN_DIR"]) / "train"
    else:
        log_dir = LOCAL_TRAIN / time.strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)

    eval_pool = args.eval_pool
    if eval_pool is None and args.pool.name.endswith("_train.npz"):
        sibling = args.pool.with_name(args.pool.name.replace("_train.npz", "_test.npz"))
        eval_pool = sibling if sibling.is_file() else None

    model = XleRobotModel.load(args.model)
    cfg = me.env_cfg(
        model,
        num_envs=num_envs,
        start_pool=args.pool,
        pool_weights=weights,
        object_noise_m=args.object_noise_mm / 1000,
        size_error_m=args.size_error_mm / 1000,
        randomize=True,
    )
    cfg.seed = args.seed
    env = me.make_env(cfg, device=device)
    wrapped = RslRlVecEnvWrapper(env)
    agent = runner_cfg(args.iterations, args.seed, args.logger, args.save_interval, args.steps_per_env)
    settings = {
        "pool": str(args.pool),
        "eval_pool": None if eval_pool is None else str(eval_pool),
        "num_envs": num_envs,
        "device": device,
        "pool_weights": weights or me.POOL_WEIGHTS,
        "object_noise_m": args.object_noise_mm / 1000,
        "size_error_m": args.size_error_mm / 1000,
        "reward_weights": me.REWARD_WEIGHTS,
        "action": {
            "arm_step_deg": task.ARM_STEP_DEG,
            "arm_cap_deg": task.ARM_CAP_DEG,
            "gripper_step_units": task.GRIPPER_STEP_UNITS,
            "gripper_cap_units": task.GRIPPER_CAP_UNITS,
        },
        "agent": asdict(agent),
    }
    (log_dir / "settings.json").write_text(json.dumps(settings, indent=2, default=str))
    print(f"training on {device}, {num_envs} worlds, {args.iterations} iterations: {log_dir}", flush=True)

    runner = filming_runner(MjlabOnPolicyRunner)(wrapped, asdict(agent), str(log_dir), device)
    watcher = None
    if args.watcher if args.watcher is not None else device.startswith("cuda"):
        command = [sys.executable, "-m", "lerobot.felix_jr.sim2real.watch", str(log_dir), "--worlds",
                   str(args.eval_worlds), "--video-every", str(args.video_every), "--video-worlds",
                   str(args.video_worlds), "--device", device, "--parent", str(os.getpid())]  # fmt: skip
        with open(log_dir / "watch.log", "w") as log:
            watcher = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        print(f"watching the checkpoints: pid {watcher.pid}, {log_dir / 'watch.log'}", flush=True)
    elif args.video_every > 0:
        from .videos import make_video_env

        video_env, video_wrapped = make_video_env(settings, args.video_worlds, device)
        runner.video = (video_env, video_wrapped, log_dir / "videos", args.video_every)
    started = time.perf_counter()
    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
    seconds = time.perf_counter() - started
    runner.save(str(log_dir / "model_final.pt"))

    policy = runner.get_inference_policy(device=device)
    # the training pool's starts (the training worlds themselves), and the held-out ones in an environment of their
    # own: PPO never collected a step there, its observation normalizer never saw them (the policy frozen, eval mode)
    result: dict[str, Any] = {"train_pool": evaluate(env, wrapped, policy)}
    last = log_dir / "evals" / f"model_{runner.current_learning_iteration}.json"
    if watcher is not None:
        print("waiting for the watcher (the last checkpoints)", flush=True)
        if watcher.wait() != 0:
            print(f"the watcher failed ({watcher.returncode}): {log_dir / 'watch.log'}", flush=True)
    if eval_pool is not None and watcher is not None and last.is_file():
        # (the watcher's, of the last checkpoint: the final policy)
        result["held_out"] = json.loads(last.read_text())
    elif eval_pool is not None:
        from .videos import make_video_env

        held_env, held_wrapped = make_video_env(settings, args.eval_worlds, device, eval_pool)
        result["held_out"] = evaluate(held_env, held_wrapped, policy)
        held_env.close()
    result |= {"train_seconds": seconds, "iterations": args.iterations, "num_envs": num_envs}
    (log_dir / "eval.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    env.close()


if __name__ == "__main__":
    main()
