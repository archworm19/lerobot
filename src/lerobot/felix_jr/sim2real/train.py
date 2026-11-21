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
start pool (the positive control's measure: from the easy starts it should be high).

On a pod, through `cloud/train.py` (the pool sent along, the run in the background). On the Mac's CPU (`--device cpu`,
a few worlds and iterations): to catch what is miswired before paying for a GPU.
"""

import argparse
import json
import os
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
    with torch.inference_mode():
        for _ in range(int(env.max_episode_length)):
            obs, _, dones, _ = wrapped.step(policy(obs))
            holding = task.held(env) > 0
            ever |= holding & ~done
            ended = dones.bool() & ~done
            held_at_end |= (
                ended & holding
            )  # (an ended world was reset: its held is the next episode's; approximate)
            done |= ended
        held_at_end |= ~done & (task.held(env) > 0)
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
    parser.add_argument("--num-envs", type=int, default=None, help="default 4096 on a GPU, 8 on the CPU")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--steps-per-env", type=int, default=24)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--pool-weights", nargs="+", default=None, metavar="POOL=W")
    parser.add_argument("--object-noise-mm", type=float, default=1000 * me.OBJECT_NOISE_M)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logger", choices=("tensorboard", "wandb"), default="tensorboard")
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--device", default=None, help="default cuda:0 if there is a GPU, else cpu")
    parser.add_argument("--model", type=Path, default=api.MODEL)
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

    model = XleRobotModel.load(args.model)
    cfg = me.env_cfg(
        model,
        num_envs=num_envs,
        start_pool=args.pool,
        pool_weights=weights,
        object_noise_m=args.object_noise_mm / 1000,
        randomize=True,
    )
    cfg.seed = args.seed
    env = me.make_env(cfg, device=device)
    wrapped = RslRlVecEnvWrapper(env)
    agent = runner_cfg(args.iterations, args.seed, args.logger, args.save_interval, args.steps_per_env)
    settings = {
        "pool": str(args.pool),
        "num_envs": num_envs,
        "device": device,
        "pool_weights": weights or me.POOL_WEIGHTS,
        "object_noise_m": args.object_noise_mm / 1000,
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

    runner = MjlabOnPolicyRunner(wrapped, asdict(agent), str(log_dir), device)
    started = time.perf_counter()
    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
    seconds = time.perf_counter() - started
    runner.save(str(log_dir / "model_final.pt"))

    result = evaluate(env, wrapped, runner.get_inference_policy(device=device))
    result |= {"train_seconds": seconds, "iterations": args.iterations, "num_envs": num_envs}
    (log_dir / "eval.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    env.close()


if __name__ == "__main__":
    main()
