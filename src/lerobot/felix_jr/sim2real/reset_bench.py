"""Where a multi-size pool's training step goes: the environment stepped alone, and its reset taken apart.

  python -m lerobot.felix_jr.sim2real.reset_bench --pool <one size .npz> --pool <several sizes .npz>
      [--num-envs 4096] [--steps 100] [--profile 5]

For each pool: the environment (`mjlab_env.env_cfg`, randomized as in training) stepped with random actions, no PPO:
steps a second, and how many steps reset at least one world (each such step refreshes the model's constants, for a
pool whose objects vary in size). For a pool of several sizes, its reset's parts timed alone, the GPU synchronized
around each: the constants' refresh (`sim.recompute_constants`, MuJoCo Warp's `set_const`), the object's field writes
(`MjlabCylinder.write`), a whole reset of a tenth of the worlds (`env._reset_idx`). Then `torch.profiler` over a few
steps: the operations taking the most time, the device's and the host's.
"""

import argparse
import time
from pathlib import Path
from typing import Any

import torch

from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel
from lerobot.felix_jr.harness import api

from . import mjlab_env as me, mjlab_object as mo
from .start_pool import POOLS


def _timed(fn: Any, repeats: int, device: str) -> float:
    """`fn`'s mean time (ms) over `repeats` calls, the device synchronized around them."""
    sync = torch.cuda.synchronize if device.startswith("cuda") else (lambda: None)
    fn()
    sync()
    started = time.perf_counter()
    for _ in range(repeats):
        fn()
    sync()
    return 1000 * (time.perf_counter() - started) / repeats


def bench(
    pool: Path, num_envs: int, steps: int, profile: int, device: str, skip_refresh: bool = False
) -> None:
    """One pool's numbers, printed (`skip_refresh`: the constants' refresh made a no-op, to measure what it costs)."""
    from mjlab.managers.event_manager import RecomputeLevel

    cfg = me.env_cfg(
        XleRobotModel.load(api.MODEL),
        num_envs=num_envs,
        start_pool=pool,
        pool_weights=dict.fromkeys(POOLS, 1.0),
        randomize=True,
    )
    env = me.make_env(cfg, device=device)
    if skip_refresh:  # (physics slightly off: the derived constants stale; only to time the refresh's share)
        env.sim.recompute_constants = lambda level: None
    env.reset()
    n_act = env.action_manager.total_action_dim
    act = lambda: 2 * torch.rand(num_envs, n_act, device=device) - 1  # noqa: E731
    resets, niter = [0], []

    def step() -> None:
        _, _, terminated, truncated, _ = env.step(act())
        resets[0] += int(bool((terminated | truncated).any()))
        niter.append(int(env.sim.data.solver_niter.max()))

    for _ in range(10):
        step()
    resets[0], niter[:] = 0, []
    ms = _timed(step, steps, device)
    cyl = mo.of(env)
    worst = sorted(niter)
    print(
        f"{pool.name}{' (refresh skipped)' if skip_refresh else ''}: {num_envs} worlds, sizes vary: {cyl.sized}; a step "
        f"{ms:.1f} ms ({1000 / ms:.1f} steps/s, {num_envs * 1000 / ms:,.0f} world-steps/s); steps with a reset: "
        f"{resets[0]} of {steps + 1}; the slowest world's solver iterations (a step's last substep): median "
        f"{worst[len(worst) // 2]}, max {worst[-1]}",
        flush=True,
    )
    tenth = torch.arange(0, num_envs, 10, device=device)
    print(
        f"  a reset of a tenth of the worlds (env._reset_idx): {_timed(lambda: env._reset_idx(tenth), 20, device):.1f} ms"
    )
    if cyl.sized:
        from .pool_reset import pool_reset

        term = pool_reset(env)
        rows = term.objects[term.rows[tenth]]
        print(
            f"  the constants' refresh (set_const), all worlds: "
            f"{_timed(lambda: env.sim.recompute_constants(RecomputeLevel.set_const), 20, device):.1f} ms"
        )
        print(
            f"  the object's field writes, a tenth: {_timed(lambda: cyl.write(env, tenth, rows), 20, device):.2f} ms"
        )
    if profile > 0:
        from torch.profiler import ProfilerActivity, profile as torch_profile

        activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.startswith("cuda") else [])
        with torch_profile(activities=activities) as prof:
            for _ in range(profile):
                step()
            if device.startswith("cuda"):
                torch.cuda.synchronize()
        key = "cuda_time_total" if device.startswith("cuda") else "cpu_time_total"
        print(prof.key_averages().table(sort_by=key, row_limit=12), flush=True)
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=12), flush=True)
    env.close()


def main(argv: list[str] | None = None) -> None:
    """Command line entry point; see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pool", type=Path, action="append", required=True)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--profile", type=int, default=5, help="steps profiled (0: none)")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--skip-refresh", action="store_true", help="also each pool with the constants' refresh off"
    )
    args = parser.parse_args(argv)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    for pool in args.pool:
        bench(pool, args.num_envs, args.steps, args.profile, device)
        if args.skip_refresh:
            bench(pool, args.num_envs, args.steps, 0, device, skip_refresh=True)


if __name__ == "__main__":
    main()
