"""A dataset for the vision probe: a teacher's rollouts seen by the wrist camera, each frame with the object's pose.

  python -m lerobot.felix_jr.sim2real.probe_data <a run's train/ folder> --pool <start pool .npz> --out <dir>
      [--checkpoint model_699.pt] [--worlds 1024] [--rounds 3] [--every 6] [--noise 0.3] [--size 224 168]
      [--appearance]

The teacher (a run's policy, its environment as `settings.json` has it) plays from the pool's starts, every pool equally
likely, `--rounds` times `--steps` policy steps (default an episode's length; worlds that end are reset and go on), its
actions given Gaussian noise of a per-world scale drawn from [0, `--noise`] each round (to see states the teacher
itself avoids). Every `--every`-th step, each world's wrist frame (`render_bench.camera_cfg`, RGB) and its labels:

- `in_pinch` (N, 6), `in_root` (N, 6): the object's center and axis in the pinch frame and the root frame, live and
  without noise (`MjlabCylinder.in_pinch`, `in_root`: the actor's `object_in_pinch`, `object_in_root`);
- `size` (N, 2): its radius and height (true);
- `pinch_pos` (N, 3), `pinch_quat` (N, 4, w x y z): the gripper's pinch frame in the root frame (what the robot knows
  from its joints: a base-frame estimate put in each frame's pinch frame, `head_prior`);
- `visible` (N,): the share of the frame that is the object (the camera's segmentation);
- `held` (N,): held as the reward counts it; `pool` (N,): the start's pool (an index into `pools`); `row` (N,): the
  start's row in the pool file; `step` (N,): the step in the episode; `world`, `round` (N,).

Writes `<out>/frames.npy` (N, H, W, 3) uint8 (a memory map: ~110 kB a frame at 224 x 168) and `<out>/labels.npz`
(with `pools`, the settings and the checkpoint). Split by placement through the pool files (record the run's training
pool and its held-out pool into two folders).
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import mjlab_object as obj, mjlab_task as task
from .pool_reset import pool_reset
from .render_bench import SENSOR, camera_cfg
from .start_pool import POOLS
from .videos import make_video_env

MJ_OBJ_GEOM = 5  # (mujoco.mjtObj.mjOBJ_GEOM: the segmentation's object type for geoms)


def object_share(env: Any, cyl: obj.MjlabCylinder) -> torch.Tensor:
    """Each world's share of the wrist frame that is the object (its geoms in the camera's segmentation)."""
    seg = env.scene.sensors[SENSOR].data.segmentation  # (N, H, W, 2): ids, types
    ids = env.scene[cyl.entity].indexing.geom_ids.to(seg.device)
    hit = torch.isin(seg[..., 0], ids) & (seg[..., 1] == MJ_OBJ_GEOM)
    return hit.flatten(1).float().mean(dim=1)


def record(
    train_dir: Path,
    pool: Path,
    out: Path,
    checkpoint: str | None = None,
    worlds: int = 1024,
    rounds: int = 3,
    steps: int | None = None,
    every: int = 6,
    noise: float = 0.3,
    size: tuple[int, int] = (224, 168),
    device: str = "cuda:0",
    seed: int = 0,
    appearance: bool = False,
) -> dict[str, Any]:
    """The dataset (the module docstring) written into `out`; returns its summary."""
    from mjlab.rl import MjlabOnPolicyRunner

    from .train import runner_cfg

    torch.manual_seed(seed)
    settings = json.loads((train_dir / "settings.json").read_text())
    sensor = camera_cfg(*size, data_types=("rgb", "segmentation"))
    env, wrapped = make_video_env(settings, worlds, device, pool, sensors=(sensor,), appearance=appearance)
    runner = MjlabOnPolicyRunner(wrapped, asdict(runner_cfg(1, 0, "tensorboard", 1, 24)), None, device)
    ckpt = train_dir / (checkpoint or "model_final.pt")
    runner.load(str(ckpt), map_location=device)
    policy = runner.get_inference_policy(device=device)
    cyl, term = obj.of(env), pool_reset(env)
    assert term is not None
    steps = steps or int(env.max_episode_length)
    per_round = len(range(0, steps, every))
    total = rounds * per_round * worlds
    out.mkdir(parents=True, exist_ok=True)
    frames = np.lib.format.open_memmap(
        out / "frames.npy", mode="w+", dtype=np.uint8, shape=(total, size[1], size[0], 3)
    )
    labels: dict[str, list[np.ndarray]] = {
        k: []
        for k in (
            "in_pinch",
            "in_root",
            "size",
            "pinch_pos",
            "pinch_quat",
            "visible",
            "held",
            "pool",
            "row",
            "step",
            "world",
            "round",
        )
    }
    pool_index = {p: i for i, p in enumerate(POOLS)}
    started, at = time.perf_counter(), 0
    world = np.arange(worlds)
    with torch.inference_mode():
        for r in range(rounds):
            scale = noise * torch.rand(worlds, 1, device=device)
            obs, _ = wrapped.reset()
            for k in range(steps):
                action = policy(obs)
                obs, _, _, _ = wrapped.step(action + scale * torch.randn_like(action))
                if k % every:
                    continue
                frames[at : at + worlds] = env.scene.sensors[SENSOR].data.rgb.cpu().numpy()
                at += worlds
                # (copies: on the CPU device `.cpu().numpy()` is a view, and these buffers change in place)
                rows = term.rows.cpu().numpy().copy()
                labels["in_pinch"].append(cyl.in_pinch(env).cpu().numpy())
                pos, quat = task.pinch_pose(env)
                labels["pinch_pos"].append(pos.cpu().numpy().copy())
                labels["pinch_quat"].append(quat.cpu().numpy().copy())
                labels["in_root"].append(cyl.in_root(env).cpu().numpy())
                labels["size"].append(cyl.size(env).cpu().numpy())
                labels["visible"].append(object_share(env, cyl).cpu().numpy())
                labels["held"].append((task.held(env, cyl) > 0).cpu().numpy())
                labels["pool"].append(np.array([pool_index[str(p)] for p in term.pool.pool[rows]]))
                labels["row"].append(rows)
                labels["step"].append(env.episode_length_buf.cpu().numpy().copy())
                labels["world"].append(world)
                labels["round"].append(np.full(worlds, r))
            print(f"round {r + 1}/{rounds}: {at} frames ({time.perf_counter() - started:.0f} s)", flush=True)
    frames.flush()
    arrays = {k: np.concatenate(v) for k, v in labels.items()}
    summary = {
        "frames": at,
        "size": list(size),
        "train_dir": str(train_dir),
        "checkpoint": ckpt.name,
        "pool": str(pool),
        "noise": noise,
        "appearance": appearance,
        "every": every,
        "seconds": time.perf_counter() - started,
        "visible_share": float((arrays["visible"] > 0).mean()),
        "held_share": float(arrays["held"].mean()),
    }
    np.savez(out / "labels.npz", **arrays, pools=np.array(POOLS), summary=json.dumps(summary))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    env.close()
    return summary


def main(argv: list[str] | None = None) -> None:
    """The command line (the module docstring)."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("train_dir", type=Path)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", default=None, help="default: model_final.pt")
    parser.add_argument("--worlds", type=int, default=1024)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--steps", type=int, default=None, help="policy steps a round (default: an episode's)"
    )
    parser.add_argument("--every", type=int, default=6)
    parser.add_argument("--noise", type=float, default=0.3)
    parser.add_argument("--size", type=int, nargs=2, default=[224, 168], metavar=("W", "H"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--appearance", action="store_true", help="the look randomized (`appearance`)")
    args = parser.parse_args(argv)
    summary = record(
        args.train_dir,
        args.pool,
        args.out,
        args.checkpoint,
        args.worlds,
        args.rounds,
        args.steps,
        args.every,
        args.noise,
        tuple(args.size),
        args.device,
        args.seed,
        args.appearance,
    )
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
