"""A policy's episodes filmed in the mjlab environment: a tile a world, a camera on its object, labelled.

`record(env, policy, path)`: one episode in each of `env`'s worlds (a few: the training's own worlds are too many to
watch), from fresh resets, rendered every policy step (~30 fps) by MuJoCo's renderer from a camera aimed at each world's
object, the tiles side by side, each labelled with its start pool and what is happening (holding, both fingers on, the
object tipped, the arm on the table), into an mp4. `sim2real/train.py` calls it at every checkpoint (`--video-every`),
on the pod's GPU (EGL); `main` films a checkpoint after the fact.

  uv run --extra felix_jr_sim --extra felix_jr_gpu python -m lerobot.felix_jr.sim2real.videos <run's train/ folder>
      [--checkpoint model_1400.pt] [--worlds 6] [--device cpu]
"""

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from . import mjlab_task as task
from .start_pool import POOLS, StartPool

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

TILE = (320, 240)
CAMERA = {
    "distance": 0.55,
    "azimuth": 90.0,
    "elevation": -35.0,
}  # (from the side, above: the arm, jaws and object)
COLORS = {"held": (0, 200, 0), "fingers": (255, 200, 0), "tipped": (230, 0, 0), "table": (230, 0, 0)}


def _pools_of_worlds(env: "ManagerBasedRlEnv") -> list[str]:
    """Each world's start pool (the start it holds, found in the pool by its state)."""
    cache = env._felix_pool  # type: ignore[attr-defined]
    pool = StartPool.load(Path(cache["path"]))
    layout = pool.layout(env.sim.mj_model)[0]
    now = env.sim.data.qpos.cpu().numpy()
    return [str(pool.pool[int(np.abs(layout - now[w]).max(axis=1).argmin())]) for w in range(env.num_envs)]


def _label(image: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    for k, (text, color) in enumerate(lines):
        for c, width in (((0, 0, 0), 3), (color, 1)):
            cv2.putText(image, text, (6, 18 + 18 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, width, cv2.LINE_AA)


def record(
    env: "ManagerBasedRlEnv", wrapped: Any, policy: Callable[[Any], Any], path: Path, title: str = ""
) -> Path:
    """One episode in each world of `env` (`wrapped`: its rsl_rl wrapper) by `policy`, filmed into `path` (mp4)."""
    import mujoco
    import torch

    obs, _ = wrapped.reset()
    m = env.sim.mj_model
    d = mujoco.MjData(m)
    renderer = mujoco.Renderer(m, height=TILE[1], width=TILE[0])
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = (
        CAMERA["distance"],
        CAMERA["azimuth"],
        CAMERA["elevation"],
    )
    pools = _pools_of_worlds(env)
    n = env.num_envs
    ended: list[str | None] = [None] * n
    fps = max(1, int(round(1.0 / env.step_dt)))
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (TILE[0] * n, TILE[1]))
    try:
        with torch.inference_mode():
            for step in range(int(env.max_episode_length)):
                qpos = env.sim.data.qpos.cpu().numpy()
                qvel = env.sim.data.qvel.cpu().numpy()
                obj = task._object(env)[0].cpu().numpy()
                held = (task.held(env) > 0).cpu().numpy()
                fingers = task.finger_contacts(env).cpu().numpy()
                tiles = []
                for w in range(n):
                    d.qpos[:], d.qvel[:] = qpos[w], qvel[w]
                    mujoco.mj_forward(m, d)
                    camera.lookat[:] = obj[w] + np.array([0.0, 0.0, 0.02])
                    renderer.update_scene(d, camera)
                    tile = renderer.render().copy()
                    state = (
                        [(f"ended: {ended[w]}", COLORS["tipped"])]
                        if ended[w]
                        else [("holding", COLORS["held"])]
                        if held[w]
                        else [(f"fingers on: {int(fingers[w].sum())}", COLORS["fingers"])]
                    )
                    _label(
                        tile,
                        [(f"{title} {pools[w]}  t {step * env.step_dt:4.1f} s", (255, 255, 255)), *state],
                    )
                    tiles.append(tile)
                writer.write(cv2.cvtColor(np.hstack(tiles), cv2.COLOR_RGB2BGR))
                obs, _, dones, _ = wrapped.step(policy(obs))
                tm = env.termination_manager
                for w in np.flatnonzero(dones.cpu().numpy()):
                    if ended[w] is None:
                        reason = next(
                            (name for name in ("tipped", "arm_on_table") if bool(tm.get_term(name)[w])),
                            "time",
                        )
                        ended[w] = reason
                if all(e is not None for e in ended):
                    break
    finally:
        writer.release()
        renderer.close()
    return path


def make_video_env(settings: dict[str, Any], worlds: int, device: str) -> tuple[Any, Any]:
    """A small environment as a run's (`settings.json`), every pool equally likely: (the env, its rsl_rl wrapper)."""
    from mjlab.rl import RslRlVecEnvWrapper

    from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel
    from lerobot.felix_jr.harness import api

    from . import mjlab_env as me

    cfg = me.env_cfg(
        XleRobotModel.load(api.MODEL),
        num_envs=worlds,
        start_pool=settings["pool"],
        pool_weights=dict.fromkeys(POOLS, 1.0),
        object_noise_m=float(settings.get("object_noise_m", me.OBJECT_NOISE_M)),
        randomize=True,
    )
    env = me.make_env(cfg, device=device)
    return env, RslRlVecEnvWrapper(env)


def main(argv: list[str] | None = None) -> None:
    """Film a checkpoint of a run (its `train/` folder) after the fact."""
    from dataclasses import asdict

    import torch
    from mjlab.rl import MjlabOnPolicyRunner

    from .train import runner_cfg

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("train_dir", type=Path, help="a run's train/ folder (settings.json, model_*.pt)")
    parser.add_argument("--checkpoint", default=None, help="default: the latest model_*.pt")
    parser.add_argument("--worlds", type=int, default=6)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    settings = json.loads((args.train_dir / "settings.json").read_text())
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoints = sorted(
        args.train_dir.glob("model_*.pt"), key=lambda p: p.stat().st_mtime
    )  # (the last saved last)
    checkpoint = args.train_dir / args.checkpoint if args.checkpoint else checkpoints[-1]
    env, wrapped = make_video_env(settings, args.worlds, device)
    agent = runner_cfg(1, 0, "tensorboard", 1, 24)
    runner = MjlabOnPolicyRunner(wrapped, asdict(agent), str(args.train_dir / "videos"), device)
    runner.load(str(checkpoint))
    out = record(
        env,
        wrapped,
        runner.get_inference_policy(device=device),
        args.train_dir / "videos" / f"{checkpoint.stem}.mp4",
        checkpoint.stem,
    )
    print(f"filmed {checkpoint.name}: {out}")


if __name__ == "__main__":
    main()
