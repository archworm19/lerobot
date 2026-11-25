"""A policy's episodes filmed in the mjlab environment: a tile a world, a camera on its object, labelled.

`record(env, policy, path)`: one episode in each of `env`'s worlds (a few: the training's own worlds are too many to
watch), from fresh resets, rendered every policy step (~30 fps) by MuJoCo's renderer from a camera aimed at each world's
object, the tiles side by side, each labelled with its start pool and what is happening (holding, both fingers on, the
object tipped, the arm on the table), into an mp4; and into tensorboard (`log_to_tensorboard`: an animated GIF and a
contact sheet at the checkpoint's iteration, in its Images tab). `sim2real/train.py` calls it at every checkpoint (`--video-every`),
on the pod's GPU (EGL); `main` films a checkpoint after the fact.

  uv run --extra felix_jr_sim --extra felix_jr_gpu python -m lerobot.felix_jr.sim2real.videos <run's train/ folder>
      [--checkpoint model_1400.pt] [--worlds 6] [--device cpu] [--tensorboard]
"""

import argparse
import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from . import mjlab_object as obj, mjlab_task as task
from .mjlab_object import SIZE_ERROR_M
from .start_pool import POOLS

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

TILE = (320, 240)
CAMERA = {
    "distance": 0.55,
    "azimuth": 90.0,
    "elevation": -35.0,
}  # (from the side, above: the arm, jaws and object)
GIF_EVERY, GIF_SCALE = 2, 0.5  # (tensorboard's: every other frame, half size; the mp4 keeps them all)
SHEET_AT_S = (0.0, 1.5, 4.0, 11.5)  # the contact sheet's moments
COLORS = {"held": (0, 200, 0), "fingers": (255, 200, 0), "tipped": (230, 0, 0), "table": (230, 0, 0)}


def _label(image: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    for k, (text, color) in enumerate(lines):
        for c, width in (((0, 0, 0), 3), (color, 1)):
            cv2.putText(image, text, (6, 18 + 18 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, width, cv2.LINE_AA)


class SideView:
    """Each world's side camera (`CAMERA`, aimed at its object), rendered by MuJoCo on a copy of the model."""

    def __init__(self, env: "ManagerBasedRlEnv") -> None:
        """The renderer for `env`'s worlds (each world's object size given to the copy as it renders it)."""
        import mujoco

        self.env = env
        self.m = copy.deepcopy(env.sim.mj_model)
        self.d = mujoco.MjData(self.m)
        self.renderer = mujoco.Renderer(self.m, height=TILE[1], width=TILE[0])
        self.camera = mujoco.MjvCamera()
        self.camera.distance, self.camera.azimuth, self.camera.elevation = (
            CAMERA["distance"],
            CAMERA["azimuth"],
            CAMERA["elevation"],
        )
        self.cyl = obj.of(env)
        _, self.object_geom = self.cyl.ids(env)
        self.sizes = env.sim.model.geom_size[..., self.object_geom, :].cpu().numpy().reshape(-1, 3)

    def tiles(self, worlds: range) -> list[np.ndarray]:
        """The worlds' current views (RGB, `TILE`)."""
        import mujoco

        env = self.env
        qpos = env.sim.data.qpos.cpu().numpy()
        qvel = env.sim.data.qvel.cpu().numpy()
        centers = self.cyl.pose(env)[0].cpu().numpy()
        out = []
        for w in worlds:
            self.d.qpos[:], self.d.qvel[:] = qpos[w], qvel[w]
            self.m.geom_size[self.object_geom] = self.sizes[min(w, len(self.sizes) - 1)]
            mujoco.mj_forward(self.m, self.d)
            self.camera.lookat[:] = centers[w] + np.array([0.0, 0.0, 0.02])
            self.renderer.update_scene(self.d, self.camera)
            out.append(self.renderer.render().copy())
        return out

    def close(self) -> None:
        """The renderer released."""
        self.renderer.close()


def label(
    tile: np.ndarray,
    env: "ManagerBasedRlEnv",
    w: int,
    pool: str,
    step: int,
    ended: str | None,
    title: str = "",
) -> None:
    """`tile` (world `w`'s) labelled: the title, its start pool and time, then how it ended or what is happening."""
    cyl = obj.of(env)
    held = bool(task.held(env, cyl)[w] > 0)
    fingers = int(task.finger_contacts(env)[w].sum())
    state = (
        [(f"ended: {ended}", COLORS["tipped"])]
        if ended
        else [("holding", COLORS["held"])]
        if held
        else [(f"fingers on: {fingers}", COLORS["fingers"])]
    )
    _label(tile, [(f"{title} {pool}  t {step * env.step_dt:4.1f} s", (255, 255, 255)), *state])


def end_reasons(env: "ManagerBasedRlEnv", dones: np.ndarray, ended: list[str | None]) -> None:
    """`ended` updated: why each world's episode ended (tipped, arm on the table, or time), the first time it does."""
    tm = env.termination_manager
    for w in np.flatnonzero(dones):
        if w < len(ended) and ended[w] is None:
            ended[w] = next(
                (name for name in ("tipped", "arm_on_table") if bool(tm.get_term(name)[w])),
                "time",
            )


def record(
    env: "ManagerBasedRlEnv",
    wrapped: Any,
    policy: Callable[[Any], Any],
    path: Path,
    title: str = "",
    frames: list[np.ndarray] | None = None,
) -> Path:
    """One episode in each world of `env` (`wrapped`: its rsl_rl wrapper) by `policy`, filmed into `path` (mp4).

    `frames`: every `GIF_EVERY`-th frame appended to it too, at `GIF_SCALE` (for tensorboard: `log_to_tensorboard`).
    """
    import torch

    from .pool_reset import pool_reset

    obs, _ = wrapped.reset()
    view = SideView(env)
    term = pool_reset(env)
    n = env.num_envs
    pools = term.pools_of_worlds() if term is not None else [""] * n
    ended: list[str | None] = [None] * n
    fps = max(1, int(round(1.0 / env.step_dt)))
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (TILE[0] * n, TILE[1]))
    try:
        with torch.inference_mode():
            for step in range(int(env.max_episode_length)):
                tiles = view.tiles(range(n))
                for w, tile in enumerate(tiles):
                    label(tile, env, w, pools[w], step, ended[w], title)
                strip = np.hstack(tiles)
                writer.write(cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
                if frames is not None and step % GIF_EVERY == 0:
                    frames.append(
                        cv2.resize(strip, None, fx=GIF_SCALE, fy=GIF_SCALE, interpolation=cv2.INTER_AREA)
                    )
                obs, _, dones, _ = wrapped.step(policy(obs))
                end_reasons(env, dones.cpu().numpy(), ended)
                if all(e is not None for e in ended):
                    break
    finally:
        writer.release()
        view.close()
    return path


def log_to_tensorboard(writer: Any, tag: str, frames: list[np.ndarray], step: int, fps: float) -> None:
    """`frames` into tensorboard at `step`: an animated GIF (its Images tab plays them) and a contact sheet.

    The GIF goes in as an image summary of GIF bytes, which is what `add_video` writes (without needing moviepy).
    """
    import io

    import imageio.v3 as iio
    from tensorboard.compat.proto.summary_pb2 import Summary

    if not frames:
        return
    buffer = io.BytesIO()
    iio.imwrite(buffer, np.stack(frames), extension=".gif", duration=1000.0 * GIF_EVERY / fps, loop=0)
    h, w = frames[0].shape[:2]
    gif = Summary.Image(height=h, width=w, colorspace=3, encoded_image_string=buffer.getvalue())
    writer._get_file_writer().add_summary(
        Summary(value=[Summary.Value(tag=f"{tag}/episode", image=gif)]), step
    )
    picks = [frames[min(len(frames) - 1, int(t * fps / GIF_EVERY))] for t in SHEET_AT_S]
    writer.add_image(f"{tag}/sheet", np.vstack(picks), step, dataformats="HWC")
    writer.flush()


def checkpoint_step(name: str, final_step: int) -> int:
    """A checkpoint's training iteration from its name (`model_<n>`; `model_final`: `final_step`)."""
    tail = name.removeprefix("model_")
    return int(tail) if tail.isdigit() else final_step


def make_video_env(
    settings: dict[str, Any],
    worlds: int,
    device: str,
    pool: str | Path | None = None,
    sensors: tuple[Any, ...] = (),
    appearance: bool | None = None,
) -> tuple[Any, Any]:
    """An environment as a run's (`settings.json`), every pool equally likely: (the env, its rsl_rl wrapper).

    `pool`: its starts (default: the run's held-out pool, `eval_pool`, if it has one, else its training pool).
    `sensors`: mjlab sensors added to the scene's (e.g. the wrist camera, `render_bench.camera_cfg`). `appearance`: the
    look randomized (`mjlab_env.env_cfg`; default: as the run's settings have it, else not).
    """
    from mjlab.rl import RslRlVecEnvWrapper

    from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel
    from lerobot.felix_jr.harness import api

    from . import mjlab_env as me

    cfg = me.env_cfg(
        XleRobotModel.load(api.MODEL),
        num_envs=worlds,
        start_pool=str(pool or settings.get("eval_pool") or settings["pool"]),
        pool_weights=dict.fromkeys(POOLS, 1.0),
        object_noise_m=float(settings.get("object_noise_m", me.OBJECT_NOISE_M)),
        size_error_m=float(settings.get("size_error_m", SIZE_ERROR_M)),
        actor_exclude=tuple(settings.get("actor_exclude", ())),
        object_pose=str(settings.get("object_pose", "live")),
        estimate_noise_m=float(settings.get("estimate_noise_m", 0.003)),
        grasp_point_fit=str(settings.get("grasp_point_fit", "side")),
        appearance=bool(settings.get("appearance", False)) if appearance is None else appearance,
        randomize=True,
    )
    cfg.scene.sensors = (*cfg.scene.sensors, *sensors)
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
    parser.add_argument("--tensorboard", action="store_true", help="also log it into the run's tensorboard")
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
    runner.load(str(checkpoint), map_location=device)
    frames: list[np.ndarray] = []
    out = record(
        env,
        wrapped,
        runner.get_inference_policy(device=device),
        args.train_dir / "videos" / f"{checkpoint.stem}.mp4",
        checkpoint.stem,
        frames,
    )
    print(f"filmed {checkpoint.name}: {out}")
    if args.tensorboard:  # (an event file of its own in the run's folder: tensorboard reads them all)
        from torch.utils.tensorboard import SummaryWriter

        step = checkpoint_step(checkpoint.stem, runner.current_learning_iteration)
        tb = SummaryWriter(log_dir=str(args.train_dir))
        log_to_tensorboard(tb, "policy", frames, step, 1.0 / env.step_dt)
        tb.close()
        print(f"in tensorboard at step {step}: {args.train_dir}")


if __name__ == "__main__":
    main()
