"""A training run's checkpoints evaluated on its held-out starts and filmed as they are saved, beside the trainer.

  uv run --extra felix_jr_sim --extra felix_jr_gpu python -m lerobot.felix_jr.sim2real.watch <run's train/ folder>
      [--worlds 2048] [--video-every 1] [--video-worlds 6] [--device cuda:0|cpu] [--parent PID] [--once]

A process of its own (`sim2real/train.py` starts it, `--watcher`), so training never waits on it and nothing it does
reaches PPO: its environments are its own (the held-out pool's, `settings.json`'s `eval_pool`, and a few worlds to film),
the policy loaded from each checkpoint frozen (eval mode, its observation normalizer as saved). For each `model_<n>.pt`
once written: one episode in every world (`train.evaluate`: held at the end and ever, by start pool), from the same
starts every time (the random draws seeded alike), into `evals/model_<n>.json` and the run's tensorboard
(`held_out/<pool>/held_at_end`, `.../held_ever` at iteration n); every `--video-every`-th filmed (`videos.record`:
`videos/model_<n>.mp4`, and a GIF and a contact sheet in tensorboard). It sharing the GPU slows training by about its
share of the work (a held-out pass of 2048 worlds is ~7-10 training iterations' simulation; one a checkpoint every 100
iterations: ~5-10 %). It stops after the last checkpoint once `model_final.pt` is there (a copy of the last), when the
trainer (`--parent`) is gone, or, `--once`, after the checkpoints already there (a finished run evaluated after the
fact).
"""

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.utils.import_utils import require_package

from .videos import checkpoint_step, log_to_tensorboard, make_video_env, record

SETTLE_S = 5.0  # a checkpoint is read once unchanged this long (torch.save writes it in place)


def checkpoints(train_dir: Path) -> list[Path]:
    """The run's numbered checkpoints (`model_<n>.pt`; not `model_final.pt`, a copy of the last), by iteration."""
    found = [p for p in train_dir.glob("model_*.pt") if p.stem.removeprefix("model_").isdigit()]
    return sorted(found, key=lambda p: int(p.stem.removeprefix("model_")))


def settled(path: Path, seen: dict[Path, tuple[int, float]], now: float) -> bool:
    """`path` the same size since the last look (`seen`, updated) and untouched for `SETTLE_S`."""
    stat = path.stat()
    before = seen.get(path)
    seen[path] = (stat.st_size, stat.st_mtime)
    return before == (stat.st_size, stat.st_mtime) and now - stat.st_mtime >= SETTLE_S


def alive(pid: int | None) -> bool:
    """The process `pid` still running (None: no parent to watch)."""
    if pid is None:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def log_eval(writer: Any, result: dict[str, Any], step: int, tag: str = "held_out") -> None:
    """`train.evaluate`'s result into tensorboard at `step`: `<tag>/<pool>/held_at_end`, `.../held_ever`."""
    for name, row in result.items():
        if isinstance(row, dict):
            for key in ("held_at_end", "held_ever"):
                writer.add_scalar(f"{tag}/{name}/{key}", row[key], step)
    writer.flush()


class Watcher:
    """The environments and a runner to load checkpoints into, kept across checkpoints."""

    def __init__(self, train_dir: Path, worlds: int, video_worlds: int, device: str, seed: int = 0) -> None:
        """From `train_dir`'s `settings.json`: `worlds` held-out worlds, `video_worlds` to film (0: none)."""
        from mjlab.rl import MjlabOnPolicyRunner
        from torch.utils.tensorboard import SummaryWriter

        from .train import runner_cfg

        self.train_dir, self.device, self.seed = train_dir, device, seed
        settings = json.loads((train_dir / "settings.json").read_text())
        # (no held-out pool: the training pool's starts)
        self.pool = settings.get("eval_pool") or settings["pool"]
        self.tag = "held_out" if settings.get("eval_pool") else "train_pool"
        self.env, self.wrapped = make_video_env(settings, worlds, device, self.pool)
        self.film = make_video_env(settings, video_worlds, device) if video_worlds > 0 else None
        self.runner = MjlabOnPolicyRunner(
            self.wrapped, asdict(runner_cfg(1, 0, "tensorboard", 1, 24)), None, device
        )
        # (an event file of its own beside the trainer's: tensorboard reads every file)
        self.writer = SummaryWriter(log_dir=str(train_dir), filename_suffix=".watch")
        (train_dir / "evals").mkdir(exist_ok=True)

    def done(self, checkpoint: Path) -> bool:
        """`checkpoint` already evaluated (its `evals/<name>.json` there)."""
        return (self.train_dir / "evals" / f"{checkpoint.stem}.json").is_file()

    def process(self, checkpoint: Path, film: bool) -> dict[str, Any]:
        """`checkpoint` evaluated (`evals/<name>.json`, tensorboard) and, `film`, filmed."""
        import torch

        from .train import evaluate

        step = checkpoint_step(checkpoint.stem, 0)
        self.runner.load(str(checkpoint), map_location=self.device)
        policy = self.runner.get_inference_policy(device=self.device)
        started = time.perf_counter()
        torch.manual_seed(self.seed)  # (the same starts and physics draws for every checkpoint)
        np.random.seed(self.seed)
        result = evaluate(self.env, self.wrapped, policy) | {"pool": self.tag, "iteration": step}
        result["seconds"] = time.perf_counter() - started
        (self.train_dir / "evals" / f"{checkpoint.stem}.json").write_text(json.dumps(result, indent=2))
        log_eval(self.writer, result, step, self.tag)
        line = f"{checkpoint.name}: {self.tag} held at the end {100 * result['all']['held_at_end']:.1f} %"
        if film and self.film is not None:
            env, wrapped = self.film
            torch.manual_seed(self.seed)
            frames: list[np.ndarray] = []
            try:  # (a failed film never stops the evaluations)
                record(
                    env,
                    wrapped,
                    self.runner.get_inference_policy(device=self.device),
                    self.train_dir / "videos" / f"{checkpoint.stem}.mp4",
                    checkpoint.stem,
                    frames,
                )
                log_to_tensorboard(self.writer, "policy", frames, step, 1.0 / env.step_dt)
                line += ", filmed"
            except Exception as e:  # noqa: BLE001
                line += f", filming failed: {e!r}"
        print(f"{line} ({result['seconds']:.0f} s)", flush=True)
        return result

    def close(self) -> None:
        """The tensorboard writer and the environments closed."""
        self.writer.close()
        self.env.close()
        if self.film is not None:
            self.film[0].close()


def watch(
    train_dir: Path,
    worlds: int = 2048,
    video_every: int = 1,
    video_worlds: int = 6,
    device: str | None = None,
    parent: int | None = None,
    once: bool = False,
    poll_s: float = 10.0,
) -> list[dict[str, Any]]:
    """Evaluate (and film) `train_dir`'s checkpoints as they come; see the module docstring. The results, in order."""
    import torch

    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    while not (train_dir / "settings.json").is_file():  # (the trainer writes it before its first checkpoint)
        if once or not alive(parent):
            raise FileNotFoundError(f"{train_dir / 'settings.json'}: not a training run's folder")
        time.sleep(poll_s)
    watcher = Watcher(train_dir, worlds, video_worlds if video_every > 0 else 0, device)
    seen: dict[Path, tuple[int, float]] = {}
    results: list[dict[str, Any]] = []
    count = 0  # (checkpoints met, for every n-th filmed)
    try:
        while True:
            finished = once or (train_dir / "model_final.pt").is_file() or not alive(parent)
            pending = [p for p in checkpoints(train_dir) if not watcher.done(p)]
            for checkpoint in pending:
                if not (finished or settled(checkpoint, seen, time.time())):
                    break  # (in order: a later one waits for this one)
                count += 1
                last = finished and checkpoint == pending[-1]
                film = video_every > 0 and ((count - 1) % video_every == 0 or last)
                results.append(watcher.process(checkpoint, film))
            if finished and all(watcher.done(p) for p in checkpoints(train_dir)):
                return results
            time.sleep(poll_s)
    finally:
        watcher.close()


def main(argv: list[str] | None = None) -> None:
    """Command line entry point; see the module docstring."""
    require_package("mjlab", extra="felix_jr_gpu")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("train_dir", type=Path, help="a run's train/ folder (settings.json, model_*.pt)")
    parser.add_argument("--worlds", type=int, default=2048, help="the held-out evaluation's worlds")
    parser.add_argument(
        "--video-every", type=int, default=1, help="film every n-th checkpoint (and the last); 0: none"
    )
    parser.add_argument("--video-worlds", type=int, default=6)
    parser.add_argument("--device", default=None, help="default cuda:0 if there is a GPU, else cpu")
    parser.add_argument("--parent", type=int, default=None, help="the trainer's pid: stop once it is gone")
    parser.add_argument("--once", action="store_true", help="the checkpoints already there, then stop")
    parser.add_argument("--poll-s", type=float, default=10.0)
    args = parser.parse_args(argv)
    watch(
        args.train_dir,
        args.worlds,
        args.video_every,
        args.video_worlds,
        args.device,
        args.parent,
        args.once,
        args.poll_s,
    )


if __name__ == "__main__":
    main()
