"""The closed-loop test of the vision probe: a teacher that sees the object only through the probe.

  python -m lerobot.felix_jr.sim2real.probe_loop <a run's train/ folder> --heads <probe_train's heads.pt>
      --pool <held-out pool .npz> --out <dir> [--worlds 4096] [--conditions truth truth_path probe ...] [--hub <dir>]

The teacher (a run's policy; experiment A's: its actor sees no finger contacts) plays one episode in every world from
fresh resets, every start pool equally likely, the wrist camera rendering (`render_bench.camera_cfg`). Its object
inputs (`object_in_root`, `object_in_pinch`: center and axis; `object_size`: radius and height) are replaced each step
according to the condition; everything else (physics, rewards, the success check) uses the true state:

- `truth`: none replaced (the run's own: the true pose with 1 mm of noise, the size with the episode's error);
- `truth_path`: the same values, computed here and written into the observation (the plumbing: matches `truth`);
- `probe`: the probe's estimate from the frame each step (`probe_train`'s attention head on frozen DINOv2-S); the pose
  in the root frame from it and the gripper's pose (the robot's own); out of view (its logit), the last estimate kept;
- `probe_frozen`: the probe's estimate at the episode's first frame, kept in the root frame (B1's situation);
- `matched_noise`: the truth plus noise as large and as correlated in time as the probe's errors (each axis of the
  pinch frame, the axis's components, the size: a first-order autoregression whose spread and step-to-step
  correlation are measured on the `probe` condition's own steps; it runs first);
- `probe_smoothed`: the probe's estimate averaged over steps in the root frame (an exponential average, `SMOOTH`).

The frame a step's estimate comes from is rendered at the end of the step before, after any reset: the same delay
as the true pose's. Reports per condition: held at the end and at any point, by start pool; how the episodes ended
(held, held then lost, never held); for the probe's conditions, its error along the episodes (pinch frame,
per axis, held and not) and at the step both fingers first touch, for episodes held at the end and not. Writes
`<out>/probe_loop.json`.
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
from .mjlab_object import rotate, rotate_inverse
from .pool_reset import pool_reset
from .probe_train import HEADS, decode, dinov2_small, tokens
from .render_bench import SENSOR, camera_cfg
from .start_pool import POOLS
from .videos import make_video_env

CONDITIONS = ("truth", "truth_path", "probe", "probe_frozen", "matched_noise", "probe_smoothed")
OBJECT_TERMS = ("object_in_root", "object_in_pinch", "object_size")
SMOOTH = 0.5  # `probe_smoothed`: the new estimate's weight each step
# `matched_noise`'s columns (center 3, axis 3, radius, height) by name, to add its noise to some inputs only
COLUMNS = {"center_x": [0], "center_y": [1], "center_z": [2], "axis": [3, 4, 5], "radius": [6], "height": [7]}


class Probe:
    """The trained attention head on frozen DINOv2-S: wrist frames -> the object in the pinch frame and its size."""

    def __init__(self, heads: Path, device: str, hub: Path | None = None, head: str = "attention") -> None:
        """The head `head` of `heads` (`probe_train`'s heads.pt) and DINOv2-S."""
        saved = torch.load(heads, map_location=device)
        self.scale = saved["scale"].to(device)
        self.head = HEADS[head]().to(device).eval()
        self.head.load_state_dict(saved["heads"][head])
        self.dino = dinov2_small(device, hub)

    @torch.no_grad()
    def __call__(self, rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        """Center (N, 3), axis (N, 3), size (N, 2) in the pinch frame (m), in_view (N,), from frames (N, H, W, 3)."""
        return decode(self.head(*tokens(self.dino, rgb)), self.scale)


def term_slices(env: Any, group: str = "actor") -> dict[str, slice]:
    """Each observation term's columns in `group`'s concatenated observation."""
    names = env.observation_manager.active_terms[group]
    dims = env.observation_manager.group_obs_term_dim[group]
    out, at = {}, 0
    for name, dim in zip(names, dims, strict=True):
        width = int(np.prod(dim))
        out[name] = slice(at, at + width)
        at += width
    return out


def _to_root(pos: torch.Tensor, quat: torch.Tensor, center: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """(N, 6): a pinch-frame center and axis in the root frame, the pinch at `pos`, `quat`."""
    return torch.cat([pos + rotate(quat, center), rotate(quat, axis)], dim=1)


def _to_pinch(pos: torch.Tensor, quat: torch.Tensor, root: torch.Tensor) -> torch.Tensor:
    """(N, 6): a root-frame center and axis (N, 6) in the pinch frame."""
    return torch.cat([rotate_inverse(quat, root[:, :3] - pos), rotate_inverse(quat, root[:, 3:])], dim=1)


class Inputs:
    """One condition's object inputs, step by step (the module docstring)."""

    def __init__(
        self,
        condition: str,
        env: Any,
        probe: Probe | None,
        noise_m: float,
        matched: dict[str, list[float]] | None = None,
        columns: tuple[str, ...] = tuple(COLUMNS),
    ) -> None:
        """`noise_m`: the run's pose noise (`truth_path`); `matched`: the probe's error spread and correlation."""
        self.condition, self.env, self.probe, self.noise_m = condition, env, probe, noise_m
        self.cyl = obj.of(env)
        n, dev = env.num_envs, env.device
        self.root = torch.zeros(n, 6, device=dev)  # (the probe's conditions: the estimate kept, root frame)
        self.size = torch.zeros(n, 2, device=dev)
        self.have = torch.zeros(n, dtype=torch.bool, device=dev)
        if matched is not None:
            mask = torch.zeros(8, device=dev)  # (the noise only in `columns`)
            for name in columns:
                mask[COLUMNS[name]] = 1.0
            self.sigma = mask * torch.tensor(matched["sigma"], device=dev)  # (8,): center 3, axis 3, size 2
            self.rho = torch.tensor(matched["rho"], device=dev)
            self.noise = self.sigma * torch.randn(n, 8, device=dev)
        self.last: dict[str, torch.Tensor] | None = None  # the probe's raw estimate this step

    def __call__(self, fresh: torch.Tensor) -> dict[str, torch.Tensor] | None:
        """The object inputs for this step (None: leave the observation as it is); `fresh`: worlds just reset."""
        env, cyl, c = self.env, self.cyl, self.condition
        if c == "truth":
            return None
        pos, quat = task.pinch_pose(env)
        if c == "truth_path":
            g = torch.tensor([self.noise_m] * 3 + [self.noise_m / 10] * 3, device=env.device)
            return {
                "object_in_root": cyl.in_root(env) + g * torch.randn_like(g.expand(env.num_envs, 6)),
                "object_in_pinch": cyl.in_pinch(env) + g * torch.randn_like(g.expand(env.num_envs, 6)),
                "object_size": cyl.size(env, estimated=True),
            }
        if c == "matched_noise":
            self.noise[fresh] = self.sigma * torch.randn(int(fresh.sum()), 8, device=env.device)
            step = self.sigma * torch.sqrt(1 - self.rho**2) * torch.randn_like(self.noise)
            self.noise = torch.where(fresh[:, None], self.noise, self.rho * self.noise + step)
            truth = cyl.in_pinch(env)
            center = truth[:, :3] + self.noise[:, :3]
            axis = torch.nn.functional.normalize(truth[:, 3:] + self.noise[:, 3:6], dim=1)
            size = cyl.size(env) + self.noise[:, 6:]
            return {
                "object_in_root": _to_root(pos, quat, center, axis),
                "object_in_pinch": torch.cat([center, axis], dim=1),
                "object_size": size,
            }
        assert self.probe is not None
        est = self.probe(env.scene.sensors[SENSOR].data.rgb)
        self.last = est
        seen = est["in_view"]
        new_root = _to_root(pos, quat, est["center"], est["axis"])
        if c == "probe":
            take = seen | ~self.have | fresh
        elif c == "probe_frozen":
            take = fresh | ~self.have
        elif c == "probe_smoothed":
            take = seen | ~self.have | fresh
            blend = ~fresh & self.have
            mixed = SMOOTH * new_root + (1 - SMOOTH) * self.root
            mixed[:, 3:] = torch.nn.functional.normalize(mixed[:, 3:], dim=1)
            new_root = torch.where(blend[:, None], mixed, new_root)
            est = dict(
                est,
                size=torch.where(
                    blend[:, None], SMOOTH * est["size"] + (1 - SMOOTH) * self.size, est["size"]
                ),
            )
        else:
            raise ValueError(f"no condition {c!r} (there are {CONDITIONS})")
        self.root = torch.where(take[:, None], new_root, self.root)
        self.size = torch.where(take[:, None], est["size"], self.size)
        self.have |= take
        return {
            "object_in_root": self.root.clone(),
            "object_in_pinch": _to_pinch(pos, quat, self.root),
            "object_size": self.size.clone(),
        }


def run(
    env: Any,
    wrapped: Any,
    policy: Any,
    inputs: Inputs,
    seed: int,
    steps: int | None = None,
) -> dict[str, Any]:
    """One episode in every world from fresh resets under `inputs`; the report (the module docstring)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    cyl = inputs.cyl
    cols = term_slices(env)
    missing = [
        t for t in OBJECT_TERMS[:2] if t not in cols
    ]  # (the size: only an actor of several sizes has it)
    if missing:
        raise ValueError(f"the actor has no {missing} to replace (it has {sorted(cols)})")
    obs, _ = wrapped.reset()
    term = pool_reset(env)
    if term is None:
        raise ValueError("the closed-loop test needs an environment whose starts are a pool's")
    which = np.array(term.pools_of_worlds())
    n, dev = env.num_envs, env.device
    ever = torch.zeros(n, dtype=torch.bool, device=dev)
    done = torch.zeros_like(ever)
    held_at_end = torch.zeros_like(ever)
    first_touch: list[torch.Tensor | None] = [None] * n
    touched = torch.zeros_like(ever)
    # a step: (the estimate's error, pinch frame (N, 8); counted (N,): episode running, not just reset; held (N,))
    errors: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    fresh = torch.ones(n, dtype=torch.bool, device=dev)
    holding = task.held(env, cyl) > 0
    with torch.inference_mode():
        for _ in range(steps or int(env.max_episode_length)):
            new = inputs(fresh)
            if new is not None:
                actor = obs["actor"].clone()
                for name, value in new.items():
                    if name in cols:
                        actor[:, cols[name]] = value
                obs["actor"] = actor
            if inputs.last is not None:
                truth = torch.cat([cyl.in_pinch(env), cyl.size(env)], dim=1)
                est = torch.cat([inputs.last["center"], inputs.last["axis"], inputs.last["size"]], dim=1)
                errors.append(((est - truth).cpu().numpy(), (~done).cpu().numpy(), holding.cpu().numpy()))
                both = (task.both_fingers(env) > 0) & ~touched & ~done
                for i in torch.nonzero(both).flatten().tolist():
                    first_touch[i] = (est - truth)[i].cpu()
                touched |= both
            before = holding
            obs, _, dones, _ = wrapped.step(policy(obs))
            holding = task.held(env, cyl) > 0
            ended = dones.bool() & ~done
            held_at_end |= ended & before
            ever |= (holding & ~done & ~ended) | (ended & before)
            done |= ended
            fresh = dones.bool()
        held_at_end |= ~done & holding
    out: dict[str, Any] = {"worlds": n}
    for name in ("all", *POOLS):
        mask = np.ones(n, bool) if name == "all" else which == name
        if mask.any():
            m = torch.as_tensor(mask, device=dev)
            out[name] = {
                "n": int(mask.sum()),
                "held_at_end": float(held_at_end[m].float().mean()),
                "held_ever": float(ever[m].float().mean()),
            }
    end = held_at_end.cpu().numpy()
    ev = ever.cpu().numpy()
    out["ended"] = {
        "held": float(end.mean()),
        "held_then_lost": float((ev & ~end).mean()),
        "never_held": float((~ev).mean()),
    }
    if errors:
        out["probe_error"] = _error_report(errors, first_touch, end)
    return out


AXES = ("x", "y", "z")


def _error_report(
    errors: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    first_touch: list[torch.Tensor | None],
    end: np.ndarray,
) -> dict[str, Any]:
    """The estimate's error along the episodes and at first touch; the columns' spread and correlation.

    The error in the pinch frame (mm), held and not; each column's spread and step-to-step correlation (center 3, axis
    3, size 2: for `matched_noise`).
    """
    e = np.stack([x for x, _, _ in errors])  # (T, N, 8)
    ok = np.stack([v for _, v, _ in errors])  # (T, N)
    h = np.stack([y for _, _, y in errors]).astype(bool)
    flat, flat_held = e[ok], h[ok]
    rep: dict[str, Any] = {}
    for label, m in (("not_held", ~flat_held), ("held", flat_held)):
        if not m.any():
            continue
        rep[label] = {
            f"center_{a}_mm": {
                "median_abs": float(np.median(np.abs(1000 * flat[m, i]))),
                "bias": float(1000 * flat[m, i].mean()),
            }
            for i, a in enumerate(AXES)
        }
        rep[label]["radius_mm_median_abs"] = float(np.median(np.abs(1000 * flat[m, 6])))
        rep[label]["height_mm_median_abs"] = float(np.median(np.abs(1000 * flat[m, 7])))
    pair = ok[:-1] & ok[1:]  # (the same episode at consecutive steps: a world counts until it ends)
    a, b = e[:-1][pair], e[1:][pair]
    rho = [float(np.corrcoef(a[:, k], b[:, k])[0, 1]) for k in range(8)]
    rep["matched"] = {
        "sigma": [float(s) for s in flat.std(axis=0)],
        "rho": [float(np.clip(r, 0.0, 0.999)) for r in rho],
    }
    touch = [(t.numpy(), bool(end[i])) for i, t in enumerate(first_touch) if t is not None]
    if touch:
        ft = np.array([t for t, _ in touch])
        held = np.array([k for _, k in touch])
        rep["at_first_touch"] = {
            label: {
                "n": int(m.sum()),
                **{
                    f"center_{ax}_mm_median_abs": float(np.median(np.abs(1000 * ft[m, i])))
                    for i, ax in enumerate(AXES)
                },
            }
            for label, m in (("held_at_end", held), ("not_held_at_end", ~held))
            if m.any()
        }
    return rep


def main(argv: list[str] | None = None) -> None:
    """The command line (the module docstring)."""
    from mjlab.rl import MjlabOnPolicyRunner

    from .train import runner_cfg

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("train_dir", type=Path)
    parser.add_argument("--heads", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", default="model_final.pt")
    parser.add_argument("--worlds", type=int, default=4096)
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS)
    parser.add_argument("--size", type=int, nargs=2, default=[224, 168], metavar=("W", "H"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hub", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=None, help="cap the episodes (a check; default: whole)")
    parser.add_argument(
        "--matched-from",
        type=Path,
        default=None,
        help="a probe_loop.json whose `probe` run's error statistics to use",
    )
    parser.add_argument(
        "--matched-sets",
        nargs="+",
        default=[],
        metavar="COL,COL",
        help=f"also `matched_noise` on these columns only, one run a set ({', '.join(COLUMNS)})",
    )
    args = parser.parse_args(argv)
    settings = json.loads((args.train_dir / "settings.json").read_text())
    env, wrapped = make_video_env(
        settings, args.worlds, args.device, args.pool, sensors=(camera_cfg(*args.size),)
    )
    runner = MjlabOnPolicyRunner(wrapped, asdict(runner_cfg(1, 0, "tensorboard", 1, 24)), None, args.device)
    runner.load(str(args.train_dir / args.checkpoint), map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)
    given = (
        json.loads(args.matched_from.read_text())["probe"]["probe_error"]["matched"]
        if args.matched_from
        else None
    )
    needs_probe = any(c.startswith("probe") for c in args.conditions) or (
        given is None and ("matched_noise" in args.conditions or args.matched_sets)
    )
    probe = Probe(args.heads, args.device, args.hub) if needs_probe else None
    noise_m = float(settings.get("object_noise_m", 0.001))
    # (`matched_noise` takes the probe's error statistics from the `probe` condition: that runs first)
    order = sorted(args.conditions, key=lambda c: (c == "matched_noise", CONDITIONS.index(c)))
    if given is None and ("matched_noise" in order or args.matched_sets) and "probe" not in order:
        order.insert(0, "probe")
    sets = {f"matched_noise[{s}]": tuple(s.split(",")) for s in args.matched_sets}
    for cols in sets.values():
        unknown = set(cols) - set(COLUMNS)
        if unknown:
            parser.error(f"no columns {sorted(unknown)} (there are {list(COLUMNS)})")
    order += list(sets)
    results: dict[str, Any] = {"settings": vars(args) | {"train_dir": str(args.train_dir)}}
    args.out.mkdir(parents=True, exist_ok=True)
    for c in order:
        started = time.perf_counter()
        kind, cols = ("matched_noise", sets[c]) if c in sets else (c, tuple(COLUMNS))
        matched = None
        if kind == "matched_noise":
            matched = given or results.get("probe", {}).get("probe_error", {}).get("matched")
        r = run(env, wrapped, policy, Inputs(kind, env, probe, noise_m, matched, cols), args.seed, args.steps)
        r["seconds"] = time.perf_counter() - started
        results[c] = r
        print(
            f"{c}: held at the end {100 * r['all']['held_at_end']:.1f} % (held ever {100 * r['all']['held_ever']:.1f}); "
            + ", ".join(f"{p} {100 * r[p]['held_at_end']:.1f}" for p in POOLS if p in r)
            + f" ({r['seconds']:.0f} s)",
            flush=True,
        )
        (args.out / "probe_loop.json").write_text(json.dumps(results, indent=1, default=str))


if __name__ == "__main__":
    main()
