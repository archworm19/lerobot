"""The bank's starts precomputed for the mjlab environment: drawn on CPU MuJoCo, written into worlds at a reset.

  uv run --extra felix_jr_sim python -m lerobot.felix_jr.sim2real.start_pool <bank.jsonl> <out.npz>
      [--per-pool attack=1024 at_object=1024 in_jaws=1024 lifted=1024] [--seed 0] [--workers 6]
      [--holdout 0.15 [--test-per-pool 1024]]   # a held-out split by placement: <out>_train.npz, <out>_test.npz
      [--split random|ahead|side|diameter|height] [--holdout-at high|low|middle]
      [--plot [PNG]] [--plot-only]   # the placements drawn (`placement_plot`): where, what size, which held out

The held-out split is by placement (a placement's starts are near copies of each other: a split by start would leak):
at random (`--split random`: placements among the trained ones, an interpolation test), or a region of the placements
by one of their coordinates (`SPLIT_KEYS`: how far ahead of the arm's base, how far to its side, the object's diameter,
its height), its `--holdout` share at the high end, the low end or the middle of it (`split_placements`): the ends an
extrapolation test, the middle a band never trained on with trained ones either side.

A start (`starts.StartSampler`: an anchor of the bank, one of its pools, the arm arrived as the real one does, settled,
checked) is a whole simulator state; drawing one is a short simulation in itself (the attack pool's arrival), too slow
for a reset. So they are drawn here, ahead (OmniReset's precomputed resets, `docs/sim_deploy.md`), and saved as each
joint's position and velocity and each actuator's command, by name: the environment's model lays them out its own way
(mjlab's prefixes, its order), and `StartPool.layout` maps them by name.

The pool's file is `StartPool`'s (`save`, `load`: an .npz, `FORMAT` its version, checked when read); the bank's is
`bank.py`'s (.jsonl). This module needs no mjlab: drawing runs on CPU MuJoCo. In the environment, `pool_reset.PoolReset`
(an mjlab reset event term) reads a pool and writes its starts into the worlds.
"""

import argparse
import multiprocessing
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel

if TYPE_CHECKING:
    import mujoco

POOLS = ("attack", "at_object", "in_jaws", "lifted")
# the pool file's version: 1 one object's size (`object_size`, older pools, read still), 2 a size a start (`objects`)
FORMAT = 2
SPLIT_KEYS = ("random", "ahead", "side", "diameter", "height")
FREE_JOINT = (
    "floating_base_joint"  # (the object's: unnamed in `scene`, named by mjlab; `mjlab_env._joint_slices`)
)


def _layout(m: "mujoco.MjModel") -> dict[str, Any]:
    """`m`'s joints (name, qpos address and width, dof address and width) and actuators, by name."""
    import mujoco

    qwidth = {mujoco.mjtJoint.mjJNT_FREE: 7, mujoco.mjtJoint.mjJNT_BALL: 4}
    vwidth = {mujoco.mjtJoint.mjJNT_FREE: 6, mujoco.mjtJoint.mjJNT_BALL: 3}
    joints = []
    for j in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        name = name.split("/", 1)[-1] if name else FREE_JOINT
        kind = mujoco.mjtJoint(int(m.jnt_type[j]))
        joints.append(
            (name, int(m.jnt_qposadr[j]), qwidth.get(kind, 1), int(m.jnt_dofadr[j]), vwidth.get(kind, 1))
        )
    actuators = [
        (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or f"#{a}").split("/", 1)[-1]
        for a in range(m.nu)
    ]
    return {"joints": joints, "actuators": actuators}


@dataclass
class StartPool:
    """Starts, by name: per joint its position (`qpos[name]`, (N, width)) and velocity, per actuator its command.

    And each start's pool and anchor (the bank's), its object's size (`objects`, (N, 3): radius, height (m), mass (kg);
    one size given, every start's), the table's height.
    """

    qpos: dict[str, np.ndarray]
    qvel: dict[str, np.ndarray]
    ctrl: dict[str, np.ndarray]
    pool: np.ndarray
    anchor: np.ndarray
    objects: np.ndarray  # (N, 3): radius, height (m), mass (kg); a (3,) one for every start
    table_z: float
    placements: np.ndarray | None = (
        None  # the bank's placements drawn from (a split's: `--holdout`), None: all
    )

    @property
    def sizes_vary(self) -> bool:
        """The starts' objects not all one size."""
        return bool(len(self) and (self.objects != self.objects[0]).any())

    @property
    def object_size(self) -> tuple[float, float, float]:
        """The one object's radius, height (m) and mass (kg); a pool whose sizes vary has none (`objects`)."""
        if self.sizes_vary:
            raise ValueError("the starts' objects vary in size: see `objects`")
        r, h, m = self.objects[0]
        return float(r), float(h), float(m)

    def __len__(self) -> int:
        """How many starts."""
        return len(self.pool)

    def save(self, path: Path) -> None:
        """As an .npz (each array under "qpos/<joint>", "qvel/<joint>", "ctrl/<actuator>")."""
        arrays = {f"qpos/{k}": v for k, v in self.qpos.items()}
        arrays |= {f"qvel/{k}": v for k, v in self.qvel.items()}
        arrays |= {f"ctrl/{k}": v for k, v in self.ctrl.items()}
        np.savez_compressed(
            path,
            format=np.array(FORMAT),
            pool=self.pool,
            anchor=self.anchor,
            objects=self.objects,
            table_z=np.array(self.table_z),
            placements=np.array([] if self.placements is None else self.placements, dtype=int),
            **arrays,
        )

    @classmethod
    def load(cls, path: Path) -> "StartPool":
        """`save`'s file read back (checked: its version, every start with each of its arrays)."""
        z = np.load(path)
        version = int(z["format"]) if "format" in z.files else 1
        if version > FORMAT:
            raise ValueError(f"{path}: pool format {version}, newer than this code reads ({FORMAT})")
        part = {
            kind: {k.split("/", 1)[1]: z[k] for k in z.files if k.startswith(f"{kind}/")}
            for kind in ("qpos", "qvel", "ctrl")
        }
        objects = z["objects"] if "objects" in z.files else z["object_size"]  # (older pools: one size)
        return cls(
            part["qpos"],
            part["qvel"],
            part["ctrl"],
            z["pool"],
            z["anchor"],
            objects,
            float(z["table_z"]),
            z["placements"] if "placements" in z.files and len(z["placements"]) else None,
        )

    def __post_init__(self) -> None:
        """`objects` one row a start (one size given: broadcast to every start); every array one row a start."""
        n = len(self.pool)
        self.objects = np.broadcast_to(np.asarray(self.objects, dtype=np.float64), (n, 3)).copy()
        lengths = {
            f"{kind}/{name}": len(v)
            for kind in ("qpos", "qvel", "ctrl")
            for name, v in getattr(self, kind).items()
        }
        wrong = {k: v for k, v in (lengths | {"anchor": len(self.anchor)}).items() if v != n}
        if wrong:
            raise ValueError(f"a pool of {n} starts with arrays of other lengths: {wrong}")
        unknown = set(np.unique(self.pool).tolist()) - set(POOLS)
        if unknown:
            raise ValueError(f"unknown start pools {sorted(unknown)} (known: {POOLS})")

    def layout(self, m: "mujoco.MjModel") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Every start as `m` lays out its state: (N, nq), (N, nv), (N, nu); what the pool lacks at `m`'s defaults."""
        lay = _layout(m)
        n = len(self)
        qpos = np.tile(np.asarray(m.qpos0, dtype=np.float64), (n, 1))
        qvel = np.zeros((n, m.nv))
        ctrl = np.zeros((n, m.nu))
        for name, qadr, qw, vadr, vw in lay["joints"]:
            if name in self.qpos:
                qpos[:, qadr : qadr + qw] = self.qpos[name]
                qvel[:, vadr : vadr + vw] = self.qvel[name]
        for a, name in enumerate(lay["actuators"]):
            if name in self.ctrl:
                ctrl[:, a] = self.ctrl[name][:, 0]
        return qpos, qvel, ctrl


# ---- drawing them (CPU) ---------------------------------------------------------------------------------------------

_sampler: Any = None


def _init(model_path: str, bank_path: str, placements: "set[int] | None" = None) -> None:
    global _sampler
    from .starts import StartSampler

    _sampler = StartSampler(XleRobotModel.load(Path(model_path)), Path(bank_path), placements=placements)


def _draw(task: tuple[str, int]) -> dict[str, Any]:
    pool, seed = task
    start = _sampler.sample(np.random.default_rng(seed), pool)
    m, d = start.scene.m, start.scene.d
    lay = _layout(m)
    line = _sampler.anchors[start.anchor]["placement"]
    return {
        "pool": pool,
        "anchor": start.anchor,
        "qpos": {name: d.qpos[q : q + qw].copy() for name, q, qw, _, _ in lay["joints"]},
        "qvel": {name: d.qvel[v : v + vw].copy() for name, _, _, v, vw in lay["joints"]},
        "ctrl": {name: d.ctrl[a : a + 1].copy() for a, name in enumerate(lay["actuators"])},
        "object": (line["radius"], line["height"], line["mass_kg"]),
        "table_z": line["bottom_root"][2],
    }


def draw_pool(
    model_path: Path,
    bank_path: Path,
    per_pool: Mapping[str, int],
    seed: int = 0,
    workers: int = 1,
    placements: "set[int] | None" = None,
) -> StartPool:
    """`per_pool` starts of each pool from the bank's anchors (`placements`' only, if given), on `workers` processes."""
    tasks = [(pool, seed * 1_000_003 + i) for pool, n in per_pool.items() for i in range(n)]
    if workers > 1:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(
            workers, initializer=_init, initargs=(str(model_path), str(bank_path), placements)
        ) as pool:
            rows = pool.map(_draw, tasks, chunksize=8)
    else:
        _init(str(model_path), str(bank_path), placements)
        rows = [_draw(t) for t in tasks]
    stack = {
        kind: {name: np.stack([r[kind][name] for r in rows]) for name in rows[0][kind]}
        for kind in ("qpos", "qvel", "ctrl")
    }
    return StartPool(
        stack["qpos"],
        stack["qvel"],
        stack["ctrl"],
        np.array([r["pool"] for r in rows]),
        np.array([r["anchor"] for r in rows]),
        np.array([r["object"] for r in rows]),
        float(np.median([r["table_z"] for r in rows])),
        None if placements is None else np.array(sorted(placements)),
    )


def placement_coordinate(placement: Mapping[str, Any], base_xy: Sequence[float], key: str) -> float:
    """A bank placement's `key` (`SPLIT_KEYS`, not random; m).

    Ahead of the arm's base (along the robot's facing, -x), to its side (+y), the object's diameter or height.
    """
    x, y = placement["bottom_root"][:2]
    if key == "ahead":
        return float(base_xy[0] - x)
    if key == "side":
        return float(y - base_xy[1])
    if key == "diameter":
        return 2.0 * float(placement["radius"])
    if key == "height":
        return float(placement["height"])
    raise ValueError(f"unknown split {key!r} (known: {SPLIT_KEYS})")


def split_placements(
    placements: Sequence[Mapping[str, Any]],
    share: float,
    key: str = "random",
    at: str = "high",
    base_xy: Sequence[float] = (0.0, 0.0),
    seed: int = 0,
) -> tuple[set[int], set[int]]:
    """The bank placements (`placement` records) split by index: (train, held out).

    `share` of them held out, at random or at the `at` end (high, low, or the middle) of `key`
    (`placement_coordinate`; `base_xy` the arm's base, root frame).
    """
    index = np.array([int(p["index"]) for p in placements])
    n_test = max(1, int(round(share * len(index))))
    if key == "random":
        test = set(np.random.default_rng(seed).permutation(index)[:n_test].tolist())
    else:
        value = np.array([placement_coordinate(p, base_xy, key) for p in placements])
        order = np.argsort(value, kind="stable")
        if at == "high":
            picked = order[::-1][:n_test]
        elif at == "low":
            picked = order[:n_test]
        elif at == "middle":
            start = (len(order) - n_test) // 2
            picked = order[start : start + n_test]
        else:
            raise ValueError(f"unknown end {at!r} (high, low, middle)")
        test = set(index[picked].tolist())
    return set(index.tolist()) - test, test


def main(argv: Sequence[str] | None = None) -> None:
    """Command line entry point; see the module docstring."""
    from lerobot.felix_jr.harness import api

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bank", type=Path)
    parser.add_argument("out", type=Path, help="the pool (.npz; under a scratch/ folder)")
    parser.add_argument("--per-pool", nargs="+", default=[f"{p}=1024" for p in POOLS], metavar="POOL=N")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--model", type=Path, default=api.MODEL)
    parser.add_argument(
        "--holdout",
        type=float,
        default=0.0,
        help="this share of the bank's anchored placements held out: <out>_train.npz and <out>_test.npz",
    )
    parser.add_argument("--test-per-pool", type=int, default=1024, help="the held-out pool's starts a pool")
    parser.add_argument(
        "--split", choices=SPLIT_KEYS, default="random", help="held out at random or by a region"
    )
    parser.add_argument("--holdout-at", choices=("high", "low", "middle"), default="high")
    parser.add_argument(
        "--plot",
        type=Path,
        nargs="?",
        const=Path(""),
        default=None,
        help="the placements drawn (where, what size, which held out) into this png (default: <out>_placements.png)",
    )
    parser.add_argument(
        "--plot-only", action="store_true", help="only the plot (a split previewed): no pools drawn"
    )
    args = parser.parse_args(argv)
    per_pool = {k: int(v) for k, v in (item.split("=") for item in args.per_pool)}
    unknown = set(per_pool) - set(POOLS)
    if unknown:
        parser.error(f"unknown pools {sorted(unknown)} (known: {POOLS})")
    started = time.perf_counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    from . import bank

    config, lines = bank.load(args.bank)
    base = XleRobotModel.load(args.model)
    base_xy = base.frame_pose(f"{config.arm}_base_link", base.joint_vector())[:2, 3]
    test: set[int] | None = None
    jobs: list[tuple[Path, dict[str, int], set[int] | None]]
    if args.holdout <= 0:
        jobs = [(args.out, per_pool, None)]
    else:  # (split by placement: a placement's starts are near copies of each other, a split by start would leak)
        anchored = sorted(
            (line["placement"] for line in lines if line["anchor"] is not None), key=lambda p: p["index"]
        )
        train, test = split_placements(
            anchored, args.holdout, args.split, args.holdout_at, base_xy, args.seed
        )
        if args.split != "random":
            values = [placement_coordinate(p, base_xy, args.split) for p in anchored if p["index"] in test]
            print(
                f"held out: {len(test)} of {len(anchored)} placements, {args.split} "
                f"{1000 * min(values):.0f}-{1000 * max(values):.0f} mm ({args.holdout_at})",
                flush=True,
            )
        stem = args.out.with_suffix("")
        jobs = [
            (stem.with_name(stem.name + "_train.npz"), per_pool, train),
            (stem.with_name(stem.name + "_test.npz"), dict.fromkeys(per_pool, args.test_per_pool), test),
        ]
    if args.plot is not None or args.plot_only:
        from .placement_plot import plot_placements

        stem = args.out.with_suffix("")
        png = (
            args.plot if args.plot not in (None, Path("")) else stem.with_name(stem.name + "_placements.png")
        )
        split = (
            "no held-out split"
            if test is None
            else f"held out: {args.split} {args.holdout_at if args.split != 'random' else ''} {args.holdout:.0%}"
        )
        print(f"placements drawn: {plot_placements(lines, base_xy, test, png, f'{args.bank.stem}, {split}')}")
        if args.plot_only:
            return
    for out, counts_wanted, placements in jobs:
        pool = draw_pool(args.model, args.bank, counts_wanted, args.seed, args.workers, placements)
        pool.save(out)
        counts = {p: int((pool.pool == p).sum()) for p in POOLS}
        which = "all" if placements is None else len(placements)
        print(
            f"{len(pool)} starts {counts} from {len(set(pool.anchor.tolist()))} anchors ({which} placements) in "
            f"{time.perf_counter() - started:.0f} s: {out}",
            flush=True,
        )


if __name__ == "__main__":
    main()
