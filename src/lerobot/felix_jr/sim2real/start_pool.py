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

In the environment, `reset_from_pool` (an mjlab reset event) draws a start for each world being reset: a pool by
`pool_weights`, then a start of it at random; its state written, the object's height there kept
(`env.felix_object_z0`: what its rise is measured from).
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
    import torch
    from mjlab.envs import ManagerBasedRlEnv

POOLS = ("attack", "at_object", "in_jaws", "lifted")
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

    def __post_init__(self) -> None:
        """`objects` one row a start (one size given: broadcast to every start)."""
        self.objects = np.broadcast_to(np.asarray(self.objects, dtype=np.float64), (len(self.pool), 3)).copy()

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
            pool=self.pool,
            anchor=self.anchor,
            objects=self.objects,
            table_z=np.array(self.table_z),
            placements=np.array([] if self.placements is None else self.placements, dtype=int),
            **arrays,
        )

    @classmethod
    def load(cls, path: Path) -> "StartPool":
        """`save`'s file read back."""
        z = np.load(path)
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


# ---- in the environment ---------------------------------------------------------------------------------------------


def reset_from_pool(
    env: "ManagerBasedRlEnv",
    env_ids: "torch.Tensor | None",
    path: str,
    pool_weights: Mapping[str, float],
) -> "torch.Tensor | None":
    """Mjlab reset event: each world in `env_ids` given a start (a pool by `pool_weights`, then a start of it).

    Each world's object size kept (`env.felix_object_size`, (num_envs, 3): radius, height, mass), and its resting
    height (`env.felix_object_rest_z`); the object itself resized by `resize_objects` (`mjlab_env`'s sized reset). The
    rows drawn (None: no world reset).
    """
    import torch

    sim = env.sim
    cache = getattr(env, "_felix_pool", None)
    if cache is None or cache["path"] != path:
        pool = StartPool.load(Path(path))
        qpos, qvel, ctrl = pool.layout(sim.mj_model)
        dev = sim.data.qpos.device
        by_pool = [np.flatnonzero(pool.pool == p) for p in POOLS]
        weights = np.array(
            [pool_weights.get(p, 0.0) if len(by_pool[i]) else 0.0 for i, p in enumerate(POOLS)]
        )
        if weights.sum() <= 0:
            raise ValueError(f"no weighted pool has starts: {dict(pool_weights)}")
        cache = {
            "path": path,
            "qpos": torch.as_tensor(qpos, device=dev, dtype=sim.data.qpos.dtype),
            "qvel": torch.as_tensor(qvel, device=dev, dtype=sim.data.qvel.dtype),
            "ctrl": torch.as_tensor(ctrl, device=dev, dtype=sim.data.ctrl.dtype),
            "rows": [torch.as_tensor(r, device=dev) for r in by_pool],
            "weights": torch.as_tensor(weights / weights.sum(), device=dev, dtype=torch.float32),
            "object_z": _object_z_index(sim.mj_model),
            # mjlab derives the controls from the robot's joint position targets (each step, and at a reset): the
            # start's commands go in as targets, by name (each actuator drives the joint it is named after)
            "targets": _targets(env, sim.mj_model),
            "objects": torch.as_tensor(pool.objects, device=dev, dtype=torch.float32),
            "table_z": float(pool.table_z),
        }
        env._felix_pool = cache
        env.felix_object_z0 = torch.zeros(env.num_envs, device=dev)
        # resting on the table: what the object's rise is measured from (`mjlab_task.rise`)
        env.felix_object_rest_z = torch.full(
            (env.num_envs,), pool.table_z + float(pool.objects[0, 1]) / 2, device=dev
        )
        env.felix_object_size = cache["objects"][0].repeat(env.num_envs, 1)
    ids = torch.arange(env.num_envs, device=sim.data.qpos.device) if env_ids is None else env_ids
    if len(ids) == 0:
        return None
    which = torch.multinomial(cache["weights"], len(ids), replacement=True)
    rows = torch.empty(len(ids), dtype=torch.long, device=ids.device)
    for i, members in enumerate(cache["rows"]):
        mask = which == i
        if mask.any():
            rows[mask] = members[torch.randint(len(members), (int(mask.sum()),), device=ids.device)]
    sim.data.qpos[ids] = cache["qpos"][rows]
    sim.data.qvel[ids] = cache["qvel"][rows]
    sim.data.ctrl[ids] = cache["ctrl"][rows]
    columns, joint_ids, entity = cache["targets"]
    env.scene[entity].set_joint_position_target(
        cache["ctrl"][rows][:, columns], joint_ids=joint_ids, env_ids=ids
    )
    env.felix_object_z0[ids] = cache["qpos"][rows, cache["object_z"]].float()
    size = cache["objects"][rows]
    env.felix_object_size[ids] = size
    env.felix_object_rest_z[ids] = cache["table_z"] + size[:, 1] / 2
    return rows


def resize_objects(
    env: "ManagerBasedRlEnv",
    env_ids: "torch.Tensor",
    sizes: "torch.Tensor",
    mass_scale: "torch.Tensor | None" = None,
) -> None:
    """The object (a cylinder) in each world of `env_ids` made `sizes` ((n, 3): radius, height (m), mass (kg)).

    Its geom's size, bounding sphere and box (what the collision broadphase reads), its body's mass (times
    `mass_scale`, if given) and inertia (a solid cylinder's). The model fields must be per world (`mjlab_env`'s sized
    reset asks for them) and the constants recomputed after (mjlab does, for a reset event that declares it).
    """
    import torch

    m = env.sim.model
    cache = env._felix_pool
    if "object_geom" not in cache:
        cache["object_body"], cache["object_geom"] = _object_ids(env.sim.mj_model)
    body, geom = cache["object_body"], cache["object_geom"]
    r, h, mass = sizes[:, 0], sizes[:, 1], sizes[:, 2]
    if mass_scale is not None:
        mass = mass * mass_scale
    half = h / 2
    m.geom_size[env_ids, geom, 0] = r.to(m.geom_size.dtype)
    m.geom_size[env_ids, geom, 1] = half.to(m.geom_size.dtype)
    m.geom_rbound[env_ids, geom] = torch.sqrt(r * r + half * half).to(m.geom_rbound.dtype)
    m.geom_aabb[env_ids, geom, 1] = torch.stack([r, r, half], dim=1).to(m.geom_aabb.dtype)
    m.body_mass[env_ids, body] = mass.to(m.body_mass.dtype)
    side = mass * (3 * r * r + h * h) / 12
    m.body_inertia[env_ids, body] = torch.stack([side, side, mass * r * r / 2], dim=1).to(
        m.body_inertia.dtype
    )


def _object_ids(m: "mujoco.MjModel") -> tuple[int, int]:
    """The object's body (its free joint's) and its one geom, a cylinder."""
    import mujoco

    joints = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    if len(joints) != 1:
        raise ValueError(f"expected one free joint (the object), found {len(joints)}")
    body = int(m.jnt_bodyid[joints[0]])
    geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == body]
    if len(geoms) != 1 or m.geom_type[geoms[0]] != mujoco.mjtGeom.mjGEOM_CYLINDER:
        raise ValueError(f"expected the object's body to have one cylinder, found geoms {geoms}")
    return body, geoms[0]


def _targets(env: "ManagerBasedRlEnv", m: "mujoco.MjModel", entity: str = "robot") -> tuple[Any, Any, str]:
    """The robot's actuated joints: the ctrl columns of their actuators and their indices in the entity's joints."""
    import torch

    names = [n.split("/", 1)[-1] for n in env.scene[entity].joint_names]
    actuators = _layout(m)["actuators"]
    pairs = [(a, names.index(n)) for a, n in enumerate(actuators) if n in names]
    dev = env.sim.data.qpos.device
    return (
        torch.as_tensor([a for a, _ in pairs], device=dev),
        torch.as_tensor([j for _, j in pairs], device=dev),
        entity,
    )


def _object_z_index(m: "mujoco.MjModel") -> int:
    """The qpos index of the object's height (its free joint's z)."""
    lay = _layout(m)
    found = [q for name, q, qw, _, _ in lay["joints"] if name == FREE_JOINT and qw == 7]
    if len(found) != 1:
        raise ValueError(f"expected one free joint (the object), found {len(found)}")
    return found[0] + 2


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
