"""A grasp refined in the simulator: robust samples around it, the first that holds taken (`docs/sim_harness.md`).

The grasp is parameterized in the object's terms: its primitive's family (`Primitive.grasp_space`, `grasp_at`: a
cylinder's `t`, `phi`, `psi`) and the gripper's own, `depth` (the object's center back from the fingertips:
`GraspSettings.depth`) and `offset` (the grasp's center moved along its closing line, m). Each sample runs the
harness's own chain (`grasping._evaluate_target`: the filters and the IK, the wrist camera's clearance at the attack
only, the grasp's collisions the simulator's; `plan_pick`; `prepare_execution`: the moves `execute` would send) and is
played in the simulator by the shadow (`shadow.Shadow`: the server's move code, the fitted servos, the camera's real
board) from the arm's reading, under `draws` draws of the world, the same for every sample: the object moved by the
perception's error (`object_sigma_m`, on the table; the first draw the estimate itself), the physics drawn as the
bank's (`grasp_check.physics_draws`). Its score: the share of draws the object was lifted (`lifted`: up
`HELD_FRACTION` of the planned lift, held `HOLD_S`), plus a tenth of how far it rose on average.

**The search is greedy, as the harness's grasp choice is** (`grasping.first_feasible`): a batch of `batch` samples
(the start itself first, the rest drawn around it, `sigma` each), run in parallel (`workers` processes); the best
sample of a batch lifted in at least `accept` of the draws is taken. Else the next batch is drawn around the best so
far, its spread `widen` times wider, up to `max_batches`; then the best seen, its share said (`accepted` False). The
lifted share is how forgiving the grasp is: the robot's misses are the draws' offsets.

  uv run --extra felix_jr --extra felix_jr_sim python -m lerobot.felix_jr.sim2real.refine <bank.jsonl> <out.jsonl>
      [--starts greedy anchor] [--placements N] [--workers 6]

The command line: every anchored placement of a bank (`bank.py`) refined from the right arm's rest, from each start:
`greedy` (`first_feasible`'s first, at the fingertips: what the harness starts from) and `anchor` (the bank's grasp,
held when simulated from its exact pose); a table, and a line per run in `out.jsonl`.

From the head (`phase="head"`): the arm from where it is through the stage, the attack, the grasp, the lift; every
parameter free. From the wrist (`phase="wrist"`): the arm already at the attack, the approach's direction kept (`phi`,
`psi`: the arm committed to it). TODO: rewards beyond lifting (placing, a pose held).
"""

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any

import numpy as np

from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel
from lerobot.felix_jr.harness.execute import prepare_execution
from lerobot.felix_jr.harness.grasping import (
    LIFT_M,
    PICK_GRIP,
    GraspCandidate,
    GraspSettings,
    _default_obstacles,
    _evaluate_target,
    _Timer,
    plan_pick,
)
from lerobot.felix_jr.harness.objects import Execution
from lerobot.felix_jr.harness.primitives import GraspParameter, Primitive
from lerobot.felix_jr.harness.reach import select_poses

from .finger_models import settings as physics
from .grasp_check import physics_draws
from .scene import from_primitive
from .shadow import Shadow

HELD_FRACTION = 0.75  # the object up this much of the planned lift: lifted
HOLD_S = 2.0  # held this long after the lift
PHASES = ("head", "wrist")
WRIST_FIXED = ("phi", "psi")  # from the wrist, the approach's direction is the attack's
# the samples' spread about the start per parameter: a few mm, a few deg; `phi` narrow (the 5-joint arm's approach stays
# in its plane: 8 deg put 15 of 24 samples out of it)
SIGMA = {"t": 0.008, "phi": 2.0, "psi": 5.0, "depth": 0.006, "offset": 0.003}
DEPTH_PAST_RADIUS_M = 0.005  # the deepest tried: just past the object's radius (the bank's)


@dataclass(frozen=True)
class RefineSettings:
    """The search's settings, and the world's draws (the module docstring)."""

    batch: int = 6
    max_batches: int = 5
    draws: int = 8
    accept: float = 7 / 8  # the lifted share a sample needs to be taken
    widen: float = 1.5  # the spread of the next batch, after one that took nothing
    workers: int = 6  # processes (1: in this one)
    object_sigma_m: float = 0.007  # the object's position error on the table (each way; the head's ~1 cm)
    mass_kg: float = 0.08
    max_vel: float = 20.0  # the harness's `execute` speed
    sigma: Mapping[str, float] = field(default_factory=lambda: dict(SIGMA))
    table_z: float | None = None  # the table (None: the model's floor)
    seed: int = 0


@dataclass(frozen=True)
class Sample:
    """One sample: its parameters, its score (lifted share + a tenth of the rise), each draw's rise (m), or why not."""

    params: dict[str, float]
    score: float
    lifted: float  # the share of draws lifted
    rises: tuple[float, ...] = ()
    refused: str = ""  # why the chain refused it (score 0)


@dataclass(frozen=True, eq=False)
class RefinedPlan:
    """The sample taken (or the best seen), its grasp and its execution (what `execute` runs), and the search."""

    best: Sample
    candidate: GraspCandidate | None
    execution: Execution | None
    accepted: bool  # lifted in `accept` of the draws (else the best seen)
    batches: int
    samples: tuple[Sample, ...]
    seconds: float

    def describe(self) -> str:
        """One line: the grasp taken (or the best seen), its lifted share, the search's size."""
        p = ", ".join(f"{k} {_show(k, v)}" for k, v in self.best.params.items())
        verdict = "taken" if self.accepted else "none held enough; the best seen"
        return (
            f"{verdict}: {p}; lifted in {100 * self.best.lifted:.0f} % of draws (score {self.best.score:.2f}); "
            f"{len(self.samples)} samples in {self.batches} batches, {self.seconds:.0f} s"
        )


def _show(name: str, value: float) -> str:
    return f"{1000 * value:.1f} mm" if name in ("t", "depth", "offset") else f"{value:.1f} deg"


def space(
    primitive: Primitive, where: Mapping[str, tuple[float, float]] | None = None
) -> dict[str, GraspParameter]:
    """The parameters searched: the primitive's family's, then the gripper's `depth` and `offset`."""
    out = dict(primitive.grasp_space(where))
    radius = float(getattr(primitive, "radius", 0.02))
    out["depth"] = GraspParameter("depth", 0.0, radius + DEPTH_PAST_RADIUS_M, unit="m")
    out["offset"] = GraspParameter("offset", -radius / 2, radius / 2, unit="m")
    return out


def _clip(params: Mapping[str, float], bounds: Mapping[str, GraspParameter]) -> dict[str, float]:
    out = {}
    for name, value in params.items():
        q = bounds[name]
        if q.period is not None and q.high - q.low >= q.period - 1e-9:
            out[name] = float(value % q.period)  # (all the way round: wrapped, not clipped)
        else:
            out[name] = float(np.clip(value, q.low, q.high))
    return out


def plan_for(
    model: XleRobotModel,
    arm: str,
    primitive: Primitive,
    params: Mapping[str, float],
    fixed_side: int,
    grasp_settings: GraspSettings,
    start: Mapping[str, float],
    phase: str = "head",
) -> tuple[GraspCandidate, Execution] | str:
    """The harness's plan for `params` (the module docstring): the candidate and its execution, or why there is none."""
    family = {k: v for k, v in params.items() if k not in ("depth", "offset")}
    grasp = primitive.grasp_at(family)
    grasp = replace(grasp, center_root=grasp.center_root + params.get("offset", 0.0) * grasp.closing_root)
    settings = replace(grasp_settings, depth=float(params.get("depth", grasp_settings.depth)))
    capture: Any = SimpleNamespace(arms={arm: dict(start)}, head_ticks=None)
    found = _evaluate_target(
        model,
        primitive,
        grasp,
        arm,
        fixed_side,
        settings,
        capture,
        {},
        _Timer(),
        tuple(_default_obstacles(primitive)),
    )
    if not isinstance(found, GraspCandidate):
        return f"{found.stage}: {found.reason}"
    try:
        poses = plan_pick(model, found, capture)
        if phase == "wrist":
            poses = select_poses(poses, "pick.back", "pick", "lift")
        execution = prepare_execution(model, poses, PICK_GRIP)
    except ValueError as e:
        return f"the execution: {e}"
    return found, execution


MODES = ("lifted", "missed", "one jaw", "knocked over", "pushed away", "slipped")
KNOCKED_DEG = 30.0  # the object tilted this far: knocked over
PUSHED_M = 0.01  # the object moved this far on the table before the jaws closed: pushed away


@dataclass(frozen=True)
class Outcome:
    """How a rollout ended: how far the object rose (m), and how (`MODES`), with what the verdict rests on."""

    rise: float
    mode: str
    tilt_deg: float  # the object's axis from vertical, at the end
    pushed_m: float  # how far it moved on the table before the close
    jaws_touching: int  # the fingers whose pads touched the object when the jaws had closed (0, 1 or 2)
    # the jaw's `gripper.pos` past the object's contact then (not a miss's sign: the fingers give ~13 units)
    past_contact: float


def outcome(
    model: XleRobotModel,
    arm: str,
    primitive: Primitive,
    execution: Execution,
    start: Mapping[str, float],
    offset_m: np.ndarray,
    draw: Mapping[str, float],
    refine_settings: RefineSettings,
    on_tick: Callable[[Shadow, str], None] | None = None,
) -> Outcome:
    """`execution` played in the simulator with the object moved `offset_m` (x, y) under `draw`, and how it ended.

    Lifted: up `HELD_FRACTION` of the planned lift. Else, in order: knocked over (tilted past `KNOCKED_DEG`), missed
    (neither finger's pads touching it when the jaws had closed), pushed away (moved `PUSHED_M` on the table before
    the close), one jaw (only one finger touching it), slipped (both on it; it did not come up, or slid out in the
    hold). `on_tick(shadow, label)`: each tick (to film it).
    """
    s = refine_settings
    moved = from_primitive(primitive, mass=s.mass_kg * draw.get("mass_scale", 1.0))
    moved = replace(moved, x=moved.x + float(offset_m[0]), y=moved.y + float(offset_m[1]))
    with physics({k: v for k, v in draw.items() if k.isupper()}):
        shadow = Shadow(model, arm, start, s.table_z, [moved])
    if on_tick is not None:
        shadow.on_tick = lambda label: on_tick(shadow, label)
    start_xyz = shadow.object_position()
    pushed, past, touching = 0.0, float("nan"), 0
    contact = model.gripper_pos_for(
        arm, model.opening_for_width(arm, 2 * float(getattr(primitive, "radius", 0.0)))
    )
    for move in execution.moves:
        closing = "gripper close" in move.label
        if closing:
            pushed = float(np.linalg.norm(shadow.object_position()[:2] - start_xyz[:2]))
        replayed = shadow.move(move.label, move.targets, s.max_vel, 2 * s.max_vel, move.tol, move.timeout)
        if closing:
            past = contact - replayed.gripper_pos
            touching = _jaws_touching(shadow)
    shadow.hold(HOLD_S)
    rise = float(shadow.object_position()[2] - start_xyz[2])
    body = shadow.scene.m.body("object").id
    axis = shadow.scene.d.xmat[body].reshape(3, 3)[:, 2]
    tilt = float(np.degrees(np.arccos(np.clip(abs(axis[2]), 0.0, 1.0))))
    if rise > HELD_FRACTION * LIFT_M:
        mode = "lifted"
    elif tilt > KNOCKED_DEG:
        mode = "knocked over"
    elif touching == 0:
        mode = "missed"
    elif pushed > PUSHED_M:
        mode = "pushed away"
    elif touching == 1:
        mode = "one jaw"
    else:
        mode = "slipped"
    return Outcome(rise, mode, tilt, pushed, touching, past)


def _jaws_touching(shadow: Shadow) -> int:
    """How many fingers' pads (`robot.finger_pads`: `finger_fixed_pad<k>`, `finger_moving_pad<k>`) touch the object."""
    m, d = shadow.scene.m, shadow.scene.d
    body = m.body("object").id
    fingers = set()
    for c in d.contact[: d.ncon]:
        for mine, other in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
            name = m.geom(mine).name
            if m.geom_bodyid[other] == body and "_pad" in name:
                fingers.add(name.split("_pad")[0])
    return len(fingers)


def rollout(
    model: XleRobotModel,
    arm: str,
    primitive: Primitive,
    execution: Execution,
    start: Mapping[str, float],
    offset_m: np.ndarray,
    draw: Mapping[str, float],
    refine_settings: RefineSettings,
) -> float:
    """`execution` played in the simulator with the object moved `offset_m` (x, y) under `draw`: how far it rose (m)."""
    return outcome(model, arm, primitive, execution, start, offset_m, draw, refine_settings).rise


def evaluate(
    model: XleRobotModel,
    arm: str,
    primitive: Primitive,
    params: Mapping[str, float],
    fixed_side: int,
    grasp_settings: GraspSettings,
    start: Mapping[str, float],
    world: Sequence[tuple[np.ndarray, Mapping[str, float]]],
    refine_settings: RefineSettings,
    phase: str = "head",
) -> tuple[Sample, tuple[GraspCandidate, Execution] | None]:
    """One sample: its plan, played under each of `world`'s draws (object offset, physics): its score."""
    planned = plan_for(model, arm, primitive, params, fixed_side, grasp_settings, start, phase)
    if isinstance(planned, str):
        return Sample(dict(params), 0.0, 0.0, refused=planned), None
    rises = tuple(
        rollout(model, arm, primitive, planned[1], start, offset, draw, refine_settings)
        for offset, draw in world
    )
    lifted = float(np.mean([r > HELD_FRACTION * LIFT_M for r in rises]))
    rise = float(np.mean([min(max(r, 0.0), LIFT_M) for r in rises])) / LIFT_M
    return Sample(dict(params), lifted + 0.1 * rise, lifted, rises), planned


def draws(
    rng: np.random.Generator, n: int, object_sigma_m: float, estimate_first: bool = True
) -> list[tuple[np.ndarray, dict[str, float]]]:
    """`n` draws of the world: the object's offset on the table and the physics (`grasp_check.physics_draws`).

    The first offset is 0 (the estimate itself) unless not `estimate_first`.
    """
    return [
        (np.zeros(2) if i == 0 and estimate_first else rng.normal(0.0, object_sigma_m, 2), draw)
        for i, draw in enumerate(physics_draws(rng, n))
    ]


def validate(
    model: XleRobotModel,
    arm: str,
    primitive: Primitive,
    params: Mapping[str, float],
    fixed_side: int,
    start: Mapping[str, float],
    grasp_settings: GraspSettings,
    refine_settings: RefineSettings,
    n_draws: int = 32,
    seed: int = 1_000_003,
    phase: str = "head",
) -> Sample:
    """`params` scored on `n_draws` fresh draws, none the search saw: an unbiased lifted share.

    The draws are split over `refine_settings.workers` processes.
    """
    s = refine_settings
    g = replace(grasp_settings, camera_clearance_at_grasp=False)
    world = draws(np.random.default_rng(seed), n_draws, s.object_sigma_m, estimate_first=False)
    k = max(1, s.workers)
    args = [
        (model, arm, primitive, dict(params), fixed_side, g, dict(start), world[i::k], s, phase)
        for i in range(k)
        if world[i::k]
    ]
    if s.workers > 1:
        import multiprocessing

        with multiprocessing.get_context("spawn").Pool(s.workers) as pool:
            results = pool.starmap(evaluate, args)
    else:
        results = [evaluate(*a) for a in args]
    if results[0][0].refused:
        return results[0][0]
    rises = tuple(r for result in results for r in result[0].rises)
    lifted = float(np.mean([r > HELD_FRACTION * LIFT_M for r in rises]))
    rise = float(np.mean([min(max(r, 0.0), LIFT_M) for r in rises])) / LIFT_M
    return Sample(dict(params), lifted + 0.1 * rise, lifted, rises)


_WORKER: dict[str, Any] = {}


def _init_worker(job: dict[str, Any]) -> None:
    _WORKER.update(job)


def _evaluate_job(params: dict[str, float]) -> tuple[Sample, tuple[GraspCandidate, Execution] | None]:
    w = _WORKER
    return evaluate(
        w["model"],
        w["arm"],
        w["primitive"],
        params,
        w["fixed_side"],
        w["grasp_settings"],
        w["start"],
        w["world"],
        w["settings"],
        w["phase"],
    )


def refine(
    model: XleRobotModel,
    arm: str,
    primitive: Primitive,
    initial: Mapping[str, float],
    fixed_side: int,
    start: Mapping[str, float],
    grasp_settings: GraspSettings | None = None,
    refine_settings: RefineSettings | None = None,
    phase: str = "head",
    where: Mapping[str, tuple[float, float]] | None = None,
) -> RefinedPlan:
    """The grasp refined from `initial` (the module docstring): the first robust sample, or the best seen.

    `initial`: the family's parameters (a cylinder's `t`, `phi`, `psi`) and, if given, `depth` and `offset` (else 0):
    the harness's greedy candidate's (`first_feasible`). `start`: the arm's reading now. `fixed_side`: which contact
    the fixed jaw takes (the candidate's, kept). `grasp_settings`: the filters' (the camera's clearance at the attack
    only, whatever given).
    """
    if phase not in PHASES:
        raise ValueError(f"phase: one of {PHASES}, not {phase!r}")
    s = RefineSettings() if refine_settings is None else refine_settings
    g = replace(
        GraspSettings() if grasp_settings is None else grasp_settings, camera_clearance_at_grasp=False
    )
    rng = np.random.default_rng(s.seed)
    bounds = space(primitive, where)
    fixed = WRIST_FIXED if phase == "wrist" else ()
    center = _clip({name: float(initial.get(name, 0.0)) for name in bounds}, bounds)
    free = [name for name in bounds if name not in fixed]
    world = draws(rng, s.draws, s.object_sigma_m)
    job = {
        "model": model,
        "arm": arm,
        "primitive": primitive,
        "fixed_side": fixed_side,
        "grasp_settings": g,
        "start": dict(start),
        "world": world,
        "settings": s,
        "phase": phase,
    }
    started = time.perf_counter()
    samples: list[Sample] = []
    best: tuple[Sample, tuple[GraspCandidate, Execution] | None] | None = None
    scale, batches = 1.0, 0
    pool = None
    if s.workers > 1:
        import multiprocessing

        pool = multiprocessing.get_context("spawn").Pool(s.workers, _init_worker, (job,))
    else:
        _init_worker(job)
    try:
        for batches in range(1, s.max_batches + 1):
            drafts = [dict(center)] if batches == 1 else []
            while len(drafts) < s.batch:
                drafts.append({**center, **{n: center[n] + scale * s.sigma[n] * rng.normal() for n in free}})
            params = [_clip(d, bounds) for d in drafts]
            results = (
                pool.map(_evaluate_job, params) if pool is not None else [_evaluate_job(p) for p in params]
            )
            samples += [r[0] for r in results]
            top = max(results, key=lambda r: r[0].score)
            if best is None or top[0].score > best[0].score:
                best = top
            if best[0].lifted >= s.accept:
                break
            center, scale = dict(best[0].params), scale * s.widen
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    assert best is not None
    planned = best[1]
    return RefinedPlan(
        best[0],
        planned[0] if planned else None,
        planned[1] if planned else None,
        best[0].lifted >= s.accept,
        batches,
        tuple(samples),
        time.perf_counter() - started,
    )


def _greedy_start(
    model: XleRobotModel, primitive: Primitive, start: Mapping[str, float], settings: GraspSettings
):
    from lerobot.felix_jr.harness.grasping import first_feasible

    capture: Any = SimpleNamespace(arms={"right": dict(start)}, head_ticks=None)
    found = first_feasible(
        primitive,
        model,
        capture,
        np.random.default_rng(0),
        arms=("right",),
        where={"psi": (0.0, 30.0)},
        settings=settings,
        depths=(0.0,),
        max_tries=3000,
    ).candidate
    if found is None:
        return None
    return {**found.claw.grasp.params, "depth": found.claw.depth, "offset": 0.0}, found.claw.fixed_side


def main() -> None:
    """Command line entry point; see the module docstring."""
    import argparse
    import json
    from pathlib import Path

    from lerobot.felix_jr.calibration.joint_sim import REST
    from lerobot.felix_jr.harness import api

    from . import bank

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bank", type=Path)
    parser.add_argument("out", type=Path, help="a line per run (.jsonl; under a scratch/ folder)")
    parser.add_argument("--starts", nargs="+", default=["greedy", "anchor"], choices=["greedy", "anchor"])
    parser.add_argument("--placements", type=int, default=None, help="the first N anchored placements")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--fresh", type=int, default=32, help="fresh draws each start and taken grasp is scored on"
    )
    parser.add_argument("--model", type=Path, default=api.MODEL)
    args = parser.parse_args()
    model = XleRobotModel.load(args.model)
    config, lines = bank.load(args.bank)
    anchored = [line for line in lines if line["anchor"] is not None][: args.placements]
    start = {**REST, "gripper.pos": 2.0}
    grasp_settings = config.settings()
    print(
        "| placement | bottle | start | start: search's draws | start: fresh | taken | taken: search's | "
        "taken: fresh | batches | s | grasp taken |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")

    def share(x: Sample, n: int) -> str:
        return f"refused ({x.refused[:24]})" if x.refused else f"{n * x.lifted:.0f}/{n}"

    with args.out.open("a") as out:
        for line in anchored:
            placement = bank.Placement(
                **{k: tuple(v) if isinstance(v, list) else v for k, v in line["placement"].items()}
            )
            primitive = placement.primitive()
            bottle = f"{2000 * placement.radius:.0f} x {1000 * placement.height:.0f} mm, {1000 * placement.mass_kg:.0f} g"
            for kind in args.starts:
                seed = 1000 * placement.index + ("greedy", "anchor").index(kind)  # (each run its own draws)
                settings = RefineSettings(
                    workers=args.workers,
                    table_z=placement.bottom_root[2],
                    mass_kg=placement.mass_kg,
                    seed=seed,
                )
                if kind == "greedy":
                    found = _greedy_start(model, primitive, start, grasp_settings)
                    if found is None:
                        print(f"| {placement.index} | {bottle} | greedy | (none found) |" + " |" * 7)
                        continue
                    initial, side = found
                else:
                    a = line["anchor"]
                    initial, side = {**a["params"], "depth": a["depth"], "offset": 0.0}, a["fixed_side"]
                plan = refine(model, "right", primitive, initial, side, start, grasp_settings, settings)
                first = plan.samples[0]
                fresh = [
                    validate(
                        model,
                        "right",
                        primitive,
                        x,
                        side,
                        start,
                        grasp_settings,
                        settings,
                        args.fresh,
                        seed + 7,
                    )
                    for x in (first.params, plan.best.params)
                ]
                n = settings.draws
                taken = ", ".join(f"{k} {_show(k, v)}" for k, v in plan.best.params.items())
                print(
                    f"| {placement.index} | {bottle} | {kind} | {share(first, n)} | {share(fresh[0], args.fresh)} | "
                    f"{'yes' if plan.accepted else 'no'} | {share(plan.best, n)} | {share(fresh[1], args.fresh)} | "
                    f"{plan.batches} | {plan.seconds:.0f} | {taken} |",
                    flush=True,
                )
                record = {
                    "placement": placement.index,
                    "start": kind,
                    "seed": seed,
                    "initial": initial,
                    "fixed_side": side,
                    "start_lifted": first.lifted,
                    "start_refused": first.refused,
                    "start_fresh": fresh[0].lifted,
                    "accepted": plan.accepted,
                    "lifted": plan.best.lifted,
                    "taken_fresh": fresh[1].lifted,
                    "fresh_draws": args.fresh,
                    "params": plan.best.params,
                    "batches": plan.batches,
                    "seconds": plan.seconds,
                    "samples": len(plan.samples),
                }
                out.write(json.dumps(record) + "\n")
                out.flush()


if __name__ == "__main__":
    main()
