"""Whether a grasp holds in the simulator: the bank's last check (`docs/sim2real_grasp.md`, the grasp environment's starts).

A grasp the harness's filters pass (`harness.grasping.first_feasible`: reachable, the paths clear, the object in the
wrist camera's view) is simulated from its exact grasp pose: the object stood where its primitive is, the arm at the
grasp with the jaws open (`ClawTarget.open_apart`), the jaws closed `reach.SQUEEZE_UNITS` past the object's contact, the
arm moved to the solved lift, held; it held if the object rose with the jaws (`HELD_FRACTION` of how far the jaws rose,
the jaws up `JAWS_MIN_FRACTION` of the lift at least: the simulated arm sags and stops short of a lift as the real one
does, the jaws ~25 of a 30 mm lift, and the planned lift's fraction scored grasps that held as slips; 2026-11-21). Not
the start of
an episode: whether the grasp is physically possible at all (the real arm will not arrive there; the episodes' starts
are offset from it).

Under a few draws of the physics around the fitted values (`physics_draws`: the fingers' friction and stiffness, the
object's mass): a success rate, not a pass (a grasp that holds only with lucky friction is marginal). `SimAccept` is
`first_feasible`'s `accept`: it simulates each grasp the filters pass, records every trial (its depth too: which depths
hold, for the harness's default later), and accepts at `min_rate`.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel
from lerobot.felix_jr.harness.grasping import GraspCandidate
from lerobot.felix_jr.harness.primitives import Cylinder as CylinderPrimitive
from lerobot.felix_jr.harness.reach import close_gripper_pos
from lerobot.utils.import_utils import _mujoco_available, require_package

from . import robot
from .finger_models import settings
from .scene import build, from_primitive

if _mujoco_available:
    import mujoco

SETTLE_S = 0.3  # the object and the open jaws left to settle before the close
MOVED_M = 0.01  # the object moved this much while settling: the start was not clean (the jaws on it)
# the open gripper this far into the object at the start: the grasp is impossible as posed (the harness's filters check
# the fingers' faces, not the gripper's body: a grasp from above, partway down a tall cylinder, puts the jaws, the
# camera's mount and the wrist inside it; 6-29 mm on pyri-like cylinders, 2026-11-18)
START_OVERLAP_M = 0.001
BLEW_UP_M = 0.5  # the object moved this far in a trial: the simulation became unstable
# the states `simulate_grasp(snapshots=True)` keeps: open at the grasp after settling, closed on the object, halfway
# up the lift (the bank's "at the object", "in the jaws" and "lifted" pools)
SNAPSHOTS = ("at_object", "in_jaws", "lifted")
CLOSE_S = 2.0
LIFT_MOVE_S = 1.5
LIFT_HOLD_S = 2.0  # the success criterion's hold (`docs/sim2real_grasp.md`)
HELD_FRACTION = 0.75  # the object up by this much of the jaws' rise: held
JAWS_MIN_FRACTION = 0.5  # the jaws up this much of the planned lift at least (else the lift itself failed)
# the physics drawn about the fitted values (`robot.py`, `docs/sim2real_grasp.md`): each a factor, uniform in log
FRICTION_RANGE = (0.8, 1.25)
STIFFNESS_RANGE = (0.75, 1.33)
MASS_RANGE = (0.8, 1.25)


@dataclass(frozen=True)
class GraspTrial:
    """One simulated grasp: the physics drawn, whether it held, how far the object rose, and why not if it did not."""

    depth: float
    draw: dict[str, float]
    held: bool
    rose_m: float
    note: str = ""
    jaws_rose_m: float = float("nan")  # how far the jaws rose (the gripper's body)
    # with `snapshots`: the simulator's state ("qpos", "qvel", "ctrl") at each stage (`SNAPSHOTS`), for the bank's pools
    states: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)


def physics_draws(rng: np.random.Generator, n: int) -> list[dict[str, float]]:
    """`n` draws of the physics about the fitted values: the fingers' friction and stiffness, the object's mass factor."""

    def log_uniform(lo: float, hi: float) -> float:
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))

    return [
        {
            "FINGER_FRICTION": robot.FINGER_FRICTION * log_uniform(*FRICTION_RANGE),
            "FINGER_STIFFNESS_NM_PER_RAD": robot.FINGER_STIFFNESS_NM_PER_RAD * log_uniform(*STIFFNESS_RANGE),
            "mass_scale": log_uniform(*MASS_RANGE),
        }
        for _ in range(n)
    ]


def simulate_grasp(
    model: XleRobotModel,
    candidate: GraspCandidate,
    target: CylinderPrimitive,
    mass_kg: float,
    table_z: float,
    draw: Mapping[str, float],
    finger_model: str = "sprung",
    snapshots: bool = False,
) -> GraspTrial:
    """`candidate`'s grasp of `target` simulated once, under `draw` (`physics_draws`; the module docstring).

    `snapshots`: the state kept at each of `SNAPSHOTS` (`GraspTrial.states`).
    """
    require_package("mujoco", extra="felix_jr_sim")
    if abs(float(target.axis_root[0, 2])) < 0.99:  # (the target: a batch of one)
        raise NotImplementedError("only upright cylinders are simulated so far")
    claw, arm = candidate.claw, candidate.claw.arm
    width = claw.grasp.opening
    obj = from_primitive(target, mass=mass_kg * draw.get("mass_scale", 1.0))
    with settings({k: v for k, v in draw.items() if k.isupper()}):
        scene = build(model, arm, [obj], table_z=table_z, finger_model=finger_model)
    m, d = scene.m, scene.d
    gripper = m.actuator(robot.GRIPPER).id
    open_pos = model.gripper_pos_for(
        arm, model.opening_for_width(arm, min(claw.open_apart, 2 * model.jaw_radius(arm)))
    )
    scene.set_pose({**candidate.grasp.joints, "gripper.pos": float(np.clip(open_pos, 0.0, 100.0))})
    body = m.body("object").id

    def run(seconds: float) -> None:
        for _ in range(int(seconds / m.opt.timestep)):
            mujoco.mj_step(m, d)

    states: dict[str, dict[str, np.ndarray]] = {}

    def keep(stage: str) -> None:
        if snapshots:
            states[stage] = {"qpos": d.qpos.copy(), "qvel": d.qvel.copy(), "ctrl": d.ctrl.copy()}

    start = d.xpos[body].copy()
    touching = _touching(m, d, body)  # (what the object starts in contact with, for the note)
    run(SETTLE_S)
    if np.linalg.norm(d.xpos[body] - start) > MOVED_M:
        note = f"the object moved before the close (touching at the start: {touching})"
        return GraspTrial(claw.depth, dict(draw), False, 0.0, note)
    keep("at_object")
    d.ctrl[gripper] = np.radians(model.opening_deg(arm, close_gripper_pos(model, arm, width)))
    run(CLOSE_S)
    keep("in_jaws")
    z0 = float(d.xpos[body][2])
    jaws = m.body(robot.GRIPPER).id
    jaws_z0 = float(d.xpos[jaws][2])
    assert candidate.lift is not None
    begin = np.array(d.ctrl).copy()
    steps = int(LIFT_MOVE_S / m.opt.timestep)
    for k in range(steps):
        f = (k + 1) / steps
        for joint, angle in candidate.lift.joints.items():
            a = m.actuator(joint.removesuffix(".pos")).id
            d.ctrl[a] = begin[a] + f * (np.radians(angle) - begin[a])
        mujoco.mj_step(m, d)
        if k == steps // 2:
            keep("lifted")
    run(LIFT_HOLD_S)
    rose = float(d.xpos[body][2]) - z0
    if np.linalg.norm(d.xpos[body] - start) > BLEW_UP_M:
        return GraspTrial(claw.depth, dict(draw), False, 0.0, "the simulation became unstable")
    lift = float(candidate.lift.root_from_pinch[2, 3] - candidate.grasp.root_from_pinch[2, 3])
    jaws_rose = float(d.xpos[jaws][2]) - jaws_z0
    if jaws_rose < JAWS_MIN_FRACTION * lift:
        note = f"the jaws rose {1000 * jaws_rose:.0f} of {1000 * lift:.0f} mm"
        return GraspTrial(claw.depth, dict(draw), False, rose, note, jaws_rose_m=jaws_rose, states=states)
    held = rose > HELD_FRACTION * jaws_rose
    note = "" if held else "the object did not come up with the jaws"
    return GraspTrial(claw.depth, dict(draw), held, rose, note, jaws_rose_m=jaws_rose, states=states)


def start_overlap(
    model: XleRobotModel, candidate: GraspCandidate, target: CylinderPrimitive, table_z: float
) -> tuple[float, str]:
    """How deep (m) the open gripper at the grasp is in the object, and what of it (no physics: one build, a forward).

    The grasp is impossible as posed past `START_OVERLAP_M`: `SimAccept` checks it once, before its physics draws.
    """
    require_package("mujoco", extra="felix_jr_sim")
    claw, arm = candidate.claw, candidate.claw.arm
    obj = from_primitive(target)
    scene = build(model, arm, [obj], table_z=table_z)
    apart = min(claw.open_apart, 2 * model.jaw_radius(arm))
    open_pos = model.gripper_pos_for(arm, model.opening_for_width(arm, apart))
    scene.set_pose({**candidate.grasp.joints, "gripper.pos": float(np.clip(open_pos, 0.0, 100.0))})
    m, d = scene.m, scene.d
    body = m.body("object").id
    deepest = 0.0
    for c in d.contact[: d.ncon]:
        if body in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]):
            other = c.geom2 if m.geom_bodyid[c.geom1] == body else c.geom1
            if m.geom(other).name != "table":
                deepest = max(deepest, -float(c.dist))
    return deepest, _touching(m, d, body)


def _touching(m: "mujoco.MjModel", d: "mujoco.MjData", body: int) -> str:
    """The bodies the object is in contact with (beyond the table), deepest penetration first, as text."""
    hits: dict[str, float] = {}
    for c in d.contact[: d.ncon]:
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if body not in (b1, b2):
            continue
        other = m.geom(c.geom2 if b1 == body else c.geom1)
        name = other.name or m.body(m.geom_bodyid[other.id]).name
        if name != "table":
            hits[name] = min(hits.get(name, 0.0), float(c.dist))
    return (
        ", ".join(f"{k} {1000 * v:.1f} mm" for k, v in sorted(hits.items(), key=lambda kv: kv[1]))
        or "nothing"
    )


def _grasp_key(candidate: GraspCandidate) -> tuple[Any, ...]:
    """What makes two grasps the same for the bank: the depth (mm), the height (cm), the tilt and roll (10 deg)."""
    p = candidate.claw.grasp.params
    return (
        round(1000 * candidate.claw.depth),
        round(100 * float(p.get("t", 0.0))),
        round(float(p.get("psi", 0.0)) / 10),
        round(float(p.get("phi", 0.0)) / 10),
    )


@dataclass
class SimAccept:
    """`first_feasible`'s `accept`: a grasp held in at least `min_rate` of `n_draws` physics draws (the module docstring).

    Every trial is kept (`trials`, by the candidate's index in the order tried; `candidates`, the grasps, alike), the
    rejected grasps' too. `snapshots`: each trial keeps its states (`simulate_grasp`; the bank's pools).
    """

    model: XleRobotModel
    target: CylinderPrimitive
    mass_kg: float
    table_z: float
    rng: np.random.Generator
    n_draws: int = 3
    min_rate: float = 2 / 3
    finger_model: str = "sprung"
    snapshots: bool = False
    # how many distinct grasps to accept before `first_feasible` stops (the bank's anchors per placement: diverse
    # grasps, OmniReset's lever); each accepted one's index in `candidates` is in `accepted`
    want: int = 1
    trials: list[list[GraspTrial]] = field(default_factory=list)
    candidates: list[GraspCandidate] = field(default_factory=list)
    accepted: list[int] = field(default_factory=list)

    def __call__(self, candidate: GraspCandidate) -> bool:
        """Simulate `candidate` under `n_draws` draws; accept at `min_rate` (refused at once if it starts in the object)."""
        self.candidates.append(candidate)
        deepest, touching = start_overlap(self.model, candidate, self.target, self.table_z)
        if deepest > START_OVERLAP_M:
            note = f"the open gripper {1000 * deepest:.0f} mm into the object at the grasp ({touching})"
            self.trials.append([GraspTrial(candidate.claw.depth, {}, False, 0.0, note)])
            return False
        runs = [
            simulate_grasp(
                self.model,
                candidate,
                self.target,
                self.mass_kg,
                self.table_z,
                draw,
                self.finger_model,
                self.snapshots,
            )
            for draw in physics_draws(self.rng, self.n_draws)
        ]
        self.trials.append(runs)
        if sum(r.held for r in runs) < self.min_rate * len(runs):
            return False
        key = _grasp_key(candidate)
        if all(_grasp_key(self.candidates[i]) != key for i in self.accepted):  # (a distinct grasp)
            self.accepted.append(len(self.candidates) - 1)
        return len(self.accepted) >= self.want

    def summary(self) -> dict[str, Any]:
        """Per depth tried: grasps simulated, and how many were accepted."""
        out: dict[str, Any] = {}
        for runs in self.trials:
            depth = f"{1000 * runs[0].depth:g} mm"
            row = out.setdefault(depth, {"simulated": 0, "accepted": 0})
            row["simulated"] += 1
            row["accepted"] += int(sum(r.held for r in runs) >= self.min_rate * len(runs))
        return out
