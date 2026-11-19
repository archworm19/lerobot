"""The grasp environment's starts, drawn from a bank (`bank.py`; `docs/sim2real_grasp.md`, the grasp environment's starts).

A start is a pool by `StartConfig.pool_weights`, an anchor of the bank (a placement whose grasp held in the simulator),
and the simulator's state for it, ready to step (`Start.scene`):

- **attack:** the arm at the anchor's attack pose, its joints perturbed as the real robot misses: the perception's error
  (the gripper off the object: `perception_*`, a Cartesian offset in the gripper's frame, mapped to the joints through
  the pinch point's Jacobian at the attack) and the execution's spread (`joint_std_deg`, per joint); both fitted from
  the harness's sessions (`shortfalls.py`, `sessions.md`) and sampled wider (`*_width`). The arm arrives there as the
  real one did (`arrival_deg` back, each joint from the side the sessions' attacks came from, `ARRIVAL_UP`; ramped at
  the harness's speed): the servos stop short on that side as the real ones do (`robot.ARM_KP_NM_PER_RAD`), so the
  execution's bias is the arrival's, not drawn. The grasp stays the anchor's: it is reachable, whatever the draw (no
  IK at a reset).
- **at_object / in_jaws / lifted:** the anchor's snapshot (`grasp_check.SNAPSHOTS`), the object nudged a few mm in the
  open jaws (`at_object_mm`).

Then a short settle (`SETTLE_S`) and a cheap check (`check`): the arm not in the object nor on the table (the open
pools), the object not moved off on its own; a start that fails is drawn again (`max_draws`).
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel, arm_joint, pinch_frame
from lerobot.utils.import_utils import _mujoco_available, require_package

from . import bank
from .scene import Cylinder, Scene, build

if _mujoco_available:
    import mujoco

POOLS = ("attack", "at_object", "in_jaws", "lifted")
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
# the execution's spread at the attack (`shortfalls`, 28 holds of 13 sessions: read - commanded's std, deg)
JOINT_STD_DEG = (0.35, 0.35, 0.5, 0.3, 0.35)
# the perception's miss (`sessions.md`: the jaws ~1-1.5 cm short along the approach, bottles 0.3-2.3 cm ahead of the
# aim point, 0.5-1.5 cm across; the sag the encoders miss, pen2's 1.3 cm against 0.9), m: in the gripper's frame,
# along its axis (+: toward the tips) and across the jaws, and up
PERCEPTION_MEAN_M = {"along": -0.010, "across": 0.0, "up": -0.004}
PERCEPTION_STD_M = {"along": 0.007, "across": 0.007, "up": 0.003}
# how often each joint came into the sessions' attacks moving up its range (`shortfalls`, 28 holds: the lift 22 of 22
# that moved, wrist_flex 1 of 22, the elbow 9 of 18; the pan and the roll moved too little to say)
ARRIVAL_UP = {
    "shoulder_pan": 0.5,
    "shoulder_lift": 1.0,
    "elbow_flex": 0.5,
    "wrist_flex": 0.05,
    "wrist_roll": 0.5,
}
ARRIVAL_DEG_S = 20.0  # the harness's `execute` speed
SETTLE_S = 0.3
MOVED_M = 0.005  # the object moved this far in the settle: the start was not clean
PENETRATION_M = 0.0005  # the arm this far into the object or the table: rejected


@dataclass(frozen=True)
class StartConfig:
    """How starts are drawn (the module docstring); the defaults the sessions' fits, sampled half again as wide."""

    pool_weights: Mapping[str, float] = field(
        default_factory=lambda: {"attack": 0.5, "at_object": 0.2, "in_jaws": 0.2, "lifted": 0.1}
    )
    joint_std_deg: tuple[float, ...] = JOINT_STD_DEG
    execution_width: float = 1.5
    perception_mean_m: Mapping[str, float] = field(default_factory=lambda: dict(PERCEPTION_MEAN_M))
    perception_std_m: Mapping[str, float] = field(default_factory=lambda: dict(PERCEPTION_STD_M))
    perception_width: float = 1.5
    arrival_deg: float = 3.0  # the attack pose arrived at from this far back, each joint (`ARRIVAL_UP`)
    at_object_mm: float = 3.0  # the object's nudge in the open jaws (std, each way on the table)
    max_draws: int = 20


@dataclass
class Start:
    """One start: its pool, the anchor's index in the bank (`StartSampler.anchors`), the scene in its state.

    And what was drawn: `offset_m` the perception's offset, `joints_deg` the joints commanded, `draws` how many tries.
    """

    pool: str
    anchor: int
    scene: Scene
    offset_m: dict[str, float] = field(default_factory=dict)
    joints_deg: dict[str, float] = field(default_factory=dict)
    draws: int = 1


class StartSampler:
    """Starts from the anchors of the bank at `path` (the module docstring)."""

    def __init__(self, model: XleRobotModel, path: Path, config: StartConfig | None = None) -> None:
        """The bank read (refused if built under other servos: its snapshots would not match) and its anchors kept."""
        require_package("mujoco", extra="felix_jr_sim")
        self.model = model
        self.config = StartConfig() if config is None else config
        self.bank_config, lines = bank.load(path)
        meta = bank.meta(path)
        if meta.get("servos") is not None and meta["servos"] != bank.servos():
            raise ValueError(
                f"{path} was built with other servos ({meta['servos']}; now {bank.servos()}): its snapshots "
                "would not match the scene; rebuild it"
            )
        self.anchors = [line for line in lines if line["anchor"] is not None]
        if not self.anchors:
            raise ValueError(f"{path} has no anchors")
        weights = np.array([self.config.pool_weights.get(p, 0.0) for p in POOLS], dtype=float)
        if weights.sum() <= 0:
            raise ValueError(f"no pool has weight: {self.config.pool_weights}")
        self._weights = weights / weights.sum()

    def sample(self, rng: np.random.Generator, pool: str | None = None) -> Start:
        """A start (its pool drawn by the weights unless given), drawn again until it passes `check`."""
        pool = POOLS[int(rng.choice(len(POOLS), p=self._weights))] if pool is None else pool
        for draw in range(1, self.config.max_draws + 1):
            index = int(rng.integers(len(self.anchors)))
            start = self._draw(pool, index, rng)
            if check(start.scene, pool):
                start.draws = draw
                return start
        raise RuntimeError(f"no {pool} start passed the check in {self.config.max_draws} draws")

    def scene(self, index: int) -> Scene:
        """Anchor `index`'s scene: its object where it stood, the arm at zero."""
        line = self.anchors[index]
        p = line["placement"]
        x, y, z = p["bottom_root"]
        obj = Cylinder("object", x, y, p["radius"], p["height"], p["mass_kg"])
        return build(
            self.model, self.bank_config.arm, [obj], table_z=z, finger_model=self.bank_config.finger_model
        )

    def _draw(self, pool: str, index: int, rng: np.random.Generator) -> Start:
        anchor = self.anchors[index]["anchor"]
        scene = self.scene(index)
        m, d = scene.m, scene.d
        if pool == "attack":
            offset, joints = self.attack_joints(anchor, rng)
            jaws = open_gripper_pos(self.model, self.bank_config.arm, anchor)
            up = np.array([rng.random() < ARRIVAL_UP[j] for j in JOINTS])
            before = {
                j: joints[j] - (1 if up[i] else -1) * self.config.arrival_deg for i, j in enumerate(JOINTS)
            }
            scene.set_pose({**{f"{j}.pos": v for j, v in before.items()}, "gripper.pos": jaws})
            if _penetration(m, d, ("table",)) <= PENETRATION_M:  # (else `check` refuses it as it stands)
                arrive(scene, before, joints)
            start = Start(pool, index, scene, offset, joints)
        else:
            state = anchor["states"][pool]
            d.qpos[:], d.qvel[:], d.ctrl[:] = state["qpos"], state["qvel"], state["ctrl"]
            if pool == "at_object" and self.config.at_object_mm > 0:
                adr = m.jnt_qposadr[m.body("object").jntadr[0]]
                d.qpos[adr : adr + 2] += rng.normal(0.0, self.config.at_object_mm / 1000.0, 2)
            mujoco.mj_forward(m, d)
            start = Start(pool, index, scene)
        return start

    def attack_joints(
        self, anchor: Mapping[str, Any], rng: np.random.Generator
    ) -> tuple[dict[str, float], dict[str, float]]:
        """The perception's offset drawn (m, the gripper's frame), and the attack's joints that make it (deg).

        The execution's spread is added to the joints.
        """
        c, arm = self.config, self.bank_config.arm
        attack = anchor["attack"]["joints"]
        offset = {
            k: c.perception_mean_m[k] + c.perception_width * c.perception_std_m[k] * float(rng.normal())
            for k in ("along", "across", "up")
        }
        q = self.model.joint_vector({arm: attack})
        pose, jac = self.model.frame_jacobian(pinch_frame(arm), q, [arm_joint(arm, j) for j in JOINTS])
        # the gripper's frame: x across the jaws, z along its axis (`XleRobotModel.grasp_offset`)
        move = pose[:3, :3] @ np.array([offset["across"], 0.0, offset["along"]]) + np.array(
            [0.0, 0.0, offset["up"]]
        )
        dq = np.linalg.pinv(jac[:3]) @ move
        spread = c.execution_width * np.array(c.joint_std_deg) * rng.normal(size=len(JOINTS))
        joints = {j: float(attack[f"{j}.pos"] + np.degrees(dq[i]) + spread[i]) for i, j in enumerate(JOINTS)}
        return offset, joints


def arrive(scene: Scene, start: Mapping[str, float], goal: Mapping[str, float]) -> None:
    """The arm's servos ramped from `start` to `goal` (deg, `JOINTS`) at `ARRIVAL_DEG_S`: where it stops is the servos'."""
    m, d = scene.m, scene.d
    act = [m.actuator(j).id for j in JOINTS]
    a = np.radians([start[j] for j in JOINTS])
    b = np.radians([goal[j] for j in JOINTS])
    steps = max(1, int(float(np.degrees(np.abs(b - a)).max()) / ARRIVAL_DEG_S / m.opt.timestep))
    for k in range(steps):
        d.ctrl[act] = a + (k + 1) / steps * (b - a)
        mujoco.mj_step(m, d)


def open_gripper_pos(model: XleRobotModel, arm: str, anchor: Mapping[str, Any]) -> float:
    """The jaws open for the anchor's approach (`ClawTarget.open_apart`, as `grasp_check.simulate_grasp` opens them)."""
    apart = min(anchor["open_apart"], 2 * model.jaw_radius(arm))
    return float(np.clip(model.gripper_pos_for(arm, model.opening_for_width(arm, apart)), 0.0, 100.0))


def _penetration(m: "mujoco.MjModel", d: "mujoco.MjData", names: tuple[str, ...]) -> float:
    """The deepest any of the arm's geoms is into the object or the table (m; 0: none).

    A geom passed right through the object reads shallow (MuJoCo's depth: a finger's pad shut through a bottle, 0.1
    mm): fine for the starts' offsets, which are small against the attack's distance back.
    """
    obj = m.body("object").id
    deepest = 0.0
    for c in d.contact[: d.ncon]:
        g1, g2 = int(c.geom1), int(c.geom2)
        b1, b2 = m.geom_bodyid[g1], m.geom_bodyid[g2]
        for mine, other, other_body in ((g1, g2, b2), (g2, g1, b1)):
            if m.geom_bodyid[mine] in (obj, 0):
                continue  # (the object's or the world's own side)
            if other_body == obj or (m.geom(other).name in names):
                deepest = max(deepest, -float(c.dist))
    return deepest


def check(scene: Scene, pool: str) -> bool:
    """Whether the start is clean (it is stepped `SETTLE_S` here).

    With the jaws open (attack, at_object), the arm not in the object nor on the table, before and after the settle;
    for every pool, the object not moved off in it.
    """
    m, d = scene.m, scene.d
    open_pools = pool in ("attack", "at_object")
    if open_pools and _penetration(m, d, ("table",)) > PENETRATION_M:
        return False
    body = m.body("object").id
    start = d.xpos[body].copy()
    for _ in range(int(SETTLE_S / m.opt.timestep)):
        mujoco.mj_step(m, d)
    if np.linalg.norm(d.xpos[body] - start) > MOVED_M:
        return False
    return not (open_pools and _penetration(m, d, ("table",)) > PENETRATION_M)
