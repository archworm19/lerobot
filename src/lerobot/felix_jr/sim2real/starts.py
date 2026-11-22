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

  uv run --extra felix_jr --extra felix_jr_sim python -m lerobot.felix_jr.sim2real.starts <bank.jsonl> <out.mp4>
      [--pairs 6] [--clips attack=2 at_object=1 in_jaws=1 lifted=1] [--seed 0]

`video`: starts played forward with the anchor's script from their stage on (to the grasp, close, lift, hold), a view
of the scene and the wrist camera's: `--pairs` attack starts each below the same anchor unperturbed (its exact attack,
arrived at the same way), the attack's miss kept through the script (no correction: what the policy has to close);
`--clips` single starts per pool.
"""

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel, arm_joint, pinch_frame
from lerobot.felix_jr.harness.primitives import RigidBody
from lerobot.felix_jr.harness.reach import close_gripper_pos
from lerobot.utils.import_utils import _mujoco_available, require_package

from . import bank, robot
from .scene import Scene, build, from_body

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

    def __init__(
        self,
        model: XleRobotModel,
        path: Path,
        config: StartConfig | None = None,
        placements: "set[int] | None" = None,
    ) -> None:
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
        # each grasp a placement accepted is an anchor of its own (`anchors`; a bank before it: `anchor` alone)
        # (`placements`: only those placements' anchors, by index: a train or held-out split, `start_pool --holdout`)
        self.anchors = [
            {**line, "anchor": a}
            for line in lines
            if placements is None or line["placement"]["index"] in placements
            for a in (line.get("anchors") or ([line["anchor"]] if line["anchor"] is not None else []))
        ]
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
        body = RigidBody.from_record(line["placement"])
        return build(
            self.model,
            self.bank_config.arm,
            [from_body(body)],
            table_z=float(body.shape.bottom_root[2]),
            finger_model=self.bank_config.finger_model,
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


# ---- video ------------------------------------------------------------------------------------------------------

VIDEO_FPS, TILE = 30, (480, 360)
SCRIPT_S = {"start": 0.7, "grasp": 1.5, "close": 1.5, "lift": 1.5, "hold": 1.5, "result": 1.0}
# each pool's script from its stage on
SCRIPTS = {
    "attack": ("grasp", "close", "lift", "hold"),
    "at_object": ("close", "lift", "hold"),
    "in_jaws": ("lift", "hold"),
    "lifted": ("lift", "hold"),
}


# why a pool's clip has no pair (`video`'s bottom half)
POOL_NOTES = {
    "attack": ["(a single attack start: --pairs shows one against its anchor)"],
    "at_object": ["the anchor's snapshot, the object nudged a few mm (`at_object_mm`)"],
    "in_jaws": [
        "the anchor's snapshot exactly: this pool is not perturbed yet",
        "(poor grips planned: the plan's pools)",
    ],
    "lifted": ["the anchor's snapshot exactly: this pool is not perturbed yet"],
}


def _text(image: np.ndarray, lines: Sequence[str]) -> None:
    for k, line in enumerate(lines):
        for color, width in (((0, 0, 0), 3), ((255, 255, 255), 1)):
            cv2.putText(
                image, line, (8, 22 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, width, cv2.LINE_AA
            )


class _Run:
    """One scene playing an anchor's script, its joints shifted by `shift` (deg), drawn from beside and the wrist."""

    def __init__(
        self,
        model: XleRobotModel,
        scene: Scene,
        line: Mapping[str, Any],
        shift: Mapping[str, float],
        title: list[str],
    ) -> None:
        from .view import COLLISION, SITES, VISUAL

        self.model, self.scene, self.anchor, self.placement = model, scene, line["anchor"], line["placement"]
        self.shift, self.title = shift, title
        m, d = scene.m, scene.d
        self.body = m.body("object").id
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = d.xpos[self.body] + [0.0, 0.0, 0.02]
        self.camera.distance, self.camera.azimuth, self.camera.elevation = 0.45, 120.0, -20.0
        self.renderers = [mujoco.Renderer(m, TILE[1], TILE[0]) for _ in range(2)]
        self.options = []
        for own in (False, True):  # (the wrist camera's module left out of its own view)
            option = mujoco.MjvOption()
            option.geomgroup[VISUAL], option.geomgroup[COLLISION] = True, False
            option.geomgroup[robot.OWN_MODULE_GROUPS[VISUAL]] = not own
            option.geomgroup[robot.OWN_MODULE_GROUPS[COLLISION]] = False
            option.sitegroup[SITES] = False
            self.options.append(option)
        self.act = [m.actuator(j).id for j in JOINTS]

    def target(self, pose: str) -> np.ndarray:
        """The anchor's `pose` joints, shifted (rad)."""
        return np.radians([self.anchor[pose]["joints"][f"{j}.pos"] + self.shift.get(j, 0.0) for j in JOINTS])

    def off_table_cm(self) -> float:
        """How far the object's center is above where it stands (cm)."""
        shape = RigidBody.from_record(self.placement).shape
        resting = float(shape.bottom_root[2]) + float(shape.center_obj[2])
        return 100 * (float(self.scene.d.xpos[self.body][2]) - resting)

    def frame(self, phase: str) -> np.ndarray:
        """The two views side by side, labeled."""
        d = self.scene.d
        views = []
        for renderer, option, camera in zip(
            self.renderers, self.options, (self.camera, robot.WRIST_CAMERA), strict=True
        ):
            renderer.update_scene(d, camera=camera, scene_option=option)
            views.append(cv2.resize(renderer.render(), TILE))
        _text(views[0], [*self.title, phase, f"object {self.off_table_cm():+.1f} cm off the table"])
        _text(views[1], ["wrist camera"])
        return np.hstack(views)

    def close(self) -> None:
        """Free the renderers."""
        for renderer in self.renderers:
            renderer.close()


def _play(runs: Sequence[_Run], script: Sequence[str], writer: "cv2.VideoWriter") -> list[str]:
    """`runs` stepped together through `script` (`SCRIPT_S`), stacked top to bottom; each one's result."""
    m = runs[0].scene.m
    every = max(1, int(round(1 / VIDEO_FPS / m.opt.timestep)))
    count = [0]

    def emit(phase: str) -> None:
        writer.write(cv2.cvtColor(np.vstack([r.frame(phase) for r in runs]), cv2.COLOR_RGB2BGR))

    for _ in range(int(SCRIPT_S["start"] * VIDEO_FPS)):
        emit("the start")
    for stage in script:
        steps = int(SCRIPT_S[stage] / m.opt.timestep)
        begin = [r.scene.d.ctrl[r.act].copy() for r in runs]
        goal = [r.target(stage) if stage in ("grasp", "lift") else None for r in runs]
        if stage == "close":
            for r in runs:
                width = close_gripper_pos(r.model, r.scene.arm, r.anchor["opening"])
                r.scene.d.ctrl[r.scene.m.actuator(robot.GRIPPER).id] = np.radians(
                    r.model.opening_deg(r.scene.arm, width)
                )
        for k in range(steps):
            for r, b, g in zip(runs, begin, goal, strict=True):
                if g is not None:
                    r.scene.d.ctrl[r.act] = b + (k + 1) / steps * (g - b)
                mujoco.mj_step(r.scene.m, r.scene.d)
            count[0] += 1
            if count[0] % every == 0:
                emit(stage)
    results = [
        ("held" if r.off_table_cm() > 1.0 else "not held") + f" ({r.off_table_cm():.1f} cm)" for r in runs
    ]
    for _ in range(int(SCRIPT_S["result"] * VIDEO_FPS)):
        emit("result: " + " / ".join(results))
    return results


def _describe(line: Mapping[str, Any]) -> str:
    body, a = RigidBody.from_record(line["placement"]), line["anchor"]
    params = body.shape.shape_params()
    return (
        f"{2000 * params['radius']:.0f} mm x {1000 * params['height']:.0f} mm, {1000 * body.mass_kg:.0f} g, "
        f"depth {1000 * a['depth']:.0f} mm"
    )


def unperturbed(sampler: StartSampler, index: int) -> Scene:
    """Anchor `index`'s scene at its exact attack, the jaws open, arrived at as `StartSampler` arrives."""
    anchor, arm = sampler.anchors[index]["anchor"], sampler.bank_config.arm
    scene = sampler.scene(index)
    attack = {j: anchor["attack"]["joints"][f"{j}.pos"] for j in JOINTS}
    back = sampler.config.arrival_deg
    before = {j: attack[j] - (back if ARRIVAL_UP[j] >= 0.5 else -back) for j in JOINTS}
    jaws = open_gripper_pos(sampler.model, arm, anchor)
    scene.set_pose({**{f"{j}.pos": v for j, v in before.items()}, "gripper.pos": jaws})
    arrive(scene, before, attack)
    return scene


def video(
    sampler: StartSampler,
    out: Path,
    rng: np.random.Generator,
    pairs: int = 6,
    clips: Mapping[str, int] | None = None,
) -> list[str]:
    """Starts played forward into `out` (mp4; the module docstring): `pairs` attack pairs, then `clips` per pool.

    Returns a line per clip: what it was and how it ended.
    """
    require_package("mujoco", extra="felix_jr_sim")
    model = sampler.model
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, (2 * TILE[0], 2 * TILE[1]))
    lines = []
    try:
        n = 0
        for _ in range(pairs):
            n += 1
            start = sampler.sample(rng, "attack")
            line = sampler.anchors[start.anchor]
            o = start.offset_m
            offset = f"offset (cm): along {100 * o['along']:+.1f}, across {100 * o['across']:+.1f}, up {100 * o['up']:+.1f}"
            attack = line["anchor"]["attack"]["joints"]
            shift = {j: start.joints_deg[j] - attack[f"{j}.pos"] for j in JOINTS}
            runs = [
                _Run(
                    model,
                    unperturbed(sampler, start.anchor),
                    line,
                    {},
                    [f"{n}. unperturbed", _describe(line)],
                ),
                _Run(model, start.scene, line, shift, [f"{n}. perturbed (no correction)", offset]),
            ]
            results = _play(runs, SCRIPTS["attack"], writer)
            lines.append(f"{n}. attack pair, {_describe(line)}, {offset}: {results[0]} / {results[1]}")
            for r in runs:
                r.close()
        for pool, count in (clips or {}).items():
            for _ in range(count):
                n += 1
                start = sampler.sample(rng, pool)
                line = sampler.anchors[start.anchor]
                title = [f"{n}. {pool}", _describe(line)]
                if start.offset_m:
                    o = start.offset_m
                    title.append(
                        f"offset (cm): along {100 * o['along']:+.1f}, across {100 * o['across']:+.1f}"
                    )
                attack = line["anchor"]["attack"]["joints"]
                shift = {j: start.joints_deg[j] - attack[f"{j}.pos"] for j in start.joints_deg}
                run = _Run(model, start.scene, line, shift, title)

                results = _play([run], SCRIPTS[pool], _Stacked(writer, POOL_NOTES[pool]))
                lines.append(f"{n}. {pool}, {_describe(line)}: {results[0]}")
                run.close()
    finally:
        writer.release()
    return lines


class _Stacked:
    """A writer that pads a single run's frames (half height) with a bottom half saying why there is no pair."""

    def __init__(self, writer: "cv2.VideoWriter", note: Sequence[str]) -> None:
        self.writer = writer
        self.blank = np.zeros((TILE[1], 2 * TILE[0], 3), dtype=np.uint8)
        _text(self.blank, ["no unperturbed run to compare:", *note])

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(np.vstack([frame, self.blank]))


def main() -> None:
    """Command line entry point; see the module docstring."""
    from lerobot.felix_jr.harness import api

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bank", type=Path)
    parser.add_argument("out", type=Path, help="the video (.mp4; under a scratch/ folder)")
    parser.add_argument(
        "--pairs", type=int, default=6, help="attack starts, each below the anchor unperturbed"
    )
    parser.add_argument("--clips", nargs="*", default=[], help="single starts per pool: pool=count ...")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=Path, default=api.MODEL)
    args = parser.parse_args()
    clips = {k: int(v) for k, v in (c.split("=") for c in args.clips)}
    unknown = set(clips) - set(POOLS)
    if unknown:
        parser.error(f"no pool {', '.join(sorted(unknown))}: {', '.join(POOLS)}")
    sampler = StartSampler(XleRobotModel.load(args.model), args.bank)
    for line in video(sampler, args.out, np.random.default_rng(args.seed), args.pairs, clips):
        print(line, flush=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
