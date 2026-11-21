"""Our simulator as an mjlab environment, for training a grasp policy on a GPU (`docs/sim_deploy.md`, step 5).

The same robot, physics and control as the CPU simulator (`robot`, `scene`, `shadow`), in mjlab's terms:

- **Entities.** "robot": `robot.arm_spec` itself (the calibrated arm, the sprung fingers and their pads, the camera's
  board), its servos as fitted kept as they are (`XmlActuatorCfg`: no gains, armature or friction restated); "object":
  an upright cylinder as `scene` builds one, its pose the entity's initial state (mjlab resets a free body to that);
  "table": `scene_spec`'s table. mjlab wraps a fixed-base entity in a mocap body at the origin, so each keeps its pose
  in the root frame (no terrain: the environments' origins are all at 0).
- **Physics** (`SIM`): `robot.arm_spec`'s options restated, which mjlab's attach drops (its defaults differ: an
  implicit integrator, a pyramidal cone, impratio 1).
- **Control:** joint position targets (rad; the gripper's opening through the arm's `GripperMap`, as `robot.set_pose`)
  every 17 physics steps (`DECIMATION`: ~30 Hz, the server's waypoint rate, `trajectory.WAYPOINT_HZ`).
- **Resets and randomization:** the arm at `start` (LeRobot's readings), the object within `object_sigma_m` of its
  spot, the fingers' friction and give and the object's mass drawn as `grasp_check.physics_draws` draws them.
- **Reward / termination:** the object's rise towards the lift (`refine`'s lifted: `HELD_FRACTION` of `LIFT_M`); the
  episode ends at its length or when the object tips past `refine.KNOCKED_DEG`.

`model_differences` compares this environment's compiled model with `scene.build`'s, field by field: the check that
nothing changed on the way (`docs/sim_deploy.md`).
"""

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.felix_jr.calibration.joint_sim import REST
from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel
from lerobot.felix_jr.harness.grasping import LIFT_M
from lerobot.utils.import_utils import require_package

from . import robot, scene as sc
from .grasp_check import FRICTION_RANGE, MASS_RANGE, STIFFNESS_RANGE
from .refine import HELD_FRACTION, KNOCKED_DEG

if TYPE_CHECKING:
    import mujoco
    import torch
    from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
    from mjlab.sim import SimulationCfg

DECIMATION = (
    17  # 17 x 2 ms: ~30 Hz, the server's waypoint rate (`trajectory.WAYPOINT_HZ`; the shadow's per tick)
)
NCONMAX, NJMAX = 64, 400  # per world (`cloud/warp_check.py`: ~1 MB a world)
EPISODE_S = 12.0
# the start pool's mix (`start_pool`): mostly the jaws open around the object, the positive control's easy start
POOL_WEIGHTS = {"attack": 0.2, "at_object": 0.5, "in_jaws": 0.2, "lifted": 0.1}
OBJECT_SIGMA_M = 0.007  # the object's spread about its spot (`refine.RefineSettings.object_sigma_m`)
FINGER_PADS = r"finger_.*_pad\d+"
FINGER_GIVES = r"finger_.*_give"
# the fingers' give in mjlab x this (`env_cfg`'s default): MuJoCo Warp's contact squeezed ~1.5 points less load than CPU
# MuJoCo's at each grip trial's close (2026-11-21); x1.3 brings the stalls and loads to the robot's as near as CPU
# MuJoCo's or nearer (stall / load RMS: pyri6 0.29 units / 2.7 points, CPU 0.28 / 2.9; feno2 0.21 / 2.9, CPU 0.45 /
# 4.4), both lifts right (x1.15: 0.33 / 3.4, 0.36 / 3.4; x1.5: 0.35 / 2.2, 0.17 / 2.1)
MJLAB_GIVE_SCALE = 1.3


def sim_cfg() -> "SimulationCfg":
    """`robot.arm_spec`'s physics options, as mjlab takes them (its attach drops a spec's own)."""
    from mjlab.sim import MujocoCfg, SimulationCfg

    return SimulationCfg(
        nconmax=NCONMAX,
        njmax=NJMAX,
        mujoco=MujocoCfg(
            timestep=0.002,
            integrator="euler",  # (MuJoCo's default, which the fits were made with)
            cone="elliptic",
            impratio=10.0,
            solver="newton",
            iterations=100,
            tolerance=1e-8,
            ls_iterations=50,
            ls_tolerance=0.01,
            ccd_iterations=35,  # (MuJoCo's; mjlab's 50)
        ),
    )


def joint_targets(model: XleRobotModel, arm: str, pose: Mapping[str, float]) -> dict[str, float]:
    """LeRobot's readings as the simulator's joint angles (rad), as `robot.set_pose` sets them."""
    angles = {j: float(np.radians(pose[f"{j}.pos"])) for j in robot.JOINTS if f"{j}.pos" in pose}
    if "gripper.pos" in pose:
        g = pose["gripper.pos"]
        angles[robot.GRIPPER] = float(
            np.radians(model.opening_deg(arm, g) if arm in model.gripper_maps else g)
        )
    return angles


def _without_options(spec: "mujoco.MjSpec") -> "mujoco.MjSpec":
    """`spec` with MuJoCo's default options (they are `sim_cfg`'s to set: mjlab warns about a spec's own)."""
    import mujoco

    default = mujoco.MjSpec().option
    spec.option.cone, spec.option.impratio = default.cone, default.impratio
    return spec


def robot_spec(model: XleRobotModel, arm: str, give_scale: float = 1.0) -> "mujoco.MjSpec":
    """The arm as the CPU simulator builds it (`robot.arm_spec`), without its options, its fingers' give x `give_scale`."""
    spec = _without_options(robot.arm_spec(model, arm))
    for joint in spec.joints:
        if joint.name.startswith("finger_") and "_give" in joint.name:
            joint.stiffness = joint.stiffness * give_scale
    return spec


def object_spec(o: sc.Cylinder) -> "mujoco.MjSpec":
    """The object as `scene` builds it, its body at the origin (its pose is the entity's initial state)."""
    import mujoco

    spec = mujoco.MjSpec()
    sc._add_object(spec, replace(o, x=0.0, y=0.0), table_z=-o.height / 2)
    return spec


def table_spec(model: XleRobotModel, arm: str, table_z: float) -> "mujoco.MjSpec":
    """`scene_spec`'s table, in a body of its own (an entity needs one)."""
    import mujoco

    built = sc.scene_spec(model, arm, (), table_z, fingers=None)
    table = next(g for g in built.worldbody.geoms if g.name == "table")
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="table_body", pos=list(table.pos))
    body.add_geom(name="table", type=table.type, size=list(table.size), rgba=list(table.rgba))
    return spec


# ---- the task's terms ----------------------------------------------------------------------------------------------


def object_rise(env: "ManagerBasedRlEnv", object_name: str = "object") -> "torch.Tensor":
    """How far the object is above where it started, as a fraction of the lift, capped at 1 (the reward)."""
    obj = env.scene[object_name]
    # (where it started: the start pool's height for each world, `start_pool.reset_from_pool`; else its default)
    start = getattr(env, "felix_object_z0", None)
    if start is None:
        start = obj.data.default_root_state[:, 2]
    return ((obj.data.root_link_pos_w[:, 2] - start) / LIFT_M).clamp(min=0.0, max=1.0)


def object_lifted(env: "ManagerBasedRlEnv", object_name: str = "object") -> "torch.Tensor":
    """1 where the object is up `HELD_FRACTION` of the lift (`refine`'s lifted)."""
    return (object_rise(env, object_name) >= HELD_FRACTION).float()


def object_tipped(env: "ManagerBasedRlEnv", object_name: str = "object") -> "torch.Tensor":
    """Where the object's axis is past `refine.KNOCKED_DEG` from upright (knocked over: the episode ends)."""
    import torch

    quat = env.scene[object_name].data.root_link_quat_w  # (w, x, y, z)
    up_z = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)  # the body z axis's world z
    return up_z < torch.cos(torch.deg2rad(torch.tensor(KNOCKED_DEG, device=quat.device)))


def object_position(env: "ManagerBasedRlEnv", object_name: str = "object") -> "torch.Tensor":
    """The object's position in the root frame (an observation; the policy's estimate, noise added by the group)."""
    return env.scene[object_name].data.root_link_pos_w


def env_cfg(
    model: XleRobotModel,
    arm: str = "right",
    obj: sc.Cylinder | None = None,
    table_z: float = sc.TABLE_Z,
    start: Mapping[str, float] | None = None,
    num_envs: int = 1,
    object_sigma_m: float = OBJECT_SIGMA_M,
    randomize: bool = True,
    give_scale: float = MJLAB_GIVE_SCALE,
    start_pool: str | Path | None = None,
    pool_weights: Mapping[str, float] | None = None,
) -> "ManagerBasedRlEnvCfg":
    """The grasp-and-lift environment: `obj` on the table at `table_z`, the arm from `start` (the module docstring).

    `start_pool` (a `start_pool.StartPool` file): each reset draws a start from it instead (a pool by `pool_weights`,
    default `POOL_WEIGHTS`), the object and the table the pool's; `obj`, `table_z`, `start`, `object_sigma_m` unused.

    `give_scale`: mjlab's correction to the fingers' give (`MJLAB_GIVE_SCALE`); 1.0 for the CPU simulator's model as is
    (`model_differences` then finds none).
    """
    require_package("mjlab", extra="felix_jr_gpu")
    from mjlab.actuator import XmlActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
    from mjlab.envs import ManagerBasedRlEnvCfg, mdp as base_mdp
    from mjlab.envs.mdp import dr
    from mjlab.envs.mdp.actions import JointPositionActionCfg
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from mjlab.managers.termination_manager import TerminationTermCfg
    from mjlab.scene import SceneCfg
    from mjlab.utils.noise import UniformNoiseCfg as Unoise

    if start_pool is not None:
        from .start_pool import StartPool

        pool = StartPool.load(Path(start_pool))
        radius, height, mass = pool.object_size
        table_z = pool.table_z
        obj = sc.Cylinder("object", -0.35, 0.13, radius, height, mass)  # (where: each start's own)
    obj = obj or sc.Cylinder("object", -0.356, 0.132, 0.0265, 0.11, 0.116)  # feno at one of sim3's spots
    start = start or {**REST, "gripper.pos": 30.0}
    entities = {
        "robot": EntityCfg(
            spec_fn=lambda: robot_spec(model, arm, give_scale),
            init_state=EntityCfg.InitialStateCfg(
                joint_pos={**joint_targets(model, arm, start), FINGER_GIVES: 0.0}
            ),
            articulation=EntityArticulationInfoCfg(actuators=(XmlActuatorCfg(target_names_expr=(".*",)),)),
        ),
        "object": EntityCfg(
            spec_fn=lambda: object_spec(obj),
            init_state=EntityCfg.InitialStateCfg(pos=(obj.x, obj.y, table_z + obj.height / 2)),
        ),
        "table": EntityCfg(spec_fn=lambda: table_spec(model, arm, table_z)),
    }
    terms = {
        "joint_pos": ObservationTermCfg(func=base_mdp.joint_pos_rel, noise=Unoise(n_min=-0.005, n_max=0.005)),
        "joint_vel": ObservationTermCfg(func=base_mdp.joint_vel_rel),
        "object_pos": ObservationTermCfg(func=object_position, noise=Unoise(n_min=-0.005, n_max=0.005)),
        "actions": ObservationTermCfg(func=base_mdp.last_action),
    }
    events: dict[str, EventTermCfg] = {
        "reset_scene": EventTermCfg(func=base_mdp.reset_scene_to_default, mode="reset"),
        "reset_object": EventTermCfg(
            func=base_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-object_sigma_m, object_sigma_m),
                    "y": (-object_sigma_m, object_sigma_m),
                },
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("object"),
            },
        ),
    }
    if start_pool is not None:
        from .start_pool import reset_from_pool

        events = {
            "reset_scene": events["reset_scene"],
            "reset_from_pool": EventTermCfg(
                func=reset_from_pool,
                mode="reset",
                params={"path": str(start_pool), "pool_weights": dict(pool_weights or POOL_WEIGHTS)},
            ),
        }
    if randomize:  # (`grasp_check.physics_draws`' ranges, about the fitted values; log-uniform as there)
        events |= {
            "finger_friction": EventTermCfg(
                func=dr.geom_friction,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", geom_names=FINGER_PADS),
                    "operation": "scale",
                    "distribution": "log_uniform",
                    "axes": [0],
                    "ranges": FRICTION_RANGE,
                },
            ),
            "finger_give": EventTermCfg(
                func=dr.joint_stiffness,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", joint_names=FINGER_GIVES),
                    "operation": "scale",
                    "distribution": "log_uniform",
                    "ranges": STIFFNESS_RANGE,
                },
            ),
            "object_mass": EventTermCfg(
                func=dr.body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("object"),
                    "operation": "scale",
                    "distribution": "log_uniform",
                    "ranges": MASS_RANGE,
                },
            ),
        }
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(num_envs=num_envs, env_spacing=0.0, entities=entities),
        observations={
            "actor": ObservationGroupCfg(terms, enable_corruption=True),
            "critic": ObservationGroupCfg(dict(terms), enable_corruption=False),
        },
        actions={
            "joint_pos": JointPositionActionCfg(
                entity_name="robot", actuator_names=(".*",), scale=1.0, use_default_offset=False
            )
        },
        events=events,
        rewards={
            "rise": RewardTermCfg(func=object_rise, weight=1.0),
            "lifted": RewardTermCfg(func=object_lifted, weight=1.0),
            "action_rate": RewardTermCfg(func=base_mdp.action_rate_l2, weight=-0.01),
        },
        terminations={
            "time_out": TerminationTermCfg(func=base_mdp.time_out, time_out=True),
            "tipped": TerminationTermCfg(func=object_tipped),
        },
        sim=sim_cfg(),
        decimation=DECIMATION,
        episode_length_s=EPISODE_S,
    )


# ---- the check that nothing changed on the way ----------------------------------------------------------------------

OPTION_FIELDS = (
    "timestep", "integrator", "cone", "impratio", "solver", "iterations", "tolerance", "ls_iterations",
    "ls_tolerance", "noslip_iterations", "ccd_iterations", "gravity", "disableflags", "enableflags", "jacobian",
)  # fmt: skip
JOINT_FIELDS = (
    "jnt_type",
    "jnt_stiffness",
    "jnt_range",
    "jnt_limited",
    "jnt_solref",
    "jnt_solimp",
    "jnt_margin",
)
DOF_FIELDS = ("dof_frictionloss", "dof_damping", "dof_armature", "dof_solref", "dof_solimp")
ACTUATOR_FIELDS = ("actuator_gainprm", "actuator_biasprm", "actuator_forcerange", "actuator_ctrlrange",
                   "actuator_gear", "actuator_dyntype", "actuator_gaintype", "actuator_biastype")  # fmt: skip
GEOM_FIELDS = ("geom_type", "geom_size", "geom_friction", "geom_condim", "geom_priority", "geom_solref",
               "geom_solimp", "geom_solmix", "geom_margin", "geom_gap", "geom_contype", "geom_conaffinity")  # fmt: skip
BODY_FIELDS = ("body_mass", "body_inertia")


def _strip(name: str) -> str:
    """A name without mjlab's entity prefix ("robot/shoulder_pan" -> "shoulder_pan")."""
    return name.split("/", 1)[1] if "/" in name else name


def _geoms_by_key(m: "mujoco.MjModel") -> dict[tuple[str, int], int]:
    """Each geom by (its body's name, its index among that body's geoms): unnamed geoms matched across models."""
    import mujoco

    out: dict[tuple[str, int], int] = {}
    count: dict[str, int] = {}
    for g in range(m.ngeom):
        body = _strip(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or "")
        if body in ("world", "", "table_body"):
            name = _strip(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
            if name != "table":
                continue  # (the world's own geoms, none but the table between the two scenes)
            body = "table"
        k = count.get(body, 0)
        count[body] = k + 1
        out[(body, k)] = g
    return out


def model_differences(
    cpu: "mujoco.MjModel", env: "mujoco.MjModel", rtol: float = 1e-9, atol: float = 1e-12
) -> list[str]:
    """Every field that differs between the CPU simulator's model and the environment's (names matched): none, if equal.

    The options; per joint (by name) its type, stiffness, range and limits; per degree of freedom its friction loss,
    damping, armature and constraint settings; per actuator its gains, bias, ranges; per geom (by its body and order)
    its shape and contact settings; per body its mass and inertia. Bodies, joints and geoms only one model has, too.
    """
    import mujoco

    out: list[str] = []

    def compare(label: str, a: Any, b: Any) -> None:
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        if a.shape != b.shape or not np.allclose(a, b, rtol=rtol, atol=atol):
            out.append(f"{label}: {a.round(9).tolist()} != {b.round(9).tolist()}")

    for f in OPTION_FIELDS:
        compare(f"opt.{f}", getattr(cpu.opt, f), getattr(env.opt, f))

    def by_name(m: "mujoco.MjModel", kind: Any, n: int) -> dict[str, int]:
        def name(i: int) -> str:
            found = mujoco.mj_id2name(m, kind, i)
            if (
                not found
                and kind == mujoco.mjtObj.mjOBJ_JOINT
                and m.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE
            ):
                return "floating_base_joint"  # (the object's free joint: unnamed in `scene`, named by mjlab)
            return _strip(found or f"#{i}")

        return {name(i): i for i in range(n)}

    for kind, n_attr, fields, extra in (
        (mujoco.mjtObj.mjOBJ_JOINT, "njnt", JOINT_FIELDS, DOF_FIELDS),
        (mujoco.mjtObj.mjOBJ_ACTUATOR, "nu", ACTUATOR_FIELDS, ()),
        (mujoco.mjtObj.mjOBJ_BODY, "nbody", BODY_FIELDS, ()),
    ):
        a, b = by_name(cpu, kind, getattr(cpu, n_attr)), by_name(env, kind, getattr(env, n_attr))
        a.pop("world", None), b.pop("world", None)
        b = {k: v for k, v in b.items() if not k.startswith("mocap") and k not in ("table_body",)}
        for name in sorted(set(a) ^ set(b)):
            if kind == mujoco.mjtObj.mjOBJ_BODY and name in ("robot", "table", ""):
                continue  # (mjlab's mocap wrappers)
            out.append(
                f"{kind.name.lower()} {name}: only in {'the CPU model' if name in a else 'the environment'}"
            )
        for name in sorted(set(a) & set(b)):
            for f in fields:
                compare(f"{name}.{f}", getattr(cpu, f)[a[name]], getattr(env, f)[b[name]])
            for f in extra:  # (per degree of freedom: a joint's first)
                compare(
                    f"{name}.{f}",
                    getattr(cpu, f)[cpu.jnt_dofadr[a[name]]],
                    getattr(env, f)[env.jnt_dofadr[b[name]]],
                )
    ga, gb = _geoms_by_key(cpu), _geoms_by_key(env)
    for key in sorted(set(ga) ^ set(gb)):
        out.append(f"geom {key}: only in {'the CPU model' if key in ga else 'the environment'}")
    for key in sorted(set(ga) & set(gb)):
        for f in GEOM_FIELDS:
            compare(f"geom {key}.{f}", getattr(cpu, f)[ga[key]], getattr(env, f)[gb[key]])
    return out


def _joint_slices(m: "mujoco.MjModel") -> dict[str, slice]:
    """Each joint's qpos slice, by name (the object's free joint as "floating_base_joint", as `model_differences`)."""
    import mujoco

    widths = {
        mujoco.mjtJoint.mjJNT_FREE: 7,
        mujoco.mjtJoint.mjJNT_BALL: 4,
        mujoco.mjtJoint.mjJNT_SLIDE: 1,
        mujoco.mjtJoint.mjJNT_HINGE: 1,
    }
    out = {}
    for j in range(m.njnt):
        found = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        name = _strip(found) if found else "floating_base_joint"
        adr = int(m.jnt_qposadr[j])
        out[name] = slice(adr, adr + widths[mujoco.mjtJoint(int(m.jnt_type[j]))])
    return out


def map_qpos(qpos: np.ndarray, src: "mujoco.MjModel", dst: "mujoco.MjModel", into: np.ndarray) -> np.ndarray:
    """`src`'s `qpos` written into `dst`'s layout (`into`, kept where `src` has no such joint), by joint name."""
    a, b = _joint_slices(src), _joint_slices(dst)
    out = np.array(into, dtype=np.float64)
    for name in set(a) & set(b):
        out[b[name]] = qpos[a[name]]
    return out


def map_ctrl(ctrl: np.ndarray, src: "mujoco.MjModel", dst: "mujoco.MjModel") -> np.ndarray:
    """`src`'s actuator commands in `dst`'s actuator order, by name (`ctrl`: (..., src.nu))."""
    import mujoco

    order = [
        mujoco.mj_name2id(
            src, mujoco.mjtObj.mjOBJ_ACTUATOR, _strip(mujoco.mj_id2name(dst, mujoco.mjtObj.mjOBJ_ACTUATOR, i))
        )
        for i in range(dst.nu)
    ]
    return np.asarray(ctrl)[..., order]


def pose_differences(
    cpu: "mujoco.MjModel", env: "mujoco.MjModel", qpos: np.ndarray, atol_m: float = 1e-9
) -> list[str]:
    """Bodies whose world position differs between the two models at the CPU model's `qpos` (the kinematics' check)."""
    import mujoco

    dc, de = mujoco.MjData(cpu), mujoco.MjData(env)
    dc.qpos[:] = qpos
    de.qpos[:] = map_qpos(qpos, cpu, env, de.qpos)
    mujoco.mj_forward(cpu, dc)
    mujoco.mj_forward(env, de)
    out = []
    for b in range(1, cpu.nbody):
        name = mujoco.mj_id2name(cpu, mujoco.mjtObj.mjOBJ_BODY, b)
        match = [
            i
            for i in range(env.nbody)
            if _strip(mujoco.mj_id2name(env, mujoco.mjtObj.mjOBJ_BODY, i) or "") == name
        ]
        if not match:
            continue
        gap = float(np.linalg.norm(dc.xpos[b] - de.xpos[match[0]]))
        if gap > atol_m:
            out.append(f"body {name}: {1000 * gap:.6f} mm apart")
    return out


class Recording:
    """A CPU simulator's run, step by step: its start (`qpos`) and each physics step's actuator commands (`ctrl`).

    `record(module)`: within it, every `mujoco.mj_step` the module calls (`shadow`, `grip_sim`) is recorded, the
    model's and the data's of the first call kept (the scene stepped).
    """

    def __init__(self) -> None:
        """Empty: `record` fills it."""
        self.m: mujoco.MjModel | None = None
        self.qpos: np.ndarray | None = None
        self.ctrl: list[np.ndarray] = []
        self.qpos_every: list[np.ndarray] = []  # (every `DECIMATION` steps, after the step)

    def record(self, module: Any) -> Any:
        """A context manager recording `module`'s `mujoco.mj_step` calls."""
        import contextlib

        import mujoco

        recording = self

        class _Mujoco:
            def __getattr__(self, name: str) -> Any:
                return getattr(mujoco, name)

            @staticmethod
            def mj_step(m: "mujoco.MjModel", d: "mujoco.MjData", nstep: int = 1) -> None:
                for _ in range(nstep):
                    if recording.m is None:
                        recording.m, recording.qpos = m, d.qpos.copy()
                    recording.ctrl.append(d.ctrl.copy())
                    mujoco.mj_step(m, d)
                    if len(recording.ctrl) % DECIMATION == 0:
                        recording.qpos_every.append(d.qpos.copy())

        @contextlib.contextmanager
        def patched() -> Any:
            old = module.mujoco
            module.mujoco = _Mujoco()
            try:
                yield recording
            finally:
                module.mujoco = old

        return patched()


def replay_in_env(
    env: "ManagerBasedRlEnv", recordings: Sequence[Recording], probes: Sequence[Sequence[int]] | None = None
) -> list[dict[str, Any]]:
    """Each recording played open loop in its own world of `env` (its start, then its commands step by step).

    Per recording: the most the arm's joints (deg) and the object (mm) came apart from the CPU run's, checked every
    `DECIMATION` steps, and the object's rise (m) at the end in each. A recording's world holds its last command after
    its end. `env.num_envs` must be at least `len(recordings)`. `probes`: per recording, after which steps to keep its
    world's state (`row["probes"]`: `{step: (qpos, actuator_force)}`, the environment's layouts).
    """
    import torch

    sim = env.sim
    em = sim.mj_model
    n = len(recordings)
    assert env.num_envs >= n
    qpos = sim.data.qpos.cpu().numpy().copy()
    for w, r in enumerate(recordings):
        assert r.m is not None and r.qpos is not None
        qpos[w] = map_qpos(r.qpos, r.m, em, qpos[w])
    sim.data.qpos[:] = torch.as_tensor(qpos, device=sim.data.qpos.device, dtype=sim.data.qpos.dtype)
    sim.data.qvel[:] = 0.0
    sim.forward()
    steps = max(len(r.ctrl) for r in recordings)
    ctrls = [map_ctrl(np.array(r.ctrl), r.m, em) for r in recordings]
    slices = [_joint_slices(r.m) for r in recordings]
    eslices = _joint_slices(em)
    arm = [*robot.JOINTS, robot.GRIPPER]
    worst = [{"arm_deg": 0.0, "object_mm": 0.0} for _ in range(n)]
    ends: list[np.ndarray | None] = [None] * n
    ctrl_now = sim.data.ctrl.cpu().numpy().copy()
    wanted = [set(p) for p in probes] if probes is not None else [set() for _ in range(n)]
    kept: list[dict[int, tuple[np.ndarray, np.ndarray]]] = [{} for _ in range(n)]
    for k in range(steps):
        for w in range(n):
            ctrl_now[w] = ctrls[w][min(k, len(ctrls[w]) - 1)]
        sim.data.ctrl[:] = torch.as_tensor(ctrl_now, device=sim.data.ctrl.device, dtype=sim.data.ctrl.dtype)
        sim.step()
        if probes is not None:
            for w in range(n):
                if k + 1 in wanted[w]:
                    kept[w][k + 1] = (
                        sim.data.qpos[w].cpu().numpy().copy(),
                        sim.data.actuator_force[w].cpu().numpy().copy(),
                    )
        if (k + 1) % DECIMATION == 0 or k + 1 == steps:
            now = sim.data.qpos.cpu().numpy()
            for w, r in enumerate(recordings):
                i = (k + 1) // DECIMATION - 1
                if (k + 1) % DECIMATION == 0 and i < len(r.qpos_every):
                    cpu_q, s, e = r.qpos_every[i], slices[w], eslices
                    gap = max(abs(cpu_q[s[j]][0] - now[w][e[j]][0]) for j in arm if j in s and j in e)
                    worst[w]["arm_deg"] = max(worst[w]["arm_deg"], float(np.degrees(gap)))
                    if "floating_base_joint" in s:
                        d = np.linalg.norm(
                            cpu_q[s["floating_base_joint"]][:3] - now[w][e["floating_base_joint"]][:3]
                        )
                        worst[w]["object_mm"] = max(worst[w]["object_mm"], 1000 * float(d))
                if k + 1 >= len(r.ctrl) and ends[w] is None:
                    ends[w] = now[w].copy()
    out = []
    for w, r in enumerate(recordings):
        s, e = slices[w], eslices
        row: dict[str, Any] = dict(worst[w])
        if probes is not None:
            row["probes"] = kept[w]
        if "floating_base_joint" in s and r.qpos_every:
            z0 = r.qpos[s["floating_base_joint"]][2]
            row["cpu_rise_m"] = float(r.qpos_every[-1][s["floating_base_joint"]][2] - z0)
            row["env_rise_m"] = float(ends[w][e["floating_base_joint"]][2] - z0)  # type: ignore[index]
        out.append(row)
    return out


def set_world_physics(
    env: "ManagerBasedRlEnv", world: int, cpu: "mujoco.MjModel", give_scale: float = MJLAB_GIVE_SCALE
) -> None:
    """World `world`'s geoms' friction and joints' stiffness as `cpu`'s (a model built under a physics draw).

    The fingers' give x `give_scale` (mjlab's correction, as `env_cfg`).

    Those fields must be per world in `env` (its config randomizes them: `env_cfg(randomize=True)`).
    """
    import torch

    em, model = env.sim.mj_model, env.sim.model
    ga, gb = _geoms_by_key(cpu), _geoms_by_key(em)
    for key in set(ga) & set(gb):
        model.geom_friction[world, gb[key]] = torch.as_tensor(
            cpu.geom_friction[ga[key]], dtype=model.geom_friction.dtype
        )
    a, b = _joint_slices(cpu), _joint_slices(em)
    import mujoco

    for name in set(a) & set(b):
        ja = mujoco.mj_name2id(cpu, mujoco.mjtObj.mjOBJ_JOINT, name)
        jb = [
            j
            for j in range(em.njnt)
            if _strip(mujoco.mj_id2name(em, mujoco.mjtObj.mjOBJ_JOINT, j) or "") == name
        ]
        if ja >= 0 and jb:
            scale = give_scale if name.startswith("finger_") and "_give" in name else 1.0
            model.jnt_stiffness[world, jb[0]] = float(cpu.jnt_stiffness[ja]) * scale


def cpu_model(
    model: XleRobotModel, arm: str = "right", obj: sc.Cylinder | None = None, table_z: float = sc.TABLE_Z
) -> "mujoco.MjModel":
    """The CPU simulator's compiled model of the same scene (`scene.build`)."""
    obj = obj or sc.Cylinder("object", -0.356, 0.132, 0.0265, 0.11, 0.116)
    return sc.build(model, arm, [obj], table_z=table_z).m


def make_env(cfg: "ManagerBasedRlEnvCfg", device: str = "cpu") -> "ManagerBasedRlEnv":
    """The environment, on `device` ("cpu": Warp's CPU device, slow, for checks; "cuda:0" on a GPU)."""
    from mjlab.envs import ManagerBasedRlEnv

    return ManagerBasedRlEnv(cfg, device=device)


def main(argv: Sequence[str] | None = None) -> None:
    """Build the environment on the CPU and print `model_differences` against the CPU simulator's model."""
    from lerobot.felix_jr.harness import api

    model = XleRobotModel.load(api.MODEL)
    env = make_env(env_cfg(model, randomize=False, give_scale=1.0))
    diffs = model_differences(cpu_model(model), env.sim.mj_model)
    print("\n".join(diffs) if diffs else "the environment's model equals the CPU simulator's")


if __name__ == "__main__":
    main()
