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
- **The task** (`mjlab_task`): delta joint targets, the privileged observations, the staged rewards, the terminations.

`model_differences` compares this environment's compiled model with `scene.build`'s, field by field: the check that
nothing changed on the way (`docs/sim_deploy.md`).
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.felix_jr.calibration.joint_sim import REST
from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel
from lerobot.felix_jr.harness.primitives import ObjectRecord
from lerobot.utils.import_utils import require_package

from . import robot, scene as sc
from .grasp_check import FRICTION_RANGE, MASS_RANGE, STIFFNESS_RANGE
from .mjlab_object import SIZE_ERROR_M

if TYPE_CHECKING:
    import mujoco
    from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
    from mjlab.sim import SimulationCfg

DECIMATION = (
    17  # 17 x 2 ms: ~30 Hz, the server's waypoint rate (`trajectory.WAYPOINT_HZ`; the shadow's per tick)
)
NCONMAX, NJMAX = 64, 400  # per world (`cloud/warp_check.py`: ~1 MB a world)
EPISODE_S = 12.0
OBJECT_NOISE_M = 0.001  # the object pose's noise in the actor's observations: the positive control's, ~1 mm
REWARD_WEIGHTS = {  # (per step; a starting point, to tune from the curves)
    "reach": 0.2,
    "between_jaws": 0.3,
    "both_fingers": 0.5,
    "rise": 1.0,
    "held": 2.0,
    "action_rate": -0.01,
    "swing": -0.2,  # (the object's angular speed while lifted, rad/s: a bottle swinging at ~2 rad/s costs a fifth of held)
    "failed": -5.0,  # (an episode ended by a failure: the arm on the table, the object tipped)
}
# the start pool's mix (`start_pool`): uniform, as OmniReset samples its resets (no curriculum; 2026-11-21: the first
# runs' mix, attack 0.2 / at_object 0.5 / in_jaws 0.2 / lifted 0.1, had the easy pools solved in 20 iterations)
POOL_WEIGHTS = {"attack": 0.25, "at_object": 0.25, "in_jaws": 0.25, "lifted": 0.25}
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


def table_spec(model: XleRobotModel, arm: str, table_z: float, appearance: bool = False) -> "mujoco.MjSpec":
    """`scene_spec`'s table, in a body of its own (an entity needs one).

    `appearance`: with the room around it and the materials its look is randomized from (`appearance.room_spec`:
    drawn only, the physics unchanged).
    """
    import mujoco

    built = sc.scene_spec(model, arm, (), table_z, fingers=None)
    table = next(g for g in built.worldbody.geoms if g.name == "table")
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="table_body", pos=list(table.pos))
    body.add_geom(name="table", type=table.type, size=list(table.size), rgba=list(table.rgba))
    if appearance:
        from . import appearance as look

        look.room_spec(spec, body, tuple(2 * float(v) for v in table.size))
    return spec


# ---- the task's terms ----------------------------------------------------------------------------------------------


def env_cfg(
    model: XleRobotModel,
    arm: str = "right",
    body: ObjectRecord | None = None,
    table_z: float = sc.TABLE_Z,
    start: Mapping[str, float] | None = None,
    num_envs: int = 1,
    object_sigma_m: float = OBJECT_SIGMA_M,
    randomize: bool = True,
    give_scale: float = MJLAB_GIVE_SCALE,
    start_pool: str | Path | None = None,
    pool_weights: Mapping[str, float] | None = None,
    object_noise_m: float = OBJECT_NOISE_M,
    size_error_m: float = SIZE_ERROR_M,
    actor_exclude: Sequence[str] = (),
    object_pose: str = "live",
    estimate_noise_m: float = 0.003,
    grasp_point_fit: str = "side",
    appearance: bool = False,
) -> "ManagerBasedRlEnvCfg":
    """The grasp-and-lift environment: `body` on the table at `table_z`, the arm from `start` (the module docstring).

    `start_pool` (a `start_pool.StartPool` file): each reset draws a start from it instead (a pool by `pool_weights`,
    default `POOL_WEIGHTS`), the object and the table the pool's; `body`, `table_z`, `start`, `object_sigma_m` unused.
    `body`: the object, standing where it stands (`harness.primitives.ObjectRecord`; default
    `mjlab_object.default_object`).

    `give_scale`: mjlab's correction to the fingers' give (`MJLAB_GIVE_SCALE`); 1.0 for the CPU simulator's model as is
    (`model_differences` then finds none). The task's action, observations, rewards and terminations: `mjlab_task`;
    `object_noise_m` the object pose's noise in the actor's observations (std, m; the axis's a tenth, unitless).
    `object_pose`: the object's pose the actor is told: "live", or "estimated" (at the episode's reset, as perception
    would: the start's perception offset and `estimate_noise_m` of noise; then frozen: `MjlabCylinder.draw_estimate`).
    `actor_exclude`: observation terms the actor goes without (privileged ones dropped: the critic keeps them).
    `appearance`: the scene's look randomized at each reset for the wrist camera (`appearance`: the table's, floor's and
    walls' textures, the lights, the object's colour; a room around the table, drawn only).
    `grasp_point_fit`: where a grasp holds the object, by radius (`mjlab_object.GRASP_POINT_FITS`: "side" for side
    grasps, multi1's; "steep" for grasps up to 60 deg from above, multi2's): the reach reward's target.
    `size_error_m`: for a pool whose objects vary in size, the error in the radius and height the actor is told, drawn
    at each reset and kept for the episode (std, m; with `randomize` only).
    """
    require_package("mjlab", extra="felix_jr_gpu")
    from mjlab.actuator import XmlActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
    from mjlab.envs import ManagerBasedRlEnvCfg, mdp as base_mdp
    from mjlab.envs.mdp import dr
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from mjlab.managers.termination_manager import TerminationTermCfg
    from mjlab.scene import SceneCfg
    from mjlab.utils.noise import GaussianNoiseCfg, UniformNoiseCfg as Unoise

    from . import mjlab_object as obj, mjlab_task as task

    if start_pool is not None:
        from .start_pool import StartPool

        pool = StartPool.load(Path(start_pool))
        sized = pool.sizes_vary  # (the object resized per world at each reset: `pool_reset.PoolReset`)
        table_z = pool.table_z
        # the object the environment is built with: the first start's (where it stands: each start's own state)
        body = pool.record(0).placed_upright(np.array([-0.35, 0.13, table_z]))
    else:
        sized = False
    body = body or obj.default_object(table_z)
    # the object: everything about it that knows its shape (construction, contacts, reset, observations, geometry)
    cyl = obj.MjlabCylinder(
        body,
        sized=sized,
        mass_range=MASS_RANGE if randomize and sized else (1.0, 1.0),
        size_error_m=size_error_m if randomize and sized else 0.0,
        pose_estimate=object_pose == "estimated",
        estimate_noise_m=estimate_noise_m,
        grasp_point_fit=grasp_point_fit,
    )
    if grasp_point_fit not in obj.GRASP_POINT_FITS:
        raise ValueError(f"grasp_point_fit: one of {sorted(obj.GRASP_POINT_FITS)}, not {grasp_point_fit!r}")
    if object_pose not in ("live", "estimated"):
        raise ValueError(f"object_pose: live or estimated, not {object_pose!r}")
    start = start or {**REST, "gripper.pos": 30.0}
    entities = {
        "robot": EntityCfg(
            spec_fn=lambda: robot_spec(model, arm, give_scale),
            init_state=EntityCfg.InitialStateCfg(
                joint_pos={**joint_targets(model, arm, start), FINGER_GIVES: 0.0}
            ),
            articulation=EntityArticulationInfoCfg(actuators=(XmlActuatorCfg(target_names_expr=(".*",)),)),
        ),
        cyl.entity: cyl.entity_cfg(),
        "table": EntityCfg(spec_fn=lambda: table_spec(model, arm, table_z, appearance)),
    }
    pose_noise = (
        GaussianNoiseCfg(
            std=tuple(
                [object_noise_m] * 3 + [object_noise_m / 10] * 3
            )  # (the center's m; the axis's, unitless)
        )
        if object_noise_m > 0
        else None
    )

    def group(object_terms: dict[str, Any]) -> dict[str, Any]:
        # (in the order checkpoints were trained with: the object's pose after the joints, its size, if any, last)
        size = {"object_size": object_terms["object_size"]} if "object_size" in object_terms else {}
        return {
            "joint_pos": ObservationTermCfg(
                func=base_mdp.joint_pos_rel, noise=Unoise(n_min=-0.002, n_max=0.002)
            ),
            "joint_vel": ObservationTermCfg(func=base_mdp.joint_vel_rel),
            "object_in_root": object_terms["object_in_root"],
            "object_in_pinch": object_terms["object_in_pinch"],
            "servo_loads": ObservationTermCfg(func=task.servo_loads),
            "finger_contacts": ObservationTermCfg(func=task.finger_contacts),
            "actions": ObservationTermCfg(func=base_mdp.last_action),
            **size,
        }

    actor_object, critic_object = cyl.observation_terms(
        pose_noise
    )  # (the actor's size off by the episode's error)
    terms, critic_terms = group(actor_object), group(critic_object)
    unknown = set(actor_exclude) - set(terms)
    if unknown:
        raise ValueError(f"no actor observation {sorted(unknown)} to leave out (there are: {sorted(terms)})")
    terms = {k: v for k, v in terms.items() if k not in actor_exclude}  # (the critic keeps them: privileged)
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
                "asset_cfg": SceneEntityCfg(cyl.entity),
            },
        ),
    }
    if start_pool is not None:
        from .pool_reset import TERM, PoolReset

        # the pool's starts written at each reset; for objects of several sizes each world's object resized too (the
        # object's `write`, its fields made per world: its `declaration`, and why `set_const` after)
        events = {
            "reset_scene": events["reset_scene"],
            TERM: EventTermCfg(
                func=PoolReset,
                mode="reset",
                params={
                    "path": str(start_pool),
                    "pool_weights": dict(pool_weights or POOL_WEIGHTS),
                    "robot": SceneEntityCfg("robot"),
                    "object": cyl,
                },
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
            **(
                {}
                if sized
                else {
                    "object_mass": EventTermCfg(
                        func=dr.body_mass,
                        mode="startup",
                        params={
                            "asset_cfg": SceneEntityCfg(cyl.entity),
                            "operation": "scale",
                            "distribution": "log_uniform",
                            "ranges": MASS_RANGE,
                        },
                    )
                }
            ),  # (a sized pool's mass drawn at each reset instead: the object's `write`)
        }
    if appearance:
        from . import appearance as look

        events.update(look.events())
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            num_envs=num_envs, env_spacing=0.0, entities=entities, sensors=task.sensors(cyl.contact_match())
        ),
        observations={
            "actor": ObservationGroupCfg(terms, enable_corruption=True),
            "critic": ObservationGroupCfg(critic_terms, enable_corruption=False),
        },
        actions={"joint_targets": task.delta_action_cfg(model, arm)},
        events=events,
        rewards={  # (staged, each stage paying more than the one before: `mjlab_task`'s docstring)
            "reach": RewardTermCfg(func=task.reach, weight=REWARD_WEIGHTS["reach"], params={"obj": cyl}),
            "between_jaws": RewardTermCfg(
                func=task.enclosed, weight=REWARD_WEIGHTS["between_jaws"], params={"obj": cyl}
            ),
            "both_fingers": RewardTermCfg(func=task.both_fingers, weight=REWARD_WEIGHTS["both_fingers"]),
            "rise": RewardTermCfg(func=task.rise, weight=REWARD_WEIGHTS["rise"], params={"obj": cyl}),
            "held": RewardTermCfg(func=task.held, weight=REWARD_WEIGHTS["held"], params={"obj": cyl}),
            "action_rate": RewardTermCfg(func=base_mdp.action_rate_l2, weight=REWARD_WEIGHTS["action_rate"]),
            "swing": RewardTermCfg(func=task.swing, weight=REWARD_WEIGHTS["swing"], params={"obj": cyl}),
            "failed": RewardTermCfg(func=base_mdp.is_terminated, weight=REWARD_WEIGHTS["failed"]),
        },
        terminations={
            "time_out": TerminationTermCfg(func=base_mdp.time_out, time_out=True),
            "tipped": TerminationTermCfg(func=task.tipped, params={"obj": cyl}),
            "arm_on_table": TerminationTermCfg(func=task.arm_on_table),
            # (a world whose physics went NaN or infinite, reset before its observation is computed; a truncation, not
            # the policy's failure: multi2's small, light objects diverged once in 42 M steps, 2026-11-25)
            "nan": TerminationTermCfg(func=base_mdp.nan_detection, time_out=True),
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
    model: XleRobotModel, arm: str = "right", body: ObjectRecord | None = None, table_z: float = sc.TABLE_Z
) -> "mujoco.MjModel":
    """The CPU simulator's compiled model of the same scene (`scene.build`; default `mjlab_object.default_object`)."""
    from .mjlab_object import default_object

    return sc.build(model, arm, [sc.from_record(body or default_object(table_z))], table_z=table_z).m


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
