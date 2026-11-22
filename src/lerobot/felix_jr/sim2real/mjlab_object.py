"""The object in the mjlab environment: everything about it that knows its shape (an upright cylinder), in one place.

What the object *is* (its shape's parameters, bounds, inertia, its record in the bank and the pools) is
`harness.primitives` (`Cylinder`, `RigidBody`); this module is how that object lives in mjlab, by stage:

- **Construction** (`entity_cfg`): the mjlab entity for a `RigidBody`: its spec (`object_spec`: a body at the origin,
  one cylinder geom, a free joint; `scene._add_object`, the CPU simulator's own builder), its pose the initial state.
- **Reset** (`object_ids`, `sized_declaration`, `write_bodies`): for objects of several sizes, the model fields a
  world's object needs rewritten (`SIZED_FIELDS`), declared per world, and written from `RigidBody.vector` rows: the
  cylinder's size, its bounds (`Cylinder.bounding_radius`, `half_extents`), its mass and inertia
  (`Cylinder.solid_inertia`). `pool_reset.PoolReset` calls these; it draws the starts, which know nothing of shape.
- **Observations** (`object_pose`, `object_in_root`, `object_in_pinch`, `object_dims`, `object_size`): its center and
  axis (a cylinder's pose up to its own spin, which nothing sees), in the root and the pinch frames; its radius and
  height (read from the model: each world's own once resized), what the actor is told of them (off by an episode's
  error, `SIZE_ERROR_M`).
- **Task geometry** (`axis_from_pinch`, `grasp_point`, `between_jaws`, `rest_z`, `tilt_cos`, `upright`): where its axis
  passes the pinch point, where a grasp puts it (fit against the radius), the box between the jaws, its height standing
  on the table, how far it tilts from upright. `mjlab_task` builds the rewards and terminations from these (`reach`,
  `rise`, `held`, `tipped`); what knows the task stays there, what knows the shape is here.

A second shape would mean filling in this module and `harness.primitives` (and the bank's grasp planning).
"""

from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.felix_jr.harness.primitives import RigidBody

from . import scene as sc

if TYPE_CHECKING:
    import mujoco
    import torch
    from mjlab.entity import EntityCfg
    from mjlab.envs import ManagerBasedRlEnv

ENTITY = "object"  # its entity's name in the scene
# the model fields a world's object needs rewritten when sizes vary: the cylinder's size, its bounds, its mass, inertia
SIZED_FIELDS = ("geom_size", "geom_rbound", "geom_aabb", "body_mass", "body_inertia")
# the size the actor is told, off by an error drawn at each reset and kept for the episode (std, m): what a size
# estimated from a camera is off by (a bias, not noise that averages away); the critic told the true size
SIZE_ERROR_M = 0.002

# where the object's axis passes the pinch point in a grasp, in the pinch frame (x across the jaws, z along the gripper):
# measured over the pyri pool's starts (2026-11-21; the pinch to the axis line, the grasp's height along the axis apart):
# in the jaws (-16, 1, -11) mm (5th-95th: x -18 to -15, y +-7, z -18 to +2); the jaws open around it (-30, 2, -12) (x -36
# to -24); at the attack (-30, -1, +34) (z +20 to +52). For objects of several sizes, refit over the multi1 pool's
# starts (radius 20-30 mm; 2026-11-21): across the jaws the axis sits about a radius from the pinch point, in the jaws
# x = -0.79 r + 2.0 mm (residual 1.0 mm; at pyri's 22.5 mm -15.8, as measured), z = +0.67 r - 24.1 (residual 6.7); the
# jaws open around it x = -1.10 r - 4.9 (residual 3.2). Each axis (per metre of radius, at no radius; m):
GRASP_POINT_FIT = ((-0.79, 0.0020), (0.0, 0.001), (0.67, -0.0241))
# the box "between the jaws", from the open jaws' starts to the closed ones': across the jaws its ends follow the radius
# as those do (at pyri's radius -38 and -12 mm, as measured); y and z fixed
BETWEEN_X_FIT = ((-1.10, -0.0049 - 0.0084), (-0.79, 0.0020 + 0.0038))  # low, high
BETWEEN_LOW_M = (None, -0.015, -0.025)
BETWEEN_HIGH_M = (None, 0.015, 0.006)
ON_BODY_MARGIN_M = 0.005  # the grasp's height this far inside the object's ends at least


# ---- construction ---------------------------------------------------------------------------------------------------


def default_body(table_z: float) -> RigidBody:
    """The object with no pool: feno (53 x 110 mm, 116 g), standing on the table at one of sim3's spots."""
    return RigidBody.upright(
        "cylinder", {"radius": 0.0265, "height": 0.11}, 0.116, np.array([-0.356, 0.132, table_z])
    )


def object_spec(body: RigidBody) -> "mujoco.MjSpec":
    """The object's spec: a body at the origin (its pose is the entity's initial state), as `scene` builds objects."""
    import mujoco

    params = body.shape.shape_params()
    spec = mujoco.MjSpec()
    obj = sc.Cylinder(ENTITY, 0.0, 0.0, params["radius"], params["height"], body.mass_kg)
    sc._add_object(spec, obj, table_z=-params["height"] / 2)  # (the body's origin at the cylinder's center)
    return spec


def entity_cfg(body: RigidBody) -> "EntityCfg":
    """The mjlab entity for `body`, standing where it stands (its center its initial position)."""
    from mjlab.entity import EntityCfg

    center = body.shape.bottom_root + body.shape.root_from_obj[:3, :3] @ body.shape.center_obj
    return EntityCfg(
        spec_fn=lambda: object_spec(body),
        init_state=EntityCfg.InitialStateCfg(pos=tuple(float(v) for v in center)),
    )


# ---- reset ----------------------------------------------------------------------------------------------------------


def object_ids(env: "ManagerBasedRlEnv", entity: str = ENTITY) -> tuple[int, int]:
    """The object entity's body and its one geom (a cylinder), as indices into the model's fields."""
    import mujoco

    obj = env.scene[entity]
    bodies, geoms = obj.indexing.body_ids.tolist(), obj.indexing.geom_ids.tolist()
    if len(bodies) != 1 or len(geoms) != 1:
        raise ValueError(
            f"expected the object to be one body with one geom, found bodies {bodies}, geoms {geoms}"
        )
    if env.sim.mj_model.geom_type[geoms[0]] != mujoco.mjtGeom.mjGEOM_CYLINDER:
        raise ValueError("expected the object's geom to be a cylinder")
    return int(bodies[0]), int(geoms[0])


def sized_declaration() -> tuple[tuple[str, ...], Any]:
    """What an event term that resizes the object declares to mjlab: (its `model_fields`, its `recompute`).

    MuJoCo Warp keeps one compiled model for every world; each model field is stored with a world dimension, one copy
    by default (every world reads it) or one a world once expanded. mjlab expands the fields an event term names in
    `model_fields` (gathered by its event manager right after building each term; `sim.expand_model_fields` then
    allocates them before the first step, and re-captures its CUDA graphs since the arrays move). Unexpanded, a write to
    `model.geom_size[world]` would land on the one shared copy. The fields (`SIZED_FIELDS`): the cylinder's size
    (radius, half height), its bounding sphere and box (`geom_rbound`, `geom_aabb`: what the collision broadphase reads,
    not `geom_size`; stale, contacts are missed or phantom), its body's mass and inertia. `requires_model_fields` adds
    the fields derived from those (`body_subtreemass`, `body_invweight0`, `dof_invweight0`, ...: per world too).
    `recompute`: after the reset events fire, mjlab recomputes those derived constants (`sim.recompute_constants`,
    MuJoCo Warp's `set_const`, which saves and restores the simulation state): the object's mass and inertia change its
    subtree mass and the inverse weights the constraint solver scales its contacts by.
    """
    from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields

    def declared() -> None:  # (only to carry the declaration, its derived fields included; never called)
        pass

    fields = requires_model_fields(*SIZED_FIELDS, recompute=RecomputeLevel.set_const)(declared).model_fields
    return tuple(fields), RecomputeLevel.set_const


def write_bodies(
    env: "ManagerBasedRlEnv",
    env_ids: "torch.Tensor",
    rows: "torch.Tensor",
    mass_scale: "torch.Tensor | None" = None,
    entity: str = ENTITY,
) -> None:
    """The object in each world of `env_ids` made the body of its row (`RigidBody.vector`: radius, height, mass).

    Its geom's size (radius, half height), its bounds (`Cylinder.bounding_radius`, `half_extents`, batched), its body's
    mass (times `mass_scale`, if given) and a solid cylinder's inertia (`Cylinder.solid_inertia`, batched). The fields
    must be per world (`sized_declaration`); mjlab recomputes the constants after.
    """
    import torch

    m = env.sim.model
    body, geom = object_ids(env, entity)
    r, h, mass = rows[:, 0], rows[:, 1], rows[:, 2]
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


# ---- observations ---------------------------------------------------------------------------------------------------


def rotate(quat: "torch.Tensor", v: "torch.Tensor") -> "torch.Tensor":
    """`v` rotated by `quat` (w, x, y, z), batched."""
    import torch

    w, xyz = quat[:, :1], quat[:, 1:]
    t = 2.0 * torch.cross(xyz, v, dim=1)
    return v + w * t + torch.cross(xyz, t, dim=1)


def rotate_inverse(quat: "torch.Tensor", v: "torch.Tensor") -> "torch.Tensor":
    """`v` rotated by the inverse of `quat` (w, x, y, z), batched."""
    import torch

    w, xyz = quat[:, :1], quat[:, 1:]
    t = 2.0 * torch.cross(-xyz, v, dim=1)
    return v + w * t + torch.cross(-xyz, t, dim=1)


def object_pose(env: "ManagerBasedRlEnv") -> tuple["torch.Tensor", "torch.Tensor"]:
    """The object's center and axis (its body z), world frame (= the root frame here)."""
    import torch

    data = env.scene[ENTITY].data
    z = torch.zeros_like(data.root_link_pos_w)
    z[:, 2] = 1.0
    return data.root_link_pos_w, rotate(data.root_link_quat_w, z)


def object_in_root(env: "ManagerBasedRlEnv") -> "torch.Tensor":
    """The object's center (3) and axis (3) in the root frame: an observation."""
    import torch

    center, axis = object_pose(env)
    return torch.cat([center, axis], dim=1)


def object_in_pinch(env: "ManagerBasedRlEnv") -> "torch.Tensor":
    """The object's center (3) and axis (3) in the pinch frame (`mjlab_task.pinch_pose`): an observation."""
    import torch

    from .mjlab_task import pinch_pose

    center, axis = object_pose(env)
    pos, quat = pinch_pose(env)
    return torch.cat([rotate_inverse(quat, center - pos), rotate_inverse(quat, axis)], dim=1)


def object_dims(env: "ManagerBasedRlEnv", entity: str = ENTITY) -> "torch.Tensor":
    """Each world's object's radius and height ((N, 2), m), read from the model.

    The cylinder's size field: one for every world, or each world's own once objects of several sizes have resized it.
    """
    geom = int(env.scene[entity].indexing.geom_ids[0])
    size = env.sim.model.geom_size[..., geom, :2].reshape(-1, 2).float()
    dims = size * size.new_tensor([1.0, 2.0])  # (radius, half height): radius, height
    return dims.expand(env.num_envs, 2) if len(dims) == 1 else dims


def object_size(env: "ManagerBasedRlEnv", estimated: bool = False) -> "torch.Tensor":
    """Each world's object's radius and height (m, `object_dims`): an observation, for objects of several sizes.

    `estimated`: off by the episode's error (`pool_reset.PoolReset.size_error`: the actor's view).
    """
    from .pool_reset import pool_reset

    dims = object_dims(env)
    term = pool_reset(env) if estimated else None
    return dims if term is None else dims + term.size_error


# ---- task geometry --------------------------------------------------------------------------------------------------


def axis_from_pinch(env: "ManagerBasedRlEnv") -> tuple["torch.Tensor", "torch.Tensor"]:
    """Where the object's axis passes the pinch point, and the grasp's height along it.

    The pinch to the axis line (pinch frame, (N, 3)) and how far along the axis from the object's center (N,): a
    cylinder's grasp is the first, at a height the second.
    """
    o = object_in_pinch(env)
    center, axis = o[:, :3], o[:, 3:]
    along = (center * axis).sum(dim=1)
    return center - along[:, None] * axis, along


def grasp_point(env: "ManagerBasedRlEnv") -> "torch.Tensor":
    """Where the object's axis passes the pinch point in a grasp, for each world's radius (pinch frame, (N, 3))."""
    import torch

    r = object_dims(env)[:, 0]
    return torch.stack([a * r + b for a, b in GRASP_POINT_FIT], dim=1)


def between_jaws(env: "ManagerBasedRlEnv") -> "torch.Tensor":
    """1 where the object's axis is in the box between the jaws (for its radius) and the grasp on its body."""
    import torch

    perp, along = axis_from_pinch(env)
    dims = object_dims(env)
    r, height = dims[:, 0], dims[:, 1]
    (a_lo, b_lo), (a_hi, b_hi) = BETWEEN_X_FIT
    n = len(r)
    low = torch.stack(
        [a_lo * r + b_lo, *(torch.full((n,), v, device=env.device) for v in BETWEEN_LOW_M[1:])], dim=1
    )
    high = torch.stack(
        [a_hi * r + b_hi, *(torch.full((n,), v, device=env.device) for v in BETWEEN_HIGH_M[1:])], dim=1
    )
    inside = ((perp > low) & (perp < high)).all(dim=1)
    return (inside & (along.abs() < height / 2 - ON_BODY_MARGIN_M)).float()


def rest_z(env: "ManagerBasedRlEnv", entity: str = ENTITY) -> "torch.Tensor":
    """Each world's object's center height standing on the table (N,).

    The entity's default height (the table and half the object the environment was built with), moved by how much
    taller each world's object is than that one (`object_dims`; the host model keeps the built size).
    """
    obj = env.scene[entity]
    geom = int(obj.indexing.geom_ids[0])
    built_half = float(env.sim.mj_model.geom_size[geom, 1])
    return obj.data.default_root_state[:, 2] + object_dims(env, entity)[:, 1] / 2 - built_half


def tilt_cos(env: "ManagerBasedRlEnv") -> "torch.Tensor":
    """The cosine of the object's tilt from upright (its axis's world z)."""
    _, axis = object_pose(env)
    return axis[:, 2]


def upright(env: "ManagerBasedRlEnv", max_tilt_deg: float) -> "torch.Tensor":
    """Where the object's axis is within `max_tilt_deg` of vertical (bool, (N,))."""
    return tilt_cos(env) > float(np.cos(np.radians(max_tilt_deg)))
