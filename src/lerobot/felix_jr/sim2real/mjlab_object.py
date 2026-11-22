"""The object in the mjlab environment: `MjlabCylinder`, everything about it that knows its shape, in one place.

What the object *is* (its shape's geometry, batched; how it is written down in the bank and the pools) is
`harness.primitives` (`Cylinder`, `ObjectRecord`). `MjlabCylinder` is how that object lives in mjlab: a plain,
stateless configuration (mjlab's managers each copy their configs: per-world state lives in the model's fields and in
`pool_reset.PoolReset`), built from the record of the object the environment is built with, by stage:

- **Construction** (`entity_cfg`, `spec`): the mjlab entity, its spec a body at the origin with one cylinder geom and a
  free joint (`scene._add_object`, the CPU simulator's own builder), its pose the initial state.
- **Contacts** (`contact_match`): which elements are the object, for the contact sensors that look for it (the finger
  pads' sensors: `mjlab_task.sensors`).
- **Reset** (`declaration`, `write`, `draw_size_error`): for objects of several sizes, the model fields a world's object
  needs rewritten (`SIZED_FIELDS`), declared per world (and the constants recomputed after), and written from
  `ObjectRecord.vectors` rows: the primitive's own batched geometry on the GPU tensors where they are, in MuJoCo's
  encoding (a cylinder's size: radius, half height); its mass times a draw in `mass_range`; the error in the size the
  actor is told (`size_error_m`). `pool_reset.PoolReset` calls these: it draws the starts, which know nothing of shape.
- **Observations** (`observation_terms`: `in_root`, `in_pinch`, `size`): its center and axis (a cylinder's pose up to its
  own spin, which nothing sees), in the root and the pinch frames; its radius and height (each world's own, read from
  the model), what the actor is told of them (off by the episode's error) and the critic (the truth).
- **Geometry** (`pose`, `dims`, `rest_z`, `tilt_cos`, `upright`): where it is, how big, its height standing on the
  table, how far it tilts. What `mjlab_task`'s rewards and terminations are built from (`rise`, `held`, `tipped`): the
  task names what it needs, the object says it.
- **In our claw** (`axis_from_pinch`, `grasp_point`, `between_jaws`, and `in_pinch`): the object relative to the
  gripper's pinch frame (`mjlab_task.pinch_pose`), and where a grasp puts it (fit over our claw's starts, against the
  radius). Facts about this cylinder in this gripper: a different gripper would refit them, a different shape recompute.

`of(env)`: the environment's object (from its reset term's parameters). A second shape would mean another class here
with the same methods, and its `harness.primitives` geometry (and the bank's grasp planning).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from lerobot.felix_jr.harness.primitives import Cylinder, ObjectRecord

from . import scene as sc

if TYPE_CHECKING:
    import mujoco
    import torch
    from mjlab.entity import EntityCfg
    from mjlab.envs import ManagerBasedRlEnv

ENTITY = "object"  # the object's entity's name in the scene, by default
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


def default_object(table_z: float) -> ObjectRecord:
    """The object with no pool: feno (53 x 110 mm, 116 g), standing on the table at one of sim3's spots."""
    return ObjectRecord.upright(
        "cylinder", {"radius": 0.0265, "height": 0.11}, 0.116, np.array([-0.356, 0.132, table_z])
    )


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


@dataclass(frozen=True, eq=False)
class MjlabCylinder:
    """An upright cylinder in the mjlab environment (the module docstring).

    Construction, contacts, reset, observations, geometry, and the object in our claw. `template`: the object the environment is built with (one; where it stands its initial state). With a pool whose
    objects vary in size (`sized`), each world's own is written at its reset; `mass_range` (its mass times a log-uniform
    draw, then) and `size_error_m` (the error in the size the actor is told) are those resets' randomization.
    """

    template: ObjectRecord
    entity: str = ENTITY
    sized: bool = False
    mass_range: tuple[float, float] = (1.0, 1.0)
    size_error_m: float = 0.0

    # ---- construction ----

    def spec(self) -> "mujoco.MjSpec":
        """Its spec: a body at the origin (its pose is the entity's initial state), as `scene` builds objects."""
        import mujoco

        radius, height = (float(v) for v in self.template.primitive.params[0])
        spec = mujoco.MjSpec()
        obj = sc.Cylinder(self.entity, 0.0, 0.0, radius, height, float(self.template.mass_kg[0]))
        sc._add_object(spec, obj, table_z=-height / 2)  # (the body's origin at the cylinder's center)
        return spec

    def entity_cfg(self) -> "EntityCfg":
        """The mjlab entity: the spec, its centroid where the template stands its initial position."""
        from mjlab.entity import EntityCfg

        p = self.template.primitive
        center = p.root_from_obj[0, :3, 3] + p.root_from_obj[0, :3, :3] @ p.centroid_obj()[0]
        return EntityCfg(
            spec_fn=self.spec, init_state=EntityCfg.InitialStateCfg(pos=tuple(float(v) for v in center))
        )

    # ---- contacts ----

    def contact_match(self) -> Any:
        """Which elements are the object, for a contact sensor's side (`mjlab.sensor.ContactMatch`): its one body."""
        from mjlab.sensor import ContactMatch

        return ContactMatch(mode="body", pattern=self.entity, entity=self.entity)

    # ---- reset ----

    def ids(self, env: "ManagerBasedRlEnv") -> tuple[int, int]:
        """Its body and its one geom (a cylinder), as indices into the model's fields."""
        import mujoco

        obj = env.scene[self.entity]
        bodies, geoms = obj.indexing.body_ids.tolist(), obj.indexing.geom_ids.tolist()
        if len(bodies) != 1 or len(geoms) != 1:
            raise ValueError(
                f"expected the object to be one body with one geom: bodies {bodies}, geoms {geoms}"
            )
        if env.sim.mj_model.geom_type[geoms[0]] != mujoco.mjtGeom.mjGEOM_CYLINDER:
            raise ValueError("expected the object's geom to be a cylinder")
        return int(bodies[0]), int(geoms[0])

    def declaration(self) -> tuple[tuple[str, ...], Any]:
        """What a reset term that resizes it declares to mjlab: (its `model_fields`, its `recompute`).

        MuJoCo Warp keeps one compiled model for every world; each model field is stored with a world dimension, one
        copy by default (every world reads it) or one a world once expanded. mjlab expands the fields an event term
        names in `model_fields` (gathered by its event manager right after building each term; `sim.expand_model_fields`
        then allocates them before the first step, and re-captures its CUDA graphs since the arrays move). Unexpanded, a
        write to `model.geom_size[world]` would land on the one shared copy. The fields (`SIZED_FIELDS`): the
        cylinder's size (radius, half height), its bounding sphere and box (`geom_rbound`, `geom_aabb`: what the
        collision broadphase reads, not `geom_size`; stale, contacts are missed or phantom), its body's mass and
        inertia. `requires_model_fields` adds the fields derived from those (`body_subtreemass`, `body_invweight0`,
        `dof_invweight0`, ...: per world too). `recompute`: after the reset events fire, mjlab recomputes those derived
        constants (`sim.recompute_constants`, MuJoCo Warp's `set_const`, which saves and restores the simulation
        state): the object's mass and inertia change its subtree mass and the inverse weights the constraint solver
        scales its contacts by.
        """
        from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields

        def declared() -> None:  # (only to carry the declaration, its derived fields included; never called)
            pass

        fields = requires_model_fields(*SIZED_FIELDS, recompute=RecomputeLevel.set_const)(
            declared
        ).model_fields
        return tuple(fields), RecomputeLevel.set_const

    def write(self, env: "ManagerBasedRlEnv", env_ids: "torch.Tensor", rows: "torch.Tensor") -> None:
        """The object in each world of `env_ids` made its row's (`ObjectRecord.vectors`: radius, height, mass).

        The primitive's batched geometry (`bounding_radius`, `half_extents`, `inertia_per_kg`) on the rows' device,
        written as MuJoCo keeps it: the geom's size (radius, half height), its bounding sphere and box, the body's mass
        (times a log-uniform draw in `mass_range`) and its inertia. The fields must be per world (`declaration`); mjlab
        recomputes the constants after.
        """
        import torch

        m = env.sim.model
        body, geom = self.ids(env)
        record = ObjectRecord.from_vectors(rows, self.template.primitive.SHAPE)
        p = cast("Cylinder", record.primitive)
        lo, hi = np.log(self.mass_range[0]), np.log(self.mass_range[1])
        mass = record.mass_kg * torch.exp(lo + (hi - lo) * torch.rand(len(env_ids), device=rows.device))
        m.geom_size[env_ids, geom, 0] = p.radius.to(
            m.geom_size.dtype
        )  # (MuJoCo's cylinder: radius, half height)
        m.geom_size[env_ids, geom, 1] = (p.height / 2).to(m.geom_size.dtype)
        m.geom_rbound[env_ids, geom] = p.bounding_radius().to(m.geom_rbound.dtype)
        m.geom_aabb[env_ids, geom, 1] = p.half_extents().to(m.geom_aabb.dtype)
        m.body_mass[env_ids, body] = mass.to(m.body_mass.dtype)
        m.body_inertia[env_ids, body] = (mass[:, None] * p.inertia_per_kg()).to(m.body_inertia.dtype)

    def draw_size_error(self, n: int, device: Any) -> "torch.Tensor":
        """`n` episodes' errors in the radius and height the actor is told ((n, 2), m: normal, `size_error_m`)."""
        import torch

        return self.size_error_m * torch.randn(n, 2, device=device)

    # ---- geometry ----

    def pose(self, env: "ManagerBasedRlEnv") -> tuple["torch.Tensor", "torch.Tensor"]:
        """Its center and axis (its body z), world frame (= the root frame here)."""
        import torch

        data = env.scene[self.entity].data
        z = torch.zeros_like(data.root_link_pos_w)
        z[:, 2] = 1.0
        return data.root_link_pos_w, rotate(data.root_link_quat_w, z)

    def dims(self, env: "ManagerBasedRlEnv") -> "torch.Tensor":
        """Each world's radius and height ((N, 2), m), read from the model (each world's own once resized)."""
        geom = int(env.scene[self.entity].indexing.geom_ids[0])
        size = env.sim.model.geom_size[..., geom, :2].reshape(-1, 2).float()
        dims = size * size.new_tensor([1.0, 2.0])  # (MuJoCo's radius, half height): radius, height
        return dims.expand(env.num_envs, 2) if len(dims) == 1 else dims

    def rest_z(self, env: "ManagerBasedRlEnv") -> "torch.Tensor":
        """Each world's center height standing on the table (N,).

        The entity's default height (the table and half the object the environment was built with), moved by how much
        taller each world's object is than that one (the host model keeps the built size).
        """
        obj = env.scene[self.entity]
        geom = int(obj.indexing.geom_ids[0])
        built_half = float(env.sim.mj_model.geom_size[geom, 1])
        return obj.data.default_root_state[:, 2] + self.dims(env)[:, 1] / 2 - built_half

    def tilt_cos(self, env: "ManagerBasedRlEnv") -> "torch.Tensor":
        """The cosine of its tilt from upright (its axis's world z)."""
        _, axis = self.pose(env)
        return axis[:, 2]

    def upright(self, env: "ManagerBasedRlEnv", max_tilt_deg: float) -> "torch.Tensor":
        """Where its axis is within `max_tilt_deg` of vertical (bool, (N,))."""
        return self.tilt_cos(env) > float(np.cos(np.radians(max_tilt_deg)))

    # ---- observations ----

    def in_root(self, env: "ManagerBasedRlEnv") -> "torch.Tensor":
        """Its center (3) and axis (3) in the root frame: an observation."""
        import torch

        center, axis = self.pose(env)
        return torch.cat([center, axis], dim=1)

    def size(self, env: "ManagerBasedRlEnv", estimated: bool = False) -> "torch.Tensor":
        """Each world's radius and height (m, `dims`): an observation, for objects of several sizes.

        `estimated`: off by the episode's error (drawn at its reset, kept by `pool_reset.PoolReset`: the actor's view).
        """
        from .pool_reset import pool_reset

        dims = self.dims(env)
        term = pool_reset(env) if estimated else None
        return dims if term is None else dims + term.size_error

    def observation_terms(self, noise: Any = None) -> tuple[dict[str, Any], dict[str, Any]]:
        """Its observation terms: (the actor's, the critic's).

        In the root and the pinch frames (`noise` on the actor's), and, for objects of several sizes, its size (the
        actor's estimated, the critic's true).
        """
        from mjlab.managers.observation_manager import ObservationTermCfg

        actor = {
            "object_in_root": ObservationTermCfg(func=self.in_root, noise=noise),
            "object_in_pinch": ObservationTermCfg(func=self.in_pinch, noise=noise),
        }
        critic = dict(actor)
        if self.sized:  # (one object's runs keep their observations: their checkpoints still load)
            actor["object_size"] = ObservationTermCfg(func=self.size, params={"estimated": True})
            critic["object_size"] = ObservationTermCfg(func=self.size)
        return actor, critic

    # ---- in our claw ----

    def in_pinch(self, env: "ManagerBasedRlEnv") -> "torch.Tensor":
        """Its center (3) and axis (3) in the gripper's pinch frame (`mjlab_task.pinch_pose`): an observation."""
        import torch

        from .mjlab_task import pinch_pose

        center, axis = self.pose(env)
        pos, quat = pinch_pose(env)
        return torch.cat([rotate_inverse(quat, center - pos), rotate_inverse(quat, axis)], dim=1)

    def axis_from_pinch(self, env: "ManagerBasedRlEnv") -> tuple["torch.Tensor", "torch.Tensor"]:
        """Where its axis passes the pinch point, and the grasp's height along it.

        The pinch to the axis line (pinch frame, (N, 3)) and how far along the axis from its center (N,): a cylinder's
        grasp is the first, at a height the second.
        """
        o = self.in_pinch(env)
        center, axis = o[:, :3], o[:, 3:]
        along = (center * axis).sum(dim=1)
        return center - along[:, None] * axis, along

    def grasp_point(self, env: "ManagerBasedRlEnv") -> "torch.Tensor":
        """Where its axis passes the pinch point in our claw's grasp, for each world's radius (pinch frame, (N, 3))."""
        import torch

        r = self.dims(env)[:, 0]
        return torch.stack([a * r + b for a, b in GRASP_POINT_FIT], dim=1)

    def between_jaws(self, env: "ManagerBasedRlEnv") -> "torch.Tensor":
        """1 where its axis is in the box between our claw's jaws (for its radius) and the grasp on its body."""
        import torch

        perp, along = self.axis_from_pinch(env)
        dims = self.dims(env)
        r, height = dims[:, 0], dims[:, 1]
        (a_lo, b_lo), (a_hi, b_hi) = BETWEEN_X_FIT
        n = len(r)
        fixed = env.device
        low = torch.stack(
            [a_lo * r + b_lo, *(torch.full((n,), v, device=fixed) for v in BETWEEN_LOW_M[1:])], dim=1
        )
        high = torch.stack(
            [a_hi * r + b_hi, *(torch.full((n,), v, device=fixed) for v in BETWEEN_HIGH_M[1:])], dim=1
        )
        inside = ((perp > low) & (perp < high)).all(dim=1)
        return (inside & (along.abs() < height / 2 - ON_BODY_MARGIN_M)).float()


def of(env: "ManagerBasedRlEnv") -> MjlabCylinder:
    """The environment's object: the one its terms were configured with (its reset term's, else any term's)."""
    for group in ("events", "rewards", "terminations"):
        for term in (getattr(env.cfg, group, None) or {}).values():
            for value in getattr(term, "params", {}).values():
                if isinstance(value, MjlabCylinder):
                    return value
    raise ValueError("the environment's terms name no object (`MjlabCylinder`)")
