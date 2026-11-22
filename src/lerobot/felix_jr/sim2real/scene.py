"""The scene for the final grasp: one calibrated arm (`robot.arm_spec`) over a table, with objects on it.

World = the model's root frame. The table's top is at `table_z` (the head's table estimate in ruler2: 0.747 m); it
starts `TABLE_NEAR_M` ahead of the arm's base (the robot faces -x) and is wider than the arms reach. Objects are
`Cylinder` and `Box`, at (x, y) in the root frame, standing on the table, free to move.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel
from lerobot.utils.import_utils import _mujoco_available, require_package

from .robot import FINGER_STL, arm_spec, set_pose

if TYPE_CHECKING or _mujoco_available:
    import mujoco

if TYPE_CHECKING:
    from lerobot.felix_jr.harness.primitives import ObjectRecord, Primitive

TABLE_Z = 0.747  # the table's top in the root frame (ruler2's head estimate)
TABLE_NEAR_M = 0.06  # the table's near edge this far ahead of the arm's base
TABLE_SIZE_M = (0.8, 1.2, 0.03)  # along x (away from the robot), y, thickness
TABLE_RGBA = (0.72, 0.55, 0.38, 1.0)


@dataclass(frozen=True)
class Cylinder:
    """An upright cylinder standing on the table at (x, y) (m, the root frame)."""

    name: str
    x: float
    y: float
    radius: float
    height: float
    mass: float = 0.05
    rgba: tuple[float, float, float, float] = (0.2, 0.4, 0.9, 1.0)


@dataclass(frozen=True)
class Box:
    """A box resting on the table at (x, y) (m, the root frame), half sizes `half` (m)."""

    name: str
    x: float
    y: float
    half: tuple[float, float, float]
    mass: float = 0.03
    rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)


def from_primitive(
    primitive: "Primitive", name: str = "object", mass: float = 0.05, table_z: float | None = None
) -> Cylinder:
    """The simulator's object for a harness primitive (`harness.primitives`), standing on the table where it is.

    Only an upright `Cylinder` so far (its bottom end's x, y; its radius and length); others are refused. With
    `table_z`, it stands on the table up to the primitive's top: a fit of the part marked (`fit_cylinder`: the middle of
    a bottle) is a bottle standing there (a fit 3 cm above the table, 6 cm long, stood on the table 6 cm tall, put its
    grasps above it: sim1, 2026-11-19). The mass is not geometry: an estimate, or a physics draw's.
    """
    from lerobot.felix_jr.harness.primitives import Cylinder as CylinderPrimitive

    if not isinstance(primitive, CylinderPrimitive):
        raise NotImplementedError(f"no simulator object for a {type(primitive).__name__} yet")
    if len(primitive) != 1:
        raise ValueError(f"one object at a time: got {len(primitive)}")
    axis = primitive.axis_root[0]
    if abs(float(axis[2])) < 0.99:
        raise NotImplementedError("only upright cylinders stand in the simulator so far")
    bottom = primitive.bottom_root[0] if axis[2] > 0 else primitive.top_root[0]
    length = float(primitive.length[0])
    height = length
    if table_z is not None:
        height = float(bottom[2]) + length - table_z
        if height <= 0:
            raise ValueError(f"the object's top ({float(bottom[2]) + length:.3f} m) is under the table")
    return Cylinder(name, float(bottom[0]), float(bottom[1]), float(primitive.radius[0]), height, mass)


def from_record(record: "ObjectRecord", name: str = "object") -> Cylinder:
    """The simulator's object for an object record (`harness.primitives.ObjectRecord`, one object).

    Standing where it stands (a bank's placement, a pool's start); its mass the record's.
    """
    return from_primitive(record.primitive, name, float(record.mass_kg[0]))


@dataclass
class Scene:
    """A compiled scene and its state, for one arm."""

    model: XleRobotModel
    arm: str
    m: "mujoco.MjModel"
    d: "mujoco.MjData"

    def set_pose(self, pose: Mapping[str, float]) -> None:
        """The arm at LeRobot's readings (`robot.set_pose`)."""
        set_pose(self.model, self.arm, self.m, self.d, pose)


def _add_object(spec: "mujoco.MjSpec", o: Cylinder | Box, table_z: float) -> None:
    rgba = list(o.rgba)
    if isinstance(o, Cylinder):
        body = spec.worldbody.add_body(name=o.name, pos=[o.x, o.y, table_z + o.height / 2])
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[o.radius, o.height / 2, 0], mass=o.mass, rgba=rgba
        )
    else:
        body = spec.worldbody.add_body(name=o.name, pos=[o.x, o.y, table_z + o.half[2]])
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=list(o.half), mass=o.mass, rgba=rgba)
    body.add_freejoint()


def scene_spec(
    model: XleRobotModel,
    arm: str,
    objects: Sequence[Cylinder | Box] = (),
    table_z: float = TABLE_Z,
    head_ticks: Mapping[str, int] | None = None,
    fingers: Path | None = FINGER_STL,
    finger_model: str = "sprung",
) -> "mujoco.MjSpec":
    """The arm's spec (`arm_spec`) with the table, the objects and lights.

    `fingers` if given, giving as `finger_model` (`robot.FINGER_MODELS`).
    """
    require_package("mujoco", extra="felix_jr_sim")
    spec = arm_spec(model, arm, head_ticks, fingers=fingers, finger_model=finger_model)
    base = np.array(spec.body("base").pos)
    lx, ly, lz = TABLE_SIZE_M
    near = base[0] - TABLE_NEAR_M
    spec.worldbody.add_geom(
        name="table",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[lx / 2, ly / 2, lz / 2],
        pos=[near - lx / 2, base[1], table_z - lz / 2],
        rgba=list(TABLE_RGBA),
    )
    spec.worldbody.add_light(
        pos=[near - lx / 2, base[1], table_z + 1.5], dir=[0, 0, -1], diffuse=[0.7, 0.7, 0.7]
    )
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.4, 0.4, 0.4]
    for o in objects:
        _add_object(spec, o, table_z)
    return spec


def build(
    model: XleRobotModel,
    arm: str,
    objects: Sequence[Cylinder | Box] = (),
    table_z: float = TABLE_Z,
    head_ticks: Mapping[str, int] | None = None,
    fingers: Path | None = FINGER_STL,
    finger_model: str = "sprung",
) -> Scene:
    """The compiled scene (`scene_spec`), the arm at all-zero readings."""
    m = scene_spec(model, arm, objects, table_z, head_ticks, fingers, finger_model).compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return Scene(model, arm, m, d)
