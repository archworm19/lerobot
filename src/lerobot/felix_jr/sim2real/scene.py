"""The scene for the final grasp: one calibrated arm (`robot.arm_spec`) over a table, with objects on it.

World = the model's root frame. The table's top is at `table_z` (the head's table estimate in ruler2: 0.747 m); it
starts `TABLE_NEAR_M` ahead of the arm's base (the robot faces -x) and is wider than the arms reach. Objects are
`Cylinder` and `Box`, at (x, y) in the root frame, standing on the table, free to move.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel
from lerobot.utils.import_utils import _mujoco_available, require_package

from .robot import arm_spec, set_pose

if TYPE_CHECKING or _mujoco_available:
    import mujoco

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
) -> "mujoco.MjSpec":
    """The arm's spec (`arm_spec`) with the table, the objects and lights."""
    require_package("mujoco", extra="felix_jr_sim")
    spec = arm_spec(model, arm, head_ticks)
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
) -> Scene:
    """The compiled scene (`scene_spec`), the arm at all-zero readings."""
    m = scene_spec(model, arm, objects, table_z, head_ticks).compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return Scene(model, arm, m, d)
