"""A MuJoCo scene for the final grasp: the SO-101 (`assets.so101_model`) on a table, with objects on it.

The arm's base is at the origin on the table's top (z = 0, the arm along +x). Objects are free bodies:
`Cylinder` (a bottle, a chapstick) and `Box`. Still to come (`docs/sim2real_grasp.md`): the arm placed as the
calibrated model has it, the wrist camera from its calibration, and pads on the jaws for their contacts (their
meshes collide as convex hulls, which fill the claw's inner curve).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from lerobot.utils.import_utils import _mujoco_available, require_package

from .assets import so101_model

if TYPE_CHECKING or _mujoco_available:
    import mujoco

TABLE_HALF_M = (0.5, 0.5, 0.02)  # the table's half sizes (x, y, thickness)
TABLE_CENTER_X = 0.25  # ahead of the arm's base


@dataclass(frozen=True)
class Cylinder:
    """An upright cylinder standing on the table at (x, y) (m), its bottom at the table's top."""

    name: str
    x: float
    y: float
    radius: float
    height: float
    mass: float = 0.05
    rgba: tuple[float, float, float, float] = (0.2, 0.4, 0.9, 1.0)

    def xml(self) -> str:
        """Its MJCF body."""
        return (
            f'<body name="{self.name}" pos="{self.x} {self.y} {self.height / 2}"><freejoint/>'
            f'<geom type="cylinder" size="{self.radius} {self.height / 2}" mass="{self.mass}" '
            f'rgba="{" ".join(map(str, self.rgba))}"/></body>'
        )


@dataclass(frozen=True)
class Box:
    """A box resting on the table at (x, y) (m), half sizes `half` (m)."""

    name: str
    x: float
    y: float
    half: tuple[float, float, float]
    mass: float = 0.03
    rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)

    def xml(self) -> str:
        """Its MJCF body."""
        return (
            f'<body name="{self.name}" pos="{self.x} {self.y} {self.half[2]}"><freejoint/>'
            f'<geom type="box" size="{" ".join(map(str, self.half))}" mass="{self.mass}" '
            f'rgba="{" ".join(map(str, self.rgba))}"/></body>'
        )


def scene_xml(objects: tuple[Cylinder | Box, ...] = (), model: Path | None = None) -> str:
    """The scene's MJCF (absolute paths: it loads from a string)."""
    arm = model or so101_model()
    hx, hy, hz = TABLE_HALF_M
    bodies = "\n    ".join(o.xml() for o in objects)
    return f"""<mujoco model="so101 on a table">
  <include file="{arm}"/>
  <compiler meshdir="{arm.parent / "assets"}"/>
  <visual><headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/></visual>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1" directional="true"/>
    <geom name="table" type="box" size="{hx} {hy} {hz}" pos="{TABLE_CENTER_X} 0 {-hz}" rgba="0.6 0.45 0.3 1"/>
    {bodies}
  </worldbody>
</mujoco>
"""


def build(objects: tuple[Cylinder | Box, ...] = (), model: Path | None = None) -> "mujoco.MjModel":
    """The compiled scene."""
    require_package("mujoco", extra="felix_jr_sim")
    return mujoco.MjModel.from_xml_string(scene_xml(objects, model))
