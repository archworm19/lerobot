"""Shapes an object is made of, and the antipodal grasps each one allows (`docs/object_grasping.md`, section 2).

A primitive is pure geometry with a pose in the root frame (`root_from_obj`, 4x4: the shape's own axes and origin); it
knows nothing about the robot. Its grasps form a **family** with a few named parameters (a cylinder's `t`, `phi`,
`psi`), generated on a grid so the same shape always gives the same grasps. A grasp is antipodal: two contacts with
opposed normals on one closing line. Which contact takes the fixed jaw, how deep the jaws go and whether the arm
reaches are decided later (`grasping.py`).

`where` narrows a family to ranges of its parameters, `spacing` sets the grid's step per parameter; both are keyed by
the parameter names the primitive documents (angles in degrees, lengths in meters). A range's ends are always on the
grid.

Names say their frame (`docs/object_grasping.md`, "Frames and names"): `root_from_obj` maps the object frame to the root
frame, `center_root` is a point in the root frame.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

Ranges = Mapping[str, tuple[float, float]]
Spacing = Mapping[str, float]


def _cm(value: float) -> str:
    return f"{value * 100:.1f}"


def _grid(span: tuple[float, float], step: float, wrap: float | None = None) -> np.ndarray:
    """`span`'s ends and evenly spaced values between, no further apart than `step`.

    With `wrap` (a period), a full turn keeps one end only.
    """
    lo, hi = span
    if hi < lo:
        raise ValueError(f"empty range {span}")
    full_turn = False
    if wrap is not None and hi - lo >= wrap - 1e-9:
        full_turn, hi = True, lo + wrap
    n = max(int(np.ceil((hi - lo) / step - 1e-9)), 1) + 1 if hi > lo else 1
    values = np.linspace(lo, hi, n)
    return values[:-1] if full_turn else values


def frame_along(axis_root: np.ndarray, origin_root: np.ndarray) -> np.ndarray:
    """root_from_obj (4x4): an object frame whose z is `axis_root` at `origin_root`.

    Its x is the root x made perpendicular to the axis (the root y if the axis is along x), so for an upright shape
    the object frame's axes are the root's.
    """
    z = np.asarray(axis_root, dtype=float)
    z = z / np.linalg.norm(z)
    ref = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = ref - (ref @ z) * z
    x /= np.linalg.norm(x)
    root_from_obj = np.eye(4)
    root_from_obj[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    root_from_obj[:3, 3] = origin_root
    return root_from_obj


@dataclass(frozen=True, eq=False)
class AntipodalGrasp:
    """Two contacts with opposed normals on one closing line (root frame, m).

    `center_root` is midway between the contacts, `closing_root` the unit direction from the first contact to the
    second, `approach_root` the unit direction the jaws move in to reach it (perpendicular to the closing), `opening`
    the distance between the contacts (what the jaws must close on). `params` are the family's parameters that made it:
    the grasp in the object's terms.
    """

    center_root: np.ndarray  # (3,)
    closing_root: np.ndarray  # (3,) unit
    approach_root: np.ndarray  # (3,) unit
    opening: float
    params: Mapping[str, float] = field(default_factory=dict)

    @property
    def contacts_root(self) -> tuple[np.ndarray, np.ndarray]:
        """The two contact points: the center -/+ half the opening along the closing direction."""
        half = 0.5 * self.opening * self.closing_root
        return self.center_root - half, self.center_root + half

    def describe(self) -> str:
        """The family parameters and the opening."""
        params = ", ".join(f"{name} {value:g}" for name, value in self.params.items())
        return f"grasp ({params}): opening {_cm(self.opening)} cm"


class Primitive(Protocol):
    """A shape with a pose in the root frame that generates its own antipodal grasps."""

    @property
    def root_from_obj(self) -> np.ndarray:
        """(4, 4): the shape's own axes and origin in the root frame."""
        ...

    def surface_distance(self, points_root: np.ndarray) -> np.ndarray:
        """Signed distance (m) of each point (N, 3) to the surface: negative inside."""
        ...

    def antipodal_grasps(
        self, where: Ranges | None = None, spacing: Spacing | None = None, max_opening: float | None = None
    ) -> list[AntipodalGrasp]:
        """The family's grasps on a grid, within `where`, no wider than `max_opening`."""
        ...

    def describe(self) -> str:
        """The shape and where it is, in words and numbers."""
        ...


@dataclass(frozen=True, eq=False)
class Cylinder:
    """A solid cylinder along the object frame's z.

    The axis runs from the origin (the "bottom" end: on the table for an upright object) to `length` along it.

    Grasp family, closing straight through the axis so the opening is the diameter:

    - `t` (m): where along the axis, from the bottom end; default the whole length, every 1 cm.
    - `phi` (deg): which side the jaws come from, around the axis from the object frame's x (for an upright cylinder
      the root x: 0 is from the robot's side, as the robot faces -x); default all the way round, every 30 deg.
    - `psi` (deg): how far the approach tilts down the axis, toward the bottom end: 0 is square to the axis, 90 along
      it from the top end (from above, for an upright cylinder), -90 from the bottom; default -90..90, every 15 deg.

    The closing direction is square to both the axis and the side the jaws come from.
    """

    root_from_obj: np.ndarray  # (4, 4)
    radius: float
    length: float

    SPACING = {"t": 0.01, "phi": 30.0, "psi": 15.0}

    def __post_init__(self) -> None:
        """Refuse a frame that is not a rigid transform, or a size that is not positive."""
        rotation = self.root_from_obj[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
            raise ValueError("the cylinder's frame is not a rotation")
        if self.radius <= 0 or self.length <= 0:
            raise ValueError(f"radius {self.radius} and length {self.length} must be positive")

    @classmethod
    def upright(cls, bottom_root: np.ndarray, height: float, radius: float) -> "Cylinder":
        """A cylinder standing on its end: its axis vertical up from `bottom_root`, the bottom end's center."""
        return cls(
            frame_along(np.array([0.0, 0.0, 1.0]), np.asarray(bottom_root, dtype=float)), radius, height
        )

    @classmethod
    def between(cls, bottom_root: np.ndarray, top_root: np.ndarray, radius: float) -> "Cylinder":
        """A cylinder whose axis runs from `bottom_root` to `top_root`, its ends' centers."""
        bottom_root, top_root = np.asarray(bottom_root, dtype=float), np.asarray(top_root, dtype=float)
        return cls(
            frame_along(top_root - bottom_root, bottom_root),
            radius,
            float(np.linalg.norm(top_root - bottom_root)),
        )

    @property
    def axis_root(self) -> np.ndarray:
        """The unit axis, bottom to top."""
        return self.root_from_obj[:3, 2]

    @property
    def bottom_root(self) -> np.ndarray:
        """The bottom end's center."""
        return self.root_from_obj[:3, 3]

    @property
    def top_root(self) -> np.ndarray:
        """The top end's center."""
        return self.bottom_root + self.length * self.axis_root

    def surface_distance(self, points_root: np.ndarray) -> np.ndarray:
        """Signed distance (m) of each point (N, 3) to the surface, ends included: negative inside."""
        points_obj = (np.atleast_2d(points_root) - self.bottom_root) @ self.root_from_obj[:3, :3]
        radial = np.hypot(points_obj[:, 0], points_obj[:, 1]) - self.radius
        axial = np.maximum(-points_obj[:, 2], points_obj[:, 2] - self.length)
        outside = np.hypot(np.maximum(radial, 0.0), np.maximum(axial, 0.0))
        return outside + np.minimum(np.maximum(radial, axial), 0.0)

    def antipodal_grasps(
        self, where: Ranges | None = None, spacing: Spacing | None = None, max_opening: float | None = None
    ) -> list[AntipodalGrasp]:
        """The grasps on the grid of `t` x `phi` x `psi`, within `where`; none if the diameter is over `max_opening`."""
        opening = 2.0 * self.radius
        if max_opening is not None and opening > max_opening:
            return []
        where, step = dict(where or {}), {**self.SPACING, **(spacing or {})}
        unknown = set(where) - set(self.SPACING)
        if unknown:
            raise ValueError(f"a cylinder's grasps have no parameter {sorted(unknown)}: use t, phi, psi")
        t_span = where.get("t", (0.0, self.length))
        if t_span[0] < 0 or t_span[1] > self.length + 1e-9:
            raise ValueError(f"t {t_span} is outside the cylinder (0..{self.length:.3f} m)")
        psi_span = where.get("psi", (-90.0, 90.0))
        if psi_span[0] < -90 or psi_span[1] > 90:
            raise ValueError(f"psi {psi_span} is outside -90..90 deg")
        rot_root_from_obj, axis_root = self.root_from_obj[:3, :3], self.axis_root
        grasps = []
        for t in _grid(t_span, step["t"]):
            center_root = self.bottom_root + t * axis_root
            for phi in _grid(where.get("phi", (0.0, 360.0)), step["phi"], wrap=360.0):
                side_root = rot_root_from_obj @ np.array(
                    [np.cos(np.radians(phi)), np.sin(np.radians(phi)), 0.0]
                )
                closing_root = np.cross(axis_root, side_root)
                for psi in _grid(psi_span, step["psi"]):
                    p = np.radians(psi)
                    approach_root = -np.cos(p) * side_root - np.sin(p) * axis_root
                    params = {
                        "t": round(float(t), 4),
                        "phi": round(float(phi), 2),
                        "psi": round(float(psi), 2),
                    }
                    grasps.append(AntipodalGrasp(center_root, closing_root, approach_root, opening, params))
        return grasps

    def describe(self) -> str:
        """The radius, the bottom end, the length and the axis's tilt from vertical (root frame)."""
        b, a = self.bottom_root, self.axis_root
        tilt = float(np.degrees(np.arccos(np.clip(abs(a[2]), 0.0, 1.0))))
        where = f"from ({', '.join(_cm(float(v)) for v in b)}) cm (root frame), {_cm(self.length)} cm along "
        where += (
            "the vertical"
            if tilt < 1.0
            else f"({', '.join(f'{v:.2f}' for v in a)}), {tilt:.0f} deg from vertical"
        )
        return f"cylinder, radius {_cm(self.radius)} cm, {where}"
