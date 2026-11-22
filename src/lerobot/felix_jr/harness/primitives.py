"""Shapes an object is made of, and the antipodal grasps each one allows (`docs/object_grasping.md`, section 2).

A primitive is pure geometry with a pose in the root frame (`root_from_obj`, 4x4: the shape's own axes and origin); it
knows nothing about the robot. Its grasps form a **family** with a few named parameters (a cylinder's `t`, `phi`,
`psi`), generated on a grid so the same shape always gives the same grasps. A grasp is antipodal: two contacts with
opposed normals on one closing line. Which contact takes the fixed jaw, how deep the jaws go and whether the arm
reaches are decided later (`grasping.py`).

`where` narrows a family to ranges of its parameters, `spacing` sets the grid's step per parameter; both are keyed by
the parameter names the primitive documents (angles in degrees, lengths in meters). A range's ends are always on the
grid. The family is also continuous: `grasp_space` names its parameters with their ranges, `grasp_at` makes the grasp
at any values in it (what an optimizer samples: `sim2real.refine`); the grid is `grasp_at` on `_grid`s.

Names say their frame (`docs/object_grasping.md`, "Frames and names"): `root_from_obj` maps the object frame to the root
frame, `center_root` is a point in the root frame.

A primitive also says what it is, in numbers anything else can use: its shape's name (`SHAPE`) and parameters
(`shape_params`, e.g. a cylinder's radius and height), its bounds (`bounding_radius`, `half_extents`) and a solid's
inertia for a mass (`solid_inertia`). A `RigidBody` is a primitive with a mass, and its record (`to_record`,
`from_record`) is the one way it is written down: the grasp bank's placements (`sim2real/bank.py`) and the start pools
(`sim2real/start_pool.py`) both go through it, so the simulators read the object the bank planned on.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

Ranges = Mapping[str, tuple[float, float]]
Spacing = Mapping[str, float]


@dataclass(frozen=True)
class GraspParameter:
    """One parameter of a grasp family: its range, its period if it wraps (an angle all the way round), its unit."""

    name: str
    low: float
    high: float
    period: float | None = None
    unit: str = ""


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


class Surface(Protocol):
    """Something in the scene to keep clear of: its signed distance from points (`grasp_candidates`' obstacles)."""

    def surface_distance(self, points_root: np.ndarray) -> np.ndarray:
        """Signed distance (m) of each point (N, 3) to the surface: negative inside (or below)."""
        ...

    def describe(self) -> str:
        """The surface and where it is, in words and numbers."""
        ...


class Primitive(Surface, Protocol):
    """A shape with a pose in the root frame that generates its own antipodal grasps: a `Surface` to grasp."""

    @property
    def root_from_obj(self) -> np.ndarray:
        """(4, 4): the shape's own axes and origin in the root frame."""
        ...

    def antipodal_grasps(
        self, where: Ranges | None = None, spacing: Spacing | None = None, max_opening: float | None = None
    ) -> list[AntipodalGrasp]:
        """The family's grasps on a grid, within `where`, no wider than `max_opening`."""
        ...

    def grasp_space(self, where: Ranges | None = None) -> dict[str, GraspParameter]:
        """The family's parameters, each with its range (within `where`): what `grasp_at` takes."""
        ...

    def grasp_at(self, params: Mapping[str, float]) -> AntipodalGrasp:
        """The family's grasp at `params` (every one of `grasp_space`'s, any value in its range)."""
        ...

    def approach_preference(self, approach_root: np.ndarray) -> float:
        """How far (deg) the gripper coming in along `approach_root` (root frame, unit) is from the shape's best way in.

        0 the best. `grasp_candidates` ranks by it after how antipodal a grasp is.
        """
        ...


@dataclass(frozen=True)
class HorizontalPlane:
    """A level plane at height `z` (root frame), e.g. a table measured another way than its `TableEstimate`."""

    z: float
    label: str = "the table"

    def surface_distance(self, points_root: np.ndarray) -> np.ndarray:
        """Each point's height above the plane (m): negative below it."""
        return np.atleast_2d(np.asarray(points_root, dtype=np.float64))[:, 2] - self.z

    def describe(self) -> str:
        """The plane's height."""
        return f"{self.label}: level, at z = {100 * self.z:.1f} cm"


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
    SHAPE = "cylinder"
    PARAMS = ("radius", "height")  # (`shape_params`' names: the height is the length, end to end)

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

    def shape_params(self) -> dict[str, float]:
        """The shape's parameters by name (`PARAMS`, m): radius and height (its length, end to end)."""
        return {"radius": float(self.radius), "height": float(self.length)}

    @classmethod
    def from_shape_params(cls, params: Mapping[str, float], root_from_obj: np.ndarray) -> "Cylinder":
        """`shape_params`' values at the pose `root_from_obj` (its bottom end's center, its axis along z)."""
        return cls(np.asarray(root_from_obj, dtype=float), float(params["radius"]), float(params["height"]))

    @property
    def center_obj(self) -> np.ndarray:
        """The center (of mass, for a solid), object frame: half way up the axis."""
        return np.array([0.0, 0.0, self.length / 2])

    @property
    def bounding_radius(self) -> float:
        """The smallest sphere about the center that holds it (m)."""
        return float(np.hypot(self.radius, self.length / 2))

    @property
    def half_extents(self) -> np.ndarray:
        """The half sizes of the box about the center that holds it, object frame (m)."""
        return np.array([self.radius, self.radius, self.length / 2])

    def solid_inertia(self, mass: float) -> np.ndarray:
        """A solid cylinder's principal moments of inertia about its center, along the object frame's axes (kg m^2)."""
        side = mass * (3 * self.radius**2 + self.length**2) / 12
        return np.array([side, side, mass * self.radius**2 / 2])

    def approach_preference(self, approach_root: np.ndarray) -> float:
        """How far (deg) an approach is off square to the axis: 0 from the side, 90 along it (from above or below).

        From the side the jaws close across it, the gripper and its wrist camera level with it; a steep approach saw only
        an upright bottle's top from the wrist camera (ruler2).
        """
        a = np.asarray(approach_root, dtype=np.float64)
        along = abs(float(a @ self.axis_root)) / float(np.linalg.norm(a))
        return float(np.degrees(np.arcsin(min(1.0, along))))

    def surface_distance(self, points_root: np.ndarray) -> np.ndarray:
        """Signed distance (m) of each point (N, 3) to the surface, ends included: negative inside."""
        points_obj = (np.atleast_2d(points_root) - self.bottom_root) @ self.root_from_obj[:3, :3]
        radial = np.hypot(points_obj[:, 0], points_obj[:, 1]) - self.radius
        axial = np.maximum(-points_obj[:, 2], points_obj[:, 2] - self.length)
        outside = np.hypot(np.maximum(radial, 0.0), np.maximum(axial, 0.0))
        return outside + np.minimum(np.maximum(radial, axial), 0.0)

    def grasp_space(self, where: Ranges | None = None) -> dict[str, GraspParameter]:
        """`t` (m), `phi` (deg, all the way round), `psi` (deg), within `where` (the class docstring)."""
        where = dict(where or {})
        unknown = set(where) - set(self.SPACING)
        if unknown:
            raise ValueError(f"a cylinder's grasps have no parameter {sorted(unknown)}: use t, phi, psi")
        t_span = where.get("t", (0.0, self.length))
        if t_span[0] < 0 or t_span[1] > self.length + 1e-9:
            raise ValueError(f"t {t_span} is outside the cylinder (0..{self.length:.3f} m)")
        psi_span = where.get("psi", (-90.0, 90.0))
        if psi_span[0] < -90 or psi_span[1] > 90:
            raise ValueError(f"psi {psi_span} is outside -90..90 deg")
        phi_span = where.get("phi", (0.0, 360.0))
        return {
            "t": GraspParameter("t", float(t_span[0]), float(t_span[1]), unit="m"),
            "phi": GraspParameter("phi", float(phi_span[0]), float(phi_span[1]), period=360.0, unit="deg"),
            "psi": GraspParameter("psi", float(psi_span[0]), float(psi_span[1]), unit="deg"),
        }

    def grasp_at(self, params: Mapping[str, float]) -> AntipodalGrasp:
        """The grasp at `t` along the axis, from side `phi`, its approach tilted `psi` (the class docstring)."""
        t, phi, psi = float(params["t"]), float(params["phi"]), float(params["psi"])
        axis_root = self.axis_root
        center_root = self.bottom_root + t * axis_root
        side_root = self.root_from_obj[:3, :3] @ np.array(
            [np.cos(np.radians(phi)), np.sin(np.radians(phi)), 0.0]
        )
        closing_root = np.cross(axis_root, side_root)
        p = np.radians(psi)
        approach_root = -np.cos(p) * side_root - np.sin(p) * axis_root
        named = {"t": round(t, 4), "phi": round(phi, 2), "psi": round(psi, 2)}
        return AntipodalGrasp(center_root, closing_root, approach_root, 2.0 * self.radius, named)

    def antipodal_grasps(
        self, where: Ranges | None = None, spacing: Spacing | None = None, max_opening: float | None = None
    ) -> list[AntipodalGrasp]:
        """The grasps on the grid of `t` x `phi` x `psi`, within `where`; none if the diameter is over `max_opening`."""
        if max_opening is not None and 2.0 * self.radius > max_opening:
            return []
        space, step = self.grasp_space(where), {**self.SPACING, **(spacing or {})}
        span = {name: (q.low, q.high) for name, q in space.items()}
        return [
            self.grasp_at({"t": t, "phi": phi, "psi": psi})
            for t in _grid(span["t"], step["t"])
            for phi in _grid(span["phi"], step["phi"], wrap=360.0)
            for psi in _grid(span["psi"], step["psi"])
        ]

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


SHAPES = {Cylinder.SHAPE: Cylinder}  # the shapes a record can name


@dataclass(frozen=True)
class RigidBody:
    """A primitive with a mass: an object as a simulator needs it, and as the grasp bank and the start pools record it.

    The record (`to_record`) is the shape's name (`"shape"`), its parameters by name (`Cylinder.PARAMS`), the mass, and
    where it stands: `bottom_root`, standing upright on it (a cylinder's bottom end's center), or the full pose
    (`root_from_obj`) when it is not upright. `vector` / `from_vector` are the same numbers as a row (the start pools'
    arrays: `columns` names them).
    """

    shape: Cylinder
    mass_kg: float

    @property
    def inertia(self) -> np.ndarray:
        """Its principal moments of inertia about its center (a solid of uniform density, kg m^2)."""
        return self.shape.solid_inertia(self.mass_kg)

    @classmethod
    def upright(
        cls, shape: str, params: Mapping[str, float], mass_kg: float, bottom_root: np.ndarray
    ) -> "RigidBody":
        """`shape` of `params`, standing upright on `bottom_root`."""
        frame = frame_along(np.array([0.0, 0.0, 1.0]), np.asarray(bottom_root, dtype=float))
        return cls(SHAPES[shape].from_shape_params(params, frame), float(mass_kg))

    @staticmethod
    def columns(shape: str = Cylinder.SHAPE) -> tuple[str, ...]:
        """`vector`'s names: the shape's parameters, then the mass."""
        return (*SHAPES[shape].PARAMS, "mass_kg")

    def vector(self) -> np.ndarray:
        """Its shape's parameters and its mass as a row (`columns`)."""
        return np.array([*self.shape.shape_params().values(), self.mass_kg], dtype=np.float64)

    @classmethod
    def from_vector(
        cls, row: np.ndarray, shape: str = Cylinder.SHAPE, bottom_root: np.ndarray | None = None
    ) -> "RigidBody":
        """`vector`'s row back (standing upright on `bottom_root`, default the origin)."""
        names = SHAPES[shape].PARAMS
        params = dict(zip(names, (float(v) for v in row[: len(names)]), strict=True))
        return cls.upright(
            shape, params, float(row[len(names)]), np.zeros(3) if bottom_root is None else bottom_root
        )

    def to_record(self) -> dict[str, object]:
        """Its record (the class docstring): plain JSON values."""
        frame = self.shape.root_from_obj
        record: dict[str, object] = {
            "shape": self.shape.SHAPE,
            **self.shape.shape_params(),
            "mass_kg": self.mass_kg,
        }
        if np.allclose(frame[:3, 2], [0.0, 0.0, 1.0], atol=1e-9):
            record["bottom_root"] = [float(v) for v in frame[:3, 3]]
        else:
            record["root_from_obj"] = frame.tolist()
        return record

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> "RigidBody":
        """`to_record`'s record back; one without a shape (written before shapes were named) is a cylinder."""
        shape = str(record.get("shape", Cylinder.SHAPE))
        if shape not in SHAPES:
            raise ValueError(f"unknown shape {shape!r} (known: {sorted(SHAPES)})")
        params = {name: float(record[name]) for name in SHAPES[shape].PARAMS}  # type: ignore[arg-type]
        mass = float(record["mass_kg"])  # type: ignore[arg-type]
        if "root_from_obj" in record:
            return cls(SHAPES[shape].from_shape_params(params, np.array(record["root_from_obj"])), mass)
        return cls.upright(shape, params, mass, np.array(record["bottom_root"]))
