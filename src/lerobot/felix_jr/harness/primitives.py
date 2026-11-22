"""Shapes an object is made of, the grasps each allows, and how an object is written down.

`Primitive`, `GraspFamily` and `ObjectRecord` (`docs/object_grasping.md`, section 2).

**A primitive is geometry, batched.** A `Primitive` is N objects of one shape: their parameters (`params`, (N, k):
a cylinder's radius and height, `PARAMS`) and their poses in the root frame (`root_from_obj`, (N, 4, 4): each shape's
own axes and origin). One object is a batch of one. Everything it says is a geometric fact, for all N at once: its
centroid, volume, bounds (`bounding_radius`, `half_extents`), inertia per kilogram (a solid of uniform density), the
signed distance of points to its surface. It knows nothing about the robot, a simulator, or a mass. The arrays are
numpy (planning) or torch (the mjlab simulator, on the GPU): the geometry is written once against the Python array
API standard (`namespace`: numpy itself, or torch through `array_api_compat`), so a batch of thousands of worlds on the
GPU is computed there, never copied to the host.

**A grasp family is planning's, one object at a time.** A `GraspFamily` (`grasp_family(primitive)`; a cylinder's
`CylinderGrasps`) is the antipodal grasps one object (a batch of one) allows: a few named parameters (a cylinder's `t`,
`phi`, `psi`), generated on a grid so the same shape always gives the same grasps, and its preferred way in
(`approach_preference`). A grasp is antipodal: two contacts with opposed normals on one closing line. Which contact
takes the fixed jaw, how deep the jaws go and whether the arm reaches are decided later (`grasping.py`). `where`
narrows a family to ranges of its parameters, `spacing` sets the grid's step per parameter; both are keyed by the
parameter names the family documents (angles in degrees, lengths in meters). A range's ends are always on the grid. The
family is also continuous: `grasp_space` names its parameters with their ranges, `grasp_at` makes the grasp at any
values in it (what an optimizer samples: `sim2real.refine`); the grid is `grasp_at` on `_grid`s.

**An object record is a primitive with a mass, written down.** An `ObjectRecord` is a batch of primitives and each
one's mass (a primitive has none: a fitted one's is unknown, a simulator's is drawn). Its record (`to_record`,
`from_record`: the shape's name, its parameters by name, the mass, where it stands) and its rows (`vectors`,
`from_vectors`: the parameters then the mass) are the one way an object is written down: the grasp bank's placements
(`sim2real/bank.py`) and the start pools (`sim2real/start_pool.py`) both go through it, so the simulators read the
object the bank planned on.

Names say their frame (`docs/object_grasping.md`, "Frames and names"): `root_from_obj` maps the object frame to the root
frame, `center_root` is a point in the root frame.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol

import numpy as np

Ranges = Mapping[str, tuple[float, float]]
Spacing = Mapping[str, float]
Array = Any  # a numpy array or a torch tensor (the array API standard's)


def namespace(*arrays: Array) -> Any:
    """The array API namespace of `arrays`.

    numpy for numpy arrays (numpy 2 is the standard's), else `array_api_compat`'s (torch's, under the standard's names:
    `stack(..., axis=)`, ...).
    """
    if all(isinstance(a, np.ndarray | np.generic | float | int) for a in arrays):
        return np
    from lerobot.utils.import_utils import require_package

    require_package("array-api-compat", extra="felix_jr", import_name="array_api_compat")
    from array_api_compat import array_namespace

    return array_namespace(*arrays)


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
        """Signed distance (m) of each point (M, 3) to the surface: negative inside (or below).

        (M,), or (N, M) for a batch of N (`Primitive`: each object's).
        """
        ...

    def describe(self) -> str:
        """The surface and where it is, in words and numbers."""
        ...


class Primitive(Surface, Protocol):
    """N objects of one shape: parameters (N, k) and poses (N, 4, 4), numpy or torch; geometry only.

    Every method answers for all N at once, in the arrays' own library and device (the module docstring).
    """

    SHAPE: ClassVar[str]  # the shape's name (a record's `"shape"`)
    PARAMS: ClassVar[tuple[str, ...]]  # its parameters' names, `params`' columns

    @property
    def params(self) -> Array:
        """(N, k): each object's parameters (`PARAMS`, m)."""
        ...

    @property
    def root_from_obj(self) -> Array:
        """(N, 4, 4): each object's own axes and origin in the root frame."""
        ...

    @classmethod
    def from_params(cls, params: Array, root_from_obj: Array | None = None) -> "Primitive":
        """`params` (N, k), at `root_from_obj` (N, 4, 4; default each at the origin)."""
        ...

    def __len__(self) -> int:
        """How many objects (N)."""
        ...

    def __getitem__(self, index: Any) -> "Primitive":
        """The objects at `index` as a batch (an integer: a batch of one)."""
        ...

    def centroid_obj(self) -> Array:
        """(N, 3): each object's centroid (a solid's center of mass), object frame."""
        ...

    def volume(self) -> Array:
        """(N,): each object's volume (m^3)."""
        ...

    def bounding_radius(self) -> Array:
        """(N,): the smallest sphere about the centroid that holds each object (m)."""
        ...

    def half_extents(self) -> Array:
        """(N, 3): the half sizes of the box about the centroid that holds each object, object frame (m)."""
        ...

    def inertia_per_kg(self) -> Array:
        """(N, 3): a uniform solid's principal moments of inertia per kilogram (m^2): times a mass, its inertia.

        About its centroid, along the object frame's axes.
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


# ---- the cylinder ---------------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Cylinder:
    """N solid cylinders, each along its object frame's z.

    Each axis runs from the origin (the "bottom" end: on the table for an upright object) to `height` along it (end to
    end; a lying cylinder's length, `length`). `params` (N, 2): radius, height (m); `root_from_obj` (N, 4, 4). One cylinder is a batch of one: `upright` and
    `between` make one from a point and sizes (or N from arrays of them). Its grasps: `CylinderGrasps`.
    """

    params: Array  # (N, 2): radius, height
    root_from_obj: Array  # (N, 4, 4)

    SHAPE: ClassVar[str] = "cylinder"
    PARAMS: ClassVar[tuple[str, ...]] = ("radius", "height")

    def __post_init__(self) -> None:
        """Refuse shapes that are not batches, frames that are not rigid transforms, sizes that are not positive.

        The values are checked on numpy arrays only (planning's): a GPU batch's shapes.
        """
        xp = namespace(self.params, self.root_from_obj)
        if self.params.ndim != 2 or self.params.shape[1] != 2:
            raise ValueError(f"params must be (N, 2): radius, height; got {tuple(self.params.shape)}")
        if self.root_from_obj.shape != (self.params.shape[0], 4, 4):
            raise ValueError(
                f"root_from_obj must be ({self.params.shape[0]}, 4, 4), got {tuple(self.root_from_obj.shape)}"
            )
        if (
            xp is not np
        ):  # (on a GPU, checking values would make it wait for the host at every call: shapes only)
            return
        if bool(np.any(self.params <= 0)):
            raise ValueError(f"radius and height must be positive: {self.params}")
        rotation = self.root_from_obj[:, :3, :3]
        gap = np.matmul(np.swapaxes(rotation, -1, -2), rotation) - np.eye(3)
        if len(self) and float(np.max(np.abs(gap))) > 1e-6:
            raise ValueError("a cylinder's frame is not a rotation")

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Unpickle; a session saved before cylinders were batches, as a batch of one.

        (Its one cylinder's state: `root_from_obj`, `radius`, `length`.)
        """
        if "params" not in state:
            state = {
                "params": np.array([[float(state["radius"]), float(state["length"])]]),
                "root_from_obj": np.asarray(state["root_from_obj"], dtype=float)[None],
            }
        for key, value in state.items():
            object.__setattr__(self, key, value)

    # -- making them --

    @classmethod
    def from_params(cls, params: Array, root_from_obj: Array | None = None) -> "Cylinder":
        """`params` (N, 2), at `root_from_obj` (N, 4, 4; default each at the origin, its axis along the root z)."""
        if root_from_obj is None:
            xp = namespace(params)
            eye = xp.eye(4, dtype=params.dtype, device=getattr(params, "device", None))
            root_from_obj = xp.broadcast_to(eye, (params.shape[0], 4, 4))
        return cls(params, root_from_obj)

    @classmethod
    def upright(cls, bottom_root: np.ndarray, height: Any, radius: Any) -> "Cylinder":
        """Cylinders standing on their ends: each axis vertical up from `bottom_root`.

        `bottom_root` (3,) or (N, 3), the bottom end's center; `height` and `radius` a number or (N,) each. A point and
        numbers: a batch of one.
        """
        bottoms = np.atleast_2d(np.asarray(bottom_root, dtype=float))
        n = len(bottoms)
        heights = np.broadcast_to(np.asarray(height, dtype=float), (n,))
        radii = np.broadcast_to(np.asarray(radius, dtype=float), (n,))
        frames = np.stack([frame_along(np.array([0.0, 0.0, 1.0]), b) for b in bottoms])
        return cls(np.stack([radii, heights], axis=1), frames)

    @classmethod
    def between(cls, bottom_root: np.ndarray, top_root: np.ndarray, radius: float) -> "Cylinder":
        """One cylinder whose axis runs from `bottom_root` to `top_root`, its ends' centers."""
        bottom_root, top_root = np.asarray(bottom_root, dtype=float), np.asarray(top_root, dtype=float)
        length = float(np.linalg.norm(top_root - bottom_root))
        return cls(
            np.array([[float(radius), length]]), frame_along(top_root - bottom_root, bottom_root)[None]
        )

    def __len__(self) -> int:
        """How many cylinders."""
        return int(self.params.shape[0])

    def __getitem__(self, index: Any) -> "Cylinder":
        """The cylinders at `index` as a batch (an integer: a batch of one)."""
        if isinstance(index, int):
            index = slice(index, index + 1 if index != -1 else None)
        return Cylinder(self.params[index], self.root_from_obj[index])

    # -- what they are (each (N, ...)) --

    @property
    def radius(self) -> Array:
        """(N,): each radius (m)."""
        return self.params[:, 0]

    @property
    def height(self) -> Array:
        """(N,): each height, end to end along the axis (m)."""
        return self.params[:, 1]

    @property
    def length(self) -> Array:
        """(N,): `height`, the word for a cylinder lying down (a pen's length)."""
        return self.height

    @property
    def axis_root(self) -> Array:
        """(N, 3): each unit axis, bottom to top."""
        return self.root_from_obj[:, :3, 2]

    @property
    def bottom_root(self) -> Array:
        """(N, 3): each bottom end's center."""
        return self.root_from_obj[:, :3, 3]

    @property
    def top_root(self) -> Array:
        """(N, 3): each top end's center."""
        return self.bottom_root + self.height[:, None] * self.axis_root

    def centroid_obj(self) -> Array:
        """(N, 3): half way up each axis, object frame."""
        xp = namespace(self.params)
        zero = xp.zeros_like(self.radius)
        return xp.stack([zero, zero, self.height / 2], axis=1)

    def volume(self) -> Array:
        """(N,): pi r^2 h."""
        return np.pi * self.radius**2 * self.height

    def bounding_radius(self) -> Array:
        """(N,): the sphere about the centroid through the ends' rims."""
        xp = namespace(self.params)
        return xp.sqrt(self.radius**2 + (self.height / 2) ** 2)

    def half_extents(self) -> Array:
        """(N, 3): (r, r, h/2), object frame."""
        xp = namespace(self.params)
        return xp.stack([self.radius, self.radius, self.height / 2], axis=1)

    def inertia_per_kg(self) -> Array:
        """(N, 3): a solid cylinder's, about its centroid: (3 r^2 + h^2) / 12 across the axis, r^2 / 2 along it."""
        xp = namespace(self.params)
        side = (3 * self.radius**2 + self.height**2) / 12
        return xp.stack([side, side, self.radius**2 / 2], axis=1)

    def surface_distance(self, points_root: Array) -> Array:
        """(N, M): each point's (M, 3) signed distance (m) to each cylinder's surface, ends included: negative inside."""
        xp = namespace(self.params, points_root)
        points = points_root[None] if points_root.ndim == 2 else points_root  # (1 or N, M, 3)
        rel = points - self.bottom_root[:, None, :]
        points_obj = xp.matmul(rel, self.root_from_obj[:, :3, :3])  # (N, M, 3)
        radial = xp.sqrt(points_obj[..., 0] ** 2 + points_obj[..., 1] ** 2) - self.radius[:, None]
        axial = xp.maximum(-points_obj[..., 2], points_obj[..., 2] - self.height[:, None])
        outside = xp.sqrt(xp.maximum(radial, 0.0) ** 2 + xp.maximum(axial, 0.0) ** 2)
        return outside + xp.minimum(xp.maximum(radial, axial), 0.0)

    def describe(self) -> str:
        """Each cylinder's radius, bottom end, length and the axis's tilt from vertical (root frame)."""
        return "; ".join(self._describe_one(i) for i in range(len(self)))

    def _describe_one(self, i: int) -> str:
        b = np.asarray(self.bottom_root[i], dtype=float)
        a = np.asarray(self.axis_root[i], dtype=float)
        radius, length = float(self.radius[i]), float(self.height[i])
        tilt = float(np.degrees(np.arccos(np.clip(abs(a[2]), 0.0, 1.0))))
        where = f"from ({', '.join(_cm(float(v)) for v in b)}) cm (root frame), {_cm(length)} cm along "
        where += (
            "the vertical"
            if tilt < 1.0
            else f"({', '.join(f'{v:.2f}' for v in a)}), {tilt:.0f} deg from vertical"
        )
        return f"cylinder, radius {_cm(radius)} cm, {where}"


# ---- grasp families (planning) --------------------------------------------------------------------------------------


class GraspFamily(Protocol):
    """The antipodal grasps one object allows, for planning (the module docstring): built from a batch of one."""

    @property
    def primitive(self) -> Primitive:
        """The object (a batch of one)."""
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


@dataclass(frozen=True, eq=False)
class CylinderGrasps:
    """A cylinder's grasps, closing straight through the axis so the opening is the diameter.

    - `t` (m): where along the axis, from the bottom end; default the whole length, every 1 cm.
    - `phi` (deg): which side the jaws come from, around the axis from the object frame's x (for an upright cylinder
      the root x: 0 is from the robot's side, as the robot faces -x); default all the way round, every 30 deg.
    - `psi` (deg): how far the approach tilts down the axis, toward the bottom end: 0 is square to the axis, 90 along
      it from the top end (from above, for an upright cylinder), -90 from the bottom; default -90..90, every 15 deg.

    The closing direction is square to both the axis and the side the jaws come from.
    """

    primitive: Cylinder  # a batch of one

    SPACING: ClassVar[dict[str, float]] = {"t": 0.01, "phi": 30.0, "psi": 15.0}

    def __post_init__(self) -> None:
        """Refuse a batch that is not one cylinder."""
        if len(self.primitive) != 1:
            raise ValueError(f"a grasp family is one object's: got {len(self.primitive)} cylinders")

    @property
    def _frame(self) -> np.ndarray:
        return np.asarray(self.primitive.root_from_obj[0], dtype=float)

    @property
    def _radius(self) -> float:
        return float(self.primitive.radius[0])

    @property
    def _length(self) -> float:
        return float(self.primitive.height[0])

    def approach_preference(self, approach_root: np.ndarray) -> float:
        """How far (deg) an approach is off square to the axis: 0 from the side, 90 along it (from above or below).

        From the side the jaws close across it, the gripper and its wrist camera level with it; a steep approach saw only
        an upright bottle's top from the wrist camera (ruler2).
        """
        a = np.asarray(approach_root, dtype=np.float64)
        along = abs(float(a @ self._frame[:3, 2])) / float(np.linalg.norm(a))
        return float(np.degrees(np.arcsin(min(1.0, along))))

    def grasp_space(self, where: Ranges | None = None) -> dict[str, GraspParameter]:
        """`t` (m), `phi` (deg, all the way round), `psi` (deg), within `where` (the class docstring)."""
        where = dict(where or {})
        unknown = set(where) - set(self.SPACING)
        if unknown:
            raise ValueError(f"a cylinder's grasps have no parameter {sorted(unknown)}: use t, phi, psi")
        t_span = where.get("t", (0.0, self._length))
        if t_span[0] < 0 or t_span[1] > self._length + 1e-9:
            raise ValueError(f"t {t_span} is outside the cylinder (0..{self._length:.3f} m)")
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
        frame = self._frame
        axis_root = frame[:3, 2]
        center_root = frame[:3, 3] + t * axis_root
        side_root = frame[:3, :3] @ np.array([np.cos(np.radians(phi)), np.sin(np.radians(phi)), 0.0])
        closing_root = np.cross(axis_root, side_root)
        p = np.radians(psi)
        approach_root = -np.cos(p) * side_root - np.sin(p) * axis_root
        named = {"t": round(t, 4), "phi": round(phi, 2), "psi": round(psi, 2)}
        return AntipodalGrasp(center_root, closing_root, approach_root, 2.0 * self._radius, named)

    def antipodal_grasps(
        self, where: Ranges | None = None, spacing: Spacing | None = None, max_opening: float | None = None
    ) -> list[AntipodalGrasp]:
        """The grasps on the grid of `t` x `phi` x `psi`, within `where`; none if the diameter is over `max_opening`."""
        if max_opening is not None and 2.0 * self._radius > max_opening:
            return []
        space, step = self.grasp_space(where), {**self.SPACING, **(spacing or {})}
        span = {name: (q.low, q.high) for name, q in space.items()}
        return [
            self.grasp_at({"t": t, "phi": phi, "psi": psi})
            for t in _grid(span["t"], step["t"])
            for phi in _grid(span["phi"], step["phi"], wrap=360.0)
            for psi in _grid(span["psi"], step["psi"])
        ]


SHAPES: dict[str, type[Primitive]] = {Cylinder.SHAPE: Cylinder}  # the shapes a record can name
GRASP_FAMILIES: dict[str, type] = {Cylinder.SHAPE: CylinderGrasps}  # each shape's grasps


def grasp_family(primitive: Primitive) -> GraspFamily:
    """`primitive`'s grasps (a batch of one)."""
    return GRASP_FAMILIES[primitive.SHAPE](primitive)  # type: ignore[no-any-return]


# ---- object records -------------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class ObjectRecord:
    """Objects as the grasp bank and the start pools write them down: a batch of primitives and each one's mass.

    A record (`to_record`, one object): the shape's name (`"shape"`), its parameters by name (`PARAMS`), the mass
    (`"mass_kg"`), and where it stands: `bottom_root`, upright on it (a cylinder's bottom end's center), or the full pose
    (`root_from_obj`) when it is not upright. Rows (`vectors`, all of them; `columns`): the parameters, then the mass.
    """

    primitive: Primitive
    mass_kg: Array  # (N,)

    def __post_init__(self) -> None:
        """One mass an object."""
        if tuple(self.mass_kg.shape) != (len(self.primitive),):
            raise ValueError(f"{len(self.primitive)} objects with masses {tuple(self.mass_kg.shape)}")

    def __len__(self) -> int:
        """How many objects."""
        return len(self.primitive)

    def __getitem__(self, index: Any) -> "ObjectRecord":
        """The objects at `index` (an integer: a batch of one)."""
        mass = (
            self.mass_kg[index : index + 1 if index != -1 else None]
            if isinstance(index, int)
            else self.mass_kg[index]
        )
        return ObjectRecord(self.primitive[index], mass)

    @classmethod
    def upright(
        cls, shape: str, params: Mapping[str, float], mass_kg: float, bottom_root: np.ndarray
    ) -> "ObjectRecord":
        """One object of `shape` and `params` (by name), standing upright on `bottom_root`."""
        row = np.array([[float(params[name]) for name in SHAPES[shape].PARAMS]])
        frame = frame_along(np.array([0.0, 0.0, 1.0]), np.asarray(bottom_root, dtype=float))[None]
        return cls(SHAPES[shape].from_params(row, frame), np.array([float(mass_kg)]))

    def placed_upright(self, bottom_root: np.ndarray) -> "ObjectRecord":
        """The one object, the same shape, size and mass, standing upright on `bottom_root` instead."""
        if len(self) != 1:
            raise ValueError(f"one object at a time: got {len(self)}")
        p = self.primitive
        params = dict(zip(p.PARAMS, (float(v) for v in np.asarray(p.params[0])), strict=True))
        return ObjectRecord.upright(p.SHAPE, params, float(self.mass_kg[0]), bottom_root)

    @staticmethod
    def columns(shape: str = Cylinder.SHAPE) -> tuple[str, ...]:
        """`vectors`' names: the shape's parameters, then the mass."""
        return (*SHAPES[shape].PARAMS, "mass_kg")

    def vectors(self) -> Array:
        """(N, k + 1): each object's parameters and mass (`columns`)."""
        xp = namespace(self.primitive.params, self.mass_kg)
        return xp.concat([self.primitive.params, self.mass_kg[:, None]], axis=1)

    @classmethod
    def from_vectors(
        cls, rows: Array, shape: str = Cylinder.SHAPE, root_from_obj: Array | None = None
    ) -> "ObjectRecord":
        """`vectors`' rows back, at `root_from_obj`.

        Default: each at the origin, upright (where an object is, is a simulator's state).
        """
        k = len(SHAPES[shape].PARAMS)
        return cls(SHAPES[shape].from_params(rows[:, :k], root_from_obj), rows[:, k])

    def to_record(self) -> dict[str, object]:
        """The one object's record (the class docstring): plain JSON values."""
        if len(self) != 1:
            raise ValueError(f"a record is one object's: got {len(self)}")
        p = self.primitive
        frame = np.asarray(p.root_from_obj[0], dtype=float)
        values = [float(v) for v in np.asarray(p.params[0])]
        record: dict[str, object] = {"shape": p.SHAPE, **dict(zip(p.PARAMS, values, strict=True))}
        record["mass_kg"] = float(self.mass_kg[0])
        if np.allclose(frame[:3, 2], [0.0, 0.0, 1.0], atol=1e-9):
            record["bottom_root"] = [float(v) for v in frame[:3, 3]]
        else:
            record["root_from_obj"] = frame.tolist()
        return record

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "ObjectRecord":
        """`to_record`'s record back; one without a shape (written before shapes were named) is a cylinder."""
        shape = str(record.get("shape", Cylinder.SHAPE))
        if shape not in SHAPES:
            raise ValueError(f"unknown shape {shape!r} (known: {sorted(SHAPES)})")
        params = {name: float(record[name]) for name in SHAPES[shape].PARAMS}
        if "root_from_obj" in record:
            row = np.array([[params[name] for name in SHAPES[shape].PARAMS]])
            frame = np.array(record["root_from_obj"], dtype=float)[None]
            return cls(SHAPES[shape].from_params(row, frame), np.array([float(record["mass_kg"])]))
        return cls.upright(shape, params, float(record["mass_kg"]), np.array(record["bottom_root"]))
