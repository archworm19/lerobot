"""A wrist scan: frames from several wrist poses, their real poses from structure from motion, and 3D from them.

`docs/object_grasping.md` section 6.1-6.2. The wrist camera has no depth; a sweep of the wrist (roll and flex only:
the joints that carry the least load, the arm above them still) gives frames from several places, and
`sfm.refine` measures how the camera really moved between them. A `WristScan` then answers for the model what a
head capture's depth does:

- **Frames to look at:** `render` draws a contact sheet of the frames (or one, full size) with what has been
  measured drawn in each: a point or a fitted outline that sits right in every frame is the check.
- **Points:** `locate` gives a pixel's 3D point: from the scan's scene points around it (the table, textured
  surfaces), or from the same point marked in a second frame (triangulated: an object's feature, where a white side
  has no points of its own); `find` looks for the second frame's pixel itself (along the epipolar line). With how
  well it is known.
- **Objects:** `fit_upright_cylinder` fits an upright cylinder to its sides marked in a few frames (each side a ray
  tangent to it), its bottom on the table plane: an `Object`, which `grasp_candidates` plans on as on the head's.

**Frames.** Answers come in the pinch frame of the frame the arm is at now (`current`: after a sweep, back at its
start): where things are relative to the jaws, which is what the next move needs. And in the root frame through that
pinch frame as the model has it at the joints now (`to_root`), so a plan made there and run from where the arm is
cancels the model's error. **The scale** of everything is the model's (the images fix the motion up to one factor):
a few percent of the distances, said in the descriptions.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import cv2
import numpy as np

from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel
from lerobot.utils.import_utils import _scipy_available, require_package

from .objects import Object
from .primitives import Cylinder
from .sfm import PRIOR_DEG, PRIOR_MM, SfmResult, refine
from .wrist import WristView, move_closely, wrist_view

if TYPE_CHECKING or _scipy_available:
    from scipy.optimize import least_squares

if TYPE_CHECKING:
    from lerobot.felix_jr.server.client import FelixClient

LOOKUP_PX = 25.0  # scene points this close to a pixel (in its frame) give its depth
MIN_LOOKUP_POINTS = 4
FIND_PATCH_PX = 21  # `find`'s template, square
FIND_DEPTHS_M = (0.04, 0.6)  # the range searched along the epipolar line
FIND_MIN_NCC = 0.7
SCALE_NOTE = "the scale is the model's: a few % of the distance"
SWEEP_ROLLS_DEG = (-30.0, -15.0, 15.0, 30.0)
SWEEP_FLEXES_DEG = (-10.0, 10.0)
MAX_ROLL_DEG = 130.0  # the wrist camera's cable (as `calibration.views`)
SWEEP_VEL = 15.0


def _cm(value: float) -> str:
    return f"{100 * value:.1f}"


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


@dataclass(frozen=True, eq=False)
class WristScan:
    """Wrist frames and their refined poses (`scan_views`, `wrist_scan`). See the module docstring."""

    sensed: ClassVar[bool] = True  # the scene now (`Workspace.add`)
    views: tuple[WristView, ...]
    sfm: SfmResult
    current: int  # the frame the arm is at now
    labels: tuple[str, ...]  # each frame's, e.g. "roll +15"
    scale_note: str = SCALE_NOTE  # where the scale comes from (`rescale`)

    @property
    def camera_from_pinch(self) -> np.ndarray:
        """The pinch frame in the camera's: rigid on the gripper."""
        return np.linalg.inv(self.views[self.current].pinch_from_camera)

    def camera(self, frame: int) -> np.ndarray:
        """root_from_camera of `frame`, refined (the scan's frame: the model's, corrected)."""
        return self.sfm.root_from_camera[frame]

    def to_pinch(self, points: np.ndarray) -> np.ndarray:
        """Scan points (N, 3) in the current pinch frame."""
        pinch = self.camera(self.current) @ self.camera_from_pinch
        inv = np.linalg.inv(pinch)
        return np.atleast_2d(points) @ inv[:3, :3].T + inv[:3, 3]

    def to_root(self, points: np.ndarray) -> np.ndarray:
        """Scan points (N, 3) in the root frame, through the current pinch frame as the model has it now."""
        pinch = self.views[self.current].root_from_pinch
        return self.to_pinch(points) @ pinch[:3, :3].T + pinch[:3, 3]

    def ray(self, frame: int, uv: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        """The camera's position and the unit ray through pixel `uv` of `frame` (scan frame)."""
        pose = self.camera(frame)
        ray = self.views[frame].rays(np.asarray(uv, dtype=np.float64).reshape(1, 2))[0]
        return pose[:3, 3].copy(), _unit(pose[:3, :3] @ ray)

    def project(self, frame: int, points: np.ndarray) -> np.ndarray:
        """Pixels (N, 2) of scan points (N, 3) in `frame`; NaN behind the camera."""
        inv = np.linalg.inv(self.camera(frame))
        camera_points = np.atleast_2d(points) @ inv[:3, :3].T + inv[:3, 3]
        uv = self.views[frame].camera.project(camera_points)
        uv[camera_points[:, 2] <= 0] = np.nan
        return uv

    def describe(self) -> str:
        """The frames, the fit, the arm's motion error, the table."""
        return (
            f"wrist scan, {len(self.views)} frames ({', '.join(f'{i}: {label}' for i, label in enumerate(self.labels))}"
            f"; the arm is at frame {self.current})\n{self.sfm.describe()}\n{self.scale_note}"
        )


@dataclass(frozen=True, eq=False)
class ScanPoint:
    """A point from a scan (`locate`, `find`): in the current pinch frame and the root frame, and how well known."""

    label: str
    scan_xyz: np.ndarray  # (3,) the scan's frame
    pinch_xyz: np.ndarray  # (3,) the current pinch frame
    root_xyz: np.ndarray  # (3,) the root frame, through the pinch frame as the model has it now
    how: str  # how it was found, in words
    uncertainty_m: float  # rough, from the method (not counting the scale)
    scale_note: str = SCALE_NOTE  # the scan's (`WristScan.scale_note`)

    def describe(self) -> str:
        """Where (pinch and root frames), from what, and how well."""
        p, r = self.pinch_xyz, self.root_xyz
        return (
            f"{self.label}: ({_cm(p[0])}, {_cm(p[1])}, {_cm(p[2])}) cm in the jaws' (pinch) frame, "
            f"({_cm(r[0])}, {_cm(r[1])}, {_cm(r[2])}) cm root; {self.how}; good to ~{1000 * self.uncertainty_m:.0f} "
            f"mm ({self.scale_note})"
        )


def _point(scan: WristScan, label: str, xyz: np.ndarray, how: str, uncertainty: float) -> ScanPoint:
    return ScanPoint(
        label, xyz, scan.to_pinch(xyz)[0], scan.to_root(xyz)[0], how, uncertainty, scan.scale_note
    )


# ---- making a scan ------------------------------------------------------------------------------------------------


def scan_views(
    views: Sequence[WristView],
    method: str = "light",
    current: int = -1,
    labels: Sequence[str] | None = None,
    prior_mm: float = PRIOR_MM,
    prior_deg: float = PRIOR_DEG,
) -> WristScan:
    """A scan of `views` (one arm's frames from different poses); `current`: the frame the arm is at now."""
    if len({v.arm for v in views}) != 1:
        raise ValueError("views of one wrist camera")
    current = current % len(views)
    result = refine(
        [v.color for v in views],
        views[0].camera,
        [v.root_from_camera for v in views],
        method,
        prior_mm,
        prior_deg,
        np.linalg.inv(views[current].pinch_from_camera),
    )
    names = tuple(labels) if labels is not None else tuple(f"view {i}" for i in range(len(views)))
    return WristScan(tuple(views), result, current, names)


def sweep(
    client: "FelixClient",
    model: XleRobotModel,
    arm: str,
    rolls_deg: Sequence[float] = SWEEP_ROLLS_DEG,
    flexes_deg: Sequence[float] = SWEEP_FLEXES_DEG,
    max_vel: float = SWEEP_VEL,
) -> tuple[list[WristView], list[str]]:
    """Frames from the wrist turned by `rolls_deg` and bent by `flexes_deg` about where it is, then back.

    Only the wrist moves (each move pushed again where a joint stops short). The first frame is where it started, the
    last where it is back: the two say how well it returned.
    """
    key = f"{arm}_arm"
    start = {k: float(v) for k, v in client.get_state()[key].items() if k != "gripper.pos"}
    targets: list[tuple[str, dict[str, float]]] = []
    for roll in rolls_deg:
        value = float(np.clip(start["wrist_roll.pos"] + roll, -MAX_ROLL_DEG, MAX_ROLL_DEG))
        targets.append((f"roll {roll:+.0f}", {**start, "wrist_roll.pos": value}))
    for flex in flexes_deg:
        targets.append((f"flex {flex:+.0f}", {**start, "wrist_flex.pos": start["wrist_flex.pos"] + flex}))
    views = [wrist_view(model, client.wrist_capture(arm))]
    labels = ["start"]
    with client.session():
        for label, target in [*targets, ("back", start)]:
            status, _ = move_closely(client, key, target, max_vel)
            views.append(wrist_view(model, client.wrist_capture(arm, after=status.get("finished_at"))))
            labels.append(label)
        client.keep_holding()
    return views, labels


# ---- the scale ----------------------------------------------------------------------------------------------------


def rescale(scan: WristScan, factor: float, note: str) -> WristScan:
    """`scan` with its geometry scaled by `factor` about the current camera (poses' positions, points, table)."""
    from dataclasses import replace

    center = scan.camera(scan.current)[:3, 3]
    poses = scan.sfm.root_from_camera.copy()
    poses[:, :3, 3] = center + factor * (poses[:, :3, 3] - center)
    points = center + factor * (scan.sfm.points - center)
    plane = None
    if scan.sfm.plane is not None:
        normal, offset = scan.sfm.plane
        plane = (normal, float(normal @ center + factor * (offset - normal @ center)))
    return replace(
        scan, sfm=replace(scan.sfm, root_from_camera=poses, points=points, plane=plane), scale_note=note
    )


def scale_from_table(scan: WristScan, table_z: float) -> WristScan:
    """The scan scaled so its table is as far below the current camera as `table_z` is below the model's camera.

    `table_z`: the table's height (root frame, m), from the head's depth. Relies on the model only for the camera's
    height now (~1 cm).
    """
    if scan.sfm.plane is None:
        raise ValueError("the scan found no table plane")
    normal, offset = scan.sfm.plane
    center = scan.camera(scan.current)[:3, 3]
    seen = float(normal @ center - offset)  # the camera's height over the table, in the scan
    true = float(scan.views[scan.current].root_from_camera[2, 3] - table_z)
    if seen <= 0 or true <= 0:
        raise ValueError("the table is not below the camera")
    factor = true / seen
    return rescale(
        scan,
        factor,
        f"the scale set by the table's height ({_cm(table_z)} cm): x{factor:.3f} (the model's camera height, ~1 cm)",
    )


def scale_from_width(
    scan: WristScan, sides: Mapping[int, Sequence[Sequence[float]]], width: float
) -> WristScan:
    """The scan scaled so an upright cylinder fitted to `sides` (free radius) is `width` wide (m: the head's fit).

    Independent of the arm's model: the object's own size.
    """
    fitted = fit_upright_cylinder(scan, sides, height=0.05).part()
    assert isinstance(fitted, Cylinder)
    factor = width / (2 * fitted.radius)
    return rescale(scan, factor, f"the scale set by the object's width ({_cm(width)} cm): x{factor:.3f}")


# ---- points -------------------------------------------------------------------------------------------------------


def locate(
    scan: WristScan,
    frame: int,
    uv: Sequence[float],
    frame_b: int | None = None,
    uv_b: Sequence[float] | None = None,
    label: str = "point",
) -> ScanPoint:
    """The 3D point at pixel `uv` of `frame`: by the scene's points around it, or triangulated with `uv_b` in `frame_b`.

    Alone, the scan's points that land within `LOOKUP_PX` of the pixel give a local plane the ray meets (the table, a
    textured side), else the table plane; with a second mark, the two rays' closest point.
    """
    origin, ray = scan.ray(frame, uv)
    if frame_b is not None and uv_b is not None:
        origin_b, ray_b = scan.ray(frame_b, uv_b)
        return _triangulated(scan, label, origin, ray, origin_b, ray_b, scan.views[frame_b].camera.fx)
    points = scan.sfm.points
    near = np.linalg.norm(scan.project(frame, points) - np.asarray(uv, dtype=np.float64), axis=1) < LOOKUP_PX
    if near.sum() >= MIN_LOOKUP_POINTS:
        local = points[near]
        center = local.mean(axis=0)
        normal = np.linalg.svd(local - center)[2][-1]
        spread = float(np.sqrt(np.mean(((local - center) @ normal) ** 2)))
        if abs(ray @ normal) > 0.2:
            t = float((center - origin) @ normal / (ray @ normal))
            if t > 0:
                return _point(
                    scan,
                    label,
                    origin + t * ray,
                    f"the scene's {int(near.sum())} points around it",
                    max(spread, 0.001),
                )
    if scan.sfm.plane is not None:
        normal, offset = scan.sfm.plane
        if abs(ray @ normal) > 0.1:
            t = float((offset - normal @ origin) / (ray @ normal))
            if t > 0:
                return _point(scan, label, origin + t * ray, "the table plane", 0.003)
    raise ValueError(
        "no scene points near it and no table under it: mark it in a second frame (or `find` it)"
    )


def _triangulated(
    scan: WristScan,
    label: str,
    origin_a: np.ndarray,
    a: np.ndarray,
    origin_b: np.ndarray,
    b: np.ndarray,
    fx: float,
    how: str = "",
) -> ScanPoint:
    cos = float(a @ b)
    if cos > 1 - 1e-8:
        raise ValueError("the two frames see it along the same ray: pick frames further apart")
    w = origin_a - origin_b
    s = (cos * float(b @ w) - float(a @ w)) / (1 - cos * cos)
    t = (float(b @ w) - cos * float(a @ w)) / (1 - cos * cos)
    if s <= 0 or t <= 0:
        raise ValueError("the rays meet behind a camera: the two marks are not the same point")
    pa, pb = origin_a + s * a, origin_b + t * b
    parallax = float(np.arccos(np.clip(cos, -1.0, 1.0)))
    per_px = t / (fx * np.sin(parallax))
    miss = float(np.linalg.norm(pa - pb))
    return _point(
        scan,
        label,
        (pa + pb) / 2,
        (how or "two marks")
        + f", the rays {np.degrees(parallax):.1f} deg apart passing {1000 * miss:.0f} mm apart",
        max(per_px, miss),
    )


def find(
    scan: WristScan, frame: int, uv: Sequence[float], in_frame: int | None = None, label: str = "point"
) -> ScanPoint:
    """`locate` with the second mark found by the tool: the pixel's patch searched along its epipolar line.

    In `in_frame` (default: the frame whose camera is farthest from `frame`'s, for the most parallax), the patch turned
    by the two frames' relative roll and compared (normalized correlation) at each depth in `FIND_DEPTHS_M`. Refused
    when the best match is weak or not unique (a smooth or repeating surface): mark the second frame by hand.
    """
    if in_frame is None:
        centers = np.array([scan.camera(i)[:3, 3] for i in range(len(scan.views))])
        in_frame = int(np.argmax(np.linalg.norm(centers - centers[frame], axis=1)))
    origin, ray = scan.ray(frame, uv)
    gray_a = cv2.cvtColor(scan.views[frame].color, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_b = cv2.cvtColor(scan.views[in_frame].color, cv2.COLOR_RGB2GRAY).astype(np.float32)
    half = FIND_PATCH_PX // 2
    # the template, turned into the other frame's orientation (by the refined poses' relative roll)
    relative = scan.camera(frame)[:3, :3].T @ scan.camera(in_frame)[:3, :3]
    roll = float(np.degrees(np.arctan2(relative[1, 0], relative[0, 0])))
    big = 2 * half
    u, v = (int(round(c)) for c in uv)
    if not (big <= u < gray_a.shape[1] - big and big <= v < gray_a.shape[0] - big):
        raise ValueError("the pixel is too near the picture's edge to find")
    crop = gray_a[v - big : v + big + 1, u - big : u + big + 1]
    turned = cv2.warpAffine(crop, cv2.getRotationMatrix2D((big, big), roll, 1.0), crop.shape[::-1])
    template = turned[big - half : big + half + 1, big - half : big + half + 1]
    template = (template - template.mean()) / (template.std() + 1e-6)
    near, far = FIND_DEPTHS_M
    ends = scan.project(in_frame, np.array([origin + near * ray, origin + far * ray]))
    if not np.isfinite(ends).all():
        raise ValueError(f"the pixel's ray is not in front of frame {in_frame}'s camera")
    steps = int(np.ceil(np.linalg.norm(ends[1] - ends[0]))) + 1
    depths = near * (far / near) ** np.linspace(0.0, 1.0, max(steps, 2))  # ~1 px apart, more at the near end
    candidates = scan.project(in_frame, origin + depths[:, None] * ray)
    scores = np.full(len(depths), -1.0)
    for k, (cu, cv) in enumerate(candidates):
        iu, iv = int(round(cu)), int(round(cv))
        if half <= iu < gray_b.shape[1] - half and half <= iv < gray_b.shape[0] - half:
            patch = gray_b[iv - half : iv + half + 1, iu - half : iu + half + 1]
            patch = (patch - patch.mean()) / (patch.std() + 1e-6)
            scores[k] = float((patch * template).mean())
    best = int(np.argmax(scores))
    if scores[best] < FIND_MIN_NCC:
        raise ValueError(
            f"not found in frame {in_frame} (best correlation {scores[best]:.2f}): mark it by hand"
        )
    far_from_best = np.linalg.norm(candidates - candidates[best], axis=1) > FIND_PATCH_PX
    if far_from_best.any() and scores[far_from_best].max() > scores[best] - 0.05:
        raise ValueError(
            f"found more than once in frame {in_frame} (a repeating or smooth surface): mark it by hand"
        )
    origin_b, ray_b = scan.ray(in_frame, candidates[best])
    return _triangulated(
        scan,
        label,
        origin,
        ray,
        origin_b,
        ray_b,
        scan.views[in_frame].camera.fx,
        f"found in frame {in_frame} at ({candidates[best][0]:.0f}, {candidates[best][1]:.0f}) "
        f"(correlation {scores[best]:.2f})",
    )


# ---- objects ------------------------------------------------------------------------------------------------------


def fit_upright_cylinder(
    scan: WristScan,
    sides: Mapping[int, Sequence[Sequence[float]]],
    radius: float | None = None,
    top: tuple[int, Sequence[float]] | None = None,
    height: float | None = None,
    label: str = "object",
) -> Object:
    """An upright cylinder from its sides marked in a few frames (`sides`: {frame: [[u, v], [u, v]]}).

    Each side is a ray tangent to it: the vertical axis (and the radius, unless given) is fitted so every ray passes
    the radius from it. Its bottom is on the table plane; its top where the axis meets `top`'s ray (a frame and a
    pixel: the top's center) or `height` above the bottom. In the root frame through the jaws now (`WristScan.to_root`):
    plan on it with `grasp_candidates` (the head's fresh `capture` for the arm's pose). The note gives the fit's error.
    """
    require_package("scipy", extra="felix_jr")
    rays = [scan.ray(int(f), uv) for f, pair in sides.items() for uv in pair]
    if len(rays) < 3:
        raise ValueError("mark the sides in at least two frames")
    if scan.sfm.plane is None and top is None:
        raise ValueError("no table plane in the scan: give its top (`top`) and `height`")
    z = np.array([0.0, 0.0, 1.0])

    def distances(cx: float, cy: float) -> np.ndarray:
        out = []
        for origin, d in rays:
            n = np.cross(z, d)
            out.append(abs((origin - np.array([cx, cy, 0.0])) @ n) / np.linalg.norm(n))
        return np.array(out)

    # start: where the sides' bisectors of the first two marked frames cross (horizontally)
    frames = list(sides)
    bisectors = []
    for f in frames[:2]:
        (o1, d1), (_, d2) = scan.ray(int(f), sides[f][0]), scan.ray(int(f), sides[f][1])
        bisectors.append((o1, _unit(d1 + d2)))
    (oa, da), (ob, db) = bisectors
    a2, b2 = da[:2], db[:2]
    matrix = np.column_stack([a2, -b2])
    try:
        s, _ = np.linalg.solve(matrix, (ob - oa)[:2])
        start = (oa + s * da)[:2]
    except np.linalg.LinAlgError:
        start = (oa + 0.15 * da)[:2]
    r0 = radius if radius is not None else float(np.median(distances(*start)))

    def residuals(x: np.ndarray) -> np.ndarray:
        r = radius if radius is not None else x[2]
        return distances(x[0], x[1]) - r

    x0 = np.array([*start] + ([] if radius is not None else [r0]))
    result = least_squares(residuals, x0, x_scale=0.01)
    cx, cy = result.x[:2]
    r = float(radius if radius is not None else result.x[2])
    if r <= 0:
        raise ValueError("the fit came out with no radius: mark the sides again")
    if scan.sfm.plane is not None:
        normal, offset = scan.sfm.plane
        bottom_z = float((offset - normal[0] * cx - normal[1] * cy) / normal[2])
    else:
        bottom_z = None
    top_z = None
    if top is not None:
        origin, d = scan.ray(int(top[0]), top[1])
        # the point of the axis closest to the top's ray
        w = np.array([cx, cy, 0.0]) - origin
        b = float(z @ d)
        top_z = float((b * float(d @ w) - float(z @ w)) / (1 - b * b))
    if bottom_z is None:
        if top_z is None or height is None:
            raise ValueError("no table plane: give `top` and `height`")
        bottom_z = top_z - height
    length = height if height is not None else (top_z - bottom_z if top_z is not None else None)
    if length is None or length <= 0:
        raise ValueError("its height is not known: give `top` or `height`")
    bottom_scan = np.array([cx, cy, bottom_z])
    bottom_root = scan.to_root(bottom_scan)[0]
    rms_mm = 1000 * float(np.sqrt(np.mean(result.fun**2)))
    note = (
        f"fitted from {len(rays)} sides in {len(sides)} frames of a wrist scan: the rays {rms_mm:.1f} mm RMS from "
        f"tangent; radius {_cm(r)} cm{' (given)' if radius is not None else ''}, {_cm(length)} cm tall; "
        f"{scan.scale_note}"
    )
    return Object(label, {"body": Cylinder.upright(bottom_root, float(length), r)}, None, note)


# ---- pictures -----------------------------------------------------------------------------------------------------


def render(
    scan: WristScan,
    frames: Sequence[int] | None = None,
    points: Sequence[ScanPoint] = (),
    objects: Sequence[Object] = (),
    tile_width: int = 320,
) -> np.ndarray:
    """The scan's frames with `points` and `objects` drawn in each: one frame full size, several as a contact sheet."""
    chosen = list(range(len(scan.views))) if frames is None else [int(f) % len(scan.views) for f in frames]
    to_scan = np.linalg.inv(scan.views[scan.current].root_from_pinch) if objects else None
    tiles = []
    for f in chosen:
        image = scan.views[f].color.copy()
        for p in points:
            uv = scan.project(f, p.scan_xyz[None, :])[0]
            if np.isfinite(uv).all():
                center = (int(round(uv[0])), int(round(uv[1])))
                cv2.drawMarker(image, center, (0, 255, 0), cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
                cv2.putText(
                    image,
                    p.label,
                    (center[0] + 8, center[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )
        for obj in objects:
            for part in obj.parts.values():
                if isinstance(part, Cylinder) and to_scan is not None:
                    _draw_cylinder(image, scan, f, part, to_scan)
        label = f"{f}: {scan.labels[f]}" + (" (now)" if f == scan.current else "")
        cv2.putText(image, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(image)
    if len(tiles) == 1:
        return tiles[0]
    scale = tile_width / tiles[0].shape[1]
    small = [cv2.resize(t, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for t in tiles]
    columns = min(3, len(small))
    rows = -(-len(small) // columns)
    blank = np.zeros_like(small[0])
    small += [blank] * (rows * columns - len(small))
    return np.vstack([np.hstack(small[r * columns : (r + 1) * columns]) for r in range(rows)])


def _draw_cylinder(
    image: np.ndarray, scan: WristScan, frame: int, cylinder: Cylinder, root_to_pinch: np.ndarray
) -> None:
    """The cylinder's end circles and its two silhouette lines as `frame` sees it, in orange.

    `root_to_pinch`: the root frame to the current pinch frame as the model has it (`to_root`'s inverse); from there
    the refined pinch frame puts it in the scan's frame.
    """
    to_scan = scan.camera(scan.current) @ scan.camera_from_pinch @ root_to_pinch
    pose = to_scan @ cylinder.root_from_obj  # the cylinder's frame in the scan's
    axis, bottom, r = pose[:3, 2], pose[:3, 3], cylinder.radius
    ring = np.linspace(0.0, 2 * np.pi, 48)
    orange = (255, 150, 0)
    for height in (0.0, cylinder.length):
        local = np.column_stack([r * np.cos(ring), r * np.sin(ring), np.full(48, height)])
        uv = scan.project(frame, local @ pose[:3, :3].T + pose[:3, 3])
        good = np.isfinite(uv).all(axis=1)
        if good.sum() > 2:
            cv2.polylines(image, [uv[good].round().astype(np.int32)], True, orange, 2, cv2.LINE_AA)
    # the silhouette: the two lines along the side tangent to the rays from this camera
    across = scan.camera(frame)[:3, 3] - bottom
    across = across - (across @ axis) * axis
    d = float(np.linalg.norm(across))
    if d <= r:
        return
    e = across / d
    w = np.cross(axis, e)
    for sign in (1.0, -1.0):
        foot = bottom + (r * r / d) * e + sign * r * np.sqrt(1 - r * r / (d * d)) * w
        uv = scan.project(frame, np.array([foot, foot + cylinder.length * axis]))
        if np.isfinite(uv).all():
            a, b = (tuple(int(round(c)) for c in p) for p in uv)
            cv2.line(image, a, b, orange, 2, cv2.LINE_AA)
