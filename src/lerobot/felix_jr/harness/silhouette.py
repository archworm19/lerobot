"""A cylinder's silhouette in a wrist camera, and its position refined against the image's edges.

The head's depth fits an object best, but on the final approach the arm and the wrist camera's mount block its view.
The wrist cameras see the object up close: given a prior cylinder (the head's fit: its shape, size and rough place),
its silhouette in a wrist frame says where it is.

**Predict** (`silhouette`, `contour_edges`). A cylinder seen by a pinhole camera has two straight side edges in the
image: the lines on its surface where viewing rays graze it (its contour generators), parallel to its axis. In the
camera's frame, with `e` the perpendicular from the axis to the camera (length `d`), the grazing points on a cross
section are `b + (r^2/d) e_hat +- r sqrt(1 - r^2/d^2) (a x e_hat)`. Projected through the lens (`WristCamera.project`:
its distortion bends straight lines), sampled along the edge with each sample's outward normal in the image. The whole
outline (`silhouette`): the convex hull of both rims in the pinhole image, densified, then the lens applied.

**Measure** (`_find_edges`). Across each predicted edge sample, a short scan line along its normal (+- `band_px`): the
strongest colour change on it is where the object's side really is; its signed distance from the prediction (pixels,
outward positive) is the residual. A scan line off the image, masked (e.g. the fixed finger, rigid to the gripping
camera), or with no clear change gives none.

**Refine** (`refine`). The pose's parameters (the axis moved across the table, `x` and `y`; the radius too with
`fit_radius`), adjusted so the predicted edges pass through the measured points: Gauss-Newton on the residuals, their
derivatives by finite differences (the edges re-predicted at each parameter nudged), Huber weights against wrong
matches; a few rounds, each searching again around the new prediction over a narrower band. With one camera and the
radius known, the depth comes from the object's apparent width; with two cameras, from where their views cross, the
radius then refined too. The result says how well it fits (`RefineResult`: edge points found, their residuals) and
`ok` only when enough edge on both sides agrees: a clear "no estimate" rather than a confident wrong one.

Upright cylinders, their axis's direction and height kept from the prior. Frames: `root_from_camera` is the camera's
optical frame (x right, y down, z forward) in the root frame (`XleRobotModel.wrist_camera_pose`, `WristView`).
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from lerobot.felix_jr.calibration.xle_robot_model import WristCamera

from .primitives import Cylinder

EDGE_SAMPLES = 16  # scan lines along each side edge
END_MARGIN = 0.12  # of the height kept clear of each end (the rims, the table)
# each round's search half-width, at the object (m): the first covers the prior's error (the head's fits: ~1-2 cm);
# in pixels a view's own (its focal length over the object's depth)
BANDS_M = (0.02, 0.01, 0.004)
MIN_BAND_PX = 4.0
MIN_GRADIENT = 12.0  # the weakest colour change taken as an edge (summed over channels, a pixel's step)
COLOR_SIGMA = 35.0  # how far (colour units) a colour can be from the object's and still look like it
SIDE_PX = 3  # pixels averaged on each side of a found edge, to tell its inside from its outside
RADIUS_BOUNDS = (0.5, 1.5)  # the radius refit within these of the prior's
# for `ok`: a refit radius within these of the prior's (one side's edge alone lets the radius and the position trade
# off: a wrong fit that agrees with itself)
OK_RADIUS = (0.75, 1.25)
HUBER_PX = 2.0
MIN_POINTS_A_SIDE = 4  # for `ok`: edge points on each side edge, in at least one view
OK_RESIDUAL_PX = 2.5  # for `ok`: the weighted median residual after the fit


@dataclass(frozen=True)
class EdgeView:
    """A frame to refine against: its colour image, where the camera was, its lens, and the pixels not to search.

    `root_from_camera` the optical frame's pose; `mask` True where not to search (e.g. the fixed finger).
    """

    color: np.ndarray  # (H, W, 3) uint8
    root_from_camera: np.ndarray  # (4, 4)
    lens: WristCamera
    mask: np.ndarray | None = None  # (H, W) bool


@dataclass(frozen=True)
class Edge:
    """One side edge predicted in a view: its samples (pixels, (k, 2)), each one's outward unit normal (k, 2)."""

    uv: np.ndarray
    normal: np.ndarray


@dataclass
class RefineResult:
    """The refined cylinder, how far it moved from the prior, and how well the image agreed."""

    cylinder: Cylinder
    moved_m: np.ndarray  # (x, y) the axis moved (root frame, m)
    radius_m: float
    points: list[int] = field(default_factory=list)  # edge points found in the last round, a view each
    residual_px: float = float("nan")  # the weighted median of the last round's residuals after the fit
    ok: bool = False
    note: str = ""


def _camera_frame(cyl: Cylinder, root_from_camera: np.ndarray) -> np.ndarray:
    return np.linalg.inv(root_from_camera) @ np.asarray(cyl.root_from_obj[0], dtype=float)


def contour_edges(
    cyl: Cylinder,
    root_from_camera: np.ndarray,
    lens: WristCamera,
    n: int = EDGE_SAMPLES,
    margin: float = END_MARGIN,
) -> list[Edge] | None:
    """The cylinder's two side edges in the image (`Edge`: samples and outward normals).

    None if the camera is inside its radius (no side edges) or an edge is behind it.
    """
    cam_obj = _camera_frame(cyl, root_from_camera)
    a, b = cam_obj[:3, 2], cam_obj[:3, 3]
    r, h = float(cyl.radius[0]), float(cyl.height[0])
    w = -b - float(-b @ a) * a  # from the axis to the camera (at the origin), perpendicular to the axis
    d = float(np.linalg.norm(w))
    if d <= r * 1.001:
        return None
    e = w / d
    side = np.cross(a, e)
    t = np.linspace(margin, 1.0 - margin, n)[:, None] * h
    center = b + t * a
    axis_uv = lens.project(center)
    edges = []
    for sign in (1.0, -1.0):
        foot = b + (r * r / d) * e + sign * r * np.sqrt(1.0 - (r / d) ** 2) * side
        pts = foot + t * a
        if np.any(pts[:, 2] <= 0.01):
            return None
        uv = lens.project(pts)
        tangent = np.gradient(uv, axis=0)
        normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        outward = np.sign(np.sum((uv - axis_uv) * normal, axis=1, keepdims=True))
        edges.append(Edge(uv, normal * np.where(outward == 0, 1.0, outward)))
    return edges


def silhouette(cyl: Cylinder, root_from_camera: np.ndarray, lens: WristCamera, n: int = 64) -> np.ndarray:
    """The cylinder's outline in the image (pixels, (M, 2), a closed polygon), through the lens.

    Both rims sampled (`n` points each), the part 1 cm or more in front of the camera kept, their convex hull taken in
    the pinhole image (exactly the silhouette of a convex solid), densified so the lens bends its sides, then the lens
    applied. Empty when nothing is in front of the camera.
    """
    import cv2

    cam_obj = _camera_frame(cyl, root_from_camera)
    a, b, u, v = cam_obj[:3, 2], cam_obj[:3, 3], cam_obj[:3, 0], cam_obj[:3, 1]
    r, h = float(cyl.radius[0]), float(cyl.height[0])
    phi = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    ring = r * (np.cos(phi)[:, None] * u + np.sin(phi)[:, None] * v)
    pts = np.vstack([b + ring, b + h * a + ring])
    pts = pts[pts[:, 2] > 0.01]
    if len(pts) < 3:
        return np.zeros((0, 2))
    plane = pts[:, :2] / pts[:, 2:3]
    hull = plane[cv2.convexHull(plane.astype(np.float32), returnPoints=False).ravel()]
    dense = []
    for p, q in zip(hull, np.roll(hull, -1, axis=0), strict=True):
        k = max(2, int(np.ceil(np.linalg.norm(q - p) / 0.002)))
        dense.append(p + np.linspace(0.0, 1.0, k, endpoint=False)[:, None] * (q - p))
    flat = np.vstack(dense)
    return lens.project(np.column_stack([flat, np.ones(len(flat))]))


def _moved(prior: Cylinder, dx: float, dy: float, radius: float) -> Cylinder:
    frame = np.array(prior.root_from_obj[0], dtype=float)
    frame[0, 3] += dx
    frame[1, 3] += dy
    return Cylinder(np.array([[radius, float(prior.height[0])]]), frame[None])


def _sample(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Bilinear colours (N, C) at pixels (N, 2) (x, y); NaN off the image."""
    import cv2

    xy = pts.astype(np.float32).reshape(-1, 1, 2)
    out = cv2.remap(img.astype(np.float32), xy[..., 0], xy[..., 1], cv2.INTER_LINEAR, borderValue=np.nan)
    return out.reshape(len(pts), -1)


def _object_color(view: EdgeView, edges: list[Edge]) -> np.ndarray | None:
    """The object's colour in `view`: the median along the middle between its predicted side edges.

    None when that middle is off the image.
    """
    middle = 0.5 * (edges[0].uv + edges[1].uv)
    colors = _sample(view.color, middle)
    colors = colors[np.isfinite(colors).all(axis=1)]
    return np.median(colors, axis=0) if len(colors) >= 3 else None


def _likeness(colors: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.exp(-np.sum((colors - target) ** 2, axis=-1) / (2 * COLOR_SIGMA**2))


def _band_px(view: EdgeView, cyl: Cylinder, band_m: float) -> float:
    """`band_m` at the object's depth in `view`, in pixels."""
    center = _camera_frame(cyl, view.root_from_camera) @ np.array([0.0, 0.0, 0.5 * float(cyl.height[0]), 1.0])
    depth = max(float(center[2]), 0.02)
    return max(MIN_BAND_PX, view.lens.fx * band_m / depth)


def _find_edges(view: EdgeView, edge: Edge, band_px: float, color: np.ndarray | None) -> np.ndarray:
    """Each sample's residual (px, outward positive; NaN where none).

    Where its scan line changes from the object's colour (inside) to something else (outside): the strongest change.
    """
    steps = np.arange(-np.ceil(band_px), np.ceil(band_px) + 1.0)
    found = np.full(len(edge.uv), np.nan)
    h, w = view.color.shape[:2]
    for i, (p, n) in enumerate(zip(edge.uv, edge.normal, strict=True)):
        line = p[None, :] + steps[:, None] * n[None, :]
        if not np.isfinite(line).all():
            continue
        inside = (line[:, 0] >= 1) & (line[:, 0] < w - 2) & (line[:, 1] >= 1) & (line[:, 1] < h - 2)
        if inside.sum() < 0.8 * len(steps):
            continue
        if view.mask is not None:
            px = np.clip(np.round(line).astype(int), 0, [w - 1, h - 1])
            if view.mask[px[:, 1], px[:, 0]].any():
                continue
        colors = _sample(view.color, line)
        grad = np.linalg.norm(np.diff(colors, axis=0), axis=1)
        grad[~(inside[:-1] & inside[1:]) | ~np.isfinite(grad)] = 0.0
        score = grad.copy()
        if color is not None:  # (the object's edge: its colour inside, another outside)
            k = np.arange(len(grad))
            inner = np.array([np.nanmean(colors[max(0, j - SIDE_PX + 1) : j + 1], axis=0) for j in k])
            outer = np.array([np.nanmean(colors[j + 1 : j + 1 + SIDE_PX], axis=0) for j in k])
            score = grad * _likeness(inner, color) * (1.0 - _likeness(outer, color))
        best = int(np.argmax(score))
        if grad[best] >= MIN_GRADIENT and score[best] >= 0.25 * grad[best]:
            found[i] = 0.5 * (steps[best] + steps[best + 1])
    return found


def refine(
    prior: Cylinder,
    views: Sequence[EdgeView],
    fit_radius: bool = False,
    bands_m: Sequence[float] = BANDS_M,
    iterations: int = 4,
) -> RefineResult:
    """`prior` (one upright cylinder) moved across the table (and its radius refit, `fit_radius`) to fit `views`' edges.

    The module docstring: each round predicts the side edges, takes the object's colour between them, searches each
    scan line within the round's band (`bands_m`, at the object), then fits the parameters (Gauss-Newton, Huber
    weights) over every view's points, a step no longer than the band; the radius (`fit_radius`) only from the second
    round, within `RADIUS_BOUNDS` of the prior's.
    """
    if len(prior) != 1:
        raise ValueError(f"one cylinder at a time: got {len(prior)}")
    r0 = float(prior.radius[0])
    params = np.array([0.0, 0.0, r0])
    eps = np.array([2e-4, 2e-4, 1e-4])
    points: list[int] = []
    sides: list[tuple[int, int]] = []
    residual = float("nan")
    for round_, band_m in enumerate(bands_m):
        free = [0, 1, 2] if fit_radius and round_ > 0 else [0, 1]
        current = _moved(prior, *params)
        rows_r, rows_j = [], []
        points, sides = [], []
        for view in views:
            edges = contour_edges(current, view.root_from_camera, view.lens)
            if edges is None:
                points.append(0)
                continue
            color = _object_color(view, edges)
            band_px = _band_px(view, current, band_m)
            nudged = []
            for j in free:
                step = params.copy()
                step[j] += eps[j]
                nudged.append(contour_edges(_moved(prior, *step), view.root_from_camera, view.lens))
            count = 0
            per_side = [0, 0]
            for ei, edge in enumerate(edges):
                res = _find_edges(view, edge, band_px, color)
                good = np.isfinite(res)
                count += int(good.sum())
                per_side[ei] = int(good.sum())
                if not good.any():
                    continue
                jac = np.zeros((len(res), len(free)))
                for k, other in enumerate(nudged):
                    if other is not None:
                        jac[:, k] = np.sum((other[ei].uv - edge.uv) * edge.normal, axis=1) / eps[free[k]]
                rows_r.append(res[good])
                rows_j.append(jac[good])
            points.append(count)
            sides.append((per_side[0], per_side[1]))
        if not rows_r:
            return RefineResult(prior, np.zeros(2), r0, points, note="no edge found in any view")
        r = np.concatenate(rows_r)
        jac_all = np.concatenate(rows_j)
        delta = np.zeros(len(free))
        for _ in range(iterations):  # (iteratively reweighted least squares, the Huber loss)
            err = r - jac_all @ delta
            weight = np.minimum(1.0, HUBER_PX / np.maximum(np.abs(err), 1e-9))
            normal_eq = jac_all.T @ (weight[:, None] * jac_all) + 1e-6 * np.eye(len(free))
            delta = np.linalg.solve(normal_eq, jac_all.T @ (weight * r))
        move = np.linalg.norm(delta[:2])
        if move > band_m:  # (no further than the band searched)
            delta[:2] *= band_m / move
        params[free] += delta
        params[2] = float(np.clip(params[2], RADIUS_BOUNDS[0] * r0, RADIUS_BOUNDS[1] * r0))
        residual = float(np.median(np.abs(r - jac_all @ delta)))
    cylinder = _moved(prior, *params)
    # every view that gave points gave both sides (one side alone: the radius and the position trade off)
    used = [s for s in sides if sum(s) > 0]
    sides_ok = bool(used) and all(min(s) >= MIN_POINTS_A_SIDE for s in used)
    radius_ok = not fit_radius or OK_RADIUS[0] * r0 <= params[2] <= OK_RADIUS[1] * r0
    ok = sides_ok and radius_ok and np.isfinite(residual) and residual <= OK_RESIDUAL_PX
    if ok:
        note = ""
    elif not sides_ok:
        note = "a view saw one side only"
    elif not radius_ok:
        note = f"the radius moved too far ({1000 * params[2]:.1f} mm, prior {1000 * r0:.1f})"
    else:
        note = f"edges disagree ({residual:.1f} px)"
    return RefineResult(cylinder, params[:2].copy(), float(params[2]), points, residual, bool(ok), note)
