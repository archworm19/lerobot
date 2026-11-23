"""A cylinder's silhouette in a wrist camera, and its position refined against the image's edges.

**EXPERIMENTAL, not used by the harness or the simulator; a candidate for removal.** On the recorded wrist frames
(2026-11-23) it could not be shown to improve the head's fit: the fits that land on edges disagree with the head by
1-3 cm across and 1.5-2.5 cm in height, and from one view height, radius and distance trade off, so nothing here tells a
wrong head fit from a wrong wrist-camera pose (`docs/object_grasping.md`, "Wrist silhouettes against the head's fit").
Nothing imports this module but its test (`tests/felix_jr/test_silhouette.py`): removing it is deleting the two files.

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
# colours compared as chromaticity (r, g over r + g + b) and log brightness: the head's and the wrist's cameras
# balance and expose differently; a white bottle still differs from light wood (its hue) and from steel (darker)
CHROMA_SIGMA = 0.04
LOG_BRIGHTNESS_SIGMA = 0.45
OK_INSIDE = (
    0.6  # for `ok`: at least this share of the refined outline's inside looks like the object, a view each
)
OK_OUTSIDE = 0.35  # and at most this share of a ring just outside it does
RING_PX = (3, 10)  # the ring outside the outline: from 3 to 10 px out
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
    coverage: float = float("nan")  # (chamfer matching) the share of the refined outline on the images' edges


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


def _chroma(colors: np.ndarray) -> np.ndarray:
    """Colours (..., 3) as (r, g chromaticity, log brightness)."""
    total = np.sum(colors, axis=-1, keepdims=True) + 3.0
    return np.concatenate([colors[..., :2] / total, np.log(total / 3.0)], axis=-1)


def _likeness(colors: np.ndarray, target: np.ndarray) -> np.ndarray:
    """How much each colour (..., 3, RGB) looks like `target` (RGB), in [0, 1]: chromaticity and brightness."""
    a, b = _chroma(colors), _chroma(np.asarray(target, dtype=float))
    d2 = np.sum((a[..., :2] - b[:2]) ** 2, axis=-1) / CHROMA_SIGMA**2
    d2 = d2 + (a[..., 2] - b[2]) ** 2 / LOG_BRIGHTNESS_SIGMA**2
    return np.exp(-0.5 * d2)


def _outline_mask(cyl: Cylinder, view: EdgeView) -> np.ndarray:
    import cv2

    outline = silhouette(cyl, view.root_from_camera, view.lens)
    mask = np.zeros(view.color.shape[:2], np.uint8)
    if len(outline) >= 3 and np.isfinite(outline).all():
        cv2.fillPoly(mask, [np.round(outline).astype(np.int32)], 1)
    return mask


def object_color(
    image: np.ndarray,
    root_from_camera: np.ndarray,
    lens: WristCamera,
    cyl: Cylinder,
    depth_m: np.ndarray | None = None,
    erode_px: int = 3,
    depth_tol_m: float = 0.03,
) -> np.ndarray | None:
    """The object's colour (RGB) where `cyl` projects in `image`: the median inside its outline, eroded.

    With `depth_m` (a depth camera's: the head's), only the pixels whose depth agrees with the cylinder's within
    `depth_tol_m` (what is in front of it or behind its outline left out). None when too few pixels are left.
    """
    import cv2

    view = EdgeView(np.asarray(image), root_from_camera, lens)
    mask = cv2.erode(
        _outline_mask(cyl, view), np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8)
    ).astype(bool)
    if depth_m is not None:
        cam = _camera_frame(cyl, root_from_camera)
        center_depth = float((cam @ np.array([0.0, 0.0, 0.5 * float(cyl.height[0]), 1.0]))[2])
        reach = float(cyl.radius[0]) + depth_tol_m
        mask &= np.isfinite(depth_m) & (np.abs(depth_m - center_depth) <= reach)
    pixels = np.asarray(image)[mask].astype(float)
    return np.median(pixels, axis=0) if len(pixels) >= 20 else None


def contrast_color(
    view: EdgeView,
    prior: Cylinder,
    near_m: float = BANDS_M[0],
    far_px: tuple[int, int] = (25, 60),
    k: int = 4,
) -> np.ndarray | None:
    """The object's colour (RGB) in `view` itself, needing the prior only near the object, not on it.

    The colours near the prior (its outline widened by `near_m` at the object) clustered (`k`-means, chromaticity and
    brightness); the background sampled in a ring further out (`far_px` beyond that); the object's colour is the
    cluster most common near the prior against the ring (the table, a ruler beside it: in both). None when no
    cluster stands out.
    """
    import cv2

    inside = _outline_mask(prior, view)
    if not inside.any():
        return None
    grow = int(round(_band_px(view, prior, near_m)))
    near = cv2.dilate(inside, np.ones((2 * grow + 1,) * 2, np.uint8)).astype(bool)
    ring_in = cv2.dilate(inside, np.ones((2 * (grow + far_px[0]) + 1,) * 2, np.uint8)).astype(bool)
    ring_out = cv2.dilate(inside, np.ones((2 * (grow + far_px[1]) + 1,) * 2, np.uint8)).astype(bool)
    ring = ring_out & ~ring_in
    if view.mask is not None:
        near &= ~view.mask
        ring &= ~view.mask
    img = view.color.astype(np.float32)
    near_px, ring_px = img[near], img[ring]
    if len(near_px) < 50 or len(ring_px) < 50:
        return None
    feats = _chroma(near_px) * np.array([1 / CHROMA_SIGMA, 1 / CHROMA_SIGMA, 1 / LOG_BRIGHTNESS_SIGMA])
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    _, labels, _ = cv2.kmeans(feats.astype(np.float32), k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    best, best_score = None, 0.0
    for c in range(k):
        members = near_px[labels == c]
        if len(members) < 0.05 * len(near_px):
            continue
        color = np.median(members, axis=0)
        near_share = float((_likeness(near_px, color) > 0.5).mean())
        ring_share = float((_likeness(ring_px, color) > 0.5).mean())
        score = near_share - 2.0 * ring_share
        if score > best_score:
            best, best_score = color, score
    return best


def region_agreement(cyl: Cylinder, view: EdgeView, color: np.ndarray) -> tuple[float, float]:
    """(The share of the outline's inside that looks like the object, the share of a ring just outside that does)."""
    import cv2

    inside = _outline_mask(cyl, view)
    outer = cv2.dilate(inside, np.ones((2 * RING_PX[1] + 1,) * 2, np.uint8))
    near = cv2.dilate(inside, np.ones((2 * RING_PX[0] + 1,) * 2, np.uint8))
    core = cv2.erode(inside, np.ones((5, 5), np.uint8)).astype(bool)
    ring = (outer > 0) & (near == 0)
    if view.mask is not None:
        core &= ~view.mask
        ring &= ~view.mask
    img = view.color.astype(float)
    like = _likeness(img, color) > 0.5
    return (float(like[core].mean()) if core.any() else 0.0, float(like[ring].mean()) if ring.any() else 1.0)


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
    color: np.ndarray | None = None,
) -> RefineResult:
    """`prior` (one upright cylinder) moved across the table (and its radius refit, `fit_radius`) to fit `views`' edges.

    The module docstring: each round predicts the side edges, takes the object's colour between them, searches each
    scan line within the round's band (`bands_m`, at the object), then fits the parameters (Gauss-Newton, Huber
    weights) over every view's points, a step no longer than the band; the radius (`fit_radius`) only from the second
    round, within `RADIUS_BOUNDS` of the prior's. `color`: the object's (RGB; e.g. `object_color` on the head's
    capture): else each view's, between the predicted edges (wrong when the prior is off by more than its radius).
    `ok` also asks every contributing view's refined outline to look like the object inside and not just outside
    (`region_agreement`): edges that agree with each other can be another object's (a ruler's two sides).
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
                sides.append((0, 0))
                continue
            view_color = color if color is not None else _object_color(view, edges)
            band_px = _band_px(view, current, band_m)
            nudged = []
            for j in free:
                step = params.copy()
                step[j] += eps[j]
                nudged.append(contour_edges(_moved(prior, *step), view.root_from_camera, view.lens))
            count = 0
            per_side = [0, 0]
            for ei, edge in enumerate(edges):
                res = _find_edges(view, edge, band_px, view_color)
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
    region_ok = True
    if color is not None:
        for view, s in zip(views, sides, strict=False):
            if sum(s) > 0:
                inside, outside = region_agreement(cylinder, view, color)
                region_ok &= inside >= OK_INSIDE and outside <= OK_OUTSIDE
    ok = sides_ok and radius_ok and region_ok and np.isfinite(residual) and residual <= OK_RESIDUAL_PX
    if ok:
        note = ""
    elif not region_ok:
        note = "the outline does not hold the object's colour"
    elif not sides_ok:
        note = "a view saw one side only"
    elif not radius_ok:
        note = f"the radius moved too far ({1000 * params[2]:.1f} mm, prior {1000 * r0:.1f})"
    else:
        note = f"edges disagree ({residual:.1f} px)"
    return RefineResult(cylinder, params[:2].copy(), float(params[2]), points, residual, bool(ok), note)


# ---- chamfer matching: the whole outline against the image's edges ---------------------------------------------------

CHAMFER_TRUNC_PX = 12.0  # an outline point's distance to the nearest edge counted up to this
CHAMFER_STEP_PX = 3.0  # the outline sampled every this many pixels
ON_EDGE_PX = 2.0  # an outline point "on an edge" within this
OK_COVERAGE = (
    0.55  # for `ok`: at least this share of the refined outline on edges, in every view that sees it
)
OK_MARGIN = 0.08  # and the best place's coverage this much above the best place more than 6 mm away


def edge_map(view: EdgeView) -> np.ndarray:
    """Each pixel's distance (px) to the nearest edge.

    Canny on the blurred grey image, its thresholds from the median; the edges in `view.mask` left out.
    """
    import cv2

    gray = cv2.GaussianBlur(cv2.cvtColor(view.color, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    med = float(np.median(gray))
    edges = cv2.Canny(gray, int(max(0, 0.66 * med)), int(min(255, 1.33 * med))) > 0
    if view.mask is not None:
        edges &= ~view.mask
    return cv2.distanceTransform((~edges).astype(np.uint8), cv2.DIST_L2, 5)


def _outline_points(cyl: Cylinder, view: EdgeView) -> np.ndarray:
    """The outline sampled every `CHAMFER_STEP_PX`, the points on the image and off the mask."""
    outline = silhouette(cyl, view.root_from_camera, view.lens)
    if len(outline) < 3 or not np.isfinite(outline).all():
        return np.zeros((0, 2))
    closed = np.vstack([outline, outline[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.arange(0.0, s[-1], CHAMFER_STEP_PX)
    pts = np.column_stack([np.interp(t, s, closed[:, 0]), np.interp(t, s, closed[:, 1])])
    h, w = view.color.shape[:2]
    keep = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
    pts = pts[keep]
    if view.mask is not None and len(pts):
        px = np.clip(np.round(pts).astype(int), 0, [w - 1, h - 1])
        pts = pts[~view.mask[px[:, 1], px[:, 0]]]
    return pts


def _chamfer(
    cyl: Cylinder, views: Sequence[EdgeView], maps: Sequence[np.ndarray]
) -> tuple[float, float, int]:
    """(the mean truncated squared distance of the outline to the edges, the share of it on edges, points), all views."""
    costs, on, n = [], [], 0
    for view, dist in zip(views, maps, strict=True):
        pts = _outline_points(cyl, view)
        if len(pts) < 10:
            continue
        px = np.clip(np.round(pts).astype(int), 0, [dist.shape[1] - 1, dist.shape[0] - 1])
        d = np.minimum(dist[px[:, 1], px[:, 0]], CHAMFER_TRUNC_PX)
        costs.append(d**2)
        on.append(d <= ON_EDGE_PX)
        n += len(pts)
    if not costs:
        return float("inf"), 0.0, 0
    return float(np.mean(np.concatenate(costs))), float(np.mean(np.concatenate(on))), n


def chamfer_refine(
    prior: Cylinder,
    views: Sequence[EdgeView],
    search_m: float = 0.02,
    coarse_m: float = 0.002,
    fine_m: float = 0.0005,
    fit_radius: bool = False,
    color: np.ndarray | None = None,
    radius_range: tuple[float, float] = (0.8, 1.2),
) -> RefineResult:
    """`prior` moved across the table (and its radius refit, `fit_radius`) so its whole outline lies on the images' edges.

    The outline: the top rim with the sides. A grid search within `search_m` (every `coarse_m`), then around the best
    (every `fine_m`). `ok`: the outline mostly on edges (`OK_COVERAGE`), the best place clearly better than any other more
    than 6 mm away (`OK_MARGIN`), and with `color` the outline holding the object's colour (`region_agreement`).
    """
    maps = [edge_map(v) for v in views]
    r0 = float(prior.radius[0])

    def score(dx: float, dy: float, r: float) -> tuple[float, float, int]:
        return _chamfer(_moved(prior, dx, dy, r), views, maps)

    grid = np.arange(-search_m, search_m + 1e-9, coarse_m)
    table = {(dx, dy): score(dx, dy, r0) for dx in grid for dy in grid}
    finite = {k: v for k, v in table.items() if np.isfinite(v[0])}
    if not finite:
        return RefineResult(prior, np.zeros(2), r0, note="the outline is not in any view")
    (bx, by), _ = min(finite.items(), key=lambda kv: kv[1][0])
    radius = r0
    fine = np.arange(-2 * coarse_m, 2 * coarse_m + 1e-9, fine_m)
    radii = r0 * np.linspace(radius_range[0], radius_range[1], 13) if fit_radius else [r0]
    best = (bx, by, radius, *score(bx, by, radius))
    for dx in bx + fine:
        for dy in by + fine:
            for r in radii:
                c = score(dx, dy, r)
                if c[0] < best[3]:
                    best = (dx, dy, r, *c)
    dx, dy, radius, cost, coverage, n = best
    cylinder = _moved(prior, dx, dy, radius)
    far = [v[1] for (gx, gy), v in finite.items() if np.hypot(gx - dx, gy - dy) > 0.006]
    margin = coverage - (max(far) if far else 0.0)
    region_ok = True
    if color is not None:
        for view in views:
            if len(_outline_points(cylinder, view)) >= 10:
                inside, outside = region_agreement(cylinder, view, color)
                region_ok &= inside >= OK_INSIDE and outside <= OK_OUTSIDE
    ok = coverage >= OK_COVERAGE and margin >= OK_MARGIN and region_ok
    if ok:
        note = ""
    elif coverage < OK_COVERAGE:
        note = f"the outline is not on edges ({100 * coverage:.0f} %)"
    elif margin < OK_MARGIN:
        note = f"another place fits as well ({100 * margin:.0f} points)"
    else:
        note = "the outline does not hold the object's colour"
    return RefineResult(
        cylinder, np.array([dx, dy]), float(radius), [n], float(np.sqrt(cost)), bool(ok), note, coverage
    )
