"""Long-term tools: pure functions over the harness's objects (`docs/harness.md`, section 1.H, layer 2).

They know nothing about the workspace; `Workspace.call` records them. Lengths are meters and points are in the root
frame; pixels are view pixels unless the argument says capture.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np

from lerobot.utils.import_utils import _scipy_available, require_package

from .objects import Capture, Focus, Grid, Object, Points, TableEstimate, View
from .primitives import Cylinder

if TYPE_CHECKING or _scipy_available:
    from scipy.optimize import least_squares

MAX_GRID_LINES = 150  # per axis: more is unreadable (and slow); zoom in and use a smaller extent instead
# fit_cylinder: capture pixels around the marked region (a line of marks along a thin object is a region too)
FIT_PAD_PX = 4
FIT_MIN_POINTS = 30  # fewer points with depth in the marked region: no fit
FIT_KEEP_SIGMAS = 3.0  # a fit's inliers: within this many robust sigmas of the surface
FIT_KEEP_MIN_M = 0.003  # but never tighter than this (the depth's own noise)
FIT_ROUNDS = 4  # fit, drop the outliers, fit again
FIT_SEARCH_TRIES = (
    400  # circles tried (through three points) to find an upright side among the table around it
)
FIT_SEARCH_BAND_M = 0.0025  # a point is on a tried circle within this
FIT_RADIUS_RANGE_M = (0.003, 0.08)  # the circles tried: a pen's to a pot's
FIT_CELL_M = 0.003  # plan-view cells: a vertical side stacks its points in a cell, over a range of heights
# a cell whose points span at least this in height is on a vertical side (not the table, not a lid)
FIT_SIDE_SPAN_M = 0.01
FIT_SEED_BAND_M = 0.02  # free axis: only region points this near the line through the marks themselves


def full_view(capture: Capture, scale: float = 1.0) -> View:
    """The whole capture."""
    return View(capture, 0, 0, capture.scene.width, capture.scene.height, scale)


def focus(*views: View, reason: str = "") -> Focus:
    """Choose the views to work in (the first is primary). A selection among existing captures: nothing moves."""
    if not views:
        raise ValueError("focus needs at least one view")
    return Focus(tuple(views), reason)


def zoom(view: View, rect: Sequence[float], scale: float | None = None) -> View:
    """A crop of `view`. `rect` = (u, v, width, height) in `view`'s pixels; it is clipped to the capture.

    `scale` (view pixels per capture pixel) defaults to fitting the crop into `view`'s rendered size.
    """
    u, v, w, h = (float(x) for x in rect)
    (u0, v0), (u1, v1) = view.to_capture(np.array([[u, v], [u + w, v + h]]))
    s = view.capture.scene
    u0, v0 = int(np.clip(np.floor(u0), 0, s.width - 1)), int(np.clip(np.floor(v0), 0, s.height - 1))
    u1, v1 = int(np.clip(np.ceil(u1), u0 + 1, s.width)), int(np.clip(np.ceil(v1), v0 + 1, s.height))
    if scale is None:
        out_w, out_h = view.size
        scale = min(out_w / (u1 - u0), out_h / (v1 - v0))
    return View(view.capture, u0, v0, u1 - u0, v1 - v0, float(scale))


def table_height(
    capture: Capture,
    max_depth_m: float = 1.2,
    bin_m: float = 0.01,
    band_m: float = 0.01,
    min_h: float | None = None,
    second_fraction: float = 0.3,
) -> TableEstimate:
    """Estimate the table's height (root-frame z) from the capture's depth. The method is in `docs/harness.md` 1.B.

    Depth pixels within `max_depth_m` of the camera are lifted to the root frame; the heights are histogrammed in
    `bin_m` bins and the fullest one (above `min_h`, if given, e.g. to skip the floor) is the table. `h` is the
    median height of the points within `band_m` of it. A second peak with at least `second_fraction` of the first's
    points is reported as `other_h`. TODO: points on the robot's own links are not excluded yet; they spread over
    many bins, so they rarely move the peak.
    """
    s = capture.scene
    vv, uu = np.mgrid[0 : s.height, 0 : s.width]
    z_cam = s.depth_m
    cam = np.stack([(uu - s.ppx) / s.fx * z_cam, (vv - s.ppy) / s.fy * z_cam, z_cam], axis=-1)
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(z_cam) & (z_cam > 0) & (z_cam <= max_depth_m)
    z = np.full(z_cam.shape, np.nan)
    z[valid] = capture.to_root(cam[valid])[:, 2]
    if min_h is not None:
        with np.errstate(invalid="ignore"):
            valid &= z >= min_h
    heights = z[valid]
    if heights.size < 50:
        raise ValueError(f"only {heights.size} depth points to estimate the table from")

    edges = np.arange(heights.min(), heights.max() + 2 * bin_m, bin_m)
    counts, edges = np.histogram(heights, edges)
    # summed over 3 bins: a surface straddling two bins still makes one peak
    smooth = np.convolve(counts, np.ones(3), mode="same")
    first = int(np.argmax(smooth))
    peak = (edges[first] + edges[first + 1]) / 2

    with np.errstate(invalid="ignore"):
        near = valid & (np.abs(z - peak) <= band_m)
        h = float(np.median(z[near]))
        mask = valid & (np.abs(z - h) <= band_m)
    inliers = capture.to_root(cam[mask])
    a, *_ = np.linalg.lstsq(np.c_[inliers[:, :2], np.ones(len(inliers))], inliers[:, 2], rcond=None)
    tilt = float(np.degrees(np.arctan(np.hypot(a[0], a[1]))))

    centers = (edges[:-1] + edges[1:]) / 2
    away = np.abs(centers - peak) > 3 * bin_m
    other_h = None
    if away.any():
        second = int(np.flatnonzero(away)[np.argmax(smooth[away])])
        if smooth[second] >= second_fraction * smooth[first]:
            other_h = float(centers[second])
    return TableEstimate(capture, h, int(mask.sum()), float(inliers[:, 2].std()), tilt, mask, other_h)


def make_grid(
    view: View,
    h: float | TableEstimate,
    cell: float = 0.02,
    extent: tuple[tuple[float, float], tuple[float, float]] | None = None,
    max_range_m: float = 1.0,
) -> Grid:
    """A grid on the plane z = `h` with `cell`-meter squares.

    Pass a `TableEstimate` as `h` rather than its `.h`, so the workspace records the grid as derived from it (and
    stale with it).

    `extent` ((x_min, x_max), (y_min, y_max)) defaults to the part of the plane `view` shows, out to `max_range_m`
    horizontally from the camera. So a grid made from a zoomed view covers only the zoomed area.
    """
    if isinstance(h, TableEstimate):
        h = h.h
    if extent is None:
        w, ht = view.size
        su, sv = np.meshgrid(np.linspace(0, w - 1, 33), np.linspace(0, ht - 1, 25))
        origin, d = view.capture.rays(view.to_capture(np.stack([su, sv], axis=-1).reshape(-1, 2)))
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (h - origin[2]) / d[:, 2]
        hits = origin + t[:, None] * d
        keep = (t > 0) & (np.hypot(hits[:, 0] - origin[0], hits[:, 1] - origin[1]) <= max_range_m)
        if not keep.any():
            raise ValueError(f"the view does not show the plane z = {h:.3f} within {max_range_m} m")
        hits = hits[keep]
        extent = ((hits[:, 0].min(), hits[:, 0].max()), (hits[:, 1].min(), hits[:, 1].max()))
    (x0, x1), (y0, y1) = extent
    lines = max(x1 - x0, y1 - y0) / cell
    if lines > MAX_GRID_LINES:
        raise ValueError(f"{lines:.0f} lines per axis at {cell} m cells: use larger cells or zoom in first")
    return Grid(float(h), float(cell), (float(x0), float(x1)), (float(y0), float(y1)))


def locate_pixels(
    view: View,
    uv: np.ndarray,
    labels: Sequence[str] | None = None,
    window: int = 5,
    standoff: float = 0.0,
) -> Points:
    """Root-frame points shown at view pixels `uv` (N, 2), from depth.

    Uses `Scene.pixel_to_xyz`: NaN where depth is missing or straddles an edge. `standoff` moves each point toward
    the camera, off the surface.
    """
    uv_cap = view.to_capture(np.atleast_2d(uv))
    xyz_cam = view.capture.scene.pixel_to_xyz(uv_cap, window=window, standoff=standoff)
    names = tuple(labels) if labels is not None else tuple(f"P{i + 1}" for i in range(len(uv_cap)))
    return Points(view.capture.to_root(xyz_cam), names, uv_cap, view.capture)


def locate_grid(grid: Grid, xy: np.ndarray, labels: Sequence[str] | None = None) -> Points:
    """Root-frame points on the grid's plane at (x, y) (N, 2), meters. No depth involved."""
    xy = np.atleast_2d(np.asarray(xy, dtype=np.float64))
    names = tuple(labels) if labels is not None else tuple(f"P{i + 1}" for i in range(len(xy)))
    return Points(grid.point(xy), names)


def _region_points(view: View, uv: np.ndarray, pad_px: int) -> np.ndarray:
    """Root-frame points (N, 3) at every pixel with depth in the region the view pixels `uv` mark.

    The region is their convex hull (a line or a single mark: the marks themselves), widened by `pad_px` capture pixels.
    """
    capture = view.capture
    s = capture.scene
    uv_cap = np.round(view.to_capture(np.atleast_2d(uv))).astype(np.int32)
    mask = np.zeros((s.height, s.width), np.uint8)
    if len(uv_cap) >= 3:
        cv2.fillConvexPoly(mask, cv2.convexHull(uv_cap.reshape(-1, 1, 2)), 1)
    cv2.polylines(mask, [uv_cap.reshape(-1, 1, 2)], False, 1, 1)
    for u, v in uv_cap:
        mask[np.clip(v, 0, s.height - 1), np.clip(u, 0, s.width - 1)] = 1
    if pad_px > 0:
        mask = cv2.dilate(mask, np.ones((2 * pad_px + 1, 2 * pad_px + 1), np.uint8))
    v, u = np.nonzero(mask)
    xyz_cam = s.pixel_to_xyz(np.column_stack([u, v]).astype(np.float64), window=1, min_valid=1)
    xyz_cam = xyz_cam[np.isfinite(xyz_cam).all(axis=1)]
    return capture.to_root(xyz_cam) if len(xyz_cam) else np.zeros((0, 3))


def _marks_line(view: View, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """The line (a point, a unit direction) through the marks' own points (their depth, root frame).

    None with fewer than two of them, or all at one place.
    """
    capture = view.capture
    xyz_cam = capture.scene.pixel_to_xyz(view.to_capture(np.atleast_2d(uv)))
    marks = capture.to_root(xyz_cam[np.isfinite(xyz_cam).all(axis=1)])
    if len(marks) < 2:
        return None
    center = marks.mean(axis=0)
    _, spread, axes = np.linalg.svd(marks - center, full_matrices=False)
    if spread[0] < 1e-3:
        return None
    return center, axes[0]


def _keep(residuals: np.ndarray, inliers: np.ndarray) -> np.ndarray:
    """The points within `FIT_KEEP_SIGMAS` robust sigmas (MAD) of the surface; never tighter than `FIT_KEEP_MIN_M`.

    The sigma is taken over the current `inliers`, not everything marked: the table would widen it.
    """
    current = residuals[inliers]
    sigma = 1.4826 * float(np.median(np.abs(current - np.median(current))))
    return np.abs(residuals) <= max(FIT_KEEP_SIGMAS * sigma, FIT_KEEP_MIN_M)


def _vertical_side(points: np.ndarray) -> np.ndarray:
    """Which points are on a vertical surface: their plan-view cell spans at least `FIT_SIDE_SPAN_M` in height.

    Cells are `FIT_CELL_M` square. A table or a lid puts one height in a cell; a side stacks many.
    """
    cells = np.floor(points[:, :2] / FIT_CELL_M).astype(np.int64)
    _, cell, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    cell = cell.ravel()
    low = np.full(len(counts), np.inf)
    high = np.full(len(counts), -np.inf)
    np.minimum.at(low, cell, points[:, 2])
    np.maximum.at(high, cell, points[:, 2])
    return (high - low)[cell] >= FIT_SIDE_SPAN_M


def _search_circle(xy: np.ndarray, seed: int = 0) -> np.ndarray:
    """The circle (cx, cy, r) with the most points on it, among circles through random triples of points.

    Seen from above, an upright object's side collapses onto one arc, dense, while the table around it spreads over
    an area. `FIT_SEARCH_TRIES` circles, radius in `FIT_RADIUS_RANGE_M`, points counted within `FIT_SEARCH_BAND_M`.
    """
    rng = np.random.default_rng(seed)
    best, best_count = np.array([*xy.mean(axis=0), FIT_RADIUS_RANGE_M[0]]), -1
    for _ in range(FIT_SEARCH_TRIES):
        p = xy[rng.choice(len(xy), 3, replace=False)]
        a = 2 * (p[1:] - p[0])
        if abs(np.linalg.det(a)) < 1e-12:
            continue
        center = np.linalg.solve(a, (p[1:] ** 2).sum(axis=1) - (p[0] ** 2).sum())
        radius = float(np.linalg.norm(p[0] - center))
        if not FIT_RADIUS_RANGE_M[0] <= radius <= FIT_RADIUS_RANGE_M[1]:
            continue
        count = int((np.abs(np.linalg.norm(xy - center, axis=1) - radius) < FIT_SEARCH_BAND_M).sum())
        if count > best_count:
            best, best_count = np.array([*center, radius]), count
    return best


def _fit_upright(
    points: np.ndarray, start: np.ndarray | None = None, held: float | None = None
) -> tuple[np.ndarray, float, np.ndarray]:
    """A vertical axis's (x, y), the radius, and each point's residual (distance to the side less the radius).

    With `held`, the radius is that, and only the axis is fitted.
    """
    xy = points[:, :2]
    if start is None:
        a = np.column_stack([2 * xy, np.ones(len(xy))])
        cx, cy, k = np.linalg.lstsq(a, (xy**2).sum(axis=1), rcond=None)[0]  # algebraic circle: a start
        start = np.array([cx, cy, np.sqrt(max(k + cx**2 + cy**2, 1e-6))])

    def residuals(x: np.ndarray) -> np.ndarray:
        return np.hypot(xy[:, 0] - x[0], xy[:, 1] - x[1]) - (x[2] if held is None else held)

    x = least_squares(residuals, start if held is None else start[:2], loss="soft_l1", f_scale=0.002).x
    return x[:2], float(abs(x[2])) if held is None else held, residuals(x)


def _fit_free(
    points: np.ndarray, seed: tuple[np.ndarray, np.ndarray] | None = None, held: float | None = None
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """A point on the axis, its unit direction, the radius, and each point's residual.

    It starts from `seed` (a point and a direction) if given, else the points' principal axis. With `held`, the
    radius is that, and only the axis is fitted.
    """
    center = points.mean(axis=0)
    if seed is None:
        direction = np.linalg.svd(points - center, full_matrices=False)[2][0]
    else:
        direction = seed[1] / np.linalg.norm(seed[1])
        center = seed[0] + ((center - seed[0]) @ direction) * direction

    def distances(p0: np.ndarray, d: np.ndarray) -> np.ndarray:
        d = d / np.linalg.norm(d)
        rel = points - p0
        return np.linalg.norm(rel - np.outer(rel @ d, d), axis=1)

    start = np.r_[center, direction]
    if held is None:
        start = np.r_[start, np.median(distances(center, direction))]

    def residuals(x: np.ndarray) -> np.ndarray:
        p0, d, r = x[:3], x[3:6], (x[6] if held is None else held)
        gauge = [np.linalg.norm(d) - 1.0, (p0 - center) @ d / max(np.linalg.norm(d), 1e-9)]
        return np.r_[distances(p0, d) - r, 10.0 * np.asarray(gauge)]

    x = least_squares(residuals, start, loss="soft_l1", f_scale=0.002).x
    d = x[3:6] / np.linalg.norm(x[3:6])
    r = float(abs(x[6])) if held is None else held
    return x[:3], d, r, distances(x[:3], d) - r


def fit_cylinder(
    view: View,
    uv: np.ndarray,
    axis: Literal["up", "free"] = "up",
    label: str = "object",
    part: str = "body",
    pad_px: int = FIT_PAD_PX,
    radius: float | None = None,
) -> Object:
    """A cylinder fitted to the depth where the view pixels `uv` mark an object (`docs/object_grasping.md`).

    Mark the part that may be grasped, and only it (the pen above the rim): the region is the marks' convex hull, so
    a few marks around it, or a line of marks along a thin one. Every pixel there with depth is lifted to the root
    frame; the fit drops what is not on the surface (the table, what is behind) over `FIT_ROUNDS` rounds.

    - `axis="up"` holds the axis vertical (an object standing on the table). The side is found first: the points
      stacked over a range of heights (a table or a lid puts one height in a place), then the circle in plan with the
      most of them on it. Its top is anything over its footprint (a lid, a cap). A view looking down on the object
      barely shows its side and is refused.
    - `axis="free"` fits the direction (a pen, a handle), starting from the line through the marks' own depths and
      keeping only what is near it (the room behind a pen standing up is not the pen).

    The extent along the axis is what depth saw (2nd to 98th percentile), so a base hidden behind something is not in
    it. The camera sees one side of the object, so a fit from a narrow arc can be biased (container1: 1.4 cm from a
    partial rim): mark what shows the most of it. `radius` (m), when the model knows it (a pen: about 0.45 cm), is
    held and only the axis is fitted: a thin object's curvature is a few depth pixels across, and its fitted radius
    comes out short (pen2's pen in the book: 0.2-0.3 cm). The note gives the points, the residual and the arc of the
    side seen.
    """
    require_package("scipy", extra="felix_jr")
    held = radius
    if axis not in ("up", "free"):
        raise ValueError(f"axis is 'up' or 'free', not {axis!r}")
    points = _region_points(view, uv, pad_px)
    seed = None
    if axis == "free":  # start from the marks themselves: they are on the object, what is behind it is not
        seed = _marks_line(view, uv)
        if seed is not None:
            rel = points - seed[0]
            off = np.linalg.norm(rel - np.outer(rel @ seed[1], seed[1]), axis=1)
            points = points[off <= FIT_SEED_BAND_M]
    marked = len(points)
    if marked < FIT_MIN_POINTS:
        raise ValueError(
            f"only {marked} points with depth where marked (at least {FIT_MIN_POINTS}): mark more of it"
        )
    everything = points
    keep = np.ones(marked, dtype=bool)
    circle = None
    if axis == "up":  # the side first: the vertical surfaces, then the circle with the most of them on it
        points = points[_vertical_side(points)]
        no_side = (
            "the view does not show the object's side (looking down on it?): mark it in a view that does"
        )
        if len(points) < FIT_MIN_POINTS:
            raise ValueError(f"only {len(points)} points on a vertical side where marked: {no_side}")
        circle = _search_circle(points[:, :2])
        keep = np.abs(np.linalg.norm(points[:, :2] - circle[:2], axis=1) - circle[2]) < 2 * FIT_SEARCH_BAND_M
        if keep.sum() < FIT_MIN_POINTS:
            raise ValueError(f"no circle has {FIT_MIN_POINTS} side points on it: {no_side}")
    for _ in range(FIT_ROUNDS):
        inliers = points[keep]
        if len(inliers) < FIT_MIN_POINTS:
            raise ValueError(
                f"no cylinder fits the marked region ({len(inliers)} points on one): mark the object"
            )
        if axis == "up":
            xy, fitted, _ = _fit_upright(inliers, circle, held)
            circle = np.array([*xy, fitted])
            residuals = np.hypot(points[:, 0] - xy[0], points[:, 1] - xy[1]) - fitted
        else:
            p0, direction, fitted, _ = _fit_free(inliers, seed, held)
            rel = points - p0
            residuals = np.linalg.norm(rel - np.outer(rel @ direction, direction), axis=1) - fitted
        keep = _keep(residuals, keep)
    inliers = points[keep]
    if not FIT_RADIUS_RANGE_M[0] <= fitted <= FIT_RADIUS_RANGE_M[1]:
        raise ValueError(f"the fit's radius {fitted * 100:.1f} cm is not an object's: mark only the object")
    if axis == "up":
        # the bottom: where the side is seen down to; the top: anything over the side's footprint (a lid, a cap)
        over = np.linalg.norm(everything[:, :2] - xy, axis=1) <= fitted + FIT_KEEP_MIN_M
        z0 = float(np.percentile(inliers[:, 2], 2))
        z1 = float(np.percentile(everything[over, 2], 98))
        cylinder = Cylinder.upright(np.array([xy[0], xy[1], z0]), float(z1 - z0), fitted)
        around = inliers[:, :2] - xy
    else:
        if direction[2] < 0:
            direction = -direction
        t0, t1 = np.percentile((inliers - p0) @ direction, [2, 98])
        cylinder = Cylinder.between(p0 + t0 * direction, p0 + t1 * direction, fitted)
        frame = cylinder.root_from_obj[:3, :3]
        around = (inliers - p0) @ frame[:, :2]
    angles = np.sort(np.degrees(np.arctan2(around[:, 1], around[:, 0])) % 360.0)
    gaps = np.diff(np.r_[angles, angles[0] + 360.0])
    arc = 360.0 - float(gaps.max())
    rms = float(np.sqrt(np.mean(residuals[keep] ** 2)))
    note = (
        f"fitted to {len(inliers)} of {marked} points where marked, residual {rms * 100:.2f} cm, the side seen over "
        f"{arc:.0f} deg of it"
    )
    return Object(label, {part: cylinder}, view.capture, note)
