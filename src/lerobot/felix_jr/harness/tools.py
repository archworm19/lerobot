"""Long-term tools: pure functions over the harness's objects (`docs/harness.md`, section 1.H, layer 2).

They know nothing about the workspace; `Workspace.call` records them. Lengths are meters and points are in the root
frame; pixels are view pixels unless the argument says capture.
"""

from collections.abc import Sequence

import numpy as np

from .objects import Capture, Focus, Grid, Plan, Points, TableEstimate, View

MAX_GRID_LINES = 150  # per axis: more is unreadable (and slow); zoom in and use a smaller extent instead


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


def check_path(
    capture: Capture,
    waypoints: np.ndarray,
    radius: float = 0.02,
    tolerance: float = 0.01,
    step: float = 0.01,
) -> Plan:
    """Check the straight legs between root-frame `waypoints` (M, 3), M >= 2, against the capture's depth.

    Each sample gets FREE / UNKNOWN / BLOCKED for a ball of `radius` (`Scene.segment_check`).
    """
    waypoints = np.asarray(waypoints, dtype=np.float64)
    if waypoints.ndim != 2 or waypoints.shape[1] != 3 or len(waypoints) < 2:
        raise ValueError(f"waypoints must have shape (M, 3) with M >= 2, got {waypoints.shape}")
    cam = capture.to_camera(waypoints)
    check = capture.scene.segment_check(cam[:-1], cam[1:], radius=radius, tolerance=tolerance, step=step)
    check = check._replace(
        sample_xyz=capture.to_root(check.sample_xyz), point_first=capture.to_root(check.point_first)
    )
    return Plan(waypoints, check, capture, radius)
