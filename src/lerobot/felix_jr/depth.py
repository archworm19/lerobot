"""Depth helpers for the RealSense D415: aligning depth to color and checking that the alignment is right.

The D415's depth and color sensors are separate, with different intrinsics and about 15 mm between them, so
depth pixel (u, v) is not color pixel (u, v). `align_depth_to_color` re-projects the depth map into the color
camera's pixel grid using the calibration saved by `diagnostics/realsense_check.py` (`calibration.json`).

Conventions: raw depth is `uint16`, 0 means "no depth", and `depth_scale_m_per_unit` converts to meters. The
D415 reports zero lens distortion for both sensors, so distortion is ignored here.
"""

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def with_matrices(calib: dict[str, Any]) -> dict[str, Any]:
    """Add the depth-to-color rotation as a proper 3x3 matrix (`rotation`) and `translation` as an array."""
    # librealsense stores the 3x3 rotation column-major. Older files called this key "..._row_major"
    # by mistake; both hold the same column-major numbers.
    flat = calib.get("depth_to_color_rotation_col_major", calib.get("depth_to_color_rotation_row_major"))
    return {
        **calib,
        "rotation": np.array(flat, dtype=float).reshape(3, 3).T,
        "translation": np.array(calib["depth_to_color_translation_m"], dtype=float),
    }


def load_calibration(path: str | Path) -> dict[str, Any]:
    """Load `calibration.json` written by `diagnostics/realsense_check.py`."""
    return with_matrices(json.loads(Path(path).read_text()))


def fill_cracks(depth: np.ndarray, max_step_units: float = 30.0) -> np.ndarray:
    """Fill single-pixel gaps left by forward-warping. A zero pixel is filled with the mean of its valid
    3x3 neighbors, but only if there are at least 3 of them and they agree to within `max_step_units`
    (30 raw units = 3 cm for the D415), so real depth discontinuities and occlusion shadows are not
    bridged."""
    d = depth.astype(np.float32)
    valid = (d > 0).astype(np.float32)
    kernel = np.ones((3, 3), np.uint8)
    count = cv2.boxFilter(valid, -1, (3, 3), normalize=False)
    total = cv2.boxFilter(d, -1, (3, 3), normalize=False)
    biggest = cv2.dilate(d, kernel)
    smallest = -cv2.dilate(np.where(valid > 0, -d, -1e9).astype(np.float32), kernel)  # min over valid pixels
    fill = (d == 0) & (count >= 3) & (biggest - smallest <= max_step_units)
    out = d.copy()
    out[fill] = total[fill] / count[fill]
    return out.round().astype(np.uint16)


def align_depth_to_color(depth: np.ndarray, calib: dict[str, Any], fill: bool = True) -> np.ndarray:
    """Re-project a raw depth map into the color camera's pixel grid. Returns raw `uint16`, 0 = no depth.

    Each valid depth pixel is turned into a 3D point (depth intrinsics), moved into the color camera's frame
    (extrinsics) and projected with the color intrinsics. Where several depth pixels land on one color
    pixel, the nearest wins. Color pixels no depth pixel lands on stay 0 (occlusion shadows); with `fill`,
    one-pixel cracks caused by the resampling are filled (`fill_cracks`).
    """
    d = np.squeeze(depth)
    di, ci = calib["depth_intrinsics"], calib["color_intrinsics"]
    v, u = np.nonzero(d)
    z = d[v, u].astype(np.float64) * calib["depth_scale_m_per_unit"]
    points = np.stack([(u - di["ppx"]) / di["fx"] * z, (v - di["ppy"]) / di["fy"] * z, z])
    pc = calib["rotation"] @ points + calib["translation"][:, None]
    keep = pc[2] > 1e-6
    pc = pc[:, keep]
    uc = np.round(pc[0] / pc[2] * ci["fx"] + ci["ppx"]).astype(int)
    vc = np.round(pc[1] / pc[2] * ci["fy"] + ci["ppy"]).astype(int)
    inside = (uc >= 0) & (uc < ci["width"]) & (vc >= 0) & (vc < ci["height"])
    flat = vc[inside] * ci["width"] + uc[inside]
    nearest = np.full(ci["width"] * ci["height"], np.inf)
    np.minimum.at(nearest, flat, pc[2][inside])
    aligned = np.where(np.isfinite(nearest), nearest / calib["depth_scale_m_per_unit"], 0)
    aligned = aligned.reshape(ci["height"], ci["width"]).round().astype(np.uint16)
    return fill_cracks(aligned) if fill else aligned


def depth_edges(
    depth: np.ndarray, depth_scale: float, min_step_m: float = 0.10, median: int = 5
) -> np.ndarray:
    """Boolean mask of depth discontinuities: pixels whose depth differs from a right/lower neighbor by at
    least `min_step_m` (default 10 cm), marking the nearer of the two pixels (the object's outline, which is
    what shifts most under alignment). Only valid-to-valid steps count, so the border of a hole in the depth
    map is not reported as an edge. The map is median-filtered first (`median` is the odd kernel size, 0 =
    off) because far surfaces are noisy at the centimeter level.
    """
    d = np.squeeze(depth)
    if median:
        d = cv2.medianBlur(d, median)
    valid = d > 0
    meters = d.astype(np.float32) * depth_scale
    edges = np.zeros(d.shape, dtype=bool)
    for dy, dx in ((0, 1), (1, 0)):
        h, w = d.shape[0] - dy, d.shape[1] - dx
        a, b = meters[:h, :w], meters[dy:, dx:]  # each pixel and its right (or lower) neighbor
        jump = valid[:h, :w] & valid[dy:, dx:] & (np.abs(a - b) >= min_step_m)
        # the silhouette belongs to the nearer surface, so mark the nearer pixel of the pair
        edges[:h, :w] |= jump & (a < b)
        edges[dy:, dx:] |= jump & (a >= b)
    return edges


def smooth_depth(depth: np.ndarray, kernel: int = 5) -> np.ndarray:
    """Median-filter a raw depth map (suppresses far-field noise). Use the same smoothed map for finding edges
    (`depth_edges(..., median=0)`) and for warping them, since filtering can fill pixels that were 0."""
    return cv2.medianBlur(np.squeeze(depth), kernel)


def warp_mask_to_color(mask: np.ndarray, depth: np.ndarray, calib: dict[str, Any]) -> np.ndarray:
    """Move the pixels of `mask` (a depth-grid mask) to where those same points appear in the color image.

    Each pixel moves by an amount that depends on its own depth (parallax), so this is the "same edges,
    moved" counterpart of `align_depth_to_color`, which warps the whole depth map.
    """
    d = np.squeeze(depth)
    di, ci = calib["depth_intrinsics"], calib["color_intrinsics"]
    v, u = np.nonzero(mask & (d > 0))
    z = d[v, u].astype(np.float64) * calib["depth_scale_m_per_unit"]
    points = np.stack([(u - di["ppx"]) / di["fx"] * z, (v - di["ppy"]) / di["fy"] * z, z])
    pc = calib["rotation"] @ points + calib["translation"][:, None]
    uc = np.round(pc[0] / pc[2] * ci["fx"] + ci["ppx"]).astype(int)
    vc = np.round(pc[1] / pc[2] * ci["fy"] + ci["ppy"]).astype(int)
    inside = (uc >= 0) & (uc < ci["width"]) & (vc >= 0) & (vc < ci["height"])
    out = np.zeros((ci["height"], ci["width"]), dtype=bool)
    out[vc[inside], uc[inside]] = True
    return out


def colorize_depth(depth: np.ndarray, max_units: float | None = None) -> np.ndarray:
    """Turn raw depth into an RGB image (JET); 0 (no depth) is shown black."""
    d = np.squeeze(depth)
    top = max_units or max(1.0, float(np.percentile(d[d > 0], 98))) if (d > 0).any() else 1.0
    vis = cv2.applyColorMap(cv2.convertScaleAbs(d, alpha=255.0 / top), cv2.COLORMAP_JET)
    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    vis[d == 0] = 0
    return vis


def overlay_edges(
    color_rgb: np.ndarray, edges: np.ndarray, rgb: tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """Paint `edges` (slightly thickened) over the color image."""
    out = color_rgb.copy()
    out[cv2.dilate(edges.astype(np.uint8), np.ones((2, 2), np.uint8)).astype(bool)] = rgb
    return out


def blend(color_rgb: np.ndarray, depth_rgb: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    return cv2.addWeighted(color_rgb, 1 - alpha, depth_rgb, alpha, 0)


def _label(img: np.ndarray, text: str) -> None:
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)


def alignment_panel(
    color_rgb: np.ndarray, depth: np.ndarray, calib: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (panel, zoom).

    panel: 2x2. Left column: the depth edges (>= 10 cm jumps) painted in green over the color image, top as
    they sit in the depth image, bottom the same edges moved to their place in the color image. Right column:
    colorized depth blended 50 % over the color image, top the depth map as-is, bottom aligned (same color
    scale in both).
    zoom: a 3x enlargement of the area around the nearest objects, where the shift is largest: color only,
    then the as-is blend, then the aligned blend.

    Moved edges are thickened a little because warping a thin line leaves gaps.
    """
    scale = calib["depth_scale_m_per_unit"]
    raw = np.squeeze(depth)
    aligned = align_depth_to_color(raw, calib)
    smooth = smooth_depth(raw)
    edges = depth_edges(smooth, scale, median=0)
    moved = warp_mask_to_color(edges, smooth, calib).astype(np.uint8)
    moved = cv2.dilate(moved, np.ones((3, 3), np.uint8)).astype(bool)
    top = max(1.0, float(np.percentile(raw[raw > 0], 98))) if (raw > 0).any() else 1.0
    blends = {
        "as-is": blend(color_rgb, colorize_depth(raw, top)),
        "aligned": blend(color_rgb, colorize_depth(aligned, top)),
    }
    rows = []
    for label, mask in (("as-is", edges), ("aligned", moved)):
        left = overlay_edges(color_rgb, mask)
        right = blends[label]
        _label(left, f"{label}: depth edges (>=10 cm jumps) on color")
        _label(right, f"{label}: colorized depth blended on color")
        rows.append(np.hstack([left, right]))

    # zoom on the nearest 5 % of the aligned depth
    valid = aligned > 0
    near = valid & (aligned <= np.percentile(aligned[valid], 5)) if valid.any() else valid
    ys, xs = np.nonzero(near) if near.any() else (np.array([240]), np.array([320]))
    cy, cx = int(np.median(ys)), int(np.median(xs))
    half_w, half_h = 130, 80
    x0 = int(np.clip(cx - half_w, 0, color_rgb.shape[1] - 2 * half_w))
    y0 = int(np.clip(cy - half_h, 0, color_rgb.shape[0] - 2 * half_h))
    tiles = []
    for label, img in (
        ("color only", color_rgb),
        ("depth as-is", blends["as-is"]),
        ("depth aligned", blends["aligned"]),
    ):
        crop = img[y0 : y0 + 2 * half_h, x0 : x0 + 2 * half_w]
        tile = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        _label(tile, label)
        tiles.append(tile)
    return np.vstack(rows), np.vstack(tiles)
