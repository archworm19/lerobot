"""Draw points and paths on a scene's color image, with metric labels, for a VLM prompt or a person.

All functions return a copy and leave the input image untouched. Images are RGB `uint8`, and colors are RGB.
"""

import cv2
import numpy as np

from lerobot.felix_jr.scene import BLOCKED, FREE, UNKNOWN, Scene, SegmentCheck

STATE_COLORS = {FREE: (0, 200, 0), UNKNOWN: (255, 190, 0), BLOCKED: (230, 30, 30)}
STATE_NAMES = {FREE: "free", UNKNOWN: "unknown", BLOCKED: "blocked"}


def _text(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.5) -> None:
    """Text with a dark outline so it reads on any background."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def draw_points(
    image: np.ndarray,
    uv: np.ndarray,
    labels: list[str] | None = None,
    color: tuple[int, int, int] = (255, 220, 0),
    radius: int = 6,
) -> np.ndarray:
    """Numbered markers at pixels `uv` (N, 2). `labels` defaults to 1..N. Points with NaN or off-image
    coordinates are skipped."""
    out = image.copy()
    uv = np.atleast_2d(np.asarray(uv, dtype=np.float64))
    for i, (u, v) in enumerate(uv):
        if not (np.isfinite(u) and np.isfinite(v)):
            continue
        centre = (int(round(u)), int(round(v)))
        if not (0 <= centre[0] < out.shape[1] and 0 <= centre[1] < out.shape[0]):
            continue
        cv2.circle(out, centre, radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(out, centre, radius, color, -1, cv2.LINE_AA)
        _text(out, labels[i] if labels else str(i + 1), (centre[0] + radius + 3, centre[1] - radius - 2))
    return out


def render_plan(
    scene: Scene,
    waypoints: np.ndarray,
    check: SegmentCheck | None = None,
    legend: bool = True,
    **check_kwargs: float,
) -> np.ndarray:
    """The scene's color image with a path over it. `waypoints`: (M, 3) camera-frame points, M >= 2.

    The path is drawn as dots colored by the state of each sample (green = free, yellow = unknown, red =
    blocked) from `check` (computed with `scene.segment_check(..., **check_kwargs)` if not given).
    Waypoints are numbered with their xyz, and each leg is labeled with its length in meters.
    """
    waypoints = np.asarray(waypoints, dtype=np.float64)
    if waypoints.ndim != 2 or waypoints.shape[1] != 3 or len(waypoints) < 2:
        raise ValueError(f"waypoints must have shape (M, 3) with M >= 2, got {waypoints.shape}")
    if check is None:
        check = scene.segment_check(waypoints[:-1], waypoints[1:], **check_kwargs)

    out = scene.color.copy()
    # connecting line first (thin, so the state dots stay readable), then the state dots
    uv_way = scene.xyz_to_pixel(waypoints)
    for a, b in zip(uv_way[:-1], uv_way[1:], strict=True):
        if np.isfinite(a).all() and np.isfinite(b).all():
            start, end = tuple(int(round(c)) for c in a), tuple(int(round(c)) for c in b)
            cv2.line(out, start, end, (255, 255, 255), 1, cv2.LINE_AA)
    uv_samples = scene.xyz_to_pixel(check.sample_xyz)
    for seg_uv, seg_state in zip(uv_samples, check.sample_state, strict=True):
        for (u, v), state in zip(seg_uv, seg_state, strict=True):
            if np.isfinite(u) and np.isfinite(v):
                cv2.circle(out, (int(round(u)), int(round(v))), 2, STATE_COLORS[int(state)], -1, cv2.LINE_AA)

    lengths = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    for (a, b), length in zip(zip(uv_way[:-1], uv_way[1:], strict=True), lengths, strict=True):
        if np.isfinite(a).all() and np.isfinite(b).all():
            mid = (a + b) / 2
            _text(out, f"{length:.2f} m", (int(mid[0]) + 4, int(mid[1]) + 14), 0.45)
    labels = [f"{i + 1} ({x:+.2f},{y:+.2f},{z:.2f})" for i, (x, y, z) in enumerate(waypoints)]
    out = draw_points(out, uv_way, labels)

    if legend:
        for row, state in enumerate((FREE, UNKNOWN, BLOCKED)):
            y = 16 + 18 * row
            cv2.circle(out, (14, y - 4), 5, STATE_COLORS[state], -1, cv2.LINE_AA)
            _text(out, STATE_NAMES[state], (26, y), 0.5)
    return out
