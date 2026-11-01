"""`render(view, *overlays)`: the one function that draws (`docs/harness.md`, section 1.D).

The view's crop of the capture is scaled first and the overlays are drawn after, in view pixels, so lines and text
stay sharp when zoomed in. Root-frame objects (`Grid`, `Points`, `Plan`, `Poses`) can be drawn in any view; a
`TableEstimate`'s mask only in views of its own capture. One style per overlay type, so the VLM learns one visual
language. Images are RGB `uint8`; the capture is never modified.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .objects import STATE_NAMES, Grid, Plan, Points, Poses, TableEstimate, View
from .scene import BLOCKED, FREE, UNKNOWN

STATE_COLORS = {FREE: (0, 200, 0), UNKNOWN: (255, 190, 0), BLOCKED: (230, 30, 30)}
GRID_COLOR = (255, 230, 0)
POINT_COLOR = (0, 220, 255)
TABLE_COLOR = (0, 160, 255)
GRIPPER_COLOR = (255, 60, 220)
GRIPPER_BODY_M = 0.04  # how much of the gripper axis is drawn, back from the pinch point
JAW_CLEARANCE_M = 0.01  # how much wider than the object the open jaws are drawn


def render(view: View, *overlays: Any, legend: bool = True) -> np.ndarray:
    """The view's image with `overlays` drawn in order.

    Overlays: `Grid`, `Points`, `Plan`, `TableEstimate`, `Poses`. Text goes on last, laid out so labels don't cover
    each other or the markers (`_Layout`). A `Plan` drawn with the `Poses` it goes through names its waypoints after
    the poses (`up1 (x,y,z)`), and those poses add no second label.
    """
    s = view.capture.scene
    crop = s.color[view.v0 : view.v0 + view.height, view.u0 : view.u0 + view.width]
    interpolation = cv2.INTER_CUBIC if view.scale > 1 else cv2.INTER_AREA
    out = cv2.resize(crop, view.size, interpolation=interpolation)
    layout = _Layout(out.shape)
    all_poses = [o for o in overlays if isinstance(o, Poses)]
    named_by = {id(o): _poses_of(o, all_poses) for o in overlays if isinstance(o, Plan)}
    labeled_elsewhere = {id(m) for m in named_by.values() if m is not None}
    plans = False
    for overlay in overlays:
        if isinstance(overlay, Grid):
            _draw_grid(out, view, overlay, layout)
        elif isinstance(overlay, Points):
            labels = [_point_label(n, p) for n, p in _named(overlay)]
            _draw_points(out, view.project(overlay.xyz), labels, layout)
        elif isinstance(overlay, Plan):
            poses = named_by[id(overlay)]
            _draw_plan(out, view, overlay, layout, poses.labels if poses is not None else None)
            plans = True
        elif isinstance(overlay, TableEstimate):
            _draw_table(out, view, overlay, layout)
        elif isinstance(overlay, Poses):
            _draw_poses(out, view, overlay, layout, labeled=id(overlay) not in labeled_elsewhere)
        else:
            raise TypeError(f"cannot render a {type(overlay).__name__}")
    if legend and plans:
        layout.reserve((0, 0, 90, 60))
    layout.draw(out)
    if legend and plans:
        _draw_legend(out)
    return out


def _poses_of(plan: Plan, poses: Sequence[Poses]) -> Poses | None:
    """The poses whose points are `plan`'s waypoints, if any."""
    for candidate in poses:
        if candidate.points.shape == plan.waypoints.shape and np.allclose(candidate.points, plan.waypoints):
            return candidate
    return None


def contact_sheet(
    views: Sequence[View],
    *overlays: Any,
    labels: Sequence[str] | None = None,
    columns: int = 3,
    width: int = 320,
) -> np.ndarray:
    """All `views` as labeled thumbnails `width` pixels wide, `columns` per row, with `overlays` drawn in each.

    Overlays are drawn at thumbnail size (crisp), so only root-frame ones make sense here (`Grid`, `Points`, `Plan`):
    the same grid in every thumbnail shows whether the captures agree where they overlap.
    """
    if not views:
        raise ValueError("no views")
    thumbs = []
    for i, view in enumerate(views):
        small = View(view.capture, view.u0, view.v0, view.width, view.height, width / view.width)
        image = render(small, *overlays, legend=False)
        _text(image, labels[i] if labels is not None else str(i + 1), (6, 18), 0.5)
        thumbs.append(image)
    h = max(t.shape[0] for t in thumbs)
    blank = np.zeros((h, width, 3), dtype=np.uint8)
    thumbs = [np.pad(t, ((0, h - t.shape[0]), (0, 0), (0, 0))) for t in thumbs]
    thumbs += [blank] * (-len(thumbs) % columns)
    rows = [np.hstack(thumbs[r : r + columns]) for r in range(0, len(thumbs), columns)]
    return np.vstack(rows)


# ---- primitives ---------------------------------------------------------------------------------------------------


def _text(
    img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.5, color=(255, 255, 255)
) -> None:
    """Text with a dark outline so it reads on any background."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


@dataclass
class _Label:
    text: str
    anchor: tuple[int, int]
    scale: float
    color: tuple[int, int, int]
    priority: int  # placed in order: 0 fixed footers, 1 point labels, 2 grid labels, 3 leg lengths
    required: bool  # placed even if it must overlap (else skipped)
    gap: int  # px from the anchor to the nearest spot
    fixed: bool  # at `anchor` exactly (a footer)
    near_only: bool  # only the spot next to the anchor (grid labels sit where their line leaves the image)


class _Layout:
    """Where the text of one render goes: labels avoid the markers and each other, and are drawn last, on top.

    Overlays draw their marks at once, `block` the markers and queue their text with `label`. `draw` then places the
    labels by priority, each at the first free spot around its anchor (right, left, above, below, then farther out
    with a thin leader line back to the anchor). An optional label with no free spot is skipped; a required one is
    put at its nearest spot anyway.
    """

    def __init__(self, shape: tuple[int, ...]):
        self.height, self.width = shape[0], shape[1]
        self.taken: list[tuple[int, int, int, int]] = []
        self.labels: list[_Label] = []

    def block(self, c: tuple[int, int], radius: int) -> None:
        """Keep text off a marker of `radius` px at `c`."""
        self.taken.append((c[0] - radius, c[1] - radius, c[0] + radius, c[1] + radius))

    def reserve(self, box: tuple[int, int, int, int]) -> None:
        """Keep text off a box (u0, v0, u1, v1), e.g. the legend."""
        self.taken.append(box)

    def label(
        self,
        text: str,
        anchor: tuple[int, int],
        scale: float = 0.5,
        color: tuple[int, int, int] = (255, 255, 255),
        priority: int = 1,
        required: bool = True,
        gap: int = 8,
        fixed: bool = False,
        near_only: bool = False,
    ) -> None:
        """Queue `text` near `anchor` (or exactly at it, with `fixed`)."""
        self.labels.append(_Label(text, anchor, scale, color, priority, required, gap, fixed, near_only))

    def draw(self, img: np.ndarray) -> list[tuple[str, tuple[int, int, int, int]]]:
        """Place every queued label and draw it; returns each placed label's text and box."""
        taken = list(self.taken)
        placed: list[tuple[_Label, tuple[int, int], bool]] = []
        for label in sorted(
            self.labels, key=lambda lab: lab.priority
        ):  # stable: same priority keeps its order
            (w, h), _ = cv2.getTextSize(label.text, cv2.FONT_HERSHEY_SIMPLEX, label.scale, 1)
            spots = self._spots(label, w, h)
            choice = next(((org, far) for org, far in spots if self._free(_box(org, w, h), taken)), None)
            if choice is None:
                if not label.required:
                    continue
                choice = spots[0]
            taken.append(_box(choice[0], w, h))
            placed.append((label, *choice))
        for label, org, far in placed:
            if far:  # a leader line from the anchor to the label's nearest corner
                (w, h), _ = cv2.getTextSize(label.text, cv2.FONT_HERSHEY_SIMPLEX, label.scale, 1)
                u = int(np.clip(label.anchor[0], org[0], org[0] + w))
                v = int(np.clip(label.anchor[1], org[1] - h, org[1]))
                cv2.line(img, label.anchor, (u, v), label.color, 1, cv2.LINE_AA)
            _text(img, label.text, org, label.scale, label.color)
        return [
            (label.text, _box(org, *cv2.getTextSize(label.text, cv2.FONT_HERSHEY_SIMPLEX, label.scale, 1)[0]))
            for label, org, _ in placed
        ]

    def _spots(self, label: _Label, w: int, h: int) -> list[tuple[tuple[int, int], bool]]:
        """Candidate baselines for a `w` x `h` label, nearest first, kept in the image; True if it needs a leader."""
        x, y = label.anchor
        if label.fixed:
            return [((x, y), False)]
        if label.near_only:
            return [(self._clip(x + 3, y - 3, w, h), False)]
        spots = []
        for ring in (1, 2, 3, 5):
            g = label.gap * ring
            for dx, dy in (
                (g, -g),
                (g, h // 2),
                (g, g + h),
                (-g - w, -g),
                (-g - w, h // 2),
                (-g - w, g + h),
                (-w // 2, -g),
                (-w // 2, g + h),
            ):
                spots.append((self._clip(x + dx, y + dy, w, h), ring > 1))
        return spots

    def _clip(self, u: int, v: int, w: int, h: int) -> tuple[int, int]:
        return int(np.clip(u, 0, max(0, self.width - w - 1))), int(
            np.clip(v, h + 1, max(h + 1, self.height - 2))
        )

    @staticmethod
    def _free(box: tuple[int, int, int, int], taken: list[tuple[int, int, int, int]]) -> bool:
        return not any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3] for b in taken)


def _box(org: tuple[int, int], w: int, h: int) -> tuple[int, int, int, int]:
    """The box of text drawn at baseline `org`, with a 2 px margin."""
    return org[0] - 2, org[1] - h - 2, org[0] + w + 2, org[1] + 4


def _inside(img: np.ndarray, uv: np.ndarray) -> bool:
    return bool(np.isfinite(uv).all() and 0 <= uv[0] < img.shape[1] and 0 <= uv[1] < img.shape[0])


def _pixel(uv: np.ndarray) -> tuple[int, int]:
    return round(float(uv[0])), round(float(uv[1]))


def _polyline(img: np.ndarray, uv: np.ndarray, color, thickness: int = 1) -> None:
    """Connect consecutive finite pixels (a NaN breaks the line: that sample was behind the camera)."""
    for a, b in zip(uv[:-1], uv[1:], strict=True):
        if np.isfinite(a).all() and np.isfinite(b).all():
            cv2.line(img, _pixel(a), _pixel(b), color, thickness, cv2.LINE_AA)


def _draw_points(
    img: np.ndarray, uv: np.ndarray, labels: list[str], layout: "_Layout", color=POINT_COLOR, radius: int = 6
) -> None:
    """Markers, labeled through `layout`; points off the image or with NaN pixels are skipped."""
    for p, label in zip(np.atleast_2d(uv), labels, strict=True):
        if not _inside(img, p):
            continue
        c = _pixel(p)
        cv2.circle(img, c, radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, c, radius, color, -1, cv2.LINE_AA)
        layout.block(c, radius + 2)
        layout.label(label, c, gap=radius + 4)


def _cm_text(value: float) -> str:
    """Meters as cm with at most one decimal and no trailing zero: 0.4 -> "40", -1.075 -> "-107.5"."""
    text = f"{100 * value:.1f}".removesuffix(".0")
    return "0" if text == "-0" else text


def _cm(xyz: np.ndarray) -> str:
    return "(" + ",".join(f"{100 * c:.1f}" for c in xyz) + ")"


def _named(points: Points):
    return zip(points.labels, points.xyz, strict=True)


def _point_label(name: str, xyz: np.ndarray) -> str:
    return f"{name} {_cm(xyz)}" if np.isfinite(xyz).all() else f"{name} (no depth)"


# ---- overlays -----------------------------------------------------------------------------------------------------


def _draw_grid(
    img: np.ndarray,
    view: View,
    grid: Grid,
    layout: "_Layout",
    label_every: int = 5,
    occlusion_m: float = 0.02,
) -> None:
    """Lines on the plane; every `label_every`-th one is stronger and labeled.

    A label is the line's coordinate in cm ("x40" is the line x = 40 cm), placed where the line leaves the image:
    x lines at their lowest point, y lines at their leftmost.

    The plane is hidden where the view's depth shows a surface more than `occlusion_m` in front of it (an object,
    the arm), so the grid shows only where the plane itself is visible. Labels that would overlap anything are
    skipped (`layout`).
    """
    original, minor, major = img.copy(), img.copy(), img.copy()
    samples = max(20, int(4 * max(np.ptp(grid.x_range), np.ptp(grid.y_range)) / grid.cell))
    for axis, values, span in (("x", grid.xs, grid.y_range), ("y", grid.ys, grid.x_range)):
        t = np.linspace(span[0], span[1], samples)
        for value in values:
            const = np.full_like(t, value)
            xy = np.stack([const, t] if axis == "x" else [t, const], axis=-1)
            xyz = grid.point(xy)
            uv = view.project(xyz)
            uv[_occluded(view, xyz, occlusion_m)] = np.nan
            is_major = round(value / grid.cell) % label_every == 0
            _polyline(major if is_major else minor, uv, GRID_COLOR)
            on = np.array([_inside(img, p) for p in uv])
            if not is_major or not on.any():
                continue
            pick = np.flatnonzero(on)[np.argmax(uv[on, 1]) if axis == "x" else np.argmin(uv[on, 0])]
            text = f"{axis}{_cm_text(value)}"
            layout.label(text, _pixel(uv[pick]), 0.4, GRID_COLOR, priority=2, required=False, near_only=True)
    # half-transparent minor lines, stronger major ones: the scene stays visible under the grid
    cv2.addWeighted(minor, 0.35, img, 0.65, 0, dst=img)
    blend = cv2.addWeighted(major, 0.8, img, 0.2, 0)
    changed = (major != original).any(axis=-1)  # only where a major line was drawn
    img[changed] = blend[changed]


def _occluded(view: View, xyz: np.ndarray, margin: float) -> np.ndarray:
    """True for root-frame points that the view's depth shows behind a surface (by more than `margin`).

    Points with no depth at their pixel count as visible.
    """
    s = view.capture.scene
    cam = view.capture.to_camera(xyz)
    uv = s.xyz_to_pixel(cam)
    inside = s.in_image(uv)
    hidden: np.ndarray = np.zeros(len(xyz), dtype=bool)
    if inside.any():
        ui = np.rint(uv[inside, 0]).astype(int)
        vi = np.rint(uv[inside, 1]).astype(int)
        with np.errstate(invalid="ignore"):
            hidden[inside] = s.depth_m[vi, ui] < cam[inside, 2] - margin
    return hidden


def _draw_plan(
    img: np.ndarray, view: View, plan: Plan, layout: "_Layout", names: Sequence[str] | None = None
) -> None:
    """The path, with each sample colored by its state.

    A thin white line joins the waypoints; the waypoints are labeled with their name (`names`, default 1, 2, ...) and
    xyz in cm, and each leg shows its length in cm where there is room.
    """
    uv_way = view.project(plan.waypoints)
    _polyline(img, uv_way, (255, 255, 255))
    uv_samples = view.project(plan.check.sample_xyz)
    for leg_uv, leg_state in zip(uv_samples, plan.check.sample_state, strict=True):
        for p, state in zip(leg_uv, leg_state, strict=True):
            if _inside(img, p):
                cv2.circle(img, _pixel(p), 2, STATE_COLORS[int(state)], -1, cv2.LINE_AA)
    lengths = np.linalg.norm(np.diff(plan.waypoints, axis=0), axis=1)
    for a, b, length in zip(uv_way[:-1], uv_way[1:], lengths, strict=True):
        mid = (a + b) / 2
        if _inside(img, mid):
            layout.label(f"{100 * length:.1f} cm", _pixel(mid), 0.45, priority=3, required=False)
    names = list(names) if names is not None else [str(i + 1) for i in range(len(plan.waypoints))]
    # waypoints at the same place (down to a grasp and back up) share one marker: "1,3 (x,y,z)"
    groups: list[list[int]] = []
    for i, p in enumerate(plan.waypoints):
        same = next((g for g in groups if np.linalg.norm(plan.waypoints[g[0]] - p) < 1e-3), None)
        if same is None:
            groups.append([i])
        else:
            same.append(i)
    labels = [_point_label(",".join(names[i] for i in g), plan.waypoints[g[0]]) for g in groups]
    _draw_points(img, uv_way[[g[0] for g in groups]], labels, layout, color=(255, 220, 0))


def _draw_poses(img: np.ndarray, view: View, poses: Poses, layout: "_Layout", labeled: bool = True) -> None:
    """Each pose as a gripper; without poses (no pitch found), an X on each target.

    A gripper is the axis back from the pinch point, the open jaws (a square at the fixed jaw, the pinch point, and
    one at the moving jaw, opened past `width`), and a ring at the tool point, green if reached and red with an X if
    not, with its label (unless `labeled` is False: a plan drawn with them names its waypoints after them).
    """
    if not len(poses.points):
        for target in poses.targets if poses.targets is not None else ():
            p = view.project(target[None])[0]
            if _inside(img, p):
                _cross(img, _pixel(p), STATE_COLORS[BLOCKED])
        layout.label(f"{poses.arm} arm: no pitch", (8, img.shape[0] - 30), priority=0, fixed=True)
        return
    for frame, point, label, ok in zip(poses.frames, poses.points, poses.labels, poses.ok, strict=True):
        pinch, jaw, axis = frame[:3, 3], frame[:3, 0], frame[:3, 2]
        moving = pinch - (poses.width + JAW_CLEARANCE_M) * jaw  # the moving jaw opens along -x
        uv = view.project(np.array([pinch - GRIPPER_BODY_M * axis, pinch, moving, point]))
        _polyline(img, uv[[0, 1]], GRIPPER_COLOR, 2)
        _polyline(img, uv[[1, 2]], GRIPPER_COLOR, 2)
        for corner in uv[[1, 2]]:
            if _inside(img, corner):
                q = _pixel(corner)
                cv2.rectangle(img, (q[0] - 3, q[1] - 3), (q[0] + 3, q[1] + 3), GRIPPER_COLOR, -1)
        if _inside(img, uv[3]):
            c = _pixel(uv[3])
            color = STATE_COLORS[FREE] if ok else STATE_COLORS[BLOCKED]
            cv2.circle(img, c, 9, color, 2, cv2.LINE_AA)
            if not ok:
                _cross(img, c, color)
            layout.block(c, 11)
            if labeled:
                layout.label(label, c, 0.45, GRIPPER_COLOR, gap=13)
    pitch = "any pitch" if poses.pitch_deg is None else f"pitch {poses.pitch_deg:+.0f} deg"
    layout.label(f"{poses.arm} arm, {pitch}", (8, img.shape[0] - 30), priority=0, fixed=True)


def _cross(img: np.ndarray, c: tuple[int, int], color) -> None:
    for d in (-1, 1):
        cv2.line(img, (c[0] - 7, c[1] - 7 * d), (c[0] + 7, c[1] + 7 * d), color, 2, cv2.LINE_AA)


def _draw_table(img: np.ndarray, view: View, table: TableEstimate, layout: "_Layout") -> None:
    """Tint the inlier pixels (only in a view of the estimate's own capture)."""
    if table.capture is not view.capture:
        raise ValueError("a TableEstimate's mask can only be drawn in a view of its own capture")
    mask = table.inlier_mask[view.v0 : view.v0 + view.height, view.u0 : view.u0 + view.width]
    mask = cv2.resize(mask.astype(np.uint8), view.size, interpolation=cv2.INTER_NEAREST).astype(bool)
    tint = np.array(TABLE_COLOR, dtype=np.float32)
    img[mask] = (0.65 * img[mask] + 0.35 * tint).astype(np.uint8)
    layout.label(
        f"table z = {100 * table.h:.1f} cm", (8, img.shape[0] - 10), color=TABLE_COLOR, priority=0, fixed=True
    )


def _draw_legend(img: np.ndarray) -> None:
    for row, state in enumerate((FREE, UNKNOWN, BLOCKED)):
        y = 16 + 18 * row
        cv2.circle(img, (14, y - 4), 5, STATE_COLORS[state], -1, cv2.LINE_AA)
        _text(img, STATE_NAMES[state], (26, y), 0.5)
