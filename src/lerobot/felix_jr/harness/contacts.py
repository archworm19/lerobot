"""The contact patches at a hold, from the wrist camera: DA3's depth, a primitive on the graspable part, the move.

`docs/object_grasping.md` section 7 (bowl7, bowl8). At a hold:

1. **Views** (`hold_views`): a short wrist roll (the wrist only: +-15 deg, 3 frames) and back; or, too close to roll,
   the one current frame. Each is posed by the arm's joints through the model (`WristView.root_from_camera`): for a
   wrist-only roll that did as well as structure from motion (the ablation in section 7).
2. **Depth** (`harness.depth_anything`): DA3-SMALL on the frames with those poses; its flying pixels at depth edges and
   its least confident third dropped.
3. **Scale:** the poses' scale is the model's, which is short (~5-20%); it is set from something metric: the table's
   height (the head's depth, `table_z`) under the graspable part, or, without it, the fixed finger (rigid to the
   camera: its points' known depths in the current frame).
4. **The graspable part:** the points that land in `box` (the current frame's pixels) and within `band` (heights above
   the table): the model's choice (the body, not the cap: bowl7). An upright cylinder is fitted (the radius held if
   given: thin objects' free fits run wide, bowl8).
5. **The contact patches** (`patches_from_points`): where the jaws' closing plane (the fingers' middle) meets the
   cylinder in its widest chord: how far beyond the fingertips (along the approach), and where the fixed and the moving
   jaw would touch (along the closing line, from the fixed finger). **The move** puts that chord `into` the fingers and
   the fixed finger `gap` from its contact: `advance` along the approach, `shift` along the closing line (its sign is
   which side of the fixed finger the contact is on: bowl8 got it wrong once by hand).

Everything is in the pinch frame of the current view (the jaws now), so a move planned from it (`plan_wrist_move` in
`harness.wrist`) is relative to where the arm is: the model's errors over large moves do not enter (bowl8: a checkpoint
carried 3.4 cm through the model put the object through the fixed finger; re-measure at the pose).
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

import cv2
import numpy as np

from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel
from lerobot.utils.import_utils import _scipy_available, require_package

from .wrist import WristView

if TYPE_CHECKING or _scipy_available:
    from scipy.optimize import least_squares

if TYPE_CHECKING:
    from lerobot.felix_jr.server.client import FelixClient

    from .depth_anything import DepthAnything

EDGE_JUMP = (
    0.01  # a pixel whose depth changes by more than this fraction per pixel is on an edge (flying pixels)
)
LOW_CONFIDENCE = 33  # DA3's least confident percent of each frame is dropped
RING_PX = 40  # the table under the graspable part: a ring this wide around its box
GAP_M = 0.003  # the fixed finger this far from its contact
INTO_M = 0.01  # the widest chord this far into the fingers
FINGER_POINTS = (
    10  # along the fixed finger's camera-side face, 3 mm to 3 cm back from its tip (the finger scale)
)

_MODEL: list["DepthAnything"] = []  # DA3, loaded on first use and kept


def _cm(value: float) -> str:
    return f"{100 * value:+.2f}"


@dataclass(frozen=True, eq=False)
class HoldViews:
    """Wrist frames at a hold (`hold_views`): the frames, their labels, and the one the arm is at now."""

    sensed: ClassVar[bool] = True  # the scene now (`Workspace.add`)
    views: tuple[WristView, ...]
    labels: tuple[str, ...]
    current: int

    def describe(self) -> str:
        """The frames."""
        return (
            f"{len(self.views)} {self.views[0].arm} wrist frames at a hold "
            f"({', '.join(self.labels)}; the arm is at {self.labels[self.current]})"
        )


@dataclass(frozen=True)
class Patches:
    """The contact geometry in the jaws' frame (`patches_from_points`); lengths in m."""

    radius: float
    fit_rms_m: float
    chord: float  # the widest chord's length (the jaws' plane through the cylinder)
    ahead: float  # how far beyond the fingertips it is, along the approach (negative: between the fingers)
    fixed_contact: float  # along the closing line, from the fixed finger (positive: toward the moving jaw)
    moving_contact: float
    advance: float  # the move along the approach that puts the chord `into` the fingers
    shift: float  # the move along the closing line that leaves the fixed finger `gap` from its contact
    move_pinch: np.ndarray  # (3,) both, in the pinch frame
    axis_pinch: np.ndarray  # (3,) the cylinder's axis's foot (on the table), pinch frame
    approach_pinch: np.ndarray  # (3,)
    closing_pinch: np.ndarray  # (3,)
    middle_pinch: (
        np.ndarray
    )  # (3,) the fingers' middle at the fixed tip: the origin of `ahead` and the contacts


@dataclass(frozen=True, eq=False)
class ContactPatches:
    """The contact patches at a hold (`contact_patches`), with the move, the evidence and a picture."""

    view: WristView  # the current frame: the jaws the patches are relative to
    patches: Patches
    points: int
    scale: float
    scale_source: str
    move_root: np.ndarray  # (3,) the move in the root frame (through the current pinch frame)
    image: np.ndarray = field(repr=False)

    def describe(self) -> str:
        """The cylinder, the contacts, the move, and how the scale was set."""
        p = self.patches
        return (
            f"{self.points} points on the graspable part; a cylinder of radius {100 * p.radius:.2f} cm (fit "
            f"{1000 * p.fit_rms_m:.1f} mm); the widest chord ({100 * p.chord:.2f} cm) {_cm(p.ahead)} cm along the "
            f"approach from the fingertips (+: beyond them); the fixed jaw's contact {_cm(p.fixed_contact)} cm, the "
            f"moving jaw's {_cm(p.moving_contact)} cm along the closing line (from the fixed finger, + toward the "
            f"moving jaw) -> advance {_cm(p.advance)} cm, shift {_cm(p.shift)} cm (root "
            f"{np.round(100 * self.move_root, 2).tolist()} cm); the scale x{self.scale:.3f} from {self.scale_source}"
        )


def _jaw_axes(model: XleRobotModel, arm: str, width: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The jaws' axes in the pinch frame: the closing line, the approach, the fingers' middle.

    The closing line runs from the fixed to the moving tip at `width`, the approach forward along the fingers, and the
    middle is the closing plane's origin.
    """
    tip = model.moving_jaw(arm, model.opening_for_width(arm, max(width, 1e-3)))[:3, 3]
    closing = tip / np.linalg.norm(tip)
    hinge = np.array([0.0, 1.0, 0.0]) - closing[1] * closing
    hinge /= np.linalg.norm(hinge)
    approach = np.cross(closing, hinge)
    approach = approach if approach[2] > 0 else -approach
    return closing, approach, model.wrist_face(arm) / 2


def patches_from_points(
    model: XleRobotModel,
    arm: str,
    points_pinch: np.ndarray,
    up_pinch: np.ndarray,
    table: float,
    radius: float | None = None,
    gap: float = GAP_M,
    into: float = INTO_M,
) -> Patches:
    """An upright cylinder fitted to `points_pinch` (N, 3) and its contact patches (the module docstring).

    `up_pinch`: the vertical in the pinch frame; `table`: the table's height along it. `radius` held if given.
    """
    require_package("scipy", extra="felix_jr")
    up = np.asarray(up_pinch, dtype=np.float64) / np.linalg.norm(up_pinch)
    pts = np.asarray(points_pinch, dtype=np.float64)
    if len(pts) < 20:
        raise ValueError(f"only {len(pts)} points on the graspable part: a bigger box, a wider band")

    def horizontal(q: np.ndarray, c: np.ndarray) -> np.ndarray:
        r = q - c
        return r - np.outer(r @ up, up)

    start = np.median(pts, axis=0)
    if radius is None:
        fit = least_squares(
            lambda x: np.linalg.norm(horizontal(pts, x[:3]), axis=1) - x[3],
            [*start, 0.015],
            loss="soft_l1",
            f_scale=0.002,
        )
        center, rad = fit.x[:3], float(fit.x[3])
    else:
        fit = least_squares(
            lambda x: np.linalg.norm(horizontal(pts, x), axis=1) - radius,
            start,
            loss="soft_l1",
            f_scale=0.002,
        )
        center, rad = fit.x, float(radius)
    if rad <= 0:
        raise ValueError("the cylinder fit came out with no radius")
    center = center - (center @ up - table) * up  # the axis's foot, on the table
    rms = float(np.sqrt(np.mean(np.minimum(np.abs(fit.fun), 0.01) ** 2)))
    closing, approach, middle = _jaw_axes(model, arm, 2 * rad)
    # the closing plane through the fingers' middle meets the (vertical) cylinder in chords: the widest, and where
    rel = horizontal(middle[None], center)[0]  # the axis relative to the middle, horizontally (pinch frame)
    a_h, c_h = approach - (approach @ up) * up, closing - (closing @ up) * up
    # along the approach the cylinder's horizontal section is a circle; the plane's line at depth t along the approach
    # is middle + s * closing + t * approach: its horizontal distance from the axis is |rel_h + s c_h + t a_h|
    best = None
    for t in np.linspace(-0.06, 0.10, 641):
        base = rel + t * a_h  # the axis-to-line offset, horizontally, at s = 0 ... solve |base + s c_h| = rad
        aa, bb, cc = c_h @ c_h, 2 * base @ c_h, base @ base - rad**2
        disc = bb * bb - 4 * aa * cc
        if disc <= 0:
            continue
        s1, s2 = (-bb - np.sqrt(disc)) / (2 * aa), (-bb + np.sqrt(disc)) / (2 * aa)
        if best is None or s2 - s1 > best[1]:
            best = (float(t), float(s2 - s1), float(s1), float(s2))
    if best is None:
        raise ValueError(
            "the jaws' closing plane does not meet the cylinder: the object is not ahead of the jaws"
        )
    ahead, chord, s1, s2 = best
    # t and s are measured from the fingers' middle at the fixed tip; the fixed-side contact is s1 (the smaller s), and
    # positive s is toward the moving jaw: so the shift's sign says which side of the fixed finger the contact is on
    advance = ahead + into
    shift = s1 - gap
    move = advance * approach + shift * closing
    return Patches(rad, rms, chord, ahead, s1, s2, advance, shift, move, center, approach, closing, middle)


def hold_views(
    client: "FelixClient", model: XleRobotModel, arm: str, rolls_deg: Sequence[float] = (-15.0, 15.0)
) -> HoldViews:
    """A wrist frame now, at each wrist roll of `rolls_deg` about where it is, and back (the wrist only)."""
    from .wrist_scan import sweep

    views, labels = sweep(client, model, arm, rolls_deg, ())
    return HoldViews(tuple(views), tuple(labels), len(views) - 1)


def _model() -> "DepthAnything":
    from .depth_anything import DepthAnything

    if not _MODEL:
        _MODEL.append(DepthAnything.from_hub())
    return _MODEL[0]


def contact_patches(
    model: XleRobotModel,
    views: HoldViews | WristView,
    box: Sequence[float],
    band: Sequence[float],
    radius: float | None = None,
    gap: float = GAP_M,
    into: float = INTO_M,
    table_z: float | None = None,
    exclude_wood: bool = True,
) -> ContactPatches:
    """The contact patches at a hold (the module docstring); nothing moves.

    `box`: (u0, v0, u1, v1) around the graspable part in the current frame; `band`: (low, high) its heights above the
    table (m). `table_z`: the table's height (root frame, the head's depth) to set the scale of several (posed) views;
    a single view always takes it from the fixed finger (the table scale of an unposed frame was off by 1 cm in bowl8).
    `exclude_wood`: drop the object's pixels that look like the (wooden) table, inside the box.
    """
    hold = views if isinstance(views, HoldViews) else HoldViews((views,), ("now",), 0)
    current = hold.views[hold.current]
    arm, camera = current.arm, current.camera
    k, d = camera.matrix(), np.asarray(camera.coeffs)
    images = [cv2.undistort(v.color, k, d) for v in hold.views]
    poses = [v.root_from_camera for v in hold.views]
    out = _model().depth(images, poses if len(images) > 1 else None, k)
    # the current frame: its depth at DA3's resolution, the pixel scale to ours
    sx, sy = out.depth.shape[2] / camera.width, out.depth.shape[1] / camera.height
    cam_now = current.root_from_camera
    clouds = []
    for i in range(len(out.depth)):
        z = out.depth[i]
        gy, gx = np.gradient(z)
        edge = cv2.dilate(
            (np.hypot(gx, gy) / np.maximum(z, 1e-6) > EDGE_JUMP).astype(np.uint8), np.ones((3, 3))
        )
        bad = edge.astype(bool) | (out.conf[i] < np.percentile(out.conf[i], LOW_CONFIDENCE))
        if len(images) > 1:
            pts = out.points(i)
        else:  # one frame: DA3's own camera frame; put it on ours
            kk = out.intrinsics[i]
            v, u = np.mgrid[0 : z.shape[0], 0 : z.shape[1]]
            pc = np.dstack([(u - kk[0, 2]) / kk[0, 0] * z, (v - kk[1, 2]) / kk[1, 1] * z, z])
            pts = pc @ cam_now[:3, :3].T + cam_now[:3, 3]
        pts[bad] = np.nan
        clouds.append(pts)
    all_pts = np.vstack([c.reshape(-1, 3) for c in clouds])
    all_pts = all_pts[np.isfinite(all_pts).all(axis=1)]
    to_cam = np.linalg.inv(cam_now)
    pc = all_pts @ to_cam[:3, :3].T + to_cam[:3, 3]
    uv = camera.project(pc)
    u0, v0, u1, v1 = box
    inbox = (uv[:, 0] > u0) & (uv[:, 0] < u1) & (uv[:, 1] > v0) & (uv[:, 1] < v1)
    ring = (
        (uv[:, 0] > u0 - RING_PX)
        & (uv[:, 0] < u1 + RING_PX)
        & (uv[:, 1] > v0 - RING_PX)
        & (uv[:, 1] < v1 + RING_PX)
    ) & ~inbox
    if ring.sum() < 50:
        raise ValueError("too few table points around the box: a wider view of the table around the object")
    origin = cam_now[:3, 3]
    # the scale: about the current camera (its rays fixed): the table's height, else the fixed finger's known depth
    if table_z is not None and len(images) > 1:
        seen = origin[2] - float(np.median(all_pts[ring][:, 2]))
        scale, source = (origin[2] - table_z) / seen, f"the table's height ({100 * table_z:.1f} cm)"
    else:
        face = model.wrist_face(arm)
        finger = np.array([face + [0.0, 0.0, -b] for b in np.linspace(0.003, 0.03, FINGER_POINTS)])
        pinch_to_cam = np.linalg.inv(current.pinch_from_camera)
        fc = finger @ pinch_to_cam[:3, :3].T + pinch_to_cam[:3, 3]
        fuv = camera.project(fc)
        ratios = []
        for (fu, fv), true_z in zip(fuv, fc[:, 2], strict=True):
            iu, iv = int(fu * sx), int(fv * sy)
            patch = out.depth[hold.current][max(iv - 1, 0) : iv + 2, max(iu - 1, 0) : iu + 2]
            if patch.size and np.isfinite(patch).all():
                ratios.append(true_z / float(np.median(patch)))
        if len(ratios) < FINGER_POINTS // 2:
            raise ValueError("the fixed finger is not in the picture: give the table's height (table_z)")
        scale, source = float(np.median(ratios)), f"the fixed finger ({len(ratios)} points)"
    all_pts = origin + scale * (all_pts - origin)
    to_pinch = np.linalg.inv(current.root_from_pinch)
    pp = all_pts @ to_pinch[:3, :3].T + to_pinch[:3, 3]
    up = to_pinch[:3, :3] @ np.array([0.0, 0.0, 1.0])
    height = pp @ up
    table = float(np.median(height[ring]))
    sel = inbox & (height > table + band[0]) & (height < table + band[1])
    if exclude_wood:
        hsv = cv2.cvtColor(current.color, cv2.COLOR_RGB2HSV).astype(int)
        wood = (hsv[..., 0] < 25) & (hsv[..., 1] > 70) & (hsv[..., 2] > 60)
        wood = cv2.morphologyEx(wood.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(
            bool
        )
        iu = np.clip(np.nan_to_num(uv[:, 0]).astype(int), 0, wood.shape[1] - 1)
        iv = np.clip(np.nan_to_num(uv[:, 1]).astype(int), 0, wood.shape[0] - 1)
        sel &= ~wood[iv, iu]
    patches = patches_from_points(model, arm, pp[sel], up, table, radius, gap, into)
    move_root = current.root_from_pinch[:3, :3] @ patches.move_pinch
    image = _draw(current, patches, pp[sel], box, band, up, table)
    return ContactPatches(current, patches, int(sel.sum()), float(scale), source, move_root, image)


def _draw(
    view: WristView,
    p: Patches,
    pts: np.ndarray,
    box: Sequence[float],
    band: Sequence[float],
    up: np.ndarray,
    table: float,
) -> np.ndarray:
    """The current frame with the evidence drawn on it.

    The box, the points used, the cylinder (rings at the band's ends), the contacts, the closing line (green).
    """
    img = view.color.copy()
    u0, v0, u1, v1 = (int(c) for c in box)
    cv2.rectangle(img, (u0, v0), (u1, v1), (255, 255, 255), 1)
    for q in view.project_pinch(pts[:: max(len(pts) // 1500, 1)]):
        if np.isfinite(q).all():
            cv2.circle(img, (int(q[0]), int(q[1])), 1, (0, 255, 255), -1)
    e1 = np.cross(up, p.approach_pinch)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    ang = np.linspace(0, 2 * np.pi, 72)
    for h in band:
        ring = p.axis_pinch + h * up + p.radius * (np.outer(np.cos(ang), e1) + np.outer(np.sin(ang), e2))
        q = view.project_pinch(ring)
        good = np.isfinite(q).all(axis=1)
        if good.sum() > 2:
            cv2.polylines(img, [q[good].round().astype(np.int32)], True, (255, 140, 0), 2, cv2.LINE_AA)
    middle = p.middle_pinch
    for s, color in ((p.fixed_contact, (0, 255, 255)), (p.moving_contact, (255, 0, 255))):
        q = view.project_pinch((middle + s * p.closing_pinch + p.ahead * p.approach_pinch)[None])[0]
        if np.isfinite(q).all():
            cv2.circle(img, (int(q[0]), int(q[1])), 7, color, 2, cv2.LINE_AA)
    line = view.project_pinch(
        np.array([middle + s * p.closing_pinch for s in np.linspace(0, 2 * p.radius + 0.01, 12)])
    )
    good = np.isfinite(line).all(axis=1)
    if good.sum() > 1:
        cv2.polylines(img, [line[good].round().astype(np.int32)], False, (60, 255, 60), 2, cv2.LINE_AA)
    return img
