"""A wrist scan: frames from several wrist poses, their real poses from structure from motion, and 3D from them.

`docs/object_grasping.md` section 6.1-6.2. The wrist camera has no depth; a sweep of the wrist (roll and flex only:
the joints that carry the least load, the arm above them still) gives frames from several places, and
`sfm.refine` measures how the camera really moved between them. A `WristScan` then answers for the model what a
head capture's depth does:

- **Frames to look at:** `render` draws a contact sheet of the frames (or one, full size) with objects drawn in
  each: an outline that sits right in every frame is the check.
- **The scale:** `scale_from_table` sets it from the table's height (the head's depth).

(The sparse points and the silhouette fit that were here are gone: DA3's dense depth replaced them,
`docs/object_grasping.md` section 7.)

**Frames.** Answers come in the pinch frame of the frame the arm is at now (`current`: after a sweep, back at its
start): where things are relative to the jaws, which is what the next move needs. And in the root frame through that
pinch frame as the model has it at the joints now (`to_root`), so a plan made there and run from where the arm is
cancels the model's error. **The scale** of everything is the model's (the images fix the motion up to one factor):
a few percent of the distances, said in the descriptions.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import cv2
import numpy as np

from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel

from .objects import Object
from .primitives import Cylinder
from .sfm import PRIOR_DEG, PRIOR_MM, SfmResult, refine
from .wrist import WristView, move_closely, wrist_view

if TYPE_CHECKING:
    from lerobot.felix_jr.server.client import FelixClient

SCALE_NOTE = "the scale is the model's: a few % of the distance"
SWEEP_ROLLS_DEG = (-30.0, -15.0, 15.0, 30.0)
SWEEP_FLEXES_DEG = (-10.0, 10.0)
MAX_ROLL_DEG = 130.0  # the wrist camera's cable (as `calibration.views`)
SWEEP_VEL = 15.0


def _cm(value: float) -> str:
    return f"{100 * value:.1f}"


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


# ---- pictures -----------------------------------------------------------------------------------------------------


def render(
    scan: WristScan,
    frames: Sequence[int] | None = None,
    objects: Sequence[Object] = (),
    tile_width: int = 320,
) -> np.ndarray:
    """The scan's frames with `objects` drawn in each: one frame full size, several as a contact sheet."""
    chosen = list(range(len(scan.views))) if frames is None else [int(f) % len(scan.views) for f in frames]
    to_scan = np.linalg.inv(scan.views[scan.current].root_from_pinch) if objects else None
    tiles = []
    for f in chosen:
        image = scan.views[f].color.copy()
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
