"""A color + depth view of the world that you can ask geometric questions about.

`Scene` holds one RealSense capture in the color camera's pixel grid (depth already aligned to color, in
meters, NaN where there is no depth) and answers, for many points at once:

- `pixel_to_xyz`: which 3D point does this pixel show?
- `xyz_to_pixel`: where does this 3D point appear in the image?
- `segment_check`: is the straight path between two 3D points free, blocked, or not known to be free?

Everything takes and returns numpy arrays and works on batches. Coordinates are in the color camera's frame:
x to the right, y down, z forward (along the optical axis), meters. Pixels are (u, v) = (column, row).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import cv2
import numpy as np

from lerobot.felix_jr.depth import align_depth_to_color, load_calibration

FREE, UNKNOWN, BLOCKED = 0, 1, 2  # state of a sample point / a whole segment (worst state wins)


def read_image(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """`cv2.imread`, raising instead of returning None when the file is missing or unreadable."""
    image = cv2.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(f"cannot read image {path}")
    return image


class SegmentCheck(NamedTuple):
    """Result of `Scene.segment_check` for N segments sampled at S points each."""

    status: np.ndarray  # (N,) int: FREE, UNKNOWN or BLOCKED (the worst sample on the segment)
    # (N,) float: position along the segment (0 = start, 1 = end) of the first sample that is not FREE;
    # NaN if the whole segment is FREE
    t_first: np.ndarray
    point_first: np.ndarray  # (N, 3) float: that sample's xyz; NaN if the segment is FREE
    sample_xyz: np.ndarray  # (N, S, 3) float: the sample points
    sample_state: np.ndarray  # (N, S) int: FREE, UNKNOWN or BLOCKED for each sample


@dataclass(frozen=True)
class Scene:
    """One capture in the camera frame: color, depth aligned to color, and the color intrinsics."""

    color: np.ndarray  # (H, W, 3) uint8, RGB
    depth_m: np.ndarray  # (H, W) float32, meters, in the color pixel grid; NaN = no depth
    fx: float
    fy: float
    ppx: float
    ppy: float

    @property
    def height(self) -> int:
        """Image height in pixels."""
        return self.depth_m.shape[0]

    @property
    def width(self) -> int:
        """Image width in pixels."""
        return self.depth_m.shape[1]

    # ---- construction -------------------------------------------------------------------------------------

    @classmethod
    def from_capture(cls, color_rgb: np.ndarray, raw_depth: np.ndarray, calib: dict[str, Any]) -> "Scene":
        """Build a scene from a LeRobot RealSense capture.

        Inputs: RGB `uint8` color, raw `uint16` depth (as read from the camera, not aligned) and the calibration
        from `diagnostics/realsense_check.py`.
        """
        aligned = align_depth_to_color(raw_depth, calib)
        depth_m = np.where(aligned > 0, aligned * calib["depth_scale_m_per_unit"], np.nan).astype(np.float32)
        ci = calib["color_intrinsics"]
        return cls(color_rgb, depth_m, ci["fx"], ci["fy"], ci["ppx"], ci["ppy"])

    @classmethod
    def from_saved(cls, folder: str | Path) -> "Scene":
        """Load color.png, depth_raw.png and calibration.json saved by `realsense_check --save-dir`."""
        folder = Path(folder)
        color = cv2.cvtColor(read_image(folder / "color.png"), cv2.COLOR_BGR2RGB)
        raw = read_image(folder / "depth_raw.png", cv2.IMREAD_UNCHANGED)
        return cls.from_capture(color, raw, load_calibration(folder / "calibration.json"))

    # ---- pixel <-> 3D -------------------------------------------------------------------------------------

    def pixel_to_xyz(
        self,
        uv: np.ndarray,
        window: int = 5,
        min_valid: int = 3,
        max_spread_m: float = 0.10,
        standoff: float = 0.0,
    ) -> np.ndarray:
        """3D points (meters, camera frame) that the pixels `uv` show. Shape (..., 2) -> (..., 3).

        The depth is the median of the valid depths in a `window` x `window` patch around each pixel (so a
        single bad pixel does not matter). A point comes back as NaN when it is not trustworthy:
        - the pixel is outside the image,
        - fewer than `min_valid` pixels in the patch have depth,
        - the patch straddles a depth edge: its 10th-to-90th percentile depth spread exceeds `max_spread_m`,
          so the median would be a blend of a near and a far surface.
        The x and y of the point use the exact (possibly fractional) pixel position you passed.

        `standoff` (meters) moves each point toward the camera along its viewing ray. A point that sits on a
        surface counts as touching it in `segment_check`, so use a standoff of a few centimeters to get a
        start or goal point in the free space in front of the surface you clicked.
        """
        uv = np.asarray(uv, dtype=np.float64)
        if uv.shape[-1] != 2:
            raise ValueError(f"uv must have a last dimension of 2, got shape {uv.shape}")
        flat = uv.reshape(-1, 2)
        n = len(flat)
        finite = np.isfinite(flat).all(axis=1)  # casting NaN to int is undefined (it gives 0 on arm64)
        u0 = np.rint(np.where(finite, flat[:, 0], 0)).astype(int)
        v0 = np.rint(np.where(finite, flat[:, 1], 0)).astype(int)
        centre_in = finite & (u0 >= 0) & (u0 < self.width) & (v0 >= 0) & (v0 < self.height)

        half = max(int(window), 1) // 2
        offs = np.arange(-half, half + 1)
        vv = v0[:, None, None] + offs[None, :, None]
        uu = u0[:, None, None] + offs[None, None, :]
        inside = (vv >= 0) & (vv < self.height) & (uu >= 0) & (uu < self.width)
        samples = self.depth_m[np.clip(vv, 0, self.height - 1), np.clip(uu, 0, self.width - 1)]
        patch = np.where(inside, samples, np.nan)
        patch = patch.reshape(n, -1)

        count = np.isfinite(patch).sum(axis=1)
        enough = centre_in & (count >= max(min_valid, 1))
        z = np.full(n, np.nan)
        if enough.any():
            sub = patch[enough]
            z[enough] = np.nanmedian(sub, axis=1)
            spread = np.nanpercentile(sub, 90, axis=1) - np.nanpercentile(sub, 10, axis=1)
            z[np.flatnonzero(enough)[spread > max_spread_m]] = np.nan

        z = z - standoff  # along the viewing ray: x and y scale with z, so the point stays on its ray
        x = (flat[:, 0] - self.ppx) / self.fx * z
        y = (flat[:, 1] - self.ppy) / self.fy * z
        return np.stack([x, y, z], axis=1).reshape(uv.shape[:-1] + (3,))

    def xyz_to_pixel(self, xyz: np.ndarray) -> np.ndarray:
        """Pixels (u, v) where the 3D points appear. Shape (..., 3) -> (..., 2).

        Points at or behind the camera (z <= 0) come back as NaN. Points in front but outside the image give
        coordinates outside [0, width) x [0, height); test with `in_image`.
        """
        xyz = np.asarray(xyz, dtype=np.float64)
        if xyz.shape[-1] != 3:
            raise ValueError(f"xyz must have a last dimension of 3, got shape {xyz.shape}")
        z = xyz[..., 2]
        safe = np.where(z > 0, z, np.nan)
        u = xyz[..., 0] / safe * self.fx + self.ppx
        v = xyz[..., 1] / safe * self.fy + self.ppy
        return np.stack([u, v], axis=-1)

    def in_image(self, uv: np.ndarray) -> np.ndarray:
        """True where pixels `uv` (..., 2) fall inside the image (NaN is outside)."""
        uv = np.asarray(uv, dtype=np.float64)
        u, v = uv[..., 0], uv[..., 1]
        with np.errstate(invalid="ignore"):
            return (u >= 0) & (u <= self.width - 1) & (v >= 0) & (v <= self.height - 1)

    # ---- path checking ------------------------------------------------------------------------------------

    def segment_check(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        radius: float = 0.02,
        tolerance: float = 0.01,
        step: float = 0.01,
        max_disc_px: int = 20,
        chunk: int = 32,
    ) -> SegmentCheck:
        """Is the straight path from `p1` to `p2` free, using only what the camera saw?

        `p1`, `p2`: (N, 3) (or (3,)) points in the camera frame. The segment is sampled every `step` meters
        and a ball of `radius` meters is imagined at each sample. Each sample gets a state:

        - BLOCKED: some observed surface lies within `radius + tolerance` of the sample's depth, within a
          `radius`-sized disc around its pixel. The ball touches something.
        - UNKNOWN: it cannot be said to be free. The sample is outside the image or behind the camera, its
          pixel has no depth (a hole or a shadow), or an observed surface is in front of it (it is hidden
          behind that surface, and depth cameras do not see behind surfaces).
        - FREE: the observed surfaces around it are all farther away than the ball, so the ball sits in
          visible empty space.

        `tolerance` (meters) allows for depth noise. Distances to surfaces are measured along the viewing
        direction (a depth-image test), so this is an approximation, good to about the tolerance.
        A segment's status is the worst of its samples (BLOCKED > UNKNOWN > FREE); `t_first` and `point_first`
        say where along it the first non-free sample is. `max_disc_px` caps the neighborhood size used for
        large radii or close points.
        """
        p1 = np.atleast_2d(np.asarray(p1, dtype=np.float64))
        p2 = np.atleast_2d(np.asarray(p2, dtype=np.float64))
        if p1.shape != p2.shape or p1.shape[-1] != 3:
            raise ValueError(f"p1 and p2 must both have shape (N, 3); got {p1.shape} and {p2.shape}")
        n = len(p1)
        lengths = np.linalg.norm(p2 - p1, axis=1)
        lengths = lengths[np.isfinite(lengths)]  # a segment with a NaN end has no length (checks UNKNOWN)
        longest = float(lengths.max()) if lengths.size else 0.0
        samples = max(2, int(np.ceil(longest / step)) + 1)
        t = np.linspace(0.0, 1.0, samples)

        sample_xyz = p1[:, None, :] + (p2 - p1)[:, None, :] * t[None, :, None]
        sample_state: np.ndarray = np.empty((n, samples), dtype=np.int8)
        for start in range(0, n, chunk):
            part = slice(start, start + chunk)
            sample_state[part] = self._sample_states(sample_xyz[part], radius, tolerance, max_disc_px)

        status = sample_state.max(axis=1) if n else np.zeros(0, dtype=np.int8)
        not_free = sample_state != FREE
        any_bad = not_free.any(axis=1) if n else np.zeros(0, dtype=bool)
        first = np.argmax(not_free, axis=1) if n else np.zeros(0, dtype=int)
        t_first = np.where(any_bad, t[first], np.nan)
        point_first = np.where(any_bad[:, None], sample_xyz[np.arange(n), first], np.nan)
        return SegmentCheck(status.astype(np.int8), t_first, point_first, sample_xyz, sample_state)

    def _sample_states(
        self, pts: np.ndarray, radius: float, tolerance: float, max_disc_px: int
    ) -> np.ndarray:
        """FREE / UNKNOWN / BLOCKED for sample points `pts` of shape (n, S, 3)."""
        height, width = self.height, self.width
        z = pts[..., 2]
        with np.errstate(invalid="ignore"):
            front = (z > 1e-3) & np.isfinite(pts).all(axis=-1)  # NaN points count as not in view
        zs = np.where(front, z, 1.0)
        px = np.where(front, pts[..., 0], 0.0)
        py = np.where(front, pts[..., 1], 0.0)
        ui = np.rint(px / zs * self.fx + self.ppx).astype(int)
        vi = np.rint(py / zs * self.fy + self.ppy).astype(int)
        centre_in = front & (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
        band = radius + tolerance

        # neighbourhood: a disc whose radius in pixels is the ball's apparent radius at that depth
        r_px = np.minimum(np.ceil(self.fx * radius / zs).astype(int), max_disc_px)
        reach = int(r_px.max()) if r_px.size else 0
        offs = np.arange(-reach, reach + 1)
        dy, dx = np.meshgrid(offs, offs, indexing="ij")
        dy, dx = dy.reshape(-1), dx.reshape(-1)
        vv, uu = vi[..., None] + dy, ui[..., None] + dx
        inside = (vv >= 0) & (vv < height) & (uu >= 0) & (uu < width)
        depth = np.where(inside, self.depth_m[np.clip(vv, 0, height - 1), np.clip(uu, 0, width - 1)], np.nan)
        in_disc = (dx**2 + dy**2) <= (r_px**2)[..., None]
        with np.errstate(invalid="ignore"):
            touching = in_disc & (np.abs(depth - zs[..., None]) <= band)
        blocked = centre_in & touching.any(axis=-1)

        centre_depth = self.depth_m[np.clip(vi, 0, height - 1), np.clip(ui, 0, width - 1)]
        centre = np.where(centre_in, centre_depth, np.nan)
        with np.errstate(invalid="ignore"):
            hidden = ~np.isfinite(centre) | (centre < zs - band)

        state = np.full(z.shape, FREE, dtype=np.int8)
        state[hidden | ~centre_in] = UNKNOWN
        state[blocked] = BLOCKED
        return state
