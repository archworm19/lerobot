"""The wrist camera's real motion between frames, from the images: structure from motion with the model's poses as a prior.

`docs/object_grasping.md` section 6.1. bowl6 triangulated over the arm's moves with the camera's motion taken from the
joints through the model, and the arm does not move as modeled (backlash, link flex, a sag that changes with reach):
10-20% depth errors on 1.5-3 cm moves. Here the motion is measured: features on the scene (the table's grain) are
matched across a few wrist frames, and the camera poses are refined to them.

This is not structure from motion from scratch (COLMAP, GLOMAP): there are few frames (~8), the lens is calibrated,
and every pose is known from the model to a few millimeters. So it is a small bundle adjustment from a good start:

1. **Features and matches** for every pair of frames: SIFT with a ratio test (`method="light"`: OpenCV only) or DISK
   and LightGlue (`method="torch"`: kornia, learned, better across large viewpoint changes and on weak texture).
2. **Outliers** dropped per pair by an essential matrix (RANSAC on the lens-corrected rays).
3. **Tracks** (a scene point seen in several frames) joined over the pairs, and triangulated from a start: the
   model's poses, and the images' own turns (`_from_images`); the start that ends with the lower error is kept.
4. **The adjustment** (`_bundle_adjust`): a correction of each frame's pose (in the camera's own frame) and the
   points, fitted to the observations' pixels (Huber), with a prior pulling each correction toward none (`prior_mm`,
   `prior_deg`): the prior fixes what the images cannot (where the whole set is, and its scale), the images fix the
   rest. Levenberg-Marquardt with analytic derivatives, the points eliminated per step (Schur complement): it
   converges in tens of steps (SciPy's finite-difference `least_squares` ran out of evaluations near its start on the
   robot's frames, hil3, and kept the model's 3.6 deg errors).
5. **The table:** the dominant plane through the points (RANSAC), and how high the jaws are over it.

**One scale is the model's.** Images fix a moving camera's motion and the scene up to one scale factor; here it comes
from the prior (the model's translations, averaged over the frames). In the unit test (3 mm, 1 deg of motion error
per frame) the directions of the moves came out within 1 deg (7 deg before) and the turns within 0.25 deg (0.5), and
every move's length was off by the same ~3%, and so was the table's distance. 3% of 12 cm is ~4 mm, against bowl6's
10-20% from triangulating over the arm's moves; the object's known width or the table's height could set it too.

The result (`SfmResult`) says how far each frame's pose moved relative to the first (the arm's motion error), the
fit's error before and after, the matches, and the table.
"""

import contextlib
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import cv2
import numpy as np

from lerobot.felix_jr.calibration.xle_robot_model import WristCamera
from lerobot.utils.import_utils import require_package

if TYPE_CHECKING:
    from .wrist import WristView

METHODS = ("light", "torch")
RATIO = 0.75  # SIFT's ratio test
SIFT_FEATURES = 4000
DISK_FEATURES = 2048
RANSAC_PX = 1.0  # an essential-matrix inlier: within this many pixels (of the first frame's fx)
MIN_PAIR_INLIERS = 15  # a pair with fewer is not used
STATIC_PX = (
    2.0  # a match that moved less between two frames is on the gripper (rigid to the camera), not the scene
)
MIN_ANGLE_DEG = 0.5  # a track whose rays cross at less is not triangulated
MAX_TRACKS = 800  # the most-seen tracks are kept (the adjustment's size: 800 x 5 frames takes a few seconds)
HUBER_PX = 1.5
PRIOR_MM = 5.0
PRIOR_DEG = 1.0
PLANE_MM = (
    4.0  # a point within this of the plane is on it (the points from short baselines are a few mm noisy)
)
TABLE_TILT_DEG = 10.0  # only planes this close to level are candidates for the table


def _cm(value: float) -> str:
    return f"{100 * value:.1f}"


def _pose(rotvec: np.ndarray, translation: np.ndarray) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = cv2.Rodrigues(np.asarray(rotvec, dtype=np.float64))[0]
    pose[:3, 3] = translation
    return pose


def _normalized(camera: WristCamera, uv: np.ndarray) -> np.ndarray:
    """Lens-corrected normalized coordinates (N, 2) of pixels (N, 2)."""
    points = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)
    k, d = camera.matrix(), np.asarray(camera.coeffs)
    if camera.model == "fisheye":
        return cv2.fisheye.undistortPoints(points, k, d).reshape(-1, 2)
    return cv2.undistortPoints(points, k, d).reshape(-1, 2)


# ---- features and matches -------------------------------------------------------------------------------------


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image


def _light_features(images: Sequence[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
    sift = cv2.SIFT_create(nfeatures=SIFT_FEATURES)
    out = []
    for image in images:
        keypoints, descriptors = sift.detectAndCompute(_gray(image), None)
        uv = np.array([k.pt for k in keypoints], dtype=np.float64).reshape(-1, 2)
        out.append((uv, descriptors if descriptors is not None else np.zeros((0, 128), np.float32)))
    return out


def _light_match(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    if len(a[1]) < 2 or len(b[1]) < 2:
        return np.zeros((0, 2), dtype=int)
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(a[1], b[1], k=2)
    good = [m[0] for m in pairs if len(m) == 2 and m[0].distance < RATIO * m[1].distance]
    return np.array([(m.queryIdx, m.trainIdx) for m in good], dtype=int).reshape(-1, 2)


class _TorchMatcher:
    """DISK features and LightGlue matches (kornia), on the GPU when there is one."""

    def __init__(self) -> None:
        require_package("kornia", extra="felix_jr")
        # imported here, not at the module's top: kornia loads torch (~1 s), which the light path and the rest of
        # the harness do not need
        import torch
        from kornia.feature import DISK, LightGlueMatcher

        self.torch = torch
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.disk = DISK.from_pretrained("depth").to(self.device).eval()
        self.matcher = LightGlueMatcher("disk").to(self.device).eval()

    def features(
        self, images: Sequence[np.ndarray], turns_deg: Sequence[float], center: tuple[float, float]
    ) -> list[tuple[np.ndarray, Any]]:
        """DISK's keypoints (in each image's own pixels) and features, each image first turned back by `turns_deg`.

        About `center`, to the first frame's roll: DISK and LightGlue are not rotation invariant, and a wrist roll
        turns the picture (unturned, pairs 45 deg or more apart did not match at all).
        """
        torch = self.torch
        out = []
        with torch.inference_mode():
            for image, turn in zip(images, turns_deg, strict=True):
                rgb = image if image.ndim == 3 else np.repeat(image[..., None], 3, axis=2)
                rotation = cv2.getRotationMatrix2D(center, -float(turn), 1.0)
                turned = cv2.warpAffine(rgb, rotation, (rgb.shape[1], rgb.shape[0]))
                tensor = torch.from_numpy(turned).permute(2, 0, 1)[None].float().to(self.device) / 255.0
                found = self.disk(tensor, n=DISK_FEATURES, pad_if_not_divisible=True)[0]
                uv = found.keypoints.cpu().numpy().astype(np.float64)
                back = cv2.invertAffineTransform(rotation)
                out.append((uv @ back[:, :2].T + back[:, 2], found))
        return out

    def match(
        self, a: tuple[np.ndarray, Any], b: tuple[np.ndarray, Any], size: tuple[int, int]
    ) -> np.ndarray:
        torch = self.torch
        from kornia.feature import laf_from_center_scale_ori

        if len(a[0]) < 2 or len(b[0]) < 2:
            return np.zeros((0, 2), dtype=int)
        with torch.inference_mode():
            lafs = [
                laf_from_center_scale_ori(
                    f.keypoints[None], torch.ones(1, len(f.keypoints), 1, 1, device=self.device)
                )
                for f in (a[1], b[1])
            ]
            hw = torch.tensor(size[::-1], device=self.device)
            _, idxs = self.matcher(a[1].descriptors, b[1].descriptors, lafs[0], lafs[1], hw1=hw, hw2=hw)
        return idxs.cpu().numpy().astype(int).reshape(-1, 2)


MIN_START_MOVE_M = (
    0.005  # a frame whose model move from the reference is shorter keeps the model's pose to start
)


START_WITHIN_DEG = (
    10.0  # an image-derived turn is used only this close to the model's (else the model's pose)
)


def _from_images(
    model: np.ndarray,
    pair_matches: dict[tuple[int, int], np.ndarray],
    normalized: Sequence[np.ndarray],
    fx: float,
) -> np.ndarray:
    """The poses to start the adjustment from: each frame's turn and move from the first frame, from the images.

    The turn and the move's direction from the images, the move's length (the scale) the model's. Starting from the
    model's own poses (hil3, 2026-11-08: its relative turns up to 3.6 deg off) put the adjustment in a wrong minimum
    near them, though the matches agreed with the true poses to 1.3 px. The candidates: the essential matrix's four
    decompositions and, for a flat scene (where the essential matrix is degenerate: a table alone), the homography's;
    those whose turn is within `START_WITHIN_DEG` of the model's, and of those the one with the most points in front
    of both cameras. A frame with too few matches with the first, hardly moved from it, or no candidate near the
    model keeps the model's pose.
    """
    start = model.copy()
    for i in range(1, len(model)):
        pairs = pair_matches.get((0, i))
        model_move = np.linalg.inv(model[0]) @ model[i]  # camera 0 from camera i
        length = float(np.linalg.norm(model_move[:3, 3]))
        if pairs is None or len(pairs) < 2 * MIN_PAIR_INLIERS or length < MIN_START_MOVE_M:
            continue
        a, b = normalized[0][pairs[:, 0]], normalized[i][pairs[:, 1]]
        model_rotation = model_move[:3, :3].T  # camera i from camera 0
        candidates: list[tuple[np.ndarray, np.ndarray]] = []
        essential, _ = cv2.findEssentialMat(
            a, b, np.eye(3), method=cv2.RANSAC, prob=0.999, threshold=RANSAC_PX / fx
        )
        if essential is not None and essential.shape == (3, 3):
            r1, r2, t = cv2.decomposeEssentialMat(essential)
            candidates += [(r, sign * t.ravel()) for r in (r1, r2) for sign in (1.0, -1.0)]
        homography, _ = cv2.findHomography(a, b, cv2.RANSAC, RANSAC_PX / fx)
        if homography is not None:
            _, rotations, translations, _ = cv2.decomposeHomographyMat(homography, np.eye(3))
            candidates += [(r, t.ravel()) for r, t in zip(rotations, translations, strict=True)]
        best, best_front = None, -1
        for rotation, direction in candidates:
            off = np.linalg.norm(cv2.Rodrigues(rotation @ model_rotation.T)[0])
            if np.degrees(off) > START_WITHIN_DEG or np.linalg.norm(direction) < 1e-9:
                continue
            unit = direction / np.linalg.norm(direction)
            front = _in_front(a, b, rotation, unit)
            if front > best_front:
                best, best_front = (rotation, unit), front
        if best is None:
            continue
        # x_i = R x_0 + t: camera i from camera 0, its translation the model's length
        camera_i_from_0 = np.eye(4)
        camera_i_from_0[:3, :3] = best[0]
        camera_i_from_0[:3, 3] = best[1] * length
        start[i] = model[0] @ np.linalg.inv(camera_i_from_0)
    return start


def _in_front(a: np.ndarray, b: np.ndarray, rotation: np.ndarray, direction: np.ndarray) -> int:
    """How many matched rays (normalized, camera 0 and i) meet in front of both cameras for x_i = R x_0 + t."""
    p0 = np.hstack([np.eye(3), np.zeros((3, 1))])
    p1 = np.hstack([rotation, direction.reshape(3, 1)])
    points = cv2.triangulatePoints(p0, p1, a.T.astype(np.float64), b.T.astype(np.float64))
    points = points[:3] / np.where(np.abs(points[3]) > 1e-12, points[3], 1e-12)
    depth_0 = points[2]
    depth_1 = (rotation @ points + direction.reshape(3, 1))[2]
    return int(((depth_0 > 0) & (depth_1 > 0)).sum())


def _rolls_deg(model: np.ndarray) -> list[float]:
    """Each frame's turn about its optical axis relative to the first's, by the model (deg).

    How far its picture is turned against the first's.
    """
    out = []
    for pose in model:
        relative = model[0][:3, :3].T @ pose[:3, :3]  # frame i's axes in frame 0's camera frame
        out.append(float(np.degrees(np.arctan2(relative[1, 0], relative[0, 0]))))
    return out


def _verified(camera: WristCamera, uv_a: np.ndarray, uv_b: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """The pairs consistent with one essential matrix (RANSAC on the lens-corrected rays).

    Matches that stayed put are dropped first: the jaws are in every wrist frame, rigid to the camera, so their
    features match at the same pixel whatever the arm did.
    """
    if len(pairs):
        pairs = pairs[np.linalg.norm(uv_a[pairs[:, 0]] - uv_b[pairs[:, 1]], axis=1) >= STATIC_PX]
    if len(pairs) < MIN_PAIR_INLIERS:
        return pairs[:0]
    na, nb = _normalized(camera, uv_a[pairs[:, 0]]), _normalized(camera, uv_b[pairs[:, 1]])
    _, mask = cv2.findEssentialMat(
        na, nb, np.eye(3), method=cv2.RANSAC, prob=0.999, threshold=RANSAC_PX / camera.fx
    )
    if mask is None:
        return pairs[:0]
    kept = pairs[mask.ravel().astype(bool)]
    return kept if len(kept) >= MIN_PAIR_INLIERS else pairs[:0]


def _tracks(pair_matches: dict[tuple[int, int], np.ndarray], counts: Sequence[int]) -> list[dict[int, int]]:
    """Join the pairs' matches into tracks ({frame: keypoint}); a track with two keypoints in one frame is dropped."""
    offsets = np.concatenate([[0], np.cumsum(counts)])
    parent = np.arange(int(offsets[-1]))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return int(x)

    for (i, j), pairs in pair_matches.items():
        for a, b in pairs:
            ra, rb = find(int(offsets[i] + a)), find(int(offsets[j] + b))
            if ra != rb:
                parent[ra] = rb
    groups: dict[int, dict[int, int]] = {}
    bad: set[int] = set()
    frame_of = np.searchsorted(offsets, np.arange(int(offsets[-1])), side="right") - 1
    used = {int(offsets[i] + k) for (i, j), pairs in pair_matches.items() for k in pairs[:, 0]} | {
        int(offsets[j] + k) for (i, j), pairs in pair_matches.items() for k in pairs[:, 1]
    }
    for node in used:
        root, frame = find(node), int(frame_of[node])
        track = groups.setdefault(root, {})
        if frame in track:
            bad.add(root)
        track[frame] = int(node - offsets[frame])
    return [t for r, t in groups.items() if r not in bad and len(t) >= 2]


# ---- the adjustment -----------------------------------------------------------------------------------------------


def _triangulate(origins: np.ndarray, rays: np.ndarray) -> tuple[np.ndarray, float]:
    """The point closest to all the rays (least squares) and the largest angle between two of them (deg)."""
    a = np.zeros((3, 3))
    b = np.zeros(3)
    for o, d in zip(origins, rays, strict=True):
        p = np.eye(3) - np.outer(d, d)
        a += p
        b += p @ o
    point = np.linalg.lstsq(a, b, rcond=None)[0]
    cos = np.clip(rays @ rays.T, -1.0, 1.0)
    return point, float(np.degrees(np.arccos(cos.min())))


@dataclass(frozen=True, eq=False)
class SfmResult:
    """The frames' poses refined to the images (`refine`), the scene's points and the table."""

    sensed: ClassVar[bool] = True  # a measurement of the scene now (`Workspace.add`)
    method: str
    model_root_from_camera: np.ndarray  # (N, 4, 4) the model's, at the joints read
    root_from_camera: np.ndarray  # (N, 4, 4) refined
    points: np.ndarray  # (M, 3) root frame
    observations: int
    pair_inliers: dict[tuple[int, int], int]
    rms_before_px: float
    rms_after_px: float
    plane: (
        tuple[np.ndarray, float] | None
    )  # (unit normal, offset): normal @ x = offset, root frame; None: no table
    plane_points: int
    time_s: float
    camera_from_pinch: np.ndarray | None = None  # (4, 4) rigid on the gripper, to place the jaws
    started: str = "the model's poses"  # where the adjustment that fitted better started (`refine`)

    def relative_motion_error(self) -> tuple[np.ndarray, np.ndarray]:
        """For each frame: how far its motion from the first frame differs, refined against the model (mm, deg)."""
        mm, deg = [], []
        for model, refined in zip(self.model_root_from_camera, self.root_from_camera, strict=True):
            m = np.linalg.inv(self.model_root_from_camera[0]) @ model
            r = np.linalg.inv(self.root_from_camera[0]) @ refined
            d = np.linalg.inv(m) @ r
            mm.append(1000 * float(np.linalg.norm(d[:3, 3])))
            deg.append(float(np.degrees(np.linalg.norm(cv2.Rodrigues(d[:3, :3])[0]))))
        return np.array(mm), np.array(deg)

    def jaws_over_table_m(self, frame: int = -1) -> float | None:
        """How high the pinch point is over the table in `frame` (refined), or None."""
        if self.plane is None or self.camera_from_pinch is None:
            return None
        normal, offset = self.plane
        pinch = (self.root_from_camera[frame] @ self.camera_from_pinch)[:3, 3]
        return float(normal @ pinch - offset)

    def describe(self) -> str:
        """The fit, the arm's motion error it found, and the table."""
        mm, deg = self.relative_motion_error()
        used = sum(1 for v in self.pair_inliers.values() if v)
        rows = [
            f"sfm ({self.method}) over {len(self.root_from_camera)} frames: {len(self.points)} points, "
            f"{self.observations} observations, {used} of {len(self.pair_inliers)} pairs matched (inliers "
            f"{min(self.pair_inliers.values(), default=0)}-{max(self.pair_inliers.values(), default=0)}); "
            f"reprojection RMS {self.rms_before_px:.1f} px from the model's poses -> {self.rms_after_px:.2f} px "
            f"(started from {self.started}); "
            f"{self.time_s:.1f} s",
            "the camera's motion from the first frame, measured against the model's: "
            + ", ".join(f"{a:.1f} mm/{b:.2f} deg" for a, b in zip(mm[1:], deg[1:], strict=True))
            + " (the scale is the model's: a few % of the distances)",
        ]
        if self.plane is None:
            rows.append("no table plane among the points")
        else:
            normal, offset = self.plane
            height = self.jaws_over_table_m()
            rows.append(
                f"the table: a plane through {self.plane_points} points, at z {_cm(offset / normal[2])} cm (root "
                f"frame), tilted {np.degrees(np.arccos(min(abs(normal[2]), 1.0))):.1f} deg"
                + (
                    f"; the jaws' pinch point {_cm(height)} cm over it (last frame)"
                    if height is not None
                    else ""
                )
            )
        return "\n".join(rows)


def refine(
    images: Sequence[np.ndarray],
    camera: WristCamera,
    model_root_from_camera: Sequence[np.ndarray],
    method: str = "light",
    prior_mm: float = PRIOR_MM,
    prior_deg: float = PRIOR_DEG,
    camera_from_pinch: np.ndarray | None = None,
) -> SfmResult:
    """The frames' camera poses refined to the images, from the model's (module docstring). Nothing moves."""
    if method not in METHODS:
        raise ValueError(f"method is one of {METHODS}, not {method!r}")
    if len(images) < 2:
        raise ValueError("at least two frames")
    start = time.perf_counter()
    model = np.array([np.asarray(p, dtype=np.float64) for p in model_root_from_camera])
    n = len(images)
    size = (images[0].shape[1], images[0].shape[0])
    if method == "light":
        feats = _light_features(images)

        def match(i: int, j: int) -> np.ndarray:
            return _light_match(feats[i], feats[j])
    else:
        torch_matcher = _TorchMatcher()
        tfeats = torch_matcher.features(images, _rolls_deg(model), (camera.ppx, camera.ppy))
        feats = [(uv, None) for uv, _ in tfeats]

        def match(i: int, j: int) -> np.ndarray:
            return torch_matcher.match(tfeats[i], tfeats[j], size)

    pair_matches: dict[tuple[int, int], np.ndarray] = {}
    for i in range(n):
        for j in range(i + 1, n):
            pair_matches[(i, j)] = _verified(camera, feats[i][0], feats[j][0], match(i, j))
    tracks = _tracks(pair_matches, [len(f[0]) for f in feats])
    normalized = [_normalized(camera, f[0]) if len(f[0]) else np.zeros((0, 2)) for f in feats]

    def adjust(start_poses: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, float, float]:
        """The adjustment from `start_poses`: refined poses, points, observations, RMS before and after (px)."""
        # triangulate each track from the start poses; keep the well-crossed ones, the most-seen first
        points, kept = [], []
        for track in sorted(tracks, key=len, reverse=True):
            frames = sorted(track)
            rays = []
            for f in frames:
                x, y = normalized[f][track[f]]
                ray = start_poses[f][:3, :3] @ np.array([x, y, 1.0])
                rays.append(ray / np.linalg.norm(ray))
            point, angle = _triangulate(start_poses[frames][:, :3, 3], np.array(rays))
            in_front = all(
                ((point - start_poses[f][:3, 3]) @ start_poses[f][:3, :3])[2] > 0.02 for f in frames
            )
            if angle >= MIN_ANGLE_DEG and in_front:
                points.append(point)
                kept.append(track)
            if len(kept) >= MAX_TRACKS:
                break
        if len(kept) < 10:
            raise ValueError(
                f"too few scene points matched across the frames ({len(kept)}): more texture or overlap"
            )
        obs_frame = np.array([f for t in kept for f in sorted(t)])
        obs_point = np.array([k for k, t in enumerate(kept) for _ in t])
        obs_xy = np.array([normalized[f][t[f]] for t in kept for f in sorted(t)])
        fx = camera.fx
        rms_start = _rms(start_poses, np.array(points), obs_frame, obs_point, obs_xy, fx)

        refined, pts = _bundle_adjust(
            start_poses,
            np.array(points),
            obs_frame,
            obs_point,
            obs_xy,
            fx,
            prior_mm / 1000,
            np.radians(prior_deg),
        )
        rms_after = _rms(refined, pts, obs_frame, obs_point, obs_xy, fx)
        return refined, pts, len(obs_frame), rms_start, rms_after

    # from the model's poses and from the images' (`_from_images`): the one the images fit better
    runs = [("the model's poses", adjust(model))]
    image_start = _from_images(model, pair_matches, normalized, camera.fx)
    if not np.allclose(image_start, model):
        with contextlib.suppress(ValueError):  # too few points from that start: the model's run stands
            runs.append(("the images' turns", adjust(image_start)))
    started, (refined, pts, observations, _, rms_after) = min(runs, key=lambda run: run[1][4])
    rms_model = runs[0][1][3]
    plane, on_plane = _table(pts)
    return SfmResult(
        method,
        model,
        refined,
        pts,
        observations,
        {k: len(v) for k, v in pair_matches.items()},
        rms_model,
        rms_after,
        plane,
        on_plane,
        time.perf_counter() - start,
        camera_from_pinch,
        started,
    )


def _rms(
    poses: np.ndarray,
    points: np.ndarray,
    obs_frame: np.ndarray,
    obs_point: np.ndarray,
    obs_xy: np.ndarray,
    fx: float,
) -> float:
    """The observations' reprojection RMS (px) with `poses` (root_from_camera) and `points`, outliers capped."""
    inv = np.array([np.linalg.inv(p) for p in poses])[obs_frame]
    p = np.einsum("nij,nj->ni", inv[:, :3, :3], points[obs_point]) + inv[:, :3, 3]
    z = np.where(p[:, 2] > 1e-6, p[:, 2], 1e-6)
    err = (p[:, :2] / z[:, None] - obs_xy) * fx
    return float(np.sqrt(np.mean(np.minimum(np.abs(err), 4 * HUBER_PX) ** 2)))


BA_ITERATIONS = 60
BA_TOLERANCE = 1e-7  # stop when the cost falls by less than this fraction


def _skew(v: np.ndarray) -> np.ndarray:
    """(N, 3, 3) cross-product matrices of (N, 3) vectors: skew(v) @ w = v x w."""
    out = np.zeros((len(v), 3, 3))
    out[:, 0, 1], out[:, 0, 2], out[:, 1, 2] = -v[:, 2], v[:, 1], -v[:, 0]
    out[:, 1, 0], out[:, 2, 0], out[:, 2, 1] = v[:, 2], -v[:, 1], v[:, 0]
    return out


def _bundle_adjust(
    start: np.ndarray,
    points: np.ndarray,
    obs_frame: np.ndarray,
    obs_point: np.ndarray,
    obs_xy: np.ndarray,
    fx: float,
    prior_m: float,
    prior_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """The cameras (root_from_camera) and points that best explain the observations: Levenberg-Marquardt.

    The points eliminated per step (Schur complement), analytic derivatives, Huber weights, a prior on each camera's
    correction from `start` (in its own frame).

    SciPy's `least_squares` with finite differences used up its evaluations near its start on hil3 (1000, cost still
    falling; from the true poses it drifted to 0.97 px): this converges in tens of steps.
    """
    n, m = len(start), len(points)
    poses, points = start.copy(), points.copy()
    obs_count = len(obs_frame)
    # the pairs of observations of one point (the Schur complement's off-diagonal camera blocks), fixed
    by_point: list[list[int]] = [[] for _ in range(m)]
    for k, j in enumerate(obs_point):
        by_point[j].append(k)
    pairs = np.array([(a, b) for ks in by_point for a in ks for b in ks], dtype=int).reshape(-1, 2)

    def correction(i: int) -> np.ndarray:
        """Camera i's correction from its start, (rotvec, translation) in its start frame."""
        d = np.linalg.inv(start[i]) @ poses[i]
        return np.concatenate([cv2.Rodrigues(d[:3, :3])[0].ravel(), d[:3, 3]])

    def evaluate(poses_: np.ndarray, points_: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        inv = np.array([np.linalg.inv(p) for p in poses_])[obs_frame]
        q = np.einsum("nij,nj->ni", inv[:, :3, :3], points_[obs_point]) + inv[:, :3, 3]
        z = np.where(q[:, 2] > 1e-6, q[:, 2], 1e-6)
        r = (q[:, :2] / z[:, None] - obs_xy) * fx
        norm = np.linalg.norm(r, axis=1)
        w = np.where(norm <= HUBER_PX, 1.0, HUBER_PX / np.maximum(norm, 1e-12))
        huber = np.where(norm <= HUBER_PX, 0.5 * norm**2, HUBER_PX * norm - 0.5 * HUBER_PX**2)
        prior = 0.0
        for i in range(n):
            c = np.linalg.inv(start[i]) @ poses_[i]
            rot = cv2.Rodrigues(c[:3, :3])[0].ravel() / prior_rad
            prior += 0.5 * float(rot @ rot + (c[:3, 3] / prior_m) @ (c[:3, 3] / prior_m))
        return float(huber.sum()) + prior, r, w, q

    cost, r, w, q = evaluate(poses, points)
    damping = 1e-3
    prior_info = np.concatenate([np.full(3, 1 / prior_rad**2), np.full(3, 1 / prior_m**2)])
    for _ in range(BA_ITERATIONS):
        z = q[:, 2]
        du_dq = np.zeros((obs_count, 2, 3))
        du_dq[:, 0, 0] = fx / z
        du_dq[:, 1, 1] = fx / z
        du_dq[:, 0, 2] = -fx * q[:, 0] / z**2
        du_dq[:, 1, 2] = -fx * q[:, 1] / z**2
        # q' = Rd^T (q - t) for a correction (w, t) in the camera's frame: dq/dw = [q]x, dq/dt = -I
        dq_dcam = np.concatenate([_skew(q), -np.broadcast_to(np.eye(3), (obs_count, 3, 3))], axis=2)
        j_cam = du_dq @ dq_dcam  # (N, 2, 6)
        rotations = np.array([p[:3, :3] for p in poses])[obs_frame]
        j_pt = du_dq @ np.transpose(rotations, (0, 2, 1))  # (N, 2, 3)
        jw_cam = j_cam * w[:, None, None]
        jw_pt = j_pt * w[:, None, None]
        h_cc = np.zeros((n, 6, 6))
        np.add.at(h_cc, obs_frame, np.einsum("nki,nkj->nij", jw_cam, j_cam))
        g_c = np.zeros((n, 6))
        np.add.at(g_c, obs_frame, np.einsum("nki,nk->ni", jw_cam, r))
        h_pp = np.zeros((m, 3, 3))
        np.add.at(h_pp, obs_point, np.einsum("nki,nkj->nij", jw_pt, j_pt))
        g_p = np.zeros((m, 3))
        np.add.at(g_p, obs_point, np.einsum("nki,nk->ni", jw_pt, r))
        h_cp = np.einsum("nki,nkj->nij", jw_cam, j_pt)  # (N, 6, 3) per observation
        corrections = np.array([correction(i) for i in range(n)])
        g_c += corrections * prior_info  # the prior's gradient
        while True:
            hc = h_cc + np.eye(6) * prior_info
            hc = hc + damping * np.einsum("nii->ni", hc)[:, :, None] * np.eye(6)
            hp = h_pp + damping * np.einsum("nii->ni", h_pp)[:, :, None] * np.eye(3) + 1e-12 * np.eye(3)
            hp_inv = np.linalg.inv(hp)
            m_k = h_cp @ hp_inv[obs_point]  # (N, 6, 3)
            schur = np.zeros((n, n, 6, 6))
            schur[np.arange(n), np.arange(n)] = hc
            contrib = np.einsum("pij,pkj->pik", m_k[pairs[:, 0]], h_cp[pairs[:, 1]])
            np.add.at(schur, (obs_frame[pairs[:, 0]], obs_frame[pairs[:, 1]]), -contrib)
            reduced = g_c.copy()
            np.add.at(reduced, obs_frame, -np.einsum("nij,nj->ni", m_k, g_p[obs_point]))
            big = schur.transpose(0, 2, 1, 3).reshape(6 * n, 6 * n)
            try:
                step_c = -np.linalg.solve(big, reduced.ravel()).reshape(n, 6)
            except np.linalg.LinAlgError:
                damping *= 10
                continue
            rhs = g_p.copy()
            np.add.at(rhs, obs_point, np.einsum("nji,nj->ni", h_cp, step_c[obs_frame]))
            step_p = -np.einsum("mij,mj->mi", hp_inv, rhs)
            trial_poses = np.array([poses[i] @ _pose(step_c[i, :3], step_c[i, 3:]) for i in range(n)])
            trial_points = points + step_p
            trial_cost, trial_r, trial_w, trial_q = evaluate(trial_poses, trial_points)
            if trial_cost < cost:
                improvement = (cost - trial_cost) / max(cost, 1e-12)
                poses, points, cost, r, w, q = (
                    trial_poses,
                    trial_points,
                    trial_cost,
                    trial_r,
                    trial_w,
                    trial_q,
                )
                damping = max(damping / 3, 1e-7)
                break
            damping *= 4
            if damping > 1e6:
                return poses, points
        if improvement < BA_TOLERANCE:
            break
    return poses, points


def _table(
    points: np.ndarray, rounds: int = 2000, seed: int = 0
) -> tuple[tuple[np.ndarray, float] | None, int]:
    """The table: the near-level plane (`TABLE_TILT_DEG`) through the most points (RANSAC), then least squares.

    Only level candidates: the scene's points include the forearm, objects and the floor past the table's edge, and
    the dominant plane among all of them was a tilted slab across the table's band (hil3: 7 deg, 3-7 mm off).
    """
    rng = np.random.default_rng(seed)
    best: np.ndarray = np.zeros(len(points), dtype=bool)
    level = np.cos(np.radians(TABLE_TILT_DEG))
    for _ in range(rounds):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        if np.linalg.norm(normal) < 1e-9:
            continue
        normal /= np.linalg.norm(normal)
        if abs(normal[2]) < level:
            continue
        inliers = np.abs((points - a) @ normal) < PLANE_MM / 1000
        if inliers.sum() > best.sum():
            best = inliers
    if best.sum() < 10:
        return None, 0
    on = points[best]
    center = on.mean(axis=0)
    normal = np.linalg.svd(on - center)[2][-1]
    normal = normal if normal[2] > 0 else -normal
    if normal[2] < level:
        return None, int(best.sum())
    return (normal, float(normal @ center)), int(best.sum())


def refine_views(
    views: Sequence["WristView"],
    method: str = "light",
    prior_mm: float = PRIOR_MM,
    prior_deg: float = PRIOR_DEG,
) -> SfmResult:
    """`refine` over wrist views of one arm (their frames, the model's camera poses, the jaws placed from them)."""
    if len({v.arm for v in views}) != 1:
        raise ValueError("views of one wrist camera")
    return refine(
        [v.color for v in views],
        views[0].camera,
        [v.root_from_camera for v in views],
        method,
        prior_mm,
        prior_deg,
        np.linalg.inv(views[-1].pinch_from_camera),
    )
