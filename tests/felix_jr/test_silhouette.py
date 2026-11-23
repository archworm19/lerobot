"""A cylinder's silhouette in a wrist camera and its position refined against the image (`harness/silhouette.py`).

The frames are rendered here independently of the module: each pixel's ray (through the lens, undistorted) cast at
the true cylinder (its side and its top), the object a colour on a noisy, shaded background.
"""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from lerobot.felix_jr.calibration.xle_robot_model import WristCamera  # noqa: E402
from lerobot.felix_jr.harness.primitives import Cylinder  # noqa: E402
from lerobot.felix_jr.harness.silhouette import EdgeView, contour_edges, refine, silhouette  # noqa: E402

LENS = WristCamera(
    640, 480, 450.0, 450.0, 320.0, 240.0, "rational", (-0.12, 0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
)
TRUE = Cylinder.upright(np.array([-0.30, 0.05, 0.78]), 0.13, 0.0225)  # pyri, standing on the table
UP = np.array([0.0, 0.0, 1.0])


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """root_from_camera (optical: x right, y down, z forward) at `eye`, looking at `target`, the image's up up."""
    z = (target - eye) / np.linalg.norm(target - eye)
    y = -UP - float(-UP @ z) * z
    y /= np.linalg.norm(y)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([np.cross(y, z), y, z])
    pose[:3, 3] = eye
    return pose


def render(cyl: Cylinder, root_from_camera: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(colour image, the object's mask): rays through each pixel cast at the cylinder (side and top)."""
    v, u = np.mgrid[0 : LENS.height, 0 : LENS.width].astype(np.float64)
    px = np.column_stack([u.ravel(), v.ravel()]).reshape(-1, 1, 2)
    rays = cv2.undistortPoints(px, LENS.matrix(), np.asarray(LENS.coeffs)).reshape(-1, 2)
    d = np.column_stack([rays, np.ones(len(rays))])
    cam_obj = np.linalg.inv(root_from_camera) @ cyl.root_from_obj[0]
    a, b = cam_obj[:3, 2], cam_obj[:3, 3]
    r, h = float(cyl.radius[0]), float(cyl.height[0])
    # the side: |(s d - b) - ((s d - b).a) a| = r
    dp = d - (d @ a)[:, None] * a
    bp = -b - float(-b @ a) * a
    qa, qb, qc = np.sum(dp * dp, axis=1), 2 * dp @ bp, float(bp @ bp) - r * r
    disc = qb * qb - 4 * qa * qc
    s = np.where(disc >= 0, (-qb - np.sqrt(np.maximum(disc, 0))) / (2 * qa), np.inf)
    along = s * (d @ a) - float(b @ a)
    side = np.isfinite(s) & (s > 0) & (along >= 0) & (along <= h)
    # the top: the plane (p - b).a = h within the radius
    denom = d @ a
    st = np.where(np.abs(denom) > 1e-9, (h + float(b @ a)) / denom, np.inf)
    hit = st[:, None] * d - b - h * a
    top = np.isfinite(st) & (st > 0) & (np.linalg.norm(hit - (hit @ a)[:, None] * a, axis=1) <= r)
    mask = (side | top).reshape(LENS.height, LENS.width)
    rng = np.random.default_rng(seed)
    img = np.empty((LENS.height, LENS.width, 3))
    img[:] = np.array([120.0, 110.0, 100.0]) + 30.0 * (u / LENS.width)[..., None]  # a shaded background
    img[mask] = np.array([40.0, 80.0, 200.0])
    img += rng.normal(0.0, 4.0, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), mask


def test_the_outline_is_the_rendered_silhouette():
    """The predicted outline (filled) and the rendered object overlap almost entirely, the lens's distortion and all."""
    pose = look_at(np.array([-0.12, 0.02, 0.88]), np.array([-0.30, 0.05, 0.84]))
    _, mask = render(TRUE, pose)
    outline = silhouette(TRUE, pose, LENS)
    filled = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillPoly(filled, [np.round(outline).astype(np.int32)], 1)
    iou = (filled.astype(bool) & mask).sum() / (filled.astype(bool) | mask).sum()
    assert iou > 0.97, iou
    edges = contour_edges(TRUE, pose, LENS)
    assert edges is not None and len(edges) == 2


def test_one_camera_and_the_radius_known_recovers_a_misplaced_prior():
    """The prior 1 cm off across the table: one wrist frame, the radius known, brings it back to within 1.5 mm."""
    pose = look_at(np.array([-0.12, 0.02, 0.88]), np.array([-0.30, 0.05, 0.84]))
    img, _ = render(TRUE, pose)
    prior = Cylinder.upright(np.array([-0.30 + 0.008, 0.05 - 0.006, 0.78]), 0.13, 0.0225)
    out = refine(prior, [EdgeView(img, pose, LENS)])
    error = np.linalg.norm(out.cylinder.bottom_root[0, :2] - TRUE.bottom_root[0, :2])
    assert out.ok, out.note
    assert error < 0.0015, error


def test_two_cameras_recover_position_and_radius():
    """Two wrist frames from different sides, the radius free: the prior 1 cm off and 2 mm too wide comes back to within
    1 mm and 0.5 mm."""
    target = np.array([-0.30, 0.05, 0.84])
    poses = [look_at(np.array([-0.12, 0.02, 0.88]), target), look_at(np.array([-0.24, 0.24, 0.90]), target)]
    views = [EdgeView(render(TRUE, p, seed=i)[0], p, LENS) for i, p in enumerate(poses)]
    prior = Cylinder.upright(np.array([-0.30 - 0.007, 0.05 + 0.007, 0.78]), 0.13, 0.0245)
    out = refine(prior, views, fit_radius=True)
    error = np.linalg.norm(out.cylinder.bottom_root[0, :2] - TRUE.bottom_root[0, :2])
    assert out.ok, out.note
    assert error < 0.001, error
    assert out.radius_m == pytest.approx(0.0225, abs=0.0005)


def test_no_estimate_when_the_object_is_out_of_view():
    """A camera looking away from the object: no edge, `ok` false, the prior kept."""
    pose = look_at(np.array([-0.12, 0.02, 0.88]), np.array([0.2, -0.3, 0.84]))
    img, _ = render(TRUE, pose)
    prior = Cylinder.upright(np.array([-0.30, 0.05, 0.78]), 0.13, 0.0225)
    out = refine(prior, [EdgeView(img, pose, LENS)])
    assert not out.ok


def test_never_confidently_wrong_when_the_prior_is_too_far():
    """A prior 3.5 cm off (past the first round's 2 cm band), two cameras and the radius free: it either comes back or
    says it has no estimate, never `ok` far from the truth."""
    target = np.array([-0.30, 0.05, 0.84])
    poses = [look_at(np.array([-0.12, 0.02, 0.88]), target), look_at(np.array([-0.24, 0.24, 0.90]), target)]
    views = [EdgeView(render(TRUE, p, seed=i)[0], p, LENS) for i, p in enumerate(poses)]
    rng = np.random.default_rng(3)
    for _ in range(8):
        angle = rng.uniform(0, 2 * np.pi)
        off = 0.035 * np.array([np.cos(angle), np.sin(angle)])
        prior = Cylinder.upright(
            np.array([-0.30 + off[0], 0.05 + off[1], 0.78]), 0.13, 0.0225 + rng.uniform(-0.002, 0.002)
        )
        out = refine(prior, views, fit_radius=True)
        error = np.linalg.norm(out.cylinder.bottom_root[0, :2] - TRUE.bottom_root[0, :2])
        assert not out.ok or error < 0.003, (error, out.note)
