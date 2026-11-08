"""Structure from motion on wrist frames (`harness.sfm`), on a synthetic textured table seen from known poses.

The frames are rendered exactly (a plane maps to an image by a homography); the "model" poses given to `refine` are
the true ones perturbed by a few mm and degrees per frame (the arm's motion error, bowl6). The refined motion between
frames should be the true one, and the table its plane.
"""

import numpy as np
import pytest

pytest.importorskip("scipy")

import cv2  # noqa: E402

from lerobot.felix_jr.calibration.xle_robot_model import WristCamera  # noqa: E402
from lerobot.felix_jr.harness.sfm import _pose, refine  # noqa: E402

CAMERA = WristCamera(640, 480, 340.0, 340.0, 320.0, 240.0, "rational", (0.0,) * 8)
TABLE_Z = 0.735
TEXTURE_M = 0.40  # the texture covers this square of the table, centered under the cameras
TEXTURE_PX = 1600


def table_texture(seed: int = 0) -> np.ndarray:
    """Wood-like grain: streaks along x at several scales, plus fine noise."""
    rng = np.random.default_rng(seed)
    image = np.zeros((TEXTURE_PX, TEXTURE_PX), np.float32)
    for scale, weight in ((4, 1.0), (16, 0.6), (64, 0.4)):
        small = rng.random((TEXTURE_PX // scale, max(TEXTURE_PX // (scale * 8), 2))).astype(np.float32)
        image += weight * cv2.resize(small, (TEXTURE_PX, TEXTURE_PX), interpolation=cv2.INTER_CUBIC)
    image += 0.3 * rng.random(image.shape).astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min())
    rgb = np.stack([0.85 * image + 0.1, 0.6 * image + 0.1, 0.4 * image + 0.05], axis=2)
    return (255 * rgb).astype(np.uint8)


def render(
    texture: np.ndarray,
    root_from_camera: np.ndarray,
    camera: WristCamera = CAMERA,
    center: tuple[float, float] = (-0.40, 0.05),
    table_z: float = TABLE_Z,
) -> np.ndarray:
    """The table as the camera sees it: texture pixel -> table point -> image pixel, one homography."""
    half = TEXTURE_M / 2
    s = TEXTURE_M / TEXTURE_PX
    # table point (x, y, table_z) from the texture's pixel (u, v), the texture centered at `center`
    root_from_texture = np.array([[s, 0, center[0] - half], [0, s, center[1] - half], [0, 0, 1.0]])
    camera_from_root = np.linalg.inv(root_from_camera)
    r, t = camera_from_root[:3, :3], camera_from_root[:3, 3]
    # camera point = r @ (x, y, z0) + t = [r1 r2 (r3 z0 + t)] @ (x, y, 1)
    plane = np.column_stack([r[:, 0], r[:, 1], r[:, 2] * table_z + t])
    h = camera.matrix() @ plane @ root_from_texture
    return cv2.warpPerspective(texture, h, (camera.width, camera.height), flags=cv2.INTER_LINEAR)


def wrist_sweep(rolls_deg=(-30, -15, 0, 15, 30)) -> list[np.ndarray]:
    """A camera 7 cm off a gripper axis pointing down, ~15 cm over the table, the wrist rolled through `rolls_deg`."""
    axis_point = np.array([-0.40, 0.05, TABLE_Z + 0.15])
    poses = []
    for roll in rolls_deg:
        spin = _pose(np.radians([0.0, 0.0, roll]), np.zeros(3))
        # the camera looks down (optical z = -z root), 7 cm off the axis along x, tilted 10 deg back toward it
        look = _pose(np.radians([180.0, 0.0, 0.0]), np.zeros(3)) @ _pose(
            np.radians([-10.0, 0.0, 0.0]), np.zeros(3)
        )
        offset = _pose(np.zeros(3), np.array([0.07, 0.0, 0.0]))
        poses.append(_pose(np.zeros(3), axis_point) @ spin @ offset @ look)
    return poses


def perturbed(poses, mm=3.0, deg=1.0, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for i, p in enumerate(poses):
        if i == 0:
            out.append(p.copy())
            continue
        out.append(
            p
            @ _pose(np.radians(rng.normal(0, deg / np.sqrt(3), 3)), rng.normal(0, mm / 1000 / np.sqrt(3), 3))
        )
    return out


def motion_errors(estimate, truth):
    """Per frame, the motion from the first frame against the true motion: its turn's error (deg), its direction's
    error (deg) and its length's ratio to the true one (the images fix motion only up to one scale)."""
    turns, directions, ratios = [], [], []
    for e, t in zip(estimate[1:], truth[1:], strict=True):
        m_e, m_t = np.linalg.inv(estimate[0]) @ e, np.linalg.inv(truth[0]) @ t
        d = np.linalg.inv(m_t) @ m_e
        turns.append(np.degrees(np.linalg.norm(cv2.Rodrigues(d[:3, :3])[0])))
        a, b = m_e[:3, 3], m_t[:3, 3]
        directions.append(
            np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1.0, 1.0)))
        )
        ratios.append(np.linalg.norm(a) / np.linalg.norm(b))
    return np.array(turns), np.array(directions), np.array(ratios)


@pytest.fixture(scope="module")
def scene():
    texture = table_texture()
    truth = wrist_sweep()
    return [render(texture, p) for p in truth], truth, perturbed(truth)


def check(result, truth, model):
    """What structure from motion can recover: the turns, the moves' directions, one scale for all of it, the table."""
    turns_0, directions_0, _ = motion_errors(model, truth)
    turns, directions, ratios = motion_errors(result.root_from_camera, truth)
    assert result.rms_after_px < 1.0, result.describe()
    # a flat table leaves a small turn and a shift nearly alike: ~0.3 deg left, 0.6 mm at 12 cm
    assert turns.max() < 0.4 and turns.max() < 0.7 * turns_0.max(), (turns_0, turns)
    assert directions.max() < 1.5 and directions.max() < 0.5 * directions_0.max(), (directions_0, directions)
    assert ratios.std() < 0.01 and abs(ratios.mean() - 1) < 0.05, ratios  # one scale, the prior's, within 5%
    assert result.plane is not None, result.describe()
    normal, offset = result.plane
    assert normal[2] > 0.999, result.describe()  # level
    # the table's distance below the first camera: true up to the same scale
    seen = result.root_from_camera[0][2, 3] - offset / normal[2]
    true = truth[0][2, 3] - TABLE_Z
    assert abs(seen / true - ratios.mean()) < 0.02, (seen, true, ratios.mean())


def test_the_rendered_scene_is_the_table():
    """The renderer: the table point under the first camera's optical axis lands on the principal point."""
    truth = wrist_sweep()
    pose = truth[2]
    ray = pose[:3, 2]
    point = pose[:3, 3] + (TABLE_Z - pose[2, 3]) / ray[2] * ray
    uv = CAMERA.project(((point - pose[:3, 3]) @ pose[:3, :3])[None, :])[0]
    np.testing.assert_allclose(uv, [CAMERA.ppx, CAMERA.ppy], atol=1e-6)
    image = render(table_texture(), pose)
    assert image.std() > 20  # textured, not empty


def test_light_refines_the_motion_and_finds_the_table(scene):
    images, truth, model = scene
    result = refine(images, CAMERA, model, method="light")
    check(result, truth, model)
    assert "sfm (light)" in result.describe()


def test_torch_refines_the_motion_and_finds_the_table(scene):
    pytest.importorskip("kornia")
    images, truth, model = scene
    try:
        result = refine(images, CAMERA, model, method="torch")
    except (OSError, RuntimeError) as e:  # the weights could not be downloaded
        if "download" in str(e).lower() or "urlopen" in str(e).lower():
            pytest.skip(f"no learned matcher weights: {e}")
        raise
    check(result, truth, model)


def test_too_little_texture_is_said(scene):
    _, truth, model = scene
    blank = [np.full((480, 640, 3), 128, np.uint8) for _ in truth]
    with pytest.raises(ValueError, match="too few scene points"):
        refine(blank, CAMERA, model)
