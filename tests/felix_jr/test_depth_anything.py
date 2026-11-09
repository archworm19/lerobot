"""`harness.depth_anything`: the similarity fit that aligns DA3's poses to ours (no model needed)."""

import cv2
import numpy as np

from lerobot.felix_jr.harness.depth_anything import Depth, umeyama


def test_umeyama_recovers_a_similarity():
    rng = np.random.default_rng(0)
    source = rng.normal(size=(6, 3))
    rotation = cv2.Rodrigues(np.array([0.3, -0.2, 0.5]))[0]
    target = 2.5 * source @ rotation.T + [0.1, -0.4, 0.2]
    r, t, s = umeyama(source, target)
    np.testing.assert_allclose(r, rotation, atol=1e-9)
    np.testing.assert_allclose(t, [0.1, -0.4, 0.2], atol=1e-9)
    assert abs(s - 2.5) < 1e-9


def test_points_back_project_through_the_pose():
    k = np.array([[100.0, 0, 20], [0, 100.0, 10], [0, 0, 1]])
    c2w = np.eye(4)
    c2w[:3, 3] = [1.0, 2.0, 3.0]
    w2c = np.linalg.inv(c2w)[:3]
    d = Depth(np.full((1, 21, 41), 2.0), np.ones((1, 21, 41)), k[None], w2c[None])
    p = d.points(0)
    np.testing.assert_allclose(p[10, 20], [1.0, 2.0, 5.0])  # the principal point, 2 m ahead
