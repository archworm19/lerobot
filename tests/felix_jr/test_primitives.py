"""The primitives' geometry: a cylinder's frame, its signed distance and its antipodal grasp family
(`docs/object_grasping.md`)."""

import numpy as np
import pytest

from lerobot.felix_jr.harness.primitives import Cylinder, frame_along

JAR = Cylinder.upright(np.array([-0.344, -0.107, 0.714]), 0.079, 0.020)  # bowl1's jar, the hand fit


def test_upright_frame_keeps_the_root_axes():
    np.testing.assert_allclose(JAR.root_from_obj[:3, :3], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(JAR.bottom_root, [-0.344, -0.107, 0.714])
    np.testing.assert_allclose(JAR.top_root, [-0.344, -0.107, 0.793])
    assert JAR.length == pytest.approx(0.079)


def test_frame_along_any_axis_is_a_rotation():
    for axis in ([1.0, 0.0, 0.0], [0.3, -0.2, 0.9], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.01]):
        f = frame_along(np.array(axis), np.zeros(3))
        r = f[:3, :3]
        np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-12)
        assert np.linalg.det(r) == pytest.approx(1.0)
        np.testing.assert_allclose(r[:, 2], np.array(axis) / np.linalg.norm(axis))


def test_rejects_a_bad_frame_or_size():
    with pytest.raises(ValueError):
        Cylinder(np.diag([2.0, 1.0, 1.0, 1.0]), 0.01, 0.05)
    with pytest.raises(ValueError):
        Cylinder(np.eye(4), 0.0, 0.05)


def test_surface_distance():
    c = Cylinder.between(np.zeros(3), np.array([0.0, 0.0, 0.1]), 0.02)
    points = np.array(
        [
            [0.05, 0.0, 0.05],  # beside it: 3 cm out
            [0.0, 0.0, 0.13],  # above the top: 3 cm
            [0.05, 0.0, 0.14],  # off the top's rim: (3, 4) cm -> 5
            [0.0, 0.0, 0.05],  # on the axis, inside: 2 cm to the side wall
            [0.0, 0.015, 0.095],  # inside, nearest the top: 0.5 cm
            [0.02, 0.0, 0.05],  # on the wall
        ]
    )
    np.testing.assert_allclose(c.surface_distance(points), [0.03, 0.03, 0.05, -0.02, -0.005, 0.0], atol=1e-12)


def test_grasps_are_antipodal_through_the_axis():
    grasps = JAR.antipodal_grasps()
    assert grasps
    for g in grasps:
        assert g.opening == pytest.approx(0.04)
        assert np.linalg.norm(g.closing_root) == pytest.approx(1.0)
        assert np.linalg.norm(g.approach_root) == pytest.approx(1.0)
        assert g.closing_root @ g.approach_root == pytest.approx(0.0, abs=1e-12)
        assert g.closing_root @ JAR.axis_root == pytest.approx(
            0.0, abs=1e-12
        )  # square to the axis: through it
        for contact in g.contacts_root:
            assert JAR.surface_distance(contact[None])[0] == pytest.approx(0.0, abs=1e-12)


def test_default_grid_covers_the_family():
    grasps = JAR.antipodal_grasps()
    t = sorted({g.params["t"] for g in grasps})
    phi = sorted({g.params["phi"] for g in grasps})
    psi = sorted({g.params["psi"] for g in grasps})
    assert t[0] == 0.0 and t[-1] == pytest.approx(0.079) and np.diff(t).max() <= 0.01 + 1e-9
    assert phi == [30.0 * k for k in range(12)]  # a full turn: 0 once, 360 not again
    assert psi == [-90.0 + 15.0 * k for k in range(13)]
    assert len(grasps) == len(t) * len(phi) * len(psi)


def test_where_and_spacing_narrow_the_family():
    top = JAR.antipodal_grasps(
        where={"t": (0.059, 0.079), "psi": (0.0, 40.0)}, spacing={"phi": 90.0, "psi": 20.0}
    )
    assert sorted({g.params["phi"] for g in top}) == [0.0, 90.0, 180.0, 270.0]
    assert sorted({g.params["psi"] for g in top}) == [0.0, 20.0, 40.0]
    assert all(0.059 - 1e-9 <= g.params["t"] <= 0.079 + 1e-9 for g in top)
    with pytest.raises(ValueError, match="no parameter"):
        JAR.antipodal_grasps(where={"theta": (0.0, 1.0)})
    with pytest.raises(ValueError, match="outside the cylinder"):
        JAR.antipodal_grasps(where={"t": (0.0, 0.2)})


def test_too_wide_for_the_jaws_gives_none():
    assert JAR.antipodal_grasps(max_opening=0.039) == []
    assert JAR.antipodal_grasps(max_opening=0.041)


def test_approach_directions_upright():
    def one(phi: float, psi: float):
        (g,) = JAR.antipodal_grasps(where={"t": (0.07, 0.07), "phi": (phi, phi), "psi": (psi, psi)})
        return g

    side = one(0.0, 0.0)  # from the robot's side (+x), square to the axis: moving toward -x, closing along y
    np.testing.assert_allclose(side.approach_root, [-1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(abs(side.closing_root), [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(side.center_root, [-0.344, -0.107, 0.784], atol=1e-12)
    down = one(0.0, 40.0)  # 40 deg down from the robot's side
    np.testing.assert_allclose(
        down.approach_root, [-np.cos(np.radians(40)), 0.0, -np.sin(np.radians(40))], atol=1e-12
    )
    above = one(90.0, 90.0)  # straight down, the jaws closing along x
    np.testing.assert_allclose(above.approach_root, [0.0, 0.0, -1.0], atol=1e-12)
    np.testing.assert_allclose(abs(above.closing_root), [1.0, 0.0, 0.0], atol=1e-12)


def test_a_lying_pen():
    pen = Cylinder.between(np.array([-0.40, 0.10, 0.72]), np.array([-0.40, 0.24, 0.72]), 0.005)
    (g,) = pen.antipodal_grasps(where={"t": (0.07, 0.07), "phi": (90.0, 90.0), "psi": (0.0, 0.0)})
    np.testing.assert_allclose(g.center_root, [-0.40, 0.17, 0.72], atol=1e-12)
    assert g.closing_root @ pen.axis_root == pytest.approx(0.0, abs=1e-12)
    assert g.approach_root @ pen.axis_root == pytest.approx(0.0, abs=1e-12)
    assert "90 deg from vertical" in pen.describe()
