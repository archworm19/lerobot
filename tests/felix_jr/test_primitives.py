"""The primitives' geometry: a cylinder's frame, its signed distance and its antipodal grasp family
(`docs/object_grasping.md`)."""

import numpy as np
import pytest

from lerobot.felix_jr.harness.primitives import Cylinder, frame_along, grasp_family

JAR = Cylinder.upright(np.array([-0.344, -0.107, 0.714]), 0.079, 0.020)  # bowl1's jar, the hand fit


def test_upright_frame_keeps_the_root_axes():
    np.testing.assert_allclose(JAR.root_from_obj[0, :3, :3], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(JAR.bottom_root[0], [-0.344, -0.107, 0.714])
    np.testing.assert_allclose(JAR.top_root[0], [-0.344, -0.107, 0.793])
    assert JAR.length[0] == pytest.approx(0.079)


def test_frame_along_any_axis_is_a_rotation():
    for axis in ([1.0, 0.0, 0.0], [0.3, -0.2, 0.9], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.01]):
        f = frame_along(np.array(axis), np.zeros(3))
        r = f[:3, :3]
        np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-12)
        assert np.linalg.det(r) == pytest.approx(1.0)
        np.testing.assert_allclose(r[:, 2], np.array(axis) / np.linalg.norm(axis))


def test_rejects_a_bad_frame_or_size():
    with pytest.raises(ValueError):
        Cylinder(np.array([[0.01, 0.05]]), np.diag([2.0, 1.0, 1.0, 1.0])[None])
    with pytest.raises(ValueError):
        Cylinder(np.array([[0.0, 0.05]]), np.eye(4)[None])
    with pytest.raises(ValueError, match="N, 2"):
        Cylinder(np.array([0.01, 0.05]), np.eye(4)[None])  # (not a batch)


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
    np.testing.assert_allclose(
        c.surface_distance(points)[0], [0.03, 0.03, 0.05, -0.02, -0.005, 0.0], atol=1e-12
    )


def test_grasps_are_antipodal_through_the_axis():
    grasps = grasp_family(JAR).antipodal_grasps()
    assert grasps
    for g in grasps:
        assert g.opening == pytest.approx(0.04)
        assert np.linalg.norm(g.closing_root) == pytest.approx(1.0)
        assert np.linalg.norm(g.approach_root) == pytest.approx(1.0)
        assert g.closing_root @ g.approach_root == pytest.approx(0.0, abs=1e-12)
        assert g.closing_root @ JAR.axis_root[0] == pytest.approx(
            0.0, abs=1e-12
        )  # square to the axis: through it
        for contact in g.contacts_root:
            assert JAR.surface_distance(contact[None])[0] == pytest.approx(0.0, abs=1e-12)


def test_default_grid_covers_the_family():
    grasps = grasp_family(JAR).antipodal_grasps()
    t = sorted({g.params["t"] for g in grasps})
    phi = sorted({g.params["phi"] for g in grasps})
    psi = sorted({g.params["psi"] for g in grasps})
    assert t[0] == 0.0 and t[-1] == pytest.approx(0.079) and np.diff(t).max() <= 0.01 + 1e-9
    assert phi == [30.0 * k for k in range(12)]  # a full turn: 0 once, 360 not again
    assert psi == [-90.0 + 15.0 * k for k in range(13)]
    assert len(grasps) == len(t) * len(phi) * len(psi)


def test_where_and_spacing_narrow_the_family():
    top = grasp_family(JAR).antipodal_grasps(
        where={"t": (0.059, 0.079), "psi": (0.0, 40.0)}, spacing={"phi": 90.0, "psi": 20.0}
    )
    assert sorted({g.params["phi"] for g in top}) == [0.0, 90.0, 180.0, 270.0]
    assert sorted({g.params["psi"] for g in top}) == [0.0, 20.0, 40.0]
    assert all(0.059 - 1e-9 <= g.params["t"] <= 0.079 + 1e-9 for g in top)
    with pytest.raises(ValueError, match="no parameter"):
        grasp_family(JAR).antipodal_grasps(where={"theta": (0.0, 1.0)})
    with pytest.raises(ValueError, match="outside the cylinder"):
        grasp_family(JAR).antipodal_grasps(where={"t": (0.0, 0.2)})


def test_too_wide_for_the_jaws_gives_none():
    assert grasp_family(JAR).antipodal_grasps(max_opening=0.039) == []
    assert grasp_family(JAR).antipodal_grasps(max_opening=0.041)


def test_approach_directions_upright():
    def one(phi: float, psi: float):
        (g,) = grasp_family(JAR).antipodal_grasps(
            where={"t": (0.07, 0.07), "phi": (phi, phi), "psi": (psi, psi)}
        )
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
    (g,) = grasp_family(pen).antipodal_grasps(
        where={"t": (0.07, 0.07), "phi": (90.0, 90.0), "psi": (0.0, 0.0)}
    )
    np.testing.assert_allclose(g.center_root, [-0.40, 0.17, 0.72], atol=1e-12)
    assert g.closing_root @ pen.axis_root[0] == pytest.approx(0.0, abs=1e-12)
    assert g.approach_root @ pen.axis_root[0] == pytest.approx(0.0, abs=1e-12)
    assert "90 deg from vertical" in pen.describe()


def test_a_cylinder_prefers_the_side():
    """`approach_preference`: 0 square to the axis, 90 along it, the angle between."""
    upright = Cylinder.upright(np.array([0.0, 0.0, 0.0]), 0.1, 0.02)
    assert grasp_family(upright).approach_preference(np.array([1.0, 0.0, 0.0])) == pytest.approx(0.0)
    assert grasp_family(upright).approach_preference(np.array([0.0, 0.0, -1.0])) == pytest.approx(90.0)
    down45 = np.array([1.0, 0.0, -1.0]) / np.sqrt(2.0)
    assert grasp_family(upright).approach_preference(down45) == pytest.approx(45.0)
    assert grasp_family(upright).approach_preference(2.0 * down45) == pytest.approx(45.0)  # any length


def test_a_cylinders_grasp_space_and_its_grasp_anywhere_in_it():
    """`grasp_space` names `t`, `phi` (wrapping all the way round) and `psi` with their ranges (narrowed by `where`,
    refused outside the shape); `grasp_at` makes the grid's grasps exactly at its points and a grasp between them too."""
    cyl = Cylinder.upright(np.array([-0.3, 0.1, 0.77]), 0.13, 0.0225)
    space = grasp_family(cyl).grasp_space()
    assert set(space) == {"t", "phi", "psi"}
    assert (space["t"].low, space["t"].high, space["t"].unit) == (0.0, 0.13, "m")
    assert space["phi"].period == 360.0 and space["psi"].period is None
    narrowed = grasp_family(cyl).grasp_space({"psi": (0.0, 30.0)})
    assert (narrowed["psi"].low, narrowed["psi"].high) == (0.0, 30.0) and narrowed["t"].high == 0.13
    with pytest.raises(ValueError, match="outside"):
        grasp_family(cyl).grasp_space({"t": (0.0, 0.2)})
    with pytest.raises(ValueError, match="no parameter"):
        grasp_family(cyl).grasp_space({"depth": (0.0, 0.01)})
    for grasp in grasp_family(cyl).antipodal_grasps({"psi": (0.0, 30.0)}):
        again = grasp_family(cyl).grasp_at(grasp.params)
        assert np.allclose(again.center_root, grasp.center_root) and np.allclose(
            again.approach_root, grasp.approach_root
        )
    between = grasp_family(cyl).grasp_at({"t": 0.055, "phi": 195.0, "psi": 7.5})  # (off the grid)
    assert between.center_root[2] == pytest.approx(0.77 + 0.055) and between.opening == pytest.approx(0.045)
    assert (
        abs(between.approach_root @ between.closing_root) < 1e-12
    )  # (still antipodal: the closing square to it)
    side = -between.approach_root - np.sin(np.radians(7.5)) * np.array(
        [0.0, 0.0, 1.0]
    )  # (the approach untilted)
    assert np.degrees(np.arctan2(side[1], side[0])) % 360 == pytest.approx(195.0, abs=1e-6)


def test_a_batch_is_geometry_alike_in_numpy_and_torch():
    """A `Primitive` is a batch: each object's geometry at once, the same numbers in numpy (planning) and torch (the
    simulator, its tensors left where they are); one object is a batch of one, and indexing keeps a batch."""
    params = np.array([[0.0225, 0.13], [0.03, 0.19]])
    batch = Cylinder.from_params(params)
    assert len(batch) == 2 and len(batch[1]) == 1 and float(batch[1].radius[0]) == 0.03
    np.testing.assert_allclose(batch.bounding_radius(), np.hypot(params[:, 0], params[:, 1] / 2))
    np.testing.assert_allclose(
        batch.half_extents(), np.column_stack([params[:, 0], params[:, 0], params[:, 1] / 2])
    )
    side = (3 * params[:, 0] ** 2 + params[:, 1] ** 2) / 12
    np.testing.assert_allclose(batch.inertia_per_kg(), np.column_stack([side, side, params[:, 0] ** 2 / 2]))
    np.testing.assert_allclose(batch.volume(), np.pi * params[:, 0] ** 2 * params[:, 1])
    np.testing.assert_allclose(batch.centroid_obj()[:, 2], params[:, 1] / 2)
    torch = pytest.importorskip("torch")
    pytest.importorskip("array_api_compat")
    tensors = Cylinder.from_params(torch.tensor(params))
    for name in ("bounding_radius", "half_extents", "inertia_per_kg", "volume", "centroid_obj"):
        out = getattr(tensors, name)()
        assert isinstance(out, torch.Tensor), name
        np.testing.assert_allclose(out.numpy(), getattr(batch, name)(), err_msg=name)
    points = torch.tensor([[0.0, 0.0, 0.2], [0.1, 0.0, 0.05]], dtype=torch.float64)
    np.testing.assert_allclose(
        tensors.surface_distance(points).numpy(), batch.surface_distance(points.numpy())
    )


def test_an_old_pickled_cylinder_is_a_batch_of_one():
    """A session saved before cylinders were batches (`root_from_obj`, `radius`, `length`) loads as a batch of one."""
    old = Cylinder.__new__(Cylinder)
    old.__setstate__({"root_from_obj": np.eye(4), "radius": 0.02, "length": 0.1})
    assert len(old) == 1 and float(old.radius[0]) == 0.02 and float(old.length[0]) == 0.1


def test_an_object_is_written_down_one_way():
    """`ObjectRecord`: its record (the bank's placements, read back alike, old ones as cylinders), its rows (the
    pools'), a batch of primitives with a mass each."""
    from lerobot.felix_jr.harness.primitives import ObjectRecord

    one = ObjectRecord.upright(
        "cylinder", {"radius": 0.0225, "height": 0.13}, 0.08, np.array([-0.3, 0.2, 0.78])
    )
    record = one.to_record()
    assert record == {
        "shape": "cylinder", "radius": 0.0225, "height": 0.13, "mass_kg": 0.08, "bottom_root": [-0.3, 0.2, 0.78],
    }  # fmt: skip
    back = ObjectRecord.from_record(record)
    assert back.to_record() == record and np.allclose(
        back.primitive.root_from_obj, one.primitive.root_from_obj
    )
    old = {"index": 3, "bottom_root": [-0.3, 0.2, 0.78], "radius": 0.0225, "height": 0.13, "mass_kg": 0.08}
    assert ObjectRecord.from_record(old).to_record() == record  # (written before shapes were named)
    tilted = ObjectRecord(Cylinder.between(np.zeros(3), np.array([0.1, 0.0, 0.0]), 0.01), np.array([0.02]))
    assert np.allclose(ObjectRecord.from_record(tilted.to_record()).primitive.axis_root[0], [1.0, 0.0, 0.0])
    assert ObjectRecord.columns() == ("radius", "height", "mass_kg")
    np.testing.assert_allclose(one.vectors(), [[0.0225, 0.13, 0.08]])
    two = ObjectRecord.from_vectors(np.array([[0.02, 0.1, 0.05], [0.03, 0.2, 0.1]]))
    assert len(two) == 2 and two[1].to_record()["radius"] == 0.03
    moved = one.placed_upright(np.array([0.1, 0.2, 0.7])).to_record()
    assert moved["bottom_root"] == [0.1, 0.2, 0.7] and moved["mass_kg"] == 0.08
    with pytest.raises(ValueError, match="unknown shape"):
        ObjectRecord.from_record(record | {"shape": "torus"})
    with pytest.raises(ValueError, match="one object"):
        two.to_record()
