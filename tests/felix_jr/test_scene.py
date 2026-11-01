import numpy as np
import pytest

from lerobot.felix_jr.harness.scene import Scene
from tests.felix_jr.conftest import BOX_Z, FX, PPX, PPY, WALL_Z, H, W, make_calibration


class TestPixelToXyz:
    def test_back_projects_a_pixel_on_the_wall(self, scene):
        xyz = scene.pixel_to_xyz(np.array([[100.0, 30.0]]))
        expected = [(100 - PPX) / FX * WALL_Z, (30 - PPY) / FX * WALL_Z, WALL_Z]
        np.testing.assert_allclose(xyz[0], expected, atol=1e-6)

    def test_box_face_is_nearer(self, scene):
        xyz = scene.pixel_to_xyz(np.array([[80.0, 60.0]]))
        assert xyz[0, 2] == pytest.approx(BOX_Z)

    def test_shapes_follow_the_input(self, scene):
        assert scene.pixel_to_xyz(np.array([100.0, 30.0])).shape == (3,)
        assert scene.pixel_to_xyz(np.zeros((5, 2)) + 100).shape == (5, 3)
        assert scene.pixel_to_xyz(np.zeros((2, 4, 2)) + 100).shape == (2, 4, 3)

    def test_uses_the_exact_fractional_pixel_for_x_and_y(self, scene):
        xyz = scene.pixel_to_xyz(np.array([100.4, 30.0]))
        assert xyz[0] == pytest.approx((100.4 - PPX) / FX * WALL_Z)

    def test_standoff_moves_the_point_toward_the_camera_along_its_ray(self, scene):
        on_wall = scene.pixel_to_xyz(np.array([[100.0, 30.0]]))[0]
        in_front = scene.pixel_to_xyz(np.array([[100.0, 30.0]]), standoff=0.25)[0]
        assert in_front[2] == pytest.approx(WALL_Z - 0.25)
        np.testing.assert_allclose(in_front / in_front[2], on_wall / on_wall[2])  # same ray
        assert scene.xyz_to_pixel(in_front) == pytest.approx([100.0, 30.0])

    def test_standoff_keeps_invalid_points_nan(self, scene):
        assert np.isnan(scene.pixel_to_xyz(np.array([[20.0, 15.0]]), standoff=0.1)).all()

    def test_pixel_without_depth_is_nan(self, scene):
        assert np.isnan(scene.pixel_to_xyz(np.array([[20.0, 15.0]]))).all()

    def test_pixels_outside_the_image_are_nan(self, scene):
        uv = np.array([[-5.0, 30.0], [W + 3.0, 30.0], [50.0, -1.0], [50.0, H + 2.0], [np.nan, 30.0]])
        assert np.isnan(scene.pixel_to_xyz(uv)).all()

    def test_median_ignores_a_single_bad_pixel(self, scene):
        depth = scene.depth_m.copy()
        depth[30, 100] = 0.2  # one outlier in the 5x5 patch around (100, 30)
        noisy = Scene(scene.color, depth, scene.fx, scene.fy, scene.ppx, scene.ppy)
        assert noisy.pixel_to_xyz(np.array([100.0, 30.0]))[2] == pytest.approx(WALL_Z)

    def test_a_patch_straddling_a_depth_edge_is_rejected(self, scene):
        edge = np.array([[99.0, 60.0]])  # the box's right edge: patch holds 0.6 m and 1.0 m depths
        assert np.isnan(scene.pixel_to_xyz(edge)).all()
        assert np.isfinite(scene.pixel_to_xyz(edge, max_spread_m=1.0)).all()

    def test_needs_enough_valid_pixels_in_the_patch(self, scene):
        # (11, 15) sits just inside the hole's left edge (hole = columns 10-29): only column 9 of the
        # 5x5 patch, 5 of its 25 pixels, has depth
        near_edge = np.array([[11.0, 15.0]])
        assert np.isfinite(scene.pixel_to_xyz(near_edge, min_valid=3)).all()
        assert np.isnan(scene.pixel_to_xyz(near_edge, min_valid=6)).all()


class TestXyzToPixel:
    def test_round_trip_through_pixel_to_xyz(self, scene):
        uv = np.array([[100.0, 30.0], [33.5, 100.25], [130.0, 90.0]])
        back = scene.xyz_to_pixel(scene.pixel_to_xyz(uv))
        np.testing.assert_allclose(back, uv, atol=1e-6)

    def test_known_projection(self, scene):
        uv = scene.xyz_to_pixel(np.array([0.2, -0.1, 1.0]))
        np.testing.assert_allclose(uv, [PPX + 20.0, PPY - 10.0])

    def test_behind_or_at_the_camera_is_nan(self, scene):
        uv = scene.xyz_to_pixel(np.array([[0.0, 0.0, -1.0], [0.1, 0.1, 0.0]]))
        assert np.isnan(uv).all()

    def test_shapes_follow_the_input(self, scene):
        assert scene.xyz_to_pixel(np.ones((4, 3))).shape == (4, 2)
        assert scene.xyz_to_pixel(np.ones((2, 5, 3))).shape == (2, 5, 2)

    def test_in_image(self, scene):
        uv = np.array([[0.0, 0.0], [W - 1.0, H - 1.0], [W + 1.0, 5.0], [np.nan, np.nan]])
        assert scene.in_image(uv).tolist() == [True, True, False, False]

    def test_rejects_wrong_last_dimension(self, scene):
        with pytest.raises(ValueError):
            scene.xyz_to_pixel(np.ones((4, 2)))
        with pytest.raises(ValueError):
            scene.pixel_to_xyz(np.ones((4, 3)))


class TestFromCapture:
    def test_converts_raw_depth_to_meters_and_marks_missing_as_nan(self):
        raw = np.full((H, W, 1), 1000, dtype=np.uint16)
        raw[5:10, 5:10] = 0
        scene = Scene.from_capture(np.zeros((H, W, 3), np.uint8), raw, _calib_with_matrices())
        assert scene.depth_m[60, 80] == pytest.approx(1.0)
        assert np.isnan(scene.depth_m[7, 7])
        assert (scene.fx, scene.ppx) == (FX, PPX)


def _calib_with_matrices() -> dict:
    from lerobot.felix_jr.depth import with_matrices

    return with_matrices(make_calibration())
