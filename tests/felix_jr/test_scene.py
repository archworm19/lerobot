import numpy as np
import pytest

from lerobot.felix_jr.harness.scene import BLOCKED, FREE, UNKNOWN, Scene
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

    def test_a_point_on_a_surface_is_blocked_but_with_standoff_it_is_free(self, scene):
        uv = np.array([[100.0, 100.0]])
        goal = np.array([0.2, 0.2, 0.5])  # in front of the wall, clear of the box
        on_surface = scene.pixel_to_xyz(uv)[0]
        floating = scene.pixel_to_xyz(uv, standoff=0.05)[0]
        assert scene.segment_check(on_surface, goal).status.tolist() == [BLOCKED]
        assert scene.segment_check(floating, goal).status.tolist() == [FREE]

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


class TestSegmentCheck:
    def test_segment_in_front_of_everything_is_free(self, scene):
        check = scene.segment_check(np.array([-0.2, 0.0, 0.3]), np.array([0.2, 0.0, 0.5]))
        assert check.status.tolist() == [FREE]
        assert np.isnan(check.t_first).all() and np.isnan(check.point_first).all()

    def test_segment_into_the_wall_is_blocked_where_it_meets_it(self, scene):
        check = scene.segment_check(np.array([0.3, 0.3, 0.8]), np.array([0.3, 0.3, 1.2]))
        assert check.status.tolist() == [BLOCKED]
        assert 0.95 <= check.point_first[0, 2] <= WALL_Z  # first within radius + tolerance of the wall
        assert 0.4 < check.t_first[0] < 0.6

    def test_segment_behind_a_surface_is_unknown_not_blocked(self, scene):
        check = scene.segment_check(np.array([0.3, 0.3, 1.2]), np.array([0.3, 0.3, 1.4]))
        assert check.status.tolist() == [UNKNOWN]

    def test_segment_through_a_depth_hole_is_unknown(self, scene):
        # pixel (20, 15) has no depth; at z = 0.5 it corresponds to x = -0.3, y = -0.225
        p = np.array([-0.3, -0.225, 0.5])
        assert scene.segment_check(p, p + [0.0, 0.0, 0.01]).status.tolist() == [UNKNOWN]

    def test_segment_leaving_the_image_is_unknown(self, scene):
        check = scene.segment_check(np.array([0.0, 0.0, 0.5]), np.array([2.0, 0.0, 0.5]))
        assert check.status.tolist() == [UNKNOWN]
        assert check.t_first[0] > 0  # the part inside the view is free

    def test_segment_running_into_the_box_stops_at_its_edge(self, scene):
        check = scene.segment_check(np.array([-0.3, 0.0, BOX_Z]), np.array([0.3, 0.0, BOX_Z]))
        assert check.status.tolist() == [BLOCKED]
        assert -0.16 <= check.point_first[0, 0] <= -0.11  # box starts at x = -0.12 (column 60)

    def test_radius_decides_whether_a_near_miss_is_blocked(self, scene):
        # 4 cm to the right of the box (its edge is at x = +0.12), moving along y at the box's depth
        a, b = np.array([0.16, -0.05, BOX_Z]), np.array([0.16, 0.05, BOX_Z])
        assert scene.segment_check(a, b, radius=0.01).status.tolist() == [FREE]
        assert scene.segment_check(a, b, radius=0.06).status.tolist() == [BLOCKED]

    def test_checks_many_segments_at_once(self, scene):
        p1 = np.array([[-0.2, 0.0, 0.3], [0.3, 0.3, 0.8], [0.3, 0.3, 1.2]])
        p2 = np.array([[0.2, 0.0, 0.5], [0.3, 0.3, 1.2], [0.3, 0.3, 1.4]])
        check = scene.segment_check(p1, p2)
        assert check.status.tolist() == [FREE, BLOCKED, UNKNOWN]
        assert check.sample_xyz.shape[0] == 3 and check.sample_state.shape == check.sample_xyz.shape[:2]

    def test_results_do_not_depend_on_chunking(self, scene):
        rng = np.random.default_rng(0)
        p1 = rng.uniform([-0.3, -0.2, 0.3], [0.3, 0.2, 1.3], size=(20, 3))
        p2 = rng.uniform([-0.3, -0.2, 0.3], [0.3, 0.2, 1.3], size=(20, 3))
        whole = scene.segment_check(p1, p2, chunk=64)
        pieces = scene.segment_check(p1, p2, chunk=3)
        np.testing.assert_array_equal(whole.sample_state, pieces.sample_state)

    def test_nan_points_are_unknown_not_a_crash(self, scene):
        p1 = np.array([[np.nan, 0.0, 0.5], [0.0, 0.0, 0.3]])
        p2 = np.array([[0.1, 0.0, 0.5], [0.1, 0.0, np.nan]])
        check = scene.segment_check(p1, p2)
        assert check.status.tolist() == [UNKNOWN, UNKNOWN]

    def test_zero_length_segment_and_wrong_shapes(self, scene):
        p = np.array([0.0, 0.0, 0.3])
        assert scene.segment_check(p, p).status.tolist() == [FREE]
        with pytest.raises(ValueError):
            scene.segment_check(np.zeros((2, 3)), np.zeros((3, 3)))


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
