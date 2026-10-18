import numpy as np

from lerobot.felix_jr.depth import (
    align_depth_to_color,
    depth_edges,
    fill_cracks,
    warp_mask_to_color,
    with_matrices,
)
from tests.felix_jr.conftest import FX, H, W, make_calibration


def raw_plane(z_mm: int = 1000) -> np.ndarray:
    return np.full((H, W), z_mm, dtype=np.uint16)


class TestWithMatrices:
    def test_rotation_is_read_column_major(self):
        # librealsense stores the 3x3 rotation column-major: this list is a 90 degree turn about z
        calib = make_calibration()
        calib["depth_to_color_rotation_col_major"] = [0, 1, 0, -1, 0, 0, 0, 0, 1]
        rotation = with_matrices(calib)["rotation"]
        np.testing.assert_array_equal(rotation, [[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        np.testing.assert_allclose(rotation @ [1, 0, 0], [0, 1, 0])  # x axis maps to y axis

    def test_old_row_major_label_is_still_read_as_column_major(self):
        calib = make_calibration()
        calib["depth_to_color_rotation_row_major"] = calib.pop("depth_to_color_rotation_col_major")
        np.testing.assert_array_equal(with_matrices(calib)["rotation"], np.eye(3))

    def test_translation_becomes_an_array(self):
        out = with_matrices(make_calibration((0.015, 0.0, -0.001)))
        np.testing.assert_allclose(out["translation"], [0.015, 0.0, -0.001])


class TestAlignDepthToColor:
    def test_identical_cameras_leave_depth_unchanged(self):
        raw = raw_plane()
        raw[20:30, 40:50] = 600
        raw[70:75, 100:110] = 0
        aligned = align_depth_to_color(raw, with_matrices(make_calibration()), fill=False)
        np.testing.assert_array_equal(aligned, raw)

    def test_sideways_offset_shifts_pixels_by_fx_times_offset_over_depth(self):
        # color camera 0.1 m to the right of the depth camera, wall at 1 m: shift = 100 * 0.1 / 1 = 10 px
        calib = with_matrices(make_calibration((0.1, 0.0, 0.0)))
        aligned = align_depth_to_color(raw_plane(), calib, fill=False)
        shift = int(FX * 0.1 / 1.0)
        assert (aligned[:, :shift] == 0).all()  # nothing was seen there
        assert (aligned[:, shift:] == 1000).all()

    def test_nearer_point_wins_when_two_land_on_the_same_pixel(self):
        # offset 0.05 m: a point at 0.5 m moves 10 px, a point at 1.0 m moves 5 px
        raw = np.zeros((H, W), dtype=np.uint16)
        raw[60, 45] = 500
        raw[60, 50] = 1000
        aligned = align_depth_to_color(raw, with_matrices(make_calibration((0.05, 0.0, 0.0))), fill=False)
        assert aligned[60, 55] == 500

    def test_accepts_depth_with_a_trailing_channel_axis(self):
        aligned = align_depth_to_color(raw_plane()[..., None], with_matrices(make_calibration()))
        assert aligned.shape == (H, W)


class TestFillCracks:
    def test_fills_a_single_missing_pixel_in_a_flat_area(self):
        depth = raw_plane()
        depth[50, 50] = 0
        assert fill_cracks(depth)[50, 50] == 1000

    def test_does_not_bridge_a_real_depth_step(self):
        depth = raw_plane()
        depth[:, W // 2 :] = 500  # near surface on the right
        depth[50, W // 2] = 0  # a missing pixel exactly on the step
        assert fill_cracks(depth)[50, W // 2] == 0

    def test_leaves_a_large_hole_alone(self):
        depth = raw_plane()
        depth[40:50, 40:50] = 0
        assert (fill_cracks(depth)[43:47, 43:47] == 0).all()


class TestDepthEdges:
    def test_marks_the_nearer_side_of_a_depth_step(self):
        depth = raw_plane(1000)
        depth[:, W // 2 :] = 500  # 50 cm nearer on the right
        edges = depth_edges(depth, depth_scale=0.001, median=0)
        assert edges[:, W // 2].all()  # the near surface's first column
        assert edges.sum() == H  # and nothing else

    def test_ignores_steps_smaller_than_the_threshold(self):
        depth = raw_plane(1000)
        depth[:, W // 2 :] = 950  # 5 cm step, threshold is 10 cm
        assert not depth_edges(depth, depth_scale=0.001, median=0).any()

    def test_does_not_treat_missing_depth_as_an_edge(self):
        depth = raw_plane(1000)
        depth[:, W // 2 :] = 0
        assert not depth_edges(depth, depth_scale=0.001, median=0).any()


class TestWarpMaskToColor:
    def test_each_pixel_moves_by_its_own_parallax(self):
        calib = with_matrices(make_calibration((0.1, 0.0, 0.0)))
        depth = np.zeros((H, W), dtype=np.uint16)
        depth[60, 40] = 1000  # 1.0 m: moves 10 px
        depth[60, 70] = 500  # 0.5 m: moves 20 px
        mask = depth > 0
        moved = warp_mask_to_color(mask, depth, calib)
        assert sorted(np.flatnonzero(moved[60]).tolist()) == [50, 90]

    def test_pixels_without_depth_are_dropped(self):
        depth = np.zeros((H, W), dtype=np.uint16)
        mask = np.ones((H, W), dtype=bool)
        assert not warp_mask_to_color(mask, depth, with_matrices(make_calibration())).any()
