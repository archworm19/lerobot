import numpy as np
import pytest

from lerobot.felix_jr.overlay import STATE_COLORS, draw_points, render_plan
from lerobot.felix_jr.scene import BLOCKED, FREE
from tests.felix_jr.conftest import H, W


class TestDrawPoints:
    def test_draws_on_a_copy(self, scene):
        image = scene.color
        out = draw_points(image, np.array([[50.0, 40.0]]))
        assert (image == 0).all()
        assert out.shape == image.shape and out.dtype == np.uint8
        assert (out[40, 50] == (255, 220, 0)).all()  # the marker's color at its centre

    def test_skips_nan_and_off_image_points(self, scene):
        uv = np.array([[np.nan, 5.0], [-10.0, 5.0], [W + 10.0, 5.0], [5.0, H + 10.0]])
        assert (draw_points(scene.color, uv) == 0).all()

    def test_uses_the_given_labels_without_crashing(self, scene):
        out = draw_points(scene.color, np.array([[30.0, 30.0], [90.0, 70.0]]), labels=["A", "B"])
        assert out.any()


class TestRenderPlan:
    def test_returns_an_annotated_copy(self, scene):
        waypoints = np.array([[-0.2, 0.0, 0.3], [0.2, 0.0, 0.5]])
        out = render_plan(scene, waypoints)
        assert out.shape == (H, W, 3) and out.dtype == np.uint8
        assert out.any() and (scene.color == 0).all()

    def test_colors_samples_by_state(self, scene):
        free = render_plan(scene, np.array([[-0.2, 0.0, 0.3], [0.2, 0.0, 0.5]]), legend=False)
        blocked = render_plan(scene, np.array([[0.3, 0.3, 0.8], [0.3, 0.3, 1.2]]), legend=False)

        def has_color(img, rgb):
            return bool((img == rgb).all(axis=-1).any())

        assert has_color(free, STATE_COLORS[FREE]) and not has_color(free, STATE_COLORS[BLOCKED])
        assert has_color(blocked, STATE_COLORS[BLOCKED])

    def test_accepts_a_precomputed_check(self, scene):
        waypoints = np.array([[-0.2, 0.0, 0.3], [0.2, 0.0, 0.5]])
        check = scene.segment_check(waypoints[:-1], waypoints[1:])
        again = render_plan(scene, waypoints, check=check)
        np.testing.assert_array_equal(render_plan(scene, waypoints), again)

    def test_waypoints_behind_the_camera_do_not_crash(self, scene):
        out = render_plan(scene, np.array([[0.0, 0.0, -0.5], [0.1, 0.0, 0.5], [0.2, 0.0, 0.6]]))
        assert out.shape == (H, W, 3)

    def test_rejects_bad_waypoint_shapes(self, scene):
        with pytest.raises(ValueError):
            render_plan(scene, np.array([[0.0, 0.0, 0.5]]))  # one point is not a path
        with pytest.raises(ValueError):
            render_plan(scene, np.zeros((3, 2)))
