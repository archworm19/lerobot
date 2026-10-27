from pathlib import Path

import numpy as np
import pytest

from lerobot.felix_jr.calibration.sensitivity import (
    ARMS,
    DEFAULT_INTRINSICS,
    TOLERANCES,
    load_camera,
    make_configurations,
    monte_carlo,
    parameters,
    project_targets,
    sensitivities,
)
from lerobot.felix_jr.calibration.urdf_kinematics import UrdfTree

XLEROBOT_URDF = Path(__file__).resolve().parents[2] / "src/lerobot/felix_jr/xlerobot.urdf"


@pytest.fixture(scope="module")
def setup():
    tree = UrdfTree(XLEROBOT_URDF)
    camera = load_camera(None)
    configs = make_configurations(tree, camera, "left", per_head_pose=1, seed=3)
    return tree, camera, configs


def by_name(params, *names):
    return [p for p in params if p.name in names]


def test_parameters_cover_head_and_arm_and_name_real_joints(setup):
    tree, _, _ = setup
    for arm in ARMS:
        params = parameters(tree, arm)
        assert {p.group for p in params} == {"head", "arm"}
        assert len({p.name for p in params}) == len(params)
        assert all(p.kind in TOLERANCES for p in params)


def test_configurations_put_the_jaw_pivot_in_view(setup):
    tree, camera, configs = setup
    assert len(configs) == 9  # 3 tilts x 3 pans, one arm pose each
    for config in configs:
        uv = project_targets(tree, camera, "left", config, None)
        assert camera.in_view(uv[list(ARMS["left"]["targets"]).index("jaw pivot")])


def test_a_head_tilt_error_shifts_the_image_by_about_fx_per_radian(setup):
    tree, camera, configs = setup
    params = by_name(parameters(tree, "left"), "zero:head_tilt_joint")
    sens = sensitivities(tree, camera, "left", configs, params)
    jaw = sens[0, :, -1]
    # a pure camera rotation moves image points by about fx per radian (more away from the image center)
    assert np.nanmin(jaw) > 0.9 * DEFAULT_INTRINSICS["fx"]
    assert np.nanmax(jaw) < 1.6 * DEFAULT_INTRINSICS["fx"]


def test_a_translation_shifts_by_about_fx_over_depth(setup):
    tree, camera, configs = setup
    params = by_name(parameters(tree, "left"), "pos:fixed_Base:y")
    sens = sensitivities(tree, camera, "left", configs, params)  # px per meter
    assert np.nanmax(sens) < camera.fx / 0.15  # nothing in the configurations is nearer than 0.15 m
    assert np.nanmin(sens) > 0


def test_head_tower_and_arm_mount_translations_are_indistinguishable(setup):
    # both just move the arm relative to the camera: a calibration can not tell them apart
    tree, camera, configs = setup
    params = by_name(parameters(tree, "left"), "pos:fixed_Base:y", "pos:fixed_top_base_link:y")
    sens = sensitivities(tree, camera, "left", configs, params)
    np.testing.assert_allclose(sens[0], sens[1], rtol=1e-3)


def test_monte_carlo_is_zero_for_no_parameters_and_positive_otherwise(setup):
    tree, camera, configs = setup
    none = monte_carlo(tree, camera, "left", configs, [], samples=3)
    assert np.nanmax(none) == 0.0
    some = monte_carlo(
        tree, camera, "left", configs, by_name(parameters(tree, "left"), "zero:head_tilt_joint"), 5
    )
    assert np.nanmean(some) > 0
