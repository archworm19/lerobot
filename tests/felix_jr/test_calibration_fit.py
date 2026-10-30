"""The calibration model, the fit (on synthetic clicks with known parameters) and the saved results."""

import numpy as np
import pytest

from lerobot.felix_jr.calibration.fit import (
    FitConfig,
    fit,
    observation_residuals,
    pixel_errors,
    rms,
    split_holdout,
    vector_to_theta,
)
from lerobot.felix_jr.calibration.model import (
    ARM_JOINTS,
    ARM_PREFIX,
    CalibrationModel,
    Observation,
    Param,
    load_observations,
    parameter_set,
    save_observations,
)

INTRINSICS = {"fx": 610.97, "fy": 610.01, "ppx": 315.95, "ppy": 247.03, "width": 640, "height": 480}
SIGNS = {"head_neck": -1, "head_eyes": 1}
REFS = {"head_neck": 1520, "head_eyes": 1324}


def make_model(arm="left"):
    return CalibrationModel(arm, INTRINSICS, SIGNS, REFS)


def observation(arm_deg=0.0, neck=1520, eyes=1324, **joints):
    pose = {f"{j}.pos": arm_deg for j in ARM_JOINTS}
    pose.update({f"{k}.pos": v for k, v in joints.items()})
    return Observation(arm=pose, head_ticks={"head_neck": neck, "head_eyes": eyes}, pixel=(0.0, 0.0))


@pytest.mark.parametrize("arm", ["left", "right"])
def test_the_so101_arm_points_forward_and_up_from_its_base_at_the_center_pose(arm):
    model = make_model(arm)
    assert model.alignment_rms < 0.012  # the two URDFs' shoulder geometry differs by about this much
    center = {f"{j}.pos": 0.0 for j in ARM_JOINTS}
    base = model.head_tree.link_pose(model.base_link)[:3, 3]
    forward, up = -(model.tip_in_root(center, {}) - base)[0], (model.tip_in_root(center, {}) - base)[2]
    assert 0.30 < forward < 0.40 and 0.20 < up < 0.26  # robot forward is -x in the root frame


def test_a_positive_pan_swings_both_arms_towards_the_right_as_seen_in_the_wiggle_frames():
    for arm in ("left", "right"):
        model = make_model(arm)
        before = model.tip_in_root({f"{j}.pos": 0.0 for j in ARM_JOINTS}, {})
        after = model.tip_in_root({**{f"{j}.pos": 0.0 for j in ARM_JOINTS}, "shoulder_pan.pos": 10.0}, {})
        assert after[1] > before[1]  # +y is the robot's right


def test_head_angles_follow_the_saved_signs():
    model = make_model()
    step = 2 * np.pi / 4096
    angles = model.head_angles({"head_neck": 1520 + 100, "head_eyes": 1324 + 100})
    assert angles["head_tilt_joint"] == pytest.approx(100 * step)  # sign +1: more ticks = tilted further down
    assert angles["head_pan_joint"] == pytest.approx(-100 * step)  # sign -1
    assert model.head_angles(REFS) == {"head_pan_joint": 0.0, "head_tilt_joint": 0.0}


def test_predict_is_the_projection_of_the_tip():
    model = make_model()
    theta = {
        "zero:head_tilt_joint": 0.3,
        f"{ARM_PREFIX}zero:elbow_flex": 0.02,
        "pos:fixed_head_camera_link:y": 0.01,
    }
    obs = observation(5.0, neck=1600, eyes=1400)
    pred = model.predict(obs, theta)
    tip = model.tip_in_root(obs.arm, theta)
    expected = model.camera.project(model.camera_pose(obs.head_ticks, theta), tip[None])[0]
    np.testing.assert_allclose(pred.uv, expected)
    assert pred.z == pytest.approx(pred.tip_cam[2])


def test_every_parameter_is_wired_into_the_prediction():
    model = make_model()
    obs = observation(15.0, neck=1500, eyes=1500)
    theta0 = {"zero:head_tilt_joint": 0.3}
    base = model.predict(obs, theta0).uv
    for p in parameter_set("left"):
        moved = model.predict(obs, {**theta0, p.name: p.sigma}).uv
        assert np.linalg.norm(moved - base) > 1e-4, f"{p.name} does not change the prediction"
    names = [p.name for p in parameter_set("left")]
    assert len(set(names)) == len(names)
    assert parameter_set("right")[0].name == "zero:head_pan_joint"


def synthetic_clicks(model, theta_true, n, seed, noise_px):
    """Observations at random arm and head poses whose true tip is in view, with Gaussian click noise."""
    rng = np.random.default_rng(seed)
    observations = []
    while len(observations) < n:
        arm = {f"{j}.pos": float(rng.uniform(-25, 25)) for j in ARM_JOINTS}
        arm["shoulder_pan.pos"] += 10.0
        ticks = {
            "head_neck": int(REFS["head_neck"] + rng.integers(-300, 300)),
            "head_eyes": int(REFS["head_eyes"] + rng.integers(-250, 250)),
        }
        obs = Observation(arm=arm, head_ticks=ticks, pixel=(0.0, 0.0))
        pred = model.predict(obs, theta_true)
        if pred.z > 0.1 and model.in_view(pred.uv, margin=20):
            obs.pixel = tuple(pred.uv + rng.normal(0, noise_px, 2))
            obs.index = len(observations)
            observations.append(obs)
    return observations


def test_priors_keep_parameters_the_clicks_cannot_determine_near_zero():
    model = make_model()
    params = parameter_set("left")
    clicks = synthetic_clicks(model, {"zero:head_tilt_joint": 0.3}, 6, seed=3, noise_px=1.0)
    result = fit(model, clicks, cfg=FitConfig(sigma_px=1.0))
    for p in params:
        assert abs(result.theta[p.name]) < 4 * p.sigma  # never runs away with few clicks


def test_observations_round_trip_through_json(tmp_path):
    model = make_model()
    clicks = synthetic_clicks(model, {}, 3, seed=5, noise_px=0.0)
    clicks[0].depth_m, clicks[0].image = 0.42, "000.png"
    save_observations(tmp_path / "observations.json", model, clicks)
    model2, clicks2 = load_observations(tmp_path / "observations.json")
    assert [c.arm for c in clicks2] == [c.arm for c in clicks]
    assert clicks2[0].depth_m == 0.42 and clicks2[0].image == "000.png"
    assert model2.head_refs == REFS and model2.arm == "left"
    np.testing.assert_allclose(model2.T, model.T)


def test_rms_ignores_infinite_values_and_is_nan_when_nothing_is_left():
    assert rms(np.array([3.0, 4.0, np.inf])) == pytest.approx(np.sqrt(12.5))
    assert np.isnan(rms(np.array([np.inf, np.nan])))


def test_split_holdout_holds_out_every_nth_click_and_nothing_overlaps():
    clicks = [observation() for _ in range(12)]
    for i, c in enumerate(clicks):
        c.index = i
    train, held = split_holdout(clicks, 5)
    assert [c.index for c in held] == [0, 5, 10]
    assert sorted(c.index for c in train + held) == list(range(12))
    train, held = split_holdout(clicks, 0)
    assert len(train) == 12 and held == []


def test_vector_to_theta_names_each_value_and_refuses_a_wrong_length():
    params = [Param("zero:a", 0.1, "head"), Param("pos:b:x", 0.01, "mount")]
    assert vector_to_theta(params, np.array([0.5, -0.25])) == {"zero:a": 0.5, "pos:b:x": -0.25}
    with pytest.raises(ValueError):
        vector_to_theta(params, np.array([0.5]))


def test_params_know_their_units():
    angle, length = Param("zero:head_tilt_joint", 0.1, "head"), Param("pos:fixed_Base:x", 0.01, "mount")
    assert angle.angular and angle.unit == "deg" and angle.display(np.pi) == pytest.approx(180.0)
    assert not length.angular and length.unit == "mm" and length.display(0.02) == pytest.approx(20.0)
    assert Param(f"{ARM_PREFIX}rot:x:roll", 0.1, "arm").angular  # the arm prefix does not hide the kind


def test_prior_overrides_replace_only_the_named_parameter():
    plain = {p.name: p for p in parameter_set("left")}
    over = {p.name: p for p in parameter_set("left", {"zero:head_tilt_joint": (0.3, 0.05)})}
    assert over["zero:head_tilt_joint"].mean == 0.3 and over["zero:head_tilt_joint"].sigma == 0.05
    assert over["zero:head_tilt_joint"].group == plain["zero:head_tilt_joint"].group
    assert {n: p for n, p in over.items() if n != "zero:head_tilt_joint"} == {
        n: p for n, p in plain.items() if n != "zero:head_tilt_joint"
    }


def test_residuals_are_in_sigmas_and_vanish_at_the_truth():
    model = make_model()
    theta = {"zero:head_tilt_joint": 0.2}
    clicks = synthetic_clicks(model, theta, 4, seed=7, noise_px=0.0)
    assert np.abs(observation_residuals(model, clicks, theta, FitConfig())).max() < 1e-9
    assert np.abs(pixel_errors(model, clicks, theta)).max() < 1e-9
    off = observation_residuals(model, clicks, {}, FitConfig(sigma_px=2.0))
    wider = observation_residuals(model, clicks, {}, FitConfig(sigma_px=4.0))
    np.testing.assert_allclose(off, 2 * wider)  # a noisier click is trusted less
    assert off.shape == (4, 2)
    assert observation_residuals(model, clicks, {}, FitConfig(use_depth=True)).shape == (4, 3)


def test_a_fit_of_one_parameter_to_exact_clicks_finds_it_and_gets_surer():
    model = make_model()
    param = Param("zero:head_tilt_joint", 0.3, "head")
    clicks = synthetic_clicks(model, {param.name: 0.2}, 6, seed=8, noise_px=0.0)
    result = fit(model, clicks, params=[param], cfg=FitConfig(sigma_px=0.5))
    assert result.theta[param.name] == pytest.approx(0.2, abs=2e-3)
    assert result.sigma_post[param.name] < 0.05 * param.sigma
    assert result.cost_history[-1] <= result.cost_history[0]
    assert rms(result.pixel_errors) < 1.0


def test_a_fit_without_clicks_is_refused():
    with pytest.raises(ValueError, match="no observations"):
        fit(make_model(), [])
