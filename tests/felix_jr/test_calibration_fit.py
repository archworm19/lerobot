"""The calibration model, the fit (on synthetic clicks with known parameters) and the saved results."""

import json

import numpy as np
import pytest

from lerobot.felix_jr.calibration.fit import (
    FitConfig,
    fit,
    pixel_errors,
    report,
    rms,
    save_result,
    split_holdout,
)
from lerobot.felix_jr.calibration.model import (
    ARM_JOINTS,
    ARM_PREFIX,
    CalibrationModel,
    Observation,
    load_calibrated_model,
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


def test_the_fit_recovers_a_calibration_from_noisy_clicks_and_generalizes():
    model = make_model()
    rng = np.random.default_rng(1)
    params = parameter_set("left")
    theta_true = {p.name: float(rng.normal(0, 0.5 * p.sigma)) for p in params}
    theta_true["zero:head_tilt_joint"] = 0.3  # the head's home really is about 17 degrees down
    clicks = synthetic_clicks(model, theta_true, 80, seed=2, noise_px=1.5)
    train, held = split_holdout(clicks, 5)
    assert len(held) == 16

    uncalibrated = rms(pixel_errors(model, held, {}))
    result = fit(model, train, cfg=FitConfig(sigma_px=1.5))
    calibrated = rms(pixel_errors(model, held, result.theta))
    assert uncalibrated > 30
    assert calibrated < 3.5  # the click noise is 1.5 px: it generalizes to clicks it never saw
    assert rms(result.pixel_errors) < 3.0

    # the estimates are consistent with their own uncertainty: nearly every parameter is within 3 posterior sigmas
    sigma = result.sigma_post
    within = [abs(result.theta[p.name] - theta_true[p.name]) <= 3 * sigma[p.name] for p in params]
    assert np.mean(within) > 0.9
    # and the posterior is tighter than the prior for the parameters the clicks determine (head tilt zero, mount)
    assert sigma["zero:head_tilt_joint"] < 0.2 * parameter_set("left")[1].sigma
    assert result.cost_history[-1] < result.cost_history[0]


def test_priors_keep_parameters_the_clicks_cannot_determine_near_zero():
    model = make_model()
    params = parameter_set("left")
    clicks = synthetic_clicks(model, {"zero:head_tilt_joint": 0.3}, 6, seed=3, noise_px=1.0)
    result = fit(model, clicks, cfg=FitConfig(sigma_px=1.0))
    for p in params:
        assert abs(result.theta[p.name]) < 4 * p.sigma  # never runs away with few clicks


def test_a_mis_click_is_down_weighted_by_the_robust_loss():
    model = make_model()
    theta_true = {"zero:head_tilt_joint": 0.3, f"{ARM_PREFIX}zero:shoulder_lift": np.radians(1.5)}
    clicks = synthetic_clicks(model, theta_true, 40, seed=4, noise_px=1.0)
    bad = list(clicks)
    wrong = Observation(
        **{**clicks[7].__dict__, "pixel": (clicks[7].pixel[0] + 120.0, clicks[7].pixel[1] - 90.0)}
    )
    bad[7] = wrong
    robust = fit(model, bad, cfg=FitConfig(sigma_px=1.0, huber=2.0))
    greedy = fit(model, bad, cfg=FitConfig(sigma_px=1.0, huber=1e9))
    good_errors = lambda r: np.delete(r.pixel_errors, 7)  # noqa: E731
    assert rms(good_errors(robust)) < rms(good_errors(greedy))
    assert robust.pixel_errors[7] > 100  # it still sees the click as wrong instead of bending the model to it


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


def test_saved_calibration_rebuilds_a_model_that_reproduces_the_fit(tmp_path):
    model = make_model()
    theta_true = {
        "zero:head_tilt_joint": 0.25,
        "pos:fixed_head_camera_link:y": 0.01,
        f"{ARM_PREFIX}zero:elbow_flex": 0.03,
    }
    clicks = synthetic_clicks(model, theta_true, 30, seed=6, noise_px=0.5)
    cfg = FitConfig(sigma_px=0.5)
    result = fit(model, clicks, cfg=cfg)
    save_result(tmp_path, model, result, {"n": 6.0, "before": 40.0, "after": 1.0})
    text = report(model, clicks, result, {"n": 6.0, "before": 40.0, "after": 1.0}, cfg)
    assert "held out" in text and "`zero:head_tilt_joint`" in text and "Learned" in text

    calibrated, theta = load_calibrated_model(tmp_path)
    assert theta == result.theta
    for obs in clicks[:10]:
        np.testing.assert_allclose(
            calibrated.predict(obs, {}).uv, model.predict(obs, result.theta).uv, atol=1e-6
        )  # URDFs patched, no parameters left to apply
    saved = json.loads((tmp_path / "kinematic_calibration.json").read_text())
    assert saved["pixel_rms"]["holdout"]["after"] == 1.0
    assert set(saved["sigma_post"]) == {p.name for p in parameter_set("left")}
