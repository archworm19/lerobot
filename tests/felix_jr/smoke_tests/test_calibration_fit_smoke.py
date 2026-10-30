"""The calibration model, the fit (on synthetic clicks with known parameters) and the saved results."""

import json

import numpy as np

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
    parameter_set,
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
