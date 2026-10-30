"""The `XleRobotModel` fit: synthetic clicks of both arms with known parameters, sessions, holdout, saving."""

import numpy as np
import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("scipy")

from lerobot.felix_jr.calibration import xle_fit  # noqa: E402
from lerobot.felix_jr.calibration.model import ARM_JOINTS, CalibrationModel, Observation, save_observations  # noqa: E402
from lerobot.felix_jr.calibration.xle_fit import Click, FitConfig, Session, calibrate, fit, rms  # noqa: E402
from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel, default_parameters  # noqa: E402

INTRINSICS = {"fx": 610.97, "fy": 610.01, "ppx": 315.95, "ppy": 247.03, "width": 640, "height": 480}
SIGNS = {"head_neck": -1, "head_eyes": 1}
REFS = {"head_neck": 1520, "head_eyes": 1324}
# A small table keeps the fits fast: the head's zeros and camera rotation, and each arm's mount position.
NAMES = [
    "zero:head_pan_joint",
    "zero:head_tilt_joint",
    *(f"rot:fixed_head_camera_link:{a}" for a in "xyz"),
    *(f"pos:fixed_Base:{a}" for a in "xyz"),
    *(f"pos:fixed_Base_2:{a}" for a in "xyz"),
]
TRUE = {
    "zero:head_pan_joint": np.radians(-8.0),
    "zero:head_tilt_joint": np.radians(30.0),
    "rot:fixed_head_camera_link:z": np.radians(3.0),
    "pos:fixed_Base:y": 0.02,
    "pos:fixed_Base_2:x": -0.03,
}


def make_model(names=NAMES):
    params = [p for p in default_parameters() if p.name in names]
    return XleRobotModel.from_urdf(head_signs=SIGNS, head_refs=REFS, intrinsics=INTRINSICS, parameters=params)


def synthetic_clicks(model, arm, n, seed, noise_px=0.5, session=None):
    """Clicks at random arm and head poses whose true pinch point is in view, with Gaussian click noise."""
    rng = np.random.default_rng(seed)
    clicks = []
    while len(clicks) < n:
        pose = {f"{j}.pos": float(rng.uniform(-25, 25)) for j in ARM_JOINTS}
        ticks = {
            "head_neck": int(REFS["head_neck"] + rng.integers(-200, 200)),
            "head_eyes": int(REFS["head_eyes"] + rng.integers(-150, 150)),
        }
        truth = {k: v for k, v in TRUE.items() if k in {p.name for p in model.parameters}}
        pred = model.predict(arm, model.joint_vector({arm: pose}, ticks), truth)
        if pred.z > 0.1 and model.in_view(pred.uv, margin=20):
            pixel = pred.uv + rng.normal(0, noise_px, 2)
            clicks.append(
                Click(
                    arm,
                    pose,
                    ticks,
                    (float(pixel[0]), float(pixel[1])),
                    session=session or arm,
                    index=len(clicks),
                )
            )
    return clicks


def test_one_fit_learns_the_shared_head_and_both_arms():
    model = make_model()
    clicks = synthetic_clicks(model, "left", 15, seed=0) + synthetic_clicks(model, "right", 15, seed=1)
    uncalibrated = rms(xle_fit.pixel_errors(model, clicks, np.zeros(len(model.parameters))))
    result = fit(model, clicks, FitConfig(sigma_px=0.5))
    assert uncalibrated > 100 and rms(result.pixel_errors) < 1.0
    assert result.free.all()  # clicks of both arms: everything is fitted
    theta = dict(zip(NAMES, result.theta, strict=True))
    assert theta["zero:head_tilt_joint"] == pytest.approx(TRUE["zero:head_tilt_joint"], abs=np.radians(1.0))
    assert theta["pos:fixed_Base:y"] == pytest.approx(0.02, abs=0.005)
    assert theta["pos:fixed_Base_2:x"] == pytest.approx(-0.03, abs=0.005)
    assert np.all(result.sigma_post <= np.array([p.sigma for p in model.parameters]) + 1e-12)
    np.testing.assert_array_equal(
        model.estimates, np.zeros(len(model.parameters))
    )  # fit does not touch the model


def test_an_arm_without_clicks_keeps_its_starting_values():
    model = make_model()
    start = np.zeros(len(model.parameters))
    start[NAMES.index("pos:fixed_Base_2:z")] = 0.007
    result = fit(model, synthetic_clicks(model, "left", 10, seed=2), FitConfig(sigma_px=0.5), theta0=start)
    right = np.array([p.arm == "right" for p in model.parameters])
    assert not result.free[right].any()
    np.testing.assert_array_equal(result.theta[right], start[right])
    sigma = np.array([p.sigma for p in model.parameters])
    np.testing.assert_allclose(result.sigma_post[right], sigma[right])  # their prior, untouched


def test_a_mis_click_is_down_weighted_by_the_huber_loss():
    model = make_model(NAMES[:5])  # head only
    clicks = synthetic_clicks(model, "left", 15, seed=3)
    clicks[4].pixel = (clicks[4].pixel[0] + 120.0, clicks[4].pixel[1] - 90.0)
    robust = fit(model, clicks, FitConfig(sigma_px=0.5, huber=2.0))
    greedy = fit(model, clicks, FitConfig(sigma_px=0.5, huber=1e9))
    good = np.arange(len(clicks)) != 4
    assert rms(robust.pixel_errors[good]) < rms(greedy.pixel_errors[good])
    assert robust.pixel_errors[4] > 100  # it still sees the click as wrong instead of bending the model to it


def test_calibrate_validates_per_session_and_the_result_survives_saving(tmp_path):
    model = make_model()
    clicks = synthetic_clicks(model, "left", 12, seed=4, session="a") + synthetic_clicks(
        model, "right", 12, seed=5, session="b"
    )
    final, holdout = calibrate(model, clicks, FitConfig(sigma_px=0.5), holdout_every=4)
    assert set(holdout) == {"a", "b"} and all(
        h["n"] == 3 and h["after"] < 2.0 < h["before"] for h in holdout.values()
    )
    np.testing.assert_array_equal(model.estimates, final.theta)
    assert model.fit_info["holdout"] == holdout and model.fit_info["sessions"] == ["a", "b"]
    model.save(tmp_path / "calibration.urdf")
    loaded = XleRobotModel.load(tmp_path / "calibration.urdf")
    np.testing.assert_array_equal(loaded.estimates, final.theta)
    np.testing.assert_allclose(xle_fit.pixel_errors(loaded, clicks), final.pixel_errors)


def test_split_holdout_counts_within_each_session():
    clicks = [Click("left", {}, {}, (0.0, 0.0), session=s, index=i) for s in ("a", "b") for i in range(6)]
    train, held = xle_fit.split_holdout(clicks, 3)
    assert [(c.session, c.index) for c in held] == [("a", 0), ("a", 3), ("b", 0), ("b", 3)]
    assert len(train) == 8
    assert xle_fit.split_holdout(clicks, 0) == (clicks, [])


def test_sessions_with_different_head_references_are_refused(tmp_path):
    a = Session(tmp_path / "a", "left", [], SIGNS, REFS, INTRINSICS)
    b = Session(tmp_path / "b", "right", [], SIGNS, {**REFS, "head_neck": 1605}, INTRINSICS)
    assert xle_fit.model_for_sessions([a, a]).head_refs == REFS
    with pytest.raises(ValueError, match="head_refs"):
        xle_fit.model_for_sessions([a, b])


def test_a_version_1_session_folder_is_read_as_clicks(tmp_path):
    v1 = CalibrationModel("right", INTRINSICS, SIGNS, REFS)
    pose = {f"{j}.pos": 5.0 for j in ARM_JOINTS}
    observations = [
        Observation(
            arm=pose,
            head_ticks={"head_neck": 1500, "head_eyes": 1400},
            pixel=(10.0, 20.0),
            image="images/000.png",
        ),
        Observation(arm=pose, head_ticks=REFS, pixel=(30.0, 40.0), depth_m=0.4, index=1),
    ]
    (tmp_path / "cal_right").mkdir()
    save_observations(tmp_path / "cal_right" / "observations.json", v1, observations)
    session = xle_fit.load_v1_session(tmp_path / "cal_right")
    assert session.arm == "right" and session.head_refs == REFS and session.intrinsics == INTRINSICS
    first, second = session.clicks
    assert first.arm == "right" and first.arm_pose == pose and first.pixel == (10.0, 20.0)
    assert first.image == "images/000.png" and first.session == "cal_right"
    assert second.depth_m == 0.4 and second.index == 1
