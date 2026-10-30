"""`XleRobotModel` and the merged URDF: agreement with version 1, corrections, export, saving, derivatives."""

import json
from pathlib import Path

import numpy as np
import pytest

pin = pytest.importorskip("pinocchio")

from lerobot.felix_jr.calibration.merge_urdf import (  # noqa: E402
    MERGED_URDF,
    SO101_JOINTS as ARM_JOINTS,
    build_merged_urdf,
)
from lerobot.felix_jr.calibration.xle_robot_model import (  # noqa: E402
    XleRobotModel,
    default_parameters,
    pinch_frame,
)

INTRINSICS = {"fx": 610.97, "fy": 610.01, "ppx": 315.95, "ppy": 247.03, "width": 640, "height": 480}
SIGNS = {"head_neck": -1, "head_eyes": 1}
REFS = {"head_neck": 1520, "head_eyes": 1324}
V1_REFERENCE = Path(__file__).parent / "reference" / "v1_reference.json"


@pytest.fixture(scope="module")
def model():
    return XleRobotModel.from_urdf(head_signs=SIGNS, head_refs=REFS, intrinsics=INTRINSICS)


def random_readings(rng):
    arm = {f"{j}.pos": float(rng.uniform(-40, 40)) for j in ARM_JOINTS}
    ticks = {
        "head_neck": int(REFS["head_neck"] + rng.integers(-300, 300)),
        "head_eyes": int(REFS["head_eyes"] + rng.integers(-250, 250)),
    }
    return arm, ticks


def test_the_committed_merged_urdf_is_what_the_script_builds():
    assert MERGED_URDF.read_text() == build_merged_urdf()


def test_the_merged_urdf_has_both_arms_no_meshes_and_a_fixed_root(model):
    text = MERGED_URDF.read_text()
    assert "<mesh" not in text and 'type="floating"' not in text
    names = set(model.nominal_model.names)
    assert {"left_shoulder_pan", "right_wrist_roll", "head_pan_joint", "head_tilt_joint"} <= names
    assert "Rotation_L" not in names  # the old arm is gone


def test_placements_rebuilt_from_the_urdf_origins_match_pinocchios(model):
    model.check_structure()


def test_it_reproduces_version_1s_predictions_with_and_without_corrections(model):
    """Version 1 (`CalibrationModel`: two URDFs joined at runtime) produced the first calibrations; its predictions
    for random readings and corrections were frozen in `reference/v1_reference.json` before it was deleted."""
    reference = json.loads(V1_REFERENCE.read_text())
    assert reference["head_refs"] == REFS and reference["intrinsics"] == INTRINSICS
    for case in reference["cases"]:
        arm, theta = case["arm"], case["theta"]
        q = model.joint_vector({arm: case["arm_pose"]}, case["head_ticks"])
        got = model.predict(arm, q, theta)
        np.testing.assert_allclose(got.uv, case["uv"], atol=1e-6)
        assert got.z == pytest.approx(case["z"], abs=1e-9)
        np.testing.assert_allclose(got.point, case["pinch_point"], atol=1e-9)
        np.testing.assert_allclose(model.camera_pose(q, theta), case["camera_pose"], atol=1e-9)
        assert got.visibility == pytest.approx(case["hinge_cosine"], abs=1e-9)
    assert any(case["theta"] for case in reference["cases"])  # some cases have corrections


def test_predict_many_is_predict_at_each_configuration(model):
    rng = np.random.default_rng(4)
    theta = {p.name: float(rng.normal(0, p.sigma)) for p in model.parameters}
    qs = [
        model.joint_vector({"right": arm}, ticks) for arm, ticks in (random_readings(rng) for _ in range(4))
    ]
    uv, z = model.predict_many("right", qs, theta)
    for i, q in enumerate(qs):
        single = model.predict("right", q, theta)
        np.testing.assert_allclose(uv[i], single.uv)
        assert z[i] == pytest.approx(single.z)


def test_a_fixed_joint_correction_moves_everything_below_it(model):
    q = model.joint_vector({"left": {}, "right": {}})
    left, right = model.pinch_point("left", q, {}), model.pinch_point("right", q, {})
    # fixed_Base is a fixed joint (a frame in pinocchio) whose parent, the chassis, is aligned with the root frame
    np.testing.assert_allclose(
        model.pinch_point("left", q, {"pos:fixed_Base:x": 0.01}) - left, [0.01, 0.0, 0.0], atol=1e-12
    )
    np.testing.assert_allclose(model.pinch_point("right", q, {"pos:fixed_Base:x": 0.01}), right, atol=1e-12)
    camera = model.camera_pose(q, {})
    moved = model.camera_pose(q, {"rot:fixed_head_camera_link:z": 0.05})
    assert np.linalg.norm(moved[:3, :3] - camera[:3, :3]) > 0.01


def test_rotations_are_rotation_vectors(model):
    q = model.joint_vector()
    w = np.array([0.02, -0.03, 0.01])
    theta = {f"rot:fixed_head_camera_link:{a}": v for a, v in zip("xyz", w, strict=True)}
    before, after = (
        model.frame_pose("head_camera_link", q, {}),
        model.frame_pose("head_camera_link", q, theta),
    )
    np.testing.assert_allclose(after[:3, :3], before[:3, :3] @ pin.exp3(w), atol=1e-12)


def test_a_zero_correction_is_the_same_as_turning_the_joint(model):
    q = model.joint_vector({"left": {"elbow_flex.pos": 10.0}})
    turned = model.joint_vector({"left": {"elbow_flex.pos": 12.0}})
    np.testing.assert_allclose(
        model.pinch_point("left", q, {"zero:left_elbow_flex": np.radians(2.0)}),
        model.pinch_point("left", turned, {}),
        atol=1e-12,
    )


def test_explicit_estimates_never_leak_into_later_calls(model):
    q = model.joint_vector({"left": {"shoulder_pan.pos": 5.0}}, {"head_neck": 1600, "head_eyes": 1400})
    model.set_estimates({"zero:head_tilt_joint": 0.3})
    current = model.predict("left", q).uv
    model.predict("left", q, {"zero:head_tilt_joint": -0.2, "pos:fixed_Base:y": 0.05})
    np.testing.assert_allclose(model.predict("left", q).uv, current)
    np.testing.assert_allclose(model.predict("left", q, model.estimates).uv, current)
    model.set_estimates({})


def test_the_exported_urdf_bakes_in_the_estimates(model):
    rng = np.random.default_rng(2)
    theta = {p.name: float(rng.normal(0, p.sigma)) for p in model.parameters}
    exported = XleRobotModel(model.to_urdf(theta), SIGNS, REFS, INTRINSICS)
    for _ in range(3):
        arm_pose, ticks = random_readings(rng)
        q = model.joint_vector({"left": arm_pose, "right": arm_pose}, ticks)
        for arm in ("left", "right"):
            np.testing.assert_allclose(
                exported.predict(arm, q, {}).uv, model.predict(arm, q, theta).uv, atol=1e-6
            )


def test_save_and_load_give_back_the_same_model(model, tmp_path):
    rng = np.random.default_rng(3)
    original = XleRobotModel(
        model.urdf,
        SIGNS,
        REFS,
        INTRINSICS,
        default_parameters(["left"], {"zero:head_tilt_joint": (0.3, 0.05)}),
        floor_z=0.7,
    )
    original.set_estimates(rng.normal(0, 0.01, len(original.parameters)))
    original.covariance = np.eye(len(original.parameters)) * 1e-4
    original.fit_info = {"sessions": ["cal_left"]}
    original.save(tmp_path / "calibration.urdf")
    loaded = XleRobotModel.load(tmp_path / "calibration.urdf")
    assert loaded.parameters == original.parameters and loaded.floor_z == 0.7
    assert loaded.fit_info == original.fit_info and loaded.head_refs == REFS
    np.testing.assert_array_equal(loaded.estimates, original.estimates)
    np.testing.assert_array_equal(loaded.covariance, original.covariance)
    for is_frame, index, _ in loaded._all_placements:  # the URDF's own values came back exactly
        models = (loaded.nominal_model, original.nominal_model)
        got, expected = (m.frames[index].placement if is_frame else m.jointPlacements[index] for m in models)
        np.testing.assert_array_equal(got.homogeneous, expected.homogeneous)
    q = model.joint_vector({"left": {"wrist_flex.pos": 20.0}})
    np.testing.assert_allclose(loaded.predict("left", q).uv, original.predict("left", q).uv)
    # any URDF reader gets the calibrated robot: pinocchio ignores the extra element
    plain = XleRobotModel((tmp_path / "calibration.urdf").read_text(), SIGNS, REFS, INTRINSICS)
    np.testing.assert_allclose(plain.predict("left", q, {}).uv, original.predict("left", q).uv, atol=1e-6)


def test_the_default_parameters_cover_the_head_once_and_each_arm():
    for arm in ("left", "right"):
        params = default_parameters([arm])
        assert len(params) == 43 and len({p.name for p in params}) == 43
        assert {p.arm for p in params} == {None, arm}
    assert len(default_parameters()) == 14 + 2 * 29  # the head and camera once, then each arm
    assert not any(
        "top_base_link" in p.name for p in default_parameters()
    )  # the head tower: see the docstring
    overridden = default_parameters(["left"], {"zero:head_tilt_joint": (0.3, 0.05)})[1]
    assert (overridden.mean, overridden.sigma) == (0.3, 0.05)


def test_unknown_parameters_and_estimates_are_refused(model):
    with pytest.raises(KeyError):
        model.predict("left", model.joint_vector(), {"zero:no_such_joint": 0.1})
    with pytest.raises(ValueError):
        model.set_estimates(np.zeros(3))


def test_head_angles_and_pinch_frames():
    m = XleRobotModel.from_urdf(head_signs=SIGNS, head_refs=REFS)
    step = 2 * np.pi / 4096
    angles = m.head_angles({"head_neck": 1520 + 100, "head_eyes": 1324 + 100})
    assert angles == pytest.approx({"head_pan_joint": -100 * step, "head_tilt_joint": 100 * step})
    assert pinch_frame("right") == "right_gripper_frame_link"
    assert m.nominal("pos:left_gripper_frame_joint:z") == pytest.approx(-0.0981274)


@pytest.mark.parametrize("arm", ["left", "right"])
def test_the_pixel_jacobian_matches_finite_differences(model, arm):
    rng = np.random.default_rng(5)
    theta = np.array([rng.normal(0, p.sigma) for p in model.parameters])
    theta[[p.kind == "rot" for p in model.parameters]] *= (
        10  # large rotations: the exp map's Jacobian matters
    )
    theta[[p.name == "zero:head_tilt_joint" for p in model.parameters]] = 0.58  # look down at the arms
    joints = ["head_pan_joint", "head_tilt_joint", *(f"{arm}_{j}" for j in ARM_JOINTS)]
    while True:
        arm_pose, ticks = random_readings(rng)
        q = model.joint_vector({arm: arm_pose}, ticks)
        if model.predict(arm, q, theta).z > 0.1:
            break
    d_parameters, d_joints = model.pixel_jacobian(arm, q, theta, joints)
    step = 1e-6
    for i in range(len(theta)):
        e = np.zeros_like(theta)
        e[i] = step
        numeric = (model.predict(arm, q, theta + e).uv - model.predict(arm, q, theta - e).uv) / (2 * step)
        np.testing.assert_allclose(d_parameters[:, i], numeric, atol=1e-4, err_msg=model.parameters[i].name)
    for i, joint in enumerate(joints):
        e = np.zeros_like(q)
        e[model.nominal_model.joints[model.nominal_model.getJointId(joint)].idx_q] = step
        numeric = (model.predict(arm, q + e, theta).uv - model.predict(arm, q - e, theta).uv) / (2 * step)
        np.testing.assert_allclose(d_joints[:, i], numeric, atol=1e-4, err_msg=joint)
    many = model.pixel_jacobian_many(arm, [q, q], theta)
    np.testing.assert_array_equal(many[1], d_parameters)
