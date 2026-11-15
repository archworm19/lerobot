"""`calibration.gripper_sweep`: the moving finger's opening from its tip in a wrist frame (synthetic frames: the sim's
fingers painted in the fingers' color on the wood's; skipped without mujoco, pinocchio or the meshes)."""

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("pinocchio")

from lerobot.felix_jr.calibration import gripper_sweep as sweep  # noqa: E402
from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel  # noqa: E402
from lerobot.felix_jr.harness import api  # noqa: E402
from lerobot.felix_jr.sim2real import robot  # noqa: E402
from lerobot.felix_jr.sim2real.assets import is_cached  # noqa: E402
from lerobot.felix_jr.sim2real.view import RULER2_POSE  # noqa: E402

pytestmark = pytest.mark.skipif(
    not is_cached() or robot.FINGER_STL.read_bytes()[: len(robot.LFS_POINTER)] == robot.LFS_POINTER,
    reason="the SO-101 model is not cached, or the finger mesh not pulled",
)
WOOD, FINGER = (
    (205, 160, 120),
    (170, 190, 230),
)  # RGB: a warm wood, the fingers' cool white (as the wrist cameras see)


@pytest.fixture(scope="module")
def model():
    return XleRobotModel.load(api.MODEL)


@pytest.fixture(scope="module")
def fingers(model):
    renderer = sweep.FingerRenderer(model, "left")
    yield renderer
    renderer.close()


def _frame(fingers, pose: dict[str, float], opening_deg: float, wood=WOOD, finger=FINGER) -> np.ndarray:
    """A frame of the sim's fingers, the jaws `opening_deg` open, in the fingers' color on the wood's."""
    fingers.set_arm(pose)
    fixed, moving = fingers.masks(opening_deg)
    image = np.empty(fixed.shape + (3,), np.uint8)
    image[:] = wood
    image[fixed | moving] = finger
    rng = np.random.default_rng(0)  # a little noise: real colors spread
    return np.clip(image + rng.normal(0, 3, image.shape), 0, 255).astype(np.uint8)


def test_the_colors_are_told_apart():
    image = np.array([[WOOD, FINGER, (20, 20, 20)]], np.uint8).repeat(9, axis=0).repeat(9, axis=1)
    assert sweep.finger_pixels(image)[:, 9:18].all() and not sweep.finger_pixels(image)[:, :9].any()
    assert (
        sweep.wood_pixels(image)[:, :9].all() and not sweep.wood_pixels(image)[:, 9:].any()
    )  # dark: neither


@pytest.mark.parametrize("error_deg", [-6.0, 0.0, 8.0])
def test_the_fit_finds_the_opening_the_finger_is_at(model, fingers, error_deg):
    """The fingers drawn `error_deg` wider than the map says: the fit finds it from the tip (the sim's tip corner is
    ~1 mm from the model's wrist-side one, so within half a degree), and the fixed finger agrees."""
    pose = dict(RULER2_POSE, **{"gripper.pos": 40.0})
    map_deg = model.opening_deg("left", 40.0)
    image = _frame(fingers, pose, map_deg + error_deg)
    fit, seen = sweep.fit_frame(fingers, sweep.SweepFrame("left", pose, image), image)
    assert fit.map_deg == pytest.approx(map_deg)
    assert fit.error_deg == pytest.approx(error_deg, abs=0.5)
    assert fit.tip_px < 5.0 and fit.fixed_agreement > 0.95
    assert seen.tip is not None and np.linalg.norm(seen.tip - seen.tip_fitted) == pytest.approx(fit.tip_px)


def test_no_fit_with_the_jaws_nearly_closed(model, fingers):
    closed = model.gripper_maps["left"].pos_closed
    pose = dict(RULER2_POSE, **{"gripper.pos": closed})
    image = _frame(fingers, pose, 0.0)
    fit, _ = sweep.fit_frame(fingers, sweep.SweepFrame("left", pose, image), image)
    assert fit.fitted_deg is None and fit.error_deg is None  # (the fixed finger's check: with the jaws open)


def test_the_detected_tip_is_the_far_point_toward_the_map_and_not_one_cut_off():
    mask = np.zeros((100, 100), bool)
    mask[40:60, 20:70] = True
    mask[48:52, 70:80] = True  # a point sticking out to the right
    tip = sweep.detected_tip(mask, hinge=np.array([0.0, 50.0]), toward=np.array([90.0, 50.0]))
    assert tip is not None and tip[0] == 79
    mask[48:52, 80:100] = True  # now running off the frame
    assert sweep.detected_tip(mask, hinge=np.array([0.0, 50.0]), toward=np.array([90.0, 50.0])) is None


def test_a_constant_error_moves_where_the_tips_meet(model):
    k = model.gripper_maps["left"].deg_per_unit
    fits = [sweep.FingerFit(p, k * p, k * p + 3.0, 1.0, 1.0, 0.9) for p in (10.0, 30.0, 50.0)]
    result = sweep.summarize(fits, model, "left")
    assert result["mean_error_deg"] == pytest.approx(3.0) and result["slope"] == pytest.approx(1.0)
    assert result["pos_closed_if_constant"] == pytest.approx(result["pos_closed_now"] - 3.0 / k)


def test_the_sweep_poses_clear_the_table_and_see_it(model):
    """Each arm's sweep pose: the gripper over the table by more than the margin at every position, the table taken
        level with the arm's base (the higher); from rest (lower than the margin, beside the table's edge) the way there only
    rises; the table most of the view."""
    rest = {
        "shoulder_pan.pos": -7.2,
        "shoulder_lift.pos": -102.0,
        "elbow_flex.pos": 96.5,
        "wrist_flex.pos": 67.0,
    }
    rest |= {"wrist_roll.pos": -2.3, "gripper.pos": 1.8}
    for arm in ("left", "right"):
        renderer = sweep.FingerRenderer(model, arm)
        table = sweep.table_height(model, arm)
        assert table == pytest.approx(
            max(0.747, model.frame_pose(f"{arm}_base_link", model.joint_vector())[2, 3])
        )
        pose = sweep.SWEEP_POSES[arm]
        assert min(sweep.clearance(renderer, pose, sweep.DEFAULT_POSITIONS, table)) > sweep.MIN_CLEARANCE_M, (
            arm
        )
        at_rest, on_the_way = sweep.path_clearance(renderer, rest, {**pose, "gripper.pos": 1.8}, table)
        assert at_rest < sweep.MIN_CLEARANCE_M and on_the_way > at_rest, (
            arm
        )  # rests lower than the margin, then rises
        assert sweep.table_view(renderer, pose) > sweep.MIN_TABLE_VIEW, arm
        assert (
            min(sweep.clearance(renderer, pose, (30.0,), table + 0.06)) < sweep.MIN_CLEARANCE_M
        )  # a higher table
        renderer.close()


def test_the_fit_finds_the_hinge_the_opening_offset_and_the_mount(model, tmp_path):
    """Frames drawn with the hinge 3.9 mm off (in the plane its tip turns in), the opening 0.5 deg wide, the moving
    finger turned -12 deg on its jaw and the fixed one +4 deg: the fit finds all four, and the model it gives puts the
    tips on the drawn ones."""
    truth = sweep.corrected_model(model, "left", np.array([-0.0038, 0.0, -0.0008]), 0.5, -12.0, 4.0)
    drawn = sweep.FingerRenderer(truth, "left")
    fitter = sweep.Fitter(model, "left", tmp_path, window=False)
    for pos in (10.0, 20.0, 30.0, 40.0, 50.0):
        pose = dict(RULER2_POSE, **{"gripper.pos": pos})
        image = _frame(drawn, pose, truth.opening_deg("left", pos))
        fit, seen = sweep.fit_frame(fitter.fingers, sweep.SweepFrame("left", pose, image), image)
        fitter.seen.append((sweep.SweepFrame("left", pose, image), seen, fit))
    drawn.close()
    geometry = sweep.fit_geometry(model, "left", fitter.seen)
    assert geometry is not None
    np.testing.assert_allclose(geometry.hinge_shift_mm, (-3.8, 0.0, -0.8), atol=0.6)
    assert geometry.opening_offset_deg == pytest.approx(0.5, abs=0.3)
    assert geometry.finger_mount_deg == pytest.approx(-12.0, abs=1.0)
    assert geometry.fixed_finger_mount_deg == pytest.approx(4.0, abs=0.5)
    assert geometry.fixed_mount_scores[geometry.fixed_finger_mount_deg] > geometry.fixed_mount_scores[0.0]
    assert geometry.tip_rms_px < 1.5 < geometry.tip_rms_px_before


def test_the_hinge_and_the_mount_are_saved_and_loaded(model, tmp_path):
    corrected = sweep.corrected_model(model, "left", np.array([-0.0038, 0.0, -0.0008]), 0.5, -12.0, 4.0)
    assert {"pos:left_gripper:x", "pos:left_gripper:y", "pos:left_gripper:z"} <= {
        p.name for p in corrected.parameters
    }
    assert "pos:left_gripper:x" not in {p.name for p in model.parameters}  # the model given is left as it was
    corrected.save(tmp_path / "corrected.urdf")
    loaded = XleRobotModel.load(tmp_path / "corrected.urdf")
    assert loaded.estimate_dict()["pos:left_gripper:x"] == pytest.approx(-0.0038)
    assert loaded.gripper_maps["left"].finger_mount_deg == pytest.approx(-12.0)
    assert loaded.gripper_maps["left"].fixed_finger_mount_deg == pytest.approx(4.0)
    q = loaded.joint_vector({"left": RULER2_POSE})
    np.testing.assert_allclose(
        loaded.moving_tip("left", q, 40.0), corrected.moving_tip("left", q, 40.0), atol=1e-9
    )
    assert model.gripper_maps["left"].finger_mount_deg == 0.0  # (calibrations from before the fields: 0)
    assert model.gripper_maps["left"].fixed_finger_mount_deg == 0.0


def test_the_colors_are_learned_from_the_frame(model, fingers):
    """Near-white fingers on a grey-pink table (2026-11-15's second sweep: the fixed thresholds missed the fingers'
    pixels): the colors learned from the frame find the finger and fit it as well."""
    pose = dict(RULER2_POSE, **{"gripper.pos": 30.0})
    map_deg = model.opening_deg("left", 30.0)
    image = _frame(fingers, pose, map_deg + 3.0, wood=(190, 170, 165), finger=(235, 240, 245))
    assert not sweep.finger_pixels(image).any()  # the fixed thresholds see no finger at all
    fit, _ = sweep.fit_frame(fingers, sweep.SweepFrame("left", pose, image), image)
    assert fit.error_deg == pytest.approx(3.0, abs=0.5) and fit.fixed_agreement > 0.95


def test_a_frame_whose_tip_is_misread_is_left_out(model, tmp_path):
    """One frame's tip taken 60 px off (as when the finger's base was taken for it): left out of the geometry, which is
    found as from the good frames."""
    truth = sweep.corrected_model(model, "left", np.array([-0.0038, 0.0, -0.0008]), 0.0, -12.0)
    drawn = sweep.FingerRenderer(truth, "left")
    fitter = sweep.Fitter(model, "left", tmp_path, window=False)
    for pos in (10.0, 20.0, 30.0, 40.0, 50.0):
        pose = dict(RULER2_POSE, **{"gripper.pos": pos})
        image = _frame(drawn, pose, truth.opening_deg("left", pos))
        frame = sweep.SweepFrame("left", pose, image)
        fit, seen = sweep.fit_frame(fitter.fingers, frame, image)
        if pos == 30.0:
            seen.tip = seen.tip + np.array([0.0, 60.0])
        fitter.seen.append((frame, seen, fit))
    drawn.close()
    geometry = sweep.fit_geometry(model, "left", fitter.seen)
    assert geometry is not None and 30.0 not in geometry.used and len(geometry.used) == 4
    assert len(geometry.dropped) == 1 and "30.0" in geometry.dropped[0]
    np.testing.assert_allclose(geometry.hinge_shift_mm, (-3.8, 0.0, -0.8), atol=0.6)


class _Client:
    """A robot holding the arm at a pose and showing a given wrist frame (what `preview` asks of a client)."""

    def __init__(self, pose, image):
        self.pose, self.image, self.moves = pose, image, []

    def move_joints(self, targets, **kwargs):
        self.moves.append(targets)

    def wrist_capture(self, arm, color_format="jpeg", after=None):
        from types import SimpleNamespace

        return SimpleNamespace(color=self.image, state={f"{arm}_arm": self.pose}, timestamp=0.0)


def test_the_preview_passes_a_good_scene_and_holds_a_bad_one(model, fingers, tmp_path):
    """The preview opens the gripper to its position; a scene where the fingers are seen passes and Enter sweeps; a scene
    where they are not fails, Enter does not sweep (it looks again), f forces, q stops. It saves what it showed."""
    import copy
    from dataclasses import replace

    pose = dict(RULER2_POSE, **{"gripper.pos": sweep.PREVIEW_POS})
    good = _frame(fingers, pose, model.opening_deg("left", sweep.PREVIEW_POS))
    # the frames drawn are a pinhole's already: a lens without distortion, so the preview's undistortion keeps them
    pinhole = copy.copy(model)
    camera = model.wrist_cameras["left"]
    pinhole.wrist_cameras = dict(model.wrist_cameras) | {
        "left": replace(camera, coeffs=(0.0,) * len(camera.coeffs))
    }
    fitter = sweep.Fitter(pinhole, "left", tmp_path, window=False)
    client = _Client(pose, good)
    assert sweep.preview(client, "left", fitter, window=False, ask=lambda _: "")
    assert (
        client.moves == [{"left_arm": {"gripper.pos": sweep.PREVIEW_POS}}]
        and (tmp_path / "preview.png").exists()
    )
    bad = np.empty_like(good)
    bad[:] = WOOD  # no fingers to be seen
    answers = iter(["", "q"])
    assert not sweep.preview(_Client(pose, bad), "left", fitter, window=False, ask=lambda _: next(answers))
    assert sweep.preview(_Client(pose, bad), "left", fitter, window=False, ask=lambda _: "f")


def test_the_preview_check_holds_the_fits_limits():
    good = sweep.FingerFit(30.0, 38.0, 38.2, 2.0, 2.5, 0.85)
    assert sweep.preview_check(good) == []
    assert len(sweep.preview_check(sweep.FingerFit(30.0, 38.0, None, None, None, 0.5))) == 2
    assert "px" in sweep.preview_check(sweep.FingerFit(30.0, 38.0, 40.0, 12.0, 12.0, 0.85))[0]
