"""`calibration.joint_fit`: the residuals of a simulated session (`calibration.joint_sim`) through each camera's chain.

With no noise, every residual (the board's points in the head's and the wrist's pictures, the fingertips clicked on
each camera's face) is zero at the true robot and not at the model it was drawn from. The whole fit, on a simulated
session: `smoke_tests/test_joint_fit_smoke.py`.
"""

import cv2
import numpy as np
import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("scipy")

from lerobot.felix_jr.calibration.boards import CharucoBoard  # noqa: E402
from lerobot.felix_jr.calibration.joint_fit import HEAD, JointProblem, _vec  # noqa: E402
from lerobot.felix_jr.calibration.joint_sim import ARM, Scenario, World, draw_truth  # noqa: E402
from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel  # noqa: E402
from lerobot.felix_jr.harness import api  # noqa: E402

# the simulations start from (and draw their true robots around) this calibration, not `api.MODEL`: a new default
# calibration must not change what these tests check
SIM_START = api.MODEL.parent / "xle_calib_1008_wrist2.urdf"


@pytest.fixture(scope="module")
def session():
    model = XleRobotModel.load(SIM_START)
    rng = np.random.default_rng(1)
    world = World(model, draw_truth(model, rng), CharucoBoard(9, 7, 0.025, 0.018, "DICT_4X4_50"), rng)
    world.joint_noise_deg = world.board_noise_px = world.click_noise_px = 0.0
    sightings, clicks = world.collect(Scenario(1, 4, 3, True, "", 3))  # low poses: the head's clicks
    return world, JointProblem(model, world.board, {ARM: world.camera}, sightings, clicks)


def test_the_session_has_every_kind_of_observation(session):
    _, problem = session
    kinds = {(s.camera, "board") for s in problem.sightings} | {(c.camera, "click") for c in problem.clicks}
    assert kinds == {(HEAD, "board"), (ARM, "board"), (HEAD, "click"), (ARM, "click")}


def test_every_residual_is_zero_at_the_truth(session):
    world, problem = session
    truth = world.truth
    x = problem.start()  # the joints' errors at zero: the readings are exact here
    x[problem.slices["theta"]] = truth.theta[problem.free]
    x[problem.slices["gc"]] = _vec(truth.gripper_from_camera)
    x[problem.slices["boards"]] = np.concatenate([_vec(world.boards[p]) for p in problem.placements])
    x[problem.slices["thickness"]] = truth.thickness
    parts = problem.residual_parts(x)
    assert set(parts) == {"head board", f"{ARM} board", "head click", f"{ARM} click"}
    for kind, residuals in parts.items():
        assert np.abs(residuals).max() < 1e-6, kind
    started = problem.residual_parts(problem.start())  # the model as it is: off
    assert all(np.sqrt(np.mean(r**2)) > 1.0 for r in started.values()), {
        k: np.abs(v).max() for k, v in started.items()
    }


def test_the_wrist_click_is_one_thickness_from_the_head_click(session):
    world, problem = session
    click = next(c for c in problem.clicks if c.camera == ARM)
    q = problem._q[id(click)]
    theta = world.truth.theta
    head_side = problem.tip(type(click)(HEAD, ARM, click.look, click.pixel), q, theta, 0.01)
    wrist_side = problem.tip(click, q, theta, 0.012)
    assert np.linalg.norm(wrist_side - head_side) == pytest.approx(0.012, abs=1e-9)


def test_a_head_click_can_be_on_either_face(session):
    """The face a click is on decides its tip, whichever camera clicked it; without one, the camera's default."""
    world, problem = session
    click = next(c for c in problem.clicks if c.camera == ARM)
    q, theta = problem._q[id(click)], world.truth.theta
    make = type(click)
    pinch = problem.tip(make(HEAD, ARM, click.look, click.pixel), q, theta, 0.01)
    wrist = problem.tip(click, q, theta, 0.01)
    assert make(HEAD, ARM, click.look, click.pixel).on_face == "pinch" and click.on_face == "wrist"
    np.testing.assert_allclose(
        problem.tip(make(HEAD, ARM, click.look, click.pixel, face="wrist"), q, theta, 0.01), wrist
    )
    np.testing.assert_allclose(
        problem.tip(make(ARM, ARM, click.look, click.pixel, face="pinch"), q, theta, 0.01), pinch
    )


def test_the_sparsity_covers_every_dependence(session):
    """A finite-difference Jacobian is zero wherever the sparsity says a residual does not depend on an unknown."""
    _, problem = session
    x = problem.start()
    sparsity = problem.sparsity()
    base = problem.residuals(x)
    assert sparsity.shape == (len(base), problem.size)
    rng = np.random.default_rng(0)
    for column in rng.choice(problem.size, 40, replace=False):
        step = np.zeros(problem.size)
        step[column] = 1e-6
        changed = np.abs(problem.residuals(x + step) - base) > 1e-9
        assert not (changed & ~sparsity[:, column]).any(), column


def test_a_session_is_saved_and_read_back(session, tmp_path):
    """Sightings and clicks keep their moments shared (one joint error each) through the file."""
    from lerobot.felix_jr.calibration.joint_fit import JointSession, load_session, save_session

    world, problem = session
    model = world.model
    saved = JointSession(
        tmp_path,
        ARM,
        world.board,
        model.head_signs,
        model.head_refs,
        model.intrinsics,
        None,
        problem.sightings,
        problem.clicks,
    )
    save_session(saved)
    back = load_session(tmp_path)
    assert back.board == world.board and back.arm == ARM
    assert [c.on_face for c in back.clicks] == [c.on_face for c in problem.clicks]
    assert len(back.sightings) == len(saved.sightings) and len(back.clicks) == len(saved.clicks)
    again = JointProblem(model, back.board, {ARM: world.camera}, back.sightings, back.clicks)
    assert len(again.looks) == len(problem.looks)
    np.testing.assert_allclose(again.residuals(again.start()), problem.residuals(problem.start()))


def test_held_out_sightings_are_whole_moments_both_cameras_saw(session):
    from lerobot.felix_jr.calibration.joint_fit import split_holdout

    _, problem = session
    kept, held = split_holdout(problem.sightings, 1)  # the fixture has one such moment
    assert held and kept and len(kept) + len(held) == len(problem.sightings)
    assert not {id(s.look) for s in kept} & {id(s.look) for s in held}
    cameras: dict[int, set[str]] = {}
    for s in held:
        cameras.setdefault(id(s.look), set()).add(s.camera)
    assert all(HEAD in seen and len(seen) == 2 for seen in cameras.values())


def test_the_two_chains_agree_on_the_board_at_the_truth_only(session):
    from lerobot.felix_jr.calibration.joint_fit import agreement

    world, problem = session  # no noise: at the truth the head and the arm put the board in the same place
    cameras = {ARM: world.camera}
    true = agreement(
        problem.model,
        world.board,
        cameras,
        problem.sightings,
        world.truth.theta,
        {ARM: world.truth.gripper_from_camera},
    )
    start = agreement(
        problem.model, world.board, cameras, problem.sightings, problem.model.estimates, {ARM: world.gc0}
    )
    assert len(true) == len(start) > 0
    assert true.max(axis=0)[0] < 0.1 and true.max(axis=0)[1] < 0.05, true
    assert np.median(start[:, 0]) > 2.0, start


def test_a_moment_with_too_few_board_points_is_left_out_of_the_agreement(session):
    """A camera's board pose from a few points is unreliable: such a moment is not scored (joint_right_full4: 6 corners
    in the head's picture, 91 deg)."""
    from dataclasses import replace

    from lerobot.felix_jr.calibration.joint_fit import MIN_POSE_POINTS, agreement

    world, problem = session
    cameras, gcs = {ARM: world.camera}, {ARM: world.truth.gripper_from_camera}
    full = agreement(problem.model, world.board, cameras, problem.sightings, world.truth.theta, gcs)
    one = next(id(s.look) for s in problem.sightings if s.camera == HEAD)
    few = [
        replace(s, detection=replace(s.detection, ids=s.detection.ids[:6], pixels=s.detection.pixels[:6]))
        if s.camera == HEAD and id(s.look) == one
        else s
        for s in problem.sightings
    ]
    assert MIN_POSE_POINTS > 6
    assert len(agreement(problem.model, world.board, cameras, few, world.truth.theta, gcs)) == len(full) - 1


def test_two_sessions_placements_are_two_boards(session, tmp_path):
    """Each session names its placements p0, p1, ...: combined, the same name in two sessions is two places."""
    from lerobot.felix_jr.calibration.joint_fit import JointSession, calibrate

    world, problem = session
    model = world.model
    sessions = [
        JointSession(
            tmp_path / name,
            ARM,
            world.board,
            model.head_signs,
            model.head_refs,
            model.intrinsics,
            None,
            problem.sightings,
            problem.clicks,
        )
        for name in ("a", "b")
    ]
    _, fit, _ = calibrate(model, sessions, holdout_every=0)
    names = {s.placement for s in problem.sightings}
    assert set(fit.boards) == {f"{i}:{p}" for i in (0, 1) for p in names}


def test_a_misclick_is_named_and_left_out(session, tmp_path):
    """One head click 200 px off (the wrong spot): the report names it and the fit is done again without it; kept, it is
    named all the same."""
    from dataclasses import replace

    from lerobot.felix_jr.calibration.joint_fit import (
        MISCLICK_PX,
        JointSession,
        calibrate,
        report,
        save_misclicks,
    )

    world, problem = session
    model = world.model
    k = next(i for i, c in enumerate(problem.clicks) if c.camera == HEAD)
    clicks = list(problem.clicks)
    clicks[k] = replace(clicks[k], pixel=np.asarray(clicks[k].pixel) + [200.0, 0.0], image="images/wrong.png")
    one = JointSession(
        tmp_path / "s",
        ARM,
        world.board,
        model.head_signs,
        model.head_refs,
        model.intrinsics,
        None,
        problem.sightings,
        clicks,
    )
    calibrated, fit, numbers = calibrate(model, [one], holdout_every=0)
    assert [m["image"] for m in numbers["misclicks"]] == ["images/wrong.png"]
    assert (
        numbers["misclicks"][0]["px"] > MISCLICK_PX
        and numbers["clicks"][HEAD] == sum(c.camera == HEAD for c in problem.clicks) - 1
    )
    assert "images/wrong.png" in report(calibrated, model, numbers)
    (tmp_path / "s" / "images").mkdir(
        parents=True
    )  # its picture: saved with the click and the fitted fingertip
    cv2.imwrite(str(tmp_path / "s" / "images" / "wrong.png"), np.zeros((480, 640, 3), np.uint8))
    save_misclicks(numbers, tmp_path / "out")
    assert (tmp_path / "out" / "misclicks" / "00.png").is_file() and numbers["misclicks"][0]["shown"]
    assert "misclicks/00.png" in report(calibrated, model, numbers)
    assert not numbers["fit_failed"] and "FIT FAILED" not in report(calibrated, model, numbers)
    _, _, kept = calibrate(model, [one], holdout_every=0, drop_misclicks=False)
    assert not kept["misclicks"] and [m["image"] for m in kept["misclicks_kept"]] == ["images/wrong.png"]


def test_the_moving_fingertip_from_the_wrist_alone_leaves_the_hinge_to_a_few_mm(session):
    """The moving finger's tip in the wrist camera at a few openings (a gripper sweep's: `MOVING` clicks), drawn with the
    hinge 3 mm off and the map's opening 0.8 deg off: the fit matches every click and finds the opening; the hinge only
    to a few mm. The wrist camera rides on the gripper and sees no depth: the camera turned a little about the roll
    axis, the hinge moved and the opening nudged draw the same arc (here: 0.8 deg, 2 mm), and the fit settles where the
    priors cost least. A second view (the head's clicks on the moving tip) is the TODO
    (`docs/camera_calibration.md`, "Not done yet")."""
    from lerobot.felix_jr.calibration.joint_fit import JAW_THICKNESS_M, MOVING, Look, TipClick, fit_staged
    from lerobot.felix_jr.calibration.xle_robot_model import pinch_frame, wrist_camera_link

    world, problem = session
    model = world.model.with_gripper_hinge(ARM)
    names = [p.name for p in model.parameters]
    shift = np.array([-0.003, 0.0, 0.0015])
    truth = np.r_[world.truth.theta, shift]  # (the hinge's parameters are appended)
    assert names[-3:] == [f"pos:{ARM}_gripper:{a}" for a in "xyz"]
    looks = [c.look for c in problem.clicks if c.camera == ARM][:3]
    moving = []
    for look in looks:
        for pos in (15.0, 30.0, 45.0):
            seen = Look({ARM: {**look.arms[ARM], "gripper.pos": pos}}, look.head_ticks)
            q = model.joint_vector(seen.arms, seen.head_ticks)
            pinch = model.frame_pose(pinch_frame(ARM), q, truth)
            turned = model.moving_jaw(ARM, model.opening_deg(ARM, pos) + 0.8, truth)
            face = model.wrist_face(ARM) * (world.truth.thickness / JAW_THICKNESS_M)
            point = pinch @ turned @ np.r_[face, 1.0]
            camera = model.frame_pose(wrist_camera_link(ARM), q, truth) @ world.truth.gripper_from_camera
            pixel = world.camera.project(((point[:3] - camera[:3, 3]) @ camera[:3, :3])[None])[0]
            moving.append(TipClick(ARM, ARM, seen, pixel, "", MOVING))
    with_moving = JointProblem(
        model, world.board, {ARM: world.camera}, problem.sightings, [*problem.clicks, *moving]
    )
    assert with_moving.moving_arms == [ARM]
    fit = fit_staged(with_moving)
    hinge = fit.theta[[names.index(f"pos:{ARM}_gripper:{a}") for a in "xz"]]
    assert fit.rms_px[f"{ARM} click"] < 0.5  # every click matched
    assert fit.openings[ARM] == pytest.approx(0.8, abs=0.3)
    assert np.linalg.norm(hinge - shift[[0, 2]]) < 3e-3  # the hinge: within the ambiguity, not pinned


def test_the_wrist_camera_is_found_from_its_own_sightings(session):
    """The arm and the boards as they are: each wrist sighting's PnP, taken back to the gripper, gives the camera's pose
    on it (`fit_staged`'s retry, for a camera far off)."""
    world, problem = session
    gc = problem.wrist_cameras_from_sightings(world.truth.theta, world.boards)[ARM]
    true = world.truth.gripper_from_camera
    assert np.linalg.norm(gc[:3, 3] - true[:3, 3]) < 1e-3
    assert np.degrees(np.linalg.norm(cv2.Rodrigues(gc[:3, :3] @ true[:3, :3].T)[0])) < 0.2


def test_a_fit_with_a_kind_of_residual_far_off_has_failed():
    """A wrong solution: some kind of residual off at its median (`fit_failed`)."""
    from lerobot.felix_jr.calibration.joint_fit import FIT_FAILED_PX, JointFit, fit_failed

    fit = JointFit(np.zeros(1), {}, {}, 0.01, median_px={"head board": 0.3, f"{ARM} click": 1.0})
    fit.rms_px["head click"] = 5 * FIT_FAILED_PX  # a mis-click lifts the RMS, not the median: not failed
    assert not fit_failed(fit)
    fit.median_px[f"{ARM} board"] = 2 * FIT_FAILED_PX
    assert fit_failed(fit)


def test_the_hinge_along_its_own_axis_is_not_fitted_and_the_fit_says_what_it_learned(session):
    """Sliding the hinge along its axis moves nothing: held. The uncertainty: every reported unknown no less certain
    than its prior, the hinge's across-axis position among them."""
    from lerobot.felix_jr.calibration.joint_fit import JointFit, uncertainty

    world, problem = session
    model = world.model.with_gripper_hinge(ARM)
    hinge = JointProblem(model, world.board, {ARM: world.camera}, problem.sightings, problem.clicks)
    names = [p.name for p in model.parameters]
    free = {n for n, f in zip(names, hinge.free, strict=True) if f}
    assert f"pos:{ARM}_gripper:y" not in free and {f"pos:{ARM}_gripper:x", f"pos:{ARM}_gripper:z"} <= free
    x = hinge.start()
    theta, gcs, boards, thickness = hinge.unpack(x)
    u = uncertainty(hinge, JointFit(theta, gcs, boards, thickness, x=x))
    assert f"pos:{ARM}_gripper:x" in u["sd"] and f"pos:{ARM}_gripper:y" not in u["sd"]
    for name, v in u["sd"].items():
        if v["prior"] is not None:
            assert v["sd"] <= v["prior"] * 1.001, name
