"""`calibration.joint_fit`: the residuals of a simulated session (`calibration.joint_sim`) through each camera's chain.

With no noise, every residual (the board's points in the head's and the wrist's pictures, the fingertips clicked on
each camera's face) is zero at the true robot and not at the model it was drawn from. The whole fit, on a simulated
session: `smoke_tests/test_joint_fit_smoke.py`.
"""

import numpy as np
import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("scipy")

from lerobot.felix_jr.calibration.boards import Checkerboard  # noqa: E402
from lerobot.felix_jr.calibration.joint_fit import HEAD, JointProblem, _vec  # noqa: E402
from lerobot.felix_jr.calibration.joint_sim import ARM, Scenario, World, draw_truth  # noqa: E402
from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel  # noqa: E402
from lerobot.felix_jr.harness import api  # noqa: E402


@pytest.fixture(scope="module")
def session():
    model = XleRobotModel.load(api.MODEL)
    rng = np.random.default_rng(1)
    world = World(model, draw_truth(model, rng), Checkerboard(8, 6, 0.025), rng)
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
