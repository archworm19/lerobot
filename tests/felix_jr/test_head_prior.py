"""The head camera's cylinder, simulated (`sim2real/head_prior.py`): the frames, one error an episode, its sizes."""

import numpy as np

from lerobot.felix_jr.sim2real.head_prior import (
    HeadErrors,
    episode_errors,
    gain_curve,
    priors,
    quat_to_matrix,
    to_pinch,
)


def _random_quats(rng, n):
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def test_to_pinch_is_the_inverse_of_the_pinch_pose():
    """A cylinder put in the root frame through a pinch pose and back with `to_pinch` comes back the same."""
    rng = np.random.default_rng(0)
    n = 50
    q, p = _random_quats(rng, n), rng.normal(0, 0.3, (n, 3))
    r = quat_to_matrix(q)
    assert np.allclose(np.einsum("nij,nkj->nik", r, r), np.eye(3), atol=1e-9)  # (rotations)
    pinch = np.concatenate(
        [rng.normal(0, 0.05, (n, 3)), rng.normal(size=(n, 3)), rng.uniform(0.01, 0.2, (n, 2))], axis=1
    )
    pinch[:, 3:6] /= np.linalg.norm(pinch[:, 3:6], axis=1, keepdims=True)
    root = np.concatenate(
        [
            np.einsum("nij,nj->ni", r, pinch[:, :3]) + p,
            np.einsum("nij,nj->ni", r, pinch[:, 3:6]),
            pinch[:, 6:],
        ],
        axis=1,
    )
    assert np.allclose(to_pinch(root, p, q), pinch, atol=1e-9)


def test_one_error_an_episode_and_its_sizes():
    """Every frame of a world's round gets the same offset; across episodes the offsets and radius errors are as set;
    the weak share is about as set; the priors are upright in the root frame."""
    rng = np.random.default_rng(1)
    errors = HeadErrors()
    table = episode_errors(4, 2000, errors, rng)
    strong = ~table.weak
    assert abs(table.weak.mean() - errors.weak_share) < 0.02
    assert abs(table.offset[strong][:, 0].std() - errors.across_m) < 0.002
    assert (table.d_radius[strong] >= errors.radius_m[0]).all() and (
        table.d_radius[strong] <= errors.radius_m[1]
    ).all()
    assert (np.linalg.norm(table.offset, axis=-1) <= errors.clip_m * errors.weak_scale + 1e-12).all()
    n = 6
    labels = {
        "round": np.array([0, 0, 0, 1, 1, 1]),
        "world": np.array([5, 5, 5, 5, 5, 5]),
        "in_root": np.concatenate([rng.normal(0, 0.1, (n, 3)), np.tile([0.0, 0.0, 1.0], (n, 1))], axis=1),
        "size": np.tile([0.02, 0.1], (n, 1)),
        "pinch_pos": rng.normal(0, 0.3, (n, 3)),
        "pinch_quat": _random_quats(rng, n),
    }
    root, pinch = priors(labels, np.arange(n), table, np.array([0.025, 0.12]))
    offsets = root[:, :3] - labels["in_root"][:, :3]
    assert np.allclose(offsets[:3], offsets[0]) and np.allclose(offsets[3:], offsets[3])
    assert not np.allclose(offsets[0], offsets[3])
    assert np.allclose(root[:, 3:6], [0.0, 0.0, 1.0])
    assert np.allclose(to_pinch(root, labels["pinch_pos"], labels["pinch_quat"]), pinch)


def test_the_gain_curve_bins_by_the_prior_error():
    prior = np.array([2.0, 7.0, 7.5, 25.0])
    estimate = np.array([3.0, 2.0, 1.0, 5.0])
    curve = gain_curve(estimate, prior)
    assert [b["n"] for b in curve] == [1, 2, 1]
    assert curve[0]["worse_than_prior"] == 1.0 and curve[1]["worse_than_prior"] == 0.0
    assert curve[2]["estimate_median"] == 5.0
