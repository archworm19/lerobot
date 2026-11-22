"""The held-out split of a bank's placements: at random, or a region by a coordinate (`start_pool.split_placements`)."""

import pytest

from lerobot.felix_jr.sim2real.start_pool import placement_coordinate, split_placements


def _placement(index, ahead, side, diameter=0.05, height=0.12):
    # the arm's base at (0.1, -0.2): ahead is along -x, the side along +y
    return {
        "index": index,
        "bottom_root": [0.1 - ahead, -0.2 + side, 0.78],
        "radius": diameter / 2,
        "height": height,
        "mass_kg": 0.08,
    }


BASE = (0.1, -0.2)
PLACEMENTS = [
    _placement(i, 0.22 + 0.01 * i, -0.15 + 0.03 * i, 0.04 + 0.002 * i, 0.20 - 0.01 * i) for i in range(10)
]


def test_coordinates():
    p = _placement(3, 0.3, -0.1, 0.05, 0.13)
    assert placement_coordinate(p, BASE, "ahead") == pytest.approx(0.3)
    assert placement_coordinate(p, BASE, "side") == pytest.approx(-0.1)
    assert placement_coordinate(p, BASE, "diameter") == pytest.approx(0.05)
    assert placement_coordinate(p, BASE, "height") == pytest.approx(0.13)


@pytest.mark.parametrize(
    ("key", "at", "held_out"),
    [
        ("ahead", "high", {8, 9}),  # the farthest
        ("ahead", "low", {0, 1}),
        ("side", "middle", {4, 5}),
        ("height", "high", {0, 1}),  # (taller with a lower index)
        ("diameter", "low", {0, 1}),
    ],
)
def test_a_region_held_out(key, at, held_out):
    train, test = split_placements(PLACEMENTS, 0.2, key, at, BASE)
    assert test == held_out and train == set(range(10)) - held_out


def test_random_is_reproducible_and_disjoint():
    a = split_placements(PLACEMENTS, 0.3, seed=1)
    assert a == split_placements(PLACEMENTS, 0.3, seed=1)
    train, test = a
    assert len(test) == 3 and not train & test and train | test == set(range(10))


@pytest.mark.parametrize("vary", [False, True])
def test_the_placements_drawn(tmp_path, vary):
    """`placement_plot.plot_placements`: one object (the table alone) or several sizes (the grid by size range)."""
    pytest.importorskip("matplotlib")
    from lerobot.felix_jr.sim2real.placement_plot import plot_placements

    placements = PLACEMENTS if vary else [_placement(i, 0.22 + 0.01 * i, -0.15 + 0.03 * i) for i in range(10)]
    lines = [{"placement": p, "anchor": None if p["index"] == 3 else {}} for p in placements]
    out = plot_placements(lines, BASE, {8, 9}, tmp_path / "placements.png", "test")
    assert out.is_file() and out.stat().st_size > 10_000


def _tiny_pool(objects):
    import numpy as np

    from lerobot.felix_jr.sim2real.start_pool import StartPool

    n = 2
    return StartPool(
        {"j": np.zeros((n, 1))}, {"j": np.zeros((n, 1))}, {"a": np.zeros((n, 1))},
        np.array(["attack", "lifted"]), np.arange(n), objects, 0.78,
    )  # fmt: skip


def test_the_pool_file_round_trips_and_is_checked(tmp_path):
    """`StartPool`'s .npz: a size a start (format 2); an older one-size file read still; a newer format, or arrays of
    another length, refused."""
    import numpy as np

    from lerobot.felix_jr.sim2real.start_pool import FORMAT, StartPool

    pool = _tiny_pool(np.array([[0.02, 0.1, 0.05], [0.03, 0.2, 0.1]]))
    pool.save(tmp_path / "p.npz")
    back = StartPool.load(tmp_path / "p.npz")
    assert back.sizes_vary and np.allclose(back.objects, pool.objects)
    z = dict(np.load(tmp_path / "p.npz"))
    old = {k: v for k, v in z.items() if k not in ("format", "objects")} | {
        "object_size": np.array([0.02, 0.1, 0.05])
    }
    np.savez(tmp_path / "old.npz", **old)
    assert StartPool.load(tmp_path / "old.npz").object_size == (0.02, 0.1, 0.05)  # (format 1: one size)
    np.savez(tmp_path / "new.npz", **(z | {"format": np.array(FORMAT + 1)}))
    with pytest.raises(ValueError, match="newer"):
        StartPool.load(tmp_path / "new.npz")
    with pytest.raises(ValueError, match="other lengths"):
        StartPool(
            {"j": np.zeros((3, 1))},
            {},
            {},
            np.array(["attack", "lifted"]),
            np.arange(2),
            [0.02, 0.1, 0.05],
            0.78,
        )
