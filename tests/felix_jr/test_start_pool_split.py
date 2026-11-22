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
