"""`sim2real.bank`: the placements drawn, the file written, read back and resumed (the simulation stubbed out: the
smoke test `test_bank_smoke.py` runs it)."""

import json
from dataclasses import replace

import pytest

pytest.importorskip("pinocchio")

from lerobot.felix_jr.calibration.gripper_sweep import table_height  # noqa: E402
from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel  # noqa: E402
from lerobot.felix_jr.harness import api  # noqa: E402
from lerobot.felix_jr.sim2real import bank  # noqa: E402


@pytest.fixture(scope="module")
def model():
    return XleRobotModel.load(api.MODEL)


def test_placements_are_the_seeds_on_the_table_within_the_ranges(model):
    config = bank.BankConfig()
    first = bank.placements(model, config, 20)
    assert (
        bank.placements(model, config, 5) == first[:5]
    )  # (each its own generator: a longer run, the same start)
    assert bank.placements(model, replace(config, seed=1), 5) != first[:5]
    base = model.frame_pose("right_base_link", model.joint_vector())[:3, 3]
    for p in first:
        assert p.bottom_root[2] == pytest.approx(table_height(model, "right"))
        assert config.ahead_m[0] <= base[0] - p.bottom_root[0] <= config.ahead_m[1]
        assert config.side_m[0] <= p.bottom_root[1] - base[1] <= config.side_m[1]
        assert config.diameter_m[0] <= 2 * p.radius <= config.diameter_m[1]
        assert config.mass_kg[0] <= p.mass_kg <= config.mass_kg[1]


def test_the_depths_are_the_configs_and_one_just_past_the_radius():
    assert bank.BankConfig(depths_m=(0.0, 0.01)).depths(0.0225) == (0.0, 0.01, 0.0275)


def test_a_bank_is_written_read_back_and_resumed_without_redoing_a_placement(model, tmp_path, monkeypatch):
    ran = []

    def fake(model, config, placement):
        ran.append(placement.index)
        return {"kind": "placement", "placement": {"index": placement.index}, "anchor": None, "simulated": []}

    monkeypatch.setattr(bank, "run_placement", fake)
    out = tmp_path / "bank.jsonl"
    config = bank.BankConfig(seed=3)
    assert len(list(bank.build_bank(api.MODEL, config, 2, out))) == 2
    first = json.loads(out.read_text().splitlines()[0])
    assert first["kind"] == "config" and first["table_z"] == pytest.approx(table_height(model, "right"))
    assert bank.load(out)[0] == config  # (the tuples back as tuples)
    ran.clear()
    list(bank.build_bank(api.MODEL, config, 4, out, resume=True))
    assert ran == [2, 3]
    assert [line["placement"]["index"] for line in bank.load(out)[1]] == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="another config"):
        list(bank.build_bank(api.MODEL, replace(config, seed=4), 4, out, resume=True))
