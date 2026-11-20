"""The robot session's driver: the bottle named and confirmed, never another's resumed (`sim2real/robot_session.py`)."""

import json

import pytest

pytest.importorskip("pinocchio")

from lerobot.felix_jr.sim2real.robot_session import Bottle, check_bottle  # noqa: E402

FENO = Bottle("feno", 0.053, 0.11, 0.116)
PYRI = Bottle("pyri", 0.045, 0.13, 0.08)


def test_check_bottle_asks_and_stops_on_no(tmp_path):
    asked = []
    check_bottle(FENO, tmp_path / "results.jsonl", ask=lambda q: asked.append(q) or "y")
    assert "feno, 53 mm across, 110 mm tall, 116 g" in asked[0]
    with pytest.raises(SystemExit, match="--bottle"):
        check_bottle(FENO, tmp_path / "results.jsonl", ask=lambda q: "")


def test_check_bottle_refuses_another_bottle_on_resume(tmp_path):
    results = tmp_path / "results.jsonl"
    results.write_text(json.dumps({"spot": 1, "bottle": PYRI.__dict__}) + "\n")
    with pytest.raises(SystemExit, match="a new session"):
        check_bottle(FENO, results, ask=lambda q: "y")
    check_bottle(PYRI, results, ask=lambda q: "y")  # (the same bottle resumes)
