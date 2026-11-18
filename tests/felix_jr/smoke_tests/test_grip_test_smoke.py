"""A grip trial (`calibration.grip_test`) against the fake robot: a bottle, its closes, the hold, the lift, all saved."""

import json
import threading
from dataclasses import replace

import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("mujoco")

from lerobot.felix_jr.calibration import grip_test  # noqa: E402
from lerobot.felix_jr.calibration.gripper_sweep import table_height  # noqa: E402
from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel, arm_record, motor_record  # noqa: E402
from lerobot.felix_jr.harness import api  # noqa: E402
from lerobot.felix_jr.server.client import FelixClient  # noqa: E402
from lerobot.felix_jr.server.hardware import FakeHardware  # noqa: E402
from lerobot.felix_jr.server.server import RobotServer  # noqa: E402


def test_a_bottle_is_gripped_held_and_lifted_and_saved(tmp_path, monkeypatch):
    """The fake's jaw stalls where the map puts the bottle's contact: every close stalls there, pressing harder the
    further past contact it was sent; the person's answers are scripted."""
    monkeypatch.setattr(grip_test, "HOLD_S", 2.0)
    monkeypatch.setattr(grip_test, "LIFT_HOLD_S", 0.1)
    hardware = FakeHardware(arm_time_scale=0.02)
    server = RobotServer(hardware)
    address = server.bind("tcp://127.0.0.1:*")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    model = XleRobotModel.load(api.MODEL)
    robot = hardware.arm_backend.calibration()
    model.arm_calibration = {arm: arm_record(robot[f"{arm}_arm"]) for arm in ("left", "right")}
    model.gripper_maps = {
        arm: replace(m, calibration=motor_record(robot[f"{arm}_arm"]["gripper"]))
        for arm, m in model.gripper_maps.items()
    }
    contact = model.gripper_pos_for("right", model.opening_for_width("right", 0.065))
    hardware.arm_backend.object_pos["right_arm"] = contact
    answers = iter(["water", "65", "200", "500", "", "y", "", "y", "held", "y", ""])
    asked = []

    def ask(prompt):
        asked.append(prompt)
        return next(answers)

    try:
        with FelixClient(address, timeout_s=10.0) as client, client.session():
            test = grip_test.GripTest(client, "right", tmp_path, model, table_height(model, "right"), ask=ask)
            bottle = test.ask_bottle()
            assert bottle == grip_test.Bottle("water", 65.0, 200.0, 500.0, grip_test.DEPTH_MM)
            assert test.trial(bottle), asked
    finally:
        server.shutdown()
        thread.join(timeout=3.0)
        server.close()
    saved = json.loads((tmp_path / "results.json").read_text())
    assert len(saved) == 1
    trial = saved[0]
    assert trial["contact_pos"] == pytest.approx(contact)
    stalls = [c["stall"] for c in trial["closes"]]
    assert len(stalls) == len(grip_test.PAST_CONTACT)
    assert all(s["pos"] == pytest.approx(contact, abs=0.5) for s in stalls)  # each stalled on the bottle
    loads = [s["load_pct"] for s in stalls]
    assert loads == sorted(loads) and loads[-1] > loads[0]  # harder the further past contact
    assert trial["hold"]["samples"] and trial["lift"]["result"] == "held"
    assert all((tmp_path / c["frame"]).is_file() for c in trial["closes"])
    assert "water" in test.summary()
