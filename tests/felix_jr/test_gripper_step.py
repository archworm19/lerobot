"""`calibration.gripper_step`: gaps to opening angles, the fit, the measuring loop (against the fake server), and the map's
round trip through a calibrated URDF."""

import threading
from pathlib import Path

import numpy as np
import pytest

from lerobot.felix_jr.calibration.gripper_step import (
    arm_calibration,
    fit_map,
    measure,
    opening_from_gap,
    report,
    servo_deg_per_unit,
)
from lerobot.felix_jr.calibration.xle_robot_model import GripperMap
from lerobot.felix_jr.server.client import FelixClient
from lerobot.felix_jr.server.hardware import FakeHardware
from lerobot.felix_jr.server.server import RobotServer

URDF = Path(__file__).parents[2] / "src" / "lerobot" / "felix_jr" / "urdfs" / "xlerobot_calib10312026.urdf"
RADIUS = 0.08
# the right arm's gripper (and the fake arms'): 1447 ticks, 1.272 deg per unit
GRIPPER = {"drive_mode": 0, "homing_offset": 0, "range_min": 2046, "range_max": 3493}
TRUE = GripperMap(pos_closed=1.5, deg_per_unit=1.272, calibration=GRIPPER)


def gap_at(pos: float) -> float:
    """The gap the TRUE map gives at `pos`: zero once the tips touch."""
    opening = np.radians(max(TRUE.opening_deg(pos), 0.0))
    return float(2 * RADIUS * np.sin(opening / 2))


# ---- gaps and the fit -----------------------------------------------------------------------------------------------


def test_a_gap_is_a_chord_of_the_tip_circle():
    assert opening_from_gap(0.0, RADIUS) == 0.0
    assert opening_from_gap(RADIUS, RADIUS) == pytest.approx(60.0)  # chord = radius: an equilateral triangle
    assert opening_from_gap(2 * RADIUS, RADIUS) == pytest.approx(180.0)
    with pytest.raises(ValueError):
        opening_from_gap(2 * RADIUS + 0.001, RADIUS)
    with pytest.raises(ValueError):
        opening_from_gap(-0.001, RADIUS)


def test_the_fit_recovers_the_map_and_leaves_out_touching_tips():
    measurements = [(pos, gap_at(pos)) for pos in (0.0, 1.0, 10.0, 25.0, 40.0, 55.0, 70.0, 85.0)]
    assert measurements[0][1] == 0.0 and measurements[1][1] == 0.0  # squeezed: the tips already touch
    gripper = fit_map(measurements, RADIUS, GRIPPER)
    assert gripper.pos_closed == pytest.approx(TRUE.pos_closed, abs=1e-6)
    assert gripper.deg_per_unit == pytest.approx(TRUE.deg_per_unit, rel=1e-6)
    assert gripper.calibration == GRIPPER
    assert gripper.measurements == tuple(measurements)  # all kept, for a refit
    said: list[str] = []
    assert report(gripper, RADIUS, say=said.append) == []
    assert sum("touching" in line for line in said) == 2


def test_the_fit_needs_two_open_positions():
    with pytest.raises(ValueError, match="two or more"):
        fit_map([(0.0, 0.0), (40.0, gap_at(40.0)), (40.0, gap_at(40.0))], RADIUS, GRIPPER)


def test_a_scale_far_from_the_servos_is_reported():
    assert servo_deg_per_unit(GRIPPER) == pytest.approx(1.2717, abs=1e-4)
    halved = [(pos, gap_at(pos / 2 + TRUE.pos_closed / 2)) for pos in (20.0, 50.0, 80.0)]  # half the opening
    warnings = report(fit_map(halved, RADIUS, GRIPPER), RADIUS, say=lambda *a: None)
    assert len(warnings) == 1 and "scale" in warnings[0]


def test_the_map_converts_both_ways():
    assert TRUE.opening_deg(TRUE.pos_for(30.0)) == pytest.approx(30.0)
    assert TRUE.opening_deg(0.0) < 0  # below where the tips meet: the servo squeezes


# ---- measuring, against the fake server ---------------------------------------------------------------------------


@pytest.fixture
def client():
    server = RobotServer(FakeHardware())
    address = server.bind("tcp://127.0.0.1:*")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    with FelixClient(address, timeout_s=5.0) as c:
        yield c
    server.shutdown()
    thread.join(timeout=3.0)
    server.close()


def test_the_calibration_comes_from_the_robot(client):
    record = arm_calibration(client, "right")
    assert record["gripper"] == GRIPPER  # the fake arms' gripper is calibrated like the real right arm
    assert "id" not in record["gripper"] and set(record) >= {"shoulder_pan", "wrist_roll"}


def test_measure_steps_the_gripper_and_returns_it(client):
    start = client.get_state()["right_arm"]["gripper.pos"]
    answers = iter(["12.5", "", "x", "-1", "30", "q"])
    prompts: list[str] = []
    said: list[str] = []

    def ask(prompt):
        prompts.append(prompt)
        return next(answers)

    with client.session():
        measurements = measure(client, "right", [10.0, 25.0, 40.0, 55.0, 70.0], ask=ask, say=said.append)
    # 25 skipped; a bad answer and a negative one asked again; q at 55
    assert measurements == [(10.0, 0.0125), (40.0, 0.03)]
    assert prompts[0].startswith("gripper.pos 10.0:") and prompts[-1].startswith("gripper.pos 55.0:")
    assert said == ["not a number: 'x'", "a gap cannot be negative"]
    assert client.get_state()["right_arm"]["gripper.pos"] == pytest.approx(start)
    assert client.get_state()["left_arm"]["gripper.pos"] == pytest.approx(start)  # the other arm untouched


# ---- the map in a calibrated URDF ---------------------------------------------------------------------------------


def test_the_map_round_trips_through_the_calibrated_urdf(tmp_path):
    pytest.importorskip("pinocchio")
    from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel

    model = XleRobotModel.load(URDF)
    assert model.gripper_maps == {}  # written before gripper maps
    # the moving tip's circle: ~8 cm, as the SO-101's mesh gives (the pinch point is the calibrated one)
    assert 0.075 < model.jaw_radius("right") < 0.087
    measured = fit_map([(pos, gap_at(pos)) for pos in (10.0, 40.0, 70.0)], RADIUS, GRIPPER)
    model.gripper_maps["right"] = measured
    model.save(tmp_path / "with_gripper.urdf")
    loaded = XleRobotModel.load(tmp_path / "with_gripper.urdf")
    assert loaded.gripper_maps == {"right": measured}
    np.testing.assert_array_equal(loaded.estimates, model.estimates)
    q = model.joint_vector()
    np.testing.assert_allclose(loaded.pinch_point("right", q), model.pinch_point("right", q), atol=1e-12)
