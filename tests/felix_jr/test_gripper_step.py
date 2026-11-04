"""`calibration.gripper_step`: gaps to opening angles, the fit, the plot, the measuring loop (against the fake server),
and the map's round trip through a calibrated URDF."""

import threading
from pathlib import Path

import numpy as np
import pytest

from lerobot.felix_jr.calibration.gripper_step import (
    arm_calibration,
    chord,
    fit_map,
    measure,
    opening_from_gap,
    plot,
    report,
    scatter_m,
    servo_deg_per_unit,
)
from lerobot.felix_jr.calibration.xle_robot_model import GripperMap
from lerobot.felix_jr.server.client import FelixClient
from lerobot.felix_jr.server.hardware import FakeHardware
from lerobot.felix_jr.server.server import RobotServer

URDF = Path(__file__).parents[2] / "src" / "lerobot" / "felix_jr" / "urdfs" / "xlerobot_calib10312026.urdf"
MODEL_RADIUS = 0.0809  # the right arm's pinch point from its hinge, in the 10/31 calibration
RADIUS = 0.0868  # where the calipers touch the tips: a little farther out
# the right arm's gripper (and the fake arms'): 1447 ticks, 1.272 deg per unit
GRIPPER = {"drive_mode": 0, "homing_offset": 0, "range_min": 2046, "range_max": 3493}
TRUE = GripperMap(1.5, servo_deg_per_unit(GRIPPER), GRIPPER, radius_m=RADIUS)
# measured on the right arm (2026-11-04), calipers at the very tips
MEASURED = [(9.7, 0.020), (24.7, 0.050), (39.7, 0.075), (54.6, 0.100), (69.7, 0.125), (84.7, 0.140)]


def gap_at(pos: float, radius: float = RADIUS) -> float:
    """The gap the TRUE map gives at `pos`: zero once the tips touch."""
    return float(chord(radius, np.array(TRUE.opening_deg(pos))))


# ---- gaps and the fit -----------------------------------------------------------------------------------------------


def test_a_gap_is_a_chord_of_the_tip_circle():
    assert opening_from_gap(0.0, RADIUS) == 0.0
    assert opening_from_gap(RADIUS, RADIUS) == pytest.approx(60.0)  # chord = radius: an equilateral triangle
    assert opening_from_gap(2 * RADIUS, RADIUS) == pytest.approx(180.0)
    assert chord(RADIUS, np.array([60.0, -5.0])) == pytest.approx(
        [RADIUS, 0.0]
    )  # and back; closed past touching
    with pytest.raises(ValueError):
        opening_from_gap(2 * RADIUS + 0.001, RADIUS)
    with pytest.raises(ValueError):
        opening_from_gap(-0.001, RADIUS)


def test_the_fit_recovers_the_map_at_the_servos_scale_and_leaves_out_touching_tips():
    measurements = [(pos, gap_at(pos)) for pos in (0.0, 1.0, 10.0, 25.0, 40.0, 55.0, 70.0, 85.0)]
    assert measurements[0][1] == 0.0 and measurements[1][1] == 0.0  # squeezed: the tips already touch
    gripper = fit_map(measurements, GRIPPER, MODEL_RADIUS)
    assert gripper.pos_closed == pytest.approx(TRUE.pos_closed, abs=1e-4)
    assert gripper.radius_m == pytest.approx(RADIUS, abs=1e-6)
    assert gripper.deg_per_unit == servo_deg_per_unit(GRIPPER) == pytest.approx(1.2718, abs=1e-4)
    assert gripper.calibration == GRIPPER
    assert gripper.measurements == tuple(measurements)  # all kept, for a refit
    said: list[str] = []
    assert report(gripper, MODEL_RADIUS, say=said.append) == []
    assert sum("touching" in line for line in said) == 2


def test_the_right_arms_measurements():
    gripper = fit_map(MEASURED, GRIPPER, MODEL_RADIUS)
    assert gripper.pos_closed == pytest.approx(-1.0, abs=0.05)
    assert gripper.radius_m == pytest.approx(0.0868, abs=0.0002)
    assert scatter_m(gripper) == pytest.approx(0.0013, abs=0.0001)
    assert report(gripper, MODEL_RADIUS, say=lambda *a: None) == []


def test_the_fit_needs_two_open_positions():
    with pytest.raises(ValueError, match="two or more"):
        fit_map([(0.0, 0.0), (40.0, gap_at(40.0)), (40.0, gap_at(40.0))], GRIPPER, MODEL_RADIUS)


def test_a_radius_far_from_the_models_and_scattered_gaps_are_reported():
    wide = fit_map([(pos, gap_at(pos, 0.100)) for pos in (20.0, 50.0, 80.0)], GRIPPER, MODEL_RADIUS)
    assert wide.radius_m == pytest.approx(0.100, abs=1e-6)
    warnings = report(wide, MODEL_RADIUS, say=lambda *a: None)
    assert len(warnings) == 1 and "+19 mm from the model's" in warnings[0]
    noisy = [
        (pos, gap_at(pos) + (0.004 if i % 2 else -0.004))
        for i, pos in enumerate((10.0, 30.0, 50.0, 70.0, 90.0))
    ]
    warnings = report(fit_map(noisy, GRIPPER, MODEL_RADIUS), MODEL_RADIUS, say=lambda *a: None)
    assert len(warnings) == 1 and "scatter" in warnings[0]


def test_the_plot_shows_the_measurements_and_the_fit(tmp_path):
    figure = pytest.importorskip("matplotlib.figure")
    fig = figure.Figure()
    plot(fig, MEASURED[:1], None, MODEL_RADIUS)  # one measurement: nothing to fit yet
    assert [len(ax.lines) for ax in fig.axes] == [1, 0]
    gripper = fit_map(MEASURED, GRIPPER, MODEL_RADIUS)
    plot(fig, [(0.0, 0.0), *MEASURED], gripper, MODEL_RADIUS)
    gaps, angles = fig.axes
    labels = [t.get_text() for t in gaps.get_legend().get_texts()]
    assert labels[:2] == ["measured", "touching (not fit)"]
    assert labels[2] == "gap = 2 · 86.8 mm · sin(1.272° · (pos + 1.0) / 2)"
    assert "the model's radius 80.9 mm" in labels[3]
    assert [t.get_text() for t in angles.get_legend().get_texts()][1] == "θ = 1.272°/unit · (pos + 1.0)"
    assert "radius 86.8 mm" in fig._suptitle.get_text()
    fig.savefig(tmp_path / "gripper.png")
    assert (tmp_path / "gripper.png").stat().st_size > 0


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
    shown: list[list] = []

    def ask(prompt):
        prompts.append(prompt)
        return next(answers)

    with client.session():
        measurements = measure(
            client, "right", [10.0, 25.0, 40.0, 55.0, 70.0], ask=ask, say=said.append, on_measure=shown.append
        )
    # 25 skipped; a bad answer and a negative one asked again; q at 55
    assert measurements == [(10.0, 0.0125), (40.0, 0.03)]
    assert prompts[0].startswith("gripper.pos 10.0:") and prompts[-1].startswith("gripper.pos 55.0:")
    assert said == ["not a number: 'x'", "a gap cannot be negative"]
    assert shown == [measurements[:1], measurements]  # after each measurement, for the plot
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
    measured = fit_map([(pos, gap_at(pos)) for pos in (10.0, 40.0, 70.0)], GRIPPER, MODEL_RADIUS)
    model.gripper_maps["right"] = measured
    model.save(tmp_path / "with_gripper.urdf")
    loaded = XleRobotModel.load(tmp_path / "with_gripper.urdf")
    assert loaded.gripper_maps == {"right": measured}
    np.testing.assert_array_equal(loaded.estimates, model.estimates)
    q = model.joint_vector()
    np.testing.assert_allclose(loaded.pinch_point("right", q), model.pinch_point("right", q), atol=1e-12)
