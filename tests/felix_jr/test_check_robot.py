"""`XleRobotModel.check_robot`: a calibration refuses a robot whose head directions changed, or whose arms were
recalibrated with `lerobot-calibrate`, since it was made (not one whose head home was re-recorded); `api.Harness` checks when it connects. Against the fake
server, through a real socket."""

import logging
import threading

import numpy as np
import pytest

pytest.importorskip("pinocchio")

from lerobot.felix_jr.calibration.xle_robot_model import (  # noqa: E402
    GripperMap,
    RobotMismatchError,
    XleRobotModel,
    arm_record,
)
from lerobot.felix_jr.harness import api  # noqa: E402
from lerobot.felix_jr.server.client import FelixClient  # noqa: E402
from lerobot.felix_jr.server.hardware import FakeHardware  # noqa: E402
from lerobot.felix_jr.server.server import RobotServer  # noqa: E402

SIGNS = {"head_neck": -1, "head_eyes": 1}  # the fake head's


def serve(hardware: FakeHardware):
    server = RobotServer(hardware)
    address = server.bind("tcp://127.0.0.1:*")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, address


def stop(server, thread) -> None:
    server.shutdown()
    thread.join(timeout=3.0)
    server.close()


@pytest.fixture
def endpoint():
    server, thread, address = serve(FakeHardware())
    yield address
    stop(server, thread)


@pytest.fixture
def client(endpoint):
    with FelixClient(endpoint, timeout_s=5.0) as c:
        for name in SIGNS:
            c.head_record(name, "home")
        yield c


def model_for(client, signs=SIGNS, arm_changes=None, gripper_changes=None) -> XleRobotModel:
    """A model made for the fake robot as it is now (its head homes, both arms' calibrations, a right gripper map),
    with `{arm: {motor: {field: value}}}` / `{field: value}` changed in its copies."""
    refs = {name: servo["home"] for name, servo in client.head_state().items()}
    arms = {arm_id.removesuffix("_arm"): arm_record(m) for arm_id, m in client.arm_calibration().items()}
    for arm, motors in (arm_changes or {}).items():
        for motor, fields in motors.items():
            arms[arm][motor] |= fields
    maps = {}
    if "right" in arms:
        maps["right"] = GripperMap(1.5, 1.27, arm_record(client.arm_calibration()["right_arm"])["gripper"])
        maps["right"].calibration.update(gripper_changes or {})
    return XleRobotModel.from_urdf(head_signs=signs, head_refs=refs, gripper_maps=maps, arm_calibration=arms)


def test_a_calibration_made_for_the_robot_passes(client):
    model_for(client).check_robot(client)


def test_a_re_recorded_or_missing_head_home_is_not_a_mismatch(client):
    # the model turns raw head ticks into angles with its own references and fitted zeros; the home is where the head
    # rests (a calibration started from an older one keeps that one's references, e.g. 967 against a home of 1475)
    model = model_for(client)
    client.head_move("head_neck", delta_deg=20.0)
    client.head_record("head_neck", "home")
    assert client.head_state()["head_neck"]["home"] != model.head_refs["head_neck"]
    model.check_robot(client)
    server, thread, address = serve(FakeHardware())  # nothing recorded
    try:
        with FelixClient(address, timeout_s=5.0) as fresh:
            XleRobotModel.from_urdf(
                head_signs=SIGNS, head_refs={"head_neck": 967, "head_eyes": 1324}
            ).check_robot(fresh)
    finally:
        stop(server, thread)


def test_flipped_head_directions_are_refused(client):
    model = model_for(client, signs={"head_neck": -1, "head_eyes": -1})
    with pytest.raises(
        RobotMismatchError, match=r"head_eyes: sign 1 on the robot, -1 in the calibration \(setup.head"
    ):
        model.check_robot(client)


@pytest.mark.parametrize("field", ["homing_offset", "drive_mode", "range_min", "range_max"])
def test_a_recalibrated_arm_is_refused(client, field):
    model = model_for(client, arm_changes={"left": {"elbow_flex": {field: 7}}})
    with pytest.raises(
        RobotMismatchError, match=rf"left arm: elbow_flex recalibrated on the robot: {field} \d+ \(was 7\)"
    ):
        model.check_robot(client)


def test_every_mismatch_is_listed(client):
    model = model_for(
        client,
        signs={"head_neck": 1, "head_eyes": 1},
        arm_changes={"right": {"shoulder_lift": {"range_min": 5}, "wrist_roll": {"homing_offset": 9}}},
        gripper_changes={"range_min": 2000},
    )
    with pytest.raises(RobotMismatchError) as error:
        model.check_robot(client)
    lines = [line.strip() for line in str(error.value).splitlines()[1:]]
    assert lines[0].startswith("head_neck: sign -1 on the robot, 1 in the calibration")
    assert lines[1].startswith("right arm: shoulder_lift recalibrated on the robot: range_min 0 (was 5)")
    assert lines[2].startswith("right arm: wrist_roll recalibrated on the robot: homing_offset 0 (was 9)")
    assert lines[3].startswith(
        "right gripper map: gripper recalibrated on the robot: range_min 2046 (was 2000)"
    )
    assert "gripper_step" in lines[3] and len(lines) == 4


def test_an_arm_the_robot_lacks_is_not_checked():
    server, thread, address = serve(FakeHardware(arms=("left_arm",)))
    try:
        with FelixClient(address, timeout_s=5.0) as client:
            for name in SIGNS:
                client.head_record(name, "home")
            model = model_for(client)  # copies the left arm only
            model.arm_calibration["right"] = {"gripper": {"range_min": 0}}  # type: ignore[dict-item]
            model.check_robot(client)
    finally:
        stop(server, thread)


def test_an_arm_without_a_snapshot_is_logged_not_refused(client, caplog):
    model = model_for(client)
    del model.arm_calibration["right"]
    model.set_estimates(np.where([p.arm == "right" for p in model.parameters], 0.01, 0.0))
    with caplog.at_level(logging.WARNING):
        model.check_robot(client)
    assert "no snapshot of the right arm's LeRobot calibration" in caplog.text
    assert "left arm" not in caplog.text


def test_the_snapshot_survives_saving(client, tmp_path):
    model = model_for(client)
    model.save(tmp_path / "calibration.urdf")
    loaded = XleRobotModel.load(tmp_path / "calibration.urdf")
    assert loaded.arm_calibration == model.arm_calibration and loaded.gripper_maps == model.gripper_maps
    loaded.check_robot(client)


def test_the_harness_checks_when_it_connects(client, endpoint):
    good = api.Harness(endpoint=endpoint, model=model_for(client))
    assert good.client.get_state()["right_arm"]
    bad = api.Harness(endpoint=endpoint, model=model_for(client, gripper_changes={"range_min": 2000}))
    with pytest.raises(RobotMismatchError):
        bad.client  # noqa: B018
    assert bad._client is None  # not kept: the next use checks again
    good.client.close()
