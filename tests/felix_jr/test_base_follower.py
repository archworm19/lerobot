import contextlib
import json
import logging

import pytest

from lerobot.felix_jr import base_follower
from lerobot.felix_jr.base_follower import (
    BASE_MOTOR_IDS,
    DEFAULT_JOINT_P_GAINS,
    HEAD_MOTOR_IDS,
    SO101FollowerWithBase,
)
from lerobot.felix_jr.home import make_follower
from lerobot.robots.so_follower import SO101FollowerConfig
from lerobot.robots.so_follower.so_follower import SOFollower

ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def make_robot(tmp_path, with_base=False, with_head=False) -> SO101FollowerWithBase:
    config = SO101FollowerConfig(port="/dev/null", id="t", calibration_dir=tmp_path, use_degrees=True)
    return SO101FollowerWithBase(config, with_base=with_base, with_head=with_head)


class FakeBus:
    """Stands in for the servo bus: records what the follower would send, talks to no hardware."""

    default_baudrate = 1_000_000
    model_baudrate_table = {"sts3215": {1_000_000: 0, 500_000: 1}}

    def __init__(self, motors, present=None, fail_goal_write=False):
        self.motors = motors
        self.calls: list[tuple] = []
        self.is_connected = True
        self.present = present or {}  # {baudrate: [ids that answer a ping]}
        self.baudrate = self.default_baudrate
        self.fail_goal_write = fail_goal_write
        self.start_position = 2000

    @contextlib.contextmanager
    def torque_disabled(self, motors=None):
        self.calls.append(("torque_off", motors))
        yield
        self.calls.append(("torque_on", motors))

    def disable_torque(self, motors=None):
        self.calls.append(("disable_torque", motors))

    def configure_motors(self):
        self.calls.append(("configure_motors",))

    def write(self, register, motor, value, **kwargs):
        if register == "Goal_Position" and self.fail_goal_write and value != self.start_position:
            raise ConnectionError("lost the servo")
        self.calls.append((register, motor, value))

    def read(self, register, motor, **kwargs):
        return self.start_position

    def set_timeout(self, *_):
        pass

    def set_baudrate(self, baudrate):
        self.baudrate = baudrate

    def ping(self, motor):
        if isinstance(motor, str):  # a servo addressed by name, after it was given its id
            return 777
        return 777 if motor in self.present.get(self.baudrate, []) else None

    def setup_motor(self, name, initial_baudrate=None, initial_id=None):
        self.calls.append(("setup_motor", name, initial_baudrate, initial_id))

    def writes(self, register):
        return [(c[1], c[2]) for c in self.calls if c[0] == register]


class TestMotorLayout:
    def test_arm_only_by_default(self, tmp_path):
        robot = make_robot(tmp_path)
        assert list(robot.bus.motors) == ARM
        assert robot.extra_motors == []

    def test_ids_of_every_combination_are_unique_and_as_documented(self, tmp_path):
        robot = make_robot(tmp_path, with_base=True, with_head=True)
        ids = {name: m.id for name, m in robot.bus.motors.items()}
        assert [ids[n] for n in ARM] == [1, 2, 3, 4, 5, 6]
        assert ids["head_neck"] == 7 and ids["head_eyes"] == 8  # neck turns, eyes tilt
        assert ids["base_left_wheel"] == 9 and ids["base_right_wheel"] == 10
        assert len(set(ids.values())) == len(ids)
        assert HEAD_MOTOR_IDS == {"head_neck": 7, "head_eyes": 8} and BASE_MOTOR_IDS["base_left_wheel"] == 9

    def test_head_only_and_base_only(self, tmp_path):
        assert robot_names(make_robot(tmp_path, with_head=True)) == ARM + ["head_neck", "head_eyes"]
        base_only = robot_names(make_robot(tmp_path, with_base=True))
        assert base_only == ARM + ["base_left_wheel", "base_right_wheel"]

    def test_extra_servos_are_in_the_bus_calibration_but_not_the_saved_one(self, tmp_path):
        robot = make_robot(tmp_path, with_head=True)
        assert {"head_neck", "head_eyes"} <= set(robot.bus.calibration)
        assert "head_neck" not in robot.calibration  # what is saved to disk stays arm-only

    def test_calibrating_with_extra_servos_is_refused(self, tmp_path):
        with pytest.raises(NotImplementedError):
            make_robot(tmp_path, with_head=True).calibrate()


def robot_names(robot) -> list[str]:
    return list(robot.bus.motors)


class TestObservation:
    def test_extra_servos_are_dropped_from_the_arm_observation(self, tmp_path, monkeypatch):
        everything = {f"{n}.pos": 1.0 for n in ARM + ["head_neck", "head_eyes", "base_left_wheel"]}
        monkeypatch.setattr(SOFollower, "get_observation", lambda self: dict(everything))
        obs = make_robot(tmp_path, with_base=True, with_head=True).get_observation()
        assert sorted(obs) == sorted(f"{n}.pos" for n in ARM)


class TestHeadSigns:
    def test_default_is_positive_and_a_missing_file_warns(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING, logger=base_follower.__name__):
            robot = make_robot(tmp_path, with_head=True)
        assert robot.head_signs == {"head_neck": 1, "head_eyes": 1}
        assert "setup.head" in caplog.text

    def test_saved_signs_are_loaded_by_the_next_instance(self, tmp_path):
        path = make_robot(tmp_path, with_head=True).save_head_signs({"head_eyes": -1})
        assert path == tmp_path / "t_head.json"
        assert json.loads(path.read_text()) == {"head_neck": 1, "head_eyes": -1}
        assert make_robot(tmp_path, with_head=True).head_signs["head_eyes"] == -1

    def test_signs_are_not_loaded_when_the_head_is_not_used(self, tmp_path):
        make_robot(tmp_path, with_head=True).save_head_signs({"head_eyes": -1})
        assert make_robot(tmp_path).head_signs == {}


class TestConfigure:
    def configure(self, tmp_path, **kwargs):
        robot = make_robot(tmp_path, **kwargs)
        robot.bus = FakeBus(robot.bus.motors)
        robot.configure()
        return robot.bus

    def test_head_servos_run_in_position_mode_with_the_arm_gains(self, tmp_path):
        bus = self.configure(tmp_path, with_base=True, with_head=True)
        modes = dict(bus.writes("Operating_Mode"))
        assert modes["head_neck"] == modes["head_eyes"] == modes["shoulder_pan"]  # position mode
        assert modes["base_left_wheel"] != modes["head_neck"]  # wheels are in velocity mode
        assert dict(bus.writes("Goal_Velocity")) == {"base_left_wheel": 0, "base_right_wheel": 0}
        p_gains = [(m, v) for m, v in bus.writes("P_Coefficient")]
        assert ("head_eyes", 16) in p_gains and ("head_neck", 16) in p_gains

    def test_heavy_arm_joints_get_their_larger_gain_and_nothing_else_does(self, tmp_path):
        bus = self.configure(tmp_path, with_head=True)
        final = dict(bus.writes("P_Coefficient"))  # later writes override earlier ones
        for joint, gain in DEFAULT_JOINT_P_GAINS.items():
            assert final[joint] == gain
        assert final["wrist_flex"] == final["head_eyes"] == 16


class TestAssignId:
    def scanned(self, tmp_path, present):
        robot = make_robot(tmp_path, with_head=True)
        robot.bus = FakeBus(robot.bus.motors, present=present)
        return robot

    def setups(self, robot):
        return [c for c in robot.bus.calls if c[0] == "setup_motor"]

    def test_first_servo_is_found_by_scanning_and_addressed_by_its_current_id(self, tmp_path):
        robot = self.scanned(tmp_path, {1_000_000: [1]})  # one factory-fresh servo, id 1
        robot.setup_head_motor("head_eyes")
        assert self.setups(robot) == [("setup_motor", "head_eyes", 1_000_000, 1)]

    def test_refuses_when_the_arm_is_still_on_the_chain(self, tmp_path):
        # the bug this guards against: the stock broadcast ping picked an arm servo and renamed it
        robot = self.scanned(tmp_path, {1_000_000: [1, 2, 3, 4, 5, 6]})
        with pytest.raises(RuntimeError, match="exactly one new servo"):
            robot.setup_head_motor("head_eyes")
        assert self.setups(robot) == []

    def test_refuses_when_nothing_answers(self, tmp_path):
        robot = self.scanned(tmp_path, {})
        with pytest.raises(RuntimeError, match="exactly one new servo"):
            robot.setup_head_motor("head_eyes")
        assert self.setups(robot) == []

    def test_second_servo_is_found_by_scanning_for_the_one_that_is_new(self, tmp_path):
        # the eyes already answer as id 8 at 1 Mbaud; a factory-fresh servo answers as id 1 at 500k
        robot = self.scanned(tmp_path, {1_000_000: [8], 500_000: [1]})
        robot.setup_head_motor("head_neck", configured=("head_eyes",))
        assert self.setups(robot) == [("setup_motor", "head_neck", 500_000, 1)]

    def test_stops_when_nothing_new_is_on_the_chain(self, tmp_path):
        robot = self.scanned(tmp_path, {1_000_000: [8]})
        with pytest.raises(RuntimeError, match="exactly one new servo"):
            robot.setup_head_motor("head_neck", configured=("head_eyes",))
        assert self.setups(robot) == []

    def test_stops_when_two_unknown_servos_answer(self, tmp_path):
        robot = self.scanned(tmp_path, {1_000_000: [8, 1, 2]})
        with pytest.raises(RuntimeError, match="exactly one new servo"):
            robot.setup_head_motor("head_neck", configured=("head_eyes",))
        assert self.setups(robot) == []

    def test_wheels_use_the_same_safe_path(self, tmp_path):
        robot = make_robot(tmp_path, with_base=True)
        robot.bus = FakeBus(robot.bus.motors, present={1_000_000: [1, 2, 3]})
        with pytest.raises(RuntimeError, match="exactly one new servo"):
            robot.setup_base_motor("base_right_wheel")


class TestNudgeHead:
    def robot_with_fake_bus(self, tmp_path, monkeypatch, **bus_kwargs):
        monkeypatch.setattr(base_follower.time, "sleep", lambda *_: None)
        robot = make_robot(tmp_path, with_head=True)
        robot.bus = FakeBus(robot.bus.motors, **bus_kwargs)
        return robot

    def test_moves_a_few_ticks_forward_and_back_and_reports_the_direction(self, tmp_path, monkeypatch):
        robot = self.robot_with_fake_bus(tmp_path, monkeypatch)
        assert robot.nudge_head("head_eyes", ticks=80) == 1
        goals = [v for _, v in robot.bus.writes("Goal_Position")]
        assert goals == [2000, 2080, 2000]  # hold, move, return
        assert robot.bus.calls[-1] == ("disable_torque", ["head_eyes"])

    def test_goes_the_other_way_near_the_end_of_the_range(self, tmp_path, monkeypatch):
        robot = self.robot_with_fake_bus(tmp_path, monkeypatch)
        robot.bus.start_position = 4050
        assert robot.nudge_head("head_neck", ticks=80) == -1
        assert [v for _, v in robot.bus.writes("Goal_Position")] == [4050, 3970, 4050]

    def test_torque_is_switched_off_even_if_a_write_fails(self, tmp_path, monkeypatch):
        robot = self.robot_with_fake_bus(tmp_path, monkeypatch, fail_goal_write=True)
        with pytest.raises(ConnectionError):
            robot.nudge_head("head_eyes")
        assert robot.bus.calls[-1] == ("disable_torque", ["head_eyes"])


class FollowingBus(FakeBus):
    """A fake bus whose head servos reach any goal they are given (or stay put if `stuck`), and whose wheels
    report the speed they were last commanded."""

    def __init__(self, motors, stuck=False, **kwargs):
        super().__init__(motors, **kwargs)
        self.stuck = stuck
        self.wheel_raw = {"base_left_wheel": 0, "base_right_wheel": 0}
        self.fail_on_nth_wheel_write = None
        self.wheel_writes = 0

    def write(self, register, motor, value, **kwargs):
        super().write(register, motor, value, **kwargs)
        if register == "Goal_Position" and not self.stuck:
            self.start_position = value

    def sync_write(self, register, values, **kwargs):
        self.wheel_writes += 1
        if self.fail_on_nth_wheel_write == self.wheel_writes:
            raise ConnectionError("lost the wheels")
        self.calls.append(("sync_write", register, dict(values)))
        self.wheel_raw.update(values)

    def sync_read(self, register, motors, **kwargs):
        return {m: self.wheel_raw[m] for m in motors}


class TestWiggleHead:
    def robot(self, tmp_path, monkeypatch, signs=None, start=2000, **bus_kwargs):
        monkeypatch.setattr(base_follower.time, "sleep", lambda *_: None)
        robot = make_robot(tmp_path, with_head=True)
        robot.head_signs.update(signs or {})
        robot.bus = FollowingBus(robot.bus.motors, **bus_kwargs)
        robot.bus.start_position = start
        return robot

    def test_goes_the_positive_way_then_the_other_way_and_returns(self, tmp_path, monkeypatch):
        robot = self.robot(tmp_path, monkeypatch)
        to_positive, to_negative = robot.wiggle_head("head_eyes", degrees=9.0)  # 9 deg = 102 ticks
        assert [v for _, v in robot.bus.writes("Goal_Position")] == [2102, 2000, 1898, 2000]
        assert (to_positive, to_negative) == (pytest.approx(9.0, abs=0.1), pytest.approx(9.0, abs=0.1))

    def test_the_saved_sign_flips_which_raw_direction_is_positive(self, tmp_path, monkeypatch):
        robot = self.robot(tmp_path, monkeypatch, signs={"head_neck": -1})
        to_positive, to_negative = robot.wiggle_head("head_neck", degrees=9.0)
        assert [v for _, v in robot.bus.writes("Goal_Position")] == [1898, 2000, 2102, 2000]
        # measured in the semantic directions (down / left), so both are still positive
        assert to_positive > 8.9 and to_negative > 8.9

    def test_the_servo_keeps_holding_afterwards(self, tmp_path, monkeypatch):
        robot = self.robot(tmp_path, monkeypatch)
        robot.wiggle_head("head_eyes")
        assert not [c for c in robot.bus.calls if c[0] in ("disable_torque", "torque_off")]

    def test_goals_are_clamped_to_the_servo_range_and_the_shortfall_is_reported(self, tmp_path, monkeypatch):
        robot = self.robot(tmp_path, monkeypatch, start=4090)
        to_positive, _ = robot.wiggle_head("head_eyes", degrees=9.0)
        assert max(v for _, v in robot.bus.writes("Goal_Position")) == 4095
        assert to_positive == pytest.approx(5 * 360 / 4096, abs=0.01)  # could only move 5 ticks

    def test_a_servo_that_does_not_move_reports_no_motion(self, tmp_path, monkeypatch):
        robot = self.robot(tmp_path, monkeypatch, stuck=True)
        assert robot.wiggle_head("head_eyes") == (0.0, 0.0)


class TestExerciseBase:
    def robot(self, tmp_path, monkeypatch):
        monkeypatch.setattr(base_follower.time, "sleep", lambda *_: None)
        robot = make_robot(tmp_path, with_base=True)
        robot.base_signs.update({"base_left_wheel": -1})  # mounted mirrored
        robot.bus = FollowingBus(robot.bus.motors)
        return robot

    def test_runs_each_wheel_alone_then_both_and_measures_what_it_commanded(self, tmp_path, monkeypatch):
        results = self.robot(tmp_path, monkeypatch).exercise_base(degps=40.0)
        assert [r["label"] for r in results] == [
            "left wheel forward",
            "left wheel backward",
            "right wheel forward",
            "right wheel backward",
            "both forward",
            "both backward",
        ]
        for r in results:  # raw readings are un-signed and scaled back, so measured == commanded
            assert r["measured"] == pytest.approx(r["commanded"], abs=0.1)
        assert results[0]["commanded"] == (40.0, 0.0) and results[5]["commanded"] == (-40.0, -40.0)

    def test_the_wheel_sign_is_applied_to_what_is_sent(self, tmp_path, monkeypatch):
        robot = self.robot(tmp_path, monkeypatch)
        robot.exercise_base(degps=40.0)
        sent = [c[2] for c in robot.bus.calls if c[0] == "sync_write" and c[2]["base_left_wheel"] != 0]
        assert sent[0]["base_left_wheel"] < 0  # "forward" for the mirrored left wheel is raw negative

    def test_wheels_are_stopped_at_the_end_and_even_after_a_failure(self, tmp_path, monkeypatch):
        robot = self.robot(tmp_path, monkeypatch)
        robot.exercise_base()
        assert robot.bus.wheel_raw == {"base_left_wheel": 0, "base_right_wheel": 0}

        failing = self.robot(tmp_path, monkeypatch)
        failing.bus.fail_on_nth_wheel_write = 3  # the third write is the second step's command
        with pytest.raises(ConnectionError):
            failing.exercise_base()
        assert failing.bus.wheel_raw == {"base_left_wheel": 0, "base_right_wheel": 0}


class TestMakeFollowerAutoDetect:
    def test_extras_are_detected_from_the_saved_sign_files(self, tmp_path):
        plain = make_follower("/dev/null", "t", calibration_dir=tmp_path)
        assert (plain.with_base, plain.with_head) == (False, False)
        (tmp_path / "t_head.json").write_text("{}")
        assert make_follower("/dev/null", "t", calibration_dir=tmp_path).with_head is True
        (tmp_path / "t_base.json").write_text("{}")
        both = make_follower("/dev/null", "t", calibration_dir=tmp_path)
        assert (both.with_base, both.with_head) == (True, True)

    def test_explicit_arguments_override_the_files(self, tmp_path):
        (tmp_path / "t_head.json").write_text("{}")
        robot = make_follower("/dev/null", "t", with_head=False, with_base=True, calibration_dir=tmp_path)
        assert (robot.with_base, robot.with_head) == (True, False)
