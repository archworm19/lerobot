import pytest

from lerobot.felix_jr.diagnostics.set_servo_id import BAUDRATE, change_servo_id, probe

OK, NONE, GARBLED = 0, -6, -4  # stand-ins for the comm results of a ping


class FakePacketHandler:
    def __init__(self, bus):
        self.bus = bus

    def ping(self, _port, servo_id):
        return 777, self.bus.responses.get(servo_id, NONE), 0

    def getTxRxResult(self, comm):  # noqa: N802 (name of the SDK method being faked)
        texts = {
            OK: "Communication success!",
            NONE: "There is no status packet!",
            GARBLED: "Incorrect status packet!",
        }
        return texts[comm]


class FakeBus:
    """A chain of servos; `responses` maps id -> how a ping of that id comes back."""

    _comm_success = OK
    port_handler = None

    def __init__(self, responses):
        self.responses = dict(responses)
        self.packet_handler = FakePacketHandler(self)
        self.setups: list[tuple] = []

    def set_baudrate(self, baudrate):
        assert baudrate == BAUDRATE

    def setup_motor(self, name, initial_baudrate, initial_id):
        self.setups.append((name, initial_baudrate, initial_id))
        self.responses[initial_id] = NONE  # the servo leaves its old id ...
        self.responses[self.new_id] = OK  # ... and answers as the new one

    new_id = None


def run(responses, current, new, answer=True):
    bus = FakeBus(responses)
    bus.new_id = new
    change_servo_id(bus, current, new, confirm=lambda _q: answer)
    return bus


def test_changes_a_servo_that_answers_cleanly_to_a_free_id():
    bus = run({8: OK, 3: OK, 4: OK}, current=8, new=1)
    assert bus.setups == [("servo", BAUDRATE, 8)]
    assert probe(bus, 1) == "ok" and probe(bus, 8) == "none"


def test_refuses_when_several_servos_share_the_current_id():
    bus = FakeBus({8: GARBLED, 3: OK})
    with pytest.raises(RuntimeError, match="Several servos answer as id 8"):
        change_servo_id(bus, 8, 1, confirm=lambda _q: True)
    assert bus.setups == []


def test_refuses_when_nothing_answers_as_the_current_id():
    bus = FakeBus({3: OK})
    with pytest.raises(RuntimeError, match="No servo answers as id 8"):
        change_servo_id(bus, 8, 1, confirm=lambda _q: True)
    assert bus.setups == []


def test_refuses_when_the_new_id_is_taken():
    bus = FakeBus({8: OK, 1: OK})
    with pytest.raises(RuntimeError, match="already in use"):
        change_servo_id(bus, 8, 1, confirm=lambda _q: True)
    assert bus.setups == []


def test_refuses_when_the_new_id_is_garbled_too():
    bus = FakeBus({8: OK, 1: GARBLED})  # something is there, even if unreadable
    with pytest.raises(RuntimeError, match="already in use"):
        change_servo_id(bus, 8, 1, confirm=lambda _q: True)


def test_declining_the_confirmation_changes_nothing():
    bus = FakeBus({8: OK})
    with pytest.raises(RuntimeError, match="Cancelled"):
        change_servo_id(bus, 8, 1, confirm=lambda _q: False)
    assert bus.setups == []


def test_same_id_is_rejected():
    with pytest.raises(RuntimeError, match="the same"):
        change_servo_id(FakeBus({8: OK}), 8, 8, confirm=lambda _q: True)
