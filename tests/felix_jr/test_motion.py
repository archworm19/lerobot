"""Motion core, sessions and the server's motion commands, against simulated arms (no hardware).

`FakeArms` shortens motions by `time_scale` and counts `violations`: a use of the serial bus from a second thread
while a trajectory runs, which on the real bus would corrupt packets.
"""

import contextlib
import threading
import time

import pytest

from lerobot.felix_jr.home import DEFAULT_MAX_ACCEL as HOME_MAX_ACCEL, DEFAULT_MAX_VEL as HOME_MAX_VEL
from lerobot.felix_jr.server import motion
from lerobot.felix_jr.server.client import FelixClient
from lerobot.felix_jr.server.hardware import FakeArms, FakeHardware
from lerobot.felix_jr.server.motion import (
    MAX_BASE_SECONDS,
    MotionBusyError,
    MotionController,
    validate_targets,
)
from lerobot.felix_jr.server.protocol import (
    MotionBusyError as ClientBusyError,
    MotionFailedError,
    ServerError,
)
from lerobot.felix_jr.server.server import RobotServer

HOME = FakeArms.HOME
PAN = "shoulder_pan.pos"
FAR = {"left_arm": {PAN: 90.0}, "right_arm": {PAN: -90.0}}


def wait_until(predicate, timeout=5.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def at_home(arms):
    return all(pose == HOME for pose in arms.pos.values())


@pytest.fixture
def make_controller():
    made = []

    def make(time_scale=0.02, arms=None, **kwargs):
        arms = arms or FakeArms(time_scale=time_scale)
        controller = MotionController(arms, watch_period_s=0.05, **kwargs)
        made.append(controller)
        return controller, arms

    yield make
    for controller in made:
        controller.stop("test over")
        controller.shutdown(go_home=False)


def test_the_defaults_match_move_all():
    assert (motion.DEFAULT_MAX_VEL, motion.DEFAULT_MAX_ACCEL) == (HOME_MAX_VEL, HOME_MAX_ACCEL)


def test_validate_targets_rejects_bad_requests():
    ids = ["left_arm", "right_arm"]
    assert validate_targets({"left_arm": {PAN: 3}}, ids) == {"left_arm": {PAN: 3.0}}
    for bad, message in [
        ({}, "at least one arm"),
        ({"middle_arm": {PAN: 1.0}}, "unknown arm"),
        ({"left_arm": {}}, "at least one joint"),
        ({"left_arm": {"shoulder_pan": 1.0}}, "unknown joint"),
        ({"left_arm": {"elbow.pos": 1.0}}, "unknown joint"),
        ({"left_arm": {PAN: float("nan")}}, "finite"),
        ({"left_arm": {PAN: True}}, "finite"),
        ({"left_arm": {PAN: "10"}}, "finite"),
        ({"left_arm": {PAN: 400.0}}, "outside"),
        ({"left_arm": {"gripper.pos": 120.0}}, "outside"),
        ({"left_arm": {"gripper.pos": -5.0}}, "outside"),
    ]:
        with pytest.raises(ValueError, match=message):
            validate_targets(bad, ids)


def test_move_joints_moves_both_arms_together_with_the_default_limits(make_controller):
    controller, arms = make_controller()
    controller.move_joints(FAR)
    assert wait_until(lambda: not controller.status()["moving"])
    status = controller.status()
    assert (
        status["state"] == "done"
        and status["result"] == {"reached": True}
        and status["kind"] == "move_joints"
    )
    assert arms.pos["left_arm"][PAN] == 90.0 and arms.pos["right_arm"][PAN] == -90.0
    targets, vel, accel = arms.trajectories[-1]
    assert set(targets) == {"left_arm", "right_arm"}  # one trajectory for both arms
    assert (vel, accel) == (60.0, 120.0)
    assert controller.state()["left_arm"][PAN] == 90.0


def test_bad_limits_and_arms_without_a_home_are_refused(make_controller):
    controller, _ = make_controller()
    with pytest.raises(ValueError, match="max_vel"):
        controller.move_joints(FAR, max_vel=500.0)
    with pytest.raises(ValueError, match="max_vel"):
        controller.move_joints(FAR, max_accel=0.0)

    class NoHomeForRight(FakeArms):
        def homes(self):
            return {k: v for k, v in super().homes().items() if k == "left_arm"}

    controller, _ = make_controller(arms=NoHomeForRight(time_scale=0.02))
    with pytest.raises(ValueError, match="right_arm has no recorded home"):
        controller.move_joints(FAR)
    controller.move_joints({"left_arm": {PAN: 5.0}})  # the arm with a home is fine


def test_a_second_motion_is_refused_and_stop_holds_the_arms_where_they_are(make_controller):
    controller, arms = make_controller(time_scale=1.0)
    controller.move_joints(FAR)
    with pytest.raises(MotionBusyError):
        controller.move_joints(FAR)
    with pytest.raises(MotionBusyError):
        controller.home()
    with pytest.raises(MotionBusyError), controller.exclusive():
        pass
    time.sleep(0.3)
    assert controller.stop() == {"was_moving": True}
    assert wait_until(lambda: not controller.status()["moving"])
    status = controller.status()
    assert status["state"] == "stopped" and status["reason"] == "client stop"
    held = arms.pos["left_arm"][PAN]
    assert 0.0 < held < 90.0
    time.sleep(0.2)
    assert arms.pos["left_arm"][PAN] == held  # it stayed put
    assert controller.stop() == {"was_moving": False}
    with controller.exclusive():  # the bus is free again
        pass


def test_state_during_a_motion_never_touches_the_bus(make_controller):
    controller, arms = make_controller(time_scale=1.0)
    controller.move_joints(FAR)
    readers = [threading.Thread(target=lambda: [controller.state() for _ in range(40)]) for _ in range(3)]
    for reader in readers:
        reader.start()
    seen = []
    for _ in range(15):
        state = controller.state()
        assert state["moving"] and state["motion"]["state"] == "running"
        seen.append(state["left_arm"][PAN])
        time.sleep(0.03)
    for reader in readers:
        reader.join()
    assert arms.violations == 0
    assert seen[-1] > seen[0]  # the snapshot is refreshed by the motion as it goes
    assert state["state_age_s"] < 1.0
    controller.stop()


@pytest.mark.slow
def test_the_watchdog_stops_a_motion_when_requests_stop_arriving(make_controller):
    controller, arms = make_controller(time_scale=1.0, watchdog_s=0.3)
    controller.move_joints(FAR)
    for _ in range(8):  # a live client keeps it going
        controller.touch()
        time.sleep(0.1)
    assert controller.status()["moving"]
    assert wait_until(lambda: not controller.status()["moving"], timeout=3.0)
    status = controller.status()
    assert status["state"] == "stopped" and status["reason"].startswith("watchdog")
    assert arms.pos["left_arm"][PAN] < 90.0


@pytest.mark.slow
def test_a_session_that_asked_for_it_sends_the_arms_home_when_the_client_goes_silent(make_controller):
    controller, arms = make_controller(session_timeout_s=0.3)
    controller.open_session("home")
    controller.move_joints(FAR)
    assert wait_until(lambda: not controller.status()["moving"])
    assert arms.pos["left_arm"][PAN] == 90.0
    for _ in range(8):  # heartbeats keep the session alive well past the timeout
        controller.touch()
        time.sleep(0.1)
    assert arms.pos["left_arm"][PAN] == 90.0
    assert wait_until(lambda: at_home(arms), timeout=3.0)  # now it is silent
    assert controller.status()["kind"] == "home"
    assert arms.trajectories[-1][1:] == (motion.REST_MAX_VEL, motion.REST_MAX_ACCEL)  # slowly
    assert controller.open_session("hold")["on_disconnect"] == "hold"


@pytest.mark.slow
def test_no_session_or_a_hold_session_never_moves_the_arms_by_itself(make_controller):
    controller, arms = make_controller(session_timeout_s=0.2, watchdog_s=0.2)
    controller.move_joints(FAR)
    assert wait_until(lambda: not controller.status()["moving"])
    time.sleep(0.8)
    assert arms.pos["left_arm"][PAN] == 90.0
    controller.open_session("hold")
    time.sleep(0.8)
    assert arms.pos["left_arm"][PAN] == 90.0
    with pytest.raises(ValueError, match="on_disconnect"):
        controller.open_session("explode")


@pytest.mark.slow
def test_a_lost_session_stops_a_running_motion_then_goes_home(make_controller):
    controller, arms = make_controller(time_scale=0.3, session_timeout_s=0.3, watchdog_s=10.0)
    controller.open_session("home")
    controller.move_joints(FAR)
    assert wait_until(lambda: controller.status()["kind"] == "home", timeout=5.0)
    assert wait_until(lambda: at_home(arms), timeout=8.0)


def test_shutdown_stops_the_motion_and_goes_home_slowly(make_controller):
    controller, arms = make_controller(time_scale=0.3)
    controller.move_joints(FAR)
    time.sleep(0.2)
    controller.shutdown(go_home=True)
    assert at_home(arms)
    assert arms.trajectories[-1][1:] == (motion.REST_MAX_VEL, motion.REST_MAX_ACCEL)
    assert arms.violations == 0


def test_shutdown_can_leave_the_arms_alone(make_controller):
    controller, arms = make_controller()
    controller.move_joints(FAR)
    assert wait_until(lambda: not controller.status()["moving"])
    controller.shutdown(go_home=False)
    assert arms.pos["left_arm"][PAN] == 90.0


def test_going_home_includes_the_head_and_stops_the_wheels(make_controller):
    from lerobot.felix_jr.server.hardware import FakeHeadBackend
    from lerobot.felix_jr.server.head import HeadController, RangeStore

    head = HeadController(FakeHeadBackend(), {"head_neck": -1, "head_eyes": 1}, RangeStore())
    head.record("head_eyes", "home")
    head.move("head_eyes", delta_deg=10.0)
    arms = FakeArms(time_scale=0.02)
    arms.set_base_velocity(30.0, 30.0)
    controller = MotionController(arms, head=head, watch_period_s=0.05)
    try:
        controller.move_joints(FAR)
        assert wait_until(lambda: not controller.status()["moving"])
        controller.home()
        assert wait_until(lambda: not controller.status()["moving"])
        assert at_home(arms)
        assert head.backend.read_ticks("head_eyes") == 2048  # its recorded home
        assert arms.wheels == (0.0, 0.0)
        assert "head" in controller.state()
    finally:
        controller.shutdown(go_home=False)


def test_wheel_commands_are_bounded_and_end_with_the_wheels_stopped(make_controller):
    controller, arms = make_controller()
    for bad in [(1.0, 1.0, 0.0), (1.0, 1.0, MAX_BASE_SECONDS + 1), (float("nan"), 1.0, 1.0)]:
        with pytest.raises(ValueError):
            controller.base_velocity(*bad)
    controller.base_velocity(40.0, -40.0, 0.5)
    time.sleep(0.2)
    assert arms.wheels == (40.0, -40.0)
    assert wait_until(lambda: not controller.status()["moving"])
    status = controller.status()
    assert status["state"] == "done" and status["result"]["measured"] == [40.0, -40.0]
    assert arms.wheels == (0.0, 0.0)

    controller.base_velocity(40.0, 40.0, 5.0)
    time.sleep(0.2)
    controller.stop()
    assert wait_until(lambda: not controller.status()["moving"])
    assert arms.wheels == (0.0, 0.0) and controller.status()["state"] == "stopped"
    assert arms.violations == 0


def test_a_failing_motion_is_reported_as_failed_and_frees_the_bus(make_controller):
    class Broken(FakeArms):
        def run_trajectory(self, *args, **kwargs):
            raise RuntimeError("servo exploded")

    controller, _ = make_controller(arms=Broken())
    controller.move_joints(FAR)
    assert wait_until(lambda: not controller.status()["moving"])
    status = controller.status()
    assert status["state"] == "failed" and "servo exploded" in status["reason"]
    with controller.exclusive():
        pass


# ---- over the wire ----------------------------------------------------------------------------------------


@contextlib.contextmanager
def serve(fake):
    server = RobotServer(fake)
    address = server.bind("tcp://127.0.0.1:*")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with FelixClient(address, timeout_s=5.0) as client:
            yield fake, client
    finally:
        server.shutdown()
        thread.join(timeout=3.0)
        server.close()


@pytest.fixture
def robot():
    with serve(
        FakeHardware(arm_time_scale=0.02, watchdog_s=0.4, session_timeout_s=0.6, watch_period_s=0.05)
    ) as r:
        yield r


@pytest.fixture
def slow_robot():
    with serve(
        FakeHardware(arm_time_scale=0.5, watchdog_s=0.4, session_timeout_s=0.6, watch_period_s=0.05)
    ) as r:
        yield r


def test_the_client_moves_the_arms_and_goes_home(robot):
    fake, client = robot
    status = client.move_joints({"left_arm": {PAN: 20.0, "gripper.pos": 60.0}})
    assert status["state"] == "done"
    assert (
        client.get_state()["left_arm"][PAN] == 20.0 and client.get_state()["left_arm"]["gripper.pos"] == 60.0
    )
    assert client.home()["state"] == "done"
    assert at_home(fake.arm_backend)
    assert client.get_homes()["right_arm"] == HOME
    assert client.base_velocity(30.0, 30.0, 0.3)["result"]["measured"] == [30.0, 30.0]


def test_bad_requests_come_back_as_errors_and_the_server_carries_on(robot):
    _, client = robot
    with pytest.raises(ServerError, match="unknown joint"):
        client.move_joints({"left_arm": {"nose.pos": 1.0}})
    with pytest.raises(ServerError, match="finite"):
        client.move_joints({"left_arm": {PAN: float("inf")}})
    assert client.motion_status()["state"] == "idle"


def test_the_client_can_stop_and_ask_while_a_motion_runs(slow_robot):
    fake, client = slow_robot
    client.move_joints(FAR, wait=False)
    with pytest.raises(ClientBusyError):
        client.move_joints(FAR, wait=False)
    with pytest.raises(ClientBusyError):
        client.head_move("head_eyes", delta_deg=2.0)
    state = client.get_state()  # answered from the cache, mid-motion
    assert state["moving"] and state["motion"]["kind"] == "move_joints"
    assert client.capture().state["moving"]  # and a capture works too
    assert client.stop() == {"was_moving": True}
    assert client.wait_motion()["state"] == "stopped"
    assert fake.arm_backend.violations == 0
    assert client.head_move("head_eyes", delta_deg=2.0)["reached"]  # the head is usable again


def test_wait_motion_times_out_without_stopping_and_reports_failures(slow_robot):
    fake, client = slow_robot
    client.move_joints(FAR, wait=False)
    with pytest.raises(TimeoutError, match="still running"):
        client.wait_motion(timeout_s=0.2)
    assert client.motion_status()["moving"]
    client.stop()
    client.wait_motion()

    def explode(*args, **kwargs):
        raise RuntimeError("servo exploded")

    fake.arm_backend.run_trajectory = explode
    with pytest.raises(MotionFailedError, match="servo exploded") as info:
        client.move_joints(FAR)
    assert info.value.status["state"] == "failed"


@pytest.mark.slow
def test_the_watchdog_works_through_the_server(slow_robot):
    _, client = slow_robot
    client.move_joints(FAR, wait=False)
    assert wait_until(lambda: client._call("motion_status")[0]["state"] == "stopped", timeout=0.1) or True
    time.sleep(1.0)  # no requests at all for longer than the watchdog (0.4 s)
    status = client.motion_status()
    assert status["state"] == "stopped" and status["reason"].startswith("watchdog")


def test_a_session_sends_the_robot_home_when_the_block_ends(robot):
    fake, client = robot
    with client.session():
        client.move_joints(FAR)
        assert fake.arm_backend.pos["left_arm"][PAN] == 90.0
    assert at_home(fake.arm_backend)
    assert not fake.motion._session_home  # closed cleanly


@pytest.mark.slow
def test_ctrl_c_in_a_session_stops_the_motion_and_goes_home(slow_robot):
    fake, client = slow_robot
    with pytest.raises(KeyboardInterrupt), client.session():
        client.move_joints(FAR, wait=False)
        time.sleep(0.3)
        raise KeyboardInterrupt
    assert at_home(fake.arm_backend)
    assert client.motion_status()["kind"] == "home"


def test_a_hold_session_leaves_the_robot_where_it_is(robot):
    fake, client = robot
    with client.session(on_disconnect="hold"):
        client.move_joints(FAR)
    assert fake.arm_backend.pos["left_arm"][PAN] == 90.0


@pytest.mark.slow
def test_the_heartbeat_keeps_an_idle_session_alive(robot):
    fake, client = robot
    with client.session(heartbeat_s=0.1):
        client.move_joints(FAR)
        time.sleep(1.5)  # well past the 0.6 s session timeout, with no other requests
        assert fake.arm_backend.pos["left_arm"][PAN] == 90.0
    assert at_home(fake.arm_backend)


@pytest.mark.slow
def test_a_client_that_dies_without_ending_its_session_gets_the_robot_sent_home(robot):
    fake, client = robot
    client._call("session_open", on_disconnect="home")  # opened, but no heartbeat and no clean end
    client.move_joints(FAR)
    assert wait_until(lambda: at_home(fake.arm_backend), timeout=4.0)


@pytest.mark.slow
def test_a_client_without_a_session_is_never_moved_by_the_server(robot):
    fake, client = robot
    client.move_joints(FAR)
    time.sleep(1.5)
    assert fake.arm_backend.pos["left_arm"][PAN] == 90.0


@pytest.mark.slow
def test_a_rest_the_server_starts_itself_is_not_cut_short_by_the_watchdog(make_controller):
    controller, arms = make_controller(time_scale=0.3, session_timeout_s=0.3, watchdog_s=0.3)
    controller.open_session("home")
    controller.move_joints(FAR)
    assert wait_until(lambda: not controller.status()["moving"], timeout=5.0)
    assert wait_until(lambda: at_home(arms), timeout=8.0)  # the rest takes longer than the watchdog
    status = controller.status()
    assert status["kind"] == "home" and status["autonomous"] and status["state"] == "done"
    assert status["finished_at"] > 0


def test_head_wiggle_over_the_wire_and_it_is_refused_while_the_arms_move(slow_robot):
    _, client = slow_robot
    result = client.head_wiggle("head_eyes", 6.0)
    assert result["to_pos"] == pytest.approx(6.0, abs=0.1) and not result["blocked"]
    client.move_joints(FAR, wait=False)
    with pytest.raises(ClientBusyError):
        client.head_wiggle("head_eyes", 6.0)
    client.stop()
    status = client.wait_motion()
    assert status["state"] == "stopped" and status["finished_at"] > 0


def test_home_can_be_limited_to_some_arms(make_controller):
    controller, arms = make_controller()
    controller.move_joints(FAR)
    assert wait_until(lambda: not controller.status()["moving"])
    controller.home(arms=["left_arm"])
    assert wait_until(lambda: not controller.status()["moving"])
    assert arms.pos["left_arm"] == HOME and arms.pos["right_arm"][PAN] == -90.0
    assert set(arms.trajectories[-1][0]) == {"left_arm"}
    for bad in (["middle_arm"], []):
        with pytest.raises(ValueError, match="unknown arms"):
            controller.home(arms=bad)
