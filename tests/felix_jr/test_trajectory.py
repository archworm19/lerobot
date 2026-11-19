"""`trajectory`: a move as the robot server plans it, shared by `home.move_all` and the simulator's replays."""

import pytest

from lerobot.felix_jr.trajectory import STALE_GOAL, MovePlan, Settler, move_starts, move_time_trap


def test_a_move_starts_from_the_last_goal_unless_the_arm_is_far_from_it():
    measured = {"a.pos": 10.0, "b.pos": 50.0, "c.pos": 0.0}
    goal = {"a.pos": 11.0, "b.pos": 50.0 + STALE_GOAL + 1.0}
    starts = move_starts({"a.pos": 0.0, "b.pos": 0.0, "c.pos": 5.0}, goal, measured)
    assert starts == {"a.pos": 11.0, "b.pos": 50.0, "c.pos": 0.0}  # (b: moved by hand; c: no goal yet)


def test_every_joint_starts_and_stops_together_the_largest_move_setting_the_time():
    plan = MovePlan.plan(
        {"left": {"x": 0.0, "y": 0.0}, "right": {"z": 10.0}},
        {"left": {"x": 30.0, "y": -5.0}, "right": {"z": 0.0}},
        max_vel=20.0,
        max_accel=40.0,
    )
    assert plan.total_time == pytest.approx(move_time_trap(30.0, 20.0, 40.0))
    assert plan.waypoint(0.0) == {"left": {"x": 0.0, "y": 0.0}, "right": {"z": 10.0}}
    end = plan.waypoint(plan.total_time)
    assert end["left"] == pytest.approx({"x": 30.0, "y": -5.0}) and end["right"] == pytest.approx({"z": 0.0})
    half = plan.waypoint(plan.total_time / 2)  # (symmetric profiles: halfway at half the time)
    assert half["left"]["x"] == pytest.approx(15.0) and half["left"]["y"] == pytest.approx(-2.5)


def test_a_move_arrives_after_its_dwell_within_the_tolerance_or_fails_at_the_timeout():
    settler = Settler(timeout=1.0, dwell=0.2, began=0.0)
    assert settler.update(0.1, within=True) is None
    assert settler.update(0.2, within=False) is None  # (out again: the dwell starts over)
    assert settler.update(0.3, within=True) is None
    assert settler.update(0.5, within=True) is True
    late = Settler(timeout=1.0, dwell=0.2, began=0.0)
    assert late.update(0.5, within=False) is None and late.update(1.0, within=False) is False


def test_the_server_sends_the_plans_waypoints_and_settles_by_the_same_rule(monkeypatch):
    """`home.move_all` with a fake arm that is where it was last told to be: the first waypoint is the start (its
    last goal), the last the target, every one on the plan; it arrives."""
    from lerobot.felix_jr import home

    class FakeArm:
        id = "right"

        def __init__(self) -> None:
            self.at = {"shoulder_pan.pos": 0.0, "elbow_flex.pos": 5.0}
            self.sent: list[dict[str, float]] = []

        def get_observation(self) -> dict[str, float]:
            return dict(self.at)

        def send_action(self, action: dict[str, float]) -> None:
            self.sent.append(dict(action))
            self.at.update(action)

    arm = FakeArm()
    monkeypatch.setattr(home, "_last_goal", {"right": {"shoulder_pan.pos": 0.5, "elbow_flex.pos": 5.0}})
    target = {"shoulder_pan.pos": 4.0, "elbow_flex.pos": 3.0}
    assert home.move_all({"right": arm}, {"right": target}, max_vel=60.0, max_accel=120.0)
    plan = MovePlan.plan(
        {"right": {"shoulder_pan.pos": 0.5, "elbow_flex.pos": 5.0}}, {"right": target}, 60.0, 120.0
    )
    assert arm.sent[0] == pytest.approx(plan.waypoint(0.0)["right"], abs=0.05)
    assert arm.sent[-1] == target and home._last_goal["right"] == target
    pan = [a["shoulder_pan.pos"] for a in arm.sent]
    assert pan == sorted(pan)  # (monotonic along the trapezoid)
