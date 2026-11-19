"""The servos' moves as the robot server plans them: synchronized trapezoids (`home.move_all`), no hardware.

Shared by the server (`home.move_all`, on the robot) and the simulator's replays (`sim2real.shadow`), so the two run a
move the same way: where it starts (`move_starts`), its plan (`MovePlan`: one duration for every joint, each its own
trapezoid, a waypoint at any time), and when it has arrived (`Settler`).
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

WAYPOINT_HZ = 30.0  # a waypoint sent this often
DWELL_S = 0.2  # every joint within the tolerance this long: arrived
STALE_GOAL = 10.0  # deg: if an arm is farther than this from its last commanded goal, start from where it is


def move_time_trap(displacement: float, max_vel: float, max_accel: float) -> float:
    """Minimum time for a trapezoidal velocity profile (accel = decel = `max_accel`) over `displacement`.

    If the joint can't reach `max_vel` before it has to brake, the profile is a triangle.
    With the same limits for every joint, only the largest displacement needs to be passed in.
    """
    d = abs(displacement)
    t_cruise = d / max_vel - max_vel / max_accel
    if t_cruise < 0:
        return float(2 * np.sqrt(d / max_accel))  # triangle: never reaches max_vel
    return float(t_cruise + 2 * max_vel / max_accel)


def schedule_trapezoid(displacement: float, total_time: float, max_accel: float) -> tuple[float, float]:
    """Stretch a joint's trapezoid so it takes exactly `total_time` (>= its own minimum time).

    Acceleration stays at `max_accel`; the cruise speed is lowered instead. Solving
    d = a*t1^2 + a*t1*(T - 2*t1) for t1 (the smaller root) gives the time spent accelerating.
    Returns (t_accel, v_peak). The triangle case falls out when the discriminant is 0 (t1 = T/2).
    """
    d = abs(displacement)
    if d == 0 or total_time <= 0:
        return 0.0, 0.0
    disc = max(0.0, total_time**2 - 4 * d / max_accel)  # clamp: float rounding at the limiting joint
    t_accel = (total_time - np.sqrt(disc)) / 2
    return float(t_accel), float(max_accel * t_accel)


def trapezoid_offset(
    t: float, displacement: float, total_time: float, t_accel: float, v_peak: float, max_accel: float
) -> float:
    """Signed distance travelled from the start after `t` seconds of a scheduled trapezoid."""
    d = abs(displacement)
    t = min(max(t, 0.0), total_time)
    if t < t_accel:
        s = 0.5 * max_accel * t**2
    elif t < total_time - t_accel:
        s = 0.5 * max_accel * t_accel**2 + v_peak * (t - t_accel)
    else:
        s = d - 0.5 * max_accel * (total_time - t) ** 2
    return float(np.sign(displacement) * s)


def move_starts(
    targets: Mapping[str, float], last_goal: Mapping[str, float], measured: Mapping[str, float]
) -> dict[str, float]:
    """Where each joint's move starts: its last commanded goal, or where it is.

    Where it is if the arm is `STALE_GOAL` from its goal (moved by hand, or the first move).

    The arm rests a few degrees off its goal (gravity, P-only control): starting from the measured pose would snap the
    servo's goal to that sagging position and make the arm jump.
    """
    return {
        k: last_goal[k] if k in last_goal and abs(last_goal[k] - measured[k]) <= STALE_GOAL else measured[k]
        for k in targets
    }


@dataclass(frozen=True)
class MovePlan:
    """Several arms' moves, all joints starting and stopping together (the largest move sets the duration).

    `starts` and `targets` per arm (`{arm: {"shoulder_pan.pos": deg, ...}}`); `waypoint(t)` where each joint is told
    to be `t` seconds in.
    """

    starts: dict[str, dict[str, float]]
    targets: dict[str, dict[str, float]]
    total_time: float
    max_accel: float
    profiles: dict[str, dict[str, tuple[float, float]]]  # per arm and joint: (t_accel, v_peak)

    @classmethod
    def plan(
        cls,
        starts: Mapping[str, Mapping[str, float]],
        targets: Mapping[str, Mapping[str, float]],
        max_vel: float,
        max_accel: float,
    ) -> "MovePlan":
        """The plan from `starts` to `targets` within `max_vel` and `max_accel` (per joint)."""
        disp = {a: {k: float(targets[a][k]) - float(starts[a][k]) for k in targets[a]} for a in targets}
        biggest = max((abs(d) for per_arm in disp.values() for d in per_arm.values()), default=0.0)
        total = move_time_trap(biggest, max_vel, max_accel)
        profiles = {
            a: {k: schedule_trapezoid(d, total, max_accel) for k, d in per_arm.items()}
            for a, per_arm in disp.items()
        }
        return cls(
            {a: {k: float(starts[a][k]) for k in targets[a]} for a in targets},
            {a: {k: float(v) for k, v in targets[a].items()} for a in targets},
            total,
            max_accel,
            profiles,
        )

    def waypoint(self, t: float) -> dict[str, dict[str, float]]:
        """Each arm's joints `t` seconds in (the targets from `total_time` on)."""
        return {
            a: {
                k: start[k]
                + trapezoid_offset(
                    t, self.targets[a][k] - start[k], self.total_time, *self.profiles[a][k], self.max_accel
                )
                for k in start
            }
            for a, start in self.starts.items()
        }


class Settler:
    """Whether a move held at its targets has arrived.

    True: every joint within the tolerance for `dwell`; False: `timeout` since the hold began; None while it may still.
    """

    def __init__(self, timeout: float, dwell: float = DWELL_S, began: float | None = None):
        """Waiting from `began` (the same clock as `update`'s), else from the first `update`."""
        self.timeout, self.dwell = timeout, dwell
        self.began: float | None = began
        self.within_since: float | None = None

    def update(self, now: float, within: bool) -> bool | None:
        """At time `now` (s, any clock), whether every joint is within the tolerance: the verdict, or None yet."""
        if self.began is None:
            self.began = now
        if not within:
            self.within_since = None
        elif self.within_since is None:
            self.within_since = now
        elif now - self.within_since >= self.dwell:
            return True
        if now - self.began >= self.timeout:
            return False
        return None
