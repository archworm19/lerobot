"""Head servo control for the robot server: relative moves with limits, and a remembered range and home.

The head servos (`head_neck`, `head_eyes`, see `docs/head.md`) are worked in raw servo ticks (4096 per turn), so
this needs no calibration. A move is given in degrees in the servo's *positive* direction (DOWN for the eyes,
LEFT for the neck, as in `xlerobot.urdf`); the sign saved by `setup.head` maps that onto raw ticks.

Recorded per servo (raw ticks, saved in `<id>_head_range.json` next to the arm's calibration):

- `limit_pos` / `limit_neg`: how far the servo may go in its positive / negative direction (down / up for the
  eyes). Moves are clamped to them once recorded.
- `home`: where the head rests.

`HeadController` talks to the servos through a small `HeadBackend`, so the same code runs on the real bus
(`hardware.BusHeadBackend`) and on a simulation (`hardware.FakeHeadBackend`).
"""

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

TICKS_PER_TURN = 4096
TICKS_PER_DEG = TICKS_PER_TURN / 360.0
SUB_STEP_DEG = 5.0  # a long move is made in steps of at most this, each settled before the next
SETTLE_TOLERANCE = 8  # ticks (about 0.7 deg) from the goal that count as "arrived"
# A servo that stops moving farther than this from its goal (about 3 deg) is called blocked and told to hold. Closer
# than this it is only "short": the load or the servo's low P gain keeps it from the last few ticks, which is
# normal, so the goal stays commanded and the servo keeps pulling towards it.
BLOCKED_ERROR = 34
SETTLE_TIMEOUT_S = 1.0
STALL_POLLS = 4  # this many polls without any movement, away from the goal, mean the servo has stopped
POLL_S = 0.05
RANGE_KEYS = ("limit_pos", "limit_neg", "home")
MAX_TORQUE_LIMIT = 1000  # STS3215 Torque_Limit register: 0-1000 = 0-100 %


class HeadBackend(Protocol):
    """The servo access `HeadController` needs."""

    def names(self) -> list[str]:
        """Names of the head servos."""

    def read_ticks(self, name: str) -> int:
        """Present raw position of servo `name`."""

    def write_goal(self, name: str, ticks: int) -> None:
        """Command servo `name` to raw position `ticks`."""

    def write_torque_limit(self, name: str, limit: int) -> None:
        """Cap servo `name`'s torque (0-1000). It is a RAM register: it resets when the servo loses power."""


class RangeStore:
    """The recorded limits and home of each head servo, kept in a JSON file (or just in memory)."""

    def __init__(self, path: Path | None = None):
        """Load `path` if it exists. With `path=None` nothing is written to disk."""
        self.path = path
        self.data: dict[str, dict[str, int]] = {}
        if path is not None and path.is_file():
            self.data = json.loads(path.read_text())

    def get(self, name: str) -> dict[str, int]:
        """The recorded values of servo `name` (missing keys were never recorded)."""
        return dict(self.data.get(name, {}))

    def set(self, name: str, key: str, ticks: int) -> None:
        """Record `ticks` as `key` of servo `name` and write the file."""
        self.data.setdefault(name, {})[key] = int(ticks)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, indent=2) + "\n")


class HeadController:
    """Moves and records the head servos. Not thread-safe: one request at a time."""

    def __init__(
        self,
        backend: HeadBackend,
        signs: dict[str, int],
        store: RangeStore,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
    ):
        """`signs[name]` is +1 if the servo's raw positive direction is down / left, -1 if not (`_head.json`)."""
        self.backend = backend
        self.signs = signs
        self.store = store
        self._sleep = sleep
        self._clock = clock
        self._goals: dict[str, int] = {}  # the last goal each servo was told to hold

    def _check(self, name: str) -> None:
        if name not in self.backend.names():
            raise ValueError(f"unknown head servo {name!r}; have {self.backend.names()}")

    def state(self) -> dict[str, dict[str, Any]]:
        """Per servo: `ticks` (now), `sign`, and the recorded `limit_pos`, `limit_neg`, `home`.

        Values that were never recorded are None.
        """
        out = {}
        for name in self.backend.names():
            recorded = self.store.get(name)
            out[name] = {
                "ticks": self.backend.read_ticks(name),
                "sign": self.signs.get(name, 1),
                **{key: recorded.get(key) for key in RANGE_KEYS},
            }
        return out

    def bounds(self, name: str) -> tuple[int, int]:
        """Lowest and highest raw tick the servo may be commanded to, from the recorded limits."""
        lo, hi = 0, TICKS_PER_TURN - 1
        recorded, sign = self.store.get(name), self.signs.get(name, 1)
        for key, direction in (("limit_pos", +1), ("limit_neg", -1)):
            if key in recorded:
                if direction * sign > 0:
                    hi = min(hi, recorded[key])  # raw positive direction is the one limited
                else:
                    lo = max(lo, recorded[key])
        return lo, hi

    def _base(self, name: str) -> int:
        """Where a relative move starts.

        That is the last commanded goal while the servo is still near it, else where the servo is. Counting
        from the goal keeps repeated small moves from losing the few ticks the servo falls short each time (a
        gravity-loaded servo at a low P gain always does).
        """
        actual = self.backend.read_ticks(name)
        last = self._goals.get(name)
        return last if last is not None and abs(actual - last) <= BLOCKED_ERROR else actual

    def move(self, name: str, delta_deg: float | None = None, to: str | None = None) -> dict[str, Any]:
        """Move servo `name` and wait until it has settled.

        Give exactly one of: `delta_deg`, a move in degrees from where the servo was last told to be (positive
        = down / left); or `to`, "home" (the recorded home position). The goal is clamped to the recorded
        limits and the move is made in steps of at most `SUB_STEP_DEG`.

        Returns `{name, ticks, goal, clamped, reached, error_ticks, done_at}`: where it ended up, the
        (clamped) goal, whether the goal was clamped, whether the servo got there (False only if it is
        blocked: it stopped more than `BLOCKED_ERROR` ticks short, and was told to hold where it is), how many
        ticks it is short of the goal (`goal - ticks`), and the server time it finished.
        """
        self._check(name)
        if (delta_deg is None) == (to is None):
            raise ValueError("give exactly one of delta_deg and to")
        if to is not None:
            if to != "home":
                raise ValueError(f"`to` must be 'home', not {to!r}")
            target = self.store.get(name).get(to)
            if target is None:
                raise ValueError(f"no {to} recorded for {name}; record one first")
        else:
            target = self._base(name) + self.signs.get(name, 1) * round(float(delta_deg) * TICKS_PER_DEG)
        return self._go(name, target)

    def _go(self, name: str, target: int) -> dict[str, Any]:
        """Take servo `name` to raw `target` (clamped to the recorded limits); see `move` for the result."""
        lo, hi = self.bounds(name)
        goal = min(max(target, lo), hi)
        sub_step = round(SUB_STEP_DEG * TICKS_PER_DEG)
        commanded, blocked = (
            self.backend.read_ticks(name),
            False,
        )  # counted from what was commanded: always ends
        while commanded != goal:
            commanded += min(max(goal - commanded, -sub_step), sub_step)
            self.backend.write_goal(name, commanded)
            position, status = self._settle(name, commanded)
            if status == "blocked":
                self.backend.write_goal(name, position)  # hold here instead of pushing
                blocked = True
                break
        ticks = self.backend.read_ticks(name)
        self._goals[name] = ticks if blocked else goal
        return {
            "name": name,
            "ticks": ticks,
            "goal": goal,
            "clamped": goal != target,
            "reached": not blocked,
            "error_ticks": goal - ticks,
            "done_at": self._clock(),
        }

    def _settle(self, name: str, goal: int) -> tuple[int, str]:
        """Wait until servo `name` is at `goal`. Returns (position, status).

        Status is "arrived" (within `SETTLE_TOLERANCE`), "short" (it stopped moving, or took too long, but is
        within `BLOCKED_ERROR` of the goal) or "blocked" (stopped or too slow, farther than that).
        """
        started, last, still = self._clock(), None, 0
        while True:
            position = self.backend.read_ticks(name)
            if abs(position - goal) <= SETTLE_TOLERANCE:
                return position, "arrived"
            still = still + 1 if position == last else 0
            if still >= STALL_POLLS or self._clock() - started > SETTLE_TIMEOUT_S:
                return position, "short" if abs(position - goal) <= BLOCKED_ERROR else "blocked"
            last = position
            self._sleep(POLL_S)

    def wiggle(self, name: str, degrees: float) -> dict[str, Any]:
        """Nudge servo `name` in both directions, coming back to where it started after each.

        It goes `degrees` in the positive direction (down / left), then the same the other way. Moves are
        clamped to the recorded limits.

        Returns `to_pos` and `to_neg`: how far it actually went in each direction, in degrees measured in the
        direction it was asked to go (both close to `degrees` if it followed the command); `limited`: a move was
        shortened by a recorded limit; `blocked`: the servo stopped well short of a goal without a limit being
        the reason.
        """
        self._check(name)
        sign = self.signs.get(name, 1)
        measured_from = self.backend.read_ticks(name)
        base = self._base(name)
        achieved, limited, blocked = [], False, False
        for direction in (+1, -1):
            out = self._go(name, base + direction * sign * round(degrees * TICKS_PER_DEG))
            limited |= out["clamped"]
            blocked |= not out["reached"]
            achieved.append(direction * sign * (out["ticks"] - measured_from) / TICKS_PER_DEG)
            blocked |= not self._go(name, base)["reached"]
        return {"to_pos": achieved[0], "to_neg": achieved[1], "limited": limited, "blocked": blocked}

    def record(self, name: str, what: str) -> dict[str, Any]:
        """Save servo `name`'s present position as `what` (one of `RANGE_KEYS`). Returns the servo's state."""
        self._check(name)
        if what not in RANGE_KEYS:
            raise ValueError(f"`what` must be one of {RANGE_KEYS}, not {what!r}")
        self.store.set(name, what, self.backend.read_ticks(name))
        return self.state()[name]

    def torque_limit(self, name: str, limit: int) -> dict[str, Any]:
        """Cap servo `name`'s torque at `limit` (1-1000).

        A low cap makes a blocked servo push gently; it resets when the servo loses power.
        """
        self._check(name)
        if not 1 <= limit <= MAX_TORQUE_LIMIT:
            raise ValueError(f"torque limit must be 1-{MAX_TORQUE_LIMIT}, got {limit}")
        self.backend.write_torque_limit(name, int(limit))
        return {"name": name, "limit": int(limit)}
