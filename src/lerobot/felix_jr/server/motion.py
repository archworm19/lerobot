"""Arm, wheel and head motion for the robot server: one motion at a time, on a worker thread.

The motion runs on its own thread so the server can keep answering while the arms move: `motion_status`,
`stop`, `get_state`, and the heartbeat that keeps the watchdog quiet. Rules that make that safe:

- **One owner of the serial bus.** The worker thread holds `_bus_lock` for the whole motion. Anything else that
  needs the bus (a head move, a live read of the joints) either takes the lock without waiting or gives up:
  `exclusive()` raises `MotionBusyError`, and `state()` answers from the last snapshot the motion took. Two threads
  talking on one serial bus at once corrupt each other's packets.
- **`stop` aborts and holds.** The trajectory stops where it is and the servos keep holding their last goal
  (torque stays on). It does not go anywhere; `home` is a separate motion.
- **Rest = the saved home pose** (`setup.record_home`), the head's recorded home, and the wheels stopped. Going
  to rest happens on `home`, when a client session ends (see below), and when the server shuts down.
- **Watchdog.** A running motion is stopped when no request at all has arrived for `watchdog_s` (except the
  rest motions the server starts by itself, which are meant to run while nobody is listening). A client can
  also open a *session* with `on_disconnect="home"` and keep it alive with `heartbeat` requests: if they stop for
  `session_timeout_s` (a crashed script, a lost network), the server stops any motion and then goes to rest on
  its own, slowly. Tools that only read (`server_check`, the camera viewers) open no session and are never moved.

`ArmBackend` is the hardware side (`hardware.FakeArms` for tests, `hardware.RealArms` on the Pi).
"""

import contextlib
import logging
import math
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any, Protocol

from lerobot.felix_jr.server.head import HeadController

log = logging.getLogger("felix_motion")

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
MAX_BODY_DEG = 200.0  # sanity bound on a commanded joint position; the calibrated ranges are the real limits
MAX_BASE_SECONDS = (
    5.0  # a wheel command is always time-bounded: a lost client cannot leave the wheels spinning
)
DEFAULT_WATCHDOG_S = 3.0
DEFAULT_SESSION_TIMEOUT_S = 5.0
DEFAULT_MAX_VEL = 60.0  # deg/s; the same defaults as `home.move_all` (a test keeps them equal)
DEFAULT_MAX_ACCEL = 120.0  # deg/s^2
REST_MAX_VEL = 30.0  # deg/s: going to rest unprompted is done slowly
REST_MAX_ACCEL = 60.0
CACHE_PERIOD_S = 0.2  # how often a running motion refreshes the snapshot `state()` answers from


class MotionBusyError(RuntimeError):
    """A motion is already running (or the bus is in use), so this request cannot start."""


class ArmBackend(Protocol):
    """The arm and wheel access `MotionController` needs."""

    def arm_ids(self) -> list[str]:
        """Ids of the connected arms."""

    def observe(self) -> dict[str, dict[str, float]]:
        """Joint positions of every arm (`{"left_arm": {"shoulder_pan.pos": deg, ...}}`). Reads the bus."""

    def homes(self) -> dict[str, dict[str, float]]:
        """The recorded home pose of every arm that has one."""

    def run_trajectory(
        self,
        targets: dict[str, dict[str, float]],
        max_vel: float,
        max_accel: float,
        tol: float,
        timeout: float,
        should_stop: Callable[[], bool],
        on_tick: Callable[[], None],
    ) -> bool:
        """Move the arms to `targets` along synchronized trapezoids.

        False if `should_stop` ended it early or the arms did not settle; the servos hold their last goal
        either way.
        """

    def has_base(self) -> bool:
        """True if wheel servos are connected."""

    def set_base_velocity(self, left_degps: float, right_degps: float) -> None:
        """Command the wheels (positive = forward)."""

    def base_velocity(self) -> tuple[float, float]:
        """Measured (left, right) wheel speeds in deg/s."""

    def stop_base(self) -> None:
        """Stop the wheels."""


def validate_targets(targets: dict[str, dict[str, float]], arm_ids: list[str]) -> dict[str, dict[str, float]]:
    """Check a `move_joints` request and return it with plain floats. Raises ValueError with the reason."""
    if not isinstance(targets, dict) or not targets:
        raise ValueError("targets must map at least one arm id to its joint targets")
    clean: dict[str, dict[str, float]] = {}
    for arm, joints in targets.items():
        if arm not in arm_ids:
            raise ValueError(f"unknown arm {arm!r}; have {arm_ids}")
        if not isinstance(joints, dict) or not joints:
            raise ValueError(f"{arm}: give at least one joint target")
        clean[arm] = {}
        for key, value in joints.items():
            if key.removesuffix(".pos") not in ARM_JOINTS or not key.endswith(".pos"):
                raise ValueError(f"{arm}: unknown joint {key!r}; use e.g. 'shoulder_pan.pos'")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{arm}.{key}: {value!r} is not a finite number")
            low, high = (0.0, 100.0) if key == "gripper.pos" else (-MAX_BODY_DEG, MAX_BODY_DEG)
            if not low <= value <= high:
                raise ValueError(f"{arm}.{key}: {value} is outside [{low}, {high}]")
            clean[arm][key] = float(value)
    return clean


class MotionController:
    """Runs one motion at a time and answers status questions while it runs. See the module docstring."""

    def __init__(
        self,
        arms: ArmBackend,
        head: HeadController | None = None,
        watchdog_s: float = DEFAULT_WATCHDOG_S,
        session_timeout_s: float = DEFAULT_SESSION_TIMEOUT_S,
        clock: Callable[[], float] = time.monotonic,
        watch_period_s: float = 0.2,
    ):
        """Create the controller and start its watchdog thread (stopped by `shutdown`)."""
        self.arms, self.head = arms, head
        self.watchdog_s, self.session_timeout_s = watchdog_s, session_timeout_s
        self._clock = clock
        self._bus_lock = threading.Lock()  # held by whoever is using the serial bus
        self._lock = threading.Lock()  # guards the fields below
        self._stop = threading.Event()
        self._stop_reason: str | None = None
        self._thread: threading.Thread | None = None
        self._motion_id = 0
        self._status: dict[str, Any] = {
            "state": "idle",
            "motion_id": None,
            "kind": None,
            "reason": None,
            "result": None,
            "autonomous": False,
            "finished_at": None,
        }
        self._cache: dict[str, Any] = {}
        self._cache_time = 0.0
        self._last_touch = clock()
        self._session_home = False
        self._closing = threading.Event()
        self._watch_period_s = watch_period_s
        self._watcher = threading.Thread(target=self._watch, daemon=True, name="motion-watchdog")
        self._watcher.start()

    # ---- liveness ---------------------------------------------------------------------------------------

    def touch(self) -> None:
        """Note that a request just arrived (any request counts as a sign of life for the watchdog)."""
        self._last_touch = self._clock()

    def open_session(self, on_disconnect: str) -> dict[str, Any]:
        """Start a session.

        With `on_disconnect="home"` the server goes to rest if the client falls silent for
        `session_timeout_s` (the client must send `heartbeat` requests meanwhile); "hold" does nothing.
        """
        if on_disconnect not in ("home", "hold"):
            raise ValueError(f"on_disconnect must be 'home' or 'hold', not {on_disconnect!r}")
        self.touch()
        self._session_home = on_disconnect == "home"
        return {"on_disconnect": on_disconnect, "timeout_s": self.session_timeout_s}

    def close_session(self) -> dict[str, Any]:
        """End the session without going anywhere."""
        self._session_home = False
        return {}

    def _watch(self) -> None:
        while not self._closing.wait(self._watch_period_s):
            silent = self._clock() - self._last_touch
            running = self.status()["moving"]
            if running and not self.status()["autonomous"] and silent > self.watchdog_s:
                self.stop(f"watchdog: no request for {silent:.1f} s")
            if self._session_home and silent > self.session_timeout_s:
                if running:
                    self.stop(f"session lost: no request for {silent:.1f} s")
                else:
                    log.warning("session lost (no request for %.1f s): going to rest", silent)
                    self._session_home = False
                    try:
                        self.home(
                            max_vel=REST_MAX_VEL,
                            max_accel=REST_MAX_ACCEL,
                            reason="session lost",
                            autonomous=True,
                        )
                    except MotionBusyError:
                        self._session_home = True  # the bus is momentarily in use; try again next period
                    except Exception:
                        log.exception("could not go to rest after losing the session")

    # ---- reading ----------------------------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """The running or most recent motion.

        `{moving, state, motion_id, kind, reason, result, autonomous, finished_at}`: `state` is idle, running,
        done, stopped or failed; `autonomous` marks a rest motion the server started on its own (a lost session,
        shutdown), which the watchdog leaves alone; `finished_at` is the server time it ended, to pass to
        `capture(after=...)`.
        """
        with self._lock:
            status = dict(self._status)
        return {**status, "moving": status["state"] == "running"}

    def _snapshot(self) -> dict[str, Any]:
        out: dict[str, Any] = dict(self.arms.observe())
        if self.head is not None:
            out["head"] = self.head.state()
        return out

    def _set_cache(self, snapshot: dict[str, Any]) -> None:
        with self._lock:
            self._cache, self._cache_time = snapshot, self._clock()

    def state(self) -> dict[str, Any]:
        """Joint state of every arm (and the head), plus `moving` and `motion` (the status).

        When idle it reads the servos. While a motion runs it does not touch the bus and returns the last
        snapshot the motion took (refreshed every `CACHE_PERIOD_S`), with `state_age_s` saying how old it is.
        """
        if self._bus_lock.acquire(blocking=False):
            try:
                snapshot = self._snapshot()
                self._set_cache(snapshot)
            finally:
                self._bus_lock.release()
            age = 0.0
        else:
            with self._lock:
                snapshot, age = dict(self._cache), self._clock() - self._cache_time
        status = self.status()
        return {**snapshot, "moving": status["moving"], "motion": status, "state_age_s": round(age, 2)}

    @contextlib.contextmanager
    def exclusive(self) -> Iterator[None]:
        """Use the bus for something short (a head move) if no motion is running; raise `MotionBusyError` if one is."""
        if not self._bus_lock.acquire(blocking=False):
            raise MotionBusyError("a motion is running; stop it or wait for it to finish")
        try:
            yield
        finally:
            self._bus_lock.release()

    # ---- starting and stopping motions ------------------------------------------------------------------

    def stop(self, reason: str = "client stop") -> dict[str, Any]:
        """Abort the running motion (if any). The arms hold where they are. Returns `{was_moving}`."""
        was_moving = self.status()["moving"]
        if was_moving:
            self._stop_reason = reason
            self._stop.set()
            log.info("stopping the motion: %s", reason)
        return {"was_moving": was_moving}

    def _start(
        self, kind: str, job: Callable[[], dict[str, Any]], autonomous: bool = False
    ) -> dict[str, Any]:
        if not self._bus_lock.acquire(blocking=False):
            raise MotionBusyError("a motion is running; stop it or wait for it to finish")
        self._stop.clear()
        self._stop_reason = None
        with self._lock:
            self._motion_id += 1
            motion_id = self._motion_id
            self._status = {
                "state": "running",
                "motion_id": motion_id,
                "kind": kind,
                "reason": None,
                "result": None,
                "autonomous": autonomous,
                "finished_at": None,
            }
        self._thread = threading.Thread(
            target=self._run, args=(motion_id, kind, job), daemon=True, name=f"motion-{motion_id}"
        )
        self._thread.start()
        return {"motion_id": motion_id}

    def _run(self, motion_id: int, kind: str, job: Callable[[], dict[str, Any]]) -> None:
        state, result, reason = "failed", None, None
        autonomous = self.status()["autonomous"]
        try:
            result = job()
            if self._stop.is_set():
                state, reason = "stopped", self._stop_reason
            else:
                state = "done"
        except Exception as e:
            log.exception("%s failed", kind)
            reason = f"{type(e).__name__}: {e}"
        finally:
            with contextlib.suppress(Exception):  # a last snapshot, so `state()` right after is current
                self._set_cache(self._snapshot())
            with self._lock:
                self._status = {
                    "state": state,
                    "motion_id": motion_id,
                    "kind": kind,
                    "reason": reason,
                    "result": result,
                    "autonomous": autonomous,
                    "finished_at": time.time(),  # wall clock, for `capture(after=...)`
                }
            self._bus_lock.release()

    def _on_tick(self) -> None:
        if self._clock() - self._cache_time >= CACHE_PERIOD_S:
            self._set_cache(self._snapshot())

    def move_joints(
        self,
        targets: dict[str, dict[str, float]],
        max_vel: float | None = None,
        max_accel: float | None = None,
        tol: float = 1.5,
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        """Start moving the arms to `targets`: one target pose per arm, not a path.

        `targets` is `{arm id: {"shoulder_pan.pos": deg, ...}}`; every arm moves together, from wherever it is
        now (or from its last commanded goal), along one trapezoid. To follow a path or to replan, send the next
        target when this one ends, or `stop` and send a new one (the new motion starts from the last goal sent).

        Returns `{motion_id}` at once; follow it with `status`. Refused with `MotionBusyError` while another
        motion runs. An arm must *have* a recorded home pose (`setup.record_home`), so there is somewhere to
        send it on shutdown or when a client is lost; it does not need to be at that pose now.
        """
        clean = validate_targets(targets, self.arms.arm_ids())
        homes = self.arms.homes()
        for arm in clean:
            if arm not in homes:
                raise ValueError(f"{arm} has no recorded home; run setup.record_home first")
        limits = _limits(max_vel, max_accel)
        return self._start(
            "move_joints", lambda: self._trajectory_job(clean, **limits, tol=tol, timeout=timeout)
        )

    def _trajectory_job(
        self,
        targets: dict[str, dict[str, float]],
        max_vel: float,
        max_accel: float,
        tol: float,
        timeout: float,
    ) -> dict[str, Any]:
        self._set_cache(self._snapshot())
        reached = self.arms.run_trajectory(
            targets, max_vel, max_accel, tol, timeout, self._stop.is_set, self._on_tick
        )
        return {"reached": bool(reached)}

    def home(
        self,
        max_vel: float | None = None,
        max_accel: float | None = None,
        reason: str = "client request",
        autonomous: bool = False,
        arms: list[str] | None = None,
    ) -> dict[str, Any]:
        """Start going to rest: the arms to their recorded home poses, then the head to its home.

        `arms` picks which arms (default: every arm that has a recorded home). `autonomous` is for motions
        the server starts itself: the watchdog does not stop them.
        """
        homes = self.arms.homes()
        if arms is not None:
            unknown = [arm for arm in arms if arm not in self.arms.arm_ids()]
            if unknown or not arms:
                raise ValueError(f"unknown arms {unknown or arms}; have {self.arms.arm_ids()}")
            missing = [arm for arm in arms if arm not in homes]
            if missing:
                raise ValueError(f"{missing} have no recorded home; run setup.record_home first")
            homes = {arm: homes[arm] for arm in arms}
        if not homes:
            raise ValueError("no arm has a recorded home; run setup.record_home first")
        limits = _limits(max_vel, max_accel)
        log.info("going to rest (%s)", reason)
        return self._start("home", lambda: self._rest_job(homes, **limits), autonomous)

    def _rest_job(
        self, homes: dict[str, dict[str, float]], max_vel: float, max_accel: float
    ) -> dict[str, Any]:
        result = self._trajectory_job(homes, max_vel, max_accel, tol=1.5, timeout=10.0)
        if result["reached"] and self.head is not None and not self._stop.is_set():
            for name in self.head.backend.names():
                if self.head.store.get(name).get("home") is not None:
                    self.head.move(name, to="home")
        if self.arms.has_base():
            self.arms.stop_base()
        return result

    def base_velocity(self, left_degps: float, right_degps: float, seconds: float) -> dict[str, Any]:
        """Drive the wheels at the given speeds (deg/s, positive = forward), then stop them.

        It runs for `seconds` (at most `MAX_BASE_SECONDS`). The result is the measured `(left, right)` speeds
        late in the run.
        """
        if not self.arms.has_base():
            raise ValueError("this robot has no wheel base")
        values = (left_degps, right_degps, seconds)
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            raise ValueError("speeds and seconds must be finite numbers")
        if not 0 < seconds <= MAX_BASE_SECONDS:
            raise ValueError(f"seconds must be in (0, {MAX_BASE_SECONDS}]")
        return self._start(
            "base_velocity", lambda: self._base_job(float(left_degps), float(right_degps), seconds)
        )

    def _base_job(self, left: float, right: float, seconds: float) -> dict[str, Any]:
        measured = (0.0, 0.0)
        started = self._clock()
        try:
            self.arms.set_base_velocity(left, right)
            while (elapsed := self._clock() - started) < seconds and not self._stop.is_set():
                self.arms.set_base_velocity(left, right)  # keep commanding; a stopped server stops the wheels
                if elapsed > 0.6 * seconds:
                    measured = self.arms.base_velocity()
                time.sleep(0.05)
        finally:
            self.arms.stop_base()
        return {"measured": [float(measured[0]), float(measured[1])]}

    # ---- shutdown ---------------------------------------------------------------------------------------

    def shutdown(self, go_home: bool = True) -> None:
        """Stop any motion, then (by default) go to rest, slowly, and wait for it.

        Call before the servos are disconnected: disconnecting switches their torque off, and an arm held out
        would drop.
        """
        self._closing.set()
        self._watcher.join(timeout=2.0)
        self.stop("shutdown")
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if go_home and self.arms.homes():
            try:
                self.home(max_vel=REST_MAX_VEL, max_accel=REST_MAX_ACCEL, reason="shutdown", autonomous=True)
                if self._thread is not None:
                    self._thread.join(timeout=60.0)
            except Exception:
                log.exception(
                    "could not go to rest on shutdown; the arms may drop when torque is switched off"
                )


def _limits(max_vel: float | None, max_accel: float | None) -> dict[str, float]:
    vel = DEFAULT_MAX_VEL if max_vel is None else float(max_vel)
    accel = DEFAULT_MAX_ACCEL if max_accel is None else float(max_accel)
    if not (0 < vel <= 120 and 0 < accel <= 300):
        raise ValueError(
            f"max_vel must be in (0, 120] deg/s and max_accel in (0, 300] deg/s^2, got {vel}, {accel}"
        )
    return {"max_vel": vel, "max_accel": accel}
