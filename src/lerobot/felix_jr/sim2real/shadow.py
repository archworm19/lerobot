"""The shadow: a real execution replayed in the simulator, move by move, as the robot server runs it (`docs/sim_harness.md`).

  uv run --extra felix_jr --extra felix_jr_sim python -m lerobot.felix_jr.sim2real.shadow [--sessions <dir>]
      [--model <calibration.urdf>]

`Shadow` holds one arm's scene (`scene.build`: the fitted servos, the sprung fingers) and plays `move`s as
`home.move_all` does: from the last commanded goal, the joints' synchronized trapezoids (`trajectory`) at the move's
`max_vel` / `max_accel`, a waypoint each 1/`WAYPOINT_HZ` s, then held until every joint is within `tol` for
`DWELL_S` or the `timeout`. Each move's end: the simulated joints and `gripper.pos`, reached or not.

Phase 1 (no objects): the jaws close on nothing, so a move with them closed on an object on the robot is marked
`loaded` (the object's weight not in the simulator: compared apart). `session_records` replays the recorded sessions'
held executions (`harness/scratch/sessions/*/workspace.pkl`: the moves, the capture they were solved from, the capture
at the hold) and compares the simulated end with the real one; the command line prints it per kind of pose.
"""

import argparse
import pickle  # nosec B403: the harness's own sessions, written on this computer
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel, pinch_frame
from lerobot.felix_jr.trajectory import DWELL_S, WAYPOINT_HZ, MovePlan, Settler, move_starts
from lerobot.utils.import_utils import _mujoco_available, require_package

from . import robot
from .scene import Cylinder, Scene, build

if _mujoco_available:
    import mujoco

if TYPE_CHECKING:
    from lerobot.felix_jr.harness.objects import ExecutionLog, MoveReading

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
# a close of more than this (`gripper.pos`): the jaws on an object from there (phase 1: `loaded`), an opening ends it
CLOSE_UNITS = 5.0
SESSIONS = Path(__file__).resolve().parents[1] / "harness" / "scratch" / "sessions"
# the recorded sessions' replays without a table: their tables stood at other heights than the model's floor (a wrong
# one put bowl4's wrist camera into it, `shortfalls`)
NO_TABLE_Z = -1.0


@dataclass(frozen=True)
class MoveReplay:
    """One move's end in the simulator: its joints (deg), `gripper.pos`, whether it settled, how long it took (s)."""

    label: str
    targets: dict[str, float]
    joints: dict[str, float]
    gripper_pos: float
    reached: bool
    seconds: float
    loaded: bool = False  # the real jaws on an object through this move (phase 1: not in the simulator)


@dataclass
class Shadow:
    """One arm's scene, played move by move (the module docstring)."""

    model: XleRobotModel
    arm: str
    start: Mapping[str, float]  # the real readings (`{"shoulder_pan.pos": deg, ..., "gripper.pos": 0-100}`)
    table_z: float | None = None
    objects: Sequence[Cylinder] = ()
    finger_model: str = "sprung"
    scene: Scene = field(init=False)
    goal: dict[str, float] = field(
        init=False
    )  # the last commanded goal (where `home.move_all` starts a move)
    loaded: bool = False
    # called each tick (1/`WAYPOINT_HZ` s of simulated time) with the move's label: to film a replay (`failures`)
    on_tick: Callable[[str], None] | None = None

    def __post_init__(self) -> None:
        """Build the scene and pose the arm at the readings, the servos holding there."""
        require_package("mujoco", extra="felix_jr_sim")
        table_z = self.table_z if self.table_z is not None else self.model.floor_z
        self.scene = build(
            self.model,
            self.arm,
            list(self.objects),
            table_z=table_z if table_z is not None else -1.0,
            finger_model=self.finger_model,
        )
        self.scene.set_pose(dict(self.start))
        self.goal = dict(self.start)

    def reading(self) -> dict[str, float]:
        """The simulated joints (deg) and `gripper.pos`, as the servos would report them."""
        m, d = self.scene.m, self.scene.d
        out = {f"{j}.pos": float(np.degrees(d.qpos[m.joint(j).qposadr[0]])) for j in JOINTS}
        opening = float(np.degrees(d.qpos[m.joint(robot.GRIPPER).qposadr[0]]))
        out["gripper.pos"] = (
            self.model.gripper_pos_for(self.arm, opening) if self.arm in self.model.gripper_maps else opening
        )
        return out

    def object_position(self, name: str = "object") -> np.ndarray:
        """Where an object's center is now (root frame, m)."""
        return self.scene.d.xpos[self.scene.m.body(name).id].copy()

    def hold(self, seconds: float, label: str = "hold") -> None:
        """The servos holding their commands for `seconds` (a lift's hold)."""
        m, d = self.scene.m, self.scene.d
        per_tick = max(1, int(round(1.0 / WAYPOINT_HZ / m.opt.timestep)))
        for k in range(int(round(seconds / m.opt.timestep))):
            mujoco.mj_step(m, d)
            if self.on_tick is not None and (k + 1) % per_tick == 0:
                self.on_tick(label)

    def _command(self, pose: Mapping[str, float]) -> None:
        m, d = self.scene.m, self.scene.d
        for key, value in pose.items():
            name = key.removesuffix(".pos")
            if name == "gripper":
                deg = (
                    self.model.opening_deg(self.arm, value) if self.arm in self.model.gripper_maps else value
                )
                d.ctrl[m.actuator(robot.GRIPPER).id] = np.radians(deg)
            else:
                d.ctrl[m.actuator(name).id] = np.radians(value)

    def move(
        self,
        label: str,
        targets: Mapping[str, float],
        max_vel: float,
        max_accel: float,
        tol: float = 1.5,
        timeout: float = 10.0,
    ) -> MoveReplay:
        """`targets` as the server moves to them (the module docstring); the move's end."""
        m, d = self.scene.m, self.scene.d
        now = self.reading()
        plan = MovePlan.plan(
            {"arm": move_starts(targets, self.goal, now)}, {"arm": targets}, max_vel, max_accel
        )
        if "gripper.pos" in targets:  # (from the last goal: where the jaw was told, not where it stalled)
            change = targets["gripper.pos"] - self.goal.get("gripper.pos", now["gripper.pos"])
            if change < -CLOSE_UNITS:
                self.loaded = True
            elif change > CLOSE_UNITS:
                self.loaded = False
        tick = 1.0 / WAYPOINT_HZ  # (as `home.move_all`: a waypoint a tick, then the targets held)
        per_tick = max(1, int(round(tick / m.opt.timestep)))
        t = 0.0
        while t < plan.total_time:
            self._command(plan.waypoint(t)["arm"])
            for _ in range(per_tick):
                mujoco.mj_step(m, d)
            t += tick
            if self.on_tick is not None:
                self.on_tick(label)
        self._command(dict(targets))
        joints = list(
            targets
        )  # (every target, the jaw's too, as the server checks: a stalled jaw never arrives)
        settler, held, reached = Settler(timeout, DWELL_S, began=0.0), 0.0, None
        while reached is None:
            r = self.reading()
            reached = settler.update(held, within=all(abs(r[k] - targets[k]) < tol for k in joints))
            for _ in range(per_tick):
                mujoco.mj_step(m, d)
            held += tick
            if self.on_tick is not None:
                self.on_tick(label)
        self.goal.update({k: float(v) for k, v in targets.items()})
        r = self.reading()
        return MoveReplay(
            label,
            dict(targets),
            {k: r[k] for k in r if k != "gripper.pos"},
            r["gripper.pos"],
            bool(reached),
            plan.total_time + held,
            self.loaded,
        )


# ---- an execution's log, replayed and compared -------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class ShadowRecord:
    """An `ExecutionLog` replayed: each move's end in the simulator, and the settings it ran under (servos, fingers)."""

    log: "ExecutionLog"
    moves: tuple[MoveReplay, ...]
    settings: dict[str, Any]

    def describe(self) -> str:
        """One line: the moves replayed, reached or not, and the jaws' load."""
        parts = [
            f"{m.label}{'' if m.reached else ' (not reached)'}{' [loaded]' if m.loaded else ''}"
            for m in self.moves
        ]
        return f"{self.log.arm} arm, {len(self.moves)} moves replayed: " + ", ".join(parts)


def settings() -> dict[str, Any]:
    """The simulator's settings a replay ran under: kept with each record, readable after a refit."""
    return {
        "kp": robot.ARM_KP_NM_PER_RAD,
        "frictionloss": robot.ARM_FRICTIONLOSS_NM,
        "gripper_kp": robot.GRIPPER_KP_NM_PER_RAD,
        "finger_friction": robot.FINGER_FRICTION,
        "finger_stiffness": robot.FINGER_STIFFNESS_NM_PER_RAD,
    }


def replay(
    model: XleRobotModel,
    log: "ExecutionLog",
    table_z: float | None = None,
    objects: Sequence[Cylinder] = (),
    finger_model: str = "sprung",
) -> ShadowRecord:
    """`log`'s moves in the simulator from its start, as the server ran them (`Shadow`)."""
    shadow = Shadow(model, log.arm, dict(log.start), table_z, objects, finger_model)
    moves = tuple(
        shadow.move(m.label, m.targets, log.max_vel, log.max_accel, m.tol, m.timeout) for m in log.moves
    )
    return ShadowRecord(log, moves, settings())


def kind(reading: "MoveReading", replayed: MoveReplay) -> str:
    """A move's pose: attack (`pick.back`), grasp (`pick`), loaded (the jaws on an object), other."""
    name = reading.label.split(" ")[0]
    if replayed.loaded:
        return "loaded"
    return "attack" if name.endswith(".back") else "grasp" if name == "pick" else "other"


@dataclass(frozen=True)
class MovePair:
    """One move's end, real against simulated: read minus commanded (deg, `JOINTS`), the pinch points apart (mm)."""

    label: str
    kind: str
    real: np.ndarray
    sim: np.ndarray
    pinch_mm: float
    real_gripper: float
    sim_gripper: float


def pairs(model: XleRobotModel, record: ShadowRecord) -> list[MovePair]:
    """The moves of `record`'s log read on the robot that moved the arm, each against its replay."""
    arm, out = record.log.arm, []
    for reading, replayed in zip(record.log.moves, record.moves, strict=True):
        if reading.reading is None or not any(f"{j}.pos" in reading.targets for j in JOINTS):
            continue
        commanded = {j: reading.targets.get(f"{j}.pos", reading.reading[f"{j}.pos"]) for j in JOINTS}
        real = np.array([reading.reading[f"{j}.pos"] - commanded[j] for j in JOINTS])
        sim = np.array([replayed.joints[f"{j}.pos"] - commanded[j] for j in JOINTS])
        a, b = (
            model.frame_pose(pinch_frame(arm), model.joint_vector({arm: pose}))[:3, 3]
            for pose in (reading.reading, {**reading.reading, **replayed.joints})
        )
        out.append(
            MovePair(
                reading.label,
                kind(reading, replayed),
                real,
                sim,
                float(1000 * np.linalg.norm(a - b)),
                float(reading.reading.get("gripper.pos", float("nan"))),
                replayed.gripper_pos,
            )
        )
    return out


def compare(model: XleRobotModel, record: ShadowRecord) -> str:
    """Move by move, real against simulated: each joint's read minus commanded, the pinch points apart, the jaw."""
    lines = [f"{record.describe()}", f"read - commanded per joint ({', '.join(JOINTS)}), deg: real / sim"]
    for p in pairs(model, record):
        joints = "  ".join(f"{r:+5.2f}/{s:+5.2f}" for r, s in zip(p.real, p.sim, strict=True))
        lines.append(
            f"- {p.label} [{p.kind}]: {joints}; pinch points {p.pinch_mm:.1f} mm apart; "
            f"gripper.pos {p.real_gripper:.1f} / {p.sim_gripper:.1f}"
        )
    if len(lines) == 2:
        lines.append("(no move with a reading: a recorded session's execution has its hold's only)")
    return "\n".join(lines)


def report(model: XleRobotModel, records: Sequence[ShadowRecord]) -> str:
    """Per kind of pose, over `records`: the real and simulated means, their difference's RMS, the pinch points apart."""
    every = [p for r in records for p in pairs(model, r)]
    lines = [f"{len(every)} moves of {len(records)} executions; per joint {', '.join(JOINTS)} (deg)"]
    for name in ("attack", "grasp", "loaded", "other"):
        group = [p for p in every if p.kind == name]
        if not group:
            continue
        real, sim = np.array([p.real for p in group]), np.array([p.sim for p in group])
        miss = np.array([p.pinch_mm for p in group])
        lines += [
            f"{name} ({len(group)}):",
            "  real mean " + " ".join(f"{v:+6.2f}" for v in real.mean(0)),
            "  sim mean  " + " ".join(f"{v:+6.2f}" for v in sim.mean(0)),
            "  RMS apart " + " ".join(f"{v:6.2f}" for v in np.sqrt(((sim - real) ** 2).mean(0))),
            f"  the pinch points apart: {np.mean(miss):.1f} mm mean, {np.max(miss):.1f} max",
        ]
    return "\n".join(lines)


# ---- the recorded sessions (before the logs) ---------------------------------------------------------------


class _Gone:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)


class _Loader(pickle.Unpickler):
    """A session's pickle, the classes the harness has since dropped loaded as empty stand-ins."""

    def find_class(self, module: str, name: str) -> Any:
        try:
            return super().find_class(module, name)
        except AttributeError:
            return type(name, (_Gone,), {})


def session_logs(sessions: Path = SESSIONS) -> list["ExecutionLog"]:
    """The recorded sessions' logs: their own if they have them, else built from their held executions.

    Their own: `ExecutionLog` nodes, every move read. Built: each held execution's moves, the capture it was solved
    from, the reading at the hold only. Sessions that cannot be read are skipped, said on stderr.
    """
    from lerobot.felix_jr.harness.objects import ExecutionLog, MoveReading

    out = []
    for path in sorted(sessions.glob("*/workspace.pkl")):
        try:
            with path.open("rb") as f:
                nodes = _Loader(f).load()["ws"].nodes
        except Exception as e:  # (a session from a harness too old to load at all)
            print(f"skipped {path.parent.name}: {type(e).__name__} {e}", file=sys.stderr)
            continue
        logs = [node.obj for node in nodes.values() if isinstance(node.obj, ExecutionLog)]
        if logs:
            out += logs
            continue
        for node in nodes.values():
            if node.call != "execute" or not node.args.get("hold") or not node.parents:
                continue
            execution, capture = nodes[node.parents[0]].obj, node.obj
            if not hasattr(execution, "moves") or not hasattr(capture, "arms"):
                continue
            arm = execution.poses.arm
            start = getattr(execution.poses.capture, "arms", {}).get(arm)
            arm_moves = [
                m for m in execution.moves if any(k.removesuffix(".pos") in JOINTS for k in m.targets)
            ]
            if start is None or arm not in capture.arms or not arm_moves:
                continue
            last = arm_moves[-1]
            max_vel = float(node.args.get("arg1", node.args.get("max_vel", 20.0)))
            moves = tuple(
                MoveReading(
                    m.label,
                    dict(m.targets),
                    m.tol,
                    m.timeout,
                    "done",
                    None,
                    {k: float(v) for k, v in capture.arms[arm].items()} if m is last else None,
                )
                for m in execution.moves[: execution.moves.index(last) + 1]
            )
            out.append(ExecutionLog(arm, max_vel, 2 * max_vel, dict(start), moves, "held"))
    return out


def main() -> None:
    """Command line entry point; see the module docstring."""
    from lerobot.felix_jr.harness import api

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sessions", type=Path, default=SESSIONS)
    parser.add_argument("--model", type=Path, default=api.MODEL)
    args = parser.parse_args()
    model = XleRobotModel.load(args.model)
    records = [replay(model, log, table_z=NO_TABLE_Z) for log in session_logs(args.sessions)]
    print(report(model, records))


if __name__ == "__main__":
    main()
