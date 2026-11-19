"""The arm's shortfalls on the real robot, from the harness's sessions, and the simulator's arm against them.

  uv run --extra felix_jr --extra felix_jr_sim python -m lerobot.felix_jr.sim2real.shortfalls
      [--sessions harness/scratch/sessions] [--sim] [--kp 8.8 19.2] [--model <calibration.urdf>]

Every `execute(..., hold=True)` in the sessions (`harness/scratch/sessions/*/workspace.pkl`) ends with the arm held at
its last pose and a capture there: the last move's commanded joints against the capture's read ones (what the encoders
see: gravity against the servos' finite gain). Per kind of pose (`KINDS`: the attack, `pick.back`; the grasp, `pick`;
loaded, a lift or a stage with the object in the jaws): each joint's mean and spread, the gripper's offset (the model's
forward kinematics, the commanded pinch frame: x across the jaws, z along the gripper axis toward the tips) and its
height. The numbers the grasp environment's attack starts are drawn from (`docs/sim2real_grasp.md`).

`--sim`: each unloaded hold replayed in the simulator (`replay_hold`: the arm at the move's previous targets, the
servos ramped to the commanded joints at the harness's speed, settled), its shortfall against the real one, at
`robot.ARM_KP_NM_PER_RAD` and `robot.ARM_FRICTIONLOSS_NM`. `--fit`: those two per joint over a grid (`FIT_KP`,
`FIT_FRICTION`), on the holds that moved (`DIRECTION_DEG`: one that did not says nothing of the friction's side).
"""

import argparse
import pickle  # nosec B403: the harness's own sessions, written on this computer
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel, pinch_frame
from lerobot.utils.import_utils import _mujoco_available, require_package

from . import robot

if _mujoco_available:
    import mujoco

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
KINDS = ("attack", "grasp", "loaded", "other")
UNLOADED = ("attack", "grasp")  # (the jaws empty: the object's mass is not known for the rest)
SESSIONS = Path(__file__).resolve().parents[1] / "harness" / "scratch" / "sessions"
MOVE_DEG_S = 20.0  # the harness's `execute` speed
SETTLE_S = 1.5
NO_TABLE_Z = (
    -1.0
)  # the replays' table, out of the way (a hold is the servos' alone; some sessions' tables sat higher)
DIRECTION_DEG = 0.05  # a joint moved less than this into its hold: no direction (left out of the fit)
FIT_KP = (15.0, 25.0, 35.0, 50.0, 70.0, 100.0, 150.0)
FIT_FRICTION = (0.02, 0.05, 0.1, 0.2, 0.3, 0.45, 0.65)


@dataclass(frozen=True)
class HeldMove:
    """One held pose on the real robot: what was commanded, what was read there, and where the move came from."""

    session: str
    node: str
    arm: str
    label: str
    commanded: dict[str, float]  # `JOINTS`, LeRobot degrees
    read: dict[str, float]
    previous: dict[str, float]  # the move before's targets (the start: the same as `commanded` if none)
    gripper_pos: float  # read

    @property
    def kind(self) -> str:
        """Which of `KINDS`: the attack (`pick.back`), the grasp (`pick`), loaded (a lift or stage), other."""
        name = self.label.split(" ")[0]
        if name.endswith(".back"):
            return "attack"
        if name == "pick":
            return "grasp"
        if name.startswith(("lift", "up", "stage")):
            return "loaded"
        return "other"

    @property
    def direction(self) -> np.ndarray:
        """Which way each joint moved into the hold: +1, -1, or 0 (less than `DIRECTION_DEG`)."""
        delta = np.array([self.commanded[j] - self.previous[j] for j in JOINTS])
        return np.where(np.abs(delta) < DIRECTION_DEG, 0.0, np.sign(delta))

    @property
    def shortfall(self) -> np.ndarray:
        """Read minus commanded, per joint (deg)."""
        return np.array([self.read[j] - self.commanded[j] for j in JOINTS])


class _Gone:
    """A class the sessions' pickles name that the harness has since dropped (loaded as an empty stand-in)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)


class _Loader(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        try:
            return super().find_class(module, name)
        except AttributeError:
            return type(name, (_Gone,), {})


def held_moves(sessions: Path = SESSIONS) -> list[HeldMove]:
    """Every held execute's last move, all sessions (a session that cannot be read is skipped, said on stderr)."""
    out = []
    for path in sorted(sessions.glob("*/workspace.pkl")):
        try:
            with path.open("rb") as f:
                nodes = _Loader(f).load()["ws"].nodes
        except Exception as e:  # (a session from a harness too old to load at all)
            print(f"skipped {path.parent.name}: {type(e).__name__} {e}", file=sys.stderr)
            continue
        for name, node in nodes.items():
            if node.call != "execute" or not node.args.get("hold") or not node.parents:
                continue
            execution, capture = nodes[node.parents[0]].obj, node.obj
            if not hasattr(execution, "moves") or not hasattr(capture, "arms"):
                continue
            arm = execution.poses.arm
            moves = [m for m in execution.moves if any(k.removesuffix(".pos") in JOINTS for k in m.targets)]
            if not moves or arm not in capture.arms:
                continue
            read = capture.arms[arm]
            commanded = {j: float(moves[-1].targets.get(f"{j}.pos", read[f"{j}.pos"])) for j in JOINTS}
            previous = (
                {j: float(moves[-2].targets.get(f"{j}.pos", commanded[j])) for j in JOINTS}
                if len(moves) > 1
                else dict(commanded)
            )
            out.append(
                HeldMove(
                    path.parent.name,
                    name,
                    arm,
                    moves[-1].label,
                    commanded,
                    {j: float(read[f"{j}.pos"]) for j in JOINTS},
                    previous,
                    float(read.get("gripper.pos", float("nan"))),
                )
            )
    return out


def gripper_offset(
    model: XleRobotModel, move: HeldMove, read: Mapping[str, float]
) -> tuple[np.ndarray, float]:
    """Where the gripper is at `read` against `move`'s commanded pose: in the commanded pinch frame (m), and up (m)."""
    commanded = model.frame_pose(
        pinch_frame(move.arm),
        model.joint_vector({move.arm: {f"{j}.pos": v for j, v in move.commanded.items()}}),
    )
    actual = model.frame_pose(
        pinch_frame(move.arm), model.joint_vector({move.arm: {f"{j}.pos": v for j, v in read.items()}})
    )
    delta = actual[:3, 3] - commanded[:3, 3]
    return commanded[:3, :3].T @ delta, float(delta[2])


def stats(model: XleRobotModel, moves: Sequence[HeldMove]) -> dict[str, dict[str, Any]]:
    """Per kind: the moves, each joint's mean and spread (deg), the gripper's offset's (cm) and its height's."""
    out = {}
    for kind in KINDS:
        group = [m for m in moves if m.kind == kind]
        if not group:
            continue
        d = np.array([m.shortfall for m in group])
        offsets = [gripper_offset(model, m, m.read) for m in group]
        o = 100 * np.array([x[0] for x in offsets])
        z = 100 * np.array([x[1] for x in offsets])
        spread = (lambda a: a.std(0, ddof=1)) if len(group) > 1 else (lambda a: np.zeros_like(a.mean(0)))
        out[kind] = {
            "n": len(group),
            "sessions": sorted({m.session for m in group}),
            "mean_deg": d.mean(0),
            "std_deg": spread(d),
            "offset_mean_cm": o.mean(0),
            "offset_std_cm": spread(o),
            "up_mean_cm": float(z.mean()),
            "up_std_cm": float(spread(z)),
        }
    return out


def report(table: Mapping[str, Mapping[str, Any]]) -> str:
    """`stats` as text, a few lines a kind."""
    lines = []
    for kind, s in table.items():
        lines += [
            f"{kind}: {s['n']} moves ({', '.join(s['sessions'])})",
            "  read - commanded (deg): "
            + ", ".join(
                f"{j} {m:+.2f} +- {e:.2f}"
                for j, m, e in zip(JOINTS, s["mean_deg"], s["std_deg"], strict=True)
            ),
            f"  gripper (cm): across {s['offset_mean_cm'][0]:+.2f} +- {s['offset_std_cm'][0]:.2f}, along "
            f"{s['offset_mean_cm'][2]:+.2f} +- {s['offset_std_cm'][2]:.2f}, up {s['up_mean_cm']:+.2f} +- "
            f"{s['up_std_cm']:.2f}",
        ]
    return "\n".join(lines)


def replay_hold(model: XleRobotModel, move: HeldMove, scene: Any = None) -> dict[str, float]:
    """`move` in the simulator, the jaws empty: from its previous targets, ramped at `MOVE_DEG_S`, settled; read.

    `scene`: one built for `move.arm` to reuse (`scene.build`), else one is built.
    """
    require_package("mujoco", extra="felix_jr_sim")
    from .scene import build

    if scene is None:
        scene = build(model, move.arm, table_z=NO_TABLE_Z)
    m, d = scene.m, scene.d
    mujoco.mj_resetData(m, d)
    gripper = move.gripper_pos if np.isfinite(move.gripper_pos) else 30.0
    scene.set_pose({**{f"{j}.pos": v for j, v in move.previous.items()}, "gripper.pos": gripper})
    start = np.radians([move.previous[j] for j in JOINTS])
    goal = np.radians([move.commanded[j] for j in JOINTS])
    act = [m.actuator(j).id for j in JOINTS]
    seconds = max(float(np.degrees(np.abs(goal - start)).max()) / MOVE_DEG_S, 0.0)
    steps = int(seconds / m.opt.timestep)
    for k in range(steps):
        d.ctrl[act] = start + (k + 1) / steps * (goal - start)
        mujoco.mj_step(m, d)
    d.ctrl[act] = goal
    for _ in range(int(SETTLE_S / m.opt.timestep)):
        mujoco.mj_step(m, d)
    return {j: float(np.degrees(d.qpos[m.joint(j).qposadr[0]])) for j in JOINTS}


def compare(
    model: XleRobotModel, moves: Sequence[HeldMove], kinds: Sequence[str] = UNLOADED
) -> dict[str, dict[str, np.ndarray]]:
    """Per kind in `kinds`: the simulated shortfalls' mean against the real ones', and the RMS of their difference."""
    require_package("mujoco", extra="felix_jr_sim")
    from .scene import build

    scenes = {arm: build(model, arm, table_z=NO_TABLE_Z) for arm in {m.arm for m in moves}}
    out = {}
    for kind in kinds:
        group = [m for m in moves if m.kind == kind]
        if not group:
            continue
        real = np.array([m.shortfall for m in group])
        sim = np.array(
            [
                [r[j] - m.commanded[j] for j in JOINTS]
                for m in group
                for r in [replay_hold(model, m, scenes[m.arm])]
            ]
        )
        out[kind] = {
            "real_mean_deg": real.mean(0),
            "sim_mean_deg": sim.mean(0),
            "rms_deg": np.sqrt(((sim - real) ** 2).mean(0)),
            "sim": sim,
            "real": real,
        }
    return out


def fit(
    model: XleRobotModel,
    moves: Sequence[HeldMove],
    kps: Sequence[float] = FIT_KP,
    frictions: Sequence[float] = FIT_FRICTION,
) -> dict[str, dict[str, float]]:
    """Per joint, the gain and friction of the grid that replay the unloaded holds best (RMS, the moved ones only).

    Every joint gets the same grid point in a replay and is scored on its own (the joints' sags barely interact).
    """
    from .finger_models import settings

    require_package("mujoco", extra="felix_jr_sim")
    group = [m for m in moves if m.kind in UNLOADED]
    real = np.array([m.shortfall for m in group])
    moved = np.array([m.direction != 0 for m in group])
    best: dict[str, dict[str, float]] = {}
    for kp in kps:
        for friction in frictions:
            with settings(
                {
                    "ARM_KP_NM_PER_RAD": dict.fromkeys(JOINTS, kp),
                    "ARM_FRICTIONLOSS_NM": dict.fromkeys(JOINTS, friction),
                }
            ):
                c = compare(model, group, kinds=UNLOADED)
            sim = np.vstack([c[k]["sim"] for k in UNLOADED if k in c])
            ordered = np.vstack([c[k]["real"] for k in UNLOADED if k in c])
            assert np.allclose(np.sort(ordered, 0), np.sort(real, 0))
            for i, joint in enumerate(JOINTS):
                sel = np.vstack([moved[[g.kind == k for g in group]] for k in UNLOADED if k in c])[:, i]
                if not sel.any():
                    continue
                rms = float(np.sqrt(((sim[sel, i] - ordered[sel, i]) ** 2).mean()))
                if joint not in best or rms < best[joint]["rms_deg"]:
                    best[joint] = {"kp": kp, "frictionloss": friction, "rms_deg": rms, "n": int(sel.sum())}
    return best


def main() -> None:
    """Command line entry point; see the module docstring."""
    from lerobot.felix_jr.harness import api

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sessions", type=Path, default=SESSIONS)
    parser.add_argument("--model", type=Path, default=api.MODEL)
    parser.add_argument("--sim", action="store_true", help="the simulator's arm on the same holds")
    parser.add_argument("--fit", action="store_true", help="the arm servos' gain and friction, per joint")
    args = parser.parse_args()
    model = XleRobotModel.load(args.model)
    moves = held_moves(args.sessions)
    print(f"{len(moves)} held moves")
    print(report(stats(model, moves)))
    if args.fit:
        for joint, b in fit(model, moves).items():
            print(
                f"  {joint}: kp {b['kp']:g} N m/rad, frictionloss {b['frictionloss']:g} N m: {b['rms_deg']:.2f} deg RMS "
                f"over {b['n']} holds"
            )
    if args.sim:
        print(f"the arm's servos: kp {robot.ARM_KP_NM_PER_RAD}, frictionloss {robot.ARM_FRICTIONLOSS_NM}")
        for kind, c in compare(model, moves).items():
            print(f"{kind} in the simulator (deg, per joint {', '.join(JOINTS)}):")
            for label in ("real_mean_deg", "sim_mean_deg", "rms_deg"):
                print(f"  {label.removesuffix('_deg'):9s} " + " ".join(f"{v:+6.2f}" for v in c[label]))


if __name__ == "__main__":
    main()
