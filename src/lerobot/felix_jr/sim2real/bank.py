"""The bank of the grasp environment's starts, built offline (`docs/sim2real_grasp.md`, the grasp environment's starts).

  uv run --extra felix_jr --extra felix_jr_sim python -m lerobot.felix_jr.sim2real.bank <out.jsonl>
      [--placements 20] [--arm right] [--seed 0] [--workers 1] [--model <calibration.urdf>]
      [--object DIAMETER_MM HEIGHT_MM MASS_G] [--anchors N]

Per placement: an upright cylinder drawn (`BankConfig`: its diameter, height and mass, where it stands from the arm's
base), a feasible grasp drawn at random through the harness's filters (`harness.grasping.first_feasible`, at each of
the depths), each grasp that passes simulated under physics draws (`grasp_check.SimAccept`) until one holds at
`min_rate`: the anchor. Its states at the grasp, in the jaws and lifted (`grasp_check.SNAPSHOTS`, from a draw that
held) are the later pools' starts; every grasp simulated is kept with its trials, the rejected too (the labels for a
learned grasp quality). A placement nothing reaches is kept as well (no anchor): the yield.

The file is JSON lines: the config first (`"kind": "config"`), then one line a placement (`"kind": "placement"`), so a
run stopped part way keeps what it did and continues (`--resume`: the placements already written are skipped; the
same seed draws the same ones). `load` reads it back.
"""

import argparse
import json
import multiprocessing
import time
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from lerobot.felix_jr.calibration.gripper_sweep import table_height
from lerobot.felix_jr.calibration.joint_sim import REST
from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel
from lerobot.felix_jr.harness.grasping import GraspCandidate, GraspSettings, first_feasible
from lerobot.felix_jr.harness.primitives import Cylinder as CylinderPrimitive, RigidBody
from lerobot.felix_jr.harness.reach import FrameSolution, close_gripper_pos

from . import robot
from .grasp_check import GraspTrial, SimAccept

if TYPE_CHECKING:
    from lerobot.felix_jr.harness.objects import Capture

# the reachable table in front of the arm (from its base: ahead along the robot's facing, -x; to the side, y): a 50 mm
# cylinder, side grasps (`PSI_DEG`), none closer than ~0.22 m ahead, all of +-0.15 m to the side from 0.25 m out;
# closer in only from well above (2026-11-18)
AHEAD_M = (0.22, 0.35)
SIDE_M = (-0.15, 0.15)
# the cylinders, spread around the measured bottles (45-65 mm, 114-200 mm: `docs/sim2real_grasp.md`), none wider than
# 60 mm (wider ones hardly held in the simulator, 3 of 11, as expected of the claw); the mass about
# what the real claw lifted (pyri, 80 g, held; feno, 116 g, slipped; the simulator agrees): with 110-260 g one placement
# in six anchored, with 40-100 g three (the rest reached nothing; 2026-11-18)
DIAMETER_M = (0.040, 0.060)
HEIGHT_M = (0.10, 0.20)
MASS_KG = (0.04, 0.12)
# the depths tried (the object's center back from the tips, m), and one just past the radius (the deepest the jaws
# fit, `to_claw`: the filters refuse the rest)
DEPTHS_M = (0.0, 0.01, 0.02)
DEEP_PAST_RADIUS_M = 0.005
# the grasps tried: mostly from the side (psi, deg: 0 square to the axis, 90 from above)
PSI_DEG = (0.0, 30.0)
# most targets fail the arm-plane check, before any IK (~1 ms each): 400 tries reached ~80 IKs and anchored none of six
# placements; 3000 are ~10 s where nothing reaches (2026-11-18)
MAX_TRIES = 3000


@dataclass(frozen=True)
class BankConfig:
    """What a bank is drawn from (the module's constants by default); saved as the file's first line."""

    arm: str = "right"
    ahead_m: tuple[float, float] = AHEAD_M
    side_m: tuple[float, float] = SIDE_M
    diameter_m: tuple[float, float] = DIAMETER_M
    height_m: tuple[float, float] = HEIGHT_M
    mass_kg: tuple[float, float] = MASS_KG
    depths_m: tuple[float, ...] = DEPTHS_M
    psi_deg: tuple[float, float] = PSI_DEG
    max_tries: int = MAX_TRIES
    n_draws: int = 3
    min_rate: float = 2 / 3
    # distinct grasps accepted a placement (`grasp_check.SimAccept.want`): the pools' anchors, all of them
    anchors_per_placement: int = 1
    finger_model: str = "sprung"
    # the filters' settings that differ from the harness's: no sag (the real arm's), the solves near the ideal pose
    # (loose bounds put the gripper's body in the object), the arm-plane check before any IK (`GraspSettings`)
    rotation_bounds_deg: tuple[float, float, float] = (15.0, 10.0, 10.0)
    arm_plane_tol_deg: float = 15.0
    seed: int = 0

    def settings(self) -> GraspSettings:
        """The filters' settings.

        No wrist view (`see_object`: the object's grasp in the wrist camera's picture at the attack point, clear of the
        fingers, is for a policy steering by that camera; the grasp policy trained on the bank does not see it, and the
        check refused every grasp deeper than 10 mm, the fingers in the way of the view: 2026-11-21), and the camera's
        clearance at the attack only (the simulator judges the grasp's collisions, its board modeled; as the refine's).
        """
        return GraspSettings(
            sag_lift=0.0,
            rotation_bounds_deg=self.rotation_bounds_deg,
            arm_plane_tol_deg=self.arm_plane_tol_deg,
            see_object=False,
            camera_clearance_at_grasp=False,
        )

    def depths(self, radius: float) -> tuple[float, ...]:
        """The depths tried on a cylinder of `radius`: the config's, and just past the radius."""
        return tuple(sorted({*self.depths_m, round(radius + DEEP_PAST_RADIUS_M, 4)}))


@dataclass(frozen=True, eq=False)
class Placement:
    """One object drawn (`body`: an upright cylinder with a mass, where it stands, root frame), and its index.

    In the bank's line its record is the body's (`RigidBody.to_record`) with the index: `to_record` / `from_record`.
    """

    index: int
    body: RigidBody

    def __eq__(self, other: object) -> bool:
        """The same index and the same body (by record: the shape's pose is an array)."""
        return isinstance(other, Placement) and self.to_record() == other.to_record()

    __hash__ = None  # type: ignore[assignment]

    @property
    def bottom_root(self) -> tuple[float, float, float]:
        """Where its bottom end's center stands (root frame, m)."""
        b = self.body.shape.bottom_root
        return float(b[0]), float(b[1]), float(b[2])

    @property
    def radius(self) -> float:
        """The cylinder's radius (m)."""
        return float(self.body.shape.radius)

    @property
    def height(self) -> float:
        """The cylinder's height (m)."""
        return float(self.body.shape.length)

    @property
    def mass_kg(self) -> float:
        """Its mass (kg)."""
        return self.body.mass_kg

    def primitive(self) -> CylinderPrimitive:
        """The harness's primitive for it (its shape, posed)."""
        return self.body.shape

    def to_record(self) -> dict[str, Any]:
        """Its line's record: the index and the body's record."""
        return {"index": self.index, **self.body.to_record()}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "Placement":
        """`to_record`'s record back (and older lines', whose records name no shape: cylinders)."""
        return cls(int(record["index"]), RigidBody.from_record(record))


def placements(model: XleRobotModel, config: BankConfig, n: int) -> list[Placement]:
    """The first `n` placements of `config`'s seed (each its own generator: the same seed, the same placements)."""
    base = model.frame_pose(f"{config.arm}_base_link", model.joint_vector())[:3, 3]
    table_z = table_height(model, config.arm)
    out = []
    for i, seq in enumerate(np.random.SeedSequence(config.seed).spawn(n)):
        rng = np.random.default_rng(seq)
        bottom = (
            float(base[0] - rng.uniform(*config.ahead_m)),
            float(base[1] + rng.uniform(*config.side_m)),
            float(table_z),
        )
        radius = float(rng.uniform(*config.diameter_m) / 2)
        height = float(rng.uniform(*config.height_m))
        mass = float(rng.uniform(*config.mass_kg))
        params = {"radius": radius, "height": height}
        out.append(Placement(i, RigidBody.upright(CylinderPrimitive.SHAPE, params, mass, np.array(bottom))))
    return out


def _solution(s: FrameSolution | None) -> dict[str, Any] | None:
    if s is None:
        return None
    return {"joints": dict(s.joints), "root_from_pinch": s.root_from_pinch.tolist()}


def _trial(t: GraspTrial) -> dict[str, Any]:
    return {
        "depth": t.depth,
        "draw": t.draw,
        "held": t.held,
        "rose_m": t.rose_m,
        "jaws_rose_m": t.jaws_rose_m,
        "note": t.note,
    }


def _grasp(model: XleRobotModel, c: GraspCandidate) -> dict[str, Any]:
    """A simulated grasp as saved: the claw's target, the solved attack, grasp and lift, the gripper's open and close."""
    claw = c.claw
    return {
        "params": dict(claw.grasp.params),
        "fixed_side": claw.fixed_side,
        "depth": claw.depth,
        "opening": claw.grasp.opening,
        "open_apart": claw.open_apart,
        "close_gripper_pos": close_gripper_pos(model, claw.arm, claw.grasp.opening),
        "root_from_pinch": claw.root_from_pinch.tolist(),
        "attack": _solution(c.attack),
        "grasp": _solution(c.grasp),
        "lift": _solution(c.lift),
    }


def run_placement(model: XleRobotModel, config: BankConfig, placement: Placement) -> dict[str, Any]:
    """One placement through the filters and the simulation: its line in the bank (the module docstring)."""
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, placement.index, 1]))
    table_z = placement.bottom_root[2]
    target = placement.primitive()
    # the arm at rest, no head (what the filters read of a capture: the IK's start; no camera here)
    capture = cast("Capture", SimpleNamespace(arms={config.arm: dict(REST)}, head_ticks=None))
    accept = SimAccept(
        model,
        target,
        placement.mass_kg,
        table_z,
        rng,
        n_draws=config.n_draws,
        min_rate=config.min_rate,
        finger_model=config.finger_model,
        snapshots=True,
        want=config.anchors_per_placement,
    )
    found = first_feasible(
        target,
        model,
        capture,
        rng,
        arms=(config.arm,),
        where={"psi": config.psi_deg},
        settings=config.settings(),
        depths=config.depths(placement.radius),
        max_tries=config.max_tries,
        accept=accept,
    )
    simulated, anchors = [], []
    for i, (candidate, runs) in enumerate(zip(accept.candidates, accept.trials, strict=True)):
        row = _grasp(model, candidate) | {
            "trials": [_trial(t) for t in runs],
            "rate": sum(t.held for t in runs) / len(runs),
        }
        simulated.append(row)
        if (
            i in accept.accepted
        ):  # (every distinct grasp accepted: the first is `found.candidate` with one wanted)
            held = next(t for t in runs if t.held)
            anchors.append(
                row
                | {
                    "states": {
                        stage: {k: v.tolist() for k, v in state.items()}
                        for stage, state in held.states.items()
                    },
                    "states_draw": held.draw,
                }
            )
    return {
        "kind": "placement",
        "placement": placement.to_record(),
        "anchor": anchors[0] if anchors else None,  # (the first: a bank's readers before `anchors`)
        "anchors": anchors,
        "simulated": simulated,
        "tried": found.tried,
        "rejected_by_stage": found.rejected_by_stage,
        "seconds": found.seconds,
    }


_WORKER: dict[str, Any] = {}


def _init_worker(model_path: str, config: BankConfig) -> None:
    _WORKER["model"], _WORKER["config"] = XleRobotModel.load(Path(model_path)), config


def _work(placement: Placement) -> dict[str, Any]:
    return run_placement(_WORKER["model"], _WORKER["config"], placement)


def build_bank(
    model_path: Path, config: BankConfig, n: int, out: Path, workers: int = 1, resume: bool = False
) -> Iterator[dict[str, Any]]:
    """`n` placements into `out` (JSON lines, the module docstring), each line yielded as it is written."""
    model = XleRobotModel.load(model_path)
    done: set[int] = set()
    if resume and out.exists():
        saved, lines = load(out)
        if saved != config:
            raise ValueError(f"{out} was built with another config: {saved}")
        done = {line["placement"]["index"] for line in lines}
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "kind": "config",
            "model": str(model_path),
            "table_z": table_height(model, config.arm),
            "servos": servos(),
        }
        out.write_text(json.dumps(meta | {"config": asdict(config)}) + "\n")
    todo = [p for p in placements(model, config, n) if p.index not in done]
    with out.open("a") as f:
        if workers <= 1:
            results: Iterator[dict[str, Any]] = (run_placement(model, config, p) for p in todo)
            for line in results:
                f.write(json.dumps(line) + "\n")
                f.flush()
                yield line
        else:
            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(workers, _init_worker, (str(model_path), config)) as pool:
                for line in pool.imap_unordered(_work, todo):
                    f.write(json.dumps(line) + "\n")
                    f.flush()
                    yield line


def servos() -> dict[str, Any]:
    """The arm's servos the simulator has now (`robot.ARM_*`): saved with a bank, its snapshots made under them."""
    return {"kp": robot.ARM_KP_NM_PER_RAD, "frictionloss": robot.ARM_FRICTIONLOSS_NM}


def meta(path: Path) -> dict[str, Any]:
    """A bank's first line: its config, the model, the table, the servos (None: a bank from before they were saved)."""
    with path.open() as f:
        return json.loads(f.readline())


def load(path: Path) -> tuple[BankConfig, list[dict[str, Any]]]:
    """A bank's config and its placements' lines (`run_placement`)."""
    lines = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    raw = lines[0]["config"]
    fields: dict[str, Any] = {k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()}
    config = BankConfig(**fields)
    return config, [line for line in lines[1:] if line["kind"] == "placement"]


@dataclass
class Tally:
    """A bank's yield so far: placements, anchors, grasps simulated and accepted, time."""

    placements: int = 0
    anchors: int = 0
    anchored: int = 0
    simulated: int = 0
    seconds: float = 0.0
    by_depth: dict[str, list[int]] = field(default_factory=dict)  # depth -> [simulated, accepted]

    def add(self, line: dict[str, Any], min_rate: float) -> None:
        """Count one placement's line."""
        self.placements += 1
        self.anchors += len(line.get("anchors") or ([line["anchor"]] if line["anchor"] else []))
        self.anchored += line["anchor"] is not None
        self.simulated += len(line["simulated"])
        self.seconds += line["seconds"]
        for g in line["simulated"]:
            row = self.by_depth.setdefault(f"{1000 * g['depth']:g} mm", [0, 0])
            row[0] += 1
            row[1] += g["rate"] >= min_rate

    def __str__(self) -> str:
        """One line: the counts, the accepted by depth, the time a placement."""
        depth = ", ".join(f"{k}: {a}/{s}" for k, (s, a) in sorted(self.by_depth.items()))
        return (
            f"{self.anchored}/{self.placements} placements anchored ({self.anchors} anchors), {self.simulated} grasps simulated "
            f"(accepted by depth {depth or '-'}), {self.seconds / max(self.placements, 1):.1f} s a placement"
        )


def main() -> None:
    """Command line entry point; see the module docstring."""
    from lerobot.felix_jr.harness import api

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path, help="the bank (.jsonl; under a scratch/ folder)")
    parser.add_argument("--placements", type=int, default=20)
    parser.add_argument("--arm", default="right")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="continue a bank stopped part way")
    parser.add_argument("--model", type=Path, default=api.MODEL)
    parser.add_argument(
        "--object",
        nargs=3,
        type=float,
        default=None,
        metavar=("DIAMETER_MM", "HEIGHT_MM", "MASS_G"),
        help="one object for every placement (default: drawn over `DIAMETER_M`, `HEIGHT_M`, `MASS_KG`)",
    )
    parser.add_argument(
        "--anchors", type=int, default=1, help="distinct grasps accepted a placement (its anchors)"
    )
    args = parser.parse_args()
    one: dict[str, tuple[float, float]] = {}
    if args.object is not None:
        d, h, g = (v / 1000 for v in args.object)
        one = {"diameter_m": (d, d), "height_m": (h, h), "mass_kg": (g, g)}
    config = BankConfig(arm=args.arm, seed=args.seed, anchors_per_placement=args.anchors, **one)
    tally, started = Tally(), time.perf_counter()
    for line in build_bank(args.model, config, args.placements, args.out, args.workers, args.resume):
        tally.add(line, config.min_rate)
        p = line["placement"]
        a = line["anchor"]
        print(
            f"placement {p['index']}: {2000 * p['radius']:.0f} mm x {1000 * p['height']:.0f} mm, "
            f"{1000 * p['mass_kg']:.0f} g; {line['tried']} tried, {len(line['simulated'])} simulated, "
            + (
                f"{len(line.get('anchors') or [a])} anchors, the first at depth {1000 * a['depth']:.0f} mm "
                f"(rate {a['rate']:.2f})"
                if a
                else "no anchor"
            )
            + f"; {line['seconds']:.1f} s",
            flush=True,
        )
    print(tally, f"; {time.perf_counter() - started:.0f} s wall")


if __name__ == "__main__":
    main()
