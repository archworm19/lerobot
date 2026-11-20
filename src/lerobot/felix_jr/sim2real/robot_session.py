"""The first robot session of the simulated grasps: today's grasp against the refined one (`docs/sim_harness.md`).

  uv run --extra felix_jr --extra felix_jr_sim --extra pyzmq-dep python -m lerobot.felix_jr.sim2real.robot_session \
      <name> [--endpoint tcp://<pi>:5555] [--bottle pyri 45 130 80] [--spots 5] [--max-vel 20] [--assume-estimate]

Everything goes through one harness (`api.Harness`, `SAFE`: every move of the head or the arm shown and asked
first). The table's height first (a capture where the head rests); the spots (`placement.placement_spots`: the arm
reaches the bottle there from the side). Then per spot, both grasps in turn (which first alternating from spot to
spot), each:

1. **The bottle on its outline** (`place_object`: the head turns to the spot, its live picture shows the outline;
   Enter when it stands on it). Fitted in that capture (`fit_cylinder`, marked inside the outline; its radius known):
   the head's estimate, its miss from the spot saved. `--assume-estimate`: the spot itself (a rehearsal against the
   fake robot, whose pictures have no bottle).
2. **Planned:** `grasp_candidates` (the right arm, from the side), A; today's: `plan_pick`, `prepare_execution`;
   the refined: `refine_grasp` (its predicted lifted share saved).
3. **Today's grasp checked in the simulator first** (`check_collisions`; sim2's camera board ran into the shoulder):
   the arm into itself or the table is a no-go unless you say so; the arm but its fingers into the bottle a warning.
   Nothing else of the simulator's is used for it. (The refined plan needs no check: refine refuses any that
   collides, and its bumps of the bottle count in its lifted share.)
4. **Run, uncorrected:** the whole pick in one `execute` (the plan from before the arm moved cannot run in parts),
   every move confirmed by this script but one: before the move into the grasp, at the attack point, a wrist frame
   is saved and you are asked to go in (protocol rule 1's check 4: only the bottle between the open jaws; your call).
   The arm holds at the lift.
5. **The outcome:** asked (`refine.MODES`); you take the bottle out of the jaws; the arms go home.

Saved in `harness/scratch/sessions/<name>/` (gitignored): `workspace.pkl` (the harness's state, every `ExecutionLog`
among it: `shadow_report` reads them), `results.jsonl` (a line per grasp), `wrist_<spot>_<kind>.png` (at the attack).
"""

import argparse
import json
import pickle  # nosec B403: the harness's own state, written here
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lerobot.felix_jr.harness import api, placement
from lerobot.felix_jr.harness.objects import Object
from lerobot.felix_jr.harness.primitives import Cylinder

from .refine import MODES

SESSIONS = Path(__file__).resolve().parents[1] / "harness" / "scratch" / "sessions"
KINDS = ("today", "refined")
PICK_GRIP = {"start": "open", "pick": "close"}
SIDE_GRASPS = {"psi": (0.0, 30.0)}


@dataclass(frozen=True)
class Bottle:
    """The bottle: its label, diameter, height (m) and mass (kg)."""

    label: str
    diameter: float
    height: float
    mass: float


class GoIn:
    """The harness's `confirm` for an uncorrected pick: every move approved but the one after the attack point.

    There: a wrist frame saved (`frame`), and the person asked to go in.
    """

    def __init__(self, h: "api.Harness", arm: str, ask: Any = input) -> None:
        """Approving for `h`'s `execute(step=True)`; `frame` is set before each pick."""
        self.h, self.arm, self.ask = h, arm, ask
        self.frame: Path | None = None
        self.last = ""
        self.went_in: bool | None = None

    def __call__(self, label: str) -> bool:
        """Approve `label`'s move, unless it is the first past the attack point: then save and ask."""
        previous, self.last = self.last, label
        if not previous.startswith("pick.back"):
            return True
        if self.frame is not None:
            wrist = self.h.client.wrist_capture(self.arm)
            cv2.imwrite(str(self.frame), cv2.cvtColor(wrist.color, cv2.COLOR_RGB2BGR))
        answer = self.ask("  at the attack point: only the bottle between the open jaws? go in [y/N] ")
        self.went_in = answer.strip().lower().startswith("y")
        return self.went_in


def fit_marks(view: Any, xy: tuple[float, float], radius: float, height: float, table_z: float) -> np.ndarray:
    """Marks well inside the bottle's outline in `view` (pixels): its silhouette for `fit_cylinder`."""
    angles = np.linspace(0.0, 2 * np.pi, 12, endpoint=False)
    points = [
        [xy[0] + 0.6 * radius * np.cos(a), xy[1] + 0.6 * radius * np.sin(a), table_z + f * height]
        for f in (0.15, 0.5, 0.85)  # (the whole bottle: `fit_cylinder` fits only what is marked)
        for a in angles
    ]
    return view.project(np.array(points))


# the head's table this far from the calibration's floor: asked which to use (the head's came out 3-4 cm under the
# gripper sweeps' 77.3-77.8 cm, sim1-sim3: 73.7, 74.8)
TABLE_TOLERANCE_M = 0.015


def check_table(measured: float, floor: float | None, ask: Any = input) -> float:
    """The table's height to use: the head's measure, unless it is `TABLE_TOLERANCE_M` from the calibration's floor.

    Then the person is asked (a bad depth, or the robot moved to another table).
    """
    if floor is None or abs(measured - floor) <= TABLE_TOLERANCE_M:
        return measured
    answer = ask(
        f"the head measured the table at {100 * measured:.1f} cm, the calibration has it at {100 * floor:.1f}: "
        "use the calibration's? [Y/n] "
    )
    return measured if answer.strip().lower().startswith("n") else floor


def ask_mode(ask: Any = input) -> str:
    """The outcome, by its number in `refine.MODES` (or "not run")."""
    menu = ", ".join(f"{k} {m}" for k, m in enumerate(MODES))
    while True:
        answer = ask(f"  how did it end? ({menu}; n: not run) ").strip().lower()
        if answer == "n":
            return "not run"
        if answer.isdigit() and int(answer) < len(MODES):
            return MODES[int(answer)]


def trial(
    h: "api.Harness",
    go_in: GoIn,
    folder: Path,
    bottle: Bottle,
    spot: int,
    xy: tuple[float, float],
    kind: str,
    table_z: float,
    max_vel: float,
    assume_estimate: bool,
    ask: Any = input,
) -> dict[str, Any] | None:
    """One grasp of one spot (the module docstring); its record, or None if the person stopped at the placing."""
    radius = bottle.diameter / 2
    capture = h.call("place_object", xy[0], xy[1], bottle.diameter, bottle.height, f"spot {spot}: {kind}")
    if capture is None:
        return None
    if assume_estimate:
        target = Object(
            bottle.label, {"body": Cylinder.upright(np.array([*xy, table_z]), bottle.height, radius)}
        )
        target = h.ws[h.ws.add(target, "assumed", parents=(h.ws.require_id(capture),))]
    else:
        view = h.call("full_view", capture)
        target = h.call(
            "fit_cylinder",
            view,
            fit_marks(view, xy, radius, bottle.height, table_z),
            "up",
            bottle.label,
            radius,
        )
    fitted = target.part().bottom_root if hasattr(target.part(), "bottom_root") else np.array([*xy, table_z])
    record: dict[str, Any] = {
        "spot": spot,
        "xy": list(xy),
        "kind": kind,
        "bottle": bottle.__dict__,
        "estimate_xy": [float(fitted[0]), float(fitted[1])],
        "estimate_miss_m": float(np.hypot(fitted[0] - xy[0], fitted[1] - xy[1])),
        "time": time.time(),
    }
    print("  planning the grasps (grasp_candidates: ~1-2 min) ...", flush=True)
    candidates = h.call("grasp_candidates", target, capture, arms=("right",), where=SIDE_GRASPS)
    if not candidates.candidates:
        return {**record, "outcome": "no grasp", "why": json.dumps(candidates.rejected_by_reason())}
    if kind == "refined":
        print("  refining it in the simulator (~20-40 s) ...", flush=True)
        plan = h.call("refine_grasp", target, candidates, "A", mass_kg=bottle.mass, table_z=table_z)
        if plan.execution is None:
            return {**record, "outcome": "no grasp", "why": plan.best.refused}
        execution = plan.execution
        record |= {"predicted": plan.best.lifted, "accepted": plan.accepted, "params": plan.best.params}
    else:
        poses = h.call("plan_pick", candidates, "A")
        execution = h.call("prepare_execution", poses, PICK_GRIP)
        record |= {"params": {**candidates["A"].claw.grasp.params, "depth": candidates["A"].claw.depth}}
    if kind == "today":  # (the refined plan was simulated in choosing it: refine refuses one that collides)
        verdict = h.call("check_collisions", execution, target, table_z=table_z)
        record["sim_collisions"] = verdict
        if verdict.startswith("collides"):
            print(f"  the simulator sees the arm collide: {verdict}")
            if not ask("  run it anyway? [y/N] ").strip().lower().startswith("y"):
                return {**record, "outcome": "no-go"}
        elif verdict != "clear":
            print(f"  note, in the simulator it {verdict} (at the estimate): watch it going in")
    go_in.frame, go_in.last, go_in.went_in = folder / f"wrist_{spot}_{kind}.png", "", None
    try:
        h.call("execute", execution, max_vel, step=True, hold=True)
    except api.RefusedError:
        record["outcome"] = "not run"
    except RuntimeError as e:
        record["outcome"] = "not run" if go_in.went_in is False else "stopped"
        record["why"] = str(e)[:200]
        if go_in.went_in is not False:
            print(f"  the execution stopped: {e}")
    if "outcome" not in record:
        record["outcome"] = ask_mode(ask)
    record["went_in"] = go_in.went_in
    ask("  take the bottle out of the jaws (by hand), then Enter: the arms go home ")
    h.call("home")
    return record


def main() -> None:
    """Command line entry point; see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", help="the session's folder under harness/scratch/sessions/")
    parser.add_argument("--endpoint", default=api.ENDPOINT)
    parser.add_argument(
        "--bottle", nargs=4, default=["pyri", "45", "130", "80"], metavar=("LABEL", "MM", "MM", "G")
    )
    parser.add_argument("--spots", type=int, default=5)
    parser.add_argument("--max-vel", type=float, default=20.0)
    parser.add_argument("--assume-estimate", action="store_true", help="the spot as the estimate (no fit)")
    args = parser.parse_args()
    label, d, height, mass = args.bottle
    bottle = Bottle(label, float(d) / 1000, float(height) / 1000, float(mass) / 1000)
    folder = SESSIONS / args.name
    folder.mkdir(parents=True, exist_ok=True)
    h = api.Harness(endpoint=args.endpoint)
    state = folder / "workspace.pkl"
    if state.is_file():
        h.restore(pickle.loads(state.read_bytes()))  # nosec B301: this session's own state
    go_in = GoIn(h, "right")
    h.confirm = go_in
    table = h.call("table_height", h.call("capture"))
    table_z = check_table(table.h, h.model.floor_z)
    spots = placement.placement_spots(h.model, bottle.diameter / 2, bottle.height, table_z, n=args.spots)
    print(f"table at {100 * table_z:.1f} cm; spots: " + "; ".join(f"({x:+.3f}, {y:+.3f})" for x, y in spots))
    done = set()
    results = folder / "results.jsonl"
    if results.is_file():
        done = {  # (a trial that found no grasp, or stopped on its own, is tried again)
            (r["spot"], r["kind"])
            for r in map(json.loads, results.read_text().splitlines())
            if r and r["outcome"] not in ("no grasp", "stopped")
        }
    try:
        for k, xy in enumerate(spots, start=1):
            for kind in KINDS if k % 2 else tuple(reversed(KINDS)):
                if (k, kind) in done:
                    continue
                print(f"\n=== spot {k} of {len(spots)}, {kind}'s grasp ===")
                record = trial(
                    h, go_in, folder, bottle, k, xy, kind, table.h, args.max_vel, args.assume_estimate
                )
                state.write_bytes(pickle.dumps(h.state()))
                if record is None:
                    print("stopped")
                    return
                with results.open("a") as f:
                    f.write(json.dumps(record) + "\n")
                print(
                    f"  {record['outcome']}; the head's estimate {1000 * record['estimate_miss_m']:.0f} mm off"
                    + (
                        f"; predicted {100 * record['predicted']:.0f} % lifted"
                        if "predicted" in record
                        else ""
                    )
                )
    finally:
        state.write_bytes(pickle.dumps(h.state()))
    print(f"\ndone: {results}; `shadow_report` reads {state}")


if __name__ == "__main__":
    main()
