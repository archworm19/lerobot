"""Why refined grasps fail in the simulator: position against physics, how each failure happened, and examples filmed.

  uv run --extra felix_jr --extra felix_jr_sim python -m lerobot.felix_jr.sim2real.failures <bank.jsonl> \
      <bench.jsonl> <out_dir> [--draws 32] [--workers 6]

`bench.jsonl`: `refine`'s benchmark (its command line): per run the starting grasp and the one taken. Each is played
(`refine.outcome`) under three conditions, the same `--draws` draws in each (`refine.draws`): **position** (the
object moved by the perception's error, the physics at the fitted values), **physics** (the object at its estimate, the
fingers' friction and stiffness and the object's mass drawn), **both** (as the benchmark scored them). Then: the lifted
share per condition; the failures by kind (`refine.MODES`: missed, knocked over, pushed away, slipped); under both, the
failure rate by the object's offset along and across the approach and by the drawn friction and mass; and one example
of each kind filmed (`examples.mp4`: the scene from beside and the wrist camera). Everything to `out_dir` (a scratch
folder): `outcomes.jsonl`, `examples.mp4`.
"""

import argparse
import json
import multiprocessing
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lerobot.felix_jr.calibration.joint_sim import REST
from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel
from lerobot.utils.import_utils import _mujoco_available

from . import bank, robot
from .refine import MODES, RefineSettings, draws, outcome, plan_for
from .shadow import Shadow

if _mujoco_available:
    import mujoco

CONDITIONS = ("position", "physics", "both")
START = {**REST, "gripper.pos": 2.0}
FPS, TILE = 30, (480, 360)

_W: dict[str, Any] = {}


def _init(model: XleRobotModel, placements: dict[int, bank.Placement], config: bank.BankConfig) -> None:
    _W.update(model=model, placements=placements, settings=config.settings(), plans={})


def _plan(placement: int, params: Mapping[str, float], side: int) -> Any:
    key = (placement, json.dumps(params, sort_keys=True), side)
    if key not in _W["plans"]:
        p = _W["placements"][placement]
        _W["plans"][key] = plan_for(_W["model"], "right", p.primitive(), params, side, _settings(), START)
    return _W["plans"][key]


def _settings() -> Any:
    from dataclasses import replace

    return replace(_W["settings"], camera_clearance_at_grasp=False)


def _task(task: dict[str, Any]) -> dict[str, Any]:
    p = _W["placements"][task["placement"]]
    planned = _plan(task["placement"], task["params"], task["side"])
    if isinstance(planned, str):
        return {**task, "mode": "refused", "refused": planned}
    candidate, execution = planned
    s = RefineSettings(table_z=p.bottom_root[2], mass_kg=p.mass_kg)
    out = outcome(
        _W["model"], "right", p.primitive(), execution, START, np.array(task["offset"]), task["draw"], s
    )
    approach = np.array(candidate.claw.grasp.approach_root[:2])
    approach /= np.linalg.norm(approach)
    across = np.array([-approach[1], approach[0]])
    offset = np.array(task["offset"])
    return {
        **task,
        "mode": out.mode,
        "rise": out.rise,
        "tilt_deg": out.tilt_deg,
        "pushed_m": out.pushed_m,
        "past_contact": out.past_contact,
        "jaws_touching": out.jaws_touching,
        "along_m": float(offset @ approach),  # (+: farther along the approach, away from the arm)
        "across_m": float(offset @ across),
    }


def tasks(records: Sequence[Mapping[str, Any]], n: int) -> list[dict[str, Any]]:
    """Every grasp (each run's start, unless refused, and the one taken) under each condition, the same `n` draws."""
    out = []
    for r in records:
        world = draws(np.random.default_rng(10_000 + r["seed"]), n, RefineSettings().object_sigma_m, False)
        grasps = [("taken", r["params"])] + ([] if r["start_refused"] else [("start", r["initial"])])
        for which, params in grasps:
            for condition in CONDITIONS:
                for k, (offset, draw) in enumerate(world):
                    out.append(
                        {
                            "placement": r["placement"],
                            "run": r["start"],
                            "grasp": which,
                            "params": params,
                            "side": r["fixed_side"],
                            "condition": condition,
                            "draw_index": k,
                            "offset": [0.0, 0.0] if condition == "physics" else [float(v) for v in offset],
                            "draw": {} if condition == "position" else dict(draw),
                        }
                    )
    return out


def _bins(
    values: np.ndarray, failed: np.ndarray, edges: Sequence[float], unit: float, name: str
) -> list[str]:
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        sel = (values >= lo) & (values < hi)
        if sel.any():
            rows.append(
                f"| {name} {lo / unit:+.0f} to {hi / unit:+.0f} | {sel.sum()} | {100 * failed[sel].mean():.0f} % |"
            )
    return rows


def summary(results: Sequence[Mapping[str, Any]]) -> str:
    """The tables (the module docstring)."""
    lines = [
        "## Lifted share per condition",
        "",
        "| grasp | position only | physics only | both |",
        "| --- | --- | --- | --- |",
    ]
    played = [r for r in results if r["mode"] != "refused"]
    for which in ("start", "taken"):
        cells = []
        for c in CONDITIONS:
            group = [r for r in played if r["grasp"] == which and r["condition"] == c]
            cells.append(
                f"{100 * np.mean([r['mode'] == 'lifted' for r in group]):.0f} % ({len(group)})"
                if group
                else "-"
            )
        lines.append(f"| {which} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Failures by kind",
        "",
        "| condition | " + " | ".join(MODES[1:]) + " |",
        "| --- |" + " --- |" * (len(MODES) - 1),
    ]
    for c in CONDITIONS:
        counts = Counter(r["mode"] for r in played if r["condition"] == c and r["mode"] != "lifted")
        lines.append(f"| {c} | " + " | ".join(str(counts.get(m, 0)) for m in MODES[1:]) + " |")
    both = [r for r in played if r["condition"] == "both"]
    failed = np.array([r["mode"] != "lifted" for r in both])
    along = np.array([r["along_m"] for r in both])
    across = np.array([abs(r["across_m"]) for r in both])
    friction = np.array([r["draw"]["FINGER_FRICTION"] / robot.FINGER_FRICTION for r in both])
    mass = np.array([r["draw"]["mass_scale"] for r in both])
    lines += [
        "",
        "## Under both: the failure rate by the draw",
        "",
        "| draw | n | failed |",
        "| --- | --- | --- |",
    ]
    edges_mm = [-0.04, -0.01, -0.005, 0.0, 0.005, 0.01, 0.04]
    lines += _bins(along, failed, edges_mm, 0.001, "along (mm)")
    lines += _bins(across, failed, [0.0, 0.003, 0.006, 0.01, 0.04], 0.001, "across, either side (mm)")
    lines += [
        f"| friction x {lo:.2f}-{hi:.2f} | {((friction >= lo) & (friction < hi)).sum()} | "
        f"{100 * failed[(friction >= lo) & (friction < hi)].mean():.0f} % |"
        for lo, hi in ((0.8, 0.9), (0.9, 1.0), (1.0, 1.1), (1.1, 1.26))
        if ((friction >= lo) & (friction < hi)).any()
    ]
    lines += [
        f"| mass x {lo:.2f}-{hi:.2f} | {((mass >= lo) & (mass < hi)).sum()} | "
        f"{100 * failed[(mass >= lo) & (mass < hi)].mean():.0f} % |"
        for lo, hi in ((0.8, 0.9), (0.9, 1.0), (1.0, 1.1), (1.1, 1.26))
        if ((mass >= lo) & (mass < hi)).any()
    ]
    return "\n".join(lines)


def film(
    model: XleRobotModel,
    examples: Sequence[Mapping[str, Any]],
    placements: Mapping[int, bank.Placement],
    config: bank.BankConfig,
    out: Path,
) -> None:
    """Each example played again and filmed, one after another: the scene from beside, the wrist camera."""
    from dataclasses import replace

    from .view import COLLISION, SITES, VISUAL

    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (2 * TILE[0], TILE[1]))
    settings = replace(config.settings(), camera_clearance_at_grasp=False)
    for ex in examples:
        p = placements[ex["placement"]]
        candidate, execution = plan_for(
            model, "right", p.primitive(), ex["params"], ex["side"], settings, START
        )
        state: dict[str, Any] = {}

        def frame(shadow: Shadow, label: str, ex: Mapping[str, Any] = ex, state: dict = state) -> None:
            m, d = shadow.scene.m, shadow.scene.d
            if "renderers" not in state:
                cam = mujoco.MjvCamera()
                cam.lookat[:] = shadow.object_position() + [0.0, 0.0, 0.02]
                cam.distance, cam.azimuth, cam.elevation = 0.45, 120.0, -20.0
                options = []
                for own in (False, True):
                    o = mujoco.MjvOption()
                    o.geomgroup[VISUAL], o.geomgroup[COLLISION] = True, False
                    o.geomgroup[robot.OWN_MODULE_GROUPS[VISUAL]] = not own
                    o.geomgroup[robot.OWN_MODULE_GROUPS[COLLISION]] = False
                    o.sitegroup[SITES] = False
                    options.append(o)
                state.update(
                    renderers=[mujoco.Renderer(m, TILE[1], TILE[0]) for _ in range(2)],
                    cam=cam,
                    options=options,
                )
            views = []
            for renderer, option, camera in zip(
                state["renderers"], state["options"], (state["cam"], robot.WRIST_CAMERA), strict=True
            ):
                renderer.update_scene(d, camera=camera, scene_option=option)
                views.append(cv2.resize(renderer.render(), TILE))
            title = [
                f"{ex['mode'].upper()}: placement {ex['placement']}, {ex['grasp']} grasp ({ex['run']} run)",
                f"bottle off {1000 * ex['along_m']:+.0f} mm along, {1000 * ex['across_m']:+.0f} mm across",
                f"friction x {ex['draw'].get('FINGER_FRICTION', robot.FINGER_FRICTION) / robot.FINGER_FRICTION:.2f}, "
                f"mass x {ex['draw'].get('mass_scale', 1.0):.2f}",
                label.split(" (")[0][:40],
            ]
            for k, text in enumerate(title):
                for color, width in (((0, 0, 0), 3), ((255, 255, 255), 1)):
                    cv2.putText(
                        views[0],
                        text,
                        (8, 22 + 22 * k),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        width,
                        cv2.LINE_AA,
                    )
            writer.write(cv2.cvtColor(np.hstack(views), cv2.COLOR_RGB2BGR))

        s = RefineSettings(table_z=p.bottom_root[2], mass_kg=p.mass_kg)
        outcome(
            model,
            "right",
            p.primitive(),
            execution,
            START,
            np.array(ex["offset"]),
            ex["draw"],
            s,
            on_tick=frame,
        )
        for renderer in state.get("renderers", []):
            renderer.close()
    writer.release()


def main() -> None:
    """Command line entry point; see the module docstring."""
    from lerobot.felix_jr.harness import api

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bank", type=Path)
    parser.add_argument("bench", type=Path, help="refine's benchmark (.jsonl)")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--draws", type=int, default=32)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--model", type=Path, default=api.MODEL)
    args = parser.parse_args()
    model = XleRobotModel.load(args.model)
    config, lines = bank.load(args.bank)
    placements = {
        line["placement"]["index"]: bank.Placement(
            **{k: tuple(v) if isinstance(v, list) else v for k, v in line["placement"].items()}
        )
        for line in lines
    }
    records = [json.loads(s) for s in args.bench.read_text().splitlines() if s.strip()]
    todo = tasks(records, args.draws)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with multiprocessing.get_context("spawn").Pool(args.workers, _init, (model, placements, config)) as pool:
        results = pool.map(_task, todo, chunksize=8)
    with (args.out_dir / "outcomes.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(summary(results))
    examples = []
    for mode in MODES[1:]:  # (the failures, the first of each kind under both)
        found = next((r for r in results if r["condition"] == "both" and r["mode"] == mode), None)
        if found is not None:
            examples.append(found)
    film(model, examples, placements, config, args.out_dir / "examples.mp4")
    print(f"filmed {[e['mode'] for e in examples]} to {args.out_dir / 'examples.mp4'}")


if __name__ == "__main__":
    main()
