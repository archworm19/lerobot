"""The ways to model the Fin Ray fingers, compared on the real grip trials: fit, outcome, speed, and a video.

  uv run --extra felix_jr --extra felix_jr_sim python -m lerobot.felix_jr.sim2real.finger_models <results.json>
      [--video <out.mp4>] [--no-fit]

Each of `robot.FINGER_MODELS` ("sprung", the default; "rigid"; "soft"; "chain"), its softness fitted to the 45 mm
bottle's stalls (`FIT_TRIAL`) over a small grid (`GRIDS`: "rigid" has nothing to fit), then replayed on every trial in
`TRIALS` (`grip_sim.replay`): the stalls' and loads' RMS against the real ones, the lift (held or not, as the real one),
and the speed (simulated steps a second, and how much faster than real time). `--video`: the fit trial replayed by
every model, side by side, the fingers' pads shown (red).

The models (`robot.py` has what set each value): "sprung", one hinge per finger at its base on a spring; "rigid", no
give; "soft", rigid pads with a soft contact (option 1 of the common Fin Ray models: the give as penetration); "chain",
the face's pads hinged base to tip, sprung (option 2 cut down: it can wrap).
"""

import argparse
import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lerobot.felix_jr.calibration.gripper_sweep import table_height
from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel
from lerobot.utils.import_utils import _mujoco_available, require_package

from . import grip_sim, robot

if _mujoco_available:
    import mujoco

FIT_TRIAL = "pyri6"  # the 45 mm bottle, the stiffer one (`robot.FINGER_STIFFNESS_NM_PER_RAD`)
TRIALS = ("pyri6", "feno2")
# per model, the settings tried (`robot` constants, as {name: value}); the first is today's
GRIDS: dict[str, list[dict[str, Any]]] = {
    "sprung": [{"FINGER_STIFFNESS_NM_PER_RAD": k} for k in (1.2, 0.8, 1.6)],
    "rigid": [{}],
    "soft": [
        {"FINGER_SOFT_SOLREF": (tc, 1.0), "FINGER_SOFT_SOLIMP": (d0, 0.95, w, 0.5, 2.0)}
        for tc in (0.02, 0.1)
        for d0 in (0.5, 0.1)
        for w in (0.01, 0.03)
    ],
    "chain": [{"FINGER_CHAIN_STIFFNESS_NM_PER_RAD": k} for k in (2.0, 1.2, 1.6, 2.5)],
}
TILE = (320, 240)


@contextmanager
def settings(values: dict[str, Any]) -> Iterator[None]:
    """`robot`'s constants set to `values` for the block (the scene reads them as it builds)."""
    old = {k: getattr(robot, k) for k in values}
    for k, v in values.items():
        setattr(robot, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(robot, k, v)


def scores(result: dict[str, Any]) -> dict[str, float]:
    """A replay's stall and load RMS against the real trial's closes."""
    closes = result["closes"]
    return {
        "stall_rms": float(np.sqrt(np.mean([(c["sim_pos"] - c["real_pos"]) ** 2 for c in closes]))),
        "load_rms": float(np.sqrt(np.mean([(c["sim_load_pct"] - c["real_load_pct"]) ** 2 for c in closes]))),
    }


def compare(
    model: XleRobotModel, trials: dict[str, dict], table_z: float, fit: bool = True
) -> list[dict[str, Any]]:
    """Per finger model: its best setting on `FIT_TRIAL` (stall RMS), then every trial's scores, lift and speed."""
    rows = []
    for name in robot.FINGER_MODELS:
        grid = GRIDS[name] if fit else GRIDS[name][:1]
        best: tuple[float, dict[str, Any]] | None = None
        for values in grid:
            with settings(values):
                rms = scores(grip_sim.replay(model, "right", trials[FIT_TRIAL], table_z, finger_model=name))[
                    "stall_rms"
                ]
            if best is None or rms < best[0]:
                best = (rms, values)
        assert best is not None
        row: dict[str, Any] = {"model": name, "setting": best[1], "trials": {}}
        with settings(best[1]):
            for label in TRIALS:
                clock = {"t": 0.0, "dt": 0.002}

                def note(scene: Any, phase: str, clock: dict = clock) -> None:  # (twice a second: no cost)
                    clock["t"], clock["dt"] = float(scene.d.time), float(scene.m.opt.timestep)

                started = time.perf_counter()
                result = grip_sim.replay(
                    model,
                    "right",
                    trials[label],
                    table_z,
                    finger_model=name,
                    on_frame=note,
                    frame_period_s=0.5,
                )
                wall = time.perf_counter() - started  # (the scene's build included, as a reset would)
                sim_s = clock["t"]
                steps = {"n": sim_s / clock["dt"]}
                row["trials"][label] = scores(result) | {
                    "lift": result["lift"].get("sim"),
                    "real_lift": result["lift"].get("real"),
                    "steps_per_s": steps["n"] / wall,
                    "realtime_x": sim_s / wall,
                }
        rows.append(row)
    return rows


def table(rows: list[dict[str, Any]]) -> str:
    """The comparison, one line a model and trial."""
    out = [
        "| model | setting | trial | stall RMS (units) | load RMS (%) | lift (real) | steps/s | x real time |"
    ]
    out.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        setting = (
            ", ".join(f"{k.removeprefix('FINGER_').lower()}={v}" for k, v in row["setting"].items()) or "-"
        )
        for label, t in row["trials"].items():
            match = "ok" if (t["lift"] == "held") == (t["real_lift"] == "held") else "WRONG"
            out.append(
                f"| {row['model']} | {setting} | {label} | {t['stall_rms']:.1f} | {t['load_rms']:.1f} | "
                f"{t['lift']} ({t['real_lift']}: {match}) | {t['steps_per_s']:.0f} | {t['realtime_x']:.1f} |"
            )
    return "\n".join(out)


def video(model: XleRobotModel, trial: dict, table_z: float, rows: list[dict[str, Any]], out: Path) -> None:
    """`trial` replayed by every model (its fitted setting), side by side, from in front of the gripper."""
    require_package("mujoco", extra="felix_jr_sim")
    frames: dict[str, list[np.ndarray]] = {}
    for row in rows:
        name, tiles = row["model"], list[np.ndarray]()
        state: dict[str, Any] = {}

        def grab(scene: Any, phase: str, state: dict = state, tiles: list = tiles, name: str = name) -> None:
            m, d = scene.m, scene.d
            if "r" not in state:
                state["r"] = mujoco.Renderer(m, TILE[1], TILE[0])
                cam = mujoco.MjvCamera()
                cam.lookat[:] = d.xpos[m.body("bottle").id]
                cam.distance, cam.azimuth, cam.elevation = 0.30, -20.0, -30.0
                opt = mujoco.MjvOption()
                opt.geomgroup[:] = 0
                opt.geomgroup[0] = opt.geomgroup[robot.VISUAL_GROUP] = opt.geomgroup[
                    robot.COLLISION_GROUP
                ] = 1
                state["cam"], state["opt"] = cam, opt
            state["r"].update_scene(d, camera=state["cam"], scene_option=state["opt"])
            tile = state["r"].render().copy()
            for text, y in ((name, 20), (phase, TILE[1] - 8)):
                cv2.putText(tile, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(
                    tile, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
                )
            tiles.append(tile)

        with settings(row["setting"]):
            grip_sim.replay(model, "right", trial, table_z, finger_model=name, on_frame=grab)
        frames[name] = tiles
    n = min(len(t) for t in frames.values())
    names = [row["model"] for row in rows]
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 30, (2 * TILE[0], 2 * TILE[1]))
    for i in range(n):
        tiles = [frames[name][i] for name in names] + [np.zeros_like(frames[names[0]][0])] * (4 - len(names))
        grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4])])
        writer.write(cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    writer.release()


def main() -> None:
    """Command line entry point; see the module docstring."""
    from lerobot.felix_jr.harness import api

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path, help="a grip_test results.json")
    parser.add_argument("--model", type=Path, default=api.MODEL)
    parser.add_argument(
        "--video", type=Path, default=None, help="the fit trial, every model side by side (mp4)"
    )
    parser.add_argument("--no-fit", action="store_true", help="each model's first setting only")
    args = parser.parse_args()
    model = XleRobotModel.load(args.model)
    trials = {t["bottle"]["label"]: t for t in json.loads(args.results.read_text())}
    table_z = table_height(model, "right")
    rows = compare(model, trials, table_z, fit=not args.no_fit)
    print(table(rows))
    if args.video is not None:
        video(model, trials[FIT_TRIAL], table_z, rows, args.video)
        print(f"wrote {args.video}")


if __name__ == "__main__":
    main()
