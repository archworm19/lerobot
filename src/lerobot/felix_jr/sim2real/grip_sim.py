"""A real grip trial (`calibration.grip_test`) replayed in the simulator: where the jaw stalls, and whether it lifts.

  uv run --extra felix_jr --extra felix_jr_sim python -m lerobot.felix_jr.sim2real.grip_sim <results.json>
      [--trial pyri6 ...] [--kp 8.8] [--force-nm 0.67] [--friction 0.6] [--model <calibration.urdf>]

The step before fitting the simulator's grip (`docs/sim2real_grasp.md`, "Next" 3): each trial's bottle (a `Cylinder`
of its diameter, height and mass) stood where the trial's outline put it (`GripTest.bottle_outline`'s center, at the
depth used), the arm at the trial's grip pose, the jaws open; then the trial's closes, each `CLOSE_S`, the hold at its
target (`HOLD_S` here), and the lift. Printed per close, the simulated against the real stall (`gripper.pos`) and load
(% of the servo's stall torque, `STALL_NM`); and whether the bottle came up with the jaws.

The gripper's servo, as measured (2026-11-17): its position gain `--kp` (N m/rad at the jaw: ~4.6 % load a degree, as
`tuning_notes.md` 3.6 has it at P 16; the SO-101 model's 998 is ~100 times stiffer), its torque capped at `--force-nm`
(the overload protection's ~20 %, `grip_test.SUSTAINED_LOAD_PCT`, of `STALL_NM`), the fingers' and the bottle's
friction `--friction` (a guess). The fingers collide as their convex hulls, rigid (`robot._add_fingers`): the real
fingers' give (the stall moving ~13 units from first touch to 15 past contact) is not in the simulator yet.
"""

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.felix_jr.calibration.grip_test import Bottle, grip_pose
from lerobot.felix_jr.calibration.gripper_sweep import table_height
from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel, pinch_frame
from lerobot.utils.import_utils import _mujoco_available, require_package

from .robot import FINGER_FRICTION, GRIPPER, GRIPPER_FORCE_NM, GRIPPER_KP_NM_PER_RAD, GRIPPER_STALL_NM
from .scene import Cylinder, Scene, build

if _mujoco_available:
    import mujoco

# the gripper's servo and fingers as measured: `robot.GRIPPER_*`, `robot.FINGER_FRICTION` (with what set them)
STALL_NM = GRIPPER_STALL_NM  # the load's 100 %
KP_NM_PER_RAD = GRIPPER_KP_NM_PER_RAD
FORCE_NM = GRIPPER_FORCE_NM
CLOSE_S = 2.0
HOLD_S = 5.0
OPEN_S = 0.5
LIFT_S = 1.5  # the lift's move
LIFT_HOLD_S = 3.0
LIFT_M = 0.04
HELD_FRACTION = 0.75  # the bottle up by this much of the lift: held


def _trial_bottle(trial: dict[str, Any]) -> Bottle:
    b = trial["bottle"]
    return Bottle(
        b["label"], b["diameter_mm"], b["height_mm"], b["mass_g"], trial.get("depth_mm_actual", b["depth_mm"])
    )


def replay(
    model: XleRobotModel,
    arm: str,
    trial: dict[str, Any],
    table_z: float,
    kp: float = KP_NM_PER_RAD,
    force_nm: float = FORCE_NM,
    friction: float = FINGER_FRICTION,
    on_frame: "Callable[[Scene, str], None] | None" = None,
    frame_period_s: float = 1 / 30,
) -> dict[str, Any]:
    """One trial in the simulator, against the real one.

    Per close `{target, sim_pos, sim_load_pct, real_pos, real_load_pct}`, the hold's, and the lift (`held`, how far
    the bottle rose).
    """
    require_package("mujoco", extra="felix_jr_sim")
    bottle = _trial_bottle(trial)
    pose = {**trial["pose"], "gripper.pos": trial["open_pos"]}
    # where the bottle stands: the outline's center (`GripTest.bottle_outline`), at the trial's grip pose
    q = model.joint_vector({arm: pose})
    pinch = model.frame_pose(pinch_frame(arm), q)
    center = (
        pinch
        @ np.r_[
            model.grasp_offset(arm, bottle.diameter_mm / 1000.0) - [0.0, 0.0, bottle.depth_mm / 1000.0], 1.0
        ]
    )
    cylinder = Cylinder(
        "bottle",
        float(center[0]),
        float(center[1]),
        bottle.diameter_mm / 2000.0,
        bottle.height_mm / 1000.0,
        bottle.mass_g / 1000.0,
    )
    scene = build(model, arm, [cylinder], table_z=table_z)
    m, d = scene.m, scene.d
    gripper = m.actuator(GRIPPER).id
    m.actuator_gainprm[gripper, 0] = kp
    m.actuator_biasprm[gripper, 1] = -kp
    m.actuator_forcerange[gripper] = [-force_nm, force_nm]
    for i in range(m.ngeom):  # the fingers' pads (`robot.finger_pads`)
        if m.geom(i).name.startswith("finger_") and "_pad" in m.geom(i).name:
            m.geom_friction[i, 0] = friction
    bottle_geom = next(i for i in range(m.ngeom) if m.geom_bodyid[i] == m.body("bottle").id)
    m.geom_friction[bottle_geom, 0] = friction  # (MuJoCo takes the larger of the two geoms')
    scene.set_pose(pose)

    phase = "open"
    every = max(1, int(round(frame_period_s / m.opt.timestep)))

    def step() -> None:
        mujoco.mj_step(m, d)
        if on_frame is not None and int(round(d.time / m.opt.timestep)) % every == 0:
            on_frame(scene, phase)

    def run(seconds: float) -> None:
        for _ in range(int(seconds / m.opt.timestep)):
            step()

    def close_to(target: float, seconds: float) -> dict[str, float]:
        d.ctrl[gripper] = np.radians(model.opening_deg(arm, target))
        run(seconds)
        angle = float(np.degrees(d.qpos[m.joint(GRIPPER).qposadr[0]]))
        return {
            "target": target,
            "sim_pos": model.gripper_pos_for(arm, angle),
            "sim_load_pct": 100.0 * abs(float(d.actuator_force[gripper])) / STALL_NM,
        }

    run(OPEN_S)
    closes = []
    for real in trial["closes"]:
        phase = f"close to {real['target']:.1f} ({real['past_contact']:g} past contact)"
        out = close_to(real["target"], CLOSE_S)
        closes.append(out | {"real_pos": real["stall"]["pos"], "real_load_pct": real["stall"]["load_pct"]})
    hold = None
    if "hold" in trial:
        h = trial["hold"]
        phase = f"hold at {h['target']:.1f}"
        hold = close_to(h["target"], HOLD_S) | {
            "real_pos": h["last"]["pos"],
            "real_load_pct": h["last"]["load_pct"],
        }
    lift: dict[str, Any] = {"real": trial.get("lift", {}).get("result")}
    if hold is not None:
        found = grip_pose(
            model, arm, {}, table_z, trial["tip_height_m"], LIFT_M, (trial["tilt_deg"],), (trial["ahead_m"],)
        )
        if found is not None:
            start = np.array(d.ctrl).copy()
            up = {**found[0]}
            z0 = float(d.xpos[m.body("bottle").id][2])
            steps = int(LIFT_S / m.opt.timestep)
            phase = "lift"
            for k in range(steps):
                f = (k + 1) / steps
                for joint, angle in up.items():
                    name = joint.removesuffix(".pos")
                    a = m.actuator(name).id
                    d.ctrl[a] = start[a] + f * (np.radians(angle) - start[a])
                step()
            run(LIFT_HOLD_S)
            rose = float(d.xpos[m.body("bottle").id][2]) - z0
            lift |= {"rose_m": rose, "sim": "held" if rose > HELD_FRACTION * LIFT_M else "dropped"}
    return {"bottle": bottle.label, "closes": closes, "hold": hold, "lift": lift}


def report(result: dict[str, Any]) -> str:
    """One trial's replay against the real one, a few lines."""
    lines = [f"{result['bottle']}:"]
    for c in result["closes"] + (
        [result["hold"] | {"target": result["hold"]["target"]}] if result["hold"] else []
    ):
        lines.append(
            f"  to {c['target']:5.1f}: stall sim {c['sim_pos']:5.1f} real {c['real_pos']:5.1f}; "
            f"load sim {c['sim_load_pct']:5.1f} real {c['real_load_pct']:5.1f} %"
        )
    lift = result["lift"]
    if "sim" in lift:
        lines.append(
            f"  lift: sim {lift['sim']} (the bottle up {100 * lift['rose_m']:.1f} cm), real {lift['real']}"
        )
    return "\n".join(lines)


def main() -> None:
    """Command line entry point; see the module docstring."""
    from lerobot.felix_jr.harness import api

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path, help="a grip_test results.json")
    parser.add_argument("--trial", nargs="*", default=None, help="the trials' labels (default: all)")
    parser.add_argument("--arm", default="right")
    parser.add_argument("--model", type=Path, default=api.MODEL)
    parser.add_argument("--kp", type=float, default=KP_NM_PER_RAD)
    parser.add_argument("--force-nm", type=float, default=FORCE_NM)
    parser.add_argument("--friction", type=float, default=FINGER_FRICTION)
    args = parser.parse_args()
    model = XleRobotModel.load(args.model)
    trials = json.loads(args.results.read_text())
    for trial in trials:
        if args.trial and trial["bottle"]["label"] not in args.trial:
            continue
        if "tilt_deg" not in trial:  # (the first trials, before the pose's tilt was recorded)
            continue
        result = replay(
            model, args.arm, trial, table_height(model, args.arm), args.kp, args.force_nm, args.friction
        )
        print(report(result))


if __name__ == "__main__":
    main()
