"""Our scene in MuJoCo Warp on a GPU: does it step as CPU MuJoCo does, and how fast with many worlds (`docs/sim_deploy.md`).

  python -m lerobot.felix_jr.cloud.warp_check [--worlds 1 64 256 1024 4096] [--steps 200]
  (on a pod: `python -m lerobot.felix_jr.cloud.run --host ... -- python -m lerobot.felix_jr.cloud.warp_check`)

The scene: the right arm, a bottle (feno: 53 mm, 110 mm, 116 g) standing on the table. Each world from the arm at rest,
its servos driven to another pose (lift +25 deg, elbow -25 deg) for the steps. 1. **The same?** One world for 2 s on
Warp's device against CPU MuJoCo: the arm's joints and the bottle apart. 2. **How fast?** Per number of worlds: the
steps a second over all worlds (one `mjw.step` captured in a CUDA graph and replayed), the simulated seconds a wall
second, the GPU's memory. Writes `warp_check.json` into `$RUN_DIR` (the run script's) or the current folder.
A first look: the scene has no grasp in it (the fingers touch nothing); the grasps' rollouts come with the batched
rollout.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.felix_jr.calibration.joint_sim import REST
from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel
from lerobot.felix_jr.harness import api
from lerobot.felix_jr.sim2real import scene as sc
from lerobot.utils.import_utils import require_package

FENO = sc.Cylinder("object", -0.356, 0.132, 0.0265, 0.11, 0.116)
TABLE_Z = 0.7779
NCONMAX, NJMAX = 64, 400  # per world: a few contacts (the bottle on the table), 6 joint-friction rows, limits


def _start(scene: sc.Scene) -> tuple[np.ndarray, np.ndarray]:
    """The arm at rest (qpos) and its servos' targets at the other pose (ctrl)."""
    import mujoco

    m, d = scene.m, scene.d
    goal = {
        **REST,
        "shoulder_lift.pos": REST["shoulder_lift.pos"] + 25,
        "elbow_flex.pos": REST["elbow_flex.pos"] - 25,
        "gripper.pos": 30.0,
    }
    scene.set_pose(goal)
    ctrl = np.array([d.qpos[m.jnt_qposadr[m.actuator_trnid[a, 0]]] for a in range(m.nu)])
    scene.set_pose({**REST, "gripper.pos": 30.0})
    mujoco.mj_forward(m, d)
    return d.qpos.copy(), ctrl


def same(scene: sc.Scene, seconds: float = 2.0) -> dict[str, Any]:
    """One world on Warp against CPU MuJoCo for `seconds`: the arm's joints (deg) and the bottle (mm) apart, at the end."""
    import mujoco
    import mujoco_warp as mjw

    m = scene.m
    qpos, ctrl = _start(scene)
    cpu = mujoco.MjData(m)
    cpu.qpos[:], cpu.ctrl[:] = qpos, ctrl
    mujoco.mj_forward(m, cpu)
    model = mjw.put_model(m)
    data = mjw.put_data(m, cpu, nworld=1, nconmax=NCONMAX, njmax=NJMAX)
    steps = int(seconds / m.opt.timestep)
    for _ in range(steps):
        mujoco.mj_step(m, cpu)
        mjw.step(model, data)
    warp_q = data.qpos.numpy()[0]
    arm = [m.jnt_qposadr[m.actuator_trnid[a, 0]] for a in range(m.nu)]
    obj = m.jnt_qposadr[m.body_jntadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")]]
    return {
        "seconds": seconds,
        "arm_moved_deg": float(np.degrees(np.abs(cpu.qpos[arm] - qpos[arm])).max()),
        "arm_apart_deg": float(np.degrees(np.abs(cpu.qpos[arm] - warp_q[arm])).max()),
        "bottle_apart_mm": float(1000 * np.linalg.norm(cpu.qpos[obj : obj + 3] - warp_q[obj : obj + 3])),
    }


def speed(scene: sc.Scene, worlds: int, steps: int) -> dict[str, Any]:
    """`worlds` worlds stepped `steps` times (one step captured as a CUDA graph on a GPU): their throughput."""
    import mujoco
    import mujoco_warp as mjw
    import warp as wp

    m = scene.m
    qpos, ctrl = _start(scene)
    d = mujoco.MjData(m)
    d.qpos[:], d.ctrl[:] = qpos, ctrl
    mujoco.mj_forward(m, d)
    model = mjw.put_model(m)
    data = mjw.put_data(m, d, nworld=worlds, nconmax=NCONMAX, njmax=NJMAX)
    device = wp.get_device()
    graph = None
    if device.is_cuda:
        mjw.step(model, data)  # (compiles the kernels before the capture)
        with wp.ScopedCapture() as capture:
            mjw.step(model, data)
        graph = capture.graph
    for _ in range(10):  # warm up
        wp.capture_launch(graph) if graph is not None else mjw.step(model, data)
    wp.synchronize()
    started = time.perf_counter()
    for _ in range(steps):
        wp.capture_launch(graph) if graph is not None else mjw.step(model, data)
    wp.synchronize()
    wall = time.perf_counter() - started
    # the device's own count (the mempool's misses most: 1.2 GB read at 16384 worlds, out of 46 at 32768)
    used, total = (
        (device.total_memory - device.free_memory, device.total_memory) if device.is_cuda else (0, 0)
    )
    finite = bool(np.isfinite(data.qpos.numpy()).all())
    return {
        "worlds": worlds,
        "steps": steps,
        "wall_s": wall,
        "world_steps_per_s": worlds * steps / wall,
        "sim_s_per_wall_s": worlds * steps * m.opt.timestep / wall,
        "gpu_mem_used_gb": used / 1e9,
        "gpu_mem_total_gb": total / 1e9,
        "finite": finite,
    }


def main() -> None:
    """Command line entry point; see the module docstring."""
    require_package("mujoco_warp", extra="felix_jr_gpu")
    import warp as wp

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--worlds", type=int, nargs="+", default=[1, 64, 256, 1024, 4096])
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    wp.init()
    device = wp.get_device()
    model = XleRobotModel.load(api.MODEL)
    scene = sc.build(model, "right", [FENO], table_z=TABLE_Z)
    print(
        f"device {device} ({device.name}); scene: nq {scene.m.nq}, nv {scene.m.nv}, ngeom {scene.m.ngeom}",
        flush=True,
    )
    report: dict[str, Any] = {"device": str(device), "device_name": device.name}
    report["same"] = same(scene)
    s = report["same"]
    print(
        f"same: the arm moved {s['arm_moved_deg']:.1f} deg; Warp and CPU MuJoCo {s['arm_apart_deg']:.4f} deg apart, "
        f"the bottle {s['bottle_apart_mm']:.2f} mm",
        flush=True,
    )
    report["speed"] = []
    for worlds in args.worlds:
        r = speed(scene, worlds, args.steps)
        report["speed"].append(r)
        print(
            f"{worlds:6d} worlds: {r['world_steps_per_s']:12,.0f} steps/s ({r['sim_s_per_wall_s']:8,.0f}x real time), "
            f"{r['gpu_mem_used_gb']:.2f} GB, finite {r['finite']}",
            flush=True,
        )
    out = Path(os.environ.get("RUN_DIR", ".")) / "warp_check.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
