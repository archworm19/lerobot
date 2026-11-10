"""Render the SO-101 moving to smooth random joint targets over a table: how the meshes and their collisions look.

  uv run python -m lerobot.felix_jr.sim.render_random [--out <path prefix>] [--seconds 12] [--seed 0]

Four views per frame: an overview and a close-up following the gripper, each with the visual meshes and with what
MuJoCo collides with (each mesh's convex hull). Writes `<out>.mp4` and `<out>_sheet.png` (six frames); default
`felix_jr/sim/scratch/renders/random`.
"""

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from lerobot.utils.import_utils import _mujoco_available, require_package

from .scene import Box, Cylinder, build

if TYPE_CHECKING or _mujoco_available:
    import mujoco

DEFAULT_OUT = Path(__file__).parent / "scratch" / "renders" / "random"
VISUAL, COLLISION = 2, 3  # the SO-101 model's geom groups
SIZE, FPS, SEGMENT_S = (480, 360), 30, 1.5


def render_random(out: Path = DEFAULT_OUT, seconds: float = 12.0, seed: int = 0) -> tuple[Path, Path]:
    """Simulate and render (the module docstring); the video's and the contact sheet's paths."""
    require_package("mujoco", extra="felix_jr_sim")
    m = build((Cylinder("bottle", 0.25, 0.0, 0.018, 0.1), Box("cube", 0.2, 0.08, (0.02, 0.02, 0.02))))
    d = mujoco.MjData(m)
    renderer = mujoco.Renderer(m, SIZE[1], SIZE[0])
    rng = np.random.default_rng(seed)
    low, high = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    mid, half = (low + high) / 2, (high - low) / 2
    lift = [m.actuator(i).name for i in range(m.nu)].index("shoulder_lift")
    gripper = m.body("gripper").id

    def target() -> np.ndarray:
        t = mid + 0.7 * half * rng.uniform(-1, 1, m.nu)
        t[lift] = np.clip(t[lift], -1.2, 0.6)  # mostly up: the arm moves over the table
        return t

    def view(lookat: np.ndarray, azimuth: float, elevation: float, distance: float, group: int) -> np.ndarray:
        camera = mujoco.MjvCamera()
        camera.lookat[:], camera.azimuth, camera.elevation, camera.distance = (
            lookat,
            azimuth,
            elevation,
            distance,
        )
        option = mujoco.MjvOption()
        option.geomgroup[:] = 0
        option.geomgroup[0] = option.geomgroup[group] = 1  # the table and the objects, and the arm's group
        option.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = group == COLLISION
        renderer.update_scene(d, camera=camera, scene_option=option)
        return renderer.render()

    frames = []
    before, after = target(), target()
    steps = int(round(1 / FPS / m.opt.timestep))
    for f in range(int(seconds * FPS)):
        t = f / FPS
        if f and (t % SEGMENT_S) < 1 / FPS:
            before, after = after, target()
        a = (t % SEGMENT_S) / SEGMENT_S
        d.ctrl[:] = before + (after - before) * (1 - np.cos(np.pi * a)) / 2
        for _ in range(steps):
            mujoco.mj_step(m, d)
        grip = d.xpos[gripper].copy()
        tiles = []
        for label, args in (
            ("visual", ([0.12, 0, 0.12], 150, -25, 0.75, VISUAL)),
            ("collision", ([0.12, 0, 0.12], 150, -25, 0.75, COLLISION)),
            ("gripper, visual", (grip, 120 + 20 * np.sin(t / 2), -30, 0.25, VISUAL)),
            ("gripper, collision", (grip, 120 + 20 * np.sin(t / 2), -30, 0.25, COLLISION)),
        ):
            tile = view(*args)
            cv2.putText(tile, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            tiles.append(tile)
        frames.append(np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]))
    out.parent.mkdir(parents=True, exist_ok=True)
    video, sheet = out.with_suffix(".mp4"), out.parent / f"{out.name}_sheet.png"
    size = (frames[0].shape[1], frames[0].shape[0])
    for codec in ("avc1", "mp4v"):  # avc1 plays in QuickTime; not every OpenCV build has it
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*codec), FPS, size)
        if writer.isOpened():
            break
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    picks = [frames[i] for i in np.linspace(0, len(frames) - 1, 6).astype(int)]
    small = [cv2.resize(p, (p.shape[1] // 2, p.shape[0] // 2)) for p in picks]
    cv2.imwrite(
        str(sheet), cv2.cvtColor(np.vstack([np.hstack(small[:3]), np.hstack(small[3:])]), cv2.COLOR_RGB2BGR)
    )
    return video, sheet


def main() -> None:
    """Command line entry point; see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="the output's path prefix")
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    video, sheet = render_random(args.out, args.seconds, args.seed)
    print(f"Wrote {video} and {sheet}")


if __name__ == "__main__":
    main()
