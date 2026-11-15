"""Render one calibrated arm in the scene from the head camera and its wrist camera.

  uv run --extra felix_jr --extra felix_jr_sim python -m lerobot.felix_jr.sim2real.view \
      [--arm left] [--pose '{"shoulder_pan.pos": 12, ...}'] [--robot tcp://ikeabot.local:5555] [--no-fingers] \
      [--collision] [--out <png>]

Without `--robot`: the arm at `--pose` (default: ruler2's left attack pose over the bottle) with a bottle on the table;
each camera's render with the visual meshes, and with `--collision` also with what MuJoCo collides with. The jaws are
the XLeRobot's fingers, as on the robot (`robot.arm_spec`); `--no-fingers`: the SO-101's own. With `--robot`: reads the robot's joints, head and frames (nothing moves) and puts the real frames (the wrist's undistorted)
next to the renders at the same joints, and the two blended: where the calibrated sim and the robot disagree. The pinch
point is the magenta dot. Writes `<out>` (default `felix_jr/sim2real/scratch/renders/view.png`, gitignored).
"""

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from lerobot.felix_jr.calibration.xle_robot_model import WristCamera, XleRobotModel
from lerobot.felix_jr.harness import api
from lerobot.utils.import_utils import _mujoco_available

from .robot import FINGER_STL, HEAD_CAMERA, OWN_MODULE_GROUPS, WRIST_CAMERA
from .scene import Box, Cylinder, Scene, build

if TYPE_CHECKING or _mujoco_available:
    import mujoco

DEFAULT_OUT = Path(__file__).parent / "scratch" / "renders" / "view.png"
# ruler2 (2026-11-14): the left arm at its attack pose over the bottle, and the bottle as the head saw it
RULER2_POSE = {
    "shoulder_pan.pos": 12.0,
    "shoulder_lift.pos": 29.0,
    "elbow_flex.pos": -16.8,
    "wrist_flex.pos": 44.7,
    "wrist_roll.pos": -76.5,
    "gripper.pos": 43.7,
}
RULER2_OBJECTS: tuple[Cylinder | Box, ...] = (
    Cylinder("bottle", -0.455, -0.039, 0.028, 0.09, rgba=(0.95, 0.95, 0.95, 1.0)),
)
VISUAL, COLLISION, SITES = 2, 3, 4  # the SO-101's geom groups; our pinch site's group


def render(scene: Scene, camera: str, collision: bool = False) -> np.ndarray:
    """RGB (H, W, 3) from `camera` at its calibrated resolution: the visual meshes, or what collides.

    The wrist camera's module is left out of its own renders (the camera is inside it: `robot.OWN_MODULE_GROUPS`).
    """
    width, height = (int(v) for v in scene.m.cam_resolution[scene.m.camera(camera).id])
    option = mujoco.MjvOption()
    option.geomgroup[VISUAL] = not collision
    option.geomgroup[COLLISION] = collision
    own = camera == WRIST_CAMERA
    option.geomgroup[OWN_MODULE_GROUPS[VISUAL]] = not collision and not own
    option.geomgroup[OWN_MODULE_GROUPS[COLLISION]] = collision and not own
    option.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = (
        collision  # what MuJoCo collides with: each mesh's hull
    )
    option.sitegroup[SITES] = True
    with mujoco.Renderer(scene.m, height, width) as renderer:
        renderer.update_scene(scene.d, camera=camera, scene_option=option)
        return renderer.render().copy()


def undistort(image: np.ndarray, camera: WristCamera) -> np.ndarray:
    """A wrist frame through its lens's undistortion, onto the same pinhole (the renders' lens)."""
    k, coeffs = camera.matrix(), np.asarray(camera.coeffs)
    if camera.model == "fisheye":
        return cv2.fisheye.undistortImage(image, k, coeffs, Knew=k)
    return cv2.undistort(image, k, coeffs)


def _label(image: np.ndarray, text: str) -> np.ndarray:
    out = cv2.resize(image, (640, 480))
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    return out


def main() -> None:
    """The command line (the module docstring)."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--arm", default="left", choices=("left", "right"))
    parser.add_argument(
        "--pose", type=json.loads, default=None, help="LeRobot readings (JSON); default ruler2's"
    )
    parser.add_argument(
        "--robot", default=None, help="read the joints and frames from this server (nothing moves)"
    )
    parser.add_argument("--collision", action="store_true", help="also render what MuJoCo collides with")
    parser.add_argument(
        "--no-fingers", action="store_true", help="the SO-101's own jaws, not the XLeRobot's fingers"
    )
    parser.add_argument("--urdf", type=Path, default=api.MODEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    model = XleRobotModel.load(args.urdf)
    real: dict[str, np.ndarray] = {}
    head_ticks = None
    pose = dict(args.pose or RULER2_POSE)
    if args.robot:
        from lerobot.felix_jr.server.client import FelixClient

        client = FelixClient(args.robot)
        state = client.get_state()
        pose = {k: float(v) for k, v in state[f"{args.arm}_arm"].items()}
        head_ticks = {k: int(v["ticks"]) for k, v in state["head"].items()}
        real[HEAD_CAMERA] = client.capture().color
        frame = client.wrist_capture(args.arm).color
        real[WRIST_CAMERA] = undistort(frame, model.wrist_cameras[args.arm])
    objects = () if args.robot else RULER2_OBJECTS
    scene = build(
        model, args.arm, objects, head_ticks=head_ticks, fingers=None if args.no_fingers else FINGER_STL
    )
    scene.set_pose(pose)

    rows = []
    for camera in (HEAD_CAMERA, WRIST_CAMERA):
        sim = render(scene, camera)
        row = [_label(sim, f"{camera}: sim")]
        if args.collision:
            row.append(_label(render(scene, camera, collision=True), f"{camera}: sim, collision"))
        if camera in real:
            row = [_label(real[camera], f"{camera}: robot"), row[0]]
            row.append(_label(cv2.addWeighted(real[camera], 0.5, sim, 0.5, 0.0), "blend"))
        rows.append(np.hstack(row))
    sheet = np.vstack(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print(f"{args.arm} arm at {json.dumps({k: round(v, 1) for k, v in pose.items()})}; wrote {args.out}")


if __name__ == "__main__":
    main()
