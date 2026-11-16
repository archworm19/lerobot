"""A joint session's wrist fingertip clicks replaced by the fingertip a gripper sweep measured. Runs on your computer.

  uv run --extra felix_jr --extra felix_jr_sim python -m lerobot.felix_jr.calibration.auto_tips <session> \
      --from-sweep <gripper_sweep folder> --out <new session folder> [--model <calibration.urdf>]

then fit the new session as any other (`joint_fit <new session> --start-from ... --out ...`).

The wrist camera and the fixed finger are both fixed to the gripper, so the fingertip is at one pixel in every wrist
frame, whatever the arm does: a session's wrist clicks all measure that one point. By eye they did it poorly: with the
jaws closed (as the session has them) the two fingers' tips lie side by side, and joint_right_full4's 27 clicks sat
in the notch between them, 2-3 px apart and ~3 mm short of the fixed finger's end; the calibration fitted to them put
the fixed finger off in the simulation, and the gripper sweep blamed the camera (7.5 px). Detecting the tip in the
session's own pictures failed too (the jaws closed, it took the moving finger's tip in some; a white board behind
them). A gripper sweep measures that pixel with the jaws open against the table, to ~1 px (`gripper_sweep.fixed_tip`,
its open frames); every wrist click is put there. The calibration's point is the wrist face's tip (the pinch point one
finger's thickness along the hinge, `XleRobotModel.wrist_face`), which on the XLeRobot's fingers is not quite the end
the camera sees (the tip's edge leans, `sim2real.robot.FINGER_FIXED_WRIST_TIP_M`): the click goes where the model puts
the wrist face's tip, offset from where it puts the end as the end was found from where the model puts it. The sweep
has to show the camera as the session had it (the same day, the mount untouched between). The head's clicks are as
they were.

`--out` gets the session's copy (`joint_session.json`, its pictures) and `auto_tips/`, each wrist click's picture with
the click as it was (magenta) and as measured (green). The session itself is left as it was.
"""

import argparse
import shutil
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from lerobot.felix_jr.calibration.joint_fit import MOVING, Look, TipClick, load_session, save_session
from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel, pinch_frame
from lerobot.felix_jr.harness import api
from lerobot.felix_jr.sim2real.view import undistort

from .gripper_sweep import FingerRenderer, SweepFrame, _project, fit_frame, fixed_tip, load_frames, tip_pixel

MAGENTA, GREEN = (255, 0, 255), (0, 220, 0)


def to_raw(model: XleRobotModel, arm: str, pixel: np.ndarray) -> np.ndarray:
    """An undistorted wrist pixel (the calibrated pinhole) back through the lens: the picture's own pixel."""
    camera = model.wrist_cameras[arm]
    ray = np.linalg.solve(camera.matrix(), np.r_[pixel, 1.0])
    return camera.project(ray[None])[0]


def sweep_clicks(
    model: XleRobotModel, arm: str, frames: Sequence[SweepFrame]
) -> list[tuple[SweepFrame, str, np.ndarray]]:
    """A gripper sweep's open frames as clicks: (frame, face, the wrist picture's own pixel) each.

    Per frame, the fixed fingertip (face "wrist": the wrist face's tip) and the moving one (`MOVING`: the moving
    finger's wrist face, the jaws open by the frame's `gripper.pos`), where found. Each moved from the end the camera
    sees to the point the calibration fits, by the model's own offset between them (the module docstring).
    """
    fingers = FingerRenderer(model, arm)
    face = model.wrist_face(arm)
    out: list[tuple[SweepFrame, str, np.ndarray]] = []
    for frame in frames:
        image = undistort(frame.color, model.wrist_cameras[arm])
        fit, seen = fit_frame(fingers, frame, image)
        if fit.fitted_deg is None:  # closed: the tips lie together
            continue
        q = model.joint_vector({arm: frame.pose})
        pinch = model.frame_pose(pinch_frame(arm), q)
        found = fixed_tip(model, arm, frame.pose, seen)
        if found is not None:
            end, end_model = found
            face_model = _project(model, arm, q, (pinch @ np.r_[face, 1.0])[:3])
            out.append((frame, "wrist", to_raw(model, arm, end - (end_model - face_model))))
        if seen.tip is not None:
            opening = model.opening_deg(arm, frame.pose["gripper.pos"])
            turned = model.moving_jaw(arm, opening) @ np.r_[face, 1.0]  # (pinch frame)
            face_model = _project(model, arm, q, (pinch @ turned)[:3])
            out.append(
                (
                    frame,
                    MOVING,
                    to_raw(model, arm, seen.tip - (tip_pixel(model, arm, q, opening) - face_model)),
                )
            )
    fingers.close()
    return out


def sweep_tip(model: XleRobotModel, arm: str, folder: Path) -> tuple[np.ndarray, np.ndarray]:
    """The click a gripper sweep's fixed fingertip makes (a wrist picture's own pixel), and how far apart it was found.

    Over the sweep's open frames (px; the module docstring).
    """
    fixed = [pixel for _, face, pixel in sweep_clicks(model, arm, load_frames(folder)) if face == "wrist"]
    if not fixed:
        raise SystemExit(f"{folder}: no open frame showed the fixed fingertip")
    return np.mean(fixed, axis=0), np.std(fixed, axis=0)


def main() -> None:
    """Command line entry point; see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session", type=Path)
    parser.add_argument(
        "--from-sweep", type=Path, required=True, help="a gripper_sweep folder of the session's arm"
    )
    parser.add_argument("--out", type=Path, required=True, help="the new session's folder")
    parser.add_argument(
        "--model", type=Path, default=api.MODEL, help="the calibration to find the fingers with"
    )
    args = parser.parse_args()
    if args.out.resolve() == args.session.resolve() or args.out.exists():
        raise SystemExit(f"--out must be a new folder: {args.out}")

    session = load_session(args.session)
    model = XleRobotModel.load(args.model)
    arm = session.arm
    frames = load_frames(args.from_sweep)
    if not frames or frames[0].arm != arm:
        raise SystemExit(f"{args.from_sweep}: not a gripper sweep of the {arm} arm")
    found = sweep_clicks(model, arm, frames)
    fixed = [pixel for _, face, pixel in found if face == "wrist"]
    if not fixed:
        raise SystemExit(f"{args.from_sweep}: no open frame showed the fixed fingertip")
    pixel, spread = np.mean(fixed, axis=0), np.std(fixed, axis=0)
    shutil.copytree(args.session / "images", args.out / "images")
    (args.out / "auto_tips").mkdir()
    clicks, moves = [], []
    for i, click in enumerate(session.clicks):
        if click.camera != arm or click.on_face == MOVING:
            if click.camera != arm:
                clicks.append(click)
            continue
        clicks.append(replace(click, pixel=pixel))
        moves.append(pixel - np.asarray(click.pixel, dtype=float))
        shown = cv2.imread(str(args.session / click.image)) if click.image.endswith(".png") else None
        if shown is not None:  # (a sweep's click has no picture of its own)
            cv2.circle(shown, tuple(int(round(v)) for v in click.pixel), 5, MAGENTA[::-1], 2)
            cv2.circle(shown, tuple(int(round(v)) for v in pixel), 5, GREEN[::-1], 2)
            cv2.imwrite(str(args.out / "auto_tips" / f"{i:03d}.png"), shown)
    # the moving finger's tips: its arc sets the hinge and the camera's turn about the roll axis (`joint_fit.MOVING`)
    ticks = next(
        iter(session.sightings)
    ).look.head_ticks  # (any: the wrist camera's picture does not depend on it)
    moving = [(frame, p) for frame, face, p in found if face == MOVING]
    for k, (frame, p) in enumerate(moving):
        look = Look({arm: dict(frame.pose)}, dict(ticks))
        clicks.append(TipClick(arm, arm, look, p, f"{args.from_sweep.name}/frame_{k:02d}", MOVING))
    save_session(replace(session, path=args.out, clicks=clicks))
    moves_ = np.array(moves).reshape(-1, 2)
    print(
        f"the sweep's fixed fingertip: {pixel.round(1)} px (a wrist picture's own pixel; found over its open frames "
        f"{spread.round(1)} px apart)"
    )
    print(
        f"{len(moves)} wrist clicks put there: they moved by {moves_.mean(axis=0).round(1)} px on average; the "
        f"clicks had been {np.std([c.pixel for c in session.clicks if c.camera == arm], axis=0).round(1)} px apart"
    )
    print(f"{len(moving)} moving fingertips added (the hinge and the camera's turn about the roll axis)")
    print(f"wrote {args.out} (the pictures with both: {args.out / 'auto_tips'})")


if __name__ == "__main__":
    main()
