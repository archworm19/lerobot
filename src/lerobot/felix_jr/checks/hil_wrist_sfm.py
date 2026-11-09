"""Hardware in the loop: the head camera's depth against the wrist's structure from motion, a checkerboard the referee.

  uv run python -m lerobot.felix_jr.checks.hil_wrist_sfm record <name> [--arm right] [--rolls ...] [--flexes ...]
  uv run python -m lerobot.felix_jr.checks.hil_wrist_sfm analyze <name> [--methods light,torch]

`docs/object_grasping.md` section 6.3. The wrist calibration's checkerboard (8 x 6 inner corners, 25 mm) lies flat on
the table where that calibration's start pose looks (~22 cm in front of the arm's base), the bottle and a box beside
it. The board's corners give each wrist frame's true pose (PnP) and the head camera's pose relative to it, so nothing
is scored through the arm's model or either camera's calibration on the robot.

- `record` (moves the robot, each move asked first): a head capture aimed at the board with the arms at rest; the arm
  to the start pose; a wrist sweep (`wrist_scan.sweep`: the wrist only, and back). Saved in
  `checks/scratch/hil/<name>/` (gitignored): the head capture (`Capture.save`), each wrist frame and its joints.
- `analyze` (no robot): the board found in every frame; the board blanked out of the wrist frames (the methods work
  from the table and the objects, as in a session); each method's frames' motion scored against the board's (the
  turns, the moves' directions, the one scale); the table plane in the board's frame from the head's depth and from
  each method, and the jaws' height over it.
"""

import argparse
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from lerobot.felix_jr.calibration.wrist_camera import (
    START_AHEAD_M,
    START_HEIGHTS_M,
    START_PITCHES_DEG,
    Board,
    board_pose,
    find_board,
)
from lerobot.felix_jr.calibration.xle_robot_model import HEAD_JOINTS, XleRobotModel
from lerobot.felix_jr.harness import api
from lerobot.felix_jr.harness.objects import Capture
from lerobot.felix_jr.harness.wrist import WristView
from lerobot.felix_jr.harness.wrist_scan import SWEEP_FLEXES_DEG, SWEEP_ROLLS_DEG, scan_views, sweep

BOARD = Board(8, 6, 0.025)
RUNS = Path(__file__).parent / "scratch" / "hil"
BLANK_MARGIN_PX = 25  # the board blanked this far beyond its outer corners (its outer squares and border)


def _ask(question: str) -> bool:
    return input(f"{question} [y/N] ").strip().lower() == "y"


# ---- record ----------------------------------------------------------------------------------------------------


def record(
    name: str,
    arm: str,
    rolls: list[float],
    flexes: list[float],
    skip_head: bool = False,
    offset: tuple[float, float] = (0.08, 0.0),
    heights: Sequence[float] = START_HEIGHTS_M,
) -> None:
    """The head capture, the arm to the start pose over the board, the wrist sweep: all saved."""
    from lerobot.felix_jr.harness import tools

    out = RUNS / name
    out.mkdir(parents=True, exist_ok=True)
    h = api.Harness()
    model, client = h.model, h.client
    base = model.frame_pose(f"{arm}_base_link", model.joint_vector())[:3, 3]
    if skip_head:
        head = Capture.load(out / "head", model)
        print("1. the head capture taken before")
    else:
        under = np.array([base[0] - START_AHEAD_M, base[1], base[2] - 0.06])  # where the board should be
        print(f"1. the head looks at the board (around {np.round(under * 100, 1)} cm); the arms stay at rest")
        head = h.call("look_at", under)  # asks first
        head.save(out / "head")
    found = find_board(head.scene.color, BOARD)
    print(
        f"   the board {'found' if found is not None else 'NOT found: move it into view'} in the head capture"
    )
    if found is None:
        return
    # the board's center on the table (root frame, through the head's depth): the start pose's tip goes over it
    # the median of the corners' points: one pixel's depth can be missing (jnt2: the center's was)
    corners_xyz = tools.locate_pixels(tools.full_view(head), found).xyz
    corners_xyz = corners_xyz[np.isfinite(corners_xyz).all(axis=1)]
    center = np.median(corners_xyz, axis=0) if len(corners_xyz) else np.full(3, np.nan)
    if not np.isfinite(center).all():
        print("   no depth at the board's center in the head capture")
        return
    print(f"   the board's center at {np.round(center * 100, 1)} cm")
    print("2. the arm to the wrist calibration's start pose (gripper tilted down, ~18 cm over the table)")
    state = client.get_state()
    current = {k: float(v) for k, v in state[f"{arm}_arm"].items()}
    capture = SimpleNamespace(
        arms={arm: current}, head_ticks={n: int(state["head"][n]["ticks"]) for n in HEAD_JOINTS}
    )
    from lerobot.felix_jr.calibration.merge_urdf import SO101_JOINTS
    from lerobot.felix_jr.harness.reach import ArmIK, solve_pose

    def margins(joints: dict[str, float]) -> dict[str, float]:
        ik = ArmIK(model, arm, {arm: {**joints, "gripper.pos": 0.0}}, None, 0.0, 0.0)
        x = ik.radians(joints)
        room = np.degrees(np.minimum(x - ik.lower, ik.upper - x))
        return dict(zip(SO101_JOINTS, room, strict=True))

    # room for the sweep: every joint 15 deg from its bounds, the wrist's two the sweep's size more
    need = dict.fromkeys(SO101_JOINTS, 15.0)
    need["wrist_flex"] = max(15.0, max((abs(f) for f in flexes), default=0.0) + 5.0)
    need["wrist_roll"] = max(15.0, max((abs(r) for r in rolls), default=0.0) + 5.0)
    candidates = []
    for up in heights:
        for pitch in START_PITCHES_DEG:
            # the tip `offset` from over the board's center (toward the robot by default: the closed jaws cover the
            # bottom middle of the picture, so the board goes in its upper part; hil1: the board under the jaws)
            position = np.array([center[0] + offset[0], center[1] + offset[1], base[2] + up])
            poses = solve_pose(model, arm, capture, position, pitch, None, 0.0, init=current)  # type: ignore[arg-type]
            if poses.ok[0]:
                room = margins(dict(poses.joints[0]))
                short = min(room[j] - need[j] for j in SO101_JOINTS)
                candidates.append((short, position, pitch, dict(poses.joints[0]), room))
                if short >= 0:
                    break
        if candidates and candidates[-1][0] >= 0:
            break
    target = None
    if candidates:
        short, position, pitch, joints, room = max(candidates, key=lambda c: c[0])
        target = joints | {"gripper.pos": current.get("gripper.pos", 0.0)}
        print(
            f"   tip at {np.round(position * 100, 1)} cm, pitch {pitch:.0f}: {np.round(list(joints.values()), 1)}"
        )
        print("   room (deg): " + ", ".join(f"{j} {room[j]:.0f} (needs {need[j]:.0f})" for j in SO101_JOINTS))
        if short < 0:
            print("   NOT enough room for the sweep at any start pose tried: the closest shown")
    if target is None or not _ask("   move the arm there?"):
        return
    with client.session():
        client.move_joints({f"{arm}_arm": target}, max_vel=20.0, tol=1.5)
        client.keep_holding()
    print(f"3. the wrist sweep: roll {rolls}, flex {flexes}, and back (the wrist only)")
    if not _ask("   sweep?"):
        return
    views, labels = sweep(client, model, arm, rolls, flexes)
    frames = []
    for i, (view, label) in enumerate(zip(views, labels, strict=True)):
        cv2.imwrite(str(out / f"wrist_{i:02d}.png"), cv2.cvtColor(view.color, cv2.COLOR_RGB2BGR))
        frames.append({"image": f"wrist_{i:02d}.png", "label": label, "arm_pose": dict(view.arm_pose)})
    (out / "wrist.json").write_text(json.dumps({"arm": arm, "frames": frames}, indent=1) + "\n")
    seen = sum(find_board(v.color, BOARD) is not None for v in views)
    print(f"   {len(views)} frames saved in {out}; the board found in {seen} of them")
    print("   done: the arm holds where it is (send it home when finished), then `analyze`")


# ---- the head and the wrist camera together (the scale: the head's depth is metric) ------------------------------


JOINT_ROLLS_DEG = (0.0, -30.0, -15.0)  # +30 took the board out of the wrist camera's view (jnt1)


def joint_record(
    name: str, arm: str, skip_head: bool = False, heights: Sequence[float] = START_HEIGHTS_M
) -> None:
    """The arm over the board (as `record`), then a head capture and a wrist frame together at each wrist roll.

    The rolls: `JOINT_ROLLS_DEG`; the head stays still, so its depth measures where the wrist camera really is and how
    far it moves.
    """
    from lerobot.felix_jr.harness.wrist import move_closely, wrist_view

    out = RUNS / name
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "wrist.json").exists():
        record(
            name, arm, [], [], skip_head, heights=heights
        )  # the head's look, the arm to the start pose (no sweep)
    h = api.Harness()
    model, client = h.model, h.client
    key = f"{arm}_arm"
    home = {k: float(v) for k, v in client.get_state()[key].items() if k != "gripper.pos"}
    if not _ask(f"turn the wrist through {JOINT_ROLLS_DEG} deg, a head capture and a wrist frame at each?"):
        return
    frames = []
    with client.session():
        for k, roll in enumerate(JOINT_ROLLS_DEG):
            status, _ = move_closely(
                client, key, {**home, "wrist_roll.pos": home["wrist_roll.pos"] + roll}, 15.0
            )
            view = wrist_view(model, client.wrist_capture(arm, after=status.get("finished_at")))
            head = h.call("capture")
            head.save(out / f"joint_{k}_head")
            image = f"joint_{k}_wrist.png"
            cv2.imwrite(str(out / image), cv2.cvtColor(view.color, cv2.COLOR_RGB2BGR))
            found = find_board(view.color, BOARD), find_board(head.scene.color, BOARD)
            frames.append(
                {"roll": roll, "image": image, "arm_pose": dict(view.arm_pose), "head": f"joint_{k}_head"}
            )
            print(
                f"   roll {roll:+.0f}: the board {'found' if found[0] is not None else 'NOT found'} by the wrist, "
                f"{'found' if found[1] is not None else 'NOT found'} by the head"
            )
        move_closely(client, key, home, 15.0)
        client.keep_holding()
    (out / "joint.json").write_text(json.dumps({"arm": arm, "frames": frames}, indent=1) + "\n")
    print(f"saved in {out}; the arm back where it was")


# ---- the wrist camera's lens (the board held by a person, the arm still) ----------------------------------------


def lens_record(name: str, arm: str, seconds: float = 120.0, show: bool = False) -> None:
    """Stream the wrist camera for `seconds`, keeping each new view of the board (as the calibration does).

    Prints the views kept and what is missing every few seconds; `last.png` in the run's folder is the latest frame.
    """
    import time

    from lerobot.felix_jr.calibration.wrist_camera import is_new_view, lens_coverage_missing, view_signature

    out = RUNS / name
    out.mkdir(parents=True, exist_ok=True)
    client = api.Harness().client
    kept: list[list[list[float]]] = []
    signatures: list[np.ndarray] = []
    size = None
    start = last_print = time.time()
    while time.time() - start < seconds:
        frame = client.wrist_capture(arm)
        size = (frame.color.shape[1], frame.color.shape[0])
        corners = find_board(frame.color, BOARD)
        if corners is not None:
            signature = view_signature(corners, size, BOARD)
            if is_new_view(signature, signatures):
                signatures.append(signature)
                kept.append(corners.tolist())
                cv2.imwrite(
                    str(out / f"lens_{len(kept):03d}.png"), cv2.cvtColor(frame.color, cv2.COLOR_RGB2BGR)
                )
        if show:  # the picture, the board's corners when found, every kept view's outline, the count
            picture = frame.color.copy()
            for view in kept:
                hull = cv2.convexHull(np.asarray(view, dtype=np.float32)).astype(np.int32)
                cv2.polylines(picture, [hull], True, (0, 160, 0), 1, cv2.LINE_AA)
            if corners is not None:
                cv2.drawChessboardCorners(
                    picture, (BOARD.cols, BOARD.rows), corners.reshape(-1, 1, 2).astype(np.float32), True
                )
            text = f"{len(kept)} views, {seconds - (time.time() - start):.0f} s left; q to stop"
            cv2.putText(picture, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(
                picture, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
            )
            cv2.imshow(
                "lens check: move the board into every corner, tilted, near and far",
                cv2.cvtColor(picture, cv2.COLOR_RGB2BGR),
            )
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
        if time.time() - last_print > 5:
            cv2.imwrite(str(out / "last.png"), cv2.cvtColor(frame.color, cv2.COLOR_RGB2BGR))
            reach = np.array([s[:2] for s in signatures]) if signatures else np.zeros((0, 2))
            corners_hit = sum(
                bool(((np.abs(reach - c) < 0.18).all(axis=1)).any())
                for c in ((0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8))
            )
            print(
                f"   {time.time() - start:5.0f} s: {len(kept)} views; picture corners reached {corners_hit} of 4; "
                f"missing: {', '.join(lens_coverage_missing(signatures)) or 'nothing'}",
                flush=True,
            )
            last_print = time.time()
    if show:
        cv2.destroyAllWindows()
    (out / "lens.json").write_text(json.dumps({"arm": arm, "size": size, "views": kept}) + "\n")
    print(f"{len(kept)} views saved in {out}")


def lens_check(name: str, write: str | None = None) -> None:
    """The current lens on the recorded views (by where in the picture), a refit, and jnt1/jnt2's roll motions with each."""
    from lerobot.felix_jr.calibration.wrist_camera import calibrate_intrinsics

    info = json.loads((RUNS / name / "lens.json").read_text())
    arm, size = info["arm"], tuple(info["size"])
    views = [np.asarray(v, dtype=np.float64) for v in info["views"]]
    model = XleRobotModel.load(api.MODEL)
    old = model.wrist_cameras[arm]

    def per_view(camera) -> tuple[np.ndarray, np.ndarray]:
        errors, edge = [], []
        for v in views:
            pose = board_pose(camera, v, BOARD)
            points = BOARD.points() @ pose[:3, :3].T + pose[:3, 3]
            uv = camera.project(points)
            errors.append(float(np.sqrt(np.mean(np.sum((uv - v) ** 2, axis=1)))))
            center = v.mean(axis=0) / np.array(size)
            edge.append(bool(np.abs(center - 0.5).max() > 0.25))
        return np.array(errors), np.array(edge)

    new, _ = calibrate_intrinsics(views, BOARD, size, old.model)
    print(f"{len(views)} views")
    for label, camera in (("current lens", old), ("refit on these views", new)):
        e, edge = per_view(camera)
        print(
            f"{label:22s} fx {camera.fx:.1f} fy {camera.fy:.1f} center ({camera.ppx:.1f}, {camera.ppy:.1f}) | reprojection "
            f"{np.sqrt(np.mean(e**2)):.2f} px; middle {np.sqrt(np.mean(e[~edge] ** 2)) if (~edge).any() else float('nan'):.2f}, "
            f"near the edges {np.sqrt(np.mean(e[edge] ** 2)) if edge.any() else float('nan'):.2f} ({edge.sum()} views)"
        )
    for label, camera in (("current lens", old), ("refit", new)):
        ratios = []
        for run in ("jnt1", "jnt2"):
            if not (RUNS / run / "joint.json").exists():
                continue
            frames = json.loads((RUNS / run / "joint.json").read_text())["frames"]
            cams: list[np.ndarray] = []
            grips: list[np.ndarray] = []
            for f in frames:
                c = find_board(
                    cv2.cvtColor(cv2.imread(str(RUNS / run / f["image"])), cv2.COLOR_BGR2RGB), BOARD
                )
                if c is None:
                    continue
                g = model.wrist_camera_pose(arm, model.joint_vector({arm: f["arm_pose"]}))[:3, 3]
                options = [np.linalg.inv(board_pose(camera, cc, BOARD))[:3, 3] for cc in (c, c[::-1])]
                if (
                    cams
                ):  # the symmetric board: the corner order that keeps the board where the first frame had it
                    want = np.linalg.norm(g - grips[0])
                    options.sort(key=lambda p: abs(np.linalg.norm(p - cams[0]) - want))
                cams.append(options[0])
                grips.append(g)
            for a in range(len(cams)):
                for b in range(a + 1, len(cams)):
                    ratios.append(np.linalg.norm(cams[b] - cams[a]) / np.linalg.norm(grips[b] - grips[a]))
        if ratios:
            print(
                f"{label:22s} roll motions by PnP / the model: {np.mean(ratios):.3f} +- {np.std(ratios):.3f} (the head's ICP: ~1.03)"
            )
    if write:
        model.with_wrist_camera(arm, replace(new, gripper_from_camera=old.gripper_from_camera)).save(write)
        print(f"written: {write} (the refit lens, the same pose on the gripper)")


# ---- the camera's pose on the gripper, again (a hand-eye fit from the board) ------------------------------------


# joint offsets (deg) about where the arm holds, for `handeye-record --joints`: turns about every axis, a few shifts
JOINT_VIEWS: tuple[dict[str, float], ...] = (
    {},
    {"wrist_flex": 15},
    {"wrist_flex": -15},
    {"wrist_roll": 25},
    {"wrist_roll": -25},
    {"wrist_roll": 45},
    {"wrist_roll": -45},
    {"wrist_flex": 12, "wrist_roll": 30},
    {"wrist_flex": -12, "wrist_roll": -30},
    {"wrist_flex": 12, "wrist_roll": -30},
    {"wrist_flex": -12, "wrist_roll": 30},
    {"shoulder_pan": 6, "wrist_roll": 20},
    {"shoulder_pan": -6, "wrist_roll": -20},
    {"shoulder_lift": 5, "elbow_flex": -5, "wrist_flex": 8},
    {"shoulder_lift": -5, "elbow_flex": 5, "wrist_flex": -8},
    {"shoulder_pan": 5, "wrist_flex": 10, "wrist_roll": -40},
)


def handeye_joint_views(name: str, arm: str) -> None:
    """`JOINT_VIEWS` about where the arm holds, a frame of the board at each: varied poses without solving for any."""
    from lerobot.felix_jr.harness.wrist import move_closely, wrist_view

    out = RUNS / name
    out.mkdir(parents=True, exist_ok=True)
    h = api.Harness()
    model, client = h.model, h.client
    key = f"{arm}_arm"
    home = {k: float(v) for k, v in client.get_state()[key].items() if k != "gripper.pos"}
    if not _ask(
        f"walk the {arm} arm through {len(JOINT_VIEWS)} joint offsets (up to 45 deg of roll, 15 of flex)?"
    ):
        return
    frames = []
    with client.session():
        for k, offsets in enumerate(JOINT_VIEWS):
            target = {j: v + offsets.get(j.removesuffix(".pos"), 0.0) for j, v in home.items()}
            status, _ = move_closely(client, key, target, 15.0)
            view = wrist_view(model, client.wrist_capture(arm, after=status.get("finished_at")))
            corners = find_board(view.color, BOARD)
            image = f"handeye_{k:02d}.png"
            cv2.imwrite(str(out / image), cv2.cvtColor(view.color, cv2.COLOR_RGB2BGR))
            frames.append(
                {
                    "image": image,
                    "arm_pose": dict(view.arm_pose),
                    "corners": None if corners is None else corners.tolist(),
                }
            )
            print(f"   view {k} {offsets}: the board {'found' if corners is not None else 'NOT found'}")
        move_closely(client, key, home, 15.0)
        client.keep_holding()
    (out / "handeye.json").write_text(json.dumps({"arm": arm, "frames": frames}, indent=1) + "\n")
    print(
        f"{sum(f['corners'] is not None for f in frames)} views with the board, saved in {out}; back where it was"
    )


def handeye_record(name: str, arm: str, up: float = 0.0) -> None:
    """Walk the wrist calibration's views (`POSE_VIEWS`) around where the arm holds, a frame of the board at each."""
    from lerobot.felix_jr.calibration.wrist_camera import POSE_VIEWS
    from lerobot.felix_jr.calibration.xle_robot_model import pinch_frame
    from lerobot.felix_jr.harness.objects import angles_of
    from lerobot.felix_jr.harness.reach import solve_pose
    from lerobot.felix_jr.harness.wrist import move_closely, wrist_view

    out = RUNS / name
    out.mkdir(parents=True, exist_ok=True)
    h = api.Harness()
    model, client = h.model, h.client
    key = f"{arm}_arm"
    state = client.get_state()
    home = {k: float(v) for k, v in state[key].items()}
    q = model.joint_vector({arm: home})
    pinch = model.frame_pose(pinch_frame(arm), q)
    pitch0, _, roll0 = angles_of(pinch[:3, :3])
    base = pinch[:3, 3] + [0.0, 0.0, up]  # the views centered `up` above where the arm holds
    print(
        f"from the pinch at {np.round(base * 100, 1)} cm, pitch {pitch0:.0f}, roll {roll0:.0f}: {len(POSE_VIEWS)} views"
    )
    capture = SimpleNamespace(
        arms={arm: home}, head_ticks={n: int(state["head"][n]["ticks"]) for n in HEAD_JOINTS}
    )
    if not _ask(
        f"walk the {arm} arm through {len(POSE_VIEWS)} poses (2-3 cm shifts, +-20 deg tilts, turns)?"
    ):
        return
    frames = []
    with client.session():
        for k, (dx, dy, dz, dpitch, roll) in enumerate(POSE_VIEWS):
            target = base + [dx, dy, dz]
            poses = solve_pose(model, arm, capture, target, pitch0 + dpitch, None, roll0 + roll, init=home)  # type: ignore[arg-type]
            if not poses.ok[0]:
                print(f"   view {k}: out of reach")
                continue
            joints = dict(poses.joints[0])
            status, note = move_closely(client, key, joints, 15.0)
            view = wrist_view(model, client.wrist_capture(arm, after=status.get("finished_at")))
            corners = find_board(view.color, BOARD)
            image = f"handeye_{k:02d}.png"
            cv2.imwrite(str(out / image), cv2.cvtColor(view.color, cv2.COLOR_RGB2BGR))
            frames.append(
                {
                    "image": image,
                    "arm_pose": dict(view.arm_pose),
                    "corners": None if corners is None else corners.tolist(),
                }
            )
            print(f"   view {k}: the board {'found' if corners is not None else 'NOT found'} {note}")
        move_closely(client, key, {k: v for k, v in home.items() if k != "gripper.pos"}, 15.0)
        client.keep_holding()
    (out / "handeye.json").write_text(json.dumps({"arm": arm, "frames": frames}, indent=1) + "\n")
    print(
        f"{sum(f['corners'] is not None for f in frames)} views with the board, saved in {out}; back where it was"
    )


def handeye_fit(name: str, write: str | None = None, also: Sequence[str] = ()) -> None:
    """Fit the camera's pose on the gripper from `handeye_record`'s views; compare; with `write`, a new URDF."""
    from lerobot.felix_jr.calibration.wrist_camera import ArmView, fit_extrinsics
    from lerobot.felix_jr.calibration.xle_robot_model import pinch_frame

    folder = RUNS / name
    info = json.loads((folder / "handeye.json").read_text())
    arm = info["arm"]
    model = XleRobotModel.load(api.MODEL)
    camera = model.wrist_cameras[arm]
    views = [
        ArmView(f["arm_pose"], f["corners"], f["image"]) for f in info["frames"] if f["corners"] is not None
    ]
    for run in also:  # other runs' frames with the board (the board must not have moved between the runs)
        if (RUNS / run / "handeye.json").is_file():
            other = json.loads((RUNS / run / "handeye.json").read_text())["frames"]
            views += [
                ArmView(f["arm_pose"], f["corners"], f"{run}/{f['image']}") for f in other if f["corners"]
            ]
            continue
        sweep_info = json.loads((RUNS / run / "wrist.json").read_text())
        for f in sweep_info["frames"]:
            image = cv2.cvtColor(cv2.imread(str(RUNS / run / f["image"])), cv2.COLOR_BGR2RGB)
            corners = find_board(image, BOARD)
            if corners is not None:
                views.append(ArmView(f["arm_pose"], corners.tolist(), f"{run}/{f['image']}"))
    fit = fit_extrinsics(model, arm, camera, views, BOARD)
    print(f"{len(views)} views: {fit.describe()}")
    print(f"settled: {fit.settled} {fit.needs()}")
    old = np.array(camera.gripper_from_camera)
    new = fit.gripper_from_camera
    d = np.linalg.inv(old) @ new
    print(
        f"against {api.MODEL.name}: moved {1000 * np.linalg.norm(new[:3, 3] - old[:3, 3]):.1f} mm, turned "
        f"{np.degrees(np.linalg.norm(cv2.Rodrigues(d[:3, :3])[0])):.1f} deg"
    )
    fitted = model.with_wrist_camera(
        arm,
        replace(
            camera,
            gripper_from_camera=tuple(tuple(float(v) for v in row) for row in new),
            extrinsics_rms_px=float(np.sqrt(np.mean(fit.errors_px**2))),
            extrinsics_views=len(views),
        ),
    )
    for label, m in (("before", model), ("fitted", fitted)):
        q = m.joint_vector()
        in_pinch = np.linalg.inv(m.frame_pose(pinch_frame(arm), q)) @ m.wrist_camera_pose(arm, q)
        t = in_pinch[:3, 3] * 100
        print(
            f"   {label}: the camera {-t[2]:.1f} cm back from the pinch point along the fingers, "
            f"{np.hypot(t[0], t[1]):.1f} cm off them (ruler: ~11 cm back)"
        )
    if write:
        fitted.save(write)
        print(f"written: {write}")


# ---- analyze ---------------------------------------------------------------------------------------------------


def _load_views(folder: Path, model: XleRobotModel) -> tuple[list[WristView], list[str]]:
    info = json.loads((folder / "wrist.json").read_text())
    arm, camera = info["arm"], model.wrist_cameras[info["arm"]]
    from lerobot.felix_jr.calibration.xle_robot_model import pinch_frame

    views, labels = [], []
    for f in info["frames"]:
        color = cv2.cvtColor(cv2.imread(str(folder / f["image"])), cv2.COLOR_BGR2RGB)
        q = model.joint_vector({arm: f["arm_pose"]})
        views.append(
            WristView(
                arm,
                color,
                camera,
                f["arm_pose"],
                model.frame_pose(pinch_frame(arm), q),
                model.wrist_camera_pose(arm, q),
            )
        )
        labels.append(f["label"])
    return views, labels


def _blank_board(view: WristView, corners: np.ndarray) -> WristView:
    """The view with the board painted out (its corners' hull, widened): the methods must not use it."""
    hull = cv2.convexHull(corners.reshape(-1, 1, 2).astype(np.float32))
    center = hull.reshape(-1, 2).mean(axis=0)
    grown = center + (hull.reshape(-1, 2) - center) * (1 + BLANK_MARGIN_PX / max(np.ptp(hull[:, 0, 0]), 1.0))
    color = view.color.copy()
    cv2.fillConvexPoly(color, grown.round().astype(np.int32), (128, 128, 128))
    return replace(view, color=color)


def _motion_errors(
    estimate: list[np.ndarray], truth: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per frame, the motion from the first against the truth's: the turn's and the direction's errors, the length.

    In deg, and the length as a ratio (the images fix motion only up to one scale); NaN where a move is under 2 mm.
    """
    turns, directions, ratios = [], [], []
    for e, t in zip(estimate[1:], truth[1:], strict=True):
        m_e, m_t = np.linalg.inv(estimate[0]) @ e, np.linalg.inv(truth[0]) @ t
        d = np.linalg.inv(m_t) @ m_e
        turns.append(float(np.degrees(np.linalg.norm(cv2.Rodrigues(d[:3, :3])[0]))))
        a, b = m_e[:3, 3], m_t[:3, 3]
        if np.linalg.norm(b) < 0.002:
            directions.append(np.nan)
            ratios.append(np.nan)
            continue
        directions.append(
            float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))
        )
        ratios.append(float(np.linalg.norm(a) / np.linalg.norm(b)))
    return np.array(turns), np.array(directions), np.array(ratios)


def _head_table_z(folder: Path, model: XleRobotModel) -> float | None:
    """The table's height in the root frame by the head's depth at the board's center (the head's calibration)."""
    from lerobot.felix_jr.harness import tools

    capture = Capture.load(folder / "head", model)
    corners = find_board(capture.scene.color, BOARD)
    if corners is None:
        return None
    z = tools.locate_pixels(tools.full_view(capture), corners.mean(axis=0)[None, :]).xyz[0][2]
    return float(z) if np.isfinite(z) else None


def _head_table(folder: Path) -> tuple[float, float] | None:
    """The table (the board's plane) by the head's depth, in the board's frame: its height and tilt (m, deg)."""
    capture = Capture.load(folder / "head")
    s = capture.scene
    corners = find_board(s.color, BOARD)
    if corners is None:
        return None
    k = np.array([[s.fx, 0, s.ppx], [0, s.fy, s.ppy], [0, 0, 1.0]])
    ok, rvec, tvec = cv2.solvePnP(BOARD.points(), corners, k, None)
    if not ok:
        return None
    camera_from_board = np.eye(4)
    camera_from_board[:3, :3] = cv2.Rodrigues(rvec)[0]
    camera_from_board[:3, 3] = tvec.ravel()
    mask = np.zeros(s.depth_m.shape, np.uint8)
    cv2.fillConvexPoly(
        mask, cv2.convexHull(corners.reshape(-1, 1, 2).astype(np.float32)).round().astype(np.int32), 1
    )
    v, u = np.nonzero(mask & np.isfinite(s.depth_m))
    z = s.depth_m[v, u]
    points = np.column_stack([(u - s.ppx) / s.fx * z, (v - s.ppy) / s.fy * z, z])
    board_from_camera = np.linalg.inv(camera_from_board)
    in_board = points @ board_from_camera[:3, :3].T + board_from_camera[:3, 3]
    center = in_board.mean(axis=0)
    normal = np.linalg.svd(in_board - center)[2][-1]
    normal = normal if normal[2] > 0 else -normal
    return float(center[2]), float(np.degrees(np.arccos(min(abs(normal[2]), 1.0))))


def analyze(name: str, methods: list[str], prior_mm: float = 5.0, prior_deg: float = 1.0) -> None:
    """Score each method against the board (module docstring)."""
    folder = RUNS / name
    model = XleRobotModel.load(api.MODEL)
    views, labels = _load_views(folder, model)
    camera = views[0].camera
    corners = [find_board(v.color, BOARD) for v in views]
    use = [i for i, c in enumerate(corners) if c is not None]
    print(f"{name}: {len(views)} frames ({', '.join(labels)}); the board in {len(use)}: {use}")
    if len(use) < 3:
        print("too few frames see the board to score")
        return
    truth = [
        np.linalg.inv(board_pose(camera, corners[i], BOARD)) for i in use
    ]  # board_from_camera: the truth
    blanked = [_blank_board(views[i], corners[i]) for i in use]
    rows: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, tuple[float, float] | None, str | None]] = []
    model_poses = [views[i].root_from_camera for i in use]
    turns, dirs, ratios = _motion_errors(model_poses, truth)
    rows.append(("model (the joints read)", turns, dirs, ratios, None, None))
    for method in methods:
        try:
            scan = scan_views(
                blanked,
                method,
                current=len(use) - 1,
                labels=[labels[i] for i in use],
                prior_mm=prior_mm,
                prior_deg=prior_deg,
            )
        except ValueError as e:
            print(f"{method}: failed: {e}")
            continue
        turns, dirs, ratios = _motion_errors(list(scan.sfm.root_from_camera), truth)
        table = None
        if scan.sfm.plane is not None:
            # the SfM table plane, through the refined first frame, into the board's frame (the first frame's truth)
            normal, offset = scan.sfm.plane
            board_from_scan = truth[0] @ np.linalg.inv(scan.sfm.root_from_camera[0])
            n = board_from_scan[:3, :3] @ normal
            point = board_from_scan[:3, :3] @ (normal * offset) + board_from_scan[:3, 3]
            table = (
                float(n @ point / n[2]) if abs(n[2]) > 0.5 else float("nan"),
                float(np.degrees(np.arccos(min(abs(n[2]), 1.0)))),
            )
        rows.append((f"sfm {method}", turns, dirs, ratios, table, scan.sfm.describe()))
        table_z = _head_table_z(folder, model)
        if table_z is not None and scan.sfm.plane is not None:
            from lerobot.felix_jr.harness.wrist_scan import scale_from_table

            scaled = scale_from_table(scan, table_z)
            turns, dirs, ratios = _motion_errors(list(scaled.sfm.root_from_camera), truth)
            normal, offset = scaled.sfm.plane  # type: ignore[misc]
            board_from_scan = truth[0] @ np.linalg.inv(scaled.sfm.root_from_camera[0])
            n = board_from_scan[:3, :3] @ normal
            point = board_from_scan[:3, :3] @ (normal * offset) + board_from_scan[:3, 3]
            scaled_table = (float(n @ point / n[2]), float(np.degrees(np.arccos(min(abs(n[2]), 1.0)))))
            rows.append((f"sfm {method}, scaled", turns, dirs, ratios, scaled_table, scaled.scale_note))
    print(
        "\n                         turn error (deg)   direction error (deg)   scale (length ratio)   table in board frame"
    )
    for label, turns, dirs, ratios, table, _ in rows:
        t = f"z {1000 * table[0]:+.1f} mm, tilt {table[1]:.1f} deg" if table else "-"
        print(
            f"{label:24s} max {np.nanmax(turns):5.2f}            max {np.nanmax(dirs):5.2f}               "
            f"{np.nanmean(ratios):.3f} +- {np.nanstd(ratios):.3f}       {t}"
        )
    head = _head_table(folder)
    print(
        "head depth (the board)                                                                        "
        + (
            f"z {1000 * head[0]:+.1f} mm, tilt {head[1]:.1f} deg"
            if head
            else "board not found in the head capture"
        )
    )
    for label, *_, describe in rows:
        if describe:
            print(f"\n{label}:\n{describe}")


def main() -> None:
    """The command line (module docstring)."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("record", help="head capture, arm to the start pose, wrist sweep (moves the robot)")
    p.add_argument("name")
    p.add_argument("--arm", default="right")
    p.add_argument("--rolls", default=",".join(f"{r:g}" for r in SWEEP_ROLLS_DEG))
    p.add_argument("--flexes", default=",".join(f"{f:g}" for f in SWEEP_FLEXES_DEG))
    p.add_argument(
        "--skip-head", action="store_true", help="use the head capture already in the run's folder"
    )
    p.add_argument("--offset-cm", default="8,0", help="the tip from over the board's center, root x,y (cm)")
    p = sub.add_parser(
        "handeye-record", help="walk views of the board around where the arm holds (moves the arm)"
    )
    p.add_argument("name")
    p.add_argument("--arm", default="right")
    p.add_argument(
        "--up-cm", type=float, default=0.0, help="center the views this far above where the arm holds"
    )
    p.add_argument(
        "--joints", action="store_true", help="joint offsets about where it holds (`JOINT_VIEWS`) instead"
    )
    p = sub.add_parser("handeye-fit", help="fit the camera's pose on the gripper from them (no robot)")
    p.add_argument("name")
    p.add_argument("--write", default=None, help="write the model with it to this URDF")
    p.add_argument(
        "--also", default="", help="other runs' sweeps to use too (the board unmoved), comma-separated"
    )
    p = sub.add_parser("joint-record", help="head and wrist captures together at a few wrist rolls (moves)")
    p.add_argument("name")
    p.add_argument("--arm", default="right")
    p.add_argument(
        "--skip-head", action="store_true", help="use the head capture already in the run's folder"
    )
    p.add_argument(
        "--height-cm", type=float, default=None, help="the start pose's tip this far above the arm's base"
    )
    p = sub.add_parser("lens-record", help="keep views of the board held under the wrist camera (no motion)")
    p.add_argument("name")
    p.add_argument("--arm", default="right")
    p.add_argument("--seconds", type=float, default=120.0)
    p.add_argument(
        "--show", action="store_true", help="a live window of the wrist camera (run it from your terminal)"
    )
    p = sub.add_parser("lens-check", help="the lens on those views, a refit, the roll motions with each")
    p.add_argument("name")
    p.add_argument("--write", default=None)
    p = sub.add_parser("analyze", help="score the methods against the board (no robot)")
    p.add_argument("name")
    p.add_argument("--methods", default="light,torch")
    p.add_argument("--prior", default="5,1", help="the prior on the model's poses: mm,deg")
    args = parser.parse_args()
    if args.cmd == "record":
        floats = lambda text: [float(x) for x in text.split(",") if x]  # noqa: E731
        dx, dy = (v / 100 for v in floats(args.offset_cm))
        record(args.name, args.arm, floats(args.rolls), floats(args.flexes), args.skip_head, (dx, dy))
    elif args.cmd == "handeye-record":
        if args.joints:
            handeye_joint_views(args.name, args.arm)
        else:
            handeye_record(args.name, args.arm, args.up_cm / 100)
    elif args.cmd == "joint-record":
        heights = (args.height_cm / 100,) if args.height_cm is not None else START_HEIGHTS_M
        joint_record(args.name, args.arm, args.skip_head, heights)
    elif args.cmd == "lens-record":
        lens_record(args.name, args.arm, args.seconds, args.show)
    elif args.cmd == "lens-check":
        lens_check(args.name, args.write)
    elif args.cmd == "handeye-fit":
        handeye_fit(args.name, args.write, [r for r in args.also.split(",") if r])
    else:
        prior_mm, prior_deg = (float(v) for v in args.prior.split(","))
        analyze(args.name, args.methods.split(","), prior_mm, prior_deg)


if __name__ == "__main__":
    main()
