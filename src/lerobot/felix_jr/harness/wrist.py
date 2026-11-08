"""Lining the jaws up on an object from the wrist camera: look, measure, a conservative step, look again.

The design is in `docs/object_grasping.md` (section 6). The head camera's plan gets the gripper to the attack point
within a centimeter or three (the arm's sag, the calibration); the wrist camera, fixed to the jaws, sees what is left
in the jaws' own frame, without the head or the arm's model in between. What it does not see is depth: a pixel is a
ray. So:

- **The object's center.** The person (the model) marks the object's two sides at the grasp height (`edges`): its
  center is along the bisector of their two rays.
- **Its depth, from parallax.** The camera is several centimeters off the jaws' line (7 cm on the robot), and the
  ray to the object crosses that line at an angle, so a depth off by a fraction e puts the object off the line by
  about e times that offset, at any range (coming closer does not help). The object's width alone (the fitted
  primitive's: the two sides' angle gives `width / 2 / sin(angle / 2)`, exact for a cylinder side on) is good to its
  fit, 10-20%: 7-14 mm. Two looks with a known move between them are a stereo pair: the arm's small relative moves
  are accurate (through the model from the measured joints, its errors mostly cancel). `triangulate` locates a point
  marked in two wrist views (a `WristPoint`, with the rays' parallax and how well that pins the depth), and
  `plan_alignment` takes it as the object's center (`center`); without one, it uses the width's depth and says so.
  A sideways move makes the best pair (`sidestep_m`); an advance moves the camera mostly along the ray, so its
  parallax is a small part of the move and more of it the arm's error. The description shows the width's depth
  beside the one used: far apart, the plan's width is off.
- **Each step makes `STEP_FACTOR` of the measured correction**, so it falls short rather than overshoots, and the
  next one picks up the rest. The measurement itself uses the best depth, not a shortened one: the camera is off
  the jaws' line, so a point moved along its ray also moves across the line, and a depth shortened on purpose would
  be a sideways error that every look repeats (the steps would settle on it; found in the smoke test, 2026-11-08).
  A wrong depth still shifts the line's target the same way, but less the nearer the jaws come: advance in parts.
- **The target is a line, not a pixel.** The camera sits a few centimeters off the gripper's axis, so where the
  object should appear depends on how far ahead it is. The plan says where in the jaws the object's center belongs
  (`Poses.grasp`, the aim point in the pinch frame) and from which direction the jaws come in (pick.back -> pick):
  the object is lined up when its center is on that approach line. The step moves the gripper across the line by
  the measured offset (times `gain`), and optionally `advance` of the way along it; `roll_deg` turns the jaws about
  it (an angle needs no depth).
- **Each step reports how it went.** The view after a step carries where the object should appear if the step was
  exactly right (`WristView.expected_uv`). The next measurement compares: the fraction of the predicted shift that
  happened is the step's real gain (the depth's error and the arm's together), and its inverse a fair `gain` for the
  next step.

Corrections far from the plan are clipped (`MAX_STEP_M`, `MAX_ROLL_DEG`) and say so: past that, re-plan from the head
camera. The moves are relative, solved through the model from the measured joints (`ArmIK.solve_frame`), so the model's
own error mostly cancels.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import cv2
import numpy as np

from lerobot.felix_jr.calibration.xle_robot_model import WristCamera, XleRobotModel, pinch_frame

from .objects import Poses
from .reach import ArmIK

if TYPE_CHECKING:
    from lerobot.felix_jr.server.client import FelixClient, WristCapture

MIN_PARALLAX_DEG = (
    3.0  # rays crossing at less than this pin the depth poorly (~20 px at fx 400): a `WristPoint` says so
)
STEP_FACTOR = 0.8  # each step makes this fraction of the measured correction: short of it rather than past it
MAX_STEP_M = 0.03  # a step moves the gripper at most this far (across and along together)
MAX_ROLL_DEG = 20.0  # and turns it at most this much
STEP_POSITION_TOL_M = 0.001  # the step's solve: how closely the pinch frame must land
STEP_ROTATION_TOLS_DEG = (20.0, 10.0, 5.0)  # about the pinch frame's x (~closing), y (hinge), z (~approach)
MAX_DRIFT_DEG = 3.0  # the arm must be this close to where the view saw it
STEP_VEL = 10.0  # deg/s
TICK_M = 0.02  # the approach line's ticks
LINE_M = 0.16  # and how far ahead of the aim point it is drawn


def _cm(value: float) -> str:
    return f"{100 * value:.1f}"


def _mm(value: float) -> str:
    return f"{1000 * value:.0f}"


def _apply(pose: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) @ pose[:3, :3].T + pose[:3, 3]


@dataclass(frozen=True, eq=False)
class WristView:
    """A frame from `arm`'s wrist camera, with the joints read with it and where the model puts the camera then.

    `expected_uv` is set on the view an `align_step` returns: where the object's center should appear if the step went
    exactly as planned (`previous_uv` is where it was marked before the step).
    """

    sensed: ClassVar[bool] = True  # read off the robot: stale once the arm moves (`Workspace.add`)
    arm: str
    color: np.ndarray  # RGB uint8 (H, W, 3)
    camera: WristCamera
    arm_pose: Mapping[str, float]  # LeRobot degrees, with the gripper
    root_from_pinch: np.ndarray  # (4, 4) through the model at the measured joints
    root_from_camera: np.ndarray  # (4, 4) the same for the camera's optical frame
    timestamp: float = 0.0
    expected_uv: np.ndarray | None = None
    previous_uv: np.ndarray | None = None

    @property
    def pinch_from_camera(self) -> np.ndarray:
        """The camera in the pinch frame: fixed to the gripper (its calibrated pose), whatever the arm does."""
        return np.linalg.inv(self.root_from_pinch) @ self.root_from_camera

    def project_pinch(self, points_pinch: np.ndarray) -> np.ndarray:
        """Pixels (N, 2) of pinch-frame points (N, 3); NaN behind the camera."""
        camera_points = _apply(np.linalg.inv(self.pinch_from_camera), np.atleast_2d(points_pinch))
        uv = self.camera.project(camera_points)
        uv[camera_points[:, 2] <= 0] = np.nan
        return uv

    def root_rays(self, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The camera's position and the unit rays (N, 3) through pixels `uv` (N, 2), in the root frame."""
        return self.root_from_camera[:3, 3].copy(), self.rays(uv) @ self.root_from_camera[:3, :3].T

    def rays(self, uv: np.ndarray) -> np.ndarray:
        """Unit rays (N, 3) in the camera frame through pixels `uv` (N, 2), through the lens."""
        points = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)
        k, d = self.camera.matrix(), np.asarray(self.camera.coeffs)
        if self.camera.model == "fisheye":
            normalized = cv2.fisheye.undistortPoints(points, k, d)
        else:
            normalized = cv2.undistortPoints(points, k, d)
        rays = np.column_stack([normalized.reshape(-1, 2), np.ones(len(points))])
        return rays / np.linalg.norm(rays, axis=1, keepdims=True)

    def describe(self) -> str:
        """The arm, the frame's size, the gripper's opening, and what the last step expected."""
        height, width = self.color.shape[:2]
        text = f"{self.arm} wrist view {width}x{height}, gripper.pos {self.arm_pose.get('gripper.pos', float('nan')):.0f}"
        if self.expected_uv is not None:
            u, v = self.expected_uv
            text += f"; after the step the object's center should show near ({u:.0f}, {v:.0f})"
        return text


def wrist_view(model: XleRobotModel, capture: "WristCapture") -> WristView:
    """A `WristView` from `FelixClient.wrist_capture` (the joints read with the frame)."""
    arm = capture.arm
    camera = model.wrist_cameras.get(arm)
    if camera is None or camera.gripper_from_camera is None:
        raise ValueError(f"the model has no calibrated {arm} wrist camera: calibration.wrist_camera")
    pose = {k: float(v) for k, v in capture.state[f"{arm}_arm"].items()}
    q = model.joint_vector({arm: pose})
    return WristView(
        arm,
        capture.color,
        camera,
        pose,
        model.frame_pose(pinch_frame(arm), q),
        model.wrist_camera_pose(arm, q),
        capture.timestamp,
    )


def jaw_line(poses: Poses) -> tuple[np.ndarray, np.ndarray]:
    """From a pick plan (`plan_pick`): the aim point (where the object's center belongs) and the approach (pinch frame).

    The approach is the plan's pick.back -> pick direction, turned into the pick pose's pinch frame (it is fixed to
    the gripper); the pinch frame's +z where the plan has no attack point.
    """
    if poses.grasp is None:
        raise ValueError("these poses have no grasp point (solved before it was recorded): plan again")
    if "pick" in poses.labels and "pick.back" in poses.labels:
        pick, back = poses.labels.index("pick"), poses.labels.index("pick.back")
        direction = poses.points[pick] - poses.points[back]
        approach = poses.frames[pick][:3, :3].T @ (direction / np.linalg.norm(direction))
    else:
        approach = np.array([0.0, 0.0, 1.0])
    return np.asarray(poses.grasp, dtype=np.float64), approach


def render_wrist(
    model: XleRobotModel,
    view: WristView,
    pick: Poses | None = None,
    marks: Sequence[Sequence[float]] | None = None,
    labels: Sequence[str] | None = None,
) -> np.ndarray:
    """The wrist frame with the fingertips, the plan's approach line, `marks` and the last step's expectation.

    Cyan: the fixed jaw's tip (the pinch point); magenta: the moving one. Yellow: with `pick`, the approach line ahead
    of the aim point, a tick every `TICK_M` labeled in cm: where the object's center appears when it is lined up that
    far ahead. Green crosses: `marks` (pixels, labeled `labels` or 1, 2, ...). White circle: where the last step
    expected the object's center.
    """
    from lerobot.felix_jr.calibration.wrist_camera import fingertips_in_wrist_image

    out = view.color.copy()
    tips = fingertips_in_wrist_image(model, view.arm, view.camera, dict(view.arm_pose))
    for (u, v), color in zip(tips, ((0, 255, 255), (255, 0, 255)), strict=True):
        if np.isfinite([u, v]).all():
            cv2.circle(out, (int(round(u)), int(round(v))), 8, color, 2, cv2.LINE_AA)
    if pick is not None:
        aim, approach = jaw_line(pick)
        steps = np.arange(0.0, LINE_M + 1e-9, TICK_M / 4)
        uv = view.project_pinch(aim + steps[:, None] * approach)
        good = np.isfinite(uv).all(axis=1)
        if good.sum() >= 2:
            cv2.polylines(out, [uv[good].round().astype(np.int32)], False, (255, 220, 0), 2, cv2.LINE_AA)
        for distance, (u, v) in zip(steps, uv, strict=True):
            ticks = distance / TICK_M
            if np.isfinite([u, v]).all() and abs(ticks - round(ticks)) < 1e-6:
                center = (int(round(u)), int(round(v)))
                cv2.circle(out, center, 4, (255, 220, 0), -1, cv2.LINE_AA)
                _text(out, f"{100 * distance:.0f}", (center[0] + 6, center[1] - 6), (255, 220, 0))
    if view.expected_uv is not None and np.isfinite(view.expected_uv).all():
        u, v = view.expected_uv
        cv2.circle(out, (int(round(u)), int(round(v))), 12, (255, 255, 255), 2, cv2.LINE_AA)
        _text(out, "expected", (int(round(u)) + 14, int(round(v)) + 4), (255, 255, 255))
    for i, (u, v) in enumerate([] if marks is None else marks):
        center = (int(round(u)), int(round(v)))
        cv2.drawMarker(out, center, (0, 255, 0), cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
        label = labels[i] if labels is not None and i < len(labels) else str(i + 1)
        _text(out, label, (center[0] + 8, center[1] + 16), (0, 255, 0))
    return out


def _text(image: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, int, int]) -> None:
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


@dataclass(frozen=True, eq=False)
class Alignment:
    """One measured step (`plan_alignment`): where the object is in the jaws' frame and the move that corrects it."""

    view: WristView
    edges: np.ndarray  # (2, 2) the object's two sides at the grasp height, as marked (px)
    width: float  # m, the object's (the plan's)
    center_uv: np.ndarray  # (2,) the pixel of the object's center (the rays' bisector)
    distance_m: float  # from the camera to the object's center, as used
    depth_source: str  # where `distance_m` came from, in words
    width_distance_m: float  # what the width alone says
    object_pinch: np.ndarray  # (3,) the object's center, pinch frame
    aim_pinch: np.ndarray  # (3,) where it belongs (the plan's aim point)
    approach_pinch: np.ndarray  # (3,) the jaws' way in
    across_m: np.ndarray  # (3,) the object's offset from the approach line (pinch frame)
    ahead_m: float  # how far ahead of the aim point it is along the line
    move_pinch: np.ndarray  # (3,) the gripper's move (pinch frame)
    roll_deg: float  # the turn about the approach line through the aim point
    clipped: bool  # the move was cut to MAX_STEP_M / MAX_ROLL_DEG
    target_root_from_pinch: np.ndarray  # (4, 4)
    joints: dict[str, float]  # the step's solve, LeRobot degrees (the five arm joints)
    solve_error_m: float
    ok: bool  # the solve met the target
    last_gain: float | None  # the fraction of the last step's predicted shift that happened, if the view says

    def describe(self) -> str:
        """The measurement, in the jaws' frame, and the move."""
        across = float(np.linalg.norm(self.across_m))
        # across the line: toward the moving jaw (the aim point's side of the approach), and square to that
        toward = _unit(self.aim_pinch - (self.aim_pinch @ self.approach_pinch) * self.approach_pinch)
        jaws = float(self.across_m @ toward)
        hinge = float(self.across_m @ _unit(np.cross(self.approach_pinch, toward)))
        rows = [
            f"the object's center ~{_cm(self.distance_m)} cm from the camera (depth from {self.depth_source}; its "
            f"{_cm(self.width)} cm width says {_cm(self.width_distance_m)} cm), {_mm(across)} mm off the approach line ({_mm(jaws)} mm toward "
            f"the moving jaw, {_mm(hinge)} mm square to that), {_cm(self.ahead_m)} cm ahead of the aim point",
            f"the step: move {_mm(float(np.linalg.norm(self.move_pinch)))} mm "
            f"(pinch frame {', '.join(_mm(float(v)) for v in self.move_pinch)} mm), turn {self.roll_deg:.0f} deg"
            + (" (clipped: re-plan if this keeps happening)" if self.clipped else "")
            + (
                f"; the solve lands {_mm(self.solve_error_m)} mm off"
                if self.ok
                else f"; NOT reached: the solve lands {_mm(self.solve_error_m)} mm off"
            ),
        ]
        if self.last_gain is not None:
            suggestion = (
                f"gain {min(max(1.0 / self.last_gain, 0.5), 2.0):.1f}" if self.last_gain > 0.05 else "no gain"
            )
            rows.append(
                f"the last step moved the object {100 * self.last_gain:.0f}% of the way predicted "
                f"(a fair next {suggestion})"
            )
        return "\n".join(rows)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _roll_about(axis: np.ndarray, point: np.ndarray, degrees: float) -> np.ndarray:
    """(4, 4) the turn by `degrees` about the line through `point` along `axis`."""
    rotation, _ = cv2.Rodrigues(_unit(np.asarray(axis, dtype=np.float64)) * np.radians(degrees))
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = point - rotation @ point
    return pose


def measure(
    view: WristView,
    edges: np.ndarray,
    width: float,
    aim_pinch: np.ndarray,
    approach_pinch: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    """The object's center pixel, its distance from the camera (from `width`), and its center in the pinch frame.

    The distance is the width's own (not shortened: `plan_alignment` does that).
    """
    rays = view.rays(np.asarray(edges, dtype=np.float64).reshape(2, 2))
    angle = float(np.arccos(np.clip(rays[0] @ rays[1], -1.0, 1.0)))
    if angle < 1e-4:
        raise ValueError("the two edges are the same pixel: mark the object's two sides")
    distance = (width / 2) / np.sin(angle / 2)
    bisector = _unit(rays[0] + rays[1])
    center_uv = view.camera.project(bisector[None, :])[0]
    return center_uv, float(distance), _apply(view.pinch_from_camera, (distance * bisector)[None, :])[0]


def plan_alignment(
    model: XleRobotModel,
    view: WristView,
    pick: Poses,
    edges: Sequence[Sequence[float]],
    advance: float = 0.0,
    roll_deg: float = 0.0,
    gain: float = 1.0,
    sidestep_m: float = 0.0,
    center: "WristPoint | None" = None,
) -> Alignment:
    """Measure the object in `view` and plan the step that lines it up (the module docstring). Nothing moves.

    `advance` is the fraction (0-1) of the way to the aim point to move in along the approach; `gain` scales the move
    across; `sidestep_m` adds a move that far square to both the approach and the jaws' closing line, for the parallax
    (the next look can triangulate the depth, and corrects the sidestep back). `center`: the object's center
    triangulated (`triangulate`), for its depth; else the width's.
    """
    if not 0.0 <= advance <= 1.0:
        raise ValueError(f"advance is a fraction of the way in, 0 to 1, not {advance}")
    edges_px = np.asarray(edges, dtype=np.float64).reshape(2, 2)
    aim, approach = jaw_line(pick)
    approach = _unit(approach)
    center_uv, width_distance, _ = measure(view, edges_px, pick.width, aim, approach)
    distance, source = width_distance, "the width only: sidestep, then triangulate"
    if center is not None:
        origin, rays = view.root_rays(center_uv[None, :])
        distance = float((center.xyz - origin) @ rays[0])
        if distance <= 0:
            raise ValueError("the triangulated center is behind this view's camera: triangulate again")
        source = f"the triangulation ({center.parallax_deg:.0f} deg of parallax)"
    ray = view.rays(center_uv[None, :])[0]
    obj = _apply(view.pinch_from_camera, (distance * ray)[None, :])[0]
    offset = obj - aim
    ahead = float(offset @ approach)
    across = offset - ahead * approach
    move = STEP_FACTOR * (gain * across + advance * max(ahead, 0.0) * approach)
    toward = _unit(aim - (aim @ approach) * approach)
    move = move + sidestep_m * _unit(np.cross(approach, toward))
    roll = float(roll_deg)
    clipped = False
    if np.linalg.norm(move) > MAX_STEP_M:
        move, clipped = move * MAX_STEP_M / np.linalg.norm(move), True
    if abs(roll) > MAX_ROLL_DEG:
        roll, clipped = float(np.sign(roll) * MAX_ROLL_DEG), True
    # the gripper moves by `move` and turns about the approach line through the aim point (both in the pinch frame)
    step = _roll_about(approach, aim, roll)
    step[:3, 3] += move
    target = view.root_from_pinch @ step
    ik = ArmIK(model, view.arm, {view.arm: dict(view.arm_pose)})
    solution = ik.solve_frame(
        target,
        target[:3, :3],
        STEP_ROTATION_TOLS_DEG,
        position_tol=STEP_POSITION_TOL_M,
        init=ik.radians(view.arm_pose),
        position_bound=3 * STEP_POSITION_TOL_M,
    )
    error = float(np.linalg.norm(solution.root_from_pinch[:3, 3] - target[:3, 3]))
    last_gain = None
    if view.expected_uv is not None and view.previous_uv is not None:
        predicted = view.expected_uv - view.previous_uv
        if float(predicted @ predicted) > 1.0:
            last_gain = float((center_uv - view.previous_uv) @ predicted / (predicted @ predicted))
    return Alignment(
        view,
        edges_px,
        pick.width,
        center_uv,
        distance,
        source,
        width_distance,
        obj,
        aim,
        approach,
        across,
        ahead,
        move,
        roll,
        clipped,
        target,
        solution.joints,
        error,
        bool(solution.ok),
        last_gain,
    )


@dataclass(frozen=True, eq=False)
class WristPoint:
    """A point located in two wrist views (`triangulate`), root frame.

    A measurement, so fresh when made (`sensed`), though its views may be older (`api.triangulate_wrist` checks that
    only alignment steps came between).
    """

    sensed: ClassVar[bool] = True
    xyz: np.ndarray  # (3,) midway between the two rays' closest points
    label: str
    parallax_deg: float  # the angle between the two rays
    miss_m: float  # how far apart the rays pass: large means the marks are not the same point (or the arm drifted)
    depth_per_px_m: float  # how far the point moves along the newer ray for one pixel of marking error

    def describe(self) -> str:
        """Where, and how well the two views pin it."""
        p = self.xyz
        return (
            f"{self.label} at ({_cm(p[0])}, {_cm(p[1])}, {_cm(p[2])}) cm (root frame), from rays {self.parallax_deg:.1f} "
            f"deg apart passing {_mm(self.miss_m)} mm from each other; one pixel of marking moves it "
            f"{_mm(self.depth_per_px_m)} mm in depth"
            + (
                " (little parallax: move the camera aside more between the views)"
                if self.parallax_deg < MIN_PARALLAX_DEG
                else ""
            )
        )


def triangulate(
    view_a: WristView, uv_a: Sequence[float], view_b: WristView, uv_b: Sequence[float], label: str = "point"
) -> WristPoint:
    """The point marked at `uv_a` in `view_a` and at `uv_b` in `view_b` (same arm), from the two rays' closest points."""
    if view_a.arm != view_b.arm:
        raise ValueError("triangulate views of the same wrist camera")
    origin_a, ray_a = view_a.root_rays(np.asarray(uv_a, dtype=np.float64).reshape(1, 2))
    origin_b, ray_b = view_b.root_rays(np.asarray(uv_b, dtype=np.float64).reshape(1, 2))
    a, b = ray_a[0], ray_b[0]
    cos = float(a @ b)
    if cos > 1 - 1e-9:
        raise ValueError("the two rays are parallel: the camera did not move across them")
    w = origin_a - origin_b
    # closest points: origin_a + s a and origin_b + t b
    s = (cos * float(b @ w) - float(a @ w)) / (1 - cos * cos)
    t = (float(b @ w) - cos * float(a @ w)) / (1 - cos * cos)
    if s <= 0 or t <= 0:
        raise ValueError("the rays meet behind a camera: the marks are not the same point")
    pa, pb = origin_a + s * a, origin_b + t * b
    parallax = float(np.arccos(np.clip(cos, -1.0, 1.0)))
    per_px = t / (
        view_b.camera.fx * np.sin(parallax)
    )  # one pixel turns a ray by ~1/fx: along the other ray, /sin
    return WristPoint(
        (pa + pb) / 2,
        label,
        float(np.degrees(parallax)),
        float(np.linalg.norm(pa - pb)),
        float(per_px),
    )


def run_alignment(
    client: "FelixClient", model: XleRobotModel, alignment: Alignment, max_vel: float = STEP_VEL
) -> WristView:
    """Make the step (`alignment.joints`), hold there, and return the wrist view after it (with the expectation)."""
    view = alignment.view
    key = f"{view.arm}_arm"
    now = client.get_state()[key]
    drift = max(abs(now[k] - view.arm_pose[k]) for k in alignment.joints)
    if drift > MAX_DRIFT_DEG:
        raise RuntimeError(
            f"the {view.arm} arm is {drift:.1f} deg from where the wrist view saw it: look again"
        )
    if not alignment.ok:
        raise RuntimeError("the step's solve missed its target: plan a smaller step, or re-plan the grasp")
    with client.session():
        status = client.move_joints({key: alignment.joints}, max_vel=max_vel, max_accel=2 * max_vel, tol=1.0)
        client.keep_holding()
    if status["state"] != "done":
        raise RuntimeError(f"the step: {status['state']} ({status.get('reason')})")
    after = wrist_view(model, client.wrist_capture(view.arm, after=status.get("finished_at")))
    # where the object's center should now appear: fixed in the root frame, seen from after
    center_root = _apply(view.root_from_pinch, alignment.object_pinch[None, :])[0]
    expected = after.camera.project(_apply(np.linalg.inv(after.root_from_camera), center_root[None, :]))[0]
    return WristView(
        after.arm,
        after.color,
        after.camera,
        after.arm_pose,
        after.root_from_pinch,
        after.root_from_camera,
        after.timestamp,
        expected,
        alignment.center_uv,
    )
