"""From an object's antipodal grasps to the gripper: the claw mapping, the filters and the ranking.

The design is in `docs/object_grasping.md`, sections 2 and 3.

**The claw.** The pinch frame's origin is the fixed jaw's tip; the moving jaw turns about the hinge (the pinch y) and
its tip travels on an arc, so the line from the fixed tip to the moving tip is not the pinch -x: it tilts toward the
fingertips by an angle that depends on the opening (about 8 deg at 4 cm, 19 deg at 1 cm on this robot). `to_claw`
lines that tip-to-tip line up with the grasp's closing line, so both tips land on the contacts, and turns the hinge
square to the closing and the approach. The gripper axis (the pinch z) then leans off the approach by that tilt.

**Filters and ranking** (`grasp_candidates`). The tool measures and filters generously; the judgement is the model's
(`docs/harness.md` 1.K), through `GraspSettings`, which it can narrow and which every result prints. Each claw target
goes through filter stages (`STAGES`), cheapest first where one does not need another's result; the first it fails is
recorded with the reason (`Rejected`), and each stage is timed (`GraspCandidates.timing`):

- **jaws:** they open wide enough.
- **attack point**, then **grasp:** the arm reaches them (`ArmIK.solve_frame`), the grasp from the attack point's
  joints, so the move in has no joint jump. The solve prefers the exact antipodal pose (`ROTATION_SCALES_DEG`: the
  closing line kept square to the object most); what it accepts is the settings' bounds, generous by default.
- **joints:** every joint `min_joint_margin_deg` (15) from its bound at the attack point and the grasp. Ranked by
  the approach's preference alone, a cylinder's level grasp won with a joint at its stop (bowl1's jar; on the robot,
  ruler2's level approaches were only reachable that way).
- **wrist camera:** its optics `camera_clearance` (8 cm) from every obstacle (`Surface`s: the table's
  `TableEstimate`, other objects; without any, a level plane at the target's lower end) at the attack point and the
  grasp, through the model (`wrist_camera_position`). With the gripper upside down the camera hangs ~5 cm below the
  pinch point: ruler1 put it on the table. Not checked without the camera's pose on the gripper.
- **wrist view:** the grasp's center and both contacts in the wrist camera's picture at the attack point, clear of the
  fingers (`joint_views.wrist_sees`): a policy on the wrist camera steers by it (`GraspSettings.see_object`; skipped
  without the arm's wrist camera).
- **lift:** the arm reaches the lift (`GraspSettings.lift` above the grasp, as `plan_pick` lifts), from the grasp's
  joints, its joints `min_joint_margin_deg` from their bounds too.
- **paths:** the moves attack point -> grasp -> lift, straight in the joints as the arm makes them, keep the arm's links,
  both fingertips and the wrist camera's board (`path_points`) `GraspSettings.path_clearance` from every obstacle, or
  no nearer than a pose that is already nearer (a grasp low over the table). The joint session's wrist camera hit
  the table on a move no pose checked (2026-11-17). On bowl1's jar the two add ~3 % (1.9 of 60 s: they run on the few
  that pass the rest).

Whether the grasp is seen is not checked here: the head camera, behind the gripper, could not judge it (bowl3, bowl4),
and the wrist cameras will (`docs/object_grasping.md`, section 6).

The rest are ranked by how close to a true antipodal grip the arm gets (the solved pose's turn about the approach,
which tilts the closing line off the object's normal, and about the hinge, which slides the contacts; in steps of
`GraspSettings.rank_step_deg`), then by how far the joints stay from their bounds. The object is what the model
described: only its graspable part (the pen above the rim), so nothing else of it is generated or collided with.
Nothing checks collisions with the rest of the scene: `protocols.md` rule 1, at the attack hold, does.

**Which jaw where.** The fixed jaw does not move: closing sweeps the object onto it. So each antipodal grasp gives two
claw targets, one per contact taking the fixed jaw (`fixed_side` 0 or 1: `AntipodalGrasp.contacts_root`). A `margin`
aims the fixed tip that much clear of its contact, toward the outside; the jaws open wider, and closing pushes the
object up to `margin` onto the fixed jaw (`reach`'s module docstring: the same scheme as `solve_pose`).
"""

import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel

from .objects import Capture, Object, Poses, angles_of
from .primitives import AntipodalGrasp, HorizontalPlane, Primitive, Ranges, Spacing, Surface
from .reach import OPEN_CLEARANCE_M, POSITION_TOL_M, ROLL_TOL_DEG, ArmIK, FrameSolution, close_gripper_pos

SAG_LIFT_M = 0.01  # grasps aimed this much higher than the object (`GraspSettings.sag_lift`)
ATTACK_M = 0.05  # the attack point: this far back along the approach from the grasp (`protocols.md`, rule 1)
MARGIN_M = 0.008  # the fixed jaw this far clear of its contact (bowl1: 0.8-1.3 cm, every grasp landed)
# the wrist camera's optics at least this far from every obstacle at the attack point and the grasp: its housing (~2
# cm about them), the sag (~1 cm) and room to spare. ruler1 (2026-11-14): a grasp with the gripper upside down held at
# its attack point with the camera 2.3 cm up, and the camera hit the table; 4 cm then let an upside-down grasp pass
# at 4.3 cm (ruler2)
CAMERA_CLEARANCE_M = 0.08
MIN_JOINT_MARGIN_DEG = 15.0  # every joint this far from its bound, at the attack point and the grasp
MIN_TIP_APART_M = 1e-3  # the tip-to-tip direction of jaws this little apart, for a grasp of no opening
AXIS_NAMES = ("closing line", "approach", "hinge")  # the grasp's axes (`ClawTarget.axes_root`)
# the solve's preferences: how much a turn about each axis costs (a turn of its scale costs as much as a position error
# of `POSITION_TOL_M`): the closing line (keeps the contacts: the arm's missing sixth joint gives way here), the
# approach (tilts the closing line off the object's normal: bowl1's jar at roll -60 the worst), the hinge (slides the
# contacts along the approach). Not what is accepted: `GraspSettings.rotation_bounds_deg`.
ROTATION_SCALES_DEG = (20.0, 5.0, 10.0)
STAGES = (
    "jaws",
    "attack point",
    "grasp",
    "joints",
    "wrist camera",
    "wrist view",
    "lift",
    "paths",
)  # in order: cheap first, each after its needs
LIFT_M = 0.03  # the lift after the close (bowl1: 3 cm, then hold and look)
# the moves attack point -> grasp -> lift (straight in the joints, as the arm makes them) keep the arm this far from
# every obstacle, or, where a pose is already nearer (a grasp low over the table), no nearer than it: the arm's links,
# both fingertips (the jaws as they are on that move) and the wrist camera's board (`path_points`). The joint session's
# camera hit the table on a move no pose checked (2026-11-17)
PATH_CLEARANCE_M = 0.01
PATH_STEPS = 12  # each move checked at this many poses along it
PATH_LINKS = ("upper_arm_link", "lower_arm_link", "wrist_link", "gripper_link")  # their origins
CAMERA_BOARD_HALF_M = (
    0.0225  # the wrist camera's board, 45 mm square about the lens (`docs/hardware_risks.md`)
)


def _cm(value: float) -> str:
    return f"{value * 100:.1f}"


@dataclass(frozen=True)
class GraspSettings:
    """What `grasp_candidates` assumes, all of it the caller's (the model's) to change; printed with every result.

    The defaults filter generously and leave the judgement to the ranking and the model: a narrower bound is a choice
    about this object (its friction, how far its grip may tilt), not the tool's.
    """

    rotation_bounds_deg: tuple[float, float, float] = (
        45.0,
        45.0,
        45.0,
    )  # accepted turn about each of `AXIS_NAMES`
    position_bound_m: float = POSITION_TOL_M  # accepted miss of the grasp point
    rank_step_deg: float = (
        5.0  # antipodal errors this close rank as equal; the joint margin decides between them
    )
    margin: float = MARGIN_M  # the fixed jaw this far clear of its contact
    # the object's center this far back along the fingers from the tips (`to_claw`): 0, at the tips, the harness's way
    depth: float = 0.0
    attack: float = ATTACK_M  # the attack point this far back along the approach
    # the grasp and its attack point aimed this much higher than the object: the arm lands low (its sag; bowl6: the
    # elbow stopping 1.2-1.3 deg short under load), and the errors are not alike: too high grips higher on the
    # object, too low hits what it stands on
    sag_lift: float = SAG_LIFT_M
    # the wrist camera's optics this far from every obstacle, at least
    camera_clearance: float = CAMERA_CLEARANCE_M
    # every joint this far from its bound at the attack point, the grasp and the lift (deg)
    min_joint_margin_deg: float = MIN_JOINT_MARGIN_DEG
    # the object's grasp (its center and contacts) in the wrist camera's picture at the attack point, clear of the fingers:
    # what a policy on the wrist camera steers by (skipped without the arm's wrist camera)
    see_object: bool = True
    # the lift after the close, solved and checked with the rest (`plan_pick` lifts as much)
    lift: float = LIFT_M
    # the moves between the poses kept clear of the obstacles (`PATH_CLEARANCE_M`)
    path_clearance: float = PATH_CLEARANCE_M
    # rank by the target's `approach_preference` (after how antipodal; e.g. a cylinder from the side), in the steps of
    # `rank_step_deg`; False: not at all
    prefer_approach: bool = True

    def describe(self) -> str:
        """The settings in a line."""
        bounds = ", ".join(
            f"{name} {b:g}" for name, b in zip(AXIS_NAMES, self.rotation_bounds_deg, strict=True)
        )
        return (
            f"accepted: turns up to ({bounds}) deg, the grasp point within {_cm(self.position_bound_m)} cm; "
            f"ranked by the antipodal error in {self.rank_step_deg:g} deg steps, "
            + ("then the approach's preference in the same steps, " if self.prefer_approach else "")
            + "then the joint margin; fixed jaw "
            f"{_cm(self.margin)} cm clear, attack point {_cm(self.attack)} cm back, aimed {_cm(self.sag_lift)} cm high "
            f"against the arm's sag; the joints {self.min_joint_margin_deg:g} deg from their bounds and the wrist camera "
            f"{_cm(self.camera_clearance)} cm from every obstacle at least"
        )


@dataclass(frozen=True, eq=False)
class ClawTarget:
    """Where the gripper must be for one antipodal grasp, one contact on the fixed jaw.

    `root_from_pinch` is the pinch frame at the grasp, jaws open; `root_from_attack` the same pinch frame `attack` back
    along the approach. `aim_pinch` is the point of the pinch frame put on the grasp's center (with the margin),
    `held_pinch` where the center is once the jaws close (`XleRobotModel.grasp_offset` with and without the margin).
    `open_apart` / `close_apart` are how far apart the tips open for the approach and are told to close to (m; the
    close `reach.SQUEEZE_UNITS` past the object's contact, `reach.close_gripper_pos`).
    """

    grasp: AntipodalGrasp
    arm: str
    fixed_side: int  # which of the grasp's contacts takes the fixed jaw: 0 or 1
    root_from_pinch: np.ndarray  # (4, 4)
    root_from_attack: np.ndarray  # (4, 4)
    aim_pinch: np.ndarray  # (3,)
    held_pinch: np.ndarray  # (3,)
    margin: float
    attack: float
    open_apart: float
    close_apart: float
    depth: float = 0.0  # the object's center this far back along the fingers from the tips (m: `to_claw`)
    # the open jaws clear the object where it stands (`face_clearances`; at depth 0 the tips' width, as before)
    jaws_fit: bool = True

    @property
    def fixed_contact_root(self) -> np.ndarray:
        """The contact the fixed jaw takes."""
        return self.grasp.contacts_root[self.fixed_side]

    @property
    def axes_root(self) -> np.ndarray:
        """(3, 3) columns: the closing line (fixed to moving contact), the approach, the hinge (their cross product)."""
        d_root = self.grasp.closing_root if self.fixed_side == 0 else -self.grasp.closing_root
        return np.column_stack([d_root, self.grasp.approach_root, np.cross(d_root, self.grasp.approach_root)])

    @property
    def axis_lean_deg(self) -> float:
        """How far the gripper axis (pinch z) leans off the approach direction: the claw's tip-line tilt."""
        cos = float(self.root_from_pinch[:3, 2] @ self.grasp.approach_root)
        return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

    def describe(self) -> str:
        """The arm, which contact is on the fixed jaw, the jaw openings and the axis's lean."""
        f = self.fixed_contact_root
        return (
            f"{self.arm} arm, {self.grasp.describe()}; fixed jaw on contact {self.fixed_side} at "
            f"({', '.join(_cm(float(v)) for v in f)}) cm (root frame), {_cm(self.margin)} cm clear; jaws open "
            f"{_cm(self.open_apart)} cm, close to {_cm(self.close_apart)} cm; the gripper axis leans "
            f"{self.axis_lean_deg:.0f} deg off the approach"
        )


def to_claw(
    grasp: AntipodalGrasp,
    model: XleRobotModel,
    arm: str,
    fixed_side: int,
    margin: float = 0.0,
    attack: float = ATTACK_M,
    depth: float = 0.0,
) -> ClawTarget:
    """The claw target for `grasp`: `arm`'s fixed tip on contact `fixed_side`, the moving tip on the other.

    The fixed tip stands `margin` clear of its contact; the moving tip, at the grasp's opening, is on the other; the
    jaws come in along the grasp's approach, from the attack point `attack` back along it.

    `depth`: the object's center that far back along the fingers from the tips (the claw that much further on): 0, at
    the tips, as the harness has always grasped. Deeper, the jaws are a V narrower there: they open as wide as the
    fingers' faces need to clear the object by half `OPEN_CLEARANCE_M` (`face_clearances`), and `jaws_fit` is False
    when they cannot. Which depth holds best is the simulation's to say (`docs/sim2real_grasp.md`: a 52 mm bottle at
    15 mm was pushed out, a 45 mm one at 27.5 mm held).

    Raises ValueError if the jaws cannot open as wide as the grasp needs.
    """
    if fixed_side not in (0, 1):
        raise ValueError(f"fixed_side is 0 or 1, not {fixed_side}")
    # the claw in the pinch frame: the tip-to-tip direction at this opening, the hinge, and their cross product (toward
    # the fingertips)
    tip_pinch = model.grasp_offset(arm, max(grasp.opening, MIN_TIP_APART_M), mid_plane=False)
    u_pinch = tip_pinch / np.linalg.norm(tip_pinch)
    hinge_pinch = np.array([0.0, 1.0, 0.0]) - u_pinch[1] * u_pinch
    hinge_pinch /= np.linalg.norm(hinge_pinch)
    forward_pinch = np.cross(hinge_pinch, u_pinch)
    # the same three directions at the grasp: fixed to moving contact, the hinge square to it and the approach
    d_root = grasp.closing_root if fixed_side == 0 else -grasp.closing_root
    a_root = grasp.approach_root
    h_root = np.cross(d_root, a_root)  # as `ClawTarget.axes_root`
    rot_root_from_pinch = (
        np.column_stack([h_root, d_root, a_root]) @ np.column_stack([hinge_pinch, u_pinch, forward_pinch]).T
    )

    back = np.array([0.0, 0.0, depth])  # (the pinch z points toward the fingertips)
    aim_pinch = model.grasp_offset(arm, grasp.opening, margin) - back
    root_from_pinch = np.eye(4)
    root_from_pinch[:3, :3] = rot_root_from_pinch
    root_from_pinch[:3, 3] = grasp.center_root - rot_root_from_pinch @ aim_pinch
    root_from_attack = root_from_pinch.copy()
    root_from_attack[:3, 3] -= attack * a_root
    widest = 2 * model.jaw_radius(arm)
    open_apart, jaws_fit = min(grasp.opening + margin + OPEN_CLEARANCE_M, widest), True
    if depth > 0:
        open_apart, jaws_fit = _open_apart_at_depth(model, arm, aim_pinch, grasp.opening / 2, open_apart)
    return ClawTarget(
        grasp,
        arm,
        fixed_side,
        root_from_pinch,
        root_from_attack,
        aim_pinch,
        model.grasp_offset(arm, grasp.opening) - back,
        margin,
        attack,
        open_apart,
        _apart_at(model, arm, close_gripper_pos(model, arm, grasp.opening))
        if arm in model.gripper_maps
        else 0.0,
        depth,
        jaws_fit,
    )


def face_clearances(
    model: XleRobotModel, arm: str, center_pinch: np.ndarray, radius: float, opening_deg: float
) -> tuple[float, float]:
    """How far (m) the fixed and the moving finger's inner faces are from an object of `radius` about `center_pinch`.

    Each face the plane through its fingertip square to its finger frame's y (`XleRobotModel.finger_pose`: the STL's
    straight face, `sim2real.robot.finger_pads`), the moving one turned by the jaws' opening; signed, negative into the
    object, measured on the side toward the jaws' middle.
    """
    middle = model.grasp_offset(arm, 2 * radius, mid_plane=False)
    out = []
    for finger, turn in (("fixed", None), ("moving", model.moving_jaw(arm, opening_deg))):
        frame = model.finger_pose(arm, finger)
        if turn is not None:
            frame = turn @ frame
        origin, normal = frame[:3, 3], frame[:3, 1]
        if (middle - origin) @ normal < 0:  # toward the jaws' middle
            normal = -normal
        out.append(float((center_pinch - origin) @ normal - radius))
    return out[0], out[1]


def _open_apart_at_depth(
    model: XleRobotModel, arm: str, center_pinch: np.ndarray, radius: float, open_apart: float
) -> tuple[float, bool]:
    """The tips' opening (m apart) that clears an object deep in the jaws, and whether the jaws open that wide.

    At least `open_apart` (the tips' rule); wider until the moving finger's face is half `OPEN_CLEARANCE_M` from the
    object (`face_clearances`), up to the widest the jaws open (`widest_apart`). Without the gripper map, as given.
    """
    if arm not in model.gripper_maps:
        return open_apart, True
    radius_m = model.jaw_radius(arm)
    opening = model.opening_for_width(arm, min(open_apart, 2 * radius_m))
    widest_deg = min(model.gripper_maps[arm].opening_deg(100.0), 180.0)
    need = OPEN_CLEARANCE_M / 2
    while face_clearances(model, arm, center_pinch, radius, opening)[1] < need:
        opening += 1.0
        if opening > widest_deg:
            return widest_apart(model, arm), False
    return float(2 * radius_m * np.sin(np.radians(opening) / 2)), True


def _apart_at(model: XleRobotModel, arm: str, gripper_pos: float) -> float:
    """How far apart (m) `arm`'s tips are told to be at `gripper_pos` (0 past where they meet)."""
    opening = max(model.opening_deg(arm, gripper_pos), 0.0)
    return float(2 * model.jaw_radius(arm) * np.sin(np.radians(opening) / 2))


def widest_apart(model: XleRobotModel, arm: str) -> float:
    """How far apart (m) `arm`'s tips can open: the moving tip's circle, or its gripper map's widest opening."""
    radius = model.jaw_radius(arm)
    if arm not in model.gripper_maps:
        return 2 * radius
    widest_deg = min(model.gripper_maps[arm].opening_deg(100.0), 180.0)
    return float(2 * radius * np.sin(np.radians(widest_deg) / 2))


@dataclass(frozen=True, eq=False)
class GraspCandidate:
    """One grasp that passed every filter, with the arm's solved poses at the attack point and at the grasp."""

    name: str
    claw: ClawTarget
    attack: FrameSolution
    grasp: FrameSolution
    limit_margin_deg: float  # the closest any joint comes to its bound, over both poses
    cost: float  # the solves' summed squared residuals, in their scales: the last tie-break
    settings: GraspSettings = field(default_factory=GraspSettings)
    # the wrist camera's optics from the nearest obstacle, over both poses (nan: not checked)
    camera_clearance_m: float = float("nan")
    # how far (deg) the solved gripper axis is from the target's best way in (`Primitive.approach_preference`)
    approach_preference_deg: float = 0.0
    # the arm's solved pose at the lift (`GraspSettings.lift` above the grasp)
    lift: FrameSolution | None = None
    # the moves' nearest approach to an obstacle (m: the arm's links, the fingertips, the wrist camera's board; nan:
    # not checked)
    path_clearance_m: float = float("nan")
    # the grasp's points the wrist camera sees from the attack point, of its center and two contacts (-1: not checked)
    wrist_view_points: int = -1

    @property
    def antipodal_error_deg(self) -> float:
        """How far (deg) the solved grasp is from a true antipodal grip, the larger of its two turns that matter.

        About the approach: the closing line off the object's normal. About the hinge: the contacts slid along the
        approach. (About the closing line only tilts the approach: harmless to the grip.)
        """
        return float(max(self.grasp.rotation_error_deg[1], self.grasp.rotation_error_deg[2]))

    def rank_key(self) -> tuple[int, int, float, float]:
        """The ranking, lower first.

        The antipodal error in the settings' steps, the approach's preference in the same steps (if
        `prefer_approach`), the joint margin, the solves' cost.
        """
        step = self.settings.rank_step_deg
        antipodal = int(self.antipodal_error_deg / step + 1e-9)
        approach = int(self.approach_preference_deg / step + 1e-9) if self.settings.prefer_approach else 0
        return (antipodal, approach, -self.limit_margin_deg, self.cost)

    def describe(self) -> str:
        """The grasp in the object's terms, the jaws, the solved gripper pose (root frame), how antipodal, joints."""
        pitch, heading, roll = angles_of(self.grasp.root_from_pinch[:3, :3])
        closing, approach, hinge = self.grasp.rotation_error_deg
        return (
            f"{self.name}: {self.claw.arm} arm, {self.claw.grasp.describe()}; fixed jaw on contact "
            f"{self.claw.fixed_side}, {_cm(self.claw.margin)} cm clear; gripper at pitch {pitch:.0f}, heading "
            f"{heading:.0f}, roll {roll:.0f} (root frame); antipodal error {self.antipodal_error_deg:.0f} deg (about the "
            f"approach {approach:.0f}, about the hinge {hinge:.0f}), approach tilted {closing:.0f} deg; the gripper "
            f"{self.approach_preference_deg:.0f} deg off the object's best way in; joints >= "
            f"{self.limit_margin_deg:.0f} deg from their bounds; off the target by "
            f"{_cm(self.grasp.position_error_m)} cm"
            + (
                f"; the wrist camera {_cm(self.camera_clearance_m)} cm from the nearest obstacle"
                if np.isfinite(self.camera_clearance_m)
                else ""
            )
        )


@dataclass(frozen=True, eq=False)
class Rejected:
    """One claw target that failed a filter: the stage (`STAGES`), why, and as far as it got."""

    grasp: AntipodalGrasp
    arm: str
    fixed_side: int
    stage: str
    reason: str
    claw: ClawTarget | None = None  # None if the jaws cannot open for it
    attack: FrameSolution | None = None  # the solves, where it got that far
    at_grasp: FrameSolution | None = None
    detail: str = ""  # the numbers behind the reason, where it has them

    def describe(self) -> str:
        """The arm, the grasp, the jaw and the reason (with its numbers)."""
        detail = f" ({self.detail})" if self.detail else ""
        return f"{self.arm} arm, {self.grasp.describe()}, fixed jaw on contact {self.fixed_side}: {self.reason}{detail}"


@dataclass(frozen=True, eq=False)
class GraspCandidates:
    """The best grasps of a primitive, ranked (named A, B, ...), every claw target rejected with why, and timing.

    `timing` is each filter stage's total time and how many claw targets it ran on; `settings` what was assumed.
    """

    candidates: tuple[GraspCandidate, ...]  # the best `n`
    tried: int  # claw targets: grasps x arms x the two fixed sides
    passed: int
    rejected: tuple[Rejected, ...]
    timing: Mapping[str, tuple[float, int]] = field(
        default_factory=dict
    )  # stage -> (seconds, claw targets run)
    settings: GraspSettings = field(default_factory=GraspSettings)
    capture: Capture | None = None  # what the arm's start and the camera were (`plan_pick` uses it)
    label: str = ""  # the object's

    def __getitem__(self, name: str) -> GraspCandidate:
        """The candidate named `name` ("A", "B", ...)."""
        for candidate in self.candidates:
            if candidate.name == name:
                return candidate
        raise KeyError(f"no candidate {name!r}: {', '.join(c.name for c in self.candidates) or 'none'}")

    def rejected_by_reason(self) -> dict[str, int]:
        """How many were rejected for each reason, most first."""
        return dict(Counter(r.reason for r in self.rejected).most_common())

    def nearest_misses(self, n: int = 3) -> tuple[Rejected, ...]:
        """Up to `n` of the rejected that got furthest through the filters (the last stage any reached)."""
        if not self.rejected:
            return ()
        last = max(STAGES.index(r.stage) for r in self.rejected)
        return tuple(r for r in self.rejected if STAGES.index(r.stage) == last)[:n]

    def timing_text(self) -> str:
        """Per stage, in order: its total time and the time per claw target it ran on."""
        rows = []
        for stage in STAGES:
            if stage in self.timing:
                total, runs = self.timing[stage]
                rows.append(f"{stage} {total:.2f} s ({1000 * total / runs:.2f} ms x {runs})")
        return "; ".join(rows)

    def describe(self) -> str:
        """The settings, the candidates, the rejected by reason, the nearest misses and the time per stage."""
        rejected = "; ".join(f"{count} {reason}" for reason, count in self.rejected_by_reason().items())
        what = f"{self.label}: " if self.label else ""
        head = f"{what}{self.passed} of {self.tried} claw targets pass, best {len(self.candidates)}"
        rows = [c.describe() for c in self.candidates]
        misses = [f"  {r.describe()}" for r in self.nearest_misses()]
        return "\n".join(
            [
                f"settings: {self.settings.describe()}",
                head + ":",
                *rows,
                f"rejected: {rejected or 'none'}",
                *(["nearest misses:", *misses] if misses else []),
                f"time: {self.timing_text()}",
            ]
        )


def _rejection(solution: FrameSolution, where: str, settings: GraspSettings) -> str | None:
    """Why a solve fails the settings' bounds (None if it does not), in words without numbers, for counting."""
    if solution.ok:
        return None
    if solution.position_error_m > settings.position_bound_m:
        return f"{where} out of reach"
    over = np.asarray(solution.rotation_error_deg) / np.asarray(settings.rotation_bounds_deg)
    return f"{where}: the arm cannot turn the gripper about the {AXIS_NAMES[int(np.argmax(over))]}"


def _limit_margin_deg(ik: ArmIK, *solutions: FrameSolution) -> float:
    """The closest any joint of the solutions comes to its bound (deg)."""
    return float(min(np.degrees(np.minimum(s.x - ik.lower, ik.upper - s.x)).min() for s in solutions))


def _solve(
    ik: ArmIK, claw: ClawTarget, target: np.ndarray, settings: GraspSettings, init: np.ndarray | None = None
):
    """One frame solve for a claw target: the solve's scales, the settings' bounds."""
    return ik.solve_frame(
        target,
        claw.axes_root,
        ROTATION_SCALES_DEG,
        init=init,
        rotation_bounds_deg=settings.rotation_bounds_deg,
        position_bound=settings.position_bound_m,
    )


class _Timer:
    """Each filter stage's total time and how many claw targets it ran on."""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = defaultdict(float)
        self.runs: dict[str, int] = defaultdict(int)

    def __call__(self, stage: str, start: float) -> None:
        self.seconds[stage] += time.perf_counter() - start
        self.runs[stage] += 1

    def timing(self) -> dict[str, tuple[float, int]]:
        return {stage: (self.seconds[stage], self.runs[stage]) for stage in STAGES if self.runs[stage]}


def _shifted(claw: ClawTarget, offset_root: np.ndarray) -> ClawTarget:
    """`claw` with its grasp and attack frames moved by `offset_root` (the object's contacts stay where they are)."""
    pinch, attack = claw.root_from_pinch.copy(), claw.root_from_attack.copy()
    pinch[:3, 3] += offset_root
    attack[:3, 3] += offset_root
    return replace(claw, root_from_pinch=pinch, root_from_attack=attack)


def wrist_camera_position(model: XleRobotModel, arm: str, solution: FrameSolution) -> np.ndarray | None:
    """The wrist camera's optics (root frame, m) at a solve's joints; None without a calibrated wrist camera."""
    if arm not in model.wrist_cameras:
        return None
    return model.wrist_camera_pose(arm, model.joint_vector({arm: dict(solution.joints)}))[:3, 3]


def _evaluate(
    model: XleRobotModel,
    claw: ClawTarget,
    ik: ArmIK,
    settings: GraspSettings,
    timed: _Timer,
    init: np.ndarray | None = None,
    obstacles: Sequence[Surface] = (),
    primitive: Primitive | None = None,
) -> GraspCandidate | Rejected:
    """One claw target through the filters after the jaws (`STAGES`): a candidate, or why it is rejected."""
    grasp, arm, fixed_side = claw.grasp, claw.arm, claw.fixed_side
    start = time.perf_counter()
    at_attack = _solve(ik, claw, claw.root_from_attack, settings, init=init)
    timed("attack point", start)
    reason = _rejection(at_attack, "attack point", settings)
    if reason is not None:
        return Rejected(grasp, arm, fixed_side, "attack point", reason, claw, at_attack)

    start = time.perf_counter()
    at_grasp = _solve(ik, claw, claw.root_from_pinch, settings, init=at_attack.x)
    timed("grasp", start)
    reason = _rejection(at_grasp, "grasp", settings)
    if reason is not None:
        return Rejected(grasp, arm, fixed_side, "grasp", reason, claw, at_attack, at_grasp)

    start = time.perf_counter()
    limit_margin = _limit_margin_deg(ik, at_attack, at_grasp)
    timed("joints", start)
    if limit_margin < settings.min_joint_margin_deg:
        return Rejected(
            grasp,
            arm,
            fixed_side,
            "joints",
            "a joint too near its bound",
            claw,
            at_attack,
            at_grasp,
            f"{limit_margin:.0f} deg",
        )

    start = time.perf_counter()
    positions = [wrist_camera_position(model, arm, s) for s in (at_attack, at_grasp)]
    clearance, nearest = float("nan"), None
    if obstacles and all(p is not None for p in positions):
        points = np.array(positions)
        distances = [float(np.min(o.surface_distance(points))) for o in obstacles]
        k = int(np.argmin(distances))
        clearance, nearest = distances[k], obstacles[k]
    timed("wrist camera", start)
    if clearance < settings.camera_clearance:  # (False when unknown: nan)
        return Rejected(
            grasp,
            arm,
            fixed_side,
            "wrist camera",
            "the wrist camera too near an obstacle",
            claw,
            at_attack,
            at_grasp,
            f"{_cm(clearance)} cm from {nearest.describe() if nearest is not None else 'it'}",
        )

    start = time.perf_counter()
    seen = -1
    if settings.see_object and arm in model.wrist_cameras:
        from lerobot.felix_jr.calibration.joint_views import wrist_sees

        points = np.array([grasp.center_root, *grasp.contacts_root])
        uv = wrist_sees(
            model, arm, model.joint_vector({arm: dict(at_attack.joints)}), points, min_fraction=0.0
        )
        seen = int(np.isfinite(uv).all(axis=1).sum()) if uv is not None else 0
    timed("wrist view", start)
    if 0 <= seen < 3:
        return Rejected(
            grasp,
            arm,
            fixed_side,
            "wrist view",
            "the object out of the wrist camera's view at the attack point",
            claw,
            at_attack,
            at_grasp,
            f"{seen} of its 3 grasp points seen",
        )

    start = time.perf_counter()
    lift_target = claw.root_from_pinch.copy()
    lift_target[2, 3] += settings.lift
    at_lift = _solve(ik, claw, lift_target, settings, init=at_grasp.x)
    timed("lift", start)
    reason = _rejection(at_lift, "lift", settings)
    lift_margin = _limit_margin_deg(ik, at_lift)
    if reason is None and lift_margin < settings.min_joint_margin_deg:
        reason = "a joint too near its bound at the lift"
    if reason is not None:
        return Rejected(
            grasp, arm, fixed_side, "lift", reason, claw, at_attack, at_grasp, f"{lift_margin:.0f} deg"
        )

    start = time.perf_counter()
    path_clearance = float("nan")
    if obstacles:
        open_deg = _opening_deg(model, arm, claw.open_apart)
        closed_deg = _opening_deg(model, arm, claw.close_apart)
        moves = ((at_attack, at_grasp, open_deg), (at_grasp, at_lift, closed_deg))
        path_clearance, why = _paths_clear(model, arm, moves, obstacles, settings.path_clearance)
        timed("paths", start)
        if why is not None:
            return Rejected(
                grasp, arm, fixed_side, "paths", why, claw, at_attack, at_grasp, f"{_cm(path_clearance)} cm"
            )
    limit_margin = min(limit_margin, lift_margin)

    cost = sum(
        (s.position_error_m / POSITION_TOL_M) ** 2
        + float(np.sum((np.asarray(s.rotation_error_deg) / np.asarray(ROTATION_SCALES_DEG)) ** 2))
        for s in (at_attack, at_grasp)
    )
    preference = 0.0 if primitive is None else primitive.approach_preference(at_grasp.root_from_pinch[:3, 2])
    return GraspCandidate(
        "",
        claw,
        at_attack,
        at_grasp,
        limit_margin,
        cost,
        settings,
        clearance,
        preference,
        at_lift,
        path_clearance,
        seen,
    )


def _opening_deg(model: XleRobotModel, arm: str, apart: float) -> float | None:
    """The jaws' opening (deg) with the tips `apart` m apart; None without the arm's gripper map."""
    if arm not in model.gripper_maps:
        return None
    return float(model.opening_for_width(arm, min(apart, 2 * model.jaw_radius(arm))))


def path_points(
    model: XleRobotModel, arm: str, joints: Mapping[str, float], opening_deg: float | None
) -> np.ndarray:
    """The arm's points kept clear of obstacles on a move (root frame, (N, 3)) at `joints` (LeRobot degrees).

    The origins of its links (`PATH_LINKS`), the fixed fingertip and, with the jaws' opening known, the moving one, and
    the wrist camera's board's four corners (with its pose calibrated).
    """
    q = model.joint_vector({arm: dict(joints)})
    points = [model.frame_pose(f"{arm}_{link}", q)[:3, 3] for link in PATH_LINKS]
    points.append(model.pinch_point(arm, q))
    if opening_deg is not None:
        points.append(model.moving_tip(arm, q, opening_deg))
    if arm in model.wrist_cameras:
        camera = model.wrist_camera_pose(arm, q)
        h = CAMERA_BOARD_HALF_M
        corners = np.array([[x, y, 0.0] for x in (-h, h) for y in (-h, h)])
        points.extend(corners @ camera[:3, :3].T + camera[:3, 3])
    return np.array(points)


def _clearance(points: np.ndarray, obstacles: Sequence[Surface]) -> float:
    return float(min(np.min(o.surface_distance(points)) for o in obstacles))


def _paths_clear(
    model: XleRobotModel,
    arm: str,
    moves: Sequence[tuple[FrameSolution, FrameSolution, float | None]],
    obstacles: Sequence[Surface],
    clearance: float,
) -> tuple[float, str | None]:
    """The moves' nearest approach to the obstacles (m), and why a move fails (None: none does).

    Each move straight in the joints (`PATH_STEPS` poses), its points (`path_points`, the jaws at its opening) kept
    `clearance` from every obstacle, or no nearer than the nearer of its two ends where that is already less.
    """
    nearest = np.inf
    for a, b, opening in moves:
        ends = [_clearance(path_points(model, arm, s.joints, opening), obstacles) for s in (a, b)]
        floor = min(clearance, *ends) - 1e-3
        for t in np.linspace(0.0, 1.0, PATH_STEPS + 1)[1:-1]:
            joints = {k: a.joints[k] + t * (b.joints[k] - a.joints[k]) for k in a.joints}
            here = _clearance(path_points(model, arm, joints, opening), obstacles)
            nearest = min(nearest, here)
            if here < floor:
                return nearest, "a move passes too near an obstacle"
        nearest = min(nearest, *ends)
    return float(nearest), None


def _ranked(passed: Sequence[GraspCandidate], n: int) -> tuple[GraspCandidate, ...]:
    ranked = sorted(passed, key=GraspCandidate.rank_key)  # stable: ties keep the generation order
    return tuple(replace(c, name=chr(ord("A") + i)) for i, c in enumerate(ranked[: min(n, 26)]))


def _default_obstacles(primitive: Primitive) -> list[Surface]:
    """What a target standing on the table keeps clear of when none are given: a level plane at its lower end."""
    if hasattr(primitive, "bottom_root") and hasattr(primitive, "top_root"):
        return [HorizontalPlane(float(min(primitive.bottom_root[2], primitive.top_root[2])))]
    return []


def _evaluate_target(
    model: XleRobotModel,
    primitive: Primitive,
    grasp: AntipodalGrasp,
    arm: str,
    fixed_side: int,
    settings: GraspSettings,
    capture: Capture,
    iks: dict[tuple[str, float], ArmIK],
    timed: _Timer,
    obstacles: tuple[Surface, ...],
) -> "GraspCandidate | Rejected":
    """One claw target (an antipodal grasp, an arm, which contact on the fixed jaw) through every stage (`STAGES`).

    The jaws first, then `_evaluate`'s. `iks` caches the arms' solvers by (arm, opening) across calls.
    """
    start = time.perf_counter()
    too_wide = grasp.opening + settings.margin > widest_apart(model, arm)
    timed("jaws", start)
    if too_wide:
        return Rejected(grasp, arm, fixed_side, "jaws", "too wide for the jaws")
    claw = _shifted(
        to_claw(grasp, model, arm, fixed_side, settings.margin, settings.attack, settings.depth),
        np.array([0.0, 0.0, settings.sag_lift]),
    )
    if not claw.jaws_fit:  # (only deeper than the tips: the faces' V narrower there)
        return Rejected(grasp, arm, fixed_side, "jaws", "the open jaws do not clear it at that depth")
    key = (arm, grasp.opening)
    if key not in iks:
        iks[key] = ArmIK(model, arm, capture.arms, capture.head_ticks, grasp.opening, settings.margin)
    return _evaluate(model, claw, iks[key], settings, timed, obstacles=obstacles, primitive=primitive)


@dataclass(frozen=True)
class FirstFeasible:
    """`first_feasible`'s result.

    The grasp found (None: none in its tries), how many were tried, why the others failed (by stage), how long it took.
    """

    candidate: GraspCandidate | None
    tried: int
    rejected_by_stage: dict[str, int]
    seconds: float


def first_feasible(
    target: Object | Primitive,
    model: XleRobotModel,
    capture: Capture,
    rng: np.random.Generator,
    arms: Sequence[str] = ("right", "left"),
    where: Ranges | None = None,
    spacing: Spacing | None = None,
    part: str | None = None,
    settings: GraspSettings | None = None,
    obstacles: Sequence[Surface] | None = None,
    depths: Sequence[float] | None = None,
    max_tries: int = 300,
    accept: Callable[[GraspCandidate], bool] | None = None,
) -> FirstFeasible:
    """A feasible grasp drawn at random: the claw targets shuffled, each through every filter until one passes.

    The first to pass in a random order is a uniform draw from the feasible ones (`grasp_candidates` runs them all
    and ranks them: what the harness needs, and ~400 times the work on bowl1's jar, 377 of 5616 passing). For the
    simulation's starts (`docs/sim2real_grasp.md`). `depths`: each target tried at each of them, the (target, depth)
    pairs shuffled together (None: the settings'): a target is taken at whichever depth passes first, the others
    still tried later in the shuffle. `max_tries`: then None (a placement nothing reaches: draw another).
    `accept`: a last stage, after every filter here (the caller's: the simulation's grasp, `sim2real`), "accept" in
    `rejected_by_stage`; a target it refuses is passed over like any other.
    """
    settings = GraspSettings() if settings is None else settings
    primitive = target.part(part) if isinstance(target, Object) else target
    grasps = primitive.antipodal_grasps(where, spacing)
    obstacles = tuple(_default_obstacles(primitive) if obstacles is None else obstacles)
    at = [settings.depth] if depths is None else [float(d) for d in depths]
    targets = [(arm, k, side, d) for arm in arms for k in range(len(grasps)) for side in (0, 1) for d in at]
    order = rng.permutation(len(targets))
    timed, iks, stages = _Timer(), {}, Counter()
    started = time.perf_counter()
    for tried, index in enumerate(order[:max_tries], start=1):
        arm, k, side, depth = targets[index]
        tried_settings = settings if depth == settings.depth else replace(settings, depth=depth)
        result = _evaluate_target(
            model, primitive, grasps[k], arm, side, tried_settings, capture, iks, timed, obstacles
        )
        if isinstance(result, GraspCandidate):
            if accept is None or accept(result):
                return FirstFeasible(result, tried, dict(stages), time.perf_counter() - started)
            stages["accept"] += 1
            continue
        stages[result.stage] += 1
    return FirstFeasible(None, min(len(order), max_tries), dict(stages), time.perf_counter() - started)


def grasp_candidates(
    target: Object | Primitive,
    model: XleRobotModel,
    capture: Capture,
    arms: Sequence[str] = ("right", "left"),
    where: Ranges | None = None,
    spacing: Spacing | None = None,
    n: int = 5,
    part: str | None = None,
    settings: GraspSettings | None = None,
    obstacles: Sequence[Surface] | None = None,
) -> GraspCandidates:
    """Generate -> claw -> filters -> rank (`docs/object_grasping.md`, section 3; the stages: the module docstring).

    `target` is an `Object` (grasped by its `part`: the only one if not given) or a primitive: only what may be
    grasped. For each of its grasps in `where`, each arm and each contact on the fixed jaw. The attack point is solved
    from the arm's pose in `capture`, the grasp from the attack point's joints. The best `n` are named A, B, ...; every rejected one is kept, with its stage.
    `obstacles`: what the wrist camera keeps `camera_clearance` from (the table first); None: a level plane at the
    target's lower end (it stands on the table).
    """
    settings = GraspSettings() if settings is None else settings
    primitive = target.part(part) if isinstance(target, Object) else target
    grasps = primitive.antipodal_grasps(where, spacing)
    obstacles = _default_obstacles(primitive) if obstacles is None else obstacles
    rejected: list[Rejected] = []
    passed: list[GraspCandidate] = []
    timed = _Timer()
    tried = 0
    iks: dict[tuple[str, float], ArmIK] = {}
    for arm in arms:
        for grasp in grasps:
            for fixed_side in (0, 1):
                tried += 1
                result = _evaluate_target(
                    model, primitive, grasp, arm, fixed_side, settings, capture, iks, timed, tuple(obstacles)
                )
                if isinstance(result, GraspCandidate):
                    passed.append(result)
                else:
                    rejected.append(result)
    label = target.label if isinstance(target, Object) else ""
    return GraspCandidates(
        _ranked(passed, n), tried, len(passed), tuple(rejected), timed.timing(), settings, capture, label
    )


STAGE_UP_M = 0.06  # the staging pose: this far above the attack point (bowl1: 6 cm, every pick landed)
PICK_GRIP = {"start": "open", "pick": "close"}  # `execute.prepare_execution`'s grip for a whole `plan_pick`


def _as_poses(
    model: XleRobotModel,
    claw: ClawTarget,
    capture: Capture,
    labels: Sequence[str],
    targets: Sequence[np.ndarray],
    solutions: Sequence[FrameSolution],
    settings: GraspSettings,
) -> Poses:
    """Solved frames as the harness's `Poses` (the solve's frames and joints, its achieved angles as targets)."""
    ik = ArmIK(model, claw.arm, capture.arms, capture.head_ticks, claw.grasp.opening, claw.margin)
    angles = [angles_of(solution.root_from_pinch[:3, :3]) for solution in solutions]
    nan = np.full(len(labels), np.nan)
    return Poses(
        claw.arm,
        capture,
        np.array([ik.tool(target) for target in targets]),
        tuple(labels),
        np.array([s.root_from_pinch for s in solutions]),
        tuple(s.joints for s in solutions),
        np.array([s.position_error_m for s in solutions]),
        np.zeros(len(labels)),  # the pitch is not held apart from the rest here: in the rotation's bounds
        nan,
        nan,
        tuple(s.at_limit for s in solutions),
        np.array([s.ok for s in solutions], dtype=bool),
        tuple(angles),
        claw.grasp.opening,
        (settings.position_bound_m, ROTATION_SCALES_DEG[2], ROLL_TOL_DEG),
        ik.offset,
        ik.open_jaw,
        ik.held,
        claw.margin,
    )


def candidates_as_poses(model: XleRobotModel, candidates: GraspCandidates) -> Poses:
    """Each candidate's grasp as one pose labeled with its name (A, B, ...), to draw (`render`); the first's arm."""
    if not candidates.candidates or candidates.capture is None:
        raise ValueError("no candidates to draw")
    first = candidates.candidates[0]
    return _as_poses(
        model,
        first.claw,
        candidates.capture,
        [c.name for c in candidates.candidates],
        [c.claw.root_from_pinch for c in candidates.candidates],
        [c.grasp for c in candidates.candidates],
        first.settings,
    )


def plan_pick(
    model: XleRobotModel,
    candidate: GraspCandidate,
    capture: Capture,
    stage_up: float = STAGE_UP_M,
    lift: float = LIFT_M,
) -> Poses:
    """A chosen candidate as the poses of a pick: "stage", "pick.back", "pick", "lift".

    "stage" is `stage_up` above the attack point, "pick.back" the attack point, "pick" the grasp (both as
    `grasp_candidates` solved them), "lift" `lift` above the grasp; all at the grasp's orientation, solved with the
    candidate's scales and settings' bounds, from the pose before. The result is the harness's `Poses`, so
    `execute.prepare_execution` runs it as any other (its steps re-solved along the straight lines between them) with
    `PICK_GRIP`, or a part of it (`reach.select_poses`: `protocols.md` rule 1 stops at "pick.back"). A pose the arm
    does not meet has `ok` False, and `prepare_execution` refuses it.
    """
    claw = candidate.claw
    ik = ArmIK(model, claw.arm, capture.arms, capture.head_ticks, claw.grasp.opening, claw.margin)

    def raised(frame: np.ndarray, height: float) -> np.ndarray:
        moved = frame.copy()
        moved[2, 3] += height
        return moved

    stage_target = raised(claw.root_from_attack, stage_up)
    stage = _solve(ik, claw, stage_target, candidate.settings, init=candidate.attack.x)
    lift_target = raised(claw.root_from_pinch, lift)
    lifted = _solve(ik, claw, lift_target, candidate.settings, init=candidate.grasp.x)
    return _as_poses(
        model,
        claw,
        capture,
        ("stage", "pick.back", "pick", "lift"),
        [stage_target, claw.root_from_attack, claw.root_from_pinch, lift_target],
        [stage, candidate.attack, candidate.grasp, lifted],
        candidate.settings,
    )
