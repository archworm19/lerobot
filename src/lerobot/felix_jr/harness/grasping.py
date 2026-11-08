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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel

from .objects import Capture, Object, Poses, angles_of
from .primitives import AntipodalGrasp, Cylinder, Primitive, Ranges, Spacing
from .reach import OPEN_CLEARANCE_M, POSITION_TOL_M, ROLL_TOL_DEG, SQUEEZE_M, ArmIK, FrameSolution

SAG_LIFT_M = 0.01  # grasps aimed this much higher than the object (`GraspSettings.sag_lift`)
ATTACK_M = 0.05  # the attack point: this far back along the approach from the grasp (`protocols.md`, rule 1)
MARGIN_M = 0.008  # the fixed jaw this far clear of its contact (bowl1: 0.8-1.3 cm, every grasp landed)
MIN_TIP_APART_M = 1e-3  # the tip-to-tip direction of jaws this little apart, for a grasp of no opening
AXIS_NAMES = ("closing line", "approach", "hinge")  # the grasp's axes (`ClawTarget.axes_root`)
# the solve's preferences: how much a turn about each axis costs (a turn of its scale costs as much as a position error
# of `POSITION_TOL_M`): the closing line (keeps the contacts: the arm's missing sixth joint gives way here), the
# approach (tilts the closing line off the object's normal: bowl1's jar at roll -60 the worst), the hinge (slides the
# contacts along the approach). Not what is accepted: `GraspSettings.rotation_bounds_deg`.
ROTATION_SCALES_DEG = (20.0, 5.0, 10.0)
STAGES = ("jaws", "attack point", "grasp")  # in order: cheap first, each after its needs


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
    attack: float = ATTACK_M  # the attack point this far back along the approach
    # the grasp and its attack point aimed this much higher than the object: the arm lands low (its sag; bowl6: the
    # elbow stopping 1.2-1.3 deg short under load), and the errors are not alike: too high grips higher on the
    # object, too low hits what it stands on
    sag_lift: float = SAG_LIFT_M

    def describe(self) -> str:
        """The settings in a line."""
        bounds = ", ".join(
            f"{name} {b:g}" for name, b in zip(AXIS_NAMES, self.rotation_bounds_deg, strict=True)
        )
        return (
            f"accepted: turns up to ({bounds}) deg, the grasp point within {_cm(self.position_bound_m)} cm; "
            f"ranked by the antipodal error in {self.rank_step_deg:g} deg steps, then the joint margin; fixed jaw "
            f"{_cm(self.margin)} cm clear, attack point {_cm(self.attack)} cm back, aimed {_cm(self.sag_lift)} cm high "
            "against the arm's sag"
        )


@dataclass(frozen=True, eq=False)
class ClawTarget:
    """Where the gripper must be for one antipodal grasp, one contact on the fixed jaw.

    `root_from_pinch` is the pinch frame at the grasp, jaws open; `root_from_attack` the same pinch frame `attack` back
    along the approach. `aim_pinch` is the point of the pinch frame put on the grasp's center (with the margin),
    `held_pinch` where the center is once the jaws close (`XleRobotModel.grasp_offset` with and without the margin).
    `open_apart` / `close_apart` are how far apart the tips open for the approach and are told to close to (m).
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
) -> ClawTarget:
    """The claw target for `grasp`: `arm`'s fixed tip on contact `fixed_side`, the moving tip on the other.

    The fixed tip stands `margin` clear of its contact; the moving tip, at the grasp's opening, is on the other; the
    jaws come in along the grasp's approach, from the attack point `attack` back along it.

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

    aim_pinch = model.grasp_offset(arm, grasp.opening, margin)
    root_from_pinch = np.eye(4)
    root_from_pinch[:3, :3] = rot_root_from_pinch
    root_from_pinch[:3, 3] = grasp.center_root - rot_root_from_pinch @ aim_pinch
    root_from_attack = root_from_pinch.copy()
    root_from_attack[:3, 3] -= attack * a_root
    widest = 2 * model.jaw_radius(arm)
    return ClawTarget(
        grasp,
        arm,
        fixed_side,
        root_from_pinch,
        root_from_attack,
        aim_pinch,
        model.grasp_offset(arm, grasp.opening),
        margin,
        attack,
        min(grasp.opening + margin + OPEN_CLEARANCE_M, widest),
        max(grasp.opening - SQUEEZE_M, 0.0),
    )


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

    @property
    def antipodal_error_deg(self) -> float:
        """How far (deg) the solved grasp is from a true antipodal grip, the larger of its two turns that matter.

        About the approach: the closing line off the object's normal. About the hinge: the contacts slid along the
        approach. (About the closing line only tilts the approach: harmless to the grip.)
        """
        return float(max(self.grasp.rotation_error_deg[1], self.grasp.rotation_error_deg[2]))

    def rank_key(self) -> tuple[int, float, float]:
        """The ranking (lower first): the antipodal error in the settings' steps, the joint margin, the solves' cost."""
        step = int(self.antipodal_error_deg / self.settings.rank_step_deg + 1e-9)
        return (step, -self.limit_margin_deg, self.cost)

    def describe(self) -> str:
        """The grasp in the object's terms, the jaws, the solved gripper pose (root frame), how antipodal, joints."""
        pitch, heading, roll = angles_of(self.grasp.root_from_pinch[:3, :3])
        closing, approach, hinge = self.grasp.rotation_error_deg
        return (
            f"{self.name}: {self.claw.arm} arm, {self.claw.grasp.describe()}; fixed jaw on contact "
            f"{self.claw.fixed_side}, {_cm(self.claw.margin)} cm clear; gripper at pitch {pitch:.0f}, heading "
            f"{heading:.0f}, roll {roll:.0f} (root frame); antipodal error {self.antipodal_error_deg:.0f} deg (about the "
            f"approach {approach:.0f}, about the hinge {hinge:.0f}), approach tilted {closing:.0f} deg; joints >= "
            f"{self.limit_margin_deg:.0f} deg from their bounds; off the target by "
            f"{_cm(self.grasp.position_error_m)} cm"
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
    capture: Capture | None = None  # what the arm's start and the camera were (`plan_pick`, `refine` use it)
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


def _evaluate(
    model: XleRobotModel,
    claw: ClawTarget,
    ik: ArmIK,
    settings: GraspSettings,
    timed: _Timer,
    init: np.ndarray | None = None,
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

    cost = sum(
        (s.position_error_m / POSITION_TOL_M) ** 2
        + float(np.sum((np.asarray(s.rotation_error_deg) / np.asarray(ROTATION_SCALES_DEG)) ** 2))
        for s in (at_attack, at_grasp)
    )
    return GraspCandidate(
        "", claw, at_attack, at_grasp, _limit_margin_deg(ik, at_attack, at_grasp), cost, settings
    )


def _ranked(passed: Sequence[GraspCandidate], n: int) -> tuple[GraspCandidate, ...]:
    ranked = sorted(passed, key=GraspCandidate.rank_key)  # stable: ties keep the generation order
    return tuple(replace(c, name=chr(ord("A") + i)) for i, c in enumerate(ranked[: min(n, 26)]))


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
) -> GraspCandidates:
    """Generate -> claw -> filters -> rank (`docs/object_grasping.md`, section 3; the stages: the module docstring).

    `target` is an `Object` (grasped by its `part`: the only one if not given) or a primitive: only what may be
    grasped. For each of its grasps in `where`, each arm and each contact on the fixed jaw. The attack point is solved
    from the arm's pose in `capture`, the grasp from the attack point's joints. The best `n` are named A, B, ...; every rejected one is kept, with its stage.
    """
    settings = GraspSettings() if settings is None else settings
    primitive = target.part(part) if isinstance(target, Object) else target
    grasps = primitive.antipodal_grasps(where, spacing)
    rejected: list[Rejected] = []
    passed: list[GraspCandidate] = []
    timed = _Timer()
    tried = 0
    for arm in arms:
        widest = widest_apart(model, arm)
        iks: dict[float, ArmIK] = {}
        for grasp in grasps:
            for fixed_side in (0, 1):
                tried += 1
                start = time.perf_counter()
                too_wide = grasp.opening + settings.margin > widest
                timed("jaws", start)
                if too_wide:
                    rejected.append(Rejected(grasp, arm, fixed_side, "jaws", "too wide for the jaws"))
                    continue
                claw = _shifted(
                    to_claw(grasp, model, arm, fixed_side, settings.margin, settings.attack),
                    np.array([0.0, 0.0, settings.sag_lift]),
                )
                if grasp.opening not in iks:
                    iks[grasp.opening] = ArmIK(
                        model, arm, capture.arms, capture.head_ticks, grasp.opening, settings.margin
                    )
                result = _evaluate(model, claw, iks[grasp.opening], settings, timed)
                if isinstance(result, GraspCandidate):
                    passed.append(result)
                else:
                    rejected.append(result)
    label = target.label if isinstance(target, Object) else ""
    return GraspCandidates(
        _ranked(passed, n), tried, len(passed), tuple(rejected), timed.timing(), settings, capture, label
    )


def refine(
    model: XleRobotModel,
    candidate: GraspCandidate,
    target: Object | Primitive,
    capture: Capture,
    offset_root: np.ndarray | None = None,
    part: str | None = None,
    settings: GraspSettings | None = None,
) -> GraspCandidates:
    """The chosen grasp again, at the attack hold (`protocols.md` rule 1): from what the hold's capture shows.

    The same grasp (the same side, tilt and fixed jaw, at the same height on the object) on `target` as fitted again
    from the hold's view; solved from where the arm is now (`capture`'s joints: it is at the attack point); with its
    grasp and attack frames moved by `offset_root`, to cancel what the arm was measured to
    miss by at the hold (minus `arm_error(...).offset`: the arm sags the same way again on the short move in). The
    filters run again; the result is one candidate, "A", or none and why.
    """
    settings = candidate.settings if settings is None else settings
    primitive = target.part(part) if isinstance(target, Object) else target
    if not isinstance(primitive, Cylinder):
        raise ValueError(f"refine knows a cylinder's grasps, not a {type(primitive).__name__}'s")
    old = candidate.claw.grasp
    t = float(np.clip((old.center_root - primitive.bottom_root) @ primitive.axis_root, 0.0, primitive.length))
    where = {"t": (t, t), "phi": (old.params["phi"],) * 2, "psi": (old.params["psi"],) * 2}
    (grasp,) = primitive.antipodal_grasps(where)
    arm, timed = candidate.claw.arm, _Timer()
    claw = _shifted(
        to_claw(grasp, model, arm, candidate.claw.fixed_side, settings.margin, settings.attack),
        np.array([0.0, 0.0, settings.sag_lift]),
    )
    if offset_root is not None:
        claw = _shifted(claw, -np.asarray(offset_root, dtype=np.float64))
    ik = ArmIK(model, arm, capture.arms, capture.head_ticks, grasp.opening, settings.margin)
    result = _evaluate(model, claw, ik, settings, timed)
    passed = [result] if isinstance(result, GraspCandidate) else []
    rejected = (result,) if isinstance(result, Rejected) else ()
    label = target.label if isinstance(target, Object) else ""
    return GraspCandidates(
        _ranked(passed, 1), 1, len(passed), rejected, timed.timing(), settings, capture, label
    )


STAGE_UP_M = 0.06  # the staging pose: this far above the attack point (bowl1: 6 cm, every pick landed)
LIFT_M = 0.03  # the lift after the close (bowl1: 3 cm, then hold and look)
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
