"""From an object's antipodal grasps to the gripper: the claw mapping, the checks and the ranking.

The design is in `docs/object_grasping.md`, sections 2 and 3.

**The claw.** The pinch frame's origin is the fixed jaw's tip; the moving jaw turns about the hinge (the pinch y) and
its tip travels on an arc, so the line from the fixed tip to the moving tip is not the pinch -x: it tilts toward the
fingertips by an angle that depends on the opening (about 8 deg at 4 cm, 19 deg at 1 cm on this robot). `to_claw`
lines that tip-to-tip line up with the grasp's closing line, so both tips land on the contacts, and turns the hinge
square to the closing and the approach. The gripper axis (the pinch z) then leans off the approach by that tilt.

**Checks and ranking** (`grasp_candidates`). Each claw target goes through stages, and the first it fails is recorded
with the reason (`Rejected`): the jaws open wide enough; the gripper axis is near the arm's vertical plane (the
heading follows the shoulder pan, so a target far off it cannot be reached: skipped before solving); the arm reaches
the attack point and then the grasp; the head camera sees the jaws at both (no wrist camera plate in the way:
`protocols.md` rule 1 looks there). The rest are ranked in tiers, each only breaking the previous one's ties: jaws
not edge-on to the camera, then no joint near its bound, then how squarely the camera sees the jaws, then how closely
the arm meets the target.

**Which jaw where.** The fixed jaw does not move: closing sweeps the object onto it. So each antipodal grasp gives two
claw targets, one per contact taking the fixed jaw (`fixed_side` 0 or 1: `AntipodalGrasp.contacts_root`). A `margin`
aims the fixed tip that much clear of its contact, toward the outside; the jaws open wider, and closing pushes the
object up to `margin` onto the fixed jaw (`reach`'s module docstring: the same scheme as `solve_pose`).
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel

from .objects import EDGE_ON_VISIBILITY, Capture, angles_of
from .primitives import AntipodalGrasp, Primitive, Ranges, Spacing
from .reach import OPEN_CLEARANCE_M, POSITION_TOL_M, SQUEEZE_M, ArmIK, FrameSolution

ATTACK_M = 0.05  # the attack point: this far back along the approach from the grasp (`protocols.md`, rule 1)
MARGIN_M = 0.008  # the fixed jaw this far clear of its contact (bowl1: 0.8-1.3 cm, every grasp landed)
MIN_TIP_APART_M = 1e-3  # the tip-to-tip direction of jaws this little apart, for a grasp of no opening
# how far the solved gripper may turn off the target, about each of the grasp's axes (`ClawTarget.axes_root`): the
# closing line (keeps the contacts: the arm's missing sixth joint gives way here), the approach (rolls the closing line
# off the object: bowl1's jar at roll -60), the hinge (moves the contacts along the approach)
ROTATION_TOLS_DEG = (20.0, 5.0, 10.0)
AXIS_NAMES = ("closing line", "approach", "hinge")
OFF_PLANE_MAX_DEG = (
    25.0  # the gripper axis this far out of the arm's plane: not solved for (passes reach 18: the jar, a pen)
)
NEAR_LIMIT_DEG = 5.0  # a joint closer than this to its bound ranks lower
VISIBILITY_STEP = 0.1  # visibilities this close rank as equal (the next tier decides)
STAGES = ("jaws", "arm plane", "attack point", "grasp", "camera")  # in the order a claw target meets them


def _cm(value: float) -> str:
    return f"{value * 100:.1f}"


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
    tip_pinch = model.grasp_offset(arm, max(grasp.opening, MIN_TIP_APART_M))
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
    """One grasp that passed every check, with the arm's solved poses at the attack point and at the grasp."""

    name: str
    claw: ClawTarget
    attack: FrameSolution
    grasp: FrameSolution
    visibility: (
        float  # the jaws at the grasp, from the capture's camera (`XleRobotModel.view_of`): near 0 edge-on
    )
    limit_margin_deg: float  # the closest any joint comes to its bound, over both poses
    cost: float  # the solves' summed squared residuals, in tolerances: lower is closer to the target

    @property
    def edge_on(self) -> bool:
        """The camera sees the jaws nearly edge-on: what is between them is hidden."""
        return abs(self.visibility) < EDGE_ON_VISIBILITY

    @property
    def near_limit(self) -> bool:
        """A joint is within `NEAR_LIMIT_DEG` of its bound at the attack point or the grasp."""
        return self.limit_margin_deg < NEAR_LIMIT_DEG

    def rank_key(self) -> tuple[bool, bool, float, float]:
        """The ranking tiers (lower first): edge-on, near a joint limit, how squarely seen, the solves' cost."""
        seen = int(abs(self.visibility) / VISIBILITY_STEP + 1e-9)  # the step it is in
        return (self.edge_on, self.near_limit, -seen, self.cost)

    def describe(self) -> str:
        """The grasp in the object's terms, the jaws, the solved gripper pose (root frame), camera and joints."""
        pitch, heading, roll = angles_of(self.grasp.root_from_pinch[:3, :3])
        off = ", ".join(f"{e:.0f}" for e in self.grasp.rotation_error_deg)
        camera = "edge-on" if self.edge_on else "seen"
        return (
            f"{self.name}: {self.claw.arm} arm, {self.claw.grasp.describe()}; fixed jaw on contact "
            f"{self.claw.fixed_side}, {_cm(self.claw.margin)} cm clear; gripper at pitch {pitch:.0f}, heading "
            f"{heading:.0f}, roll {roll:.0f} (root frame); camera: jaws {camera} ({self.visibility:+.2f}); joints "
            f">= {self.limit_margin_deg:.0f} deg from their bounds; off the target by "
            f"{_cm(self.grasp.position_error_m)} cm and ({off}) deg about the {', '.join(AXIS_NAMES)}"
        )


@dataclass(frozen=True, eq=False)
class Rejected:
    """One claw target that failed a check: the stage (`STAGES`), why, and as far as it got."""

    grasp: AntipodalGrasp
    arm: str
    fixed_side: int
    stage: str
    reason: str
    claw: ClawTarget | None = None  # None if the jaws cannot open for it
    attack: FrameSolution | None = None  # the solves, where it got that far
    at_grasp: FrameSolution | None = None

    def describe(self) -> str:
        """The arm, the grasp, the jaw and the reason."""
        return (
            f"{self.arm} arm, {self.grasp.describe()}, fixed jaw on contact {self.fixed_side}: {self.reason}"
        )


@dataclass(frozen=True, eq=False)
class GraspCandidates:
    """The best grasps of a primitive, ranked (named A, B, ...), and every claw target rejected, with why."""

    candidates: tuple[GraspCandidate, ...]  # the best `n`
    tried: int  # claw targets: grasps x arms x the two fixed sides
    passed: int
    rejected: tuple[Rejected, ...]

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
        """Up to `n` of the rejected that got furthest through the checks (the last stage any reached)."""
        if not self.rejected:
            return ()
        last = max(STAGES.index(r.stage) for r in self.rejected)
        return tuple(r for r in self.rejected if STAGES.index(r.stage) == last)[:n]

    def describe(self) -> str:
        """The candidates, the rejected by reason, and the nearest misses."""
        rejected = "; ".join(f"{count} {reason}" for reason, count in self.rejected_by_reason().items())
        head = f"{self.passed} of {self.tried} claw targets pass, best {len(self.candidates)}"
        rows = [c.describe() for c in self.candidates]
        misses = [f"  {r.describe()}" for r in self.nearest_misses()]
        return "\n".join(
            [
                head + ":",
                *rows,
                f"rejected: {rejected or 'none'}",
                *(["nearest misses:", *misses] if misses else []),
            ]
        )


def _rejection(solution: FrameSolution, where: str, position_tol: float) -> str | None:
    """Why a solve fails its target (None if it does not), in words without numbers, for counting."""
    if solution.ok:
        return None
    if solution.position_error_m > position_tol:
        return f"{where} out of reach"
    worst = int(np.argmax(np.asarray(solution.rotation_error_deg) / np.asarray(ROTATION_TOLS_DEG)))
    return f"{where}: the arm cannot turn the gripper about the {AXIS_NAMES[worst]}"


def off_arm_plane_deg(model: XleRobotModel, ik: ArmIK, root_from_pinch: np.ndarray) -> float:
    """How far (deg) a target's gripper axis is out of the arm's vertical plane (through the pan axis and the target).

    The arm keeps its gripper axis in that plane (five joints: the heading follows the pan), so a target far out of it
    cannot be reached. 0 for a vertical axis (any heading will do).
    """
    pan_root = model.frame_pose(f"{ik.arm}_shoulder_link", ik.q)[:3, 3]
    radial = (root_from_pinch[:3, 3] - pan_root)[:2]
    radial /= np.linalg.norm(radial)
    normal_root = np.array([-radial[1], radial[0], 0.0])
    return float(np.degrees(np.arcsin(min(abs(float(root_from_pinch[:3, 2] @ normal_root)), 1.0))))


def _camera(model: XleRobotModel, ik: ArmIK, solution: FrameSolution) -> tuple[float, tuple[str, ...]]:
    """How the capture's camera sees a solved pose's grasp point: the jaws' visibility, the plates hiding it."""
    q = ik.q.copy()
    q[ik.index] = solution.x
    return model.view_of(ik.arm, q, ik.tool(solution.root_from_pinch))


def _limit_margin_deg(ik: ArmIK, *solutions: FrameSolution) -> float:
    """The closest any joint of the solutions comes to its bound (deg)."""
    return float(min(np.degrees(np.minimum(s.x - ik.lower, ik.upper - s.x)).min() for s in solutions))


def grasp_candidates(
    primitive: Primitive,
    model: XleRobotModel,
    capture: Capture,
    arms: Sequence[str] = ("right", "left"),
    where: Ranges | None = None,
    spacing: Spacing | None = None,
    margin: float = MARGIN_M,
    attack: float = ATTACK_M,
    n: int = 5,
) -> GraspCandidates:
    """Generate -> claw -> checks -> rank (`docs/object_grasping.md`, section 3; the stages: the module docstring).

    For each of the primitive's grasps in `where`, each arm and each contact on the fixed jaw. The attack point is
    solved from the arm's pose in `capture`, the grasp from the attack point's joints (so the move in has no joint
    jump); the camera is the capture's head pose. The best `n` are named A, B, ...; every rejected one is kept.
    """
    grasps = primitive.antipodal_grasps(where, spacing)
    rejected: list[Rejected] = []
    passed: list[GraspCandidate] = []
    tried = 0
    for arm in arms:
        widest = widest_apart(model, arm)
        iks: dict[float, ArmIK] = {}
        for grasp in grasps:
            for fixed_side in (0, 1):
                tried += 1
                if grasp.opening + margin > widest:
                    rejected.append(Rejected(grasp, arm, fixed_side, "jaws", "too wide for the jaws"))
                    continue
                claw = to_claw(grasp, model, arm, fixed_side, margin, attack)
                if grasp.opening not in iks:
                    iks[grasp.opening] = ArmIK(
                        model, arm, capture.arms, capture.head_ticks, grasp.opening, margin
                    )
                ik = iks[grasp.opening]
                if off_arm_plane_deg(model, ik, claw.root_from_pinch) > OFF_PLANE_MAX_DEG:
                    off_plane = (
                        "the gripper axis is out of the arm's plane (the heading follows the shoulder)"
                    )
                    rejected.append(Rejected(grasp, arm, fixed_side, "arm plane", off_plane, claw))
                    continue
                at_attack = ik.solve_frame(claw.root_from_attack, claw.axes_root, ROTATION_TOLS_DEG)
                reason = _rejection(at_attack, "attack point", POSITION_TOL_M)
                if reason is not None:
                    rejected.append(Rejected(grasp, arm, fixed_side, "attack point", reason, claw, at_attack))
                    continue
                at_grasp = ik.solve_frame(
                    claw.root_from_pinch, claw.axes_root, ROTATION_TOLS_DEG, init=at_attack.x
                )
                reason = _rejection(at_grasp, "grasp", POSITION_TOL_M)
                if reason is not None:
                    rejected.append(
                        Rejected(grasp, arm, fixed_side, "grasp", reason, claw, at_attack, at_grasp)
                    )
                    continue
                visibility, hidden_at_grasp = _camera(model, ik, at_grasp)
                hidden = {
                    "attack point": _camera(model, ik, at_attack)[1],
                    "grasp": hidden_at_grasp,
                }
                for where_hidden, plates in hidden.items():
                    if plates:
                        reason = (
                            f"{where_hidden}: the {' and '.join(plates)} wrist camera plate hides the jaws from the "
                            "head camera"
                        )
                        rejected.append(
                            Rejected(grasp, arm, fixed_side, "camera", reason, claw, at_attack, at_grasp)
                        )
                        break
                else:
                    cost = sum(
                        (s.position_error_m / POSITION_TOL_M) ** 2
                        + float(
                            np.sum((np.asarray(s.rotation_error_deg) / np.asarray(ROTATION_TOLS_DEG)) ** 2)
                        )
                        for s in (at_attack, at_grasp)
                    )
                    margin_deg = _limit_margin_deg(ik, at_attack, at_grasp)
                    passed.append(GraspCandidate("", claw, at_attack, at_grasp, visibility, margin_deg, cost))
    ranked = sorted(passed, key=GraspCandidate.rank_key)  # stable: ties keep the generation order
    candidates = tuple(
        GraspCandidate(chr(ord("A") + i), c.claw, c.attack, c.grasp, c.visibility, c.limit_margin_deg, c.cost)
        for i, c in enumerate(ranked[: min(n, 26)])
    )
    return GraspCandidates(candidates, tried, len(passed), tuple(rejected))
