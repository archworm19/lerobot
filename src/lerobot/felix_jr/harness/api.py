"""The harness's tools, all in one place: what the VLM may call, and what each call can do to the world.

Every tool is a function here, registered with `@tool(effect=...)`; its docstring is written for the model and goes
into its prompt (`describe_tools`). The work is done elsewhere (`tools`, `render`, `reach`, `scan`, `execute`); these
are the entry points. A tool's first argument is the `Harness` running it, which the model never passes.

**Effects.** `SOFTWARE` (computes from what is already recorded), `SENSE` (reads the robot: a capture; nothing moves),
`MOVE_HEAD` (moves the head; the world stays as it was), `MOVE_ARM` (moves the arm, and maybe the objects it
touches: everything seen before is stale after it), `END` (ends the session: the robot goes to rest).

**The runner.** `Harness.call` runs every tool: it asks a person first when the policy says so (`SAFE`: before any
move of the head or arm, showing what will run), records the result in the workspace (the DAG of objects), marks
earlier results stale after a `MOVE_ARM`, and logs every call, refused ones and queries included (`calls`). Software
tools never wait for anyone.

**Snippets.** The model can also run Python snippets (`run_snippet`) in a namespace holding the tools (each going
through `Harness.call`), the recorded objects by id and by the names it gave them, and numpy. This is a guardrail,
not a boundary: a snippet could import the robot client and bypass the runner. Fine while a person is in the loop;
before unattended runs, the robot server must enforce approval itself (`docs/open_issues.md`).
"""

import dataclasses
import enum
import functools
import inspect
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel

from . import (
    contacts as _contacts,
    execute as _execute,
    grasping as _grasping,
    look as _look,
    reach as _reach,
    scan as _scan,
    sfm as _sfm,
    tools as _tools,
    wrist as _wrist,
    wrist_scan as _wrist_scan,
)
from .contacts import ContactPatches, HoldViews
from .objects import (
    ArmPath,
    Capture,
    Execution,
    Focus,
    Grid,
    HeldObject,
    Object,
    Points,
    Poses,
    TableEstimate,
    View,
)
from .render import contact_sheet as _contact_sheet, render as _render  # the package re-exports `render`
from .sfm import SfmResult
from .workspace import Workspace
from .wrist import Alignment, WristMove, WristPoint, WristView
from .wrist_scan import WristScan

if TYPE_CHECKING:
    from lerobot.felix_jr.server.client import FelixClient

# xle_calib_1008 with the right wrist camera turned 5.2 deg to fit its fingertips in a gripper sweep (bowl5), the
# fingers' 1 cm thickness accounted for (`JAW_THICKNESS_M`: the wrist camera sees their other face). The 7.8 deg of
# `xle_calib_1008_wrist.urdf` absorbed that thickness.
MODEL = Path(__file__).parents[1] / "urdfs" / "xle_calib_1008_wrist2.urdf"
ENDPOINT = "tcp://ikeabot.local:5555"


class Effect(enum.Enum):
    """What a tool can do to the robot and the world."""

    SOFTWARE = "software"  # computes from what is already recorded
    SENSE = "sense"  # reads the robot (a capture); nothing moves
    MOVE_HEAD = "move head"  # moves the head; the world stays as it was
    MOVE_ARM = "move arm"  # moves the arm and maybe what it touches: earlier results become stale
    END = "end"  # ends the session; the robot goes to rest


@dataclass(frozen=True)
class Tool:
    """A registered tool."""

    name: str
    fn: Callable[..., Any]
    effect: Effect
    records: bool  # is the result an object for the workspace? (queries return text and are only logged)
    deprecated: str = (
        ""  # what replaces it, if it is on its way out (shown in `describe_tools`, logged when called)
    )


TOOLS: dict[str, Tool] = {}


def tool(
    effect: Effect, records: bool = True, deprecated: str = ""
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register the decorated function as a tool with `effect`; `deprecated` says what replaces it, if anything."""

    def register(fn: Callable[..., Any]) -> Callable[..., Any]:
        if not fn.__doc__:
            raise ValueError(f"tool {fn.__name__} needs a docstring: it is the model's documentation")
        TOOLS[fn.__name__] = Tool(fn.__name__, fn, effect, records, deprecated)
        return fn

    return register


@dataclass(frozen=True)
class Policy:
    """Which effects need a person's approval before they run."""

    ask: frozenset[Effect]


SAFE = Policy(frozenset({Effect.MOVE_HEAD, Effect.MOVE_ARM}))  # the default
UNATTENDED = Policy(frozenset())  # for tests against fake hardware


class NotStartedError(ValueError):
    """A tool that moves refused its input before anything moved (it is stale, say): nothing becomes stale."""


class RefusedError(RuntimeError):
    """A person did not approve the call (or the session has ended)."""


def ask_in_terminal(tool: Tool, preview: str) -> bool:
    """Show what a call will do and ask whether to run it."""
    print(f"\n--- {tool.name} ({tool.effect.value}) ---\n{preview}")
    return input(f"run {tool.name}? [y/N] ").strip().lower() == "y"


def confirm_in_terminal(label: str) -> bool:
    """Ask before one move of an execution (`execute(..., step=True)`)."""
    return input(f"{label}: Enter to run, n to stop (the robot goes home) ").strip().lower() != "n"


class Harness:
    """Runs tools: approval, recording, staleness and the call log (see the module docstring)."""

    def __init__(
        self,
        ws: Workspace | None = None,
        policy: Policy = SAFE,
        ask: Callable[[Tool, str], bool] = ask_in_terminal,
        confirm: Callable[[str], bool] = confirm_in_terminal,
        endpoint: str = ENDPOINT,
        model_path: str | Path = MODEL,
        client: "FelixClient | None" = None,
        model: XleRobotModel | None = None,
    ):
        """A runner over `ws` (default: a new one). The model and the robot client are made when first needed."""
        self.ws = ws if ws is not None else Workspace()
        self.policy, self.ask, self.confirm = policy, ask, confirm
        self.endpoint, self.model_path = endpoint, Path(model_path)
        self._client, self._model = client, model
        self.names: dict[str, str] = {}  # the model's names for objects -> node ids
        self.calls: list[dict[str, Any]] = []  # every call, in order
        self.ended = False

    @property
    def model(self) -> XleRobotModel:
        """The calibrated robot model."""
        if self._model is None:
            self._model = XleRobotModel.load(self.model_path)
        return self._model

    @property
    def client(self) -> "FelixClient":
        """The robot client (connected on first use, and checked against the model: `XleRobotModel.check_robot`)."""
        if self._client is None:
            from lerobot.felix_jr.server.client import FelixClient

            client = FelixClient(self.endpoint)
            try:
                self.model.check_robot(client)
            except Exception:
                client.close()
                raise
            self._client = client
        return self._client

    # ---- running tools -------------------------------------------------------------------------------------

    def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Run tool `name` (see the module docstring); raises `RefusedError` if a person says no."""
        spec = TOOLS[name]
        entry: dict[str, Any] = {
            "tool": name,
            "effect": spec.effect.value,
            "args": [self.ws._summarize(a) for a in args],
            "kwargs": {k: self.ws._summarize(v) for k, v in kwargs.items()},
            "time": time.time(),
        }
        if spec.deprecated:
            entry["deprecated"] = spec.deprecated
        self.calls.append(entry)
        if self.ended:
            entry["outcome"] = "refused: the session has ended"
            raise RefusedError("the session has ended")
        if spec.effect in self.policy.ask:
            approved = self.ask(spec, self._preview(spec, args, kwargs))
            entry["approved"] = approved
            if not approved:
                entry["outcome"] = "refused"
                raise RefusedError(f"{name} was not approved")

        @functools.wraps(spec.fn)
        def run(*a: Any, **k: Any) -> Any:
            started = True
            try:
                return spec.fn(self, *a, **k)
            except NotStartedError:
                started = False  # refused before anything moved: what was seen is still fresh
                raise
            finally:
                if (
                    started and spec.effect is Effect.MOVE_ARM
                ):  # before the result is recorded: an after-capture is fresh
                    self.ws.moved()

        try:
            result = self.ws.call(run, *args, **kwargs) if spec.records else run(*args, **kwargs)
        except Exception as e:
            entry["outcome"] = f"error: {type(e).__name__}: {e}"
            stopped_at = getattr(e, "capture", None)  # a move that did not settle: what the camera saw there
            if isinstance(stopped_at, Capture):  # recorded after the arm moved, so it is the fresh one
                entry["result"] = [self.ws.add(stopped_at, name)]
                e.args = (f"{e} (capture: {entry['result'][0]})",)
            raise
        entry["outcome"] = "ok"
        ids = [self.ws.id_of(r) for r in (result if isinstance(result, (list, tuple)) else [result])]
        if any(ids):
            entry["result"] = [i for i in ids if i]
        return result

    def _preview(self, spec: Tool, args: Sequence[Any], kwargs: Mapping[str, Any]) -> str:
        """What a call will do, for the person asked to approve it."""
        rows = [
            f"{spec.name}("
            + ", ".join(
                [str(self.ws._summarize(a)) for a in args]
                + [f"{k}={self.ws._summarize(v)}" for k, v in kwargs.items()]
            )
            + ")"
        ]
        for value in [*args, *kwargs.values()]:
            if isinstance(value, Execution):
                rows += ["", *value.lines()]
            elif hasattr(value, "describe"):
                rows.append(f"{self.ws.id_of(value) or type(value).__name__}: {value.describe()}")
        return "\n".join(rows)

    # ---- snippets ------------------------------------------------------------------------------------------

    def namespace(self) -> dict[str, Any]:
        """What a snippet sees: the tools, the recorded objects by id and by name, numpy."""
        ns: dict[str, Any] = {"np": np}
        for name, spec in TOOLS.items():
            ns[name] = functools.wraps(spec.fn)(functools.partial(self.call, name))
        ns |= {node_id: node.obj for node_id, node in self.ws.nodes.items()}
        ns |= {name: self.ws[node_id] for name, node_id in self.names.items()}
        return ns

    def run_snippet(self, code: str, extra: Mapping[str, Any] | None = None) -> list[str]:
        """Run `code` in the namespace (plus `extra`); keep the names it gives to recorded objects.

        Returns the ids of the nodes it made.
        """
        before_nodes = len(self.ws.nodes)
        ns = self.namespace() | dict(extra or {})
        before = dict(ns)
        exec(code, ns)  # nosec B102: the model's snippet; a guardrail, not a sandbox (module docstring)
        for name, value in ns.items():
            node_id = self.ws.id_of(value)
            if node_id is not None and not name.startswith("_") and before.get(name) is not value:
                self.names[name] = node_id
        return list(self.ws.nodes)[before_nodes:]

    # ---- persistence ---------------------------------------------------------------------------------------

    def state(self) -> dict[str, Any]:
        """What to pickle: the workspace, the names and the call log (not the client or the model)."""
        return {"ws": self.ws, "names": self.names, "calls": self.calls, "ended": self.ended}

    def restore(self, state: Mapping[str, Any]) -> None:
        """Load `state()` output."""
        self.ws, self.names = state["ws"], dict(state["names"])
        self.calls, self.ended = list(state["calls"]), bool(state["ended"])


def describe_tools() -> str:
    """Every tool's signature (without the runner), effect and docstring: the model's tool documentation."""
    blocks = []
    for name, spec in TOOLS.items():
        params = list(inspect.signature(spec.fn).parameters.values())[1:]
        signature = re.sub(r"\b(?:[a-z_]\w*\.)+(?=\w)", "", f"{name}({', '.join(str(p) for p in params)})")
        note = f"DEPRECATED: {spec.deprecated}\n" if spec.deprecated else ""
        blocks.append(f"{signature}  [{spec.effect.value}]\n{note}{inspect.cleandoc(spec.fn.__doc__ or '')}")
    return "\n\n".join(blocks)


# ---- software: views, the table, grids, points, paths, pictures ---------------------------------------------------


@tool(Effect.SOFTWARE)
def full_view(h: Harness, capture: Capture, scale: float = 1.0) -> View:
    """A view of the whole capture, `scale` image pixels per capture pixel."""
    return _tools.full_view(capture, scale)


@tool(Effect.SOFTWARE)
def zoom(h: Harness, view: View, rect: Sequence[float], scale: float | None = None) -> View:
    """A crop of `view`: `rect` = (u, v, width, height) in `view`'s pixels, scaled to fill the view by default."""
    return _tools.zoom(view, rect, scale)


@tool(Effect.SOFTWARE)
def focus(h: Harness, *views: View, reason: str = "") -> Focus:
    """Choose the views to work in (the first is primary) and say why. Nothing moves."""
    return _tools.focus(*views, reason=reason)


@tool(Effect.SOFTWARE)
def table_height(h: Harness, capture: Capture, min_h: float | None = None) -> TableEstimate:
    """The table's height (root-frame z, m) from the capture's depth, with how flat and level it is."""
    return _tools.table_height(capture, min_h=min_h)


@tool(Effect.SOFTWARE)
def make_grid(
    h: Harness,
    view: View,
    height: float | TableEstimate,
    cell: float = 0.02,
    extent: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> Grid:
    """A grid of `cell`-meter squares on the plane z = `height` (or a table estimate), over what `view` shows.

    Drawn with `render`, it labels root-frame x and y in cm: read positions off it. `extent` = ((x0, x1), (y0, y1))
    in m overrides what the view shows.
    """
    return _tools.make_grid(view, height, cell, extent)


@tool(Effect.SOFTWARE)
def locate_pixels(h: Harness, view: View, uv: np.ndarray, labels: Sequence[str] | None = None) -> Points:
    """Root-frame points (m) at `view` pixels `uv` (N, 2), from the depth; NaN where there is none."""
    return _tools.locate_pixels(view, uv, labels)


@tool(Effect.SOFTWARE)
def locate_grid(h: Harness, grid: Grid, xy: np.ndarray, labels: Sequence[str] | None = None) -> Points:
    """Root-frame points on the grid's plane at (x, y) (N, 2), meters. No depth involved."""
    return _tools.locate_grid(grid, xy, labels)


@tool(Effect.SOFTWARE)
def render(h: Harness, view: View, *overlays: Any, labels: bool = True) -> np.ndarray:
    """The view's image with the overlays you choose drawn on it.

    Overlays: `Grid`, `Points`, `TableEstimate`, `Poses`, `ArmPath`, `Object`, `GraspCandidates`.

    An `Object` is drawn as its cylinder (orange: the two end circles and lines along the side), labeled with its
    width; `GraspCandidates` as the gripper at each candidate's grasp, labeled A, B, ... (as `Poses`, below).

    `Poses` are drawn as the path through them (in order) and a gripper at each, open as for the approach: the fixed
    finger pink, the moving one cyan, turned about the hinge behind them (a claw, so the gap is a wedge, widest at the
    tips; a roll of the wrist shows as the colors swapping sides), a ring at the grasp point (green if reachable, red
    with an X if not) and the label with its xyz in cm. `ArmPath` draws the arm's body
    (from `arm_path`). The text of `Poses` gives each pose's wrist roll; `held_object` says what a roll does to what
    the gripper carries. Pick what helps: poses alone, poses with the arm, a grid to read positions;
    `labels=False` drops the text. Nothing is checked against the depth: judge clearance from the images.
    """
    drawn = [
        _grasping.candidates_as_poses(h.model, o) if isinstance(o, _grasping.GraspCandidates) else o
        for o in overlays
    ]
    return _render(view, *drawn, labels=labels)


@tool(Effect.SOFTWARE)
def contact_sheet(
    h: Harness, views: Sequence[View], *overlays: Any, labels: Sequence[str] | None = None
) -> np.ndarray:
    """Several views rendered side by side in one image, each with the overlays and its label."""
    return _contact_sheet(list(views), *overlays, labels=labels)


# ---- software: where the arm can put its gripper ----------------------------------------------------------------


@tool(
    Effect.SOFTWARE,
    deprecated="to grasp, use fit_cylinder, grasp_candidates and plan_pick; still for a carry or a place, until plan_place (docs/object_grasping.md, phase D)",
)
def solve_pose(
    h: Harness,
    arm: str,
    capture: Capture,
    position: Sequence[float],
    pitch_deg: float,
    heading_deg: float | None = None,
    roll_deg: float | None = 0.0,
    width: float = 0.0,
    position_tol: float = 0.005,
    heading_tol_deg: float = 10.0,
    roll_tol_deg: float = 10.0,
    init: Any = None,
    label: str = "1",
    margin: float = 0.0,
) -> Poses:
    """Solve the joints that put `arm`'s gripper at a pose: one constrained optimization, from `init`.

    The target: `position` (root frame, m) for the grasp point, where the center of an object `width` m wide is held:
    midway between the fingertips, with the jaws opened until the tips are `width` apart (the jaws close on the
    object at the tips; 0 for none). Only the moving jaw opens: the fixed fingertip touches the object's side, so
    an aim error toward it misses the object. `margin` (m, e.g. 0.01) aims that much toward the moving jaw instead:
    the fixed jaw clears the object by `margin`, the jaws open as much wider, and closing sweeps the object onto the
    fixed jaw (pushing it up to `margin` sideways: fine for a leaning pen, not for something that tips over); `pitch_deg`, the gripper axis above horizontal (-90 straight down, 0 horizontal);
    `heading_deg`, the axis' direction about the vertical (deg from +x toward +y; None: free); `roll_deg`, the jaws
    about the axis (0: jaws horizontal; 180: the jaws swapped; None: free, but then it can land anywhere: for a path,
    hold the roll the first pose got, or the wrist swings between poses). The pitch is held; position, heading and
    roll are met as closely as the arm allows, each weighted by its tolerance. On this arm the heading follows from
    the position (the gripper stays in the vertical plane through the shoulder), so leave it free unless it matters.
    `init`: where the solver starts (joints in LeRobot degrees, or `Poses` to continue from its last pose; default the
    arm's pose in the capture). It decides the branch (elbow up or down). One local solve, no retries: if a pose
    misses, the description says by how much and which joints are at a limit; try another pitch, start or tolerance.
    """
    return _reach.solve_pose(
        h.model,
        arm,
        capture,
        np.asarray(position, dtype=np.float64),
        pitch_deg,
        heading_deg,
        roll_deg,
        width,
        position_tol,
        heading_tol_deg,
        roll_tol_deg,
        init,
        label,
        margin,
    )


@tool(
    Effect.SOFTWARE,
    deprecated="plan_pick gives a pick's poses whole; still for joining a carry or a place, until plan_place",
)
def join_poses(h: Harness, *poses: Poses) -> Poses:
    """Solved poses as one path, in the order given (same arm, capture, object width and margin; labels unique).

    Draw it with `render`, run it with `prepare_execution`.
    """
    return _reach.join_poses(*poses)


@tool(Effect.SOFTWARE)
def select_poses(h: Harness, poses: Poses, *labels: str) -> Poses:
    """Some of `poses`, by label, in the order given: e.g. one pose of a path, to go only there."""
    return _reach.select_poses(poses, *labels)


@tool(Effect.SOFTWARE, deprecated="plan_pick gives the attack point (pick.back) and the lift")
def approach(
    h: Harness, pose: Poses, back: float = 0.05, up: float = 0.05, order: Sequence[str] = ("back", "at", "up")
) -> Poses:
    """A solved pose with its neighbors: `back` m back along the gripper axis and `up` m above, same orientation.

    In `order`. For a pose labeled "pick": "pick.back", "pick", "pick.up". ("back", "at", "up") goes in along the axis and leaves
    upward (a pick); ("up", "at", "back") comes down and leaves along the axis (a place).
    """
    return _reach.approach(h.model, pose, back, up, order)


@tool(Effect.SOFTWARE)
def prepare_execution(
    h: Harness, poses: Poses, grip: Mapping[str, str] | None = None, pose_tol_deg: float = 2.5
) -> Execution:
    """The exact moves `execute` would send to go through `poses` in order, solved every ~1 cm.

    `grip` maps a pose label (or "start", before the first move) to "open" / "close", e.g. {"start": "open",
    "pick": "close", "place": "open"}: open puts the tips 2 cm wider apart than the poses' object width and margin,
    close 3 mm narrower than the object (it holds it). Fails if a step misses or a joint would jump. Nothing moves.
    `execute` counts a labeled pose reached when every joint is within `pose_tol_deg` (the ~1 cm steps between
    them: 4 deg); a loaded, extended arm sags further than 2.5 deg, so loosen it (e.g. 4) for a carry.
    """
    # solved from a capture taken before the arm moved? (the capture, not the poses' node)
    solved_from = poses.capture if h.ws.id_of(poses.capture) is not None else poses
    if h.ws.is_stale(solved_from):
        raise ValueError(
            "these poses come from before the arm last moved: capture again and solve them again"
        )
    return _execute.prepare_execution(h.model, poses, grip, pose_tol_deg=pose_tol_deg)


def _settings(
    changes: Mapping[str, Any], base: "_grasping.GraspSettings | None" = None
) -> "_grasping.GraspSettings":
    """`base` (default the defaults) with `changes` by field name; an unknown name is refused."""
    known = set(_grasping.GraspSettings.__dataclass_fields__)
    if set(changes) - known:
        raise ValueError(f"unknown settings {sorted(set(changes) - known)}: {sorted(known)}")
    changes = dict(changes)
    if "rotation_bounds_deg" in changes:
        changes["rotation_bounds_deg"] = tuple(float(b) for b in changes["rotation_bounds_deg"])
    return dataclasses.replace(base or _grasping.GraspSettings(), **changes)


def _aligned_since(h: Harness, obj: Any, what: str) -> None:
    """Refuse `obj` if any arm move other than an `align_step` came after it (or a scan: the scene may have changed).

    Alignment steps move the arm by a few mm short of the object, so wrist views and points from before them still
    describe the object where it is (the workspace counts every arm move as stale: this is the narrower rule).
    """
    node_id = h.ws.id_of(obj)
    if node_id is None or not h.ws.is_stale(obj):
        return
    made = next((i for i, call in enumerate(h.calls) if node_id in call.get("result", [])), None)
    if made is None:
        raise ValueError(f"{what} comes from before the arm last moved: look again")
    for call in h.calls[made + 1 :]:
        refused = str(call.get("outcome", "")).startswith("refused")
        if call["tool"] == "align_step" or refused:
            continue
        if call["effect"] in (Effect.MOVE_ARM.value, Effect.END.value) or call["tool"] == "scan":
            raise ValueError(
                f"{what} comes from before {call['tool']}, which may have moved things: look again"
            )


def _fresh(h: Harness, obj: Any, what: str) -> None:
    """Refuse `obj` if it comes from before the arm last moved."""
    if h.ws.id_of(obj) is not None and h.ws.is_stale(obj):
        raise ValueError(f"{what} comes from before the arm last moved: capture again (and fit again)")


@tool(Effect.SOFTWARE)
def fit_cylinder(
    h: Harness,
    view: View,
    uv: np.ndarray,
    axis: str = "up",
    label: str = "object",
    radius: float | None = None,
) -> Object:
    """An object as a cylinder, fitted to the depth where you mark it in `view` (pixels `uv`, (N, 2)).

    Mark only the part that may be grasped (a pen above the cup's rim, above the book): the tool knows nothing else of
    the object, so nothing else of it is grasped. A few marks around it, or a line of marks along a thin one: the
    region is their convex hull, and what in it is not the object (the table, what is behind) is dropped.
    `axis="up"` for an object standing on the table (its axis vertical); `"free"` for one lying or leaning (a pen, a
    handle). `radius` (m) if you know it (a pen: 0.0045): thin objects' fitted radius comes out short. The description
    gives its points, residual and how much of its side the camera saw: a narrow arc means a biased fit, so mark it
    in a view that shows more of it. A view looking straight down on an upright object is refused (no side to fit).
    Draw it with `render`. Nothing moves.
    """
    if axis not in ("up", "free"):
        raise ValueError(f"axis is 'up' or 'free', not {axis!r}")
    return _tools.fit_cylinder(view, np.asarray(uv, dtype=np.float64), axis, label, radius=radius)  # type: ignore[arg-type]


@tool(Effect.SOFTWARE)
def grasp_candidates(
    h: Harness,
    target: Object,
    capture: Capture,
    arms: Sequence[str] = ("right", "left"),
    where: Mapping[str, Sequence[float]] | None = None,
    n: int = 5,
    part: str | None = None,
    **settings: Any,
) -> _grasping.GraspCandidates:
    """The best grasps of `target` (from `fit_cylinder`), ranked and named A, B, ...: you choose one by name.

    Every antipodal grasp of the cylinder (the jaws closing straight across it) on a grid, for each arm in `arms` and
    either jaw on either side, kept if the arm reaches the attack point (5 cm back) and the grasp, and the head
    camera (as in `capture`, the arms where it saw them) sees the grasp's center and contacts past both arms. Ranked
    by how close to a true antipodal grip the arm gets, then by how far its joints stay from their limits. Each says
    its arm, pitch, heading and roll, how antipodal, and its joint margin; the rejected are counted by reason.

    Nothing about the scene is checked: say from where. `where` narrows the grasps, in the cylinder's terms: "t" (m
    along its axis from its bottom end), "phi" (deg: the side the jaws come from, around the axis; for an upright one
    0 is the robot's side), "psi" (deg: the approach's tilt down the axis: 0 level, 90 from above, -90 from below),
    each (low, high), e.g. {"psi": (0, 90)} for an object in a book or a cup (not from below, through the support).
    `**settings` change what is assumed (`GraspSettings`, printed with the result): `rotation_bounds_deg` (how far
    the arm may turn off the exact grasp, about the closing line, the approach, the hinge; default 45 each: narrow
    the approach for a slippery object), `position_bound_m`, `rank_step_deg`, `margin` (the fixed jaw's clearance,
    m), `attack` (m). Whether the grasp is seen is not checked: judge it at the hold. Draw them with `render`. Nothing
    moves.
    """
    _fresh(h, capture, "the capture")
    _fresh(h, target, "the object")
    ranges = {k: (float(v[0]), float(v[1])) for k, v in (where or {}).items()}
    return _grasping.grasp_candidates(
        target,
        h.model,
        capture,
        tuple(arms),
        ranges or None,
        n=n,
        part=part,
        settings=_settings(settings),
    )


@tool(Effect.SOFTWARE)
def plan_pick(
    h: Harness,
    candidates: _grasping.GraspCandidates,
    name: str = "A",
    stage_up: float = 0.06,
    lift: float = 0.03,
) -> Poses:
    """The chosen candidate (`name`) as the poses of a pick: "stage", "pick.back", "pick", "lift".

    "stage" is `stage_up` m above the attack point, "pick.back" the attack point, "pick" the grasp, "lift" `lift` m
    above it.
    For `prepare_execution` with grip {"start": "open", "pick": "close"}, or a part of it: following the protocol,
    first `select_poses(pick, "stage", "pick.back")` and hold there, then refine from the wrist camera
    (`wrist_hold_views`, `wrist_contacts`) before going in. Nothing moves.
    """
    if candidates.capture is None:
        raise ValueError("these candidates have no capture")
    _fresh(h, candidates.capture, "the candidates' capture")
    return _grasping.plan_pick(h.model, candidates[name], candidates.capture, stage_up, lift)


@tool(Effect.SOFTWARE)
def render_wrist(
    h: Harness,
    view: WristView,
    pick: Poses | None = None,
    marks: Sequence[Sequence[float]] | None = None,
    labels: Sequence[str] | None = None,
    midline_top: Sequence[float] | None = None,
    midline_sides: Sequence[Sequence[float]] | None = None,
    object_axis: Sequence[float] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """The wrist camera's picture (`wrist_look`) with what lines the jaws up drawn on it.

    Cyan / magenta circles: where the model puts the fixed jaw's tip (the pinch point) and the moving one. Yellow, with
    the pick plan (`plan_pick`): the approach line, a dot every 2 cm ahead of where the object's center belongs,
    labeled in cm: the object is lined up when its center sits on the line (at the dot for how far ahead it is).
    Green crosses: `marks`, pixels you want to check before passing them on (e.g. the object's two sides). White
    circle: where the last `align_step` expected the object's center. Green, with `pick`: the closing line (fixed tip to
    where the moving tip touches an object the plan's width: the object's two sides should fall on its ends), the
    moving tip's sweep to the close, and where the object's center is once held. These are fixed to the gripper:
    the last check before closing, with no depth. Orange, with `midline_top` (the top of the object's axis as seen:
    an upright bottle's cap center) and `midline_sides` (the top rim's two ends across): the object's middle line down
    from there, ticked every cm below the top; the "held" circle should sit on it at the grasp's height (off it
    sideways: off across the jaws; at the wrong tick: short of or past the middle). Its direction is the model's
    (the camera's tilt), its anchor what you mark. The camera is a few cm off the gripper's axis,
    so the line is not through the picture's middle.
    """
    return _wrist.render_wrist(h.model, view, pick, marks, labels, midline_top, midline_sides, object_axis)


_SFM_DEPRECATED = "structure from motion is not needed at a hold: DA3 posed by the arm's joints does as well (object_grasping.md 7); use wrist_hold_views and wrist_contacts"


@tool(
    Effect.SOFTWARE,
    deprecated="measuring from marked edges is superseded by wrist_contacts; for a given move, plan_wrist_move (object_grasping.md 7)",
)
def plan_alignment(
    h: Harness,
    view: WristView,
    pick: Poses,
    edges: Sequence[Sequence[float]],
    advance: float = 0.0,
    roll_deg: float = 0.0,
    gain: float = 1.0,
    sidestep_m: float = 0.0,
    center: WristPoint | None = None,
    hinge_gain: float | None = None,
    object_axis: Sequence[float] | None = None,
    nudge_m: Sequence[float] | None = None,
) -> Alignment:
    """Measure the object in a wrist view against the pick plan, and plan one step that lines the jaws up on it.

    (Its measurement from marked edges, with the depth from the width or `center`, is superseded by the DA3
    checkpoint (object_grasping.md 7); as a planner of a move given in `nudge_m` it is current.)

    `edges`: the object's two sides in `view`, at the height (or place) it is to be grasped, as two pixels [[u, v],
    [u, v]] (mark them on `render_wrist` first). How far away it is matters (the camera is ~7 cm off the jaws'
    line: a depth 10% off is ~7 mm sideways). `center`, the object's center triangulated from two wrist views
    (`wrist.triangulate`, no longer a tool), gives it; without one, the plan's width (`pick.width`, the fitted
    primitive's, good to 10-20%) does. `sidestep_m` moves across the jaws (toward the moving one; negative: the fixed
    one; the next step corrects it back). `hinge_gain=0` leaves the offset along the
    hinge (square to the closing line) to the plan: for an upright object grasped from the side it is the height on
    it, which any height on the body serves and which marks do not pin down. But give the object's axis too
    (`object_axis=(0, 0, 1)` for an upright one): the measured center slides along it to the jaws' line, so how far
    ahead it is is measured where the jaws close (unless the approach is level, the hinge is not the object's axis,
    and the marks' height shifts "ahead": bowl6 went 1 cm too far in on it). `nudge_m`: a root-frame move added as
    it is (m), e.g. back along the approach and up, to look again. Each step makes 80% of the correction, so it falls
    short rather than overshoots. The step moves the gripper across the approach line by the object's offset from it (times `gain`), in
    `advance` (0-1) of the way along it, and turns the jaws `roll_deg` about it (positive: counterclockwise, looking
    along the approach). At most 3 cm and 20 deg a step (more is clipped and said: re-plan from the head camera).

    The description says how far off the object is (toward the moving jaw, and square to that), how far ahead, the
    step, and, after an `align_step`, how much of the predicted shift the last step made (its real gain, with a fair
    `gain` for the next step). Nothing moves: `align_step` runs it.
    """
    _fresh(h, view, "the wrist view")
    if center is not None:
        _aligned_since(h, center, "the triangulated center")
    return _wrist.plan_alignment(
        h.model,
        view,
        pick,
        edges,
        advance,
        roll_deg,
        gain,
        sidestep_m,
        center,
        hinge_gain,
        object_axis,
        nudge_m,
    )


@tool(Effect.SOFTWARE, deprecated=_SFM_DEPRECATED)
def sfm_torch(
    h: Harness, views: Sequence[WristView], prior_mm: float = _sfm.PRIOR_MM, prior_deg: float = _sfm.PRIOR_DEG
) -> SfmResult:
    """The wrist camera's real motion between `views`: structure from motion (DISK features, LightGlue matches).

    `views`: several frames of one arm from different poses; the model's poses are the start and a prior. The
    description gives the fit's error before and after, how far each frame's motion was off, and the table: its plane
    and how high the jaws are over it. Views from before alignment steps count. ~10 s (the weights load on each call;
    OpenCV's SIFT, the light alternative, failed on the robot's frames, hil3). The frames
    are first turned back by the wrist's roll as the model has it (these features are not rotation invariant). Nothing
    moves.
    """
    for i, view in enumerate(views):
        _aligned_since(h, view, f"view {i}")
    return _sfm.refine_views(views, "torch", prior_mm, prior_deg)


@tool(Effect.SOFTWARE)
def wrist_contacts(
    h: Harness,
    views: HoldViews | WristView,
    box: Sequence[float],
    band: Sequence[float],
    radius: float | None = None,
    gap: float = _contacts.GAP_M,
    into: float = _contacts.INTO_M,
    table_z: float | None = None,
) -> ContactPatches:
    """Where the jaws would touch the object, and the move that gets them there, from the wrist camera at a hold.

    `views`: `wrist_hold_views`' frames (a short wrist roll), or one `wrist_look` when too close to roll. `box`: (u0, v0,
    u1, v1) around the graspable part in the current frame (the body, not a cap or a lid); `band`: (low, high) its
    heights above the table (m); `radius` (m) if known (thin objects' free fits run wide). `table_z`: the table's height
    (m, root frame, from the head's depth) sets the scale of a hold's views; without it, and always for a single
    view, the fixed finger does (it must be in the picture).

    Depth Anything 3 gives dense depth on every frame (posed by the arm's joints), an upright cylinder is fitted to the
    graspable part, and its widest chord in the jaws' closing plane gives the contacts. The move puts that chord `into`
    the fingers (m) and the fixed finger `gap` (m) from its contact, both relative to where the jaws are now; plan it
    with `plan_wrist_move`. The description says the cylinder, how far beyond the fingertips the chord is, both
    contacts along the closing line (+ toward the moving jaw) and the scale; `render_contacts` draws it. Re-measure
    after every move (a measurement carried through the arm's model over centimeters goes wrong). Nothing moves.
    """
    _aligned_since(h, views, "the views")
    return _contacts.contact_patches(h.model, views, box, band, radius, gap, into, table_z)


@tool(Effect.SOFTWARE)
def render_contacts(h: Harness, contacts: ContactPatches) -> np.ndarray:
    """The current wrist frame with `wrist_contacts`' evidence drawn on it.

    The box (white), the points used (yellow), the fitted cylinder (orange rings at the band's ends), the fixed (cyan)
    and moving (magenta) contacts at the widest chord, and the closing line now (green). Nothing moves.
    """
    return contacts.image


@tool(Effect.SOFTWARE)
def plan_wrist_move(
    h: Harness,
    view: WristView | HoldViews,
    contacts: ContactPatches | None = None,
    move_m: Sequence[float] | None = None,
    fraction: float = 1.0,
) -> WristMove:
    """A small move of the gripper from where `view` saw it (a hold's: its current view), solved for `align_step`.

    `contacts`' move (times `fraction`: e.g. 0.5 to stop short and look again), or `move_m` (m, root frame: e.g.
    straight up, or forward with no drop toward the table). The gripper's turn is kept; at most 3 cm (more is clipped
    and said). Nothing moves.
    """
    _fresh(h, view, "the wrist view")
    if contacts is not None:
        _fresh(h, contacts, "the contacts")
    if (contacts is None) == (move_m is None):
        raise ValueError("give either the contacts or a move")
    move = np.asarray(move_m, dtype=np.float64) if move_m is not None else fraction * contacts.move_root  # type: ignore[union-attr]
    label = f"{fraction:g} of the contacts' move" if contacts is not None else "the move given"
    now = view.views[view.current] if isinstance(view, HoldViews) else view
    return _wrist.plan_move(h.model, now, move, label)


@tool(Effect.SOFTWARE, deprecated=_SFM_DEPRECATED)
def scan_views(h: Harness, views: Sequence[WristView], method: str = "light", current: int = -1) -> WristScan:
    """A wrist scan of `views` already taken (one arm's frames from different poses): structure from motion over them.

    `method` "light" (SIFT) or "torch" (DISK and LightGlue: far more matches on the robot's frames, ~10 s);
    `current`: the frame the arm is at now (default the last). As `wrist_scan`, without moving. Views from before
    alignment steps count.
    """
    for i, view in enumerate(views):
        _aligned_since(h, view, f"view {i}")
    return _wrist_scan.scan_views(views, method, current)


@tool(Effect.SOFTWARE, deprecated=_SFM_DEPRECATED)
def scan_scale(
    h: Harness,
    scan: WristScan,
    table_z: float,
) -> WristScan:
    """A wrist scan with its scale set by the table's height, instead of the arm's model.

    Structure from motion fixes everything but one scale, which comes from the arm's moves as the model has them (on
    the robot ~10% off: 1-2 cm at a grasp's distances). `table_z` (m, root frame: the head's depth on the table) sets
    it; it relies on the model only for the camera's height now. Nothing moves.
    """
    return _wrist_scan.scale_from_table(scan, table_z)


@tool(Effect.SOFTWARE, deprecated=_SFM_DEPRECATED)
def render_scan(
    h: Harness,
    scan: WristScan,
    frames: Sequence[int] | None = None,
    objects: Sequence[Object] = (),
) -> np.ndarray:
    """A wrist scan's frames, with objects drawn in each.

    All as a contact sheet (default), or the ones in `frames` (one: full size), each labeled (its number, the wrist's
    turn, "now" where the arm is), with `objects` (their outline, orange): what sits right in every frame is right.
    Nothing moves.
    """
    return _wrist_scan.render(scan, frames, objects)


@tool(Effect.SOFTWARE)
def arm_path(h: Harness, execution: Execution, every: int = 5) -> ArmPath:
    """The arm's body along an execution, to draw with `render`: where the shoulder, elbow, wrist and gripper go.

    Skeletons along the first move (from where the capture saw the arm, all joints at once: it arcs, it does not
    follow a straight line), at every pose, and at every `every`-th step between. The return home is not included.
    """
    return _execute.arm_path(h.model, execution, every)


@tool(
    Effect.SOFTWARE,
    deprecated="still for a carry, until plan_place knows the object's pose in the jaws (phase D)",
)
def held_object(
    h: Harness,
    poses: Poses,
    length: float,
    grasp: str,
    release: str,
    axis: Sequence[float] = (0.0, 0.0, 1.0),
) -> HeldObject:
    """How the object taken at pose `grasp` and let go at `release` turns while the jaws carry it (text, not drawn).

    `length` (m) and `axis` (its long direction at the grasp; default vertical, an upright object) describe it; its
    center is the grasp point. The description gives its tilt from how it was grasped at each pose: a roll of the
    wrist on the way turns it, and could put it down tilted or upside down.
    """
    return _execute.held_object(poses, length, grasp, release, axis)


# ---- software: the history ----------------------------------------------------------------------------------------


@tool(Effect.SOFTWARE, records=False)
def history(h: Harness, last: int = 20) -> str:
    """The last `last` recorded objects: id, the call that made it, what it was made from, and its description."""
    rows = []
    for entry in h.ws.log()[-last:]:
        stale = " [stale]" if entry["stale"] else ""
        rows.append(
            f"{entry['id']}{stale} = {entry['call']}({', '.join(entry['parents'])}): {entry.get('describe', '')}"
        )
    return "\n".join(rows)


@tool(Effect.SOFTWARE, records=False)
def lineage(h: Harness, obj: Any) -> str:
    """What `obj` (an object or its id) was made from, nearest first, with the calls."""
    node_id = obj if isinstance(obj, str) else h.ws.require_id(obj)
    return "\n".join(
        f"{i} = {h.ws.nodes[i].call}({', '.join(h.ws.nodes[i].parents)})" for i in h.ws.ancestors(node_id)
    )


@tool(Effect.SOFTWARE, records=False)
def is_stale(h: Harness, obj: Any) -> bool:
    """True if `obj` (an object or its id) comes from before the arm last moved: what it shows may have changed."""
    return h.ws.is_stale(obj)


# ---- hardware -----------------------------------------------------------------------------------------------------


@tool(Effect.SENSE)
def capture(h: Harness) -> Capture:
    """A new color + depth frame from where the head is now, with the robot's pose. Nothing moves."""
    return Capture.from_snapshot(h.client.capture(), h.model)


@tool(Effect.MOVE_HEAD)
def scan(
    h: Harness, tilts: Sequence[float] = _scan.TILTS_DEG, pans: Sequence[float] = _scan.PANS_DEG
) -> list[Capture]:
    """Look around: a capture at each head pose, then the head goes back home.

    The head steps through `tilts` (deg down from home) x `pans` (deg left of home). A new scan also marks
    everything from the earlier ones stale (the scene may have changed). Moves the head: a person approves first.
    """
    if any(isinstance(node.obj, Capture) for node in h.ws.nodes.values()):
        h.ws.moved()
    return _scan.scan(h.client, h.model, tilts, pans)


@tool(Effect.MOVE_HEAD)
def look_at(h: Harness, target: Any) -> Capture:
    """Point the camera at `target` (a root-frame xyz in m, or `Points`: the first one) and take a capture there.

    The head turns and tilts to center the target, within its limits (if it cannot, the target is off center in the
    new capture: `render` it with the point to see). Moves the head: a person approves first.
    """
    xyz = target.xyz[0] if isinstance(target, Points) else np.asarray(target, dtype=np.float64)
    if not np.isfinite(xyz).all():
        raise ValueError("the target has no position (no depth there?): use look_at_pixel")
    state = h.client.head_state()
    goal, _ = _look.aim_head(h.model, {name: int(state[name]["ticks"]) for name in state}, xyz)
    done_at = None
    for name, delta_deg in _look.head_moves(state, goal).items():
        done_at = h.client.head_move(name, delta_deg=delta_deg)["done_at"]
    return Capture.from_snapshot(h.client.capture(after=done_at), h.model)


@tool(Effect.MOVE_HEAD)
def look_at_pixel(h: Harness, view: View, uv: Sequence[float], distance: float = 0.5) -> Capture:
    """Point the camera at what `view` shows at pixel `uv` and take a capture there.

    The target is the depth at that pixel, or, where there is none, the point `distance` m along its viewing ray.
    Moves the head: a person approves first.
    """
    xyz = _tools.locate_pixels(view, np.array([uv], dtype=np.float64)).xyz[0]
    if not np.isfinite(xyz).all():
        origin, direction = view.capture.rays(view.to_capture(np.asarray(uv, dtype=np.float64)))
        xyz = origin + distance * direction
    return look_at(h, xyz)


@tool(Effect.SENSE)
def wrist_look(h: Harness, arm: str) -> WristView:
    """A new frame from `arm`'s wrist camera ("left" or "right") with its joints. Nothing moves.

    The camera rides on the gripper, so it sees the object in the jaws' own frame (no head, no arm model in between):
    `render_wrist` it with the pick plan to see how far off the jaws are.
    """
    return _wrist.wrist_view(h.model, h.client.wrist_capture(arm))


@tool(Effect.MOVE_ARM, deprecated=_SFM_DEPRECATED)
def wrist_scan(
    h: Harness,
    arm: str,
    method: str = "torch",
    rolls_deg: Sequence[float] = _wrist_scan.SWEEP_ROLLS_DEG,
    flexes_deg: Sequence[float] = _wrist_scan.SWEEP_FLEXES_DEG,
) -> WristScan:
    """Sweep the wrist where the arm holds (the attack point), a wrist frame at each pose, and back: a wrist scan.

    Only the wrist moves: turned by `rolls_deg` and bent by `flexes_deg` about where it is (the camera, ~7 cm off the
    gripper's axis, swings through a few cm: the parallax for depth), then back to the start. The frames' real
    poses come from the images (structure from motion, `method` as `scan_views`), so the arm's motion error drops out.
    The description says how far its motion was off, the table plane and how high the jaws are over it, and how well
    the wrist came back (the first and last frames). Then: `render_scan` to see the frames. Moves the arm: a person
    approves first.
    """
    views, labels = _wrist_scan.sweep(h.client, h.model, arm, rolls_deg, flexes_deg)
    return _wrist_scan.scan_views(views, method, len(views) - 1, labels)


@tool(Effect.MOVE_ARM)
def wrist_hold_views(h: Harness, arm: str, rolls_deg: Sequence[float] = (-15.0, 15.0)) -> HoldViews:
    """At a hold: a wrist frame now, at each wrist roll of `rolls_deg` (deg) about where the wrist is, and back.

    Only the wrist roll moves (the fixed fingertip is on its axis; the moving jaw swings ~1 cm per 15 deg: not when
    the jaws are already around the object). For `wrist_contacts`. Moves the arm: a person approves first.
    """
    return _contacts.hold_views(h.client, h.model, arm, rolls_deg)


@tool(Effect.MOVE_ARM)
def align_step(h: Harness, alignment: Alignment | WristMove, max_vel: float = _wrist.STEP_VEL) -> WristView:
    """Make one small move (`plan_wrist_move`, or `plan_alignment`): the arm moves, then holds there.

    Returns the wrist view after it, which also says where the object's center should appear if the step went as
    planned (`render_wrist` draws it): measure again with `plan_alignment` on it. Everything seen before becomes
    stale. Moves the arm: a person approves first, seeing the step.
    """
    # an alignment's center may be older (triangulated): the view it was planned from must be fresh; a move's
    # view may be a hold's (not recorded on its own): the move itself must be
    source = alignment if isinstance(alignment, WristMove) else alignment.view
    if h.ws.id_of(source) is None or h.ws.is_stale(source):
        raise NotStartedError("this alignment comes from before the arm last moved: look and measure again")
    return _wrist.run_alignment(h.client, h.model, alignment, max_vel)


@tool(Effect.MOVE_ARM)
def execute(
    h: Harness, execution: Execution, max_vel: float = 20.0, step: bool = False, hold: bool = False
) -> Capture:
    """Run an execution (`prepare_execution`): every move in order at `max_vel` deg/s, then home.

    Returns a capture taken after; everything seen before becomes stale. With `hold`, the arm stays at the last pose
    and the capture is taken there: render the planned poses on it to see where the gripper really is (then `home`).
    If a move does not settle, the arm stops and holds there, and the error says which joints missed by how much and
    names the capture taken there: look, and re-plan from it. With `step`, a person confirms each move. Moves the
    arm: a person approves first, seeing every move.
    """
    capture = execution.poses.capture  # what it was solved from (see `prepare_execution`)
    if h.ws.is_stale(capture if h.ws.id_of(capture) is not None else execution):
        raise NotStartedError(
            "this execution comes from before the arm last moved: capture again and plan again"
        )
    confirm = h.confirm if step else None
    return _execute.run_execution(h.client, h.model, execution, max_vel, confirm=confirm, hold=hold)


@tool(Effect.MOVE_ARM)
def home(h: Harness) -> Capture:
    """Send the arms to rest (their recorded home poses) and the head home, and take a capture there.

    Everything seen before becomes stale. Moves the arm: a person approves first.
    """
    status = h.client.home()
    return Capture.from_snapshot(h.client.capture(after=status.get("finished_at")), h.model)


@tool(Effect.END, records=False)
def end_session(h: Harness, reason: str = "") -> str:
    """End the session: the robot goes to rest, and no tool runs after this. Say why (done, stuck, unsafe, ...)."""
    if h._client is not None:
        h._client.home()
    h.ended = True
    return f"session ended{': ' + reason if reason else ''}"
