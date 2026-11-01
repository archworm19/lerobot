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

from . import execute as _execute, reach as _reach, scan as _scan, tools as _tools
from .objects import Capture, Execution, Focus, Grid, Plan, Points, Poses, TableEstimate, View
from .render import contact_sheet as _contact_sheet, render as _render  # the package re-exports `render`
from .workspace import Workspace

if TYPE_CHECKING:
    from lerobot.felix_jr.server.client import FelixClient

MODEL = Path(__file__).parents[1] / "urdfs" / "xlerobot_calib10312026.urdf"
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


TOOLS: dict[str, Tool] = {}


def tool(effect: Effect, records: bool = True) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register the decorated function as a tool with `effect`."""

    def register(fn: Callable[..., Any]) -> Callable[..., Any]:
        if not fn.__doc__:
            raise ValueError(f"tool {fn.__name__} needs a docstring: it is the model's documentation")
        TOOLS[fn.__name__] = Tool(fn.__name__, fn, effect, records)
        return fn

    return register


@dataclass(frozen=True)
class Policy:
    """Which effects need a person's approval before they run."""

    ask: frozenset[Effect]


SAFE = Policy(frozenset({Effect.MOVE_HEAD, Effect.MOVE_ARM}))  # the default
UNATTENDED = Policy(frozenset())  # for tests against fake hardware


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
        """The robot client (connected on first use)."""
        if self._client is None:
            from lerobot.felix_jr.server.client import FelixClient

            self._client = FelixClient(self.endpoint)
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
            try:
                return spec.fn(self, *a, **k)
            finally:
                if (
                    spec.effect is Effect.MOVE_ARM
                ):  # before the result is recorded, so an after-capture is fresh
                    self.ws.moved()

        try:
            result = self.ws.call(run, *args, **kwargs) if spec.records else run(*args, **kwargs)
        except Exception as e:
            entry["outcome"] = f"error: {type(e).__name__}: {e}"
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
        """Load `state()` output (older sessions may lack the call log)."""
        self.ws, self.names = state["ws"], dict(state["names"])
        self.calls, self.ended = list(state.get("calls", [])), bool(state.get("ended", False))


def describe_tools() -> str:
    """Every tool's signature (without the runner), effect and docstring: the model's tool documentation."""
    blocks = []
    for name, spec in TOOLS.items():
        params = list(inspect.signature(spec.fn).parameters.values())[1:]
        signature = re.sub(r"\b(?:[a-z_]\w*\.)+(?=\w)", "", f"{name}({', '.join(str(p) for p in params)})")
        blocks.append(f"{signature}  [{spec.effect.value}]\n{inspect.cleandoc(spec.fn.__doc__ or '')}")
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
def check_path(h: Harness, capture: Capture, waypoints: np.ndarray, radius: float = 0.02) -> Plan:
    """Check the straight legs between root-frame `waypoints` (M, 3, meters) against the capture's depth.

    Each sample along a leg is free, unknown (no depth there) or blocked, for a ball of `radius` meters around it.
    The object being grasped or carried counts as blocked too.
    """
    return _tools.check_path(capture, waypoints, radius)


@tool(Effect.SOFTWARE)
def render(h: Harness, view: View, *overlays: Any, legend: bool = True) -> np.ndarray:
    """The view's image with overlays drawn on it: `Grid`, `Points`, `Plan`, `TableEstimate`, `Poses`."""
    return _render(view, *overlays, legend=legend)


@tool(Effect.SOFTWARE)
def contact_sheet(
    h: Harness, views: Sequence[View], *overlays: Any, labels: Sequence[str] | None = None
) -> np.ndarray:
    """Several views rendered side by side in one image, each with the overlays and its label."""
    return _contact_sheet(list(views), *overlays, labels=labels)


# ---- software: where the arm can put its gripper ----------------------------------------------------------------


@tool(Effect.SOFTWARE)
def check_poses(
    h: Harness,
    arm: str,
    capture: Capture,
    points: np.ndarray,
    pitch_deg: float | None = -90.0,
    width: float = 0.0,
    across: Sequence[float] | None = (0.0, 0.0, 1.0),
) -> Poses:
    """Can `arm` ("left" / "right") put its tool point at each of `points` (M, 3, root frame, m)?

    The tool point is the midpoint between the jaws, opened `width` meters (the object's width: its center goes
    there). `pitch_deg` is the gripper axis' angle above horizontal: -90 straight down, 0 horizontal (None: any).
    The jaws close across `across` (default vertical: they close horizontally; None: any). The heading is not a
    choice: it follows from the shoulder. `capture` gives the robot's pose. Returns the solved poses: per pose,
    reachable or by how much it misses.
    """
    return _reach.check_poses(h.model, arm, capture, points, pitch_deg, width, across)


@tool(Effect.SOFTWARE)
def approach_poses(
    h: Harness,
    arm: str,
    capture: Capture,
    targets: np.ndarray,
    pitch_deg: Sequence[float] = (0.0, -90.0),
    width: float = 0.0,
    back: float = 0.05,
    up: float = 0.05,
    across: Sequence[float] | None = (0.0, 0.0, 1.0),
    order: Sequence[str] | None = None,
) -> Poses:
    """Find one gripper pitch at which `arm` reaches every target (K, 3, root frame, m), from either side.

    Each target is reached from `back` m along the gripper axis and left `up` m above (or the reverse). Use it to
    pick and place: targets = [where the object is, where it goes].

    `pitch_deg` = (preferred, other end): (0, -90) prefers from the side, (-90, 0) from above. `width`, `across` as
    in `check_poses`. Returns poses labeled back1, at1, up1 (in along the axis, leave upward), then up2, at2, back2
    for the next target (come down, leave along the axis); `order` changes the order. If no pitch works for all,
    there are no poses, and the description says which pitches each target would need.
    """
    return _reach.approach_poses(
        h.model, arm, capture, targets, pitch_deg, width, back, up, across, order=order
    )


@tool(Effect.SOFTWARE)
def prepare_execution(
    h: Harness,
    poses: Poses,
    grip: Mapping[str, str] | None = None,
    gripper_open: float = 60.0,
    roll_deg: float | None = None,
) -> Execution:
    """The exact moves `execute` would send to go through `poses` in order, checked step by step (~1 cm apart).

    `grip` maps a pose label (or "start") to "open" / "close"; by default for `approach_poses` output: open at the
    start, close at at1, open at at2. Fails if a step is out of reach or a joint would jump. Nothing moves.
    """
    if h.ws.is_stale(poses):
        raise ValueError(
            "these poses come from before the arm last moved: capture again and solve them again"
        )
    return _execute.prepare_execution(h.model, poses, grip, gripper_open, roll_deg=roll_deg)


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


@tool(Effect.MOVE_ARM)
def execute(h: Harness, execution: Execution, max_vel: float = 20.0, step: bool = False) -> Capture:
    """Run an execution (`prepare_execution`): every move in order at `max_vel` deg/s, then home.

    Returns a capture taken after; everything seen before becomes stale. With `step`, a person confirms each move.
    Moves the arm: a person approves first, seeing every move.
    """
    if h.ws.is_stale(execution):
        raise ValueError("this execution comes from before the arm last moved: capture again and plan again")
    return _execute.run_execution(h.client, h.model, execution, max_vel, confirm=h.confirm if step else None)


@tool(Effect.END, records=False)
def end_session(h: Harness, reason: str = "") -> str:
    """End the session: the robot goes to rest, and no tool runs after this. Say why (done, stuck, unsafe, ...)."""
    if h._client is not None:
        h._client.home()
    h.ended = True
    return f"session ended{': ' + reason if reason else ''}"
