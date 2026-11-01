"""The workspace: the DAG of everything a reasoning chain made (`docs/harness.md`, sections 1.B, 1.C, 1.H layer 3).

`ws.call(fn, *args, **kwargs)` runs any function, a long-term tool or a scratch function the VLM wrote, and records
its result as a node: an id, the call, its arguments (objects replaced by their ids), and its parents (the
arguments that are workspace objects). Objects stay plain Python objects; the workspace only adds ids, the graph,
freshness and the log.

Freshness: every `Capture` is stamped with the current epoch, and every node inherits the oldest epoch among its
parents. `ws.moved()` (called after the robot moves) starts a new epoch, which makes everything derived from older
captures stale. A node that depends on no capture (a grid at a height typed in by hand) is never stale.
"""

import json
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .objects import Capture, Focus, Grid, Plan, Points, Poses, Reach, TableEstimate, View

OBJECT_TYPES = (Capture, View, Grid, TableEstimate, Points, Plan, Focus, Poses, Reach)
MAX_LOGGED_ELEMENTS = 64  # arrays with more elements are logged by shape only


@dataclass(frozen=True)
class Node:
    """One recorded object: how it was made and what from."""

    id: str
    obj: Any
    call: str
    args: dict[str, Any]  # json-ready: objects as "@<id>", big arrays as their shape
    parents: tuple[str, ...]
    epoch: int | None  # oldest capture epoch it depends on; None if it depends on no capture


class Workspace:
    """The DAG store for one reasoning chain. See the module docstring."""

    def __init__(self) -> None:
        """An empty workspace at epoch 0."""
        self.nodes: dict[str, Node] = {}
        self.epoch = 0
        self._ids: dict[int, str] = {}  # id(obj) -> node id; nodes keep the objects alive, so ids stay unique
        self._counts: Counter[str] = Counter()

    # ---- recording ----------------------------------------------------------------------------------------------

    def add(
        self, obj: Any, call: str = "add", parents: tuple[str, ...] = (), args: dict | None = None
    ) -> str:
        """Record `obj` (once: adding it again returns its id) and return its id, e.g. "view3"."""
        if (existing := self.id_of(obj)) is not None:
            return existing
        kind = "image" if isinstance(obj, np.ndarray) else type(obj).__name__.lower()
        self._counts[kind] += 1
        node_id = f"{kind}{self._counts[kind]}"
        if isinstance(obj, Capture):
            epoch: int | None = self.epoch
        else:
            epochs = [e for p in parents if (e := self.nodes[p].epoch) is not None]
            epoch = min(epochs) if epochs else None
        self.nodes[node_id] = Node(node_id, obj, call, args or {}, tuple(parents), epoch)
        self._ids[id(obj)] = node_id
        return node_id

    def call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run `fn(*args, **kwargs)` and record its result (unless None) with its arguments as parents.

        Harness objects passed in that the workspace has not seen are added first (call "external"). Returns the
        result itself; `id_of` gives its id.
        """
        parents: list[str] = []
        for value in _walk([*args, *kwargs.values()]):
            node_id = self.id_of(value)
            if node_id is None and isinstance(value, OBJECT_TYPES):
                node_id = self.add(value, "external")
            if node_id is not None and node_id not in parents:
                parents.append(node_id)
        result = fn(*args, **kwargs)
        if result is not None:
            logged = {f"arg{i}": self._summarize(a) for i, a in enumerate(args)} | {
                k: self._summarize(v) for k, v in kwargs.items()
            }
            self.add(result, getattr(fn, "__name__", repr(fn)), tuple(parents), logged)
        return result

    def moved(self) -> None:
        """The robot moved (or the scene may have changed): everything from earlier captures is now stale."""
        self.epoch += 1

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Unpickling: objects get new `id()`s, so rebuild the object -> node id map."""
        self.__dict__.update(state)
        self._ids = {id(node.obj): node.id for node in self.nodes.values()}

    # ---- queries --------------------------------------------------------------------------------------------------

    def id_of(self, obj: Any) -> str | None:
        """The node id of a recorded object, or None."""
        return self._ids.get(id(obj))

    def __getitem__(self, node_id: str) -> Any:
        """The object recorded under `node_id`."""
        return self.nodes[node_id].obj

    def is_stale(self, ref: Any) -> bool:
        """`ref`: a node id or a recorded object."""
        node = self.nodes[ref if isinstance(ref, str) else self.require_id(ref)]
        return node.epoch is not None and node.epoch < self.epoch

    def ancestors(self, ref: Any) -> list[str]:
        """Ids of everything `ref` was derived from, nearest first."""
        start = ref if isinstance(ref, str) else self.require_id(ref)
        seen: list[str] = []
        frontier = list(self.nodes[start].parents)
        while frontier:
            node_id = frontier.pop(0)
            if node_id not in seen:
                seen.append(node_id)
                frontier.extend(self.nodes[node_id].parents)
        return seen

    # ---- log ------------------------------------------------------------------------------------------------------

    def log(self) -> list[dict[str, Any]]:
        """One json-ready entry per node, in creation order, with the object's `describe()` when it has one."""
        entries = []
        for node in self.nodes.values():
            entry = {
                "id": node.id,
                "call": node.call,
                "args": node.args,
                "parents": list(node.parents),
                "stale": self.is_stale(node.id),
            }
            if hasattr(node.obj, "describe"):
                entry["describe"] = node.obj.describe()
            elif isinstance(node.obj, np.ndarray):
                entry["describe"] = f"image {node.obj.shape[1]}x{node.obj.shape[0]}"
            entries.append(entry)
        return entries

    def save_log(self, path: str | Path) -> None:
        """Write `log()` as JSON lines."""
        Path(path).write_text("".join(json.dumps(e) + "\n" for e in self.log()))

    def require_id(self, obj: Any) -> str:
        """The node id of a recorded object; KeyError if it isn't recorded."""
        node_id = self.id_of(obj)
        if node_id is None:
            raise KeyError(f"{type(obj).__name__} is not in the workspace")
        return node_id

    def _summarize(self, value: Any) -> Any:
        if (node_id := self.id_of(value)) is not None:
            return f"@{node_id}"
        if isinstance(value, np.ndarray):
            return value.tolist() if value.size <= MAX_LOGGED_ELEMENTS else f"array{list(value.shape)}"
        if isinstance(value, (list, tuple)):
            return [self._summarize(v) for v in value]
        if isinstance(value, dict):
            return {str(k): self._summarize(v) for k, v in value.items()}
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, np.generic):
            return value.item()
        return repr(value)[:200]


def _walk(values: list[Any]) -> Iterator[Any]:
    """The values and, inside lists / tuples / dicts, their elements (so overlays passed in a list are parents)."""
    for value in values:
        yield value
        if isinstance(value, (list, tuple)):
            yield from _walk(list(value))
        elif isinstance(value, dict):
            yield from _walk(list(value.values()))
