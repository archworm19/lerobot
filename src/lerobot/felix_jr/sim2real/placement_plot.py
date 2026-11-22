"""A bank's placements drawn: where on the table, what size, and which are held out (`start_pool --plot`).

The top row: the table from above (to the arm's side against ahead of its base, mm) and the objects' sizes (diameter
against height, mm), each placement marked training (blue circle), held out (orange triangle) or never anchored (grey
cross: no grasp held, not in any pool). When the sizes vary, below: the table view again for each size range, a column
per diameter range and a row per height range (`bins` each), so where the held-out and the unanchored placements fall
can be read by size.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

# the reference palette's categorical slots 1 and 2 (light), a recessive grey for what is in no pool; each with its
# own marker, so identity is never colour alone
STYLES = {
    "training": {"color": "#2a78d6", "marker": "o", "s": 14},
    "held out": {"color": "#eb6834", "marker": "^", "s": 20},
    "unanchored": {"color": "#a3a29a", "marker": "x", "s": 14},
}
SURFACE, INK, MUTED = "#fcfcfb", "#1f1f1d", "#6b6a64"


def _coordinates(placements: Sequence[Mapping[str, Any]], base_xy: Sequence[float]) -> dict[str, np.ndarray]:
    from .start_pool import placement_coordinate

    return {
        key: 1000 * np.array([placement_coordinate(p, base_xy, key) for p in placements])
        for key in ("ahead", "side", "diameter", "height")
    }


def _style(ax: Any, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.set_xlabel(xlabel, fontsize=8, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=8, color=MUTED)
    ax.tick_params(labelsize=7, colors=MUTED, length=2)
    ax.grid(True, color="#e6e5e0", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c1")


def _scatter(ax: Any, x: np.ndarray, y: np.ndarray, kinds: np.ndarray) -> None:
    for kind, style in STYLES.items():
        mask = kinds == kind
        if mask.any():
            edge = {} if style["marker"] == "x" else {"edgecolors": SURFACE, "linewidths": 0.5}
            ax.scatter(x[mask], y[mask], label=f"{kind} ({int(mask.sum())})", **style, **edge)


def _edges(values: np.ndarray, bins: int) -> np.ndarray:
    lo, hi = float(values.min()), float(values.max())
    return np.linspace(lo, hi, bins + 1) if hi > lo else np.array([lo, hi])


def plot_placements(
    lines: Sequence[Mapping[str, Any]],
    base_xy: Sequence[float],
    held_out: set[int] | None,
    path: Path,
    title: str = "",
    bins: int = 3,
) -> Path:
    """The bank's `lines` (`bank.load`'s) drawn into `path` (png); `held_out` the placements held out (None: none)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    placements = [line["placement"] for line in lines]
    c = _coordinates(placements, base_xy)
    held_out = held_out or set()
    kinds = np.array(
        [
            "unanchored"
            if line["anchor"] is None
            else "held out"
            if int(line["placement"]["index"]) in held_out
            else "training"
            for line in lines
        ]
    )
    d_edges, h_edges = _edges(c["diameter"], bins), _edges(c["height"], bins)
    sizes_vary = len(d_edges) > 2 or len(h_edges) > 2
    rows = 1 + (len(h_edges) - 1 if sizes_vary else 0)
    cols = max(2, len(d_edges) - 1) if sizes_vary else 1
    fig = plt.figure(
        figsize=(3.6 * cols, 2.5 * rows + 1.2) if sizes_vary else (6.0, 4.6),
        facecolor=SURFACE,
        layout="constrained",
    )
    grid = fig.add_gridspec(rows, cols)
    half = max(1, cols // 2)

    table = fig.add_subplot(grid[0, :half] if sizes_vary else grid[0, 0])
    _scatter(table, c["side"], c["ahead"], kinds)
    _style(table, "to the side of the arm's base (mm, +y)", "ahead of it (mm)", "on the table, from above")
    if (
        not sizes_vary
    ):  # (alone: true to scale; in the grid the facets share its axes instead, rows kept short)
        table.set_aspect("equal", adjustable="box")
    if sizes_vary:
        sizes = fig.add_subplot(grid[0, half:])
        _scatter(sizes, c["diameter"], c["height"], kinds)
        _style(sizes, "diameter (mm)", "height (mm)", "the objects' sizes")
        for r in range(len(h_edges) - 1):  # (the tallest on top)
            row = len(h_edges) - 2 - r
            h_lo, h_hi = h_edges[row], h_edges[row + 1]
            top = row == len(h_edges) - 2
            in_h = (c["height"] >= h_lo) & ((c["height"] <= h_hi) if top else (c["height"] < h_hi))
            for k in range(len(d_edges) - 1):
                d_lo, d_hi = d_edges[k], d_edges[k + 1]
                last = k == len(d_edges) - 2
                in_d = (c["diameter"] >= d_lo) & ((c["diameter"] <= d_hi) if last else (c["diameter"] < d_hi))
                mask = in_h & in_d
                ax = fig.add_subplot(grid[1 + r, k], sharex=table, sharey=table)
                _scatter(ax, c["side"][mask], c["ahead"][mask], kinds[mask])
                _style(
                    ax,
                    "side (mm)" if r == len(h_edges) - 2 else "",
                    "ahead (mm)" if k == 0 else "",
                    f"diameter {d_lo:.0f}-{d_hi:.0f}, height {h_lo:.0f}-{h_hi:.0f} mm ({int(mask.sum())})",
                )
    counts = {kind: int((kinds == kind).sum()) for kind in STYLES}
    p0 = placements[0]
    one = (
        ""
        if sizes_vary
        else f"; one object: {2000 * p0['radius']:.0f} x {1000 * p0['height']:.0f} mm, {1000 * p0['mass_kg']:.0f} g"
    )
    fig.suptitle(
        f"{title}\n{len(lines)} placements: {counts['training']} training, {counts['held out']} held out, "
        f"{counts['unanchored']} unanchored{one}",
        fontsize=9,
        color=INK,
        x=0.02,
        ha="left",
    )
    handles, labels = table.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8, frameon=False, labelcolor=INK)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path
