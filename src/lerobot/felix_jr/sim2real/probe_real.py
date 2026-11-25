"""The vision probe (trained in simulation) on real wrist frames: how far its estimates are from making sense.

  uv run --extra felix_jr_sim python -m lerobot.felix_jr.sim2real.probe_real --heads <heads.pt> [--heads <heads.pt> ...]
      --out <dir> [--sessions bowl5 bowl6 bowl7 bowl8 ruler2] [--sheets attention attn_L9_q4] [--hub <dir>]

The real frames: the harness sessions' wrist frames (`harness/scratch/sessions/<name>/workspace.pkl`: the single
views and each wrist scan's), each with the head camera's cylinder fitted before it, prepared as the simulator renders
them (`docs/sim2real_grasp.md`): undistorted with the frame's calibrated lens, resized to 224 x 168. Every head of
every `--heads` file runs on them (`probe_train.load_heads`; DINOv2-S once a frame). Per head:

- **scan consistency** (no ground truth needed): a wrist scan's frames see one static object from a few wrist rolls;
  the head's estimates of its center, taken to the root frame through each frame's own pinch pose, should agree: the
  spread (the largest distance from their mean, mm), the median over the scans. The arm's own error between the
  frames is a few mm (`object_grasping.md`: the model's wrist roll 5 % off).
- **the radius** against the object's measured one (`MEASURED_DIAMETER_MM`; the sessions' notes);
- **the offset from the head camera's fit** in the pinch frame (median |x|, |y|, |z|, mm): a rough reference only (the
  head and the wrist cameras disagree by 1-3 cm: `object_grasping.md`, the wrist silhouettes).

Frames from the left arm are reported apart (the probe was trained on the right arm's camera, whose view the left's
mirrors). Writes `<out>/probe_real.json` and, for the `--sheets` heads, a sheet a session: each frame with the head's
cylinder (red) and the head camera's (blue), and the head's attention (summed over its queries).
"""

import argparse
import json
import pickle
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .probe_train import HeadSpec, decode, dinov2_small, features, head_inputs, load_heads

SESSIONS_DIR = Path(__file__).resolve().parents[1] / "harness" / "scratch" / "sessions"
SESSIONS = ("bowl5", "bowl6", "bowl7", "bowl8", "ruler2")
# the object each session's frames are of, measured (`docs/sessions.md`): bowl4's white bottle's body (its cap wider),
# bowl8's chapstick, ruler2's Aleve bottle (tapered, 50-65 mm)
MEASURED_DIAMETER_MM = {"bowl5": 29.0, "bowl6": 29.0, "bowl7": 29.0, "bowl8": 17.5, "ruler2": 58.0}
W, H = 224, 168


@dataclass
class RealFrame:
    """A real wrist frame prepared as the simulator renders (224 x 168, undistorted), with what is known about it."""

    session: str
    label: str
    scan: str | None  # the wrist scan it belongs to, if any
    arm: str
    rgb: np.ndarray  # (H, W, 3) uint8
    k: np.ndarray  # (3, 3): its pinhole at 224 x 168
    cam_from_pinch: np.ndarray  # (4, 4)
    root_from_pinch: np.ndarray  # (4, 4)
    head_fit: tuple[
        np.ndarray, np.ndarray, float, float
    ]  # the head camera's cylinder: center, axis (pinch), r, h


def real_frames(sessions: Sequence[str], root: Path = SESSIONS_DIR) -> list[RealFrame]:
    """The sessions' wrist frames (the single views and the scans'), each after a head fit of a cylinder."""
    import cv2

    from lerobot.felix_jr.harness.primitives import Cylinder

    out = []
    for name in sessions:
        with open(root / name / "workspace.pkl", "rb") as fh:
            nodes = list(pickle.load(fh)["ws"].nodes.values())  # nosec B301: our own sessions' files
        fits = sorted(
            (n.obj.capture.timestamp, n)
            for n in nodes
            if type(n.obj).__name__ == "Object"
            and n.obj.capture is not None
            and any(isinstance(p, Cylinder) for p in n.obj.parts.values())
        )
        views: list[tuple[str, str | None, Any]] = []
        for n in nodes:
            if type(n.obj).__name__ == "WristView":
                views.append((n.id, None, n.obj))
            elif n.call == "wrist_scan":
                views += [(f"{n.id}.{i}", n.id, v) for i, v in enumerate(n.obj.views)]
        for label, scan, view in views:
            before = [n for t, n in fits if t <= view.timestamp]
            if not before:
                continue
            cyl = next(p for p in before[-1].obj.parts.values() if isinstance(p, Cylinder))
            k640, dist = view.camera.matrix(), np.asarray(view.camera.coeffs)
            und = cv2.undistort(np.asarray(view.color), k640, dist)
            k = k640.copy()
            k[:2] *= W / view.camera.width
            pinch_from_root = np.linalg.inv(view.root_from_pinch)
            center = cyl.bottom_root[0] + cyl.axis_root[0] * float(cyl.height[0]) / 2
            fit = (
                pinch_from_root[:3, :3] @ center + pinch_from_root[:3, 3],
                pinch_from_root[:3, :3] @ cyl.axis_root[0],
                float(cyl.radius[0]),
                float(cyl.height[0]),
            )
            out.append(
                RealFrame(
                    name,
                    label,
                    scan,
                    view.arm,
                    cv2.resize(und, (W, H), interpolation=cv2.INTER_AREA),
                    k,
                    np.linalg.inv(view.pinch_from_camera),
                    np.asarray(view.root_from_pinch),
                    fit,
                )
            )
    return out


def estimate(
    frames: Sequence[RealFrame],
    heads: dict[str, torch.nn.Module],
    specs: dict[str, HeadSpec],
    scale: torch.Tensor,
    dino: torch.nn.Module,
    device: str,
    batch: int = 32,
) -> dict[str, dict[str, np.ndarray]]:
    """Each head's estimates for `frames`: center, axis, size (pinch frame, m), in view; its attention (N, patches)."""
    blocks = tuple(sorted({b for s in specs.values() for b in s.blocks}))
    out: dict[str, dict[str, list[np.ndarray]]] = {n: {} for n in heads}
    with torch.no_grad():
        for start in range(0, len(frames), batch):
            rgb = torch.from_numpy(np.stack([f.rgb for f in frames[start : start + batch]])).to(device)
            feats = features(dino, rgb, blocks)
            for name, head in heads.items():
                d = decode(head(*head_inputs(feats, specs[name])), scale)
                weights = getattr(head, "last_weights", None)
                if weights is not None:
                    d["attention"] = weights.float().sum(dim=1)
                for k, v in d.items():
                    out[name].setdefault(k, []).append(v.float().cpu().numpy())
    return {n: {k: np.concatenate(v) for k, v in d.items()} for n, d in out.items()}


def scores(frames: Sequence[RealFrame], est: dict[str, np.ndarray]) -> dict[str, Any]:
    """One head's numbers on `frames` (the module docstring)."""
    center = est["center"]
    root = np.array(
        [
            f.root_from_pinch[:3, :3] @ c + f.root_from_pinch[:3, 3]
            for f, c in zip(frames, center, strict=True)
        ]
    )
    spreads = []
    for scan in sorted({f.scan for f in frames if f.scan is not None}):
        idx = [i for i, f in enumerate(frames) if f.scan == scan]
        if len(idx) >= 3:
            pts = root[idx]
            spreads.append(float(1000 * np.linalg.norm(pts - pts.mean(axis=0), axis=1).max()))
    radius_err = np.array(
        [
            1000 * r - MEASURED_DIAMETER_MM[f.session] / 2
            for f, r in zip(frames, est["size"][:, 0], strict=True)
        ]
    )
    offset = 1000 * np.abs(center - np.array([f.head_fit[0] for f in frames]))
    return {
        "frames": len(frames),
        "scan_spread_mm": {"median": float(np.median(spreads)), "n_scans": len(spreads)} if spreads else None,
        "radius_error_mm": {
            "median": float(np.median(radius_err)),
            "median_abs": float(np.median(np.abs(radius_err))),
        },
        "height_mm_median": float(1000 * np.median(est["size"][:, 1])),
        "offset_from_head_fit_mm": dict(
            zip(("x", "y", "z"), np.median(offset, axis=0).round(1).tolist(), strict=True)
        ),
        "in_view": float(np.mean(est["in_view"])),
    }


def sheet(frames: Sequence[RealFrame], est: dict[str, np.ndarray], title: str, up: int = 2) -> np.ndarray:
    """Each frame: the head's cylinder (red) and the head camera's (blue) drawn on it, its attention beside it."""
    import cv2

    def ring(
        img: np.ndarray,
        center: np.ndarray,
        axis: np.ndarray,
        r: float,
        h: float,
        f: RealFrame,
        color: tuple[int, int, int],
    ) -> None:
        a = axis / max(np.linalg.norm(axis), 1e-9)
        u = np.cross(a, [1.0, 0.0, 0.0])
        u = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-6 else np.array([0.0, 1.0, 0.0])
        v = np.cross(a, u)
        t = np.linspace(0, 2 * np.pi, 40)
        k = f.k.copy()
        k[:2] *= up
        for end in (-0.5, 0.5):
            pts = center + end * h * a + r * (np.outer(np.cos(t), u) + np.outer(np.sin(t), v))
            c = pts @ f.cam_from_pinch[:3, :3].T + f.cam_from_pinch[:3, 3]
            if (c[:, 2] > 1e-3).all():
                uv = np.column_stack(
                    [k[0, 0] * c[:, 0] / c[:, 2] + k[0, 2], k[1, 1] * c[:, 1] / c[:, 2] + k[1, 2]]
                )
                cv2.polylines(img, [np.round(uv).astype(np.int32)], True, color, 1, cv2.LINE_AA)
        c = center @ f.cam_from_pinch[:3, :3].T + f.cam_from_pinch[:3, 3]
        if c[2] > 1e-3:
            cv2.circle(
                img,
                (int(k[0, 0] * c[0] / c[2] + k[0, 2]), int(k[1, 1] * c[1] / c[2] + k[1, 2])),
                3,
                color,
                -1,
            )

    tiles = []
    for i, f in enumerate(frames):
        img = cv2.cvtColor(cv2.resize(f.rgb, (W * up, H * up)), cv2.COLOR_RGB2BGR)
        ring(img, *f.head_fit, f, (255, 120, 0))
        ring(
            img,
            est["center"][i],
            est["axis"][i],
            float(est["size"][i, 0]),
            float(est["size"][i, 1]),
            f,
            (0, 0, 255),
        )
        r, h = 1000 * est["size"][i]
        for col, wd in (((0, 0, 0), 3), ((255, 255, 255), 1)):
            cv2.putText(
                img,
                f"{f.session} {f.label} ({f.arm}) {title}",
                (5, 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                col,
                wd,
            )
            cv2.putText(
                img,
                f"red: r {r:.0f} h {h:.0f} mm (true d {MEASURED_DIAMETER_MM[f.session]:.0f})",
                (5, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                col,
                wd,
            )
        tile = img
        if "attention" in est:
            m = est["attention"][i].reshape(12, 16)
            m = cv2.resize(m / max(m.max(), 1e-9), (W * up, H * up), interpolation=cv2.INTER_CUBIC).clip(0, 1)
            heat = cv2.applyColorMap((255 * m).astype(np.uint8), cv2.COLORMAP_INFERNO)
            tile = np.hstack(
                [
                    img,
                    cv2.addWeighted(
                        cv2.cvtColor(cv2.resize(f.rgb, (W * up, H * up)), cv2.COLOR_RGB2BGR),
                        0.45,
                        heat,
                        0.55,
                        0,
                    ),
                ]
            )
        tiles.append(tile)
    cols = 2
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    return np.vstack([np.hstack(tiles[j : j + cols]) for j in range(0, len(tiles), cols)])


def main(argv: list[str] | None = None) -> None:
    """The command line (the module docstring)."""
    import cv2

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--heads", type=Path, action="append", required=True, help="a probe_train heads.pt (again: more)"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sessions", nargs="+", default=list(SESSIONS))
    parser.add_argument(
        "--sheets", nargs="*", default=["attention"], help="heads to draw (default: attention)"
    )
    parser.add_argument("--hub", type=Path, default=None)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args(argv)
    frames = real_frames(args.sessions)
    right = [f for f in frames if f.arm == "right"]
    left = [f for f in frames if f.arm != "right"]
    dino = dinov2_small(args.device, args.hub)
    args.out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"frames": {"right": len(right), "left": len(left)}, "heads": {}}
    for path in args.heads:
        heads, specs, scale = load_heads(path, args.device)
        tag = path.parent.name
        for group, fs in (("right", right), ("left", left)):
            if not fs:
                continue
            est = estimate(fs, heads, specs, scale, dino, args.device)
            for name, e in est.items():
                report["heads"].setdefault(f"{tag}/{name}", {})[group] = scores(fs, e)
                if name in args.sheets and group == "right":
                    for session in sorted({f.session for f in fs}):
                        idx = [i for i, f in enumerate(fs) if f.session == session]
                        img = sheet([fs[i] for i in idx], {k: v[idx] for k, v in e.items()}, f"{tag}/{name}")
                        cv2.imwrite(
                            str(args.out / f"{tag}_{name}_{session}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85]
                        )
    (args.out / "probe_real.json").write_text(json.dumps(report, indent=1))
    print(f"{len(right)} right-arm frames, {len(left)} left-arm")
    print(
        f"{'head':34s} {'scan spread':>12s} {'radius err':>11s} {'height':>7s}  offset from the head fit x/y/z"
    )
    for name, by in report["heads"].items():
        s = by.get("right")
        if s is None:
            continue
        spread = s["scan_spread_mm"]["median"] if s["scan_spread_mm"] else float("nan")
        o = s["offset_from_head_fit_mm"]
        print(
            f"{name:34s} {spread:10.1f}mm {s['radius_error_mm']['median']:+9.1f}mm {s['height_mm_median']:5.0f}mm"
            f"  {o['x']:.0f}/{o['y']:.0f}/{o['z']:.0f} mm"
        )


if __name__ == "__main__":
    main()
