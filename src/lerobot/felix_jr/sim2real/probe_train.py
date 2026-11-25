"""The vision probe: heads on a frozen DINOv2-S, trained to tell the object's pose in the pinch frame from a wrist frame.

  python -m lerobot.felix_jr.sim2real.probe_train --train <probe_data dir> --test <probe_data dir> --out <dir>
      [--epochs 20] [--batch 512] [--hub <torch.hub dir>] [--heads basic|grid] [--augment]

The frames (`probe_data`: 224 x 168, 16 x 12 patches of 14 px) go through DINOv2 ViT-S/14 (`torch.hub`,
facebookresearch/dinov2, frozen, bf16) once a batch, its tokens taken after the blocks the heads need (normalized as
its last are: `features`); the heads (`HeadSpec`) read them, trained side by side on the same batches, each with its own
optimizer:

- `linear`: the class token, a linear map;
- `mean`: the class token and the patch tokens' mean, a small MLP;
- `attention`: learned queries attending over the patch tokens (a learned position embedding added), then a small MLP
  (its attention maps say where it looks); over one block's tokens or several's side by side.

`--heads basic`: the three on the last block (4 queries); `grid` (`GRID_SPECS`): those and attention over blocks 6, 9,
12 and 9-12 together, with 1, 4 and 16 queries. Each predicts the object's center (3) and axis (3) in the pinch frame
(`in_pinch`), its radius and height, and whether it is in view (the segmentation's share above `VISIBLE_SHARE`).
Losses: Huber on the center and size, standardized by the training set's spread; 1 - cosine on the axis; those only
for frames with the object in view; cross-entropy on in view. `--augment`: the training frames' colours jittered
(brightness, contrast, saturation, each channel's gain), noise and now and then a blur (`augment`).
Reported on the held-out placements (`--test`), the object in view: the center's error (mm; median, 90th
percentile), by start pool and by distance from the pinch point; the axis's (deg); the radius's and height's (mm); in
view's accuracy. Writes `<out>/heads.pt` (the heads and their specs, the standardization, the settings: `load_heads`),
`<out>/metrics.json`.
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .start_pool import POOLS

DIM = 384  # DINOv2-S's width
BLOCKS = 12  # its depth
GRID = (12, 16)  # (rows, columns) of patches at 168 x 224
QUERIES = 4
VISIBLE_SHARE = 0.002  # in view: at least this share of the frame is the object (~75 px at 224 x 168)
OUT = 3 + 3 + 2 + 1  # center, axis, radius and height, in view (a logit)
MEAN = (0.485, 0.456, 0.406)  # ImageNet's, as DINOv2 was trained
STD = (0.229, 0.224, 0.225)
NEAR_M = (0.03, 0.06, 0.12)  # the error reported by the center's distance from the pinch point, under each


@dataclass(frozen=True)
class HeadSpec:
    """A head: its kind (linear, mean, attention), the blocks whose tokens it reads (1-based), its queries."""

    name: str
    kind: str
    blocks: tuple[int, ...] = (BLOCKS,)
    queries: int = QUERIES


BASIC_SPECS = (
    HeadSpec("linear", "linear"),
    HeadSpec("mean", "mean"),
    HeadSpec("attention", "attention"),
)
GRID_SPECS = BASIC_SPECS + tuple(
    HeadSpec(f"attn_L{'-'.join(map(str, (b[0], b[-1]) if len(b) > 1 else b))}_q{q}", "attention", b, q)
    for b in ((6,), (9,), (12,), (9, 10, 11, 12))
    for q in (1, 4, 16)
    if not (b == (12,) and q == QUERIES)  # (the basic attention head)
)
SPECS = {"basic": BASIC_SPECS, "grid": GRID_SPECS}
# the heads of files written before the specs were (their names: the basic ones)
LEGACY = {s.name: s for s in BASIC_SPECS}


def dinov2_small(device: str, hub: Path | None = None) -> nn.Module:
    """DINOv2 ViT-S/14 from torch.hub, frozen, in bf16 on `device` (CPU: float32)."""
    if hub is not None:
        torch.hub.set_dir(str(hub))
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")  # nosec B614: the published model
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = model.to(device, dtype).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _normalized(dino: nn.Module, rgb: torch.Tensor) -> torch.Tensor:
    """uint8 frames (B, H, W, 3) as DINOv2 takes them (B, 3, H, W), in its dtype."""
    dtype = next(dino.parameters()).dtype
    x = rgb.permute(0, 3, 1, 2).to(dtype).div(255.0)
    mean = torch.tensor(MEAN, device=x.device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x.device, dtype=dtype).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def tokens(dino: nn.Module, rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(class token (B, DIM), patch tokens (B, rows * columns, DIM)), float32, the last block's, of uint8 frames."""
    out = dino.forward_features(_normalized(dino, rgb))
    return out["x_norm_clstoken"].float(), out["x_norm_patchtokens"].float()


@torch.no_grad()
def features(
    dino: nn.Module, rgb: torch.Tensor, blocks: tuple[int, ...]
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Each of `blocks` (1-based): (class token (B, DIM), patch tokens (B, P, DIM)), float32, in one pass.

    Normalized as the last block's are.
    """
    wanted = sorted(set(blocks))
    outs = dino.get_intermediate_layers(
        _normalized(dino, rgb), n=[b - 1 for b in wanted], return_class_token=True, norm=True
    )
    return {b: (cls.float(), patches.float()) for b, (patches, cls) in zip(wanted, outs, strict=True)}


def head_inputs(
    feats: dict[int, tuple[torch.Tensor, torch.Tensor]], spec: HeadSpec
) -> tuple[torch.Tensor, torch.Tensor]:
    """(class token, patch tokens) for `spec`: its blocks' side by side (B, len(blocks) * DIM)."""
    cls = torch.cat([feats[b][0] for b in spec.blocks], dim=-1)
    patches = torch.cat([feats[b][1] for b in spec.blocks], dim=-1)
    return cls, patches


class LinearHead(nn.Module):
    """The class token, a linear map."""

    def __init__(self, width: int = DIM) -> None:
        """The map."""
        super().__init__()
        self.out = nn.Linear(width, OUT)

    def forward(self, cls: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        """The outputs (B, OUT)."""
        return self.out(cls)


class MeanHead(nn.Module):
    """The class token and the patch tokens' mean, a small MLP."""

    def __init__(self, width: int = DIM, hidden: int = 256) -> None:
        """The MLP."""
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * width, hidden), nn.GELU(), nn.Linear(hidden, OUT))

    def forward(self, cls: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        """The outputs (B, OUT)."""
        return self.mlp(torch.cat([cls, patches.mean(dim=1)], dim=1))


class AttentionHead(nn.Module):
    """Learned queries attending over the patch tokens (a learned position embedding added), a small MLP."""

    def __init__(self, width: int = DIM, queries: int = QUERIES, hidden: int = 256, heads: int = 6) -> None:
        """The queries, the position embedding, the attention and the MLP."""
        super().__init__()
        self.queries = nn.Parameter(0.02 * torch.randn(queries, width))
        self.position = nn.Parameter(0.02 * torch.randn(GRID[0] * GRID[1], width))
        self.norm = nn.LayerNorm(width)
        self.attend = nn.MultiheadAttention(width, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(queries * width, hidden), nn.GELU(), nn.Linear(hidden, OUT))
        self.last_weights: torch.Tensor | None = None  # (B, queries, patches): the last forward's

    def forward(self, cls: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        """The outputs (B, OUT)."""
        keys = self.norm(patches + self.position)
        q = self.queries.expand(len(patches), -1, -1)
        pooled, weights = self.attend(q, keys, keys, need_weights=True, average_attn_weights=True)
        self.last_weights = weights.detach()
        return self.mlp(pooled.flatten(1))


def build(spec: HeadSpec) -> nn.Module:
    """The head `spec` describes."""
    width = DIM * len(spec.blocks)
    if spec.kind == "linear":
        return LinearHead(width)
    if spec.kind == "mean":
        return MeanHead(width)
    if spec.kind == "attention":
        return AttentionHead(width, spec.queries)
    raise ValueError(f"no head kind {spec.kind!r}")


# the basic heads by name (their defaults: the last block, 4 queries)
HEADS = {"linear": LinearHead, "mean": MeanHead, "attention": AttentionHead}


def load_heads(path: Path, device: str) -> tuple[dict[str, nn.Module], dict[str, HeadSpec], torch.Tensor]:
    """(heads, their specs, the standardization) of a `heads.pt`, on `device`, in eval mode."""
    saved = torch.load(path, map_location=device)
    raw = saved.get("specs")
    specs = (
        {n: HeadSpec(**{**s, "blocks": tuple(s["blocks"])}) for n, s in raw.items()}
        if raw
        else {n: LEGACY[n] for n in saved["heads"]}
    )
    heads = {}
    for name, spec in specs.items():
        head = build(spec).to(device).eval()
        head.load_state_dict(saved["heads"][name])
        heads[name] = head
    return heads, specs, saved["scale"].to(device)


def augment(rgb: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """uint8 frames (B, H, W, 3) with their colours jittered, noise added and a tenth blurred.

    Brightness, contrast, saturation and each channel's gain: what a cheap camera's exposure and white balance do.
    """
    b = len(rgb)
    dev = rgb.device

    def u(lo: float, hi: float, *shape: int) -> torch.Tensor:
        return lo + (hi - lo) * torch.rand(b, *shape, device=dev, generator=generator)

    x = rgb.float() / 255.0
    x = x * u(0.7, 1.3, 1, 1, 1)  # brightness
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * u(0.7, 1.3, 1, 1, 1) + mean  # contrast
    grey = x.mean(dim=3, keepdim=True)
    x = (x - grey) * u(0.6, 1.4, 1, 1, 1) + grey  # saturation
    x = x * u(0.9, 1.1, 1, 1, 3)  # each channel's gain (white balance)
    x = x + u(0.0, 0.025, 1, 1, 1) * torch.randn(x.shape, device=dev, generator=generator)
    blur = torch.rand(b, device=dev, generator=generator) < 0.1
    if blur.any():
        k = torch.full((3, 1, 3, 3), 1.0 / 9.0, device=dev)
        y = x[blur].permute(0, 3, 1, 2)
        y = nn.functional.conv2d(nn.functional.pad(y, (1, 1, 1, 1), mode="replicate"), k, groups=3)
        x[blur] = y.permute(0, 2, 3, 1)
    return (x.clamp(0, 1) * 255).round().to(torch.uint8)


class ProbeData:
    """A `probe_data` folder: its frames (memory-mapped) and labels."""

    def __init__(self, folder: Path) -> None:
        """The frames and labels of `folder`."""
        self.frames = np.load(folder / "frames.npy", mmap_mode="r")
        labels = np.load(folder / "labels.npz")
        self.labels = {k: labels[k] for k in labels.files if k not in ("pools", "summary")}
        self.n = len(self.labels["in_pinch"])
        self.visible = self.labels["visible"] > VISIBLE_SHARE

    def targets(self) -> np.ndarray:
        """(N, 8): the center, the axis, the radius and height."""
        return np.concatenate([self.labels["in_pinch"], self.labels["size"]], axis=1).astype(np.float32)

    def batch(self, idx: np.ndarray, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(frames (B, H, W, 3) uint8, targets (B, 8), in view (B,)) on `device`."""
        idx = np.sort(idx)  # (the memory map read in order)
        rgb = torch.from_numpy(np.ascontiguousarray(self.frames[idx])).to(device)
        target = torch.from_numpy(self.targets()[idx]).to(device)
        return rgb, target, torch.from_numpy(self.visible[idx]).to(device)


def loss(out: torch.Tensor, target: torch.Tensor, seen: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """The objective (the module docstring); `scale`: the targets' standardization (mean (8,), std (8,))."""
    mean, std = scale
    in_view = nn.functional.binary_cross_entropy_with_logits(out[:, 8], seen.float())
    if not seen.any():
        return in_view
    o, t = out[seen], target[seen]
    z = (t - mean) / std
    center = nn.functional.huber_loss(o[:, :3], z[:, :3], delta=1.0)
    size = nn.functional.huber_loss(o[:, 6:8], z[:, 6:8], delta=1.0)
    axis = (1 - nn.functional.cosine_similarity(o[:, 3:6], t[:, 3:6], dim=1)).mean()
    return center + axis + size + in_view


def decode(out: torch.Tensor, scale: torch.Tensor) -> dict[str, torch.Tensor]:
    """The heads' outputs in meters: center (B, 3), axis (B, 3, unit), radius and height (B, 2), in view (B,)."""
    mean, std = scale
    center = out[:, :3] * std[:3] + mean[:3]
    axis = nn.functional.normalize(out[:, 3:6], dim=1)
    size = out[:, 6:8] * std[6:8] + mean[6:8]
    return {"center": center, "axis": axis, "size": size, "in_view": out[:, 8] > 0}


def _blocks(specs: dict[str, HeadSpec]) -> tuple[int, ...]:
    return tuple(sorted({b for s in specs.values() for b in s.blocks}))


def evaluate(
    dino: nn.Module,
    heads: dict[str, nn.Module],
    specs: dict[str, HeadSpec],
    data: ProbeData,
    scale: torch.Tensor,
    device: str,
    batch: int,
) -> dict[str, Any]:
    """Each head's errors on `data` (the module docstring)."""
    preds: dict[str, list[dict[str, np.ndarray]]] = {name: [] for name in heads}
    for h in heads.values():
        h.eval()
    blocks = _blocks(specs)
    with torch.no_grad():
        for start in range(0, data.n, batch):
            idx = np.arange(start, min(start + batch, data.n))
            rgb, _, _ = data.batch(idx, device)
            feats = features(dino, rgb, blocks)
            for name, h in heads.items():
                out = decode(h(*head_inputs(feats, specs[name])), scale)
                preds[name].append({k: v.cpu().numpy() for k, v in out.items()})
    target = data.targets()
    seen = data.visible
    dist = np.linalg.norm(target[:, :3], axis=1)
    pool = data.labels["pool"]
    held = data.labels["held"].astype(bool)
    result: dict[str, Any] = {}
    for name, chunks in preds.items():
        p = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
        err_mm = 1000 * np.linalg.norm(p["center"] - target[:, :3], axis=1)
        cos = np.clip(np.sum(p["axis"] * target[:, 3:6], axis=1), -1, 1)
        axis_deg = np.degrees(np.arccos(cos))
        radius_mm = 1000 * np.abs(p["size"][:, 0] - target[:, 6])
        height_mm = 1000 * np.abs(p["size"][:, 1] - target[:, 7])

        def stats(mask: np.ndarray, err: np.ndarray = err_mm) -> dict[str, float]:
            m = mask & seen
            if not m.any():
                return {"n": 0}
            return {
                "n": int(m.sum()),
                "median": float(np.median(err[m])),
                "p90": float(np.percentile(err[m], 90)),
            }

        everything = np.ones(data.n, bool)
        result[name] = {
            "center_mm": stats(everything),
            "center_mm_not_held": stats(~held),
            "by_pool": {pl: stats(pool == i) for i, pl in enumerate(POOLS)},
            "by_distance": {f"under_{int(100 * d)}cm": stats(dist < d) for d in NEAR_M},
            "axis_deg": stats(everything, axis_deg),
            "radius_mm": stats(everything, radius_mm),
            "height_mm": stats(everything, height_mm),
            "in_view_accuracy": float(np.mean(p["in_view"] == seen)),
        }
    for h in heads.values():
        h.train()
    return result


def train(
    train_dir: Path,
    test_dir: Path,
    out: Path,
    epochs: int = 20,
    batch: int = 512,
    lr: float = 3e-4,
    device: str = "cuda:0",
    hub: Path | None = None,
    seed: int = 0,
    specs: tuple[HeadSpec, ...] = BASIC_SPECS,
    augmented: bool = False,
) -> dict[str, Any]:
    """The heads trained (the module docstring); returns the metrics."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dino = dinov2_small(device, hub)
    data, test = ProbeData(train_dir), ProbeData(test_dir)
    seen_targets = data.targets()[data.visible]
    mean = torch.tensor(seen_targets.mean(axis=0), device=device)
    std = torch.tensor(seen_targets.std(axis=0) + 1e-6, device=device)
    std[3:6] = 1.0  # (the axis: a unit vector, compared by angle)
    mean[3:6] = 0.0
    scale = torch.stack([mean, std])
    by_name = {s.name: s for s in specs}
    heads = {s.name: build(s).to(device) for s in specs}
    blocks = _blocks(by_name)
    steps = epochs * (data.n // batch)
    optims = {name: torch.optim.AdamW(h.parameters(), lr=lr, weight_decay=1e-4) for name, h in heads.items()}
    sched = {name: torch.optim.lr_scheduler.CosineAnnealingLR(o, max(steps, 1)) for name, o in optims.items()}
    out.mkdir(parents=True, exist_ok=True)
    history = []
    started = time.perf_counter()
    for epoch in range(epochs):
        order = rng.permutation(data.n)
        losses = dict.fromkeys(heads, 0.0)
        nb = max(data.n // batch, 1)
        for b in range(nb):
            rgb, target, seen = data.batch(order[b * batch : (b + 1) * batch], device)
            if augmented:
                rgb = augment(rgb)
            feats = features(dino, rgb, blocks)
            for name, h in heads.items():
                value = loss(h(*head_inputs(feats, by_name[name])), target, seen, scale)
                optims[name].zero_grad(set_to_none=True)
                value.backward()
                optims[name].step()
                sched[name].step()
                losses[name] += float(value.detach()) / nb
        metrics = evaluate(dino, heads, by_name, test, scale, device, batch)
        history.append({"epoch": epoch, "loss": losses, "test": metrics})
        line = ", ".join(
            f"{n} {m['center_mm'].get('median', float('nan')):.1f}/{m['center_mm'].get('p90', float('nan')):.1f}"
            for n, m in metrics.items()
        )
        print(
            f"epoch {epoch + 1}/{epochs} ({time.perf_counter() - started:.0f} s): held-out center mm {line}",
            flush=True,
        )
    torch.save(
        {
            "heads": {n: h.state_dict() for n, h in heads.items()},
            "specs": {n: asdict(s) for n, s in by_name.items()},
            "scale": scale.cpu(),
            "settings": {
                "train": str(train_dir),
                "test": str(test_dir),
                "epochs": epochs,
                "batch": batch,
                "lr": lr,
                "augment": augmented,
            },
        },
        out / "heads.pt",
    )
    result = {"final": history[-1]["test"], "history": history}
    (out / "metrics.json").write_text(json.dumps(result, indent=1))
    return result


def main(argv: list[str] | None = None) -> None:
    """The command line (the module docstring)."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hub", type=Path, default=None, help="torch.hub's folder (default: its own)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--heads", choices=sorted(SPECS), default="basic", help="which heads to train")
    parser.add_argument(
        "--augment", action="store_true", help="jitter the training frames' colours (`augment`)"
    )
    args = parser.parse_args(argv)
    result = train(
        args.train,
        args.test,
        args.out,
        args.epochs,
        args.batch,
        args.lr,
        args.device,
        args.hub,
        args.seed,
        SPECS[args.heads],
        args.augment,
    )
    summary = {
        n: {"center_mm": m["center_mm"], "center_mm_not_held": m["center_mm_not_held"]}
        for n, m in result["final"].items()
    }
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
