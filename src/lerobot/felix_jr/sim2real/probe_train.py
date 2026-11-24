"""The vision probe: heads on a frozen DINOv2-S, trained to tell the object's pose in the pinch frame from a wrist frame.

  python -m lerobot.felix_jr.sim2real.probe_train --train <probe_data dir> --test <probe_data dir> --out <dir>
      [--epochs 20] [--batch 512] [--hub <torch.hub dir>]

The frames (`probe_data`: 224 x 168, 16 x 12 patches of 14 px) go through DINOv2 ViT-S/14 (`torch.hub`,
facebookresearch/dinov2, frozen, bf16); three heads read its tokens, trained side by side on the same batches, each
with its own optimizer:

- `linear`: the class token, a linear map;
- `mean`: the class token and the patch tokens' mean, a small MLP;
- `attention`: `QUERIES` learned queries attending over the patch tokens (a learned position embedding added), then a
  small MLP (its attention maps say where it looks).

Each predicts the object's center (3) and axis (3) in the pinch frame (`in_pinch`), its radius and height, and whether
it is in view (the segmentation's share above `VISIBLE_SHARE`). Losses: Huber on the center and size, standardized by
the training set's spread; 1 - cosine on the axis; those only for frames with the object in view; cross-entropy on
in view. Reported on the held-out placements (`--test`), the object in view: the center's error (mm; median, 90th
percentile), by start pool and by distance from the pinch point; the axis's (deg); the radius's (mm); in view's
accuracy. Writes `<out>/heads.pt` (the heads, the standardization, the settings), `<out>/metrics.json`.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .start_pool import POOLS

DIM = 384  # DINOv2-S's width
GRID = (12, 16)  # (rows, columns) of patches at 168 x 224
QUERIES = 4
VISIBLE_SHARE = 0.002  # in view: at least this share of the frame is the object (~75 px at 224 x 168)
OUT = 3 + 3 + 2 + 1  # center, axis, radius and height, in view (a logit)
MEAN = (0.485, 0.456, 0.406)  # ImageNet's, as DINOv2 was trained
STD = (0.229, 0.224, 0.225)
NEAR_M = (0.03, 0.06, 0.12)  # the error reported by the center's distance from the pinch point, under each


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


@torch.no_grad()
def tokens(dino: nn.Module, rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(class token (B, DIM), patch tokens (B, rows * columns, DIM)), float32, of uint8 frames (B, H, W, 3)."""
    dtype = next(dino.parameters()).dtype
    x = rgb.permute(0, 3, 1, 2).to(dtype).div(255.0)
    mean = torch.tensor(MEAN, device=x.device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x.device, dtype=dtype).view(1, 3, 1, 1)
    out = dino.forward_features((x - mean) / std)
    return out["x_norm_clstoken"].float(), out["x_norm_patchtokens"].float()


class LinearHead(nn.Module):
    """The class token, a linear map."""

    def __init__(self) -> None:
        """The map."""
        super().__init__()
        self.out = nn.Linear(DIM, OUT)

    def forward(self, cls: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        """The outputs (B, OUT)."""
        return self.out(cls)


class MeanHead(nn.Module):
    """The class token and the patch tokens' mean, a small MLP."""

    def __init__(self, hidden: int = 256) -> None:
        """The MLP."""
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * DIM, hidden), nn.GELU(), nn.Linear(hidden, OUT))

    def forward(self, cls: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        """The outputs (B, OUT)."""
        return self.mlp(torch.cat([cls, patches.mean(dim=1)], dim=1))


class AttentionHead(nn.Module):
    """`QUERIES` learned queries attending over the patch tokens (a learned position embedding added), a small MLP."""

    def __init__(self, hidden: int = 256, heads: int = 6) -> None:
        """The queries, the position embedding, the attention and the MLP."""
        super().__init__()
        self.queries = nn.Parameter(0.02 * torch.randn(QUERIES, DIM))
        self.position = nn.Parameter(0.02 * torch.randn(GRID[0] * GRID[1], DIM))
        self.norm = nn.LayerNorm(DIM)
        self.attend = nn.MultiheadAttention(DIM, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(QUERIES * DIM, hidden), nn.GELU(), nn.Linear(hidden, OUT))
        self.last_weights: torch.Tensor | None = None  # (B, QUERIES, patches): the last forward's

    def forward(self, cls: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        """The outputs (B, OUT)."""
        keys = self.norm(patches + self.position)
        q = self.queries.expand(len(patches), -1, -1)
        pooled, weights = self.attend(q, keys, keys, need_weights=True, average_attn_weights=True)
        self.last_weights = weights.detach()
        return self.mlp(pooled.flatten(1))


HEADS = {"linear": LinearHead, "mean": MeanHead, "attention": AttentionHead}


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


def evaluate(
    dino: nn.Module,
    heads: dict[str, nn.Module],
    data: ProbeData,
    scale: torch.Tensor,
    device: str,
    batch: int,
) -> dict[str, Any]:
    """Each head's errors on `data` (the module docstring)."""
    preds: dict[str, list[dict[str, np.ndarray]]] = {name: [] for name in heads}
    for h in heads.values():
        h.eval()
    with torch.no_grad():
        for start in range(0, data.n, batch):
            idx = np.arange(start, min(start + batch, data.n))
            rgb, _, _ = data.batch(idx, device)
            cls, patches = tokens(dino, rgb)
            for name, h in heads.items():
                preds[name].append({k: v.cpu().numpy() for k, v in decode(h(cls, patches), scale).items()})
    target = data.targets()
    seen = data.visible
    dist = np.linalg.norm(target[:, :3], axis=1)
    pool = data.labels["pool"]
    out: dict[str, Any] = {}
    for name, chunks in preds.items():
        p = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
        err_mm = 1000 * np.linalg.norm(p["center"] - target[:, :3], axis=1)
        cos = np.clip(np.sum(p["axis"] * target[:, 3:6], axis=1), -1, 1)
        axis_deg = np.degrees(np.arccos(cos))
        radius_mm = 1000 * np.abs(p["size"][:, 0] - target[:, 6])

        def stats(mask: np.ndarray, err: np.ndarray = err_mm) -> dict[str, float]:
            m = mask & seen
            if not m.any():
                return {"n": 0}
            return {
                "n": int(m.sum()),
                "median": float(np.median(err[m])),
                "p90": float(np.percentile(err[m], 90)),
            }

        out[name] = {
            "center_mm": stats(np.ones(data.n, bool)),
            "by_pool": {pl: stats(pool == i) for i, pl in enumerate(POOLS)},
            "by_distance": {f"under_{int(100 * d)}cm": stats(dist < d) for d in NEAR_M},
            "axis_deg": stats(np.ones(data.n, bool), axis_deg),
            "radius_mm": stats(np.ones(data.n, bool), radius_mm),
            "in_view_accuracy": float(np.mean(p["in_view"] == seen)),
        }
    for h in heads.values():
        h.train()
    return out


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
    heads = {name: cls().to(device) for name, cls in HEADS.items()}
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
            cls, patches = tokens(dino, rgb)
            for name, h in heads.items():
                value = loss(h(cls, patches), target, seen, scale)
                optims[name].zero_grad(set_to_none=True)
                value.backward()
                optims[name].step()
                sched[name].step()
                losses[name] += float(value) / nb
        metrics = evaluate(dino, heads, test, scale, device, batch)
        history.append({"epoch": epoch, "loss": losses, "test": metrics})
        line = ", ".join(
            f"{n} {m['center_mm'].get('median', float('nan')):.1f}/{m['center_mm'].get('p90', float('nan')):.1f} mm"
            for n, m in metrics.items()
        )
        print(
            f"epoch {epoch + 1}/{epochs} ({time.perf_counter() - started:.0f} s): held-out center {line}",
            flush=True,
        )
    torch.save(
        {
            "heads": {n: h.state_dict() for n, h in heads.items()},
            "scale": scale.cpu(),
            "settings": {
                "train": str(train_dir),
                "test": str(test_dir),
                "epochs": epochs,
                "batch": batch,
                "lr": lr,
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
    args = parser.parse_args(argv)
    result = train(
        args.train, args.test, args.out, args.epochs, args.batch, args.lr, args.device, args.hub, args.seed
    )
    print(json.dumps(result["final"], indent=1))


if __name__ == "__main__":
    main()
