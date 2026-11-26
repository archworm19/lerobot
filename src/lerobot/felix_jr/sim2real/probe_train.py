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
12 and 9-12 together, with 1 and 4 queries. Each predicts the object's center (3) and axis (3) in the pinch frame
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
    for q in (1, 4)  # (16 queries dropped: 4x the training time of the basic heads, 2026-11-25)
    if not (b == (12,) and q == QUERIES)  # (the basic attention head)
)
# the refinement's heads (`refine_heads`: they correct the head camera's prior, `head_prior`) on blocks 9-12, beside two
# of the grid's image-only heads as references
REFINE_BLOCKS = (9, 10, 11, 12)
REFINE_SPECS = (
    HeadSpec("attention", "attention"),
    HeadSpec("attn_L9-12_q4", "attention", REFINE_BLOCKS, 4),
    HeadSpec("residual_mlp", "residual_mlp", REFINE_BLOCKS),
    HeadSpec("prior_query", "prior_query", REFINE_BLOCKS, 4),
    HeadSpec("prior_soft", "prior_soft", REFINE_BLOCKS, 4),
    HeadSpec("prior_sample", "prior_sample", REFINE_BLOCKS),
)
SPECS = {"basic": BASIC_SPECS, "grid": GRID_SPECS, "refine": REFINE_SPECS}
PRIOR_KINDS = ("residual_mlp", "prior_query", "prior_soft", "prior_sample")
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


def build(spec: HeadSpec, camera: dict[str, Any] | None = None) -> nn.Module:
    """The head `spec` describes; a prior head's `camera` (`refine_heads.sim_camera`; loading restores it)."""
    width = DIM * len(spec.blocks)
    if spec.kind == "linear":
        return LinearHead(width)
    if spec.kind == "mean":
        return MeanHead(width)
    if spec.kind == "attention":
        return AttentionHead(width, spec.queries)
    if spec.kind in PRIOR_KINDS:
        from .refine_heads import build_prior_head

        return build_prior_head(spec.kind, width, spec.queries, camera)
    raise ValueError(f"no head kind {spec.kind!r}")


# the basic heads by name (their defaults: the last block, 4 queries)
HEADS = {"linear": LinearHead, "mean": MeanHead, "attention": AttentionHead}


def run_head(
    head: nn.Module,
    spec: HeadSpec,
    feats: dict[int, tuple[torch.Tensor, torch.Tensor]],
    prior_z: torch.Tensor | None = None,
    prior: torch.Tensor | None = None,
) -> torch.Tensor:
    """A head's outputs (B, OUT): a prior head's given the prior (standardized and in meters, pinch frame)."""
    cls, patches = head_inputs(feats, spec)
    if spec.kind in PRIOR_KINDS:
        if prior_z is None or prior is None:
            raise ValueError(f"head {spec.name!r} corrects a prior: none given")
        return head(cls, patches, prior_z, prior)
    return head(cls, patches)


def standardize(cyl: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Cylinders (B, 8: center, axis, radius, height; m) standardized as the targets (the axis as it is)."""
    mean, std = scale
    return (cyl - mean) / std


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

    @property
    def has_pinch_pose(self) -> bool:
        """Recorded with the gripper's pinch pose (`probe_data` since 2026-11-26): priors can be drawn for it."""
        return "pinch_pos" in self.labels

    def prior_table(self, seed: int) -> Any:
        """The head camera's errors, one an episode (`head_prior.episode_errors`), drawn with `seed`."""
        from .head_prior import HeadErrors, episode_errors

        rounds, worlds = int(self.labels["round"].max()) + 1, int(self.labels["world"].max()) + 1
        return episode_errors(rounds, worlds, HeadErrors(), np.random.default_rng(seed))

    def priors(self, idx: np.ndarray, table: Any, mean_size: np.ndarray, device: str) -> torch.Tensor:
        """The priors (B, 8; pinch frame, m) of the frames `idx` (sorted, as `batch` reads them)."""
        from .head_prior import priors

        _, pinch = priors(self.labels, np.sort(idx), table, mean_size)
        return torch.from_numpy(pinch.astype(np.float32)).to(device)

    def targets(self) -> np.ndarray:
        """(N, 8): the center, the axis, the radius and height."""
        return np.concatenate([self.labels["in_pinch"], self.labels["size"]], axis=1).astype(np.float32)

    def batch(self, idx: np.ndarray, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The frames' priors, as `batch` reads them.

        (frames (B, H, W, 3) uint8, targets (B, 8), in view (B,)) on `device`; `idx` sorted (the memory map read
        in order: the priors of the same frames, `priors`, take the same sorted `idx`).
        """
        idx = np.sort(idx)
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
    prior_table: Any = None,
    mean_size: np.ndarray | None = None,
) -> dict[str, Any]:
    """Each head's errors on `data`, with the priors' when given.

    Each head's errors on `data` (the module docstring); with `prior_table`, the priors too (the same for every
    evaluation: the prior alone a row of its own) and each head's gain curve against them.
    """
    preds: dict[str, list[dict[str, np.ndarray]]] = {name: [] for name in heads}
    if prior_table is not None:
        preds["(the prior alone)"] = []
    for h in heads.values():
        h.eval()
    blocks = _blocks(specs)
    with torch.no_grad():
        for start in range(0, data.n, batch):
            idx = np.arange(start, min(start + batch, data.n))
            rgb, _, _ = data.batch(idx, device)
            feats = features(dino, rgb, blocks)
            prior = prior_z = None
            if prior_table is not None:
                assert mean_size is not None
                prior = data.priors(idx, prior_table, mean_size, device)
                prior_z = standardize(prior, scale)
                p = {"center": prior[:, :3], "axis": prior[:, 3:6], "size": prior[:, 6:8]}
                p["in_view"] = torch.ones(len(prior), dtype=torch.bool, device=device)
                preds["(the prior alone)"].append({k: v.cpu().numpy() for k, v in p.items()})
            for name, h in heads.items():
                out = decode(run_head(h, specs[name], feats, prior_z, prior), scale)
                preds[name].append({k: v.cpu().numpy() for k, v in out.items()})
    target = data.targets()
    seen = data.visible
    dist = np.linalg.norm(target[:, :3], axis=1)
    pool = data.labels["pool"]
    held = data.labels["held"].astype(bool)
    result: dict[str, Any] = {}
    prior_err = None
    if prior_table is not None:
        pc = np.concatenate([c["center"] for c in preds["(the prior alone)"]])
        prior_err = 1000 * np.linalg.norm(pc - target[:, :3], axis=1)
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
        if prior_err is not None:
            from .head_prior import gain_curve

            result[name]["gain_curve"] = gain_curve(err_mm[seen], prior_err[seen])
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
    uses_prior = any(s.kind in PRIOR_KINDS for s in specs)
    camera = None
    if uses_prior:
        if not (data.has_pinch_pose and test.has_pinch_pose):
            raise ValueError(
                "prior heads need data recorded with the pinch pose (probe_data since 2026-11-26)"
            )
        from .refine_heads import sim_camera

        camera = sim_camera()
    mean_size = seen_targets[:, 6:8].mean(axis=0)
    test_table = test.prior_table(seed + 10_000) if uses_prior else None  # (the same priors every evaluation)
    heads = {s.name: build(s, camera).to(device) for s in specs}
    blocks = _blocks(by_name)
    steps = epochs * (data.n // batch)
    optims = {name: torch.optim.AdamW(h.parameters(), lr=lr, weight_decay=1e-4) for name, h in heads.items()}
    sched = {name: torch.optim.lr_scheduler.CosineAnnealingLR(o, max(steps, 1)) for name, o in optims.items()}
    out.mkdir(parents=True, exist_ok=True)
    history = []
    started = time.perf_counter()
    for epoch in range(epochs):
        order = rng.permutation(data.n)
        table = (
            data.prior_table(seed + epoch) if uses_prior else None
        )  # (the head's errors drawn again each epoch)
        losses = dict.fromkeys(heads, 0.0)
        nb = max(data.n // batch, 1)
        for b in range(nb):
            idx = order[b * batch : (b + 1) * batch]
            rgb, target, seen = data.batch(idx, device)
            prior = prior_z = None
            if table is not None:
                prior = data.priors(idx, table, mean_size, device)
                prior_z = standardize(prior, scale)
            if augmented:
                rgb = augment(rgb)
            feats = features(dino, rgb, blocks)
            for name, h in heads.items():
                value = loss(run_head(h, by_name[name], feats, prior_z, prior), target, seen, scale)
                optims[name].zero_grad(set_to_none=True)
                value.backward()
                optims[name].step()
                sched[name].step()
                losses[name] += float(value.detach()) / nb
        metrics = evaluate(dino, heads, by_name, test, scale, device, batch, test_table, mean_size)
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
