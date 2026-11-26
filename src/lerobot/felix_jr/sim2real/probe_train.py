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
    rounds: int = 1  # (the iterative prior head's: `refine_heads.IterativeSample`)
    shared: bool = True
    space: str = "pinch"


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
# sub-experiment 6: hard geometry unrolled (`refine_heads.IterativeSample`) against its one-round self
REFINE2_SPECS = (
    HeadSpec("prior_sample", "prior_sample", REFINE_BLOCKS),
    HeadSpec("iter3_shared", "prior_iter", REFINE_BLOCKS, rounds=3, shared=True),
    HeadSpec("iter3_separate", "prior_iter", REFINE_BLOCKS, rounds=3, shared=False),
    HeadSpec("iter3_shared_image", "prior_iter", REFINE_BLOCKS, rounds=3, shared=True, space="image"),
)
SPECS = {"basic": BASIC_SPECS, "grid": GRID_SPECS, "refine": REFINE_SPECS, "refine2": REFINE2_SPECS}
PRIOR_KINDS = ("residual_mlp", "prior_query", "prior_soft", "prior_sample", "prior_iter")
ROUND_WEIGHT = 0.8  # each round's loss weighted ROUND_WEIGHT ** (rounds left after it), as RAFT's
EVAL_ROUNDS = (1, 2, 3, 4, 5)  # a shared iterative head evaluated at each of these too
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

        return build_prior_head(spec.kind, width, spec.queries, camera, spec.rounds, spec.shared, spec.space)
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


class FrameBatches(torch.utils.data.Dataset):
    """A `probe_data` folder's frames served a batch at a time.

    An item is a list of frame indices (a `BatchSampler`'s), read in order from the memory-mapped file (opened in each
    worker: a memory map handed to a worker would be pickled whole). Returns (frames (B, H, W, 3) uint8, their indices
    (B,), sorted).
    """

    def __init__(self, folder: Path) -> None:
        """The frames' file."""
        self.path = folder / "frames.npy"
        self.n = int(np.load(self.path, mmap_mode="r").shape[0])
        self._frames: np.ndarray | None = None

    def __len__(self) -> int:
        """The frames."""
        return self.n

    def __getitem__(self, idx: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """A batch's frames and indices."""
        if self._frames is None:
            self._frames = np.load(self.path, mmap_mode="r")
        order = np.sort(np.asarray(idx))
        return torch.from_numpy(np.ascontiguousarray(self._frames[order])), torch.from_numpy(order)


def loader(
    folder: Path, batch: int, shuffle: bool, workers: int, seed: int = 0, device: str = "cpu"
) -> torch.utils.data.DataLoader:
    """A `DataLoader` of `FrameBatches`: `workers` processes gather the batches ahead (prefetched, pinned).

    `shuffle`: random batches of `batch` (the last partial one dropped), else in order (all of them).
    """
    data = FrameBatches(folder)
    gen = torch.Generator().manual_seed(seed)
    base = (
        torch.utils.data.RandomSampler(data, generator=gen)
        if shuffle
        else torch.utils.data.SequentialSampler(data)
    )
    sampler = torch.utils.data.BatchSampler(base, batch, drop_last=shuffle)
    return torch.utils.data.DataLoader(
        data,
        sampler=sampler,
        batch_size=None,
        num_workers=workers,
        pin_memory=device.startswith("cuda"),
        prefetch_factor=4 if workers > 0 else None,
        persistent_workers=workers > 0,
    )


class ProbeData:
    """A `probe_data` folder: its frames (memory-mapped) and labels."""

    def __init__(self, folder: Path) -> None:
        """The frames and labels of `folder`."""
        self.folder = folder
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

    def on_device(self, device: str) -> dict[str, torch.Tensor]:
        """The labels the training loop needs, as tensors on `device`, once.

        The targets, in view, and the columns the priors are drawn from (`head_prior.priors_torch`).
        """
        out = {
            "targets": torch.from_numpy(self.targets()).to(device),
            "visible": torch.from_numpy(self.visible).to(device),
        }
        if self.has_pinch_pose:
            for k in ("round", "world"):
                out[k] = torch.from_numpy(self.labels[k].astype(np.int64)).to(device)
            for k in ("in_root", "size", "pinch_pos", "pinch_quat"):
                out[k] = torch.from_numpy(self.labels[k].astype(np.float32)).to(device)
        return out

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


SIGNED = (
    "center x (pinch)",
    "center y (pinch)",
    "center z (pinch)",
    "center x (root)",
    "center y (root)",
    "center z (root, vertical)",
    "radius",
    "height",
)


def per_parameter(
    pred: dict[str, np.ndarray], prior: dict[str, np.ndarray], target: np.ndarray, pinch_quat: np.ndarray
) -> dict[str, dict[str, float]]:
    """Which of the cylinder's parameters a head's correction helps (mm; the axis in deg).

    Each parameter: the prior's and the head's median |error|, the share of frames the head improves on the prior,
    and the correction's slope against the prior's error (the correction regressed on minus the prior's error: 1 undoes
    it, 0 ignores it). The center in the pinch frame and in the root frame (x, y across the table; z the vertical).
    Read with the errors: for a prior head, a slope near 1 and a small error fixes the parameter, near 0 copies the
    prior's; an image-only head never sees the prior, so its slope is ~1 whatever (its own error is what counts).
    """
    from .head_prior import quat_to_matrix

    r = quat_to_matrix(pinch_quat)

    def signed(x: dict[str, np.ndarray]) -> np.ndarray:
        e = 1000 * (x["center"] - target[:, :3])
        e_root = np.einsum("nij,nj->ni", r, e)
        return np.column_stack([e, e_root, 1000 * (x["size"] - target[:, 6:8])])

    def angle(x: dict[str, np.ndarray]) -> np.ndarray:
        return np.degrees(np.arccos(np.clip(np.abs(np.sum(x["axis"] * target[:, 3:6], axis=1)), -1, 1)))

    e_prior, e_head = signed(prior), signed(pred)
    out: dict[str, dict[str, float]] = {}
    for i, name in enumerate(SIGNED):
        a, b = e_prior[:, i], e_head[:, i]
        correction = b - a
        var = float(np.var(a))
        out[name] = {
            "prior_median_abs": float(np.median(np.abs(a))),
            "head_median_abs": float(np.median(np.abs(b))),
            "improved": float(np.mean(np.abs(b) < np.abs(a))),
            "slope": float(-np.cov(correction, a)[0, 1] / var) if var > 1e-12 else float("nan"),
        }
    a, b = angle(prior), angle(pred)
    out["axis (deg)"] = {
        "prior_median_abs": float(np.median(a)),
        "head_median_abs": float(np.median(b)),
        "improved": float(np.mean(b < a)),
        "slope": float("nan"),
    }
    return out


def print_per_parameter(metrics: dict[str, Any]) -> None:
    """The per-parameter breakdown (`per_parameter`) of each prior head, a table a head."""
    for name, m in metrics.items():
        if "per_parameter" not in m:
            continue
        print(f"== {name}: prior |error| -> head |error| (median), improved, the correction's slope")
        for param, v in m["per_parameter"].items():
            print(
                f"   {param:26s} {v['prior_median_abs']:6.1f} -> {v['head_median_abs']:6.1f}"
                f"   improved {100 * v['improved']:3.0f} %   slope {v['slope']:5.2f}"
            )


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
    workers: int = 0,
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
    labels = data.on_device(device)
    table_t = mean_t = None
    if prior_table is not None:
        from .head_prior import priors_torch, table_to_torch

        assert mean_size is not None
        table_t = table_to_torch(prior_table, device)
        mean_t = torch.as_tensor(mean_size, dtype=torch.float32, device=device)
    with torch.no_grad():
        for frames, idx in loader(data.folder, batch, False, workers, device=device):
            rgb = frames.to(device, non_blocking=True)
            idx = idx.to(device)
            feats = features(dino, rgb, blocks)
            prior = prior_z = None
            if table_t is not None:
                prior = priors_torch(labels, idx, table_t, mean_t)
                prior_z = standardize(prior, scale)
                p = {"center": prior[:, :3], "axis": prior[:, 3:6], "size": prior[:, 6:8]}
                p["in_view"] = torch.ones(len(prior), dtype=torch.bool, device=device)
                preds["(the prior alone)"].append({k: v.cpu().numpy() for k, v in p.items()})
            for name, h in heads.items():
                out = decode(run_head(h, specs[name], feats, prior_z, prior), scale)
                preds[name].append({k: v.cpu().numpy() for k, v in out.items()})
                if (
                    specs[name].kind == "prior_iter" and specs[name].shared
                ):  # (more or fewer rounds than trained)
                    for r in EVAL_ROUNDS:
                        h.eval_rounds = r
                        out = decode(run_head(h, specs[name], feats, prior_z, prior), scale)
                        preds.setdefault(f"{name}@{r}", []).append(
                            {k: v.cpu().numpy() for k, v in out.items()}
                        )
                    h.eval_rounds = None
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
            if name != "(the prior alone)":
                prior_p = {
                    k: np.concatenate([c[k] for c in preds["(the prior alone)"]])
                    for k in ("center", "axis", "size")
                }
                result[name]["per_parameter"] = per_parameter(
                    {k: v[seen] for k, v in p.items() if k in ("center", "axis", "size")},
                    {k: v[seen] for k, v in prior_p.items()},
                    target[seen],
                    data.labels["pinch_quat"][seen],
                )
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
    workers: int = 8,
    max_steps: int | None = None,
    compiled: bool = False,
) -> dict[str, Any]:
    """The heads trained (the module docstring); returns the metrics.

    `workers`: processes gathering the batches ahead (0: in this one); `max_steps`: at most this many batches an epoch
    (a timing run); `compiled`: the heads `torch.compile`d in place (`nn.Module.compile`).
    """
    torch.manual_seed(seed)
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
        camera["scale"] = scale.cpu().numpy()  # (the iterative head re-projects between rounds)
    mean_size = seen_targets[:, 6:8].mean(axis=0)
    test_table = test.prior_table(seed + 10_000) if uses_prior else None  # (the same priors every evaluation)
    heads = {s.name: build(s, camera).to(device) for s in specs}
    if compiled:
        for h in heads.values():
            h.compile()
    blocks = _blocks(by_name)
    labels = data.on_device(device)
    mean_t = torch.as_tensor(mean_size, dtype=torch.float32, device=device)
    batches = loader(data.folder, batch, True, workers, seed, device)
    per_epoch = min(len(batches), max_steps) if max_steps else len(batches)
    steps = epochs * per_epoch
    optims = {name: torch.optim.AdamW(h.parameters(), lr=lr, weight_decay=1e-4) for name, h in heads.items()}
    sched = {name: torch.optim.lr_scheduler.CosineAnnealingLR(o, max(steps, 1)) for name, o in optims.items()}
    out.mkdir(parents=True, exist_ok=True)
    history = []
    started = time.perf_counter()
    for epoch in range(epochs):
        table = None
        if uses_prior:  # (the head's errors drawn again each epoch)
            from .head_prior import priors_torch, table_to_torch

            table = table_to_torch(data.prior_table(seed + epoch), device)
        losses = {name: torch.zeros((), device=device) for name in heads}
        epoch_started, waited = time.perf_counter(), 0.0
        fetched = time.perf_counter()
        for step, (frames, idx) in enumerate(batches):
            if step >= per_epoch:
                break
            waited += time.perf_counter() - fetched
            rgb = frames.to(device, non_blocking=True)
            idx = idx.to(device, non_blocking=True)
            target, seen = labels["targets"][idx], labels["visible"][idx]
            prior = prior_z = None
            if table is not None:
                prior = priors_torch(labels, idx, table, mean_t)
                prior_z = standardize(prior, scale)
            if augmented:
                rgb = augment(rgb)
            feats = features(dino, rgb, blocks)
            for name, h in heads.items():
                output = run_head(h, by_name[name], feats, prior_z, prior)
                rounds = getattr(h, "round_outputs", None)
                if rounds:  # (an iterative head: a loss on every round, the later weighted more)
                    w = [ROUND_WEIGHT ** (len(rounds) - 1 - r) for r in range(len(rounds))]
                    value = sum(
                        wi * loss(o, target, seen, scale) for wi, o in zip(w, rounds, strict=True)
                    ) / sum(w)
                else:
                    value = loss(output, target, seen, scale)
                optims[name].zero_grad(set_to_none=True)
                value.backward()
                optims[name].step()
                sched[name].step()
                losses[name] += value.detach() / per_epoch  # (on the device: no wait for it every step)
            fetched = time.perf_counter()
        train_s = time.perf_counter() - epoch_started
        print(
            f"epoch {epoch + 1}: {per_epoch} steps in {train_s:.0f} s ({per_epoch / train_s:.2f} a second), "
            f"{waited:.0f} s of it waiting for batches",
            flush=True,
        )
        metrics = evaluate(dino, heads, by_name, test, scale, device, batch, test_table, mean_size, workers)
        history.append({"epoch": epoch, "loss": {n: float(v) for n, v in losses.items()}, "test": metrics})
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
    parser.add_argument(
        "--workers", type=int, default=8, help="processes gathering the batches ahead (0: none)"
    )
    parser.add_argument(
        "--max-steps", type=int, default=None, help="at most this many batches an epoch (timing)"
    )
    parser.add_argument("--compile", action="store_true", help="torch.compile the heads (in place)")
    parser.add_argument(
        "--eval-only",
        type=Path,
        default=None,
        metavar="HEADS_PT",
        help="evaluate a trained heads.pt on --test (its priors as training's evaluation drew them); --train unused",
    )
    args = parser.parse_args(argv)
    if args.eval_only is not None:
        dino = dinov2_small(args.device, args.hub)
        heads, specs, scale = load_heads(args.eval_only, args.device)
        test = ProbeData(args.test)
        uses_prior = any(sp.kind in PRIOR_KINDS for sp in specs.values()) and test.has_pinch_pose
        table = test.prior_table(args.seed + 10_000) if uses_prior else None
        mean_size = test.targets()[test.visible][:, 6:8].mean(axis=0)
        metrics = evaluate(
            dino, heads, specs, test, scale, args.device, args.batch, table, mean_size, args.workers
        )
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "eval.json").write_text(json.dumps(metrics, indent=1))
        print_per_parameter(metrics)
        return
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
        args.workers,
        args.max_steps,
        args.compile,
    )
    summary = {
        n: {"center_mm": m["center_mm"], "center_mm_not_held": m["center_mm_not_held"]}
        for n, m in result["final"].items()
    }
    print(json.dumps(summary, indent=1))
    print_per_parameter(result["final"])


if __name__ == "__main__":
    main()
