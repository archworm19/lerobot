"""Heads that correct a prior: the head camera's cylinder, refined by the wrist camera.

(`docs/sim2real_grasp.md`, "The cylinder refinement: the plan", sub-experiments 2-5.)

Each takes DINOv2-S's patch tokens (blocks side by side, projected to `WIDTH`) and a prior (`head_prior`: the
cylinder in the pinch frame, standardized as `probe_train`'s targets) and returns the corrected cylinder in the same
layout as the image-only heads' outputs (center and size standardized, the axis, the in-view logit): the prior plus a
learned correction, so `probe_train.loss` and `decode` serve them unchanged. They differ in how the prior steers where
they look:

- `ResidualMLP` (2): the patches' mean and the class token beside the prior's embedding: does the prior help at all?
- `PriorQuery` (3): learned queries plus the prior's embedding attending over all patches: the prior as the query,
  no geometry.
- `PriorQuery(soft=True)` (4): the same, the attention biased toward the patches near the prior's projected points
  (falling off with the distance, its width learned): soft geometry.
- `PriorSample` (5): DINO's features read only at the prior's projected points and at offsets the prior predicts
  (bilinear, `grid_sample`): hard geometry; the background and other objects not read at all.

The projection (`Projector`): the cylinder's center, `RIM_POINTS` on each rim and its axis's two ends, through the
wrist camera's pose in the pinch frame and its lens (undistorted, at 224 x 168), kept in the head's weights (buffers).
"""

from typing import Any

import numpy as np
import torch
from torch import nn

GRID = (12, 16)  # (rows, columns) of DINO's patches at 168 x 224
IMAGE = (168, 224)
WIDTH = 256  # the projected patch features' width
RIM_POINTS = 8
POINTS = 1 + 2 * RIM_POINTS + 2  # the center, the rims, the axis's ends


class Projector(nn.Module):
    """A cylinder (pinch frame) to its points in DINO's patch grid, as `grid_sample` coordinates in [-1, 1]."""

    def __init__(self, cam_from_pinch: np.ndarray | None = None, k: np.ndarray | None = None) -> None:
        """The camera: its pose in the pinch frame (4x4) and its pinhole at 224 x 168 (3x3); identity until loaded."""
        super().__init__()
        self.register_buffer(
            "cam_from_pinch",
            torch.as_tensor(cam_from_pinch if cam_from_pinch is not None else np.eye(4), dtype=torch.float32),
        )
        self.register_buffer("k", torch.as_tensor(k if k is not None else np.eye(3), dtype=torch.float32))
        t = torch.arange(RIM_POINTS, dtype=torch.float32) * (2 * torch.pi / RIM_POINTS)
        self.register_buffer("cos", torch.cos(t))
        self.register_buffer("sin", torch.sin(t))

    def points(self, cyl: torch.Tensor) -> torch.Tensor:
        """The cylinders' (B, 8: center, axis, radius, height; m) points (B, POINTS, 3), pinch frame."""
        c, a = cyl[:, :3], nn.functional.normalize(cyl[:, 3:6], dim=1)
        r, h = cyl[:, 6:7], cyl[:, 7:8]
        helper = torch.where(
            a[:, :1].abs() < 0.9,
            torch.tensor([[1.0, 0.0, 0.0]], device=a.device),
            torch.tensor([[0.0, 1.0, 0.0]], device=a.device),
        )
        u = nn.functional.normalize(torch.cross(a, helper, dim=1), dim=1)
        v = torch.cross(a, u, dim=1)
        ring = self.cos[None, :, None] * u[:, None] + self.sin[None, :, None] * v[:, None]  # (B, RIM, 3)
        top = c[:, None] + (h / 2)[:, None] * a[:, None] + r[:, None] * ring
        bottom = c[:, None] - (h / 2)[:, None] * a[:, None] + r[:, None] * ring
        ends = torch.stack([c + h / 2 * a, c - h / 2 * a], dim=1)
        return torch.cat([c[:, None], top, bottom, ends], dim=1)

    def forward(self, cyl: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The cylinders' points in the image, and which are valid.

        (the points in `grid_sample`'s coordinates (B, POINTS, 2), x then y in [-1, 1]; valid (B, POINTS): in
        front of the camera and on the image).
        """
        p = self.points(cyl)
        cam = p @ self.cam_from_pinch[:3, :3].T + self.cam_from_pinch[:3, 3]
        z = cam[..., 2].clamp(min=1e-3)
        u = self.k[0, 0] * cam[..., 0] / z + self.k[0, 2]
        v = self.k[1, 1] * cam[..., 1] / z + self.k[1, 2]
        grid = torch.stack([2 * u / IMAGE[1] - 1, 2 * v / IMAGE[0] - 1], dim=-1)
        valid = (cam[..., 2] > 0.01) & (grid.abs() <= 1.0).all(dim=-1)
        return grid, valid


class PriorEmbed(nn.Module):
    """The prior (standardized center and size, the axis) to an embedding."""

    def __init__(self) -> None:
        """The MLP."""
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(8, WIDTH), nn.GELU(), nn.Linear(WIDTH, WIDTH))

    def forward(self, prior_z: torch.Tensor) -> torch.Tensor:
        """(B, WIDTH)."""
        return self.mlp(prior_z)


class _Corrector(nn.Module):
    """What the prior heads share: the patches projected, the prior embedded, the correction applied."""

    def __init__(self, width: int, cam_from_pinch: np.ndarray | None, k: np.ndarray | None) -> None:
        super().__init__()
        self.project_in = nn.Linear(width, WIDTH)
        self.embed = PriorEmbed()
        self.projector = Projector(cam_from_pinch, k)
        self.last_weights: torch.Tensor | None = None

    def correct(self, prior_z: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """The prior plus a correction, in the image-only heads' layout.

        The prior plus `delta` (B, 9), in the image-only heads' layout: center and size standardized, the axis (the
        prior's plus a correction), the in-view logit.
        """
        out = torch.cat([prior_z[:, :8], torch.zeros_like(prior_z[:, :1])], dim=1) + delta
        return out


def _mlp(inp: int, out: int = 9) -> nn.Module:
    head = nn.Sequential(nn.Linear(inp, WIDTH), nn.GELU(), nn.Linear(WIDTH, out))
    nn.init.zeros_(head[-1].weight)  # (the correction starts at none: the prior)
    nn.init.zeros_(head[-1].bias)
    return head


class ResidualMLP(_Corrector):
    """The patches' mean and the class token beside the prior's embedding, an MLP: the correction."""

    def __init__(
        self, width: int, cam_from_pinch: np.ndarray | None = None, k: np.ndarray | None = None
    ) -> None:
        """The layers."""
        super().__init__(width, cam_from_pinch, k)
        self.cls_in = nn.Linear(width, WIDTH)
        self.out = _mlp(3 * WIDTH)

    def forward(
        self, cls: torch.Tensor, patches: torch.Tensor, prior_z: torch.Tensor, prior: torch.Tensor
    ) -> torch.Tensor:
        """The corrected cylinder (B, 9)."""
        x = torch.cat([self.cls_in(cls), self.project_in(patches).mean(dim=1), self.embed(prior_z)], dim=1)
        return self.correct(prior_z, self.out(x))


class PriorQuery(_Corrector):
    """Learned queries plus the prior's embedding attending over all patches.

    Learned queries plus the prior's embedding attending over all patches (`soft`: biased toward the patches near the
    prior's projected points).
    """

    def __init__(
        self,
        width: int,
        cam_from_pinch: np.ndarray | None = None,
        k: np.ndarray | None = None,
        queries: int = 4,
        heads: int = 4,
        soft: bool = False,
    ) -> None:
        """The layers; `soft`: the attention's geometric bias."""
        super().__init__(width, cam_from_pinch, k)
        self.n_heads, self.soft = heads, soft
        self.queries = nn.Parameter(0.02 * torch.randn(queries, WIDTH))
        self.position = nn.Parameter(0.02 * torch.randn(GRID[0] * GRID[1], WIDTH))
        self.norm = nn.LayerNorm(WIDTH)
        self.attend = nn.MultiheadAttention(WIDTH, heads, batch_first=True)
        self.out = _mlp((queries + 1) * WIDTH)
        if soft:
            self.log_sigma = nn.Parameter(torch.tensor(np.log(2.0), dtype=torch.float32))  # (in patches)
            rows, cols = torch.meshgrid(torch.arange(GRID[0]), torch.arange(GRID[1]), indexing="ij")
            # each patch's center in grid_sample's coordinates
            centers = torch.stack(
                [(2 * cols + 1) / GRID[1] - 1, (2 * rows + 1) / GRID[0] - 1], dim=-1
            ).reshape(-1, 2)
            self.register_buffer("patch_centers", centers.float())

    def _bias(self, prior: torch.Tensor, n_queries: int) -> torch.Tensor:
        """The attention's geometric bias toward the prior's projected points.

        The additive attention bias (B * heads, queries, patches): -d^2 / (2 sigma^2), d the patch's distance (in
        patches) from the nearest valid projected point; 0 everywhere where none is valid.
        """
        grid, valid = self.projector(prior)  # (B, P, 2), (B, P)
        scale = torch.tensor([GRID[1] / 2, GRID[0] / 2], device=grid.device)  # (grid_sample units to patches)
        d = ((self.patch_centers[None, :, None] - grid[:, None]) * scale).norm(dim=-1)  # (B, patches, P)
        d = torch.where(valid[:, None], d, torch.full_like(d, 1e3)).min(dim=-1).values
        sigma = self.log_sigma.exp()
        bias = -(d**2) / (2 * sigma**2)
        bias = torch.where(valid.any(dim=1, keepdim=True), bias, torch.zeros_like(bias))
        b = len(prior)
        return (
            bias[:, None, None, :]
            .expand(b, self.n_heads, n_queries, -1)
            .reshape(b * self.n_heads, n_queries, -1)
        )

    def forward(
        self, cls: torch.Tensor, patches: torch.Tensor, prior_z: torch.Tensor, prior: torch.Tensor
    ) -> torch.Tensor:
        """The corrected cylinder (B, 9)."""
        e = self.embed(prior_z)
        keys = self.norm(self.project_in(patches) + self.position)
        q = self.queries[None] + e[:, None]
        mask = self._bias(prior, q.shape[1]) if self.soft else None
        pooled, weights = self.attend(
            q, keys, keys, attn_mask=mask, need_weights=True, average_attn_weights=True
        )
        self.last_weights = weights.detach()
        return self.correct(prior_z, self.out(torch.cat([pooled.flatten(1), e], dim=1)))


class PriorSample(_Corrector):
    """DINO's features read at the prior's projected points, pooled by attention.

    DINO's features read only at the prior's projected points and at offsets the prior predicts (bilinear), pooled
    by attention: the correction.
    """

    def __init__(
        self,
        width: int,
        cam_from_pinch: np.ndarray | None = None,
        k: np.ndarray | None = None,
        heads: int = 4,
    ) -> None:
        """The layers; the offsets start at none."""
        super().__init__(width, cam_from_pinch, k)
        self.offsets = nn.Linear(WIDTH, POINTS * 2)
        nn.init.zeros_(self.offsets.weight)
        nn.init.zeros_(self.offsets.bias)
        self.point_id = nn.Parameter(0.02 * torch.randn(POINTS, WIDTH))
        self.missing = nn.Parameter(torch.zeros(WIDTH))  # (a point behind the camera or off the image)
        self.norm = nn.LayerNorm(WIDTH)
        self.query = nn.Parameter(0.02 * torch.randn(1, WIDTH))
        self.attend = nn.MultiheadAttention(WIDTH, heads, batch_first=True)
        self.out = _mlp(2 * WIDTH)

    def forward(
        self, cls: torch.Tensor, patches: torch.Tensor, prior_z: torch.Tensor, prior: torch.Tensor
    ) -> torch.Tensor:
        """The corrected cylinder (B, 9)."""
        b = len(patches)
        e = self.embed(prior_z)
        fmap = self.project_in(patches).transpose(1, 2).reshape(b, WIDTH, *GRID)
        grid, valid = self.projector(prior)
        # (offsets in patches, at most 3 either way)
        step = torch.tensor([2 / GRID[1], 2 / GRID[0]], device=grid.device)
        grid = grid + 3 * torch.tanh(self.offsets(e).reshape(b, POINTS, 2)) * step
        sampled = nn.functional.grid_sample(fmap, grid[:, :, None], mode="bilinear", align_corners=False)
        tokens = sampled[..., 0].transpose(1, 2)  # (B, POINTS, WIDTH)
        tokens = torch.where(valid[..., None], tokens, self.missing.expand_as(tokens))
        tokens = self.norm(tokens + self.point_id)
        q = self.query[None].expand(b, -1, -1) + e[:, None]
        pooled, weights = self.attend(q, tokens, tokens, need_weights=True)
        self.last_weights = weights.detach()
        return self.correct(prior_z, self.out(torch.cat([pooled[:, 0], e], dim=1)))


class _Round(nn.Module):
    """One round of `IterativeSample`: sample, attend, update the hidden state, predict a correction.

    The samples at the current cylinder's projected points and at offsets the hidden state predicts; the correction
    starts at none.
    """

    def __init__(self, heads: int = 4) -> None:
        super().__init__()
        self.offsets = nn.Linear(WIDTH, POINTS * 2)
        nn.init.zeros_(self.offsets.weight)
        nn.init.zeros_(self.offsets.bias)
        self.point_id = nn.Parameter(0.02 * torch.randn(POINTS, WIDTH))
        self.missing = nn.Parameter(torch.zeros(WIDTH))
        self.norm = nn.LayerNorm(WIDTH)
        self.query = nn.Parameter(0.02 * torch.randn(1, WIDTH))
        self.attend = nn.MultiheadAttention(WIDTH, heads, batch_first=True)
        self.gru = nn.GRUCell(2 * WIDTH, WIDTH)
        self.out = _mlp(WIDTH)

    def forward(
        self,
        fmap: torch.Tensor,
        hidden: torch.Tensor,
        embedded: torch.Tensor,
        grid: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(the hidden state, the correction (B, 9), the attention's weights)."""
        b = len(fmap)
        step = torch.tensor([2 / GRID[1], 2 / GRID[0]], device=grid.device)
        grid = grid + 3 * torch.tanh(self.offsets(hidden).reshape(b, POINTS, 2)) * step
        sampled = nn.functional.grid_sample(fmap, grid[:, :, None], mode="bilinear", align_corners=False)
        tokens = sampled[..., 0].transpose(1, 2)
        tokens = torch.where(valid[..., None], tokens, self.missing.expand_as(tokens))
        tokens = self.norm(tokens + self.point_id)
        q = self.query[None].expand(b, -1, -1) + hidden[:, None]
        pooled, weights = self.attend(q, tokens, tokens, need_weights=True)
        hidden = self.gru(torch.cat([pooled[:, 0], embedded], dim=1), hidden)
        return hidden, self.out(hidden), weights


class IterativeSample(_Corrector):
    """Hard geometry unrolled, as RAFT refines (sub-experiment 6).

    Each round samples DINO at the current cylinder's projected points, updates a hidden state (a GRU cell) and
    corrects the cylinder, which the next round re-projects. `rounds` (at evaluation `eval_rounds` may run more, with
    `shared` weights); `shared`: one round's weights for all, else a round's own; `space`: the correction in the pinch
    frame (the center and size standardized, as the other heads) or in the image's terms ("image": the projected
    center's shift in pixels, the log of its depth, the logs of the radius and height: a pixel the same near and far).
    The estimate detached between rounds (RAFT), each round's output kept (`round_outputs`) for a loss on every round.
    """

    def __init__(
        self,
        width: int,
        cam_from_pinch: np.ndarray | None = None,
        k: np.ndarray | None = None,
        scale: np.ndarray | None = None,
        rounds: int = 3,
        shared: bool = True,
        space: str = "pinch",
    ) -> None:
        """The rounds; `scale` (2, 8): the targets' standardization (mean, std), to re-project between rounds."""
        super().__init__(width, cam_from_pinch, k)
        if space not in ("pinch", "image"):
            raise ValueError(f"space: pinch or image, not {space!r}")
        self.rounds, self.shared, self.space = rounds, shared, space
        default = np.stack([np.zeros(8), np.ones(8)])
        self.register_buffer(
            "scale", torch.as_tensor(scale if scale is not None else default, dtype=torch.float32)
        )
        self.steps = nn.ModuleList([_Round() for _ in range(1 if shared else rounds)])
        self.eval_rounds: int | None = None
        self.round_outputs: list[torch.Tensor] = []

    def _metric(self, z: torch.Tensor) -> torch.Tensor:
        mean, std = self.scale
        m = z * std + mean
        axis = nn.functional.normalize(z[:, 3:6], dim=1)
        return torch.cat([m[:, :3], axis, m[:, 6:7].clamp(min=0.003), m[:, 7:8].clamp(min=0.01)], dim=1)

    def _standard(self, m: torch.Tensor) -> torch.Tensor:
        mean, std = self.scale
        z = (m - mean) / std
        return torch.cat([z[:, :3], m[:, 3:6], z[:, 6:8]], dim=1)

    def _corrected(
        self, z: torch.Tensor, m: torch.Tensor, delta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The corrected cylinder: (standardized, in meters)."""
        if self.space == "pinch":
            new_z = z + delta[:, :8]
            return new_z, self._metric(new_z)
        t, k = self.projector.cam_from_pinch, self.projector.k
        cam = m[:, :3] @ t[:3, :3].T + t[:3, 3]
        depth = cam[:, 2:3].clamp(min=1e-3)
        u = k[0, 0] * cam[:, :1] / depth + k[0, 2] + 10.0 * delta[:, 0:1]  # (10 px a unit)
        v = k[1, 1] * cam[:, 1:2] / depth + k[1, 2] + 10.0 * delta[:, 1:2]
        depth = depth * torch.exp(0.1 * delta[:, 2:3])
        cam = torch.cat([(u - k[0, 2]) * depth / k[0, 0], (v - k[1, 2]) * depth / k[1, 1], depth], dim=1)
        center = (cam - t[:3, 3]) @ t[:3, :3]
        axis = nn.functional.normalize(m[:, 3:6] + delta[:, 3:6], dim=1)
        size = m[:, 6:8] * torch.exp(0.1 * delta[:, 6:8])
        new_m = torch.cat([center, axis, size], dim=1)
        return self._standard(new_m), new_m

    def forward(
        self, cls: torch.Tensor, patches: torch.Tensor, prior_z: torch.Tensor, prior: torch.Tensor
    ) -> torch.Tensor:
        """The last round's corrected cylinder (B, 9)."""
        b = len(patches)
        fmap = self.project_in(patches).transpose(1, 2).reshape(b, WIDTH, *GRID)
        rounds = self.eval_rounds if (self.eval_rounds and self.shared) else self.rounds
        z, m = prior_z[:, :8], prior
        hidden = torch.tanh(self.embed(prior_z))
        self.round_outputs = []
        weights = torch.zeros(0)
        for r in range(rounds):
            step = self.steps[0 if self.shared else r]
            grid, valid = self.projector(m)
            hidden, delta, weights = step(fmap, hidden, self.embed(z), grid, valid)
            z, m = self._corrected(z, m, delta)
            self.round_outputs.append(torch.cat([z, delta[:, 8:9]], dim=1))
            z, m = z.detach(), m.detach()  # (RAFT: the estimate detached between rounds)
        self.last_weights = weights.detach()
        return self.round_outputs[-1]


KINDS = {
    "residual_mlp": ResidualMLP,
    "prior_query": PriorQuery,
    "prior_soft": PriorQuery,
    "prior_sample": PriorSample,
    "prior_iter": IterativeSample,
}


def build_prior_head(
    kind: str,
    width: int,
    queries: int,
    camera: dict[str, Any] | None = None,
    rounds: int = 1,
    shared: bool = True,
    space: str = "pinch",
) -> nn.Module:
    """The prior head of `kind` on tokens `width` wide.

    `camera` ({"cam_from_pinch", "k"}, and "scale" for the iterative head): the wrist camera's; `rounds`, `shared`,
    `space`: the iterative head's (`IterativeSample`).
    """
    cam = camera or {}
    args = (width, cam.get("cam_from_pinch"), cam.get("k"))
    if kind == "residual_mlp":
        return ResidualMLP(*args)
    if kind == "prior_query":
        return PriorQuery(*args, queries=queries)
    if kind == "prior_soft":
        return PriorQuery(*args, queries=queries, soft=True)
    if kind == "prior_sample":
        return PriorSample(*args)
    if kind == "prior_iter":
        return IterativeSample(*args, scale=cam.get("scale"), rounds=rounds, shared=shared, space=space)
    raise ValueError(f"no prior head kind {kind!r}")


def sim_camera() -> dict[str, np.ndarray]:
    """The wrist camera as the simulator renders it.

    The wrist camera as the simulator renders it: its pose in the pinch frame and its lens at 224 x 168 (the
    calibrated model's, without distortion).
    """
    from lerobot.felix_jr.calibration.merge_urdf import wrist_camera_frame
    from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel, pinch_frame
    from lerobot.felix_jr.harness import api

    model = XleRobotModel.load(api.MODEL)
    q = model.joint_vector()
    cam_from_pinch = np.linalg.inv(
        np.linalg.inv(model.frame_pose(pinch_frame("right"), q))
        @ model.frame_pose(wrist_camera_frame("right"), q)
    )
    lens = model.wrist_cameras["right"]
    k = lens.matrix().copy()
    k[:2] *= IMAGE[1] / lens.width
    return {"cam_from_pinch": cam_from_pinch, "k": k}
