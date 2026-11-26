"""The head camera's cylinder, simulated: the prior the wrist camera's refinement starts from.

(`docs/sim2real_grasp.md`, "The cylinder refinement: the plan", sub-experiment 0.)

On the robot the harness fits the object with the head camera's depth before the approach; the wrist camera then
corrects that fit. In the simulator the prior is the truth with the head's errors, as measured on the robot
(`object_grasping.md`, the wrist silhouettes against the head's fit): the head and the wrist cameras 1-3 cm apart
across the table and 1.5-2.5 cm in height, fairly constant within a session; the radius too large when the fit takes
in a cap (bowl7: 38 mm wide against the body's 29); the fits always upright. `HeadErrors` draws them a little wider.

- `episode_errors`: one error per episode (`probe_data`'s frames of a world in a round), fixed for its frames; drawn
  again each epoch.
- `priors`: a batch's priors, in the root frame and in each frame's pinch frame (through the gripper's pose,
  `probe_data`'s `pinch_pos`, `pinch_quat`): center (3), axis (3), radius, height, as `probe_train`'s targets.
- `gain_curve`: an estimate's error against the prior's, by the prior's error: what the refinement buys, and whether
  it is ever worse than the prior it started from.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

UP = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class HeadErrors:
    """How the head camera's cylinder misses the truth (std or ranges, m); a little wider than measured."""

    across_m: float = 0.015  # the center across the table (x and y, root frame; each)
    height_m: float = (
        0.015  # the height (the head sees the top; the bottom from the table: the center moves half)
    )
    table_m: float = 0.005  # the table's plane (the bottom and the top alike)
    radius_m: tuple[float, float] = (
        -0.002,
        0.010,
    )  # the radius's error, uniform (a fit taking in a cap: too wide)
    clip_m: float = 0.05  # the center's offset at most this
    weak_share: float = (
        0.1  # a share of priors weak: their offsets times `weak_scale`, the size the training's mean
    )
    weak_scale: float = 3.0


@dataclass(frozen=True)
class EpisodeErrors:
    """One error per episode (a world in a round), the table `priors` reads: (rounds, worlds, ...)."""

    offset: np.ndarray  # (R, W, 3): the center's offset, root frame
    d_height: np.ndarray  # (R, W)
    d_radius: np.ndarray  # (R, W)
    weak: np.ndarray  # (R, W) bool


def episode_errors(rounds: int, worlds: int, errors: HeadErrors, rng: np.random.Generator) -> EpisodeErrors:
    """The head's error for each (round, world) episode (the module docstring)."""
    shape = (rounds, worlds)
    weak = rng.random(shape) < errors.weak_share
    scale = np.where(weak, errors.weak_scale, 1.0)
    d_height = rng.normal(0.0, errors.height_m, shape) * scale
    across = rng.normal(0.0, errors.across_m, (*shape, 2)) * scale[..., None]
    up = d_height / 2 + rng.normal(0.0, errors.table_m, shape) * scale
    offset = np.concatenate([across, up[..., None]], axis=-1)
    norm = np.linalg.norm(offset, axis=-1, keepdims=True)
    limit = (errors.clip_m * scale)[..., None]
    offset = np.where(norm > limit, offset * limit / np.maximum(norm, 1e-12), offset)
    d_radius = rng.uniform(*errors.radius_m, shape) * scale
    return EpisodeErrors(offset, d_height, d_radius, weak)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrices (N, 3, 3) of unit quaternions (N, 4, w x y z)."""
    w, x, y, z = (q[:, i] for i in range(4))
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=-1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=-1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=-1),
        ],
        axis=1,
    )


def to_pinch(root: np.ndarray, pinch_pos: np.ndarray, pinch_quat: np.ndarray) -> np.ndarray:
    """Cylinders (N, 8: center, axis, radius, height) from the root frame into each frame's pinch frame."""
    r = quat_to_matrix(pinch_quat)
    center = np.einsum("nji,nj->ni", r, root[:, :3] - pinch_pos)
    axis = np.einsum("nji,nj->ni", r, root[:, 3:6])
    return np.concatenate([center, axis, root[:, 6:8]], axis=1)


def priors(
    labels: dict[str, np.ndarray], idx: np.ndarray, table: EpisodeErrors, mean_size: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(the priors in the root frame, in the pinch frame), each (B, 8: center, axis, radius, height).

    For the frames `idx` of a `probe_data` folder's `labels`; `mean_size` (2,): a weak prior's radius and height.
    """
    r = labels["round"][idx].astype(int)
    w = labels["world"][idx].astype(int)
    truth_center = labels["in_root"][idx, :3]
    size = labels["size"][idx]
    weak = table.weak[r, w]
    radius = np.where(weak, mean_size[0], size[:, 0] + table.d_radius[r, w])
    height = np.where(weak, mean_size[1], size[:, 1] + table.d_height[r, w])
    root = np.concatenate(
        [
            truth_center + table.offset[r, w],
            np.broadcast_to(UP, (len(idx), 3)),  # (the head's fits upright)
            np.maximum(radius, 0.004)[:, None],
            np.maximum(height, 0.02)[:, None],
        ],
        axis=1,
    )
    return root, to_pinch(root, labels["pinch_pos"][idx], labels["pinch_quat"][idx])


def gain_curve(
    estimate_mm: np.ndarray,
    prior_mm: np.ndarray,
    edges_mm: tuple[float, ...] = (0, 5, 10, 20, 30, 50, np.inf),
) -> list[dict[str, Any]]:
    """An estimate's center error against its prior's (both (N,), mm), by bins of the prior's error.

    Each bin: the medians, and the share where the estimate is worse than the prior.
    """
    out = []
    for lo, hi in zip(edges_mm[:-1], edges_mm[1:], strict=True):
        m = (prior_mm >= lo) & (prior_mm < hi)
        if m.sum() == 0:
            continue
        out.append(
            {
                "prior_mm": [float(lo), float(hi)],
                "n": int(m.sum()),
                "prior_median": float(np.median(prior_mm[m])),
                "estimate_median": float(np.median(estimate_mm[m])),
                "worse_than_prior": float(np.mean(estimate_mm[m] > prior_mm[m])),
            }
        )
    return out


# ---- the same on the GPU (torch): the training loop's, the batch's priors computed where its frames are ----


def table_to_torch(table: EpisodeErrors, device: Any) -> dict[str, Any]:
    """An `EpisodeErrors` table as torch tensors on `device`."""
    import torch

    return {
        "offset": torch.as_tensor(table.offset, dtype=torch.float32, device=device),
        "d_height": torch.as_tensor(table.d_height, dtype=torch.float32, device=device),
        "d_radius": torch.as_tensor(table.d_radius, dtype=torch.float32, device=device),
        "weak": torch.as_tensor(table.weak, device=device),
    }


def quat_to_matrix_torch(q: Any) -> Any:
    """`quat_to_matrix` in torch: (N, 4, w x y z) -> (N, 3, 3)."""
    import torch

    w, x, y, z = q.unbind(dim=1)
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], dim=-1),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], dim=-1),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], dim=-1),
        ],
        dim=1,
    )


def priors_torch(labels: dict[str, Any], idx: Any, table: dict[str, Any], mean_size: Any) -> Any:
    """`priors`' pinch-frame priors (B, 8) in torch, on the GPU.

    For the frames `idx` (a tensor) of `labels` (torch tensors on the device: `round`, `world`, `in_root`, `size`,
    `pinch_pos`, `pinch_quat`) and a `table_to_torch` table.
    """
    import torch

    r, w = labels["round"][idx], labels["world"][idx]
    size = labels["size"][idx]
    weak = table["weak"][r, w]
    radius = torch.where(weak, mean_size[0], size[:, 0] + table["d_radius"][r, w]).clamp(min=0.004)
    height = torch.where(weak, mean_size[1], size[:, 1] + table["d_height"][r, w]).clamp(min=0.02)
    center = labels["in_root"][idx, :3] + table["offset"][r, w]
    up = torch.tensor([0.0, 0.0, 1.0], device=center.device).expand(len(idx), 3)
    rot = quat_to_matrix_torch(labels["pinch_quat"][idx])
    pos = labels["pinch_pos"][idx]
    c = torch.einsum("nji,nj->ni", rot, center - pos)
    a = torch.einsum("nji,nj->ni", rot, up)
    return torch.cat([c, a, radius[:, None], height[:, None]], dim=1)
