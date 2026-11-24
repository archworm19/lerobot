"""The vision probe's heads and objective (`sim2real/probe_train.py`), on made-up tokens (no DINOv2 needed)."""

import pytest

torch = pytest.importorskip("torch")

from lerobot.felix_jr.sim2real.probe_train import DIM, GRID, HEADS, OUT, decode, loss  # noqa: E402


def _tokens(b: int = 5) -> tuple["torch.Tensor", "torch.Tensor"]:
    return torch.randn(b, DIM), torch.randn(b, GRID[0] * GRID[1], DIM)


def test_each_head_maps_tokens_to_the_outputs():
    cls, patches = _tokens()
    for name, head in HEADS.items():
        assert head()(cls, patches).shape == (5, OUT), name
    attention = HEADS["attention"]()
    attention(cls, patches)
    assert attention.last_weights is not None and attention.last_weights.shape == (5, 4, GRID[0] * GRID[1])


def test_decoding_undoes_the_standardization_and_the_loss_is_zero_at_the_truth():
    mean = torch.tensor([0.01, -0.02, 0.05, 0.0, 0.0, 0.0, 0.02, 0.10])
    std = torch.tensor([0.02, 0.02, 0.03, 1.0, 1.0, 1.0, 0.005, 0.03])
    scale = torch.stack([mean, std])
    target = torch.tensor([[0.02, -0.01, 0.04, 0.0, 0.0, 1.0, 0.0225, 0.13]])
    z = (target - mean) / std
    out = torch.cat([z[:, :3], target[:, 3:6], z[:, 6:8], torch.tensor([[20.0]])], dim=1)
    d = decode(out, scale)
    assert torch.allclose(d["center"], target[:, :3], atol=1e-6)
    assert torch.allclose(d["size"], target[:, 6:8], atol=1e-6)
    assert bool(d["in_view"][0])
    assert float(loss(out, target, torch.tensor([True]), scale)) < 1e-6


def test_out_of_view_frames_teach_only_whether_the_object_is_in_view():
    scale = torch.stack([torch.zeros(8), torch.ones(8)])
    target = torch.zeros(2, 8)
    out = torch.zeros(2, OUT)
    out[:, 8] = -20.0  # (confidently out of view)
    out[1, :3] = 100.0  # (a wild center where nothing is seen: not penalized)
    unseen = torch.tensor([False, False])
    assert float(loss(out, target, unseen, scale)) < 1e-6
