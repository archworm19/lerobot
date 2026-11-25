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


def test_every_grid_head_reads_its_blocks_side_by_side():
    """Each head of the grid takes its blocks' tokens concatenated (one block: DIM wide; four: 4 x DIM)."""
    from lerobot.felix_jr.sim2real.probe_train import GRID_SPECS, build, head_inputs

    feats = {b: (torch.randn(3, DIM), torch.randn(3, GRID[0] * GRID[1], DIM)) for b in (6, 9, 10, 11, 12)}
    names = set()
    for spec in GRID_SPECS:
        cls, patches = head_inputs(feats, spec)
        assert cls.shape == (3, DIM * len(spec.blocks)) and patches.shape[-1] == DIM * len(spec.blocks)
        assert build(spec)(cls, patches).shape == (3, OUT), spec.name
        names.add(spec.name)
    assert len(names) == len(GRID_SPECS)  # (no two heads share a name)


def test_the_heads_file_round_trips_with_its_specs(tmp_path):
    """Saved with their specs, the heads load back as they were (a multi-block head too); files without specs read as
    the basic heads."""
    from dataclasses import asdict

    from lerobot.felix_jr.sim2real.probe_train import BASIC_SPECS, HeadSpec, build, load_heads

    spec = HeadSpec("attn_L9-12_q16", "attention", (9, 10, 11, 12), 16)
    head = build(spec)
    scale = torch.stack([torch.zeros(8), torch.ones(8)])
    torch.save(
        {"heads": {spec.name: head.state_dict()}, "specs": {spec.name: asdict(spec)}, "scale": scale},
        tmp_path / "a.pt",
    )
    heads, specs, back = load_heads(tmp_path / "a.pt", "cpu")
    assert specs[spec.name] == spec and torch.equal(back, scale)
    for k, v in head.state_dict().items():
        assert torch.equal(heads[spec.name].state_dict()[k], v)
    legacy = {s.name: build(s).state_dict() for s in BASIC_SPECS}
    torch.save({"heads": legacy, "scale": scale}, tmp_path / "old.pt")
    heads, specs, _ = load_heads(tmp_path / "old.pt", "cpu")
    assert set(heads) == {"linear", "mean", "attention"} and specs["attention"].blocks == (12,)


def test_augmenting_changes_colours_but_keeps_the_frame():
    """Augmented frames: the same shape and type, different values, the structure kept (still correlated)."""
    from lerobot.felix_jr.sim2real.probe_train import augment

    g = torch.Generator().manual_seed(0)
    rgb = (torch.rand(8, 168, 224, 3, generator=g) * 255).to(torch.uint8)
    rgb[:, 60:110, 90:140] = torch.tensor([30, 60, 200], dtype=torch.uint8)  # (an object)
    out = augment(rgb, g)
    assert out.shape == rgb.shape and out.dtype == torch.uint8
    assert (out != rgb).float().mean() > 0.5
    a, b = rgb.float().flatten(1), out.float().flatten(1)
    corr = ((a - a.mean(1, keepdim=True)) * (b - b.mean(1, keepdim=True))).mean(1) / (a.std(1) * b.std(1))
    assert (corr > 0.7).all(), corr
