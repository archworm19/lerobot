"""The wrist camera rendered in mjlab, and what it costs: the frames, and the step's time with and without them.

  python -m lerobot.felix_jr.sim2real.render_bench --pool <a start pool .npz> [--num-envs 256 1024 4096]
      [--size 224 168] [--steps 50] [--out <dir>] [--device cuda:0]

For each number of worlds: the environment (`mjlab_env.env_cfg`, randomized as in training) stepped with random
actions, without the camera and with it (mjlab's `CameraSensorCfg` wrapping the arm's calibrated `wrist_camera`, RGB
at `--size`, the camera's own module left out as in `view.render`); the render alone (`sim.sense`); and a ViT-S/14
(DINOv2-S's shape: 12 layers, width 384, 6 heads; random weights, the cost alone) on the batch of frames, in half
precision. Prints steps and frames a second and the GPU memory's peak; writes a sheet of frames (`--out`).

The size: the real wrist frames are 640 x 480 (4:3), so 224 x 168 (16 x 12 patches of 14 px) keeps their field of view
without stretching; the camera's intrinsics are its calibrated ones, scaled with the resolution.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel
from lerobot.felix_jr.harness import api

from . import mjlab_env as me
from .robot import COLLISION_GROUP, OWN_MODULE_GROUPS, VISUAL_GROUP, WRIST_CAMERA
from .start_pool import POOLS

SENSOR = "wrist_rgb"
# what the wrist camera sees: the default groups (the table and the object: 0) and the arm as drawn, not its collision
# hulls nor the camera's own module
GEOM_GROUPS = tuple(
    g for g in (0, 1, VISUAL_GROUP) if g != COLLISION_GROUP and g not in OWN_MODULE_GROUPS.values()
)


def camera_cfg(width: int, height: int, data_types: tuple[str, ...] = ("rgb",)) -> Any:
    """The wrist camera as an mjlab sensor: `data_types` ("rgb", "depth", "segmentation") at `width` x `height`."""
    from mjlab.sensor import CameraSensorCfg

    return CameraSensorCfg(
        name=SENSOR,
        camera_name=f"robot/{WRIST_CAMERA}",
        width=width,
        height=height,
        data_types=data_types,
        use_textures=True,
        use_shadows=False,
        enabled_geom_groups=GEOM_GROUPS,
    )


class VitS14(nn.Module):
    """DINOv2 ViT-S/14's shape (patch 14, width 384, 12 pre-norm layers of 6 heads, MLP x4, a class token)."""

    def __init__(self, width: int = 384, depth: int = 12, heads: int = 6) -> None:
        """The layers, randomly initialized."""
        super().__init__()
        self.patch = nn.Conv2d(3, width, 14, 14)
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        layer = nn.TransformerEncoderLayer(
            width, heads, 4 * width, activation="gelu", batch_first=True, norm_first=True
        )
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """`rgb` (B, H, W, 3) uint8 -> tokens (B, 1 + H/14 * W/14, width)."""
        x = rgb.permute(0, 3, 1, 2).to(self.patch.weight.dtype).div_(255.0)
        x = self.patch(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(len(x), -1, -1), x], dim=1)
        return self.norm(self.blocks(x))


def _timed(fn: Any, repeats: int, device: str) -> float:
    """`fn`'s mean time (ms) over `repeats` calls after one warm-up, the device synchronized around them."""
    sync = torch.cuda.synchronize if device.startswith("cuda") else (lambda: None)
    fn()
    sync()
    started = time.perf_counter()
    for _ in range(repeats):
        fn()
    sync()
    return 1000 * (time.perf_counter() - started) / repeats


def sheet(frames: np.ndarray, cols: int = 8) -> np.ndarray:
    """(N, H, W, 3) frames tiled `cols` wide."""
    n, h, w, _ = frames.shape
    rows = -(-n // cols)
    out = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, f in enumerate(frames):
        r, c = divmod(i, cols)
        out[r * h : (r + 1) * h, c * w : (c + 1) * w] = f
    return out


def _run(
    model: XleRobotModel,
    pool: Path,
    num_envs: int,
    size: tuple[int, int] | None,
    steps: int,
    device: str,
    out: Path | None,
) -> dict[str, float]:
    """The environment with the wrist camera at `size` (None: without), timed; with it, the render and the ViT too."""
    cfg = me.env_cfg(
        model, num_envs=num_envs, start_pool=pool, pool_weights=dict.fromkeys(POOLS, 1.0), randomize=True
    )
    if size is not None:
        cfg.scene.sensors = (*cfg.scene.sensors, camera_cfg(*size))
    cuda = device.startswith("cuda")
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    env = me.make_env(cfg, device=device)
    env.reset()
    n_act = env.action_manager.total_action_dim

    def act(scale: float = 0.3) -> torch.Tensor:
        return scale * (2 * torch.rand(num_envs, n_act, device=device) - 1)

    result = {"step_ms": _timed(lambda: env.step(act()), steps, device)}
    if size is not None:
        sensor = env.scene.sensors[SENSOR]
        result["render_ms"] = _timed(env.sim.sense, steps, device)
        rgb = sensor.data.rgb
        assert rgb is not None and rgb.shape == (num_envs, size[1], size[0], 3), rgb.shape
        if out is not None:
            import cv2

            out.mkdir(parents=True, exist_ok=True)
            env.reset()
            frames = [sensor.data.rgb[:16].cpu().numpy().copy()]
            for _ in range(10):  # (whole actions: random ones of a third average out over a few steps)
                env.step(act(1.0))
            frames.append(sensor.data.rgb[:16].cpu().numpy().copy())
            img = sheet(np.concatenate(frames))
            cv2.imwrite(str(out / f"frames_{num_envs}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        vit = VitS14().to(device).eval()
        dtype = torch.bfloat16 if cuda else torch.float32
        with torch.inference_mode(), torch.autocast(device_type=device.split(":")[0], dtype=dtype):
            result["vit_ms"] = _timed(lambda: vit(rgb), max(steps // 5, 3), device)
        if cuda:
            result["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
    env.close()
    if cuda:
        torch.cuda.empty_cache()
    return result


def bench(
    pool: Path, num_envs: int, size: tuple[int, int], steps: int, device: str, out: Path | None
) -> dict[str, float]:
    """One number of worlds: the step without and with the camera, the render, the ViT; frames saved to `out`."""
    model = XleRobotModel.load(api.MODEL)
    plain = _run(model, pool, num_envs, None, steps, device, None)
    seen = _run(model, pool, num_envs, size, steps, device, out)
    result: dict[str, float] = {"num_envs": num_envs, "step_ms": plain["step_ms"]}
    result |= {k if k != "step_ms" else "step_camera_ms": v for k, v in seen.items()}
    result["camera_extra_ms"] = result["step_camera_ms"] - result["step_ms"]
    result["frames_per_s_render"] = 1000 * num_envs / result["render_ms"]
    result["frames_per_s_vit"] = 1000 * num_envs / result["vit_ms"]
    result["env_steps_per_s_with_vit"] = 1000 * num_envs / (result["step_camera_ms"] + result["vit_ms"])
    return result


def vit_variants(batch: int, size: tuple[int, int], device: str, repeats: int = 10) -> dict[str, float]:
    """The ViT alone on `batch` random frames (ms): under autocast (as `bench`), its weights in bf16, and those compiled."""
    rgb = torch.randint(0, 256, (batch, size[1], size[0], 3), dtype=torch.uint8, device=device)
    out: dict[str, float] = {}
    with torch.inference_mode():
        vit = VitS14().to(device).eval()
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
            out["autocast"] = _timed(lambda: vit(rgb), repeats, device)
        half = VitS14().to(device, torch.bfloat16).eval()
        out["bf16"] = _timed(lambda: half(rgb), repeats, device)
    compiled = torch.compile(half)
    with torch.inference_mode():
        started = time.perf_counter()
        compiled(rgb)  # (the compile itself)
        out["compile_s"] = time.perf_counter() - started
        out["bf16_compiled"] = _timed(lambda: compiled(rgb), repeats, device)
    return out


def main(argv: list[str] | None = None) -> None:
    """The command line (the module docstring)."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, nargs="+", default=[256, 1024, 4096])
    parser.add_argument("--size", type=int, nargs=2, default=[224, 168], metavar=("W", "H"))
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=None, help="a folder for the frames and the numbers")
    parser.add_argument("--vit-only", action="store_true", help="only the ViT's variants, on random frames")
    args = parser.parse_args(argv)
    if args.vit_only:
        for n in args.num_envs:
            v = vit_variants(n, tuple(args.size), args.device)
            print(
                f"ViT-S/14 on {n} frames at {args.size[0]}x{args.size[1]}: "
                + ", ".join(
                    f"{k} {x:.1f} ms ({1000 * n / x:.0f} frames/s)" for k, x in v.items() if k != "compile_s"
                )
                + f"; compiled in {v['compile_s']:.0f} s",
                flush=True,
            )
        return
    results = []
    for n in args.num_envs:
        r = bench(args.pool, n, tuple(args.size), args.steps, args.device, args.out)
        results.append(r)
        print(
            f"{n} worlds at {args.size[0]}x{args.size[1]}: step {r['step_ms']:.1f} ms, with the camera "
            f"{r['step_camera_ms']:.1f} ms (render alone {r['render_ms']:.1f} ms, "
            f"{r['frames_per_s_render']:.0f} frames/s); ViT-S/14 {r['vit_ms']:.1f} ms ({r['frames_per_s_vit']:.0f} "
            f"frames/s); together {r['env_steps_per_s_with_vit']:.0f} env steps/s"
            + (f"; peak {r['peak_gb']:.1f} GB" if "peak_gb" in r else ""),
            flush=True,
        )
    if args.out is not None:
        (args.out / "render_bench.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
