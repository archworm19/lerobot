"""Depth Anything 3 (DA3-SMALL) on wrist frames: dense depth, from our camera poses and lens.

`docs/object_grasping.md` section 7: at a hold, a short wrist roll's frames, posed by the arm's joints (the model's
poses do as well as structure from motion for a wrist-only roll) and scaled by the table, give DA3's depth for every
frame; a primitive fitted to the graspable part gives the contact patches.

The `depth-anything-3` package is used as source only (`pyproject.toml`: its declared dependencies replaced by the ones
these modules import). Its `api.py` is not imported (it loads exporters and viewers at import time); `DepthAnything`
does what its `inference` does for posed frames: the network from the package's config, the weights from the Hub,
the package's input and output processing, the poses normalized as it does, and the prediction aligned to our poses
by a similarity (Umeyama) fit of the camera centers (the package uses `evo` for that; the same fit, no RANSAC, which
it only uses from 10 frames). `tests/felix_jr/test_depth_anything.py` checks it against the package's own output.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.utils.import_utils import _depth_anything_available, require_package

if TYPE_CHECKING or _depth_anything_available:
    import torch

MODEL = "da3-small"
REPO = (
    "depth-anything/DA3-SMALL"  # Apache-2.0 (the larger DA3 models' licenses differ: check before switching)
)
WEIGHTS = "model.safetensors"
PROCESS_RES = 504
PROCESS_RES_METHOD = "upper_bound_resize"


@dataclass(frozen=True)
class Depth:
    """DA3's output for N frames, at its processing resolution (`PROCESS_RES` on the long side).

    `depth` (N, h, w) is along each camera's optical axis, in the poses' units; `conf` (N, h, w) its confidence;
    `intrinsics` (N, 3, 3) the lens at that resolution; `extrinsics` (N, 3, 4) world-to-camera, the poses given.
    """

    depth: np.ndarray
    conf: np.ndarray
    intrinsics: np.ndarray
    extrinsics: np.ndarray

    def points(self, i: int) -> np.ndarray:
        """Frame `i`'s pixels as world points (h, w, 3)."""
        k, z = self.intrinsics[i], self.depth[i]
        v, u = np.mgrid[0 : z.shape[0], 0 : z.shape[1]]
        camera = np.dstack([(u - k[0, 2]) / k[0, 0] * z, (v - k[1, 2]) / k[1, 1] * z, z])
        w2c = np.vstack([self.extrinsics[i], [0.0, 0.0, 0.0, 1.0]])
        c2w = np.linalg.inv(w2c)
        return camera @ c2w[:3, :3].T + c2w[:3, 3]


def umeyama(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """The similarity (R, t, s) that best maps `source` points (N, 3) onto `target`: target ~ s R source + t."""
    mu_s, mu_t = source.mean(axis=0), target.mean(axis=0)
    xs, xt = source - mu_s, target - mu_t
    var_s = float((xs**2).sum(axis=1).mean())
    cov = xt.T @ xs / len(source)
    u, d, vt = np.linalg.svd(cov)
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2, 2] = -1.0
    rotation = u @ sign @ vt
    scale = float(np.trace(np.diag(d) @ sign) / var_s) if var_s > 0 else 1.0
    return rotation, mu_t - scale * rotation @ mu_s, scale


class DepthAnything:
    """DA3-SMALL, loaded once (`from_hub`); `depth` runs it on frames with our poses and lens."""

    def __init__(self, net: Any, device: "torch.device"):
        """Use `from_hub`."""
        from depth_anything_3.utils.io.input_processor import InputProcessor
        from depth_anything_3.utils.io.output_processor import OutputProcessor

        self.net, self.device = net, device
        self.input_processor, self.output_processor = InputProcessor(), OutputProcessor()

    @classmethod
    def from_hub(cls, device: str | None = None) -> "DepthAnything":
        """The network from the package's config and the weights from the Hub (downloaded once, then cached)."""
        require_package("depth-anything-3", extra="felix_jr", import_name="depth_anything_3")
        from depth_anything_3.cfg import create_object, load_config
        from depth_anything_3.registry import MODEL_REGISTRY
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        net = create_object(load_config(MODEL_REGISTRY[MODEL]))
        state = load_file(hf_hub_download(REPO, WEIGHTS))
        # the weights are the package's wrapper's: the network under "model."
        state = {k.removeprefix("model."): v for k, v in state.items()}
        missing, unexpected = net.load_state_dict(state, strict=False)
        # the checkpoint has no weights for the auxiliary (ray) output convolutions, which the posed path does not use
        # (the package's own loader is not strict either): anything else missing is a mismatch
        missing = [k for k in missing if "output_conv2_aux" not in k]
        if missing or unexpected:
            raise RuntimeError(f"DA3 weights do not fit: missing {missing[:5]}, unexpected {unexpected[:5]}")
        chosen = torch.device(
            device
            or (
                "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            )
        )
        return cls(net.to(chosen).eval(), chosen)

    def depth(
        self,
        images: Sequence[np.ndarray],
        world_from_camera: Sequence[np.ndarray] | None = None,
        intrinsics: Sequence[np.ndarray] | np.ndarray | None = None,
    ) -> Depth:
        """Depth for `images` (RGB uint8, undistorted), with the cameras' poses (4x4, world from camera) and lens.

        With poses, the depth is in their units (aligned to them: the package's `align_to_input_ext_scale`); without,
        in DA3's own (its own poses, unscaled: they did not work for our grasps, `object_grasping.md` 7).
        """
        exts = None if world_from_camera is None else np.array([np.linalg.inv(p) for p in world_from_camera])
        ks = None if intrinsics is None else np.asarray(intrinsics, dtype=np.float32)
        if ks is not None and ks.ndim == 2:
            ks = np.repeat(ks[None], len(images), axis=0)
        imgs, ex, ix = self.input_processor(
            list(images),
            None if exts is None else exts.astype(np.float32),
            ks,
            PROCESS_RES,
            PROCESS_RES_METHOD,
        )
        x = imgs.to(self.device)[None].float()
        ex_t = None if ex is None else ex.to(self.device)[None].float()
        ix_t = None if ix is None else ix.to(self.device)[None].float()
        with torch.inference_mode():
            dtype = (
                torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else torch.float16
            )
            with torch.autocast(device_type=x.device.type, dtype=dtype):
                raw = self.net(x, _normalized(ex_t), ix_t, [], False, False, "saddle_balanced")
        prediction = self.output_processor(raw)
        depth, conf = prediction.depth, prediction.conf
        if ex is None:
            return Depth(depth, conf, prediction.intrinsics, prediction.extrinsics)
        # align: the prediction's camera centers onto the given ones' (a similarity); its depth by the scale
        predicted = _centers(prediction.extrinsics)
        given = _centers(ex.numpy())
        _, _, scale = umeyama(given, predicted)
        return Depth(depth / scale, conf, ix.numpy(), ex.numpy()[:, :3, :])


def _centers(w2c: np.ndarray) -> np.ndarray:
    """Camera centers (N, 3) of world-to-camera extrinsics (N, 3 or 4, 4)."""
    out = []
    for e in w2c:
        full = np.vstack([e[:3], [0.0, 0.0, 0.0, 1.0]])
        out.append(np.linalg.inv(full)[:3, 3])
    return np.array(out)


def _normalized(ex: "torch.Tensor | None") -> "torch.Tensor | None":
    """The poses as the package feeds them: relative to the first camera, scaled so the median distance is 1."""
    if ex is None:
        return None
    from depth_anything_3.utils.geometry import affine_inverse

    ex = ex.clone()
    norm = ex @ affine_inverse(ex[:, :1])
    centers = affine_inverse(norm)[..., :3, 3]
    norm[..., :3, 3] = norm[..., :3, 3] / torch.clamp(torch.median(centers.norm(dim=-1)), min=1e-1)
    return norm
