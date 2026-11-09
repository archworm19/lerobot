"""`harness.depth_anything` against the `depth-anything-3` package's own `inference` (its full install, offline).

The frames are `test_sfm`'s synthetic wrist sweep over a textured table (rendered here: no images in git); the
reference depths (`reference/da3_small_parity.npz`, every 6th pixel) came from the package's `DepthAnything3.inference`
on the same frames, on the CPU, with the same poses and lens (2026-11-09). Skipped without the weights in the Hugging
Face cache (no download in tests): fetch them once with `DepthAnything.from_hub()`.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("depth_anything_3")

from test_sfm import CAMERA, render, table_texture, wrist_sweep  # noqa: E402

from lerobot.felix_jr.harness.depth_anything import REPO, WEIGHTS, DepthAnything  # noqa: E402

REFERENCE = Path(__file__).parents[1] / "reference" / "da3_small_parity.npz"


@pytest.fixture(scope="module")
def model():
    from huggingface_hub import try_to_load_from_cache

    if not isinstance(try_to_load_from_cache(REPO, WEIGHTS), str):
        pytest.skip("DA3-SMALL weights not in the Hugging Face cache")
    return DepthAnything.from_hub("cpu")


def test_the_posed_and_single_depth_match_the_package(model):
    ref = np.load(REFERENCE)
    poses = wrist_sweep()
    images = [render(table_texture(), p) for p in poses]
    posed = model.depth(images, poses, CAMERA.matrix())
    assert posed.depth.shape == tuple(ref["posed_shape"])
    ratio = posed.depth[:, ::6, ::6] / ref["posed_depth"]
    # per frame within 0.5% (the package's alignment uses evo, ours the same fit; float16 differences), pixels mostly 1%
    assert np.abs(np.median(ratio.reshape(len(ratio), -1), axis=1) - 1).max() < 0.005, ratio
    assert np.mean(np.abs(ratio - 1) > 0.01) < 0.02
    np.testing.assert_allclose(
        posed.extrinsics, np.array([np.linalg.inv(p) for p in poses])[:, :3], atol=1e-5
    )
    single = model.depth(images[-1:])
    assert single.depth.shape == tuple(ref["single_shape"])
    assert np.median(np.abs(single.depth[:, ::6, ::6] / ref["single_depth"] - 1)) < 0.005
