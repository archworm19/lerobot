"""`sim.render_random`, briefly: the arm moves and a video and a contact sheet are written (skipped without the
model cached, or without an OpenGL context for offscreen rendering)."""

import pytest

pytest.importorskip("mujoco")

from lerobot.felix_jr.sim.assets import is_cached  # noqa: E402
from lerobot.felix_jr.sim.render_random import render_random  # noqa: E402

pytestmark = pytest.mark.skipif(not is_cached(), reason="the SO-101 model is not cached")


def test_a_short_random_render(tmp_path):
    try:
        video, sheet = render_random(tmp_path / "random", seconds=0.5)
    except Exception as e:  # no OpenGL context (a headless machine)
        if "gl" in str(e).lower() or "context" in str(e).lower():
            pytest.skip(f"no offscreen rendering here: {e}")
        raise
    assert video.stat().st_size > 1000 and sheet.stat().st_size > 1000
