"""The scene's randomized look (`sim2real/appearance.py`): the textures, the room's spec."""

import numpy as np
import pytest

from lerobot.felix_jr.sim2real.appearance import KINDS, SURFACES, texture


def test_every_texture_kind_is_an_image_with_structure():
    """Each kind: (256, 256, 3) uint8, not one flat colour, and different from one draw to the next."""
    rng = np.random.default_rng(0)
    for kind in KINDS:
        a, b = texture(kind, rng), texture(kind, rng)
        assert a.shape == (256, 256, 3) and a.dtype == np.uint8, kind
        assert a.astype(float).std() > 1.0, kind
        assert np.abs(a.astype(int) - b.astype(int)).mean() > 2.0, kind
    for surface, weights in SURFACES.items():
        assert set(weights) <= set(KINDS) and abs(sum(weights.values()) - 1) < 1e-9, surface


def test_the_room_is_drawn_only_and_textured():
    """The room compiles: its materials each with a texture, its planes drawn only (no contact), the lights."""
    mujoco = pytest.importorskip("mujoco")
    from lerobot.felix_jr.sim2real.appearance import (
        FLOOR_MATERIALS,
        TABLE_MATERIALS,
        WALL_MATERIALS,
        room_spec,
    )

    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="table_body", pos=[-0.46, 0.0, 0.73])
    body.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.4, 0.6, 0.015])
    room_spec(spec, body, (0.8, 1.2, 0.03))
    m = spec.compile()
    assert m.nmat == TABLE_MATERIALS + FLOOR_MATERIALS + WALL_MATERIALS == m.ntex
    assert (m.mat_texid[:, mujoco.mjtTextureRole.mjTEXROLE_RGB] >= 0).all()
    assert m.nlight == 2
    for name in ("table_top", "floor", "wall_ahead", "wall_left", "wall_right"):
        g = m.geom(name)
        assert g.type[0] == mujoco.mjtGeom.mjGEOM_PLANE and g.contype[0] == 0 and g.conaffinity[0] == 0, name
    # the top a hair above the table's, covering it
    assert m.geom("table_top").pos[2] > m.geom("table").size[2]
