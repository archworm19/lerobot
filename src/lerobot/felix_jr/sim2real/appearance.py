"""The scene's look randomized for the wrist camera: procedural textures, a room, lights, the object's colour.

Why: the probe on real frames (`docs/sim2real_grasp.md`) looked at the plain table's far edge and the black past it.

Built into the table's entity (`room_spec`, added to `mjlab_env.table_spec`'s with `appearance=True`), all of it drawn
only (no collisions, so the physics and the teacher are unchanged):

- textures made here (`texture`: wood grain, planks, mottled, woven, speckled, near plain; 256 x 256), each in a material
  of its own: `TABLE_MATERIALS` for the table, `FLOOR_MATERIALS`, `WALL_MATERIALS`;
- a textured top over the table carried on past its far edge and to its sides (`EXTENSION_M`), a floor below it and
  three walls: planes (MuJoCo Warp's renderer textures planes and meshes, not boxes);
- two lights over the table.

At each reset (`events`): the table (its top and the extension alike), the floor and the walls each take one of their
materials (mjlab's `dr.geom_matid`, per world: MuJoCo Warp's renderer reads a geom's material per world); the lights
move and change brightness and tint; the object takes a colour (`object_color`: about half near white, the rest any
hue, mostly muted).
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import mujoco
    import torch
    from mjlab.envs import ManagerBasedRlEnv

TEXTURE_PX = 256
TABLE_MATERIALS, FLOOR_MATERIALS, WALL_MATERIALS = 24, 12, 12
EXTENSION_M = (1.0, 1.0)  # the table's top carried on this far past its far edge, and to each side
FLOOR_BELOW_M = 0.75  # the floor this far below the table's top
WALL_AT_M = 2.0  # the walls this far from the table's center
WHITE_SHARE = 0.5  # the share of objects near white


# ---- textures -------------------------------------------------------------------------------------------------------


def _noise(rng: np.random.Generator, size: int, octaves: int = 5, base: int = 4) -> np.ndarray:
    """Fractal value noise (size, size) in [0, 1]: random grids upsampled bilinearly, finer ones weaker."""
    out = np.zeros((size, size))
    weight, total = 1.0, 0.0
    for o in range(octaves):
        n = base * 2**o
        grid = rng.random((n + 1, n + 1))
        t = np.linspace(0, n, size, endpoint=False)
        i = t.astype(int)
        f = t - i
        f = f * f * (3 - 2 * f)  # (smoothstep)
        rows = grid[i][:, i] * (1 - f)[None, :] + grid[i][:, i + 1] * f[None, :]
        rows2 = grid[i + 1][:, i] * (1 - f)[None, :] + grid[i + 1][:, i + 1] * f[None, :]
        out += weight * (rows * (1 - f)[:, None] + rows2 * f[:, None])
        total += weight
        weight *= 0.5
    out /= total
    return (out - out.min()) / max(out.max() - out.min(), 1e-9)


def _mix(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    return a[None, None] * (1 - t[..., None]) + b[None, None] * t[..., None]


def _color(rng: np.random.Generator, lo: float, hi: float, sat: float = 0.25) -> np.ndarray:
    """An RGB colour (0-1): a grey in [lo, hi] tinted by a random hue up to `sat`."""
    grey = rng.uniform(lo, hi)
    tint = rng.dirichlet(np.ones(3)) * 3 - 1  # (mean 0)
    return np.clip(grey * (1 + sat * rng.uniform(0, 1) * tint), 0, 1)


def _wood_palette(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    light = np.array([0.78, 0.6, 0.42]) * rng.uniform(0.6, 1.25)
    return np.clip(light, 0, 1), np.clip(light * rng.uniform(0.55, 0.8), 0, 1)


def wood(rng: np.random.Generator, size: int = TEXTURE_PX) -> np.ndarray:
    """Wood grain: long streaks along one axis, a few broad rings bent by noise, fine fibres."""
    y, x = np.mgrid[0:size, 0:size] / size
    warp = _noise(rng, size, 3, 2)
    rings = np.sin(
        2
        * np.pi
        * (rng.uniform(2.5, 7) * (y + 0.08 * warp + 0.04 * np.sin(2 * np.pi * x * rng.uniform(0.5, 1.5))))
    )
    grain = (0.5 + 0.5 * rings) ** rng.uniform(1.5, 3.0)  # (dark rings narrower than the light wood between)
    fibres = _noise(rng, size, 2, 64)[:, :1] * np.ones((1, size))  # (streaks along x)
    fibres = 0.5 * fibres + 0.5 * _noise(rng, size, 4, 8)
    t = np.clip(0.55 * grain + 0.45 * fibres, 0, 1)
    light, dark = _wood_palette(rng)
    return _mix(light, dark, t)


def planks(rng: np.random.Generator, size: int = TEXTURE_PX) -> np.ndarray:
    """Wood in planks: grain, each plank its own shade, dark seams between."""
    img = wood(rng, size)
    n = int(rng.integers(3, 7))
    edges = np.linspace(0, size, n + 1).astype(int)
    for a, b in zip(
        edges[:-1], edges[1:], strict=True
    ):  # (the seams along the grain: the rings vary along y)
        img[a:b] *= rng.uniform(0.8, 1.15)
        img[a : a + 2] *= 0.45
    return np.clip(img, 0, 1)


def mottled(rng: np.random.Generator, size: int = TEXTURE_PX) -> np.ndarray:
    """Two colours blended by fractal noise (stone, concrete, worn paint, carpet)."""
    a = _color(rng, 0.25, 0.85)
    b = np.clip(a * rng.uniform(0.6, 1.3) + rng.normal(0, 0.05, 3), 0, 1)
    return _mix(a, b, _noise(rng, size, 5, int(rng.integers(2, 8))))


def woven(rng: np.random.Generator, size: int = TEXTURE_PX) -> np.ndarray:
    """A woven cloth: two crossing threads' patterns over a colour."""
    y, x = np.mgrid[0:size, 0:size] / size
    f = rng.uniform(30, 90)
    pattern = (
        0.5
        + 0.25 * np.sin(2 * np.pi * f * x) * np.sign(np.sin(2 * np.pi * f * y))
        + 0.25 * _noise(rng, size, 2, 32)
    )
    a = _color(rng, 0.2, 0.85, sat=0.5)
    return _mix(a, np.clip(a * 0.7, 0, 1), np.clip(pattern, 0, 1))


def speckled(rng: np.random.Generator, size: int = TEXTURE_PX) -> np.ndarray:
    """A laminate or granite: a colour, faint mottling, scattered specks."""
    img = mottled(rng, size) * 0.3 + _color(rng, 0.3, 0.95)[None, None] * 0.7
    specks = rng.random((size, size)) < rng.uniform(0.005, 0.04)
    img[specks] = np.clip(img[specks] * rng.uniform(0.3, 1.7), 0, 1)
    return np.clip(img, 0, 1)


def plain(rng: np.random.Generator, size: int = TEXTURE_PX) -> np.ndarray:
    """Near plain: a colour with a faint unevenness (painted walls, a white table)."""
    return np.clip(
        _color(rng, 0.3, 0.97, sat=0.15)[None, None] * (0.94 + 0.06 * _noise(rng, size, 3, 4))[..., None],
        0,
        1,
    )


KINDS: dict[str, Callable[[np.random.Generator, int], np.ndarray]] = {
    "wood": wood,
    "planks": planks,
    "mottled": mottled,
    "woven": woven,
    "speckled": speckled,
    "plain": plain,
}
# each surface's texture tile (m): MuJoCo's plane texgen repeats a texture every 2 / texrepeat meters
TILE_M = {"table": (0.25, 0.6), "floor": (0.5, 1.5), "wall": (0.6, 1.5)}
# which kinds each surface draws from, and how often
SURFACES = {
    "table": {"wood": 0.35, "planks": 0.1, "speckled": 0.2, "plain": 0.2, "mottled": 0.1, "woven": 0.05},
    "floor": {"planks": 0.3, "wood": 0.1, "mottled": 0.35, "speckled": 0.15, "woven": 0.1},
    "wall": {"plain": 0.6, "mottled": 0.3, "speckled": 0.1},
}


def texture(kind: str, rng: np.random.Generator, size: int = TEXTURE_PX) -> np.ndarray:
    """A procedural texture (size, size, 3) uint8 of `kind` (`KINDS`)."""
    return (np.clip(KINDS[kind](rng, size), 0, 1) * 255).round().astype(np.uint8)


# ---- the room -------------------------------------------------------------------------------------------------------


def _materials(spec: "mujoco.MjSpec", surface: str, n: int, rng: np.random.Generator) -> list[str]:
    """`n` textured materials for `surface` (named `<surface>_mat_<i>`), their kinds drawn by `SURFACES`."""
    import mujoco

    kinds = list(SURFACES[surface])
    p = np.array([SURFACES[surface][k] for k in kinds])
    names = []
    for i in range(n):
        kind = kinds[int(rng.choice(len(kinds), p=p / p.sum()))]
        img = texture(kind, rng)
        tex = spec.add_texture(
            name=f"{surface}_tex_{i}",
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            width=img.shape[1],
            height=img.shape[0],
            nchannel=3,
        )
        tex.data = img.tobytes()
        mat = spec.add_material(name=f"{surface}_mat_{i}")
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
        repeat = 2.0 / rng.uniform(*TILE_M[surface])
        mat.texrepeat = [repeat, repeat]
        mat.specular = float(rng.uniform(0.0, 0.25))
        mat.shininess = float(rng.uniform(0.1, 0.6))
        names.append(mat.name)
    return names


def room_spec(
    spec: "mujoco.MjSpec", body: "mujoco.MjsBody", table_size: tuple[float, float, float], seed: int = 0
) -> None:
    """The room added to the table's `body`: materials, the drawn-only extension, floor and walls, the lights.

    `body`'s origin is the table's center; `table_size` its full extents (the module docstring).
    """
    import mujoco

    rng = np.random.default_rng(seed)
    first = {
        s: _materials(spec, s, n, rng)[0]
        for s, n in (("table", TABLE_MATERIALS), ("floor", FLOOR_MATERIALS), ("wall", WALL_MATERIALS))
    }
    lx, ly, lz = table_size
    # drawn only, and planes: MuJoCo Warp's renderer textures planes and meshes (a box takes one texel's colour); a
    # plane is a rectangle (its half sizes) seen from its +z side
    drawn = {"contype": 0, "conaffinity": 0, "group": 0, "density": 0.0, "type": mujoco.mjtGeom.mjGEOM_PLANE}
    far, side = EXTENSION_M
    top = lz / 2
    # the top: over the table and carried on past its far edge (the table's -x, away from the robot) and to the sides,
    # a hair above the table's own (which it covers)
    body.add_geom(
        name="table_top",
        size=[(lx + far) / 2, ly / 2 + side, 0.05],
        pos=[-far / 2, 0.0, top + 0.0005],
        material=first["table"],
        **drawn,
    )
    floor = top - FLOOR_BELOW_M
    body.add_geom(
        name="floor", size=[6.0, 6.0, 0.05], pos=[0.0, 0.0, floor], material=first["floor"], **drawn
    )
    wall_h = 2.5
    c = float(np.cos(np.pi / 4))
    for name, pos, quat, half in (
        (
            "wall_ahead",
            [-WALL_AT_M, 0.0, floor + wall_h / 2],
            [c, 0.0, c, 0.0],
            [wall_h / 2, WALL_AT_M],
        ),  # facing +x
        (
            "wall_left",
            [0.0, WALL_AT_M, floor + wall_h / 2],
            [c, c, 0.0, 0.0],
            [WALL_AT_M, wall_h / 2],
        ),  # facing -y
        (
            "wall_right",
            [0.0, -WALL_AT_M, floor + wall_h / 2],
            [c, -c, 0.0, 0.0],
            [WALL_AT_M, wall_h / 2],
        ),  # facing +y
    ):
        body.add_geom(name=name, size=[*half, 0.05], pos=pos, quat=quat, material=first["wall"], **drawn)
    for i, (x, y) in enumerate(((-0.3, 0.4), (0.2, -0.5))):
        body.add_light(name=f"room_light_{i}", pos=[x, y, 1.2], dir=[0.0, 0.0, -1.0], diffuse=[0.5, 0.5, 0.5])


# ---- the resets -----------------------------------------------------------------------------------------------------


def object_color(
    env: "ManagerBasedRlEnv",
    env_ids: "torch.Tensor | None",
    entity: str = "object",
    white_share: float = WHITE_SHARE,
) -> None:
    """Each world's object coloured: about `white_share` near white, the rest any hue, mostly muted.

    (A white bottle's white a little warm or cool; its geoms' rgba: the object has no material.)
    """
    import torch

    model = env.sim.model
    ids = env.scene[entity].indexing.geom_ids.to(model.geom_rgba.device)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=model.geom_rgba.device)
    n = len(env_ids)
    dev = model.geom_rgba.device
    white = torch.rand(n, device=dev) < white_share
    near_white = 0.82 + 0.16 * torch.rand(n, 1, device=dev) + 0.03 * torch.randn(n, 3, device=dev)
    hue = torch.rand(n, 3, device=dev)
    grey = torch.rand(n, 1, device=dev)
    muted = grey + (hue - hue.mean(dim=1, keepdim=True)) * torch.rand(n, 1, device=dev) * 0.8
    rgb = torch.where(white[:, None], near_white, muted).clamp(0.03, 1.0)
    rgba = torch.cat([rgb, torch.ones(n, 1, device=dev)], dim=1)
    model.geom_rgba[env_ids[:, None], ids[None, :]] = rgba[:, None, :]


def events() -> dict[str, Any]:
    """The reset terms (mjlab's `EventTermCfg`s) that randomize the look (the module docstring)."""
    from mjlab.envs.mdp import dr
    from mjlab.managers.event_manager import EventTermCfg, requires_model_fields
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    def surface(geoms: tuple[str, ...], prefix: str) -> EventTermCfg:
        return EventTermCfg(
            func=dr.geom_matid,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("table", geom_names=geoms, material_names=(f"{prefix}_mat_.*",)),
                "shared_random": True,
            },
        )

    # (mjlab 1.6's `dr.light_diffuse` declares `light_dir` as the field it needs: its own here, declaring the right one)
    @requires_model_fields("light_diffuse")
    def light_diffuse(env: "ManagerBasedRlEnv", env_ids: "torch.Tensor | None", **kwargs: Any) -> None:
        dr.light_diffuse(env, env_ids, **kwargs)

    lights = SceneEntityCfg("table", light_names=("room_light_.*",))
    return {
        "look_table": surface(("table", "table_top"), "table"),
        "look_floor": surface(("floor",), "floor"),
        "look_walls": surface(("wall_.*",), "wall"),
        "look_light_pos": EventTermCfg(
            func=dr.light_pos,
            mode="reset",
            params={"ranges": {0: (-0.8, 0.4), 1: (-0.8, 0.8), 2: (0.6, 1.8)}, "asset_cfg": lights},
        ),
        "look_light_diffuse": EventTermCfg(
            func=light_diffuse,
            mode="reset",
            params={"ranges": (0.08, 0.45), "asset_cfg": lights},
        ),
        "look_object": EventTermCfg(
            func=requires_model_fields("geom_rgba")(object_color),
            mode="reset",
            params={},
        ),
    }
