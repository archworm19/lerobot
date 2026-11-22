"""The start pool in the mjlab environment: a reset event term that writes one of the pool's starts into each world.

`PoolReset` is an mjlab class-based event term (`EventTermCfg(func=PoolReset, mode="reset", params=...)`), bound to
its entities the mjlab way: `params["robot"]` and `params["object"]` are `SceneEntityCfg`s, which mjlab resolves
against the scene before building the term. Built once (`__init__`): the pool (`start_pool.StartPool`, its `.npz`)
read, laid out as the environment's model lays out its state, put on the device. At each reset (`__call__`), for each
world being reset: a pool drawn by `pool_weights`, then a start of it at random; its joints' positions and velocities,
its actuators' commands (and the robot's joint position targets, which mjlab derives the controls from) written.

A pool whose objects vary in size (`StartPool.sizes_vary`) also has each reset world's object made its start's
(`mjlab_object.write_bodies`: the shape's fields, its mass times a log-uniform draw in `mass_range`), the fields it
writes declared per world and the constants recomputed after (`mjlab_object.sized_declaration`: this term's
`model_fields` and `recompute`, read by mjlab once the term is built). Everything that knows the object's shape is
`mjlab_object`'s; this term draws starts and writes state. The model is where each world's object size is kept
(`mjlab_object.object_dims` reads it); the term keeps only what is not in the model: each world's start (`rows`: its
pool, `pools_of_worlds`) and the error in the size the actor is told (`size_error`, drawn at each reset,
`size_error_m`). A pool of one object declares nothing: no field per world, nothing recomputed.

`pool_reset(env)`: the environment's term (mjlab's event manager keeps a built term in its config's `func`).
"""

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from . import mjlab_object as obj
from .start_pool import POOLS, StartPool, _layout

if TYPE_CHECKING:
    import torch
    from mjlab.envs import ManagerBasedRlEnv

TERM = "reset_from_pool"  # the event term's name in `mjlab_env.env_cfg`


class PoolReset:
    """Mjlab reset event term: a start of the pool written into each world being reset (the module docstring)."""

    def __init__(self, cfg: Any, env: "ManagerBasedRlEnv") -> None:
        """Read the pool (`cfg.params["path"]`), lay it out as `env`'s model does, put it on the device."""
        import torch

        params = cfg.params
        self.path = str(params["path"])
        self.pool = StartPool.load(Path(self.path))
        self.sized = self.pool.sizes_vary
        if (
            self.sized
        ):  # (the object's fields per world, the constants recomputed: `mjlab_object.sized_declaration`)
            self.model_fields, self.recompute = obj.sized_declaration()
        m, dev = env.sim.mj_model, env.device
        qpos, qvel, ctrl = self.pool.layout(m)
        self.qpos = torch.as_tensor(qpos, device=dev, dtype=env.sim.data.qpos.dtype)
        self.qvel = torch.as_tensor(qvel, device=dev, dtype=env.sim.data.qvel.dtype)
        self.ctrl = torch.as_tensor(ctrl, device=dev, dtype=env.sim.data.ctrl.dtype)
        self.objects = torch.as_tensor(self.pool.objects, device=dev, dtype=torch.float32)
        weights = dict(params["pool_weights"])
        self.members = [torch.as_tensor(np.flatnonzero(self.pool.pool == p), device=dev) for p in POOLS]
        w = np.array([weights.get(p, 0.0) if len(self.members[i]) else 0.0 for i, p in enumerate(POOLS)])
        if w.sum() <= 0:
            raise ValueError(f"no weighted pool has starts: {weights}")
        self.weights = torch.as_tensor(w / w.sum(), device=dev, dtype=torch.float32)
        # mjlab derives the controls from the robot's joint position targets (each step, and at a reset): the start's
        # commands go in as targets, by name (each actuator drives the joint it is named after)
        robot = params["robot"].name
        names = [n.split("/", 1)[-1] for n in env.scene[robot].joint_names]
        pairs = [(a, names.index(n)) for a, n in enumerate(_layout(m)["actuators"]) if n in names]
        self.target_columns = torch.as_tensor([a for a, _ in pairs], device=dev)
        self.target_joints = torch.as_tensor([j for _, j in pairs], device=dev)
        self.object_entity = params["object"].name
        obj.object_ids(
            env, self.object_entity
        )  # (refuses an object that is not one cylinder, now rather than later)
        self.rows = torch.zeros(env.num_envs, dtype=torch.long, device=dev)  # each world's start
        self.size_error = torch.zeros(
            env.num_envs, 2, device=dev
        )  # the actor's radius and height off by this

    def __call__(
        self,
        env: "ManagerBasedRlEnv",
        env_ids: "torch.Tensor",
        path: str,
        pool_weights: Mapping[str, float],
        robot: Any,
        object: Any,  # noqa: A002  (the entity's role, as mjlab's own terms name theirs)
        mass_range: tuple[float, float] = (1.0, 1.0),
        size_error_m: float = 0.0,
    ) -> None:
        """A start for each world in `env_ids` (the params are `__init__`'s, as mjlab passes them each call)."""
        import torch

        if len(env_ids) == 0:
            return
        rows = self._draw(len(env_ids))
        self.rows[env_ids] = rows
        sim = env.sim
        sim.data.qpos[env_ids] = self.qpos[rows]
        sim.data.qvel[env_ids] = self.qvel[rows]
        sim.data.ctrl[env_ids] = self.ctrl[rows]
        env.scene[robot.name].set_joint_position_target(
            self.ctrl[rows][:, self.target_columns], joint_ids=self.target_joints, env_ids=env_ids
        )
        if self.sized:
            lo, hi = np.log(mass_range[0]), np.log(mass_range[1])
            scale = torch.exp(lo + (hi - lo) * torch.rand(len(env_ids), device=rows.device))
            obj.write_bodies(env, env_ids, self.objects[rows], scale, self.object_entity, self.pool.shape)
            self.size_error[env_ids] = size_error_m * torch.randn(len(env_ids), 2, device=rows.device)

    def _draw(self, n: int) -> "torch.Tensor":
        """`n` starts: a pool each by the weights, then a start of it at random."""
        import torch

        which = torch.multinomial(self.weights, n, replacement=True)
        rows = torch.empty(n, dtype=torch.long, device=self.weights.device)
        for i, members in enumerate(self.members):
            mask = which == i
            if mask.any():
                rows[mask] = members[torch.randint(len(members), (int(mask.sum()),), device=rows.device)]
        return rows

    def pools_of_worlds(self) -> list[str]:
        """Each world's start pool (its last reset's)."""
        return [str(p) for p in self.pool.pool[self.rows.cpu().numpy()]]


def pool_reset(env: "ManagerBasedRlEnv") -> PoolReset | None:
    """The environment's start-pool reset term (None: its starts are not a pool's)."""
    try:
        term = env.event_manager.get_term_cfg(TERM).func
    except (KeyError, ValueError):
        return None
    return term if isinstance(term, PoolReset) else None
