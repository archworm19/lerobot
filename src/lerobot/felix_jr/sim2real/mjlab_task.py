"""The grasp-and-lift task's terms for the mjlab environment: the action, the observations, the rewards (`mjlab_env`).

The first experiment is a positive control (`docs/sim_deploy.md`, step 6): the actor and the critic both see the
simulator's truth (the object's pose with a little noise, the fingers' contacts), starts from the bank's pools.

- **Action: delta joint targets** (`DeltaJointTargetAction`). Each step the policy nudges the servos' targets, which
  accumulate (ManiSkill's `pd_joint_target_delta_pos`; OmniReset and IndustReal step their end-effector pose the same
  way): a step of `ARM_STEP_DEG` a joint (30 deg/s at 30 Hz; the harness moves at 20) and `GRIPPER_STEP_UNITS` for the
  jaw, each target kept within `ARM_CAP_DEG` / `GRIPPER_CAP_UNITS` of the joint's reading and within its range. The
  cap is what lets small steps still push: a P servo's torque is the gap, built up over steps. The sessions' settled
  moves stop 0.9-2.7 deg from their targets (the joint) and the jaw 3 units on an object (2026-11-21, every logged
  execution): the cap twice that, and the 50 N m/rad servos at full torque within it; the jaw's room for the harness's
  squeeze (10 past contact). mjlab's own relative action steps from the reading, so it cannot build a gap past one step.
- **Observations:** the joints (position, velocity), the object's center and axis in the root frame and in the pinch
  frame (the fixed fingertip's: `robot`'s `pinch` site; x across the jaws, z along the gripper), the servos' loads (the
  actuators' torque over the servo's stall torque: the real servos report a load), each finger's pads touching the
  object (privileged), the last action.
- **Rewards** (positive and staged: an all-negative reward with episodes ended by failure teaches ending them):
  reaching the grasp point (where the object's axis passes the pinch point in a grasp: `grasp_point`, measured over
  the pool's starts in the jaws), the object between the jaws (`BETWEEN_*_M`), both fingers on it, its rise above
  resting on the table towards the lift, held (upright within `HELD_MAX_TILT_DEG`); a small action-rate penalty and
  one on the object's swing while lifted. **Terminations:**
  the arm on the table, the object tipped past `refine.KNOCKED_DEG`; the episode's length (not success: the held reward
  keeps paying, so holding is learned, not rushing).
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel
from lerobot.felix_jr.harness.grasping import LIFT_M

from . import robot
from .refine import HELD_FRACTION, KNOCKED_DEG

if TYPE_CHECKING:
    import torch
    from mjlab.envs import ManagerBasedRlEnv

try:  # (mjlab only with `--extra felix_jr_gpu`: the action term subclasses its base)
    from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
except ImportError:  # pragma: no cover
    ActionTerm, ActionTermCfg = object, object  # type: ignore[misc,assignment]

ARM_STEP_DEG = 1.0
ARM_CAP_DEG = 5.0
GRIPPER_STEP_UNITS = 2.0
GRIPPER_CAP_UNITS = 15.0
STALL_NM = robot.GRIPPER_STALL_NM  # (the STS3215's, every joint's servo)
FIXED_PADS, MOVING_PADS, ARM_TABLE = "fixed_pads", "moving_pads", "arm_table"  # the contact sensors' names
REACH_STD_M = 0.05
# where the object's axis passes the pinch point, in the pinch frame (x across the jaws, z along the gripper): measured
# over the pyri pool's starts (2026-11-21; the pinch to the axis line, the grasp's height along the axis apart): in the
# jaws (-16, 1, -11) mm (5th-95th: x -18 to -15, y +-7, z -18 to +2); the jaws open around it (-30, 2, -12) (x -36 to
# -24); at the attack (-30, -1, +34) (z +20 to +52)
# For objects of several sizes, refit over the multi1 pool's starts (radius 20-30 mm; 2026-11-21): across the jaws the
# axis sits about a radius from the pinch point, in the jaws x = -0.79 r + 2.0 mm (residual 1.0 mm; at pyri's 22.5 mm
# -15.8, as measured), z = +0.67 r - 24.1 (residual 6.7); the jaws open around it x = -1.10 r - 4.9 (residual 3.2).
# Each axis (per metre of radius, at no radius; m):
GRASP_POINT_FIT = ((-0.79, 0.0020), (0.0, 0.001), (0.67, -0.0241))
# the box "between the jaws", from the open jaws' starts to the closed ones': across the jaws its ends follow the radius
# as those do (at pyri's radius -38 and -12 mm, as measured); y and z fixed
BETWEEN_X_FIT = ((-1.10, -0.0049 - 0.0084), (-0.79, 0.0020 + 0.0038))  # low, high
BETWEEN_LOW_M = (None, -0.015, -0.025)
BETWEEN_HIGH_M = (None, 0.015, 0.006)
ON_BODY_MARGIN_M = 0.005  # the grasp's height this far inside the object's ends at least
HELD_MAX_TILT_DEG = 10.0  # held only upright within this


def gripper_rad_per_unit(model: XleRobotModel, arm: str) -> float:
    """The jaw's opening (rad) per `gripper.pos` unit, about its middle (the map is near linear there)."""
    if arm not in model.gripper_maps:
        return float(np.radians(1.0))
    return float(np.radians(model.opening_deg(arm, 30.0) - model.opening_deg(arm, 20.0)) / 10.0)


# ---- the action -----------------------------------------------------------------------------------------------------


@dataclass(kw_only=True)
class DeltaJointTargetActionCfg(ActionTermCfg):  # type: ignore[misc]
    """The delta-joint-target action (the module docstring): per actuated joint (regex) its step and cap (rad)."""

    actuator_names: tuple[str, ...] = (".*",)
    step_rad: Mapping[str, float] = field(default_factory=dict)
    cap_rad: Mapping[str, float] = field(default_factory=dict)

    def build(self, env: "ManagerBasedRlEnv") -> "DeltaJointTargetAction":
        """The term."""
        return DeltaJointTargetAction(self, env)


class DeltaJointTargetAction(ActionTerm):  # type: ignore[misc]
    """Targets that accumulate: target += clip(action, -1, 1) x step, kept within cap of the reading and the range."""

    cfg: DeltaJointTargetActionCfg

    def __init__(self, cfg: DeltaJointTargetActionCfg, env: "ManagerBasedRlEnv") -> None:
        """The joints the actuators drive, their steps and caps, the targets (from the entity's, at each reset)."""
        import re

        import torch

        super().__init__(cfg=cfg, env=env)
        ids, names = self._entity.find_joints_by_actuator_names(cfg.actuator_names)
        self._ids = torch.tensor(ids, device=self.device, dtype=torch.long)
        self._names = list(names)

        def per_joint(values: Mapping[str, float]) -> "torch.Tensor":
            out = []
            for name in self._names:
                found = [v for k, v in values.items() if re.fullmatch(k, name.split("/")[-1])]
                if not found:
                    raise ValueError(f"no step/cap for joint {name!r} in {dict(values)}")
                out.append(found[0])
            return torch.tensor(out, device=self.device, dtype=torch.float32)

        self._step = per_joint(cfg.step_rad)
        self._cap = per_joint(cfg.cap_rad)
        self._raw = torch.zeros(self.num_envs, len(ids), device=self.device)
        self._targets = self._entity.data.joint_pos_target[:, self._ids].clone()
        limits = self._entity.data.joint_pos_limits[:, self._ids]
        self._low, self._high = limits[..., 0], limits[..., 1]

    @property
    def action_dim(self) -> int:
        """One per actuated joint."""
        return len(self._names)

    @property
    def raw_action(self) -> "torch.Tensor":
        """The last action, as given."""
        return self._raw

    @property
    def targets(self) -> "torch.Tensor":
        """The targets now (rad)."""
        return self._targets

    def process_actions(self, actions: "torch.Tensor") -> None:
        """The targets stepped (once a policy step), kept within the cap of the reading and the range."""
        import torch

        self._raw[:] = actions
        step = actions.clamp(-1.0, 1.0) * self._step
        reading = self._entity.data.joint_pos[:, self._ids]
        targets = torch.clamp(self._targets + step, reading - self._cap, reading + self._cap)
        self._targets[:] = torch.clamp(targets, self._low, self._high)

    def apply_actions(self) -> None:
        """The targets to the servos (each physics step of the policy step)."""
        self._entity.set_joint_position_target(self._targets, joint_ids=self._ids)

    def reset(self, env_ids: "torch.Tensor | slice | None" = None) -> None:
        """The reset worlds' targets: what the reset set (a start pool's commands)."""
        ids = slice(None) if env_ids is None else env_ids
        self._raw[ids] = 0.0
        self._targets[ids] = self._entity.data.joint_pos_target[ids][:, self._ids]


def delta_action_cfg(model: XleRobotModel, arm: str) -> DeltaJointTargetActionCfg:
    """The action as the module docstring sets it (the jaw's units as its opening's radians)."""
    unit = gripper_rad_per_unit(model, arm)
    arm_joints = "|".join(robot.JOINTS)
    return DeltaJointTargetActionCfg(
        entity_name="robot",
        step_rad={
            f"({arm_joints})": float(np.radians(ARM_STEP_DEG)),
            robot.GRIPPER: GRIPPER_STEP_UNITS * unit,
        },
        cap_rad={f"({arm_joints})": float(np.radians(ARM_CAP_DEG)), robot.GRIPPER: GRIPPER_CAP_UNITS * unit},
    )


# ---- geometry -------------------------------------------------------------------------------------------------------


def _rotate_inverse(quat: "torch.Tensor", v: "torch.Tensor") -> "torch.Tensor":
    """`v` rotated by the inverse of `quat` (w, x, y, z), batched."""
    import torch

    w, xyz = quat[:, :1], quat[:, 1:]
    t = 2.0 * torch.cross(-xyz, v, dim=1)
    return v + w * t + torch.cross(-xyz, t, dim=1)


def _rotate(quat: "torch.Tensor", v: "torch.Tensor") -> "torch.Tensor":
    """`v` rotated by `quat` (w, x, y, z), batched."""
    import torch

    w, xyz = quat[:, :1], quat[:, 1:]
    t = 2.0 * torch.cross(xyz, v, dim=1)
    return v + w * t + torch.cross(xyz, t, dim=1)


def _pinch(env: "ManagerBasedRlEnv") -> tuple["torch.Tensor", "torch.Tensor"]:
    """The pinch site's position and orientation (w, x, y, z), world frame (= the root frame here)."""
    rob = env.scene["robot"]
    cache = getattr(env, "_felix_pinch_id", None)
    if cache is None:
        ids, _ = rob.find_sites("pinch")
        cache = env._felix_pinch_id = ids[0]  # type: ignore[attr-defined]
    return rob.data.site_pos_w[:, cache], rob.data.site_quat_w[:, cache]


def _object(env: "ManagerBasedRlEnv") -> tuple["torch.Tensor", "torch.Tensor"]:
    """The object's center and axis (its body z), world frame."""
    import torch

    data = env.scene["object"].data
    z = torch.zeros_like(data.root_link_pos_w)
    z[:, 2] = 1.0
    return data.root_link_pos_w, _rotate(data.root_link_quat_w, z)


def object_in_pinch(env: "ManagerBasedRlEnv") -> "torch.Tensor":
    """The object's center (3) and axis (3) in the pinch frame."""
    import torch

    center, axis = _object(env)
    pos, quat = _pinch(env)
    return torch.cat([_rotate_inverse(quat, center - pos), _rotate_inverse(quat, axis)], dim=1)


def object_in_root(env: "ManagerBasedRlEnv") -> "torch.Tensor":
    """The object's center (3) and axis (3) in the root frame."""
    import torch

    center, axis = _object(env)
    return torch.cat([center, axis], dim=1)


def servo_loads(env: "ManagerBasedRlEnv") -> "torch.Tensor":
    """Each actuator's torque over the servo's stall torque (the real servos' load, as a fraction)."""
    return env.sim.data.actuator_force / STALL_NM


def _touching(env: "ManagerBasedRlEnv", sensor: str) -> "torch.Tensor":
    found = env.scene[sensor].data.found
    return (found > 0).any(dim=1).float()


def finger_contacts(env: "ManagerBasedRlEnv") -> "torch.Tensor":
    """Each finger's pads touching the object (fixed, moving): 0 or 1."""
    import torch

    return torch.stack([_touching(env, FIXED_PADS), _touching(env, MOVING_PADS)], dim=1)


# ---- rewards and terminations ---------------------------------------------------------------------------------------


def axis_from_pinch(env: "ManagerBasedRlEnv") -> tuple["torch.Tensor", "torch.Tensor"]:
    """Where the object's axis passes the pinch point, and the grasp's height along it.

    The pinch to the axis line (pinch frame, (N, 3)) and how far along the axis from the object's center (N,): a
    cylinder's grasp is the first, at a height the second.
    """
    o = object_in_pinch(env)
    center, axis = o[:, :3], o[:, 3:]
    along = (center * axis).sum(dim=1)
    return center - along[:, None] * axis, along


def object_radius(env: "ManagerBasedRlEnv") -> "torch.Tensor":
    """Each world's object's radius (N,), from the model (per world once resized, `start_pool.resize_objects`)."""
    from .start_pool import _object_ids

    cache = env.__dict__.setdefault("_felix_object_ids", {})
    if "geom" not in cache:
        cache["body"], cache["geom"] = _object_ids(env.sim.mj_model)
    radius = env.sim.model.geom_size[..., cache["geom"], 0].reshape(-1)
    return radius.expand(env.num_envs) if radius.numel() == 1 else radius


def grasp_point(env: "ManagerBasedRlEnv") -> "torch.Tensor":
    """Where the object's axis passes the pinch point in a grasp, for each world's radius (pinch frame, (N, 3))."""
    import torch

    r = object_radius(env).float()
    return torch.stack([a * r + b for a, b in GRASP_POINT_FIT], dim=1)


def reach(env: "ManagerBasedRlEnv", std: float = REACH_STD_M) -> "torch.Tensor":
    """exp(-|the object's axis from the grasp point (`grasp_point`)| / std)."""
    import torch

    perp, _ = axis_from_pinch(env)
    return torch.exp(-torch.linalg.norm(perp - grasp_point(env), dim=1) / std)


def between_jaws(env: "ManagerBasedRlEnv", height: float) -> "torch.Tensor":
    """1 where the object's axis is in the box between the jaws and the grasp on its body.

    `height`: the object's; each world's own when the start pool keeps them (`env.felix_object_size`).
    """
    import torch

    sizes = getattr(env, "felix_object_size", None)
    if sizes is not None:
        height = sizes[:, 1]

    perp, along = axis_from_pinch(env)
    r = object_radius(env).float()
    (a_lo, b_lo), (a_hi, b_hi) = BETWEEN_X_FIT
    n = len(r)
    low = torch.stack(
        [a_lo * r + b_lo, *(torch.full((n,), v, device=env.device) for v in BETWEEN_LOW_M[1:])], dim=1
    )
    high = torch.stack(
        [a_hi * r + b_hi, *(torch.full((n,), v, device=env.device) for v in BETWEEN_HIGH_M[1:])], dim=1
    )
    inside = ((perp > low) & (perp < high)).all(dim=1)
    return (inside & (along.abs() < height / 2 - ON_BODY_MARGIN_M)).float()


def both_fingers(env: "ManagerBasedRlEnv") -> "torch.Tensor":
    """1 where both fingers' pads touch the object."""
    return finger_contacts(env).prod(dim=1)


def rise(env: "ManagerBasedRlEnv") -> "torch.Tensor":
    """The object's height above resting on the table, over the lift, in [0, 1].

    Resting: the start pool's (`felix_object_rest_z`: the table and half the object), else the object's default.
    """
    data = env.scene["object"].data
    rest = getattr(env, "felix_object_rest_z", None)
    if rest is None:
        rest = data.default_root_state[:, 2]
    return ((data.root_link_pos_w[:, 2] - rest) / LIFT_M).clamp(0.0, 1.0)


def tilt_cos(env: "ManagerBasedRlEnv") -> "torch.Tensor":
    """The cosine of the object's tilt from upright (its axis's world z)."""
    _, axis = _object(env)
    return axis[:, 2]


def held(env: "ManagerBasedRlEnv") -> "torch.Tensor":
    """1 where the object is up `HELD_FRACTION` of the lift, both fingers on it, and upright within `HELD_MAX_TILT_DEG`.

    (Upright since 2026-11-21: the first run's policy held bottles tilted 20-30 deg, gripped high and swinging, which
    the lift alone paid for in full.)
    """
    upright = tilt_cos(env) > float(np.cos(np.radians(HELD_MAX_TILT_DEG)))
    return (rise(env) >= HELD_FRACTION).float() * both_fingers(env) * upright.float()


def swing(env: "ManagerBasedRlEnv") -> "torch.Tensor":
    """The object's angular speed (rad/s) while it is off the table (rise past a fifth of the lift): a penalty's term."""
    import torch

    speed = torch.linalg.norm(env.scene["object"].data.root_link_ang_vel_w, dim=1)
    return speed * (rise(env) > 0.2).float()


def tipped(env: "ManagerBasedRlEnv") -> "torch.Tensor":
    """Where the object's axis is past `refine.KNOCKED_DEG` from upright."""
    _, axis = _object(env)
    return axis[:, 2] < float(np.cos(np.radians(KNOCKED_DEG)))


def arm_on_table(env: "ManagerBasedRlEnv") -> "torch.Tensor":
    """Where any of the arm touches the table."""
    return _touching(env, ARM_TABLE) > 0


def sensors() -> tuple[Any, ...]:
    """The contact sensors: each finger's pads on the object, the arm on the table."""
    from mjlab.sensor import ContactMatch, ContactSensorCfg

    def pads(name: str, which: str) -> Any:
        return ContactSensorCfg(
            name=name,
            primary=ContactMatch(mode="geom", pattern=rf"finger_{which}_pad\d+", entity="robot"),
            secondary=ContactMatch(mode="body", pattern="object", entity="object"),
            fields=("found",),
            reduce="none",
        )

    return (
        pads(FIXED_PADS, "fixed"),
        pads(MOVING_PADS, "moving"),
        ContactSensorCfg(
            name=ARM_TABLE,
            primary=ContactMatch(mode="subtree", pattern="base", entity="robot"),
            secondary=ContactMatch(mode="body", pattern="table_body", entity="table"),
            fields=("found",),
            reduce="none",
        ),
    )
