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
  reaching the grasp point (where the object's axis passes the pinch point in a grasp: `mjlab_object.grasp_point`,
  fit over the pool's starts in the jaws), the object between the jaws (`mjlab_object.between_jaws`), both fingers on it, its rise above
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

    from .mjlab_object import MjlabCylinder

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


# ---- the robot ------------------------------------------------------------------------------------------------------


def pinch_pose(env: "ManagerBasedRlEnv") -> tuple["torch.Tensor", "torch.Tensor"]:
    """The pinch site's position and orientation (w, x, y, z), world frame (= the root frame here)."""
    rob = env.scene["robot"]
    cache = getattr(env, "_felix_pinch_id", None)
    if cache is None:
        ids, _ = rob.find_sites("pinch")
        cache = env._felix_pinch_id = ids[0]  # type: ignore[attr-defined]
    return rob.data.site_pos_w[:, cache], rob.data.site_quat_w[:, cache]


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
# Generic over the object: each takes it (`obj`, an `mjlab_object.MjlabCylinder`, through the term's `params`) and
# asks it for what the stage needs: where a grasp puts it (`grasp_point`), the jaws' region (`between_jaws`), its
# height standing (`rest_z`), what upright means (`upright`).


def reach(env: "ManagerBasedRlEnv", obj: "MjlabCylinder", std: float = REACH_STD_M) -> "torch.Tensor":
    """exp(-|the object's axis from the grasp point (`grasp_point`)| / std)."""
    import torch

    perp, _ = obj.axis_from_pinch(env)
    return torch.exp(-torch.linalg.norm(perp - obj.grasp_point(env), dim=1) / std)


def enclosed(env: "ManagerBasedRlEnv", obj: "MjlabCylinder") -> "torch.Tensor":
    """1 where the object is between the jaws (`between_jaws`)."""
    return obj.between_jaws(env)


def both_fingers(env: "ManagerBasedRlEnv") -> "torch.Tensor":
    """1 where both fingers' pads touch the object (the pads' sensors look for it: `sensors`)."""
    return finger_contacts(env).prod(dim=1)


def rise(env: "ManagerBasedRlEnv", obj: "MjlabCylinder") -> "torch.Tensor":
    """The object's height above standing on the table (`rest_z`), over the lift, in [0, 1]."""
    center, _ = obj.pose(env)
    return ((center[:, 2] - obj.rest_z(env)) / LIFT_M).clamp(0.0, 1.0)


def held(env: "ManagerBasedRlEnv", obj: "MjlabCylinder") -> "torch.Tensor":
    """1 where the object is up `HELD_FRACTION` of the lift, both fingers on it, and upright within `HELD_MAX_TILT_DEG`.

    (Upright since 2026-11-21: the first run's policy held bottles tilted 20-30 deg, gripped high and swinging, which
    the lift alone paid for in full.)
    """
    upright = obj.upright(env, HELD_MAX_TILT_DEG)
    return (rise(env, obj) >= HELD_FRACTION).float() * both_fingers(env) * upright.float()


def swing(env: "ManagerBasedRlEnv", obj: "MjlabCylinder") -> "torch.Tensor":
    """The object's angular speed (rad/s) while it is off the table (rise past a fifth of the lift): a penalty's term."""
    import torch

    speed = torch.linalg.norm(env.scene[obj.entity].data.root_link_ang_vel_w, dim=1)
    return speed * (rise(env, obj) > 0.2).float()


def tipped(env: "ManagerBasedRlEnv", obj: "MjlabCylinder") -> "torch.Tensor":
    """Where the object is past `refine.KNOCKED_DEG` from upright."""
    return ~obj.upright(env, KNOCKED_DEG)


def arm_on_table(env: "ManagerBasedRlEnv") -> "torch.Tensor":
    """Where any of the arm touches the table."""
    return _touching(env, ARM_TABLE) > 0


def sensors(object_match: Any) -> tuple[Any, ...]:
    """The contact sensors: each finger's pads on the object, the arm on the table.

    `object_match`: which elements are the object (its `contact_match`).
    """
    from mjlab.sensor import ContactMatch, ContactSensorCfg

    def pads(name: str, which: str) -> Any:
        return ContactSensorCfg(
            name=name,
            primary=ContactMatch(mode="geom", pattern=rf"finger_{which}_pad\d+", entity="robot"),
            secondary=object_match,
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
