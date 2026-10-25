"""SO-101 follower whose servo bus can also carry a two-wheel base and a two-servo head.

LeRobot identifies joints by servo ID on a bus (name -> Motor(id, ...)), not by physical position, so extra
servos only need unique IDs on the bus they share with an arm. The arm keeps IDs 1-6; the head servos get 7
(neck, turns left/right) and 8 (eyes, tilts the camera up/down), following XLeRobot's numbering of 7-8 for the
head; the wheels get 9-10.

Wheels run in VELOCITY mode and are driven with `set_base_velocity`. The only wheel "calibration" is a
spin-direction sign per wheel (they are mounted mirrored), saved as `<id>_base.json` by
`python -m lerobot.felix_jr.setup.wheels`. The head servos stay in POSITION mode; their only saved data so far
is which raw direction is down / left, in `<id>_head.json` from `python -m lerobot.felix_jr.setup.head`. Their
range of motion is not calibrated yet. The extra servos are deliberately kept out of the arm interface:
`get_observation()` / `send_action()` / the saved calibration and home files only ever see the six arm
joints, so `move_all`, `record_home` and the wiggle test work unchanged.
"""

import json
import logging
import time
from pathlib import Path

import numpy as np

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

logger = logging.getLogger(__name__)

BASE_MOTOR_IDS = {"base_left_wheel": 9, "base_right_wheel": 10}
HEAD_MOTOR_IDS = {"head_neck": 7, "head_eyes": 8}  # neck turns left/right, eyes tilt up/down
STEPS_PER_DEG = 4096.0 / 360.0
# Per-joint P gain overrides (lerobot's default is 16 for every joint). With 16, shoulder_lift and
# elbow_flex fall 10-35 deg behind when lifting the arm against gravity; 32 tracks within a few degrees
# (see lift_test.py). The wrist joints carry little load and are fine at 16.
DEFAULT_JOINT_P_GAINS = {"shoulder_lift": 32, "elbow_flex": 32}


class SO101FollowerWithBase(SO101Follower):
    """`SO101Follower`, optionally with wheel servos (`with_base=True`) and head servos (`with_head=True`) on
    the same bus.

    With both False this behaves exactly like `SO101Follower`, so it is safe to use before the extra servos
    are plugged in (connecting pings every motor in the bus, and would fail on missing ones).
    """

    def __init__(
        self,
        config: SO101FollowerConfig,
        with_base: bool = True,
        max_base_degps: float = 180.0,
        joint_p_gains: dict[str, int] | None = None,
        with_head: bool = False,
    ):
        super().__init__(config)
        self.joint_p_gains = dict(DEFAULT_JOINT_P_GAINS if joint_p_gains is None else joint_p_gains)
        self.with_base = with_base
        self.with_head = with_head
        self.max_base_degps = max_base_degps
        self.base_motors: list[str] = list(BASE_MOTOR_IDS) if with_base else []
        self.head_motors: list[str] = list(HEAD_MOTOR_IDS) if with_head else []
        self.extra_motors: list[str] = self.base_motors + self.head_motors
        self.base_signs_path: Path = self.calibration_dir / f"{self.id}_base.json"
        self.head_signs_path: Path = self.calibration_dir / f"{self.id}_head.json"
        self.base_signs: dict[str, int] = dict.fromkeys(self.base_motors, 1)
        self.head_signs: dict[str, int] = dict.fromkeys(self.head_motors, 1)
        if not self.extra_motors:
            return
        self._load_signs(self.base_signs, self.base_signs_path, "wheel directions", "wheels")
        self._load_signs(self.head_signs, self.head_signs_path, "head directions", "head")

        motors = dict(self.bus.motors)
        # The extra servos are not part of the arm calibration file (`self.calibration` is what gets saved
        # to disk and stays arm-only). They get an in-memory full-range entry so that reading and writing
        # them through the bus works: wheels turn continuously, and the head range is not calibrated yet.
        extra_cal: dict[str, MotorCalibration] = {}
        for name in self.base_motors:
            motors[name] = Motor(BASE_MOTOR_IDS[name], "sts3215", MotorNormMode.RANGE_M100_100)
        for name in self.head_motors:
            motors[name] = Motor(HEAD_MOTOR_IDS[name], "sts3215", MotorNormMode.DEGREES)
        for name in self.extra_motors:
            extra_cal[name] = MotorCalibration(
                id=motors[name].id, drive_mode=0, homing_offset=0, range_min=0, range_max=4095
            )
        self.bus = FeetechMotorsBus(
            port=config.port, motors=motors, calibration={**self.calibration, **extra_cal}
        )

    def _load_signs(self, signs: dict[str, int], path: Path, what: str, script: str) -> None:
        if not signs:
            return
        if path.is_file():
            signs.update(json.loads(path.read_text()))
        else:
            logger.warning(f"No {what} at {path}; run lerobot.felix_jr.setup.{script}.")

    def calibrate(self) -> None:
        if self.extra_motors:
            raise NotImplementedError(
                "Calibrate the arm on its own (it saves an arm-only file): "
                "lerobot-calibrate --robot.type=so101_follower --robot.port=<PORT> --robot.id=<ID>"
            )
        super().calibrate()

    def configure(self) -> None:
        self._configure_motors()
        if self.joint_p_gains:
            with self.bus.torque_disabled(list(self.joint_p_gains)):  # P gain lives in EEPROM
                for joint, p_gain in self.joint_p_gains.items():
                    self.bus.write("P_Coefficient", joint, p_gain)

    def _configure_motors(self) -> None:
        if not self.extra_motors:
            return super().configure()
        # Same as SOFollower.configure for the arm joints (head servos are set up like arm joints, in
        # POSITION mode); wheels go to VELOCITY mode (stopped) instead.
        with self.bus.torque_disabled():
            self.bus.configure_motors()
            for motor in self.bus.motors:
                if motor in self.base_motors:
                    self.bus.write("Operating_Mode", motor, OperatingMode.VELOCITY.value)
                    self.bus.write("Goal_Velocity", motor, 0)
                    continue
                self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
                self.bus.write("P_Coefficient", motor, self.config.position_p_coefficient)
                self.bus.write("I_Coefficient", motor, self.config.position_i_coefficient)
                self.bus.write("D_Coefficient", motor, self.config.position_d_coefficient)
                if motor == "gripper":
                    self.bus.write("Max_Torque_Limit", motor, 500)  # 50% of max torque to avoid burnout
                    self.bus.write("Protection_Current", motor, 250)  # 50% of max current to avoid burnout
                    self.bus.write("Overload_Torque", motor, 25)  # 25% torque when overloaded

    def get_observation(self):
        obs = super().get_observation()  # also reads the extra servos' positions; drop them from the arm obs
        return {k: v for k, v in obs.items() if k.removesuffix(".pos") not in self.extra_motors}

    def scan_chain(self, max_id: int = 20, timeout_ms: int = 15) -> dict[int, list[int]]:
        """Ping IDs 0..max_id at every supported baudrate. Returns {baudrate: [ids that answered]}."""
        found: dict[int, list[int]] = {}
        if not self.bus.is_connected:
            self.bus._connect(handshake=False)
        self.bus.set_timeout(timeout_ms)  # a ping to an absent ID waits for the full timeout
        try:
            for baudrate in self.bus.model_baudrate_table["sts3215"]:
                self.bus.set_baudrate(baudrate)
                ids = [i for i in range(max_id + 1) if self.bus.ping(i) is not None]
                if ids:
                    found[baudrate] = ids
        finally:
            self.bus.set_baudrate(self.bus.default_baudrate)
            self.bus.set_timeout()
        return found

    def setup_base_motor(self, name: str, configured: tuple[str, ...] = ()) -> None:
        """Assign the servo ID for wheel `name` (see `_assign_id`)."""
        self._assign_id(name, configured)

    def setup_head_motor(self, name: str, configured: tuple[str, ...] = ()) -> None:
        """Assign the servo ID for head servo `name` (see `_assign_id`)."""
        self._assign_id(name, configured)

    def _assign_id(self, name: str, configured: tuple[str, ...] = ()) -> None:
        """Give the factory-fresh servo on the chain the ID that belongs to `name`.

        The chain is scanned first, and the one servo that is not an already `configured` one is the
        target. If anything else answers (for example the arm is still plugged into the board), nothing is
        written: re-addressing a servo that happens to answer first would rename one of the arm's own
        servos. Broadcast ping is unreliable with several servos answering at once, hence the per-ID scan.
        Servos that share an ID cannot be seen by the scan at all, which also stops here (nothing found).
        """
        known = {self.bus.motors[m].id for m in configured}
        found = self.scan_chain()
        candidates = [(b, i) for b, ids in found.items() for i in ids if i not in known]
        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected exactly one new servo on the chain (besides {sorted(known) or 'none'}), but the "
                f"scan found (baudrate -> ids): {found}. Nothing was changed. Is the arm still plugged into "
                f"the board? Check the cable and power to '{name}'."
            )
        baudrate, found_id = candidates[0]
        print(f"Found the new servo at id={found_id}, baudrate={baudrate}.")
        self.bus.setup_motor(name, initial_baudrate=baudrate, initial_id=found_id)
        if self.bus.ping(name) is None:
            raise RuntimeError(f"'{name}' did not answer at id {self.bus.motors[name].id} after being set.")
        print(f"'{name}' motor id set to {self.bus.motors[name].id}")

    def nudge_head(self, name: str, ticks: int = 80, seconds: float = 0.8, release: bool = True) -> int:
        """Move head servo `name` a few degrees (80 ticks is about 7 deg) and back. For setup.

        Works in raw servo ticks, so it needs no calibration. Returns the raw direction that was used, +1 or
        -1 (it goes -1 only if +1 would run past the end of the range), so the caller can turn the user's
        answer ("did it go down / left?") into a sign. With `release=False` the servo keeps holding its
        position afterwards (the default switches its torque off, which suits the setup script but would let
        the camera droop).
        """
        start = int(self.bus.read("Present_Position", name, normalize=False))
        direction = 1 if start + ticks <= 4095 else -1
        with self.bus.torque_disabled([name]):
            self.bus.write("Operating_Mode", name, OperatingMode.POSITION.value)
            # hold where it is when torque comes on
            self.bus.write("Goal_Position", name, start, normalize=False)
        try:
            self.bus.write("Goal_Position", name, start + direction * ticks, normalize=False)
            time.sleep(seconds)
            self.bus.write("Goal_Position", name, start, normalize=False)
            time.sleep(seconds)
        finally:
            if release:
                self.bus.disable_torque([name])
        return direction

    def wiggle_head(self, name: str, degrees: float = 8.0, seconds: float = 0.7) -> tuple[float, float]:
        """Move head servo `name` `degrees` in its positive direction (down for the eyes, left for the neck),
        then the same amount the other way, and back to where it started. The servo keeps holding its
        position. Returns how far it actually moved in each direction, in degrees, measured in those
        directions (so both should be close to `degrees` if it follows its commands)."""
        sign = self.head_signs[name]
        ticks = round(degrees * 4096.0 / 360.0)
        start = int(self.bus.read("Present_Position", name, normalize=False))
        moved = []
        for direction in (+1, -1):
            goal = int(np.clip(start + direction * sign * ticks, 0, 4095))
            self.bus.write("Goal_Position", name, goal, normalize=False)
            time.sleep(seconds)
            reached = int(self.bus.read("Present_Position", name, normalize=False))
            moved.append(direction * sign * (reached - start) * 360.0 / 4096.0)
            self.bus.write("Goal_Position", name, start, normalize=False)
            time.sleep(seconds)
        return moved[0], moved[1]

    def save_head_signs(self, signs: dict[str, int]) -> Path:
        """`signs[name]` is +1 if the raw positive direction is down (eyes) / left (neck), else -1.

        Same sign convention as the head joints in `xlerobot.urdf`: positive tilt looks down, positive pan
        turns left."""
        self.head_signs.update(signs)
        self.head_signs_path.write_text(json.dumps(self.head_signs, indent=2))
        return self.head_signs_path

    def spin_wheel(self, name: str, degps: float = 60.0, seconds: float = 1.0) -> None:
        """Spin one wheel in its raw servo direction (ignores saved signs), then stop. For setup."""
        with self.bus.torque_disabled([name]):
            self.bus.write("Operating_Mode", name, OperatingMode.VELOCITY.value)
            self.bus.write("Goal_Velocity", name, 0)
        try:
            self.bus.write("Goal_Velocity", name, int(round(degps * STEPS_PER_DEG)))
            time.sleep(seconds)
        finally:
            self.bus.write("Goal_Velocity", name, 0)
            self.bus.disable_torque([name])

    def save_base_signs(self, signs: dict[str, int]) -> Path:
        self.base_signs.update(signs)
        self.base_signs_path.write_text(json.dumps(self.base_signs, indent=2))
        return self.base_signs_path

    def set_base_velocity(self, left_degps: float, right_degps: float) -> None:
        """Command wheel speeds in deg/s; positive drives the robot forward (uses the saved wheel signs)."""
        if not self.with_base:
            raise RuntimeError("This follower was created with with_base=False.")
        speeds = {"base_left_wheel": left_degps, "base_right_wheel": right_degps}
        limit = self.max_base_degps
        raw = {
            name: int(round(self.base_signs[name] * max(-limit, min(limit, v)) * STEPS_PER_DEG))
            for name, v in speeds.items()
        }
        self.bus.sync_write("Goal_Velocity", raw)

    def get_base_velocity(self) -> dict[str, float]:
        """Measured wheel speeds in deg/s."""
        raw = self.bus.sync_read("Present_Velocity", self.base_motors, normalize=False)
        return {name: self.base_signs[name] * v / STEPS_PER_DEG for name, v in raw.items()}

    def stop_base(self) -> None:
        if self.with_base:
            self.bus.sync_write("Goal_Velocity", dict.fromkeys(self.base_motors, 0))

    def exercise_base(self, degps: float = 40.0, seconds: float = 1.0) -> list[dict]:
        """Run each wheel alone, then both, forward and back, and measure the speeds. The wheels must be able
        to spin freely (robot lifted) or have room to roll. Returns one dict per step with the label, the
        commanded (left, right) speeds and the measured ones in deg/s (positive = forward)."""
        steps = [
            ("left wheel forward", degps, 0.0),
            ("left wheel backward", -degps, 0.0),
            ("right wheel forward", 0.0, degps),
            ("right wheel backward", 0.0, -degps),
            ("both forward", degps, degps),
            ("both backward", -degps, -degps),
        ]
        results = []
        try:
            for label, left, right in steps:
                self.set_base_velocity(left, right)
                time.sleep(seconds)
                measured = self.get_base_velocity()  # sampled while still moving
                self.stop_base()
                time.sleep(0.3)
                results.append(
                    {
                        "label": label,
                        "commanded": (left, right),
                        "measured": (measured["base_left_wheel"], measured["base_right_wheel"]),
                    }
                )
        finally:
            self.stop_base()
        return results

    def disconnect(self) -> None:
        try:
            self.stop_base()  # velocity mode keeps spinning until told otherwise
        except Exception as e:
            logger.warning(f"Could not stop the base before disconnecting: {e!r}")
        super().disconnect()
