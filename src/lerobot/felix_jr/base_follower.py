"""SO-101 follower whose servo bus also carries a two-wheel base.

LeRobot identifies joints by servo ID on a bus (name -> Motor(id, ...)), not by physical position, so the
wheels only need unique IDs on the bus they share with an arm. The arm keeps IDs 1-6, the head servos are
reserved for 7-8 (XLeRobot convention) and the wheels get 9-10.

Wheels run in VELOCITY mode and are driven with `set_base_velocity`. The only wheel "calibration" is a
spin-direction sign per wheel (they are mounted mirrored), saved as `<id>_base.json` by
`python -m lerobot.felix_jr.setup_base`. The wheels are deliberately kept out of
the arm interface: `get_observation()` / `send_action()` / the saved calibration and home files only ever
see the six arm joints, so `move_all`, `record_home` and the wiggle test work unchanged.
"""

import json
import logging
import time
from pathlib import Path

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

logger = logging.getLogger(__name__)

BASE_MOTOR_IDS = {"base_left_wheel": 9, "base_right_wheel": 10}
STEPS_PER_DEG = 4096.0 / 360.0


class SO101FollowerWithBase(SO101Follower):
    """`SO101Follower`, optionally with wheel servos on the same bus (`with_base=True`).

    With `with_base=False` this behaves exactly like `SO101Follower`, so it is safe to use before the
    wheels are plugged in (connecting pings every motor in the bus, and would fail on missing wheels).
    """

    def __init__(self, config: SO101FollowerConfig, with_base: bool = True, max_base_degps: float = 180.0):
        super().__init__(config)
        self.with_base = with_base
        self.max_base_degps = max_base_degps
        self.base_motors: list[str] = []
        if not with_base:
            return
        self.base_motors = list(BASE_MOTOR_IDS)
        self.base_signs_path: Path = self.calibration_dir / f"{self.id}_base.json"
        self.base_signs: dict[str, int] = dict.fromkeys(self.base_motors, 1)
        if self.base_signs_path.is_file():
            self.base_signs.update(json.loads(self.base_signs_path.read_text()))
        else:
            logger.warning(f"No wheel directions at {self.base_signs_path}; run lerobot.felix_jr.setup_base.")
        motors = {
            **self.bus.motors,
            **{name: Motor(i, "sts3215", MotorNormMode.RANGE_M100_100) for name, i in BASE_MOTOR_IDS.items()},
        }
        # Wheels turn continuously: no homing offset, full range. This is only held in memory and
        # `self.calibration` (what gets saved to disk) stays arm-only.
        wheel_cal = {
            name: MotorCalibration(id=i, drive_mode=0, homing_offset=0, range_min=0, range_max=4095)
            for name, i in BASE_MOTOR_IDS.items()
        }
        self.bus = FeetechMotorsBus(
            port=config.port, motors=motors, calibration={**self.calibration, **wheel_cal}
        )

    def calibrate(self) -> None:
        if self.with_base:
            raise NotImplementedError(
                "Calibrate the arm on its own (it saves an arm-only file): "
                "lerobot-calibrate --robot.type=so101_follower --robot.port=<PORT> --robot.id=<ID>"
            )
        super().calibrate()

    def configure(self) -> None:
        if not self.with_base:
            return super().configure()
        # Same as SOFollower.configure for the arm joints; wheels go to VELOCITY mode (stopped) instead.
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
        obs = super().get_observation()  # also reads the wheels' positions; drop them from the arm obs
        return {k: v for k, v in obs.items() if k.removesuffix(".pos") not in self.base_motors}

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
        """Assign the servo ID for wheel `name`.

        With nothing in `configured`, only that wheel may be on the chain with the board (found by
        broadcast ping). Otherwise the chain is scanned and the one servo that is not an already
        `configured` wheel is the target, so a configured wheel can't be picked by mistake. Broadcast
        ping is unreliable with two servos answering at once, hence the per-ID scan. The arm must be off
        the chain either way (its shoulder_pan is ID 1, like a factory wheel).
        """
        if not configured:
            self.bus.setup_motor(name)
        else:
            known = {self.bus.motors[m].id for m in configured}
            found = self.scan_chain()
            candidates = [(b, i) for b, ids in found.items() for i in ids if i not in known]
            if len(candidates) != 1:
                raise RuntimeError(
                    f"Expected exactly one servo besides the configured wheels {sorted(known)}, but the "
                    f"scan found (baudrate -> ids): {found}. Check the cable/power to '{name}'."
                )
            baudrate, found_id = candidates[0]
            print(f"Found the new servo at id={found_id}, baudrate={baudrate}.")
            self.bus.setup_motor(name, initial_baudrate=baudrate, initial_id=found_id)
        print(f"'{name}' motor id set to {self.bus.motors[name].id}")

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

    def disconnect(self) -> None:
        try:
            self.stop_base()  # velocity mode keeps spinning until told otherwise
        except Exception as e:
            logger.warning(f"Could not stop the base before disconnecting: {e!r}")
        super().disconnect()
