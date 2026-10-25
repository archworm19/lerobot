"""Print the limit/gain registers of every arm servo (read-only; nothing is written, torque is untouched).

Useful for comparing joints, or the two arms, when one behaves differently (e.g. a weak shoulder_lift).

Usage: uv run python -m lerobot.felix_jr.diagnostics.dump_servo_config [--left PORT] [--right PORT]
"""

import argparse

from lerobot.felix_jr.home import make_follower

REGISTERS = [
    "Firmware_Minor_Version",
    "Operating_Mode",
    "P_Coefficient",
    "I_Coefficient",
    "D_Coefficient",
    "Max_Torque_Limit",
    "Torque_Limit",
    "Protection_Current",
    "Protective_Torque",
    "Protection_Time",
    "Overload_Torque",
    "Over_Current_Protection_Time",
    "Minimum_Startup_Force",
    "CW_Dead_Zone",
    "CCW_Dead_Zone",
    "Acceleration",
    "Goal_Velocity",
    "Maximum_Velocity_Limit",
    "Maximum_Acceleration",
    "Min_Voltage_Limit",
    "Max_Voltage_Limit",
    "Max_Temperature_Limit",
]

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--left", metavar="PORT")
parser.add_argument("--right", metavar="PORT")
args = parser.parse_args()
arms = {"left_arm": args.left, "right_arm": args.right}
arms = {robot_id: port for robot_id, port in arms.items() if port}
if not arms:
    parser.error("supply at least one of --left PORT or --right PORT")

table: dict[str, dict[str, list]] = {}  # arm -> register -> value per motor
motors: list[str] = []
for robot_id, port in arms.items():
    robot = make_follower(port, robot_id, with_base=False, with_head=False)
    robot.bus.connect()
    try:
        motors = list(robot.bus.motors)
        table[robot_id] = {
            reg: [int(v) for v in robot.bus.sync_read(reg, normalize=False).values()] for reg in REGISTERS
        }
    finally:
        robot.bus.disconnect(disable_torque=False)  # read-only: leave torque as it was

header = "".join(f"{m[:13]:>14}" for m in motors)
for robot_id, regs in table.items():
    print(f"\n{robot_id}\n{'':32}{header}")
    for reg, values in regs.items():
        print(f"{reg:32}" + "".join(f"{v:>14}" for v in values))
