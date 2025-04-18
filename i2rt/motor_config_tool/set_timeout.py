from utils import *
import argparse
from i2rt.motor_drivers.dm_driver import ControlMode, DMSingleMotorCanInterface
args = argparse.ArgumentParser()
args.add_argument("--channel", type=str, default="can0")
args.add_argument("--timeout", action='store_true')
args.add_argument("--motor_id", type=int, default=-1)

args = args.parse_args()
can_interface = RawCanInterface(
    channel=args.channel,
    bustype="socketcan",
)
control_interface = DMSingleMotorCanInterface(
    channel=args.channel, bustype="socketcan", control_mode=ControlMode.MIT
)
if args.timeout:
    timeout = 4000
else:
    timeout = 0

motor_id = args.motor_id
if motor_id < 0:
    motor_ids = [1, 2, 3, 4, 5, 6, 7]
else:
    motor_ids = [motor_id]

for motor_id in motor_ids:
    control_interface.motor_off(motor_id)
    print("#"*30)
    print(f"processing motor {motor_id}, before removing timeout")
    for reg_name in ["id", "master_id", "timeout"]:
        info = get_special_message_response(can_interface, motor_id, reg_name)
        print(f'current setting: {reg_name} = {info}')
        
    write_special_message(can_interface, motor_id, "timeout", timeout)
    print(save_to_memory(can_interface, motor_id, "timeout"))

    print("after removing timeout")
    for reg_name in ["id", "master_id", "timeout"]:
        info = get_special_message_response(can_interface, motor_id, reg_name)
        print(f'current setting: {reg_name} = {info}')