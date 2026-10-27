import time

import tyro

from i2rt.motor_config_tool.utils import (
    RawCanInterface,
    get_special_message_response,
    save_to_memory,
    write_special_message,
)
from i2rt.motor_drivers.dm_driver import ControlMode, DMSingleMotorCanInterface, ReceiveMode


def main(
    channel: str = "can0", timeout: bool = False, motor_id: int = -1, receive_mode: ReceiveMode = ReceiveMode.p16
) -> None:
    """Set (or remove) each motor's CAN timeout register and save it to Flash.

    Args:
        channel: CAN channel, or ``ecat:<nic>`` for motors behind an EtherCAT-to-CAN board.
        timeout: Set the timeout to 8000; without it the timeout is removed (0).
        motor_id: The one motor to configure; -1 configures ids 1-7.
        receive_mode: The id the motors reply on: ``p16`` (MASTER_ID = ID + 16, the i2rt default) or
            ``same`` (MASTER_ID == ID). Ignored on ``ecat:<nic>``, which always uses ``same``.
    """
    can_interface = RawCanInterface(
        channel=channel,
        bustype="socketcan",
    )

    timeout_value = 8000 if timeout else 0
    motor_ids = [1, 2, 3, 4, 5, 6, 7] if motor_id < 0 else [motor_id]

    for mid in motor_ids:
        control_interface = DMSingleMotorCanInterface(
            channel=channel, bustype="socketcan", control_mode=ControlMode.MIT, receive_mode=receive_mode
        )
        control_interface.motor_off(mid)
        print("#" * 30)
        print(f"processing motor {mid}, before removing timeout")
        for reg_name in ["id", "master_id", "timeout"]:
            info = get_special_message_response(can_interface, mid, reg_name)
            print(f"current setting: {reg_name} = {info}")
            time.sleep(0.1)

        write_special_message(can_interface, mid, "timeout", timeout_value)
        print(save_to_memory(can_interface, mid, "timeout"))

        print("after removing timeout")
        for reg_name in ["id", "master_id", "timeout"]:
            info = get_special_message_response(can_interface, mid, reg_name)
            print(f"current setting: {reg_name} = {info}")
            time.sleep(0.1)
        for _ in range(4):
            can_interface.try_receive_message(mid)
        control_interface.close()


if __name__ == "__main__":
    tyro.cli(main)
