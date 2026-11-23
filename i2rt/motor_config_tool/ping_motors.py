import tyro

from i2rt.motor_drivers.dm_driver import ControlMode, DMSingleMotorCanInterface, MotorType, ReceiveMode


def main(channel: str = "can0", motor_id: int = -1, receive_mode: ReceiveMode = ReceiveMode.p16) -> None:
    """Enable and disable each motor in turn, and report which ones answered.

    Args:
        channel: CAN channel, or ``ecat:<nic>`` for motors behind an EtherCAT-to-CAN board.
        motor_id: The one motor to ping; -1 pings ids 1-7.
        receive_mode: The id the motors reply on: ``p16`` (MASTER_ID = ID + 16, the i2rt default) or
            ``same`` (MASTER_ID == ID). Ignored on ``ecat:<nic>``, which always uses ``same``.
    """
    motor_control_interface = DMSingleMotorCanInterface(
        channel=channel, bustype="socketcan", control_mode=ControlMode.MIT, receive_mode=receive_mode
    )
    motor_ids = [1, 2, 3, 4, 5, 6, 7] if motor_id < 0 else [motor_id]

    online_motors = []
    for mid in motor_ids:
        try:
            motor_info = motor_control_interface.motor_on(mid, MotorType.DM4310)
            print(f"motor {mid} info: {motor_info}")
            online_motors.append(mid)
            motor_control_interface.motor_off(mid)
        except Exception as e:
            print(f"motor {mid} error: {e}")

    motor_control_interface.close()

    print(f"online motors: {online_motors}")


if __name__ == "__main__":
    tyro.cli(main)
