"""Set a motor's current position as its zero position.

Two things the naive sequence gets wrong, both fixed here:

- **The read-back has to be retried.** The motor does not report the new zero on the very next
  frame after the Flash write, so a read issued immediately still returns the *old* position and
  the tool looks like it did nothing. (``save_zero_position``'s own check has the same problem and
  additionally passes a frame id where a motor id belongs, so it rarely prints anything.) The
  verdict is on the *shift* rather than on landing at zero, because a joint zeroed against a
  mechanical stop springs back once torque is removed.
- **The error state has to be cleared.** Zeroing is usually done *because* a motor faulted, and a
  latched error survives the write -- so the motor is left refusing to run right after a
  successful zero. Clearing it (0xFB) also disables the motor, so it is brought back up and read
  once more to confirm: ``set_control`` raises on the error bits, so a clean read means ready.
"""

import time

import tyro

from i2rt.motor_drivers.dm_driver import ControlMode, DMSingleMotorCanInterface, MotorType

ARM_MOTOR_IDS = [1, 2, 3, 4, 5, 6, 7]


def _read_position(interface: DMSingleMotorCanInterface, motor_id: int, motor_type: str) -> float:
    """Read a motor's position by sending it an all-zero MIT command."""
    return interface.set_control(motor_id, motor_type, 0, 0, 0, 0, 0).position


def main(
    motor_id: int = -1,
    channel: str = "can0",
    motor_type: str = MotorType.DM4310,
    settle: float = 0.2,
    attempts: int = 5,
    tolerance: float = 0.01,
) -> None:
    """Set a motor's current position as its zero position, then verify it and clear its error.

    Args:
        motor_id: CAN id to zero. -1 zeros the arm's motors (1-7); a gripper is its own id.
        channel: CAN channel.
        motor_type: Motor type, which sets the scale the reported position is decoded with -- a
            wrong type makes the read-back silently mis-scaled.
        settle: Seconds to wait between read-back attempts.
        attempts: Read-back attempts before reporting that the zero did not take.
        tolerance: Position shift, in rad, that counts as the zero having landed.
    """
    motor_ids = ARM_MOTOR_IDS if motor_id < 0 else [motor_id]

    for mid in motor_ids:
        interface = DMSingleMotorCanInterface(channel=channel, bustype="socketcan", control_mode=ControlMode.MIT)
        try:
            print("#" * 30)
            print(f"processing motor {mid}")
            interface.motor_on(mid, motor_type)
            before = _read_position(interface, mid, motor_type)
            print(f"current position: {before}")

            interface.save_zero_position(mid)
            print("set zero .....")

            # Judge by the *shift*, not by the position landing on zero: a joint resting against a
            # mechanical stop springs back once torque is removed, so a correctly zeroed motor can
            # legitimately read a couple of tenths off. What a failed write looks like is a
            # read-back identical to the value from before it.
            position = before
            for _ in range(attempts):
                time.sleep(settle)
                position = _read_position(interface, mid, motor_type)
                if abs(position - before) > tolerance:
                    break
            if abs(position - before) > tolerance:
                verdict = "ok"
            else:
                verdict = f"unchanged after {attempts} reads -- already at zero, or the write did not take"
            print(f"after set to zero position, current position: {position} ({verdict})")

            # 0xFB clears the error but leaves the motor disabled, so bring it back up -- otherwise
            # the next command gets "disabled" and the tool has traded one fault for another.
            interface.clean_error(mid)
            interface.motor_on(mid, motor_type)
            # A read that returns at all means the error bits are clear: set_control raises on them.
            _read_position(interface, mid, motor_type)
            print("error state cleared, motor enabled")
        except Exception as e:
            print(f"motor {mid} error: {e}")
        finally:
            interface.close()


if __name__ == "__main__":
    tyro.cli(main)
