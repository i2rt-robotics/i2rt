"""The DM MIT codec shared by the CAN chain and the EtherCAT chain.

The golden frames were captured from ``DMSingleMotorCanInterface.set_control`` before the codec was
pulled out of it, so they pin the extraction to the bytes that were already going on the wire.
"""

import pytest

from i2rt.motor_drivers.dm_driver import ControlMode, DMSingleMotorCanInterface
from i2rt.motor_drivers.utils import MotorType, pack_mit, unpack_mit_feedback

GOLDEN = [
    ((MotorType.DM4310, 0, 0, 0, 0, 0), "7fff7ff0000007ff"),
    ((MotorType.DM4310, 1.0, -2.0, 20.0, 5.0, 3.0), "8a3c7770a3fffa65"),
    ((MotorType.DM6248, -0.5, 1.5, 0.0, 0.625, -40.0), "7ae08990001ff555"),
    ((MotorType.DM4340, 12.5, -10.0, 500.0, 0.0, 28.0), "ffff000fff000fff"),
    ((MotorType.DM4340, 99.0, 99.0, 999.0, 9.0, -99.0), "fffffffffffff000"),  # every field clamps
]


@pytest.mark.parametrize("args, frame", GOLDEN)
def test_pack_mit_reproduces_the_frames_already_on_the_wire(args: tuple, frame: str) -> None:
    motor_type, *values = args
    assert pack_mit(*values, MotorType.get_motor_constants(motor_type)).hex() == frame


class _Sent(Exception):
    pass


class _EncodeOnly:
    control_mode = ControlMode.MIT
    cmd_idoffset = 0

    def _get_frame_id(self, motor_id: int) -> int:
        return motor_id

    def _send_message_get_response(self, frame_id: int, motor_id: int, data: bytes, max_retry: int = 0) -> None:
        raise _Sent(bytes(data).hex())


@pytest.mark.parametrize("args, frame", GOLDEN)
def test_set_control_still_sends_the_golden_frame(args: tuple, frame: str) -> None:
    with pytest.raises(_Sent, match=frame):
        DMSingleMotorCanInterface.set_control(_EncodeOnly(), 1, *args)  # type: ignore[arg-type]


def test_kd_uses_the_whole_12_bit_field() -> None:
    """kd 5.0 is 0xFFF and 0.625 is 511. Packing kd on 9 bits into this 12-bit field instead would
    deliver ~1/8 of every commanded kd -- the defect this codec exists to rule out."""
    const = MotorType.get_motor_constants(MotorType.DM4310)

    def kd_field(kd: float) -> int:
        frame = pack_mit(0, 0, 0, kd, 0, const)
        return (frame[5] << 4) | (frame[6] >> 4)

    assert kd_field(5.0) == 0xFFF
    assert kd_field(0.625) == 511


def test_unpack_decodes_a_hand_built_feedback_frame() -> None:
    const = MotorType.get_motor_constants(MotorType.DM4340)
    # state 1 (enabled), pos code 0x8000, vel code 0x800, torque code 0x800, MOS 36 C, rotor 41 C
    fb = unpack_mit_feedback(bytes([0x11, 0x80, 0x00, 0x80, 0x08, 0x00, 36, 41]), const)
    assert fb.error == 0x1
    assert fb.position == pytest.approx(0.0, abs=12.5 / 32767)
    assert fb.velocity == pytest.approx(0.0, abs=10.0 / 2047)
    assert fb.torque == pytest.approx(0.0, abs=28.0 / 2047)
    assert (fb.temperature_mos, fb.temperature_rotor) == (36.0, 41.0)


@pytest.mark.parametrize("motor_type", [MotorType.DM4310, MotorType.DM4340, MotorType.DM6248])
def test_a_command_decodes_back_within_one_step(motor_type: str) -> None:
    """Command and feedback share the pos/vel/torque layout in bytes 1-5 shifted by one, so decoding
    a command's own fields through the feedback path is a round trip of the quantisation."""
    const = MotorType.get_motor_constants(motor_type)
    pos, vel, tau = 1.234, -0.75 * const.VELOCITY_MAX, 0.3 * const.TORQUE_MAX
    cmd = pack_mit(pos, vel, 0.0, 0.0, tau, const)
    as_feedback = bytes([0x10, cmd[0], cmd[1], cmd[2], (cmd[3] & 0xF0), 0, 0, 0])
    fb = unpack_mit_feedback(as_feedback, const)
    assert fb.position == pytest.approx(pos, abs=(const.POSITION_MAX - const.POSITION_MIN) / 65535)
    assert fb.velocity == pytest.approx(vel, abs=(const.VELOCITY_MAX - const.VELOCITY_MIN) / 4095)
    # torque in a command sits in bytes 6-7, in feedback in bytes 4-5: check it via its own field
    t_code = ((cmd[6] & 0xF) << 8) | cmd[7]
    t_back = t_code * (const.TORQUE_MAX - const.TORQUE_MIN) / 4095 + const.TORQUE_MIN
    assert t_back == pytest.approx(tau, abs=(const.TORQUE_MAX - const.TORQUE_MIN) / 4095)
