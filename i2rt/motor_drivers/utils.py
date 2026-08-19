import enum
from dataclasses import dataclass
from typing import List


def uint_to_float(x_int: int, x_min: float, x_max: float, bits: int) -> float:
    """Converts unsigned int to float, given range and number of bits."""
    span = x_max - x_min
    offset = x_min
    return (x_int * span / ((1 << bits) - 1)) + offset


def float_to_uint(x: float, x_min: float, x_max: float, bits: int) -> int:
    """Converts a float to an unsigned int, given range and number of bits."""
    span = x_max - x_min
    offset = x_min
    x = min(x, x_max)
    x = max(x, x_min)
    return int((x - offset) * ((1 << bits) - 1) / span)


@dataclass
class MotorConstants:
    POSITION_MAX: float = 12.5
    POSITION_MIN: float = -12.5

    VELOCITY_MAX: float = 45
    VELOCITY_MIN: float = -45

    TORQUE_MAX: float = 54
    TORQUE_MIN: float = -54

    ####### Mihgt be used for other motors #######
    CURRENT_MAX: float = 1.0
    CURRENT_MIN: float = -1.0
    KT: float = 1.0
    ##############################

    KP_MAX: float = 500.0
    KP_MIN: float = 0.0
    KD_MAX: float = 5.0
    KD_MIN: float = 0.0


@dataclass
class MotorInfo:
    """Class to represent motor information.

    Attributes:
        id (int): Motor ID.
        target_torque (int): Target torque value.
        vel (float): Motor speed.
        eff (float): Motor current.
        pos (float): Encoder value.
        voltage (float): Motor voltage.
        temperature (float): Motor temperature.

    """

    id: int
    error_code: int
    target_torque: int = 0
    vel: float = 0.0
    eff: float = 0
    pos: float = 0
    voltage: float = -1
    temp_mos: float = -1
    temp_rotor: float = -1
    timestamp: float = 0.0


@dataclass
class FeedbackFrameInfo:
    id: int
    error_code: int
    error_message: str
    position: float
    velocity: float
    torque: float
    temperature_mos: float
    temperature_rotor: float


@dataclass
class EncoderInfo:
    encoder = -1
    encoder_raw = -1
    encoder_offset = -1


class MotorErrorCode:
    disabled = 0x0
    normal = 0x1
    over_voltage = 0x8
    under_voltage = 0x9
    over_current = 0xA
    mosfet_over_temperature = 0xB
    motor_over_temperature = 0xC
    loss_communication = 0xD
    overload = 0xE

    # create a dict map error code to error message
    motor_error_code_dict = {
        normal: "normal",
        disabled: "disabled",
        over_voltage: "over voltage",
        under_voltage: "under voltage",
        over_current: "over current",
        mosfet_over_temperature: "mosfet over temperature",
        motor_over_temperature: "motor over temperature",
        loss_communication: "loss communication",
        overload: "overload",
    }
    # covert to decimal
    motor_error_code_dict = {int(k): v for k, v in motor_error_code_dict.items()}

    @classmethod
    def get_error_message(cls, error_code: int) -> str:
        return cls.motor_error_code_dict.get(int(error_code), f"Unknown error code: {error_code}")


class MotorType:
    DM8009 = "DM8009"
    DM4310 = "DM4310"
    DM4310V = "DM4310V"
    DM4340 = "DM4340"
    DM6248 = "DM6248"
    DMH6215 = "DMH6215"
    DMH6215MIT = "DMH6215MIT"
    DM3507 = "DM3507"
    DM_FLOW_WHEEL = "DM_FLOW_WHEEL"

    @classmethod
    def get_motor_constants(cls, motor_type: str) -> MotorConstants:
        if motor_type == cls.DM8009:
            return MotorConstants(
                POSITION_MAX=12.5,
                POSITION_MIN=-12.5,
                VELOCITY_MAX=45,
                VELOCITY_MIN=-45,
                TORQUE_MAX=54,
                TORQUE_MIN=-54,
            )
        elif motor_type == cls.DM4310:
            return MotorConstants(
                POSITION_MAX=12.5,
                POSITION_MIN=-12.5,
                VELOCITY_MAX=30,
                VELOCITY_MIN=-30,
                TORQUE_MAX=10,
                TORQUE_MIN=-10,
                # max kp 500
                # max kd 5
            )
        elif motor_type in [cls.DM4310V, cls.DM_FLOW_WHEEL, cls.DMH6215]:
            return MotorConstants(
                POSITION_MAX=3.1415926,
                POSITION_MIN=-3.1415926,
                VELOCITY_MAX=30,
                VELOCITY_MIN=-30,
                TORQUE_MAX=10,
                TORQUE_MIN=-10,
            )
        elif motor_type == cls.DM4340:
            return MotorConstants(
                POSITION_MAX=12.5,
                POSITION_MIN=-12.5,
                VELOCITY_MAX=10,
                VELOCITY_MIN=-10,
                TORQUE_MAX=28,
                TORQUE_MIN=-28,
                # max kp 500
                # max kd 5
            )
        elif motor_type == cls.DM6248:
            return MotorConstants(
                POSITION_MAX=12.5,
                POSITION_MIN=-12.5,
                VELOCITY_MAX=20,
                VELOCITY_MIN=-20,
                TORQUE_MAX=120,
                TORQUE_MIN=-120,
                # max kp 500
                # max kd 5
            )
        elif motor_type == cls.DMH6215MIT:
            return MotorConstants(
                POSITION_MAX=12.5,
                POSITION_MIN=-12.5,
                VELOCITY_MAX=45,
                VELOCITY_MIN=-45,
                TORQUE_MAX=10,
                TORQUE_MIN=-10,
            )
        elif motor_type == cls.DM3507:
            return MotorConstants(
                POSITION_MAX=12.5,
                POSITION_MIN=-12.5,
                VELOCITY_MAX=50,
                VELOCITY_MIN=-50,
                TORQUE_MAX=5,
                TORQUE_MIN=-5,
            )
        else:
            raise ValueError(f"Motor type '{motor_type}' not recognized.")

    @classmethod
    def get_gear_ratio(cls, motor_type: str) -> float:
        """The gear reduction ratio this motor type's ``Gr`` register (address 20) must report.

        Raises:
            ValueError: if no ratio is recorded for this type. A missing entry is a gap in
                ``_GEAR_RATIO``, not a fault in the hardware, so it is deliberately not reported as
                "unknown" for the caller to skip over -- see ``i2rt/motor_drivers/motor_check.py``.
                Mirrors ``get_motor_constants``, which also raises on a type it does not know.
        """
        try:
            return _GEAR_RATIO[motor_type]
        except KeyError:
            raise ValueError(
                f"No gear ratio recorded for motor type '{motor_type}'. Add it to _GEAR_RATIO in "
                f"i2rt/motor_drivers/utils.py, taking the value from the motor's own Gr register "
                f"(dm_motor_registers.py read Gr --motor-id <id>) rather than from the part number. "
                f"Recorded: {sorted(_GEAR_RATIO)}"
            ) from None

    @classmethod
    def known_gear_ratios(cls) -> dict[str, float]:
        """Every motor type with a recorded gear ratio, as a copy.

        Exists for the two callers that need the whole mapping rather than one lookup: the reverse
        direction (which type does a ``Gr`` reading correspond to?), which is what makes a mismatch
        message name the motor that is actually installed, and the test that asserts every motor
        type in a shipped config has an entry here.
        """
        return dict(_GEAR_RATIO)


# Gear reduction ratio each motor type reports in its `Gr` register (address 20). Read-only in
# firmware and a property of the physical gearbox, which is what makes it the one register that
# identifies a motor: PMAX/VMAX/TMAX are writable, so they can be wrong on a correct motor *and*
# right on a wrong one. Consumed by verify_motor_types in i2rt/motor_drivers/motor_check.py, which
# refuses to build a chain on a mismatch. Arms and the Flow Base both opt into it
# (check_motor_types=True); nothing else does.
#
# The DM part number ends in its gear ratio -- 4310 -> 10, 4340 -> 40, 3507 -> 7 -- and *not* the
# tempting "8009 -> 80:1", which a bench reading of 9 is what settles. Provenance is per entry below;
# only DM4310 / DM4340 / DM3507 rest on the manual alone, and all three agree with the rule.
#
# Only the types the shipped configs and the Flow Base controller actually build are listed. A type
# that is absent raises instead of being compared against a value inferred from its name -- extending
# this table is a bench reading, not a guess.
_GEAR_RATIO: dict[str, float] = {
    MotorType.DM3507: 7.0,  # DM manual
    MotorType.DM4310: 10.0,  # DM manual
    MotorType.DM4310V: 10.0,  # bench: flow base steering motors 1/3/7, 2026-08-18
    MotorType.DM4340: 40.0,  # DM manual
    MotorType.DM6248: 48.0,  # bench: big_yam motor 2, 2026-08-18
    MotorType.DM8009: 9.0,  # bench: flow base rail motor 9, 2026-08-14 and again 2026-08-18
    # DM_FLOW_WHEEL is a *role*, not a part number: i2rt/flow_base/README.md says the physical drive
    # motor may differ per unit, and Gr being read-only means it cannot be normalised across units the
    # way PMAX/VMAX/TMAX are. Recorded and asserted anyway (decided 2026-08-18, after all four drive
    # motors on one base read 10), so a unit built with a different drive motor refuses to start rather
    # than running mis-scaled -- which is the intended way to discover that such a unit exists. The
    # kinematics make that non-negotiable: N_s/N_r1/N_r2/N_w are all 1 in flow_base_controller.py, i.e.
    # the swerve model assumes the motor's own rad/s *is* the wheel's. --no-verify-motor-config is the
    # field bypass, and widening this number is the fix.
    MotorType.DM_FLOW_WHEEL: 10.0,  # bench: flow base drive motors 2/4/6/8, 2026-08-18
}


class AutoNameEnum(enum.Enum):
    def _generate_next_value_(name: str, start: int, count: int, last_values: List[str]) -> str:
        return name


class ReceiveMode(AutoNameEnum):
    p16 = enum.auto()
    same = enum.auto()
    zero = enum.auto()
    plus_one = enum.auto()

    def get_receive_id(self, motor_id: int) -> int:
        if self == ReceiveMode.p16:
            return motor_id + 16
        elif self == ReceiveMode.same:
            return motor_id
        elif self == ReceiveMode.zero:
            return 0
        elif self == ReceiveMode.plus_one:
            return motor_id + 1
        else:
            raise NotImplementedError(f"receive_mode: {self} not recognized")

    def to_motor_id(self, receive_id: int) -> int:
        if self == ReceiveMode.p16:
            return receive_id - 16
        elif self == ReceiveMode.same:
            return receive_id
        elif self == ReceiveMode.zero:
            return 0
        else:
            raise NotImplementedError(f"receive_mode: {self} not recognized")
