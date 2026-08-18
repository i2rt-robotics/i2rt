"""Tests for the Flow Base startup control-mode and scaling check.

Everything here is hardware-free. The bus-driving routine is exercised against a fake *register file* --
a dict per motor, one level above the wire -- rather than a mocked ``can`` bus, so the tests pin this
module's decisions and not python-can's behaviour.

``FakeMotors`` enforces the module's central safety rule itself: a save must directly follow that same
register's write **and** its verifying read. It signals a violation with ``ProtocolViolation``, which is
deliberately *not* one of ``BUS_ERRORS`` -- an ``AssertionError`` would be caught by the code under test
and quietly logged as a bus failure, so the test would pass for the wrong reason.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from i2rt.flow_base import motor_config_check as mcc
from i2rt.motor_config_tool.dm_motor_registers import (
    REG_BY_ADDR,
    DMRegAddr,
    Scalar,
    _data_to_value,
    _value_to_bytes,
)

BASE_MOTOR_LIST = [
    [1, "DM4310V"],
    [2, "DM_FLOW_WHEEL"],
    [3, "DM4310V"],
    [4, "DM_FLOW_WHEEL"],
    [5, "DM4310V"],
    [6, "DM_FLOW_WHEEL"],
    [7, "DM4310V"],
    [8, "DM_FLOW_WHEEL"],
]

CHANNEL = "can_test"
CTRL_MODE = int(DMRegAddr.CTRL_MODE)


def as_stored(reg: DMRegAddr, value: float) -> Scalar:
    """What a motor really answers for a float register: the value after a float32 round trip.

    ``PMAX`` 3.1415926 comes back as 3.141592502593994. Storing the float64 instead would let every
    healthy-base test pass without ever exercising ``_REL_TOL``, which is the only reason the scaling
    comparison is not exact.
    """
    spec = REG_BY_ADDR[int(reg)]
    return _data_to_value(spec, bytearray(4) + _value_to_bytes(spec, value))  # value lives in data[4:8]


def healthy_base(motor_list: list[list[Any]] = BASE_MOTOR_LIST) -> dict[int, dict[int, Scalar]]:
    """Every motor answering identity registers, speed mode, and the scaling its own type demands.

    The scaling comes from ``scaling_for`` rather than being written down again here; the numbers it
    produces are pinned separately by ``test_the_scaling_expectation_comes_from_the_driver_constants``.
    """
    return {
        int(motor_id): {
            int(DMRegAddr.SW_VER): 925970741,
            int(DMRegAddr.GR): 10,
            CTRL_MODE: mcc.CTRL_MODE_SPEED,
            **{int(reg): as_stored(reg, want) for reg, want in mcc.scaling_for(str(motor_type)).items()},
        }
        for motor_id, motor_type in motor_list
    }


class ProtocolViolation(Exception):
    """Raised when the code under test breaks the write -> verify -> save contract."""


class FakeBus:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeMotors:
    """A dict-backed register file per motor id."""

    def __init__(
        self,
        registers: dict[int, dict[int, Scalar]],
        silent: tuple[int, ...] = (),
        reject: tuple[int, ...] = (),
    ) -> None:
        self.registers = registers
        self.silent = set(silent)
        self.reject = set(reject)  # motor ids that accept a write but do not store it
        self.calls: list[tuple[str, int, int]] = []
        self.bus = FakeBus()

    def read(self, iface: Any, motor_id: int, reg: DMRegAddr) -> Scalar:
        addr = int(reg)
        self.calls.append(("read", motor_id, addr))
        if motor_id in self.silent:
            raise RuntimeError(f"Failed CAN exchange for motor {motor_id}")
        return self.registers[motor_id][addr]

    def write(self, iface: Any, motor_id: int, reg: DMRegAddr, value: Scalar) -> Scalar:
        addr = int(reg)
        self.calls.append(("write", motor_id, addr))
        if motor_id not in self.reject:
            self.registers[motor_id][addr] = value
        return value

    def save(self, iface: Any, motor_id: int, reg: DMRegAddr) -> object:
        addr = int(reg)
        if self.calls[-2:] != [("write", motor_id, addr), ("read", motor_id, addr)]:
            raise ProtocolViolation(
                f"save of {REG_BY_ADDR[addr].name} on motor {motor_id} did not directly follow that "
                f"register's own write and verifying read; last calls were {self.calls[-3:]}"
            )
        self.calls.append(("save", motor_id, addr))
        return object()

    def calls_for(self, motor_id: int) -> list[tuple[str, int, int]]:
        return [call for call in self.calls if call[1] == motor_id]

    def writes(self) -> list[tuple[str, int, int]]:
        return [call for call in self.calls if call[0] == "write"]

    def saves(self) -> list[tuple[str, int, int]]:
        return [call for call in self.calls if call[0] == "save"]


@pytest.fixture
def bank(monkeypatch: pytest.MonkeyPatch) -> Any:
    def _install(
        registers: dict[int, dict[int, Scalar]],
        silent: tuple[int, ...] = (),
        reject: tuple[int, ...] = (),
    ) -> FakeMotors:
        fake = FakeMotors(registers, silent=silent, reject=reject)
        monkeypatch.setattr(mcc, "RawCanInterface", lambda **kwargs: fake.bus)
        monkeypatch.setattr(mcc, "read_register", fake.read)
        monkeypatch.setattr(mcc, "write_register", fake.write)
        monkeypatch.setattr(mcc, "save_register_to_flash", fake.save)
        return fake

    return _install


# --------------------------------------------------------------------------------------------------
# The expectation itself
# --------------------------------------------------------------------------------------------------


def test_speed_mode_is_three_and_compares_exactly() -> None:
    """CTRL_MODE is a uint32, so there is no tolerance and no float32 round-trip to absorb."""
    assert mcc.CTRL_MODE_SPEED == 3
    assert mcc.is_speed_mode(3)
    for other in (0, 1, 2, 4):
        assert not mcc.is_speed_mode(other)


def test_the_expected_value_is_writable_and_encodable() -> None:
    """Catches a ``CTRL_MODE_SPEED = 3.0`` typo: a uint32 register rejects a float outright."""
    spec = REG_BY_ADDR[CTRL_MODE]
    assert spec.rw, "CTRL_MODE is read-only, so it could never be repaired"
    _value_to_bytes(spec, mcc.CTRL_MODE_SPEED)


def test_describe_names_both_modes() -> None:
    described = mcc.describe(CHANNEL, 7, DMRegAddr.CTRL_MODE, 1, mcc.CTRL_MODE_SPEED)
    assert "MIT" in described and "speed" in described
    assert "motor 7" in described and CHANNEL in described


def test_describe_explains_an_unwritten_flash_cell() -> None:
    """A float register that never held a value decodes to NaN, which is a reading, not a bus fault."""
    described = mcc.describe(CHANNEL, 2, DMRegAddr.PMAX, float("nan"), math.pi)
    assert "never written" in described and "NaN" in described


def test_the_scaling_expectation_comes_from_the_driver_constants() -> None:
    """Both Flow Base motor types decode through pi / 30 / 10 -- the settled value, see the README."""
    for motor_type in ("DM4310V", "DM_FLOW_WHEEL"):
        assert mcc.scaling_for(motor_type) == {
            DMRegAddr.PMAX: pytest.approx(math.pi, abs=1e-6),
            DMRegAddr.VMAX: 30.0,
            DMRegAddr.TMAX: 10.0,
        }
    # The entry the 12.5 / 45 confusion came from. It is a different motor, and neither of ours.
    assert mcc.scaling_for("DMH6215MIT")[DMRegAddr.VMAX] == 45.0


def test_only_the_registers_the_base_steers_and_navigates_on_can_block() -> None:
    """All three are read and compared; only two of them are a reason not to move the base.

    PMAX decodes the wheel angle the kinematics are rebuilt from and VMAX the rate the odometry, the
    caster-flip brake and both detectors act on. TMAX decodes MotorInfo.eff, which only leaves the
    process -- so it is still read, still compared and still reported, but never blocking.
    """
    assert set(mcc.LOOP_SCALING_REGISTERS) == {DMRegAddr.PMAX, DMRegAddr.VMAX}
    assert set(mcc.LOOP_SCALING_REGISTERS) < set(mcc.SCALING_REGISTERS)
    assert DMRegAddr.TMAX in mcc.SCALING_REGISTERS, "still read and reported on every motor"


def test_matches_absorbs_the_float32_round_trip_but_not_a_typo() -> None:
    """Why the scaling comparison is not exact, and why the tolerance is still tight enough to matter."""
    spec = REG_BY_ADDR[int(DMRegAddr.PMAX)]
    stored = as_stored(DMRegAddr.PMAX, math.pi)
    assert stored != math.pi, "the round trip must actually lose precision, or this proves nothing"
    assert mcc.matches(spec, stored, math.pi)
    assert not mcc.matches(spec, 3.1416, math.pi), "a hand-typed constant is 2.4e-6 off and must fail"
    assert not mcc.matches(spec, 12.5, math.pi)
    assert not mcc.matches(spec, float("nan"), math.pi), "an unwritten Flash cell is a mismatch"


# --------------------------------------------------------------------------------------------------
# Against the fake register file
# --------------------------------------------------------------------------------------------------


def test_a_healthy_base_writes_nothing(bank: Any) -> None:
    """The idempotence property: a correctly configured base must never be touched."""
    fake = bank(healthy_base())
    mcc.verify_base_motor_registers(CHANNEL, BASE_MOTOR_LIST)
    assert fake.writes() == []
    assert fake.saves() == []
    assert fake.bus.closed


def test_every_motor_is_checked_not_just_the_steering_ones(bank: Any) -> None:
    fake = bank(healthy_base())
    mcc.verify_base_motor_registers(CHANNEL, BASE_MOTOR_LIST)
    read_ids = {call[1] for call in fake.calls if call[0] == "read" and call[2] == CTRL_MODE}
    assert read_ids == {1, 2, 3, 4, 5, 6, 7, 8}


@pytest.mark.parametrize("motor_id", [1, 2, 8])
def test_a_wrong_mode_is_written_verified_and_saved(
    bank: Any, caplog: pytest.LogCaptureFixture, motor_id: int
) -> None:
    """Drive motors are repaired exactly like steering motors."""
    registers = healthy_base()
    registers[motor_id][CTRL_MODE] = 1
    fake = bank(registers)

    with caplog.at_level("WARNING"):
        mcc.verify_base_motor_registers(CHANNEL, BASE_MOTOR_LIST)

    assert fake.writes() == [("write", motor_id, CTRL_MODE)]
    assert fake.saves() == [("save", motor_id, CTRL_MODE)]
    assert registers[motor_id][CTRL_MODE] == mcc.CTRL_MODE_SPEED
    assert "saved to Flash" in caplog.text


def test_save_directly_follows_that_motors_own_verified_write(bank: Any) -> None:
    """Two wrong motors must not be batched into write, write, save, save.

    ``FakeMotors.save`` raises ``ProtocolViolation`` if the ordering is ever broken; asserting the exact
    call sequence here documents what the contract is rather than only that it held.
    """
    registers = healthy_base()
    registers[1][CTRL_MODE] = 1
    registers[4][CTRL_MODE] = 2
    fake = bank(registers)

    mcc.verify_base_motor_registers(CHANNEL, BASE_MOTOR_LIST)

    assert [call for call in fake.calls if call[0] in ("write", "save")] == [
        ("write", 1, CTRL_MODE),
        ("save", 1, CTRL_MODE),
        ("write", 4, CTRL_MODE),
        ("save", 4, CTRL_MODE),
    ]


def test_a_write_that_does_not_take_aborts_and_saves_nothing(bank: Any) -> None:
    """A motor that accepts a write without holding it is one we do not understand."""
    registers = healthy_base()
    registers[5][CTRL_MODE] = 1
    fake = bank(registers, reject=(5,))

    with pytest.raises(RuntimeError, match="would not take"):
        mcc.verify_base_motor_registers(CHANNEL, BASE_MOTOR_LIST)

    assert fake.saves() == []
    assert fake.bus.closed


def test_a_silent_motor_is_probed_once_and_suppresses_every_write(bank: Any) -> None:
    """One probe, not three: a dead register costs ~0.65 s of retries inside _tx_rx."""
    registers = healthy_base()
    registers[1][CTRL_MODE] = 1  # a repair that must NOT happen
    fake = bank(registers, silent=(7,))

    with pytest.raises(RuntimeError, match="did not answer"):
        mcc.verify_base_motor_registers(CHANNEL, BASE_MOTOR_LIST)

    assert fake.calls_for(7) == [("read", 7, int(DMRegAddr.SW_VER))]
    assert fake.writes() == []
    assert fake.saves() == []
    assert fake.bus.closed


def test_repairing_warns_about_a_power_cycle(bank: Any, caplog: pytest.LogCaptureFixture) -> None:
    """A 0x55 write is not established to re-latch the control mode without a reboot."""
    registers = healthy_base()
    registers[1][CTRL_MODE] = 1
    bank(registers)

    with caplog.at_level("WARNING"):
        mcc.verify_base_motor_registers(CHANNEL, BASE_MOTOR_LIST)

    assert "power-cycle motor 1" in caplog.text
    assert "E stop is not the problem" in caplog.text


def test_an_unopenable_bus_is_reported_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(**kwargs: Any) -> None:
        raise OSError("no such device")

    monkeypatch.setattr(mcc, "RawCanInterface", explode)
    with pytest.raises(RuntimeError, match="could not open"):
        mcc.verify_base_motor_registers(CHANNEL, BASE_MOTOR_LIST)


def test_the_rail_motor_is_checked_too(bank: Any) -> None:
    """LinearRailVehicle passes a 9-motor list; the 9th is the lift and is not exempt."""
    motor_list = [*BASE_MOTOR_LIST, [9, "DM8009"]]
    registers = healthy_base(motor_list)
    registers[9][CTRL_MODE] = 1
    fake = bank(registers)

    mcc.verify_base_motor_registers(CHANNEL, motor_list)
    assert fake.writes() == [("write", 9, CTRL_MODE)]


# --------------------------------------------------------------------------------------------------
# Scaling: read on every motor, written on none
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("motor_id", [1, 7])
def test_a_mis_scaled_steering_motor_refuses_to_start(
    bank: Any, caplog: pytest.LogCaptureFixture, motor_id: int
) -> None:
    """A steering motor's PMAX is the scale the swerve kinematics read the wheel angle through."""
    registers = healthy_base()
    registers[motor_id][int(DMRegAddr.PMAX)] = as_stored(DMRegAddr.PMAX, 12.5)
    fake = bank(registers)

    with caplog.at_level("ERROR"), pytest.raises(RuntimeError, match="mis-scaled") as raised:
        mcc.verify_base_motor_registers(CHANNEL, BASE_MOTOR_LIST)

    assert f"steering motor(s) [{motor_id}]" in str(raised.value)
    assert "NOT started" in str(raised.value)
    assert "swerve kinematics" in caplog.text
    assert fake.writes() == [], "a bus whose scaling we do not believe is not one to write Flash on"
    assert fake.bus.closed


@pytest.mark.parametrize("motor_id", [2, 8])
def test_a_mis_scaled_drive_motor_is_reported_and_the_base_still_starts(
    bank: Any, caplog: pytest.LogCaptureFixture, motor_id: int
) -> None:
    """A drive motor's scale is only on the way out of the loop -- it mis-scales odometry, not steering."""
    registers = healthy_base()
    registers[motor_id][int(DMRegAddr.VMAX)] = as_stored(DMRegAddr.VMAX, 45.0)
    fake = bank(registers)

    with caplog.at_level("ERROR"):
        mcc.verify_base_motor_registers(CHANNEL, BASE_MOTOR_LIST)  # must not raise

    assert "translational odometry" in caplog.text
    assert "DMH6215MIT" in caplog.text, "name the entry the 12.5 / 45 confusion came from"
    assert fake.writes() == []


@pytest.mark.parametrize("motor_id", [1, 7])
def test_a_mis_scaled_tmax_on_a_steering_motor_reports_but_does_not_block(
    bank: Any, caplog: pytest.LogCaptureFixture, motor_id: int
) -> None:
    """TMAX decodes MotorInfo.eff, and eff leaves the process without ever being read back.

    Its two readers are get_wheel_states and the linear rail state; nothing in the control loop, the
    swerve kinematics, the odometry or the caster fault check consumes a torque, and the MIT branch that
    would use TMAX to *encode* one is unreachable on a ControlMode.VEL chain. So this is reported like
    any other mismatch and the launch continues -- proven by letting a genuine CTRL_MODE repair run on
    another motor in the same pass, which the mis-scaled gate (it raises before any repair) would
    otherwise have pre-empted.
    """
    registers = healthy_base()
    registers[motor_id][int(DMRegAddr.TMAX)] = as_stored(DMRegAddr.TMAX, 54.0)
    registers[2][CTRL_MODE] = 1
    fake = bank(registers)

    with caplog.at_level("ERROR"):
        mcc.verify_base_motor_registers(CHANNEL, BASE_MOTOR_LIST)  # must not raise

    assert "reported torque" in caplog.text
    assert "MotorInfo.eff" in caplog.text
    assert "DMH6215MIT" in caplog.text, "still gets the full fix-it-by-hand line"
    # Not "swerve kinematics": the TMAX clause names it too, in the list of what does *not* read a
    # torque. The PMAX consequence is the one thing that must not be said about this register.
    assert "wheels point somewhere they do not" not in caplog.text, "TMAX is not the wheel-angle scale"
    assert "NOT starting" not in caplog.text
    assert fake.writes() == [("write", 2, CTRL_MODE)], "the run got past the scaling gate"


@pytest.mark.parametrize("motor_id", [1, 7])
def test_a_mis_scaled_vmax_on_a_steering_motor_still_refuses_to_start(
    bank: Any, caplog: pytest.LogCaptureFixture, motor_id: int
) -> None:
    """Narrowing the block to the loop registers must not narrow it to PMAX alone.

    Steering velocity is dq: the odometry integrates it, the caster-flip brake trips on it, and it is
    the measured half of both the rate and the runaway detector.
    """
    registers = healthy_base()
    registers[motor_id][int(DMRegAddr.VMAX)] = as_stored(DMRegAddr.VMAX, 45.0)
    fake = bank(registers)

    with caplog.at_level("ERROR"), pytest.raises(RuntimeError, match="mis-scaled") as raised:
        mcc.verify_base_motor_registers(CHANNEL, BASE_MOTOR_LIST)

    assert f"steering motor(s) [{motor_id}]" in str(raised.value)
    assert "NOT started" in str(raised.value)
    assert "caster-flip brake" in caplog.text
    assert "wheels point somewhere they do not" not in caplog.text, "the angle is not what is mis-scaled"
    assert fake.writes() == []


def test_one_bad_loop_register_is_enough_to_block(bank: Any, caplog: pytest.LogCaptureFixture) -> None:
    """``any`` over the bad registers: a harmless TMAX alongside a fatal PMAX must not dilute it."""
    registers = healthy_base()
    registers[3][int(DMRegAddr.TMAX)] = as_stored(DMRegAddr.TMAX, 54.0)
    registers[3][int(DMRegAddr.PMAX)] = as_stored(DMRegAddr.PMAX, 12.5)
    bank(registers)

    with caplog.at_level("ERROR"), pytest.raises(RuntimeError, match="mis-scaled"):
        mcc.verify_base_motor_registers(CHANNEL, BASE_MOTOR_LIST)

    assert "swerve kinematics" in caplog.text, "the PMAX line keeps its own consequence"
    assert "reported torque" in caplog.text, "and the TMAX line keeps its own, on the same motor"


def test_the_scaling_registers_are_never_written(bank: Any) -> None:
    """The load-bearing property: their correct value describes the motor, so a guess must not reach Flash.

    Set up the one case that would most tempt a repair -- a mis-scaled drive motor next to a genuine
    CTRL_MODE repair on another motor, so the write path is demonstrably live in the same run.
    """
    registers = healthy_base()
    registers[4][int(DMRegAddr.PMAX)] = as_stored(DMRegAddr.PMAX, 12.5)
    registers[6][int(DMRegAddr.TMAX)] = as_stored(DMRegAddr.TMAX, 54.0)
    registers[3][CTRL_MODE] = 1
    fake = bank(registers)

    mcc.verify_base_motor_registers(CHANNEL, BASE_MOTOR_LIST)

    assert fake.writes() == [("write", 3, CTRL_MODE)], "only CTRL_MODE is ever written"
    assert fake.saves() == [("save", 3, CTRL_MODE)]
    touched = {call[2] for call in fake.calls if call[0] in ("write", "save")}
    assert touched.isdisjoint({int(reg) for reg in mcc.SCALING_REGISTERS})


def test_an_unwritten_scaling_cell_is_reported_as_such(bank: Any, caplog: pytest.LogCaptureFixture) -> None:
    """An erased Flash cell reads 0xFFFFFFFF, which decodes to NaN. It is a mismatch, not a bus error."""
    registers = healthy_base()
    registers[2][int(DMRegAddr.PMAX)] = float("nan")
    bank(registers)

    with caplog.at_level("ERROR"):
        mcc.verify_base_motor_registers(CHANNEL, BASE_MOTOR_LIST)

    assert "never written" in caplog.text


def test_every_motor_has_its_scaling_read_including_the_rail(bank: Any) -> None:
    motor_list = [*BASE_MOTOR_LIST, [9, "DM8009"]]
    fake = bank(healthy_base(motor_list))

    mcc.verify_base_motor_registers(CHANNEL, motor_list)

    for reg in mcc.SCALING_REGISTERS:
        read_ids = {call[1] for call in fake.calls if call[0] == "read" and call[2] == int(reg)}
        assert read_ids == {1, 2, 3, 4, 5, 6, 7, 8, 9}, f"{REG_BY_ADDR[int(reg)].name} was not read everywhere"


# --------------------------------------------------------------------------------------------------
# The controller's side of the wiring
# --------------------------------------------------------------------------------------------------


def test_disabling_the_check_skips_it_and_says_so(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from i2rt.flow_base import flow_base_controller

    def explode(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the check must not run, let alone touch the bus, when it is disabled")

    monkeypatch.setattr(flow_base_controller, "verify_base_motor_registers", explode)
    with caplog.at_level("WARNING"):
        flow_base_controller._check_motor_config(CHANNEL, BASE_MOTOR_LIST, enabled=False)
    assert "DISABLED" in caplog.text
