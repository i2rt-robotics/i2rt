"""Tests for ``motor_check.verify_motor_config`` -- the control-mode and scaling half.

``verify_motor_types``, the other half of that module, has its own file next to this one. The Flow
Base is the worked example through most of it because it is the caller with the interesting *partial*
policy -- a VEL chain naming four of its nine motors loop-critical. Arms run the same check at the other
extreme, MIT with no ids named, and get their own section at the end. The policy under test is the
general one: the control mode decides which registers can block, and the caller decides which motors
are loop-critical.

Everything here is hardware-free. The bus-driving routine is exercised against a fake *register file* --
a dict per motor, one level above the wire -- rather than a mocked ``can`` bus, so the tests pin this
module's decisions and not python-can's behaviour.

``FakeMotors`` enforces the module's central safety rule itself: a save must directly follow that same
register's write **and** its verifying read. It signals a violation with ``ProtocolViolation``, which is
deliberately *not* one of ``BUS_ERRORS`` -- an ``AssertionError`` would be caught by the code under test
and quietly logged as a bus failure, so the test would pass for the wrong reason.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import pytest
import yaml

from i2rt.motor_config_tool.dm_motor_registers import (
    REG_BY_ADDR,
    DMRegAddr,
    Scalar,
    _data_to_value,
    _value_to_bytes,
)
from i2rt.motor_drivers import motor_check as mc
from i2rt.motor_drivers.utils import MotorType
from i2rt.robots.utils import _CONFIG_DIR

# Taken from the loader's own constant so the glob cannot drift from what _load_arm_config reads.
CONFIG_DIR = Path(_CONFIG_DIR)

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
BASE_CONTROL_MODE = "VEL"
"""Every Flow Base chain is built with ControlMode.VEL, which is what makes TMAX non-blocking here."""
STEER_MOTOR_IDS = (1, 3, 5, 7)
"""What flow_base_controller passes as loop_critical_motor_ids -- the four motors in the swerve loop."""


def verify_base(motor_list: list[list[Any]] = BASE_MOTOR_LIST) -> None:
    """Run the check the way the base's chain runs it."""
    mc.verify_motor_config(CHANNEL, motor_list, BASE_CONTROL_MODE, STEER_MOTOR_IDS)


CTRL_MODE = int(DMRegAddr.CTRL_MODE)
GR = int(DMRegAddr.GR)


def as_stored(reg: DMRegAddr, value: float) -> Scalar:
    """What a motor really answers for a float register: the value after a float32 round trip.

    ``PMAX`` 3.1415926 comes back as 3.141592502593994. Storing the float64 instead would let every
    healthy-base test pass without ever exercising ``_REL_TOL``, which is the only reason the scaling
    comparison is not exact.
    """
    spec = REG_BY_ADDR[int(reg)]
    return _data_to_value(spec, bytearray(4) + _value_to_bytes(spec, value))  # value lives in data[4:8]


def healthy_base(motor_list: list[list[Any]] = BASE_MOTOR_LIST) -> dict[int, dict[int, Scalar]]:
    """Every motor answering the identity register, speed mode, and the scaling its own type demands.

    The scaling comes from the driver's own per-type table, ``scaling_for``, rather than being written
    down again here; the numbers it produces are pinned separately by
    ``test_the_scaling_expectation_comes_from_the_driver_constants``. No ``Gr`` entry: this check no
    longer reads it -- see ``test_motor_check_types.py``.
    """
    return {
        int(motor_id): {
            int(DMRegAddr.SW_VER): 925970741,
            CTRL_MODE: mc.CTRL_MODE_SPEED,
            **{int(reg): as_stored(reg, want) for reg, want in mc.scaling_for(str(motor_type)).items()},
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
        monkeypatch.setattr(mc, "RawCanInterface", lambda **kwargs: fake.bus)
        monkeypatch.setattr(mc, "read_register", fake.read)
        monkeypatch.setattr(mc, "write_register", fake.write)
        monkeypatch.setattr(mc, "save_register_to_flash", fake.save)
        return fake

    return _install


# --------------------------------------------------------------------------------------------------
# The expectation itself
# --------------------------------------------------------------------------------------------------


def test_the_expected_mode_comes_from_the_chains_control_mode() -> None:
    """Pinned with literals, from register 10's own meaning: 1 MIT, 2 pos-speed, 3 speed, 4 torque-pos.

    This is the number the check *writes to Flash*, so a wrong entry here would persist the wrong mode on
    every motor of a healthy chain. It used to be a hardcoded 3, which was right only because the Flow
    Base was the only caller.
    """
    assert mc.expected_ctrl_mode("MIT") == 1
    assert mc.expected_ctrl_mode("POS_VEL") == 2
    assert mc.expected_ctrl_mode("VEL") == 3
    assert mc.CTRL_MODE_SPEED == 3, "the base's mode, still named because its docs cite it"


def test_an_unrecorded_control_mode_raises_rather_than_guessing() -> None:
    """The fallback would be writing *some* mode to Flash on every motor. There is no safe default."""
    with pytest.raises(ValueError, match="no CTRL_MODE recorded"):
        mc.expected_ctrl_mode("TORQUE_POS")


def test_the_mode_comparison_is_exact() -> None:
    """CTRL_MODE is a uint32, so there is no tolerance and no float32 round-trip to absorb."""
    spec = REG_BY_ADDR[CTRL_MODE]
    assert not spec.is_float
    assert mc.matches(spec, 3, 3)
    for other in (0, 1, 2, 4):
        assert not mc.matches(spec, other, 3)


def test_the_expected_value_is_writable_and_encodable() -> None:
    """Catches a ``CTRL_MODE_SPEED = 3.0`` typo: a uint32 register rejects a float outright."""
    spec = REG_BY_ADDR[CTRL_MODE]
    assert spec.rw, "CTRL_MODE is read-only, so it could never be repaired"
    _value_to_bytes(spec, mc.CTRL_MODE_SPEED)


def test_describe_names_both_modes() -> None:
    described = mc.describe(CHANNEL, 7, DMRegAddr.CTRL_MODE, 1, mc.CTRL_MODE_SPEED)
    assert "MIT" in described and "speed" in described
    assert "motor 7" in described and CHANNEL in described


def test_describe_explains_an_unwritten_flash_cell() -> None:
    """A float register that never held a value decodes to NaN, which is a reading, not a bus fault."""
    described = mc.describe(CHANNEL, 2, DMRegAddr.PMAX, float("nan"), math.pi)
    assert "never written" in described and "NaN" in described


def test_the_scaling_expectation_comes_from_the_driver_constants() -> None:
    """Both Flow Base motor types decode through pi / 30 / 10 -- the settled value, see the README."""
    for motor_type in ("DM4310V", "DM_FLOW_WHEEL"):
        assert mc.scaling_for(motor_type) == {
            DMRegAddr.PMAX: pytest.approx(math.pi, abs=1e-6),
            DMRegAddr.VMAX: 30.0,
            DMRegAddr.TMAX: 10.0,
        }
    # The entry the 12.5 / 45 confusion came from. It is a different motor, and neither of ours.
    assert mc.scaling_for("DMH6215MIT")[DMRegAddr.VMAX] == 45.0


def test_which_registers_can_block_follows_from_the_control_mode() -> None:
    """All three are always read and compared; whether TMAX can block is the mode's business.

    On the base (VEL) PMAX decodes the wheel angle the kinematics are rebuilt from and VMAX the rate the
    odometry, the caster-flip brake and both detectors act on, but the VEL frame carries no torque, so
    TMAX only decodes MotorInfo.eff and leaves the process. On a MIT chain -- an arm -- set_control
    encodes commanded torque through TMAX, so there it is the most consequential of the three.
    """
    assert set(mc.loop_registers("VEL")) == {DMRegAddr.PMAX, DMRegAddr.VMAX}
    assert set(mc.loop_registers("POS_VEL")) == {DMRegAddr.PMAX, DMRegAddr.VMAX}
    assert set(mc.loop_registers("MIT")) == set(mc.SCALING_REGISTERS)
    assert DMRegAddr.TMAX in mc.SCALING_REGISTERS, "read and reported on every motor in every mode"
    for mode in ("MIT", "VEL", "POS_VEL"):
        assert set(mc.loop_registers(mode)) <= set(mc.SCALING_REGISTERS)


def test_matches_absorbs_the_float32_round_trip_but_not_a_typo() -> None:
    """Why the scaling comparison is not exact, and why the tolerance is still tight enough to matter."""
    spec = REG_BY_ADDR[int(DMRegAddr.PMAX)]
    stored = as_stored(DMRegAddr.PMAX, math.pi)
    assert stored != math.pi, "the round trip must actually lose precision, or this proves nothing"
    assert mc.matches(spec, stored, math.pi)
    assert not mc.matches(spec, 3.1416, math.pi), "a hand-typed constant is 2.4e-6 off and must fail"
    assert not mc.matches(spec, 12.5, math.pi)
    assert not mc.matches(spec, float("nan"), math.pi), "an unwritten Flash cell is a mismatch"


# --------------------------------------------------------------------------------------------------
# Against the fake register file
# --------------------------------------------------------------------------------------------------


def test_a_healthy_base_writes_nothing(bank: Any) -> None:
    """The idempotence property: a correctly configured base must never be touched."""
    fake = bank(healthy_base())
    verify_base()
    assert fake.writes() == []
    assert fake.saves() == []
    assert fake.bus.closed


def test_every_motor_is_checked_not_just_the_steering_ones(bank: Any) -> None:
    fake = bank(healthy_base())
    verify_base()
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
        verify_base()

    assert fake.writes() == [("write", motor_id, CTRL_MODE)]
    assert fake.saves() == [("save", motor_id, CTRL_MODE)]
    assert registers[motor_id][CTRL_MODE] == mc.CTRL_MODE_SPEED
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

    verify_base()

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
        verify_base()

    assert fake.saves() == []
    assert fake.bus.closed


def test_the_first_repair_that_fails_stops_the_rest(bank: Any) -> None:
    """A bus that just dropped one Flash write is not a bus to commit the others on.

    The read phase already enforces this -- the ``unreadable`` and ``mis_scaled`` guards refuse before any
    repair -- but the write phase used to evaluate every motor regardless. On a Flow Base with motors 1, 3
    and 5 all needing the same repair, losing the bus during motor 1's write meant 3 and 5 were still
    written and Flash-saved over it, and ``save_register_to_flash`` is the call most likely to false-ack on
    a contended bus.
    """
    registers = healthy_base()
    for motor_id in (1, 3, 5):
        registers[motor_id][CTRL_MODE] = 1
    fake = bank(registers, reject=(1,))

    with pytest.raises(RuntimeError, match=r"motor\(s\) \[1\] would not take"):
        verify_base()

    assert fake.writes() == [("write", 1, CTRL_MODE)], "motors 3 and 5 must not be written"
    assert fake.saves() == []
    assert registers[3][CTRL_MODE] == 1 and registers[5][CTRL_MODE] == 1


def test_the_motors_behind_a_failed_repair_are_named_as_not_attempted(bank: Any) -> None:
    """Silence about them would read as "those were fine": they still need the same repair."""
    registers = healthy_base()
    for motor_id in (1, 3, 5):
        registers[motor_id][CTRL_MODE] = 1
    bank(registers, reject=(1,))

    with pytest.raises(RuntimeError) as excinfo:
        verify_base()

    assert "[3, 5] also needed the same repair and were NOT attempted" in str(excinfo.value)


def test_a_lone_failed_repair_says_nothing_about_others(bank: Any) -> None:
    """Nothing was behind it, so the message must not invent an empty list."""
    registers = healthy_base()
    registers[5][CTRL_MODE] = 1
    bank(registers, reject=(5,))

    with pytest.raises(RuntimeError) as excinfo:
        verify_base()

    assert "NOT attempted" not in str(excinfo.value)


def test_a_survey_reports_a_wrong_control_mode_without_writing_it(bank: Any) -> None:
    """``repair=False`` is what makes ``--survey-only`` a survey rather than a reconfiguration.

    The flag used to skip building the chain while leaving this write path untouched, so an operator
    copying the arm survey line (``--control-mode MIT``) and pointing it at a Flow Base rewrote every
    motor holding CTRL_MODE 3 to 1 and committed it to Flash -- from a command documented as read-only.
    Refused rather than passed over, so a scripted fleet sweep can still trust the exit code.
    """
    registers = healthy_base()
    for motor_id in (1, 3):
        registers[motor_id][CTRL_MODE] = 1
    fake = bank(registers)

    with pytest.raises(RuntimeError, match=r"survey on can_test: motor\(s\) \[1, 3\]"):
        mc.verify_motor_config(CHANNEL, BASE_MOTOR_LIST, BASE_CONTROL_MODE, STEER_MOTOR_IDS, repair=False)

    assert fake.writes() == []
    assert fake.saves() == []
    assert registers[1][CTRL_MODE] == 1 and registers[3][CTRL_MODE] == 1, "the survey must leave them alone"
    assert fake.bus.closed


def test_a_survey_names_the_command_that_fixes_each_motor(bank: Any, caplog: pytest.LogCaptureFixture) -> None:
    """A survey that found the fault and withheld the remedy would just move the dead end.

    Same shape ``_check_scaling`` already uses for the three registers no caller ever repairs.
    """
    registers = healthy_base()
    registers[3][CTRL_MODE] = 1
    bank(registers)

    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError):
        mc.verify_motor_config(CHANNEL, BASE_MOTOR_LIST, BASE_CONTROL_MODE, STEER_MOTOR_IDS, repair=False)

    assert (
        f"dm_motor_registers.py write CTRL_MODE --value {mc.CTRL_MODE_SPEED} --motor-id 3 --channel {CHANNEL}"
        in caplog.text
    )


def test_a_survey_of_a_healthy_chain_passes_and_claims_no_writes(bank: Any, caplog: pytest.LogCaptureFixture) -> None:
    """The pass line must not report a write count on a run that was never allowed to write.

    "0 motor(s) repaired and saved to Flash" reads as a write path that ran and found nothing to do,
    which is the wrong thing to tell an operator about a survey.
    """
    bank(healthy_base())

    with caplog.at_level(logging.INFO):
        mc.verify_motor_config(CHANNEL, BASE_MOTOR_LIST, BASE_CONTROL_MODE, STEER_MOTOR_IDS, repair=False)

    assert "survey passed on can_test" in caplog.text
    assert "nothing was written" in caplog.text
    assert "repaired and saved to Flash" not in caplog.text


def test_a_silent_motor_is_probed_once_and_suppresses_every_write(bank: Any) -> None:
    """One probe, not three: a dead register costs ~0.65 s of retries inside _tx_rx."""
    registers = healthy_base()
    registers[1][CTRL_MODE] = 1  # a repair that must NOT happen
    fake = bank(registers, silent=(7,))

    with pytest.raises(RuntimeError, match="did not answer"):
        verify_base()

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
        verify_base()

    assert "power-cycle motor 1" in caplog.text
    assert "E stop is not the problem" in caplog.text


def test_an_unopenable_bus_is_reported_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(**kwargs: Any) -> None:
        raise OSError("no such device")

    monkeypatch.setattr(mc, "RawCanInterface", explode)
    with pytest.raises(RuntimeError, match="could not open"):
        verify_base()


def test_the_rail_motor_is_checked_too(bank: Any) -> None:
    """LinearRailVehicle passes a 9-motor list; the 9th is the lift and is not exempt."""
    motor_list = [*BASE_MOTOR_LIST, [9, "DM8009"]]
    registers = healthy_base(motor_list)
    registers[9][CTRL_MODE] = 1
    fake = bank(registers)

    verify_base(motor_list)
    assert fake.writes() == [("write", 9, CTRL_MODE)]


# --------------------------------------------------------------------------------------------------
# Scaling: read on every motor, written on none
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("motor_id", [1, 7])
def test_a_mis_scaled_steering_motor_refuses_to_start(
    bank: Any, caplog: pytest.LogCaptureFixture, motor_id: int
) -> None:
    """A loop-critical motor's PMAX is the scale its reported position decodes through.

    On the base that position is the wheel angle the swerve kinematics are rebuilt from every cycle. The
    consequence line is worded from the mechanism rather than from the swerve, because an arm running this
    check gets the same clause about a joint angle -- see i2rt/flow_base/README.md for the base's version.
    """
    registers = healthy_base()
    registers[motor_id][int(DMRegAddr.PMAX)] = as_stored(DMRegAddr.PMAX, 12.5)
    fake = bank(registers)

    with caplog.at_level("ERROR"), pytest.raises(RuntimeError, match="mis-scaled") as raised:
        verify_base()

    assert f"motor(s) [{motor_id}]" in str(raised.value)
    assert "NOT started" in str(raised.value)
    assert "solving for a place this joint is not" in caplog.text
    assert "NOT starting" in caplog.text
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
        verify_base()  # must not raise

    assert "not one the caller marked as inside its control loop" in caplog.text
    assert "Starting anyway" in caplog.text
    assert "run the motor type check" in caplog.text, "a wrong scale can be a wrong part; say so"
    assert fake.writes() == []


@pytest.mark.parametrize("motor_id", [1, 7])
def test_a_mis_scaled_tmax_on_a_steering_motor_reports_but_does_not_block(
    bank: Any, caplog: pytest.LogCaptureFixture, motor_id: int
) -> None:
    """On a VEL chain TMAX decodes MotorInfo.eff, and eff leaves the process without being read back.

    Its two readers on the base are get_wheel_states and the linear rail state; nothing in the control
    loop, the kinematics, the odometry or the caster fault check consumes a torque, and the MIT branch of
    set_control that would use TMAX to *encode* one is unreachable on a ControlMode.VEL chain. So this is
    reported like any other mismatch and the launch continues -- proven by letting a genuine CTRL_MODE
    repair run on another motor in the same pass, which the mis-scaled gate (it raises before any repair)
    would otherwise have pre-empted. On a MIT chain the same register *does* block; see
    test_tmax_blocks_on_a_mit_chain.
    """
    registers = healthy_base()
    registers[motor_id][int(DMRegAddr.TMAX)] = as_stored(DMRegAddr.TMAX, 54.0)
    registers[2][CTRL_MODE] = 1
    fake = bank(registers)

    with caplog.at_level("ERROR"):
        verify_base()  # must not raise

    assert "carries no torque" in caplog.text, "a VEL frame encodes nothing through TMAX"
    assert "MotorInfo.eff" in caplog.text
    assert "fix it by hand" in caplog.text, "still gets the full fix-it-by-hand line"
    assert "solving for a place this joint is not" not in caplog.text, "TMAX is not the position scale"
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
        verify_base()

    assert f"motor(s) [{motor_id}]" in str(raised.value)
    assert "NOT started" in str(raised.value)
    assert "judging how fast it is moving" in caplog.text
    assert "solving for a place this joint is not" not in caplog.text, "the position is not what is mis-scaled"
    assert fake.writes() == []


def test_one_bad_loop_register_is_enough_to_block(bank: Any, caplog: pytest.LogCaptureFixture) -> None:
    """``any`` over the bad registers: a harmless TMAX alongside a fatal PMAX must not dilute it."""
    registers = healthy_base()
    registers[3][int(DMRegAddr.TMAX)] = as_stored(DMRegAddr.TMAX, 54.0)
    registers[3][int(DMRegAddr.PMAX)] = as_stored(DMRegAddr.PMAX, 12.5)
    bank(registers)

    with caplog.at_level("ERROR"), pytest.raises(RuntimeError, match="mis-scaled"):
        verify_base()

    assert "solving for a place this joint is not" in caplog.text, "the PMAX line keeps its own consequence"
    assert "carries no torque" in caplog.text, "and the TMAX line keeps its own, on the same motor"


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

    verify_base()

    assert fake.writes() == [("write", 3, CTRL_MODE)], "only CTRL_MODE is ever written"
    assert fake.saves() == [("save", 3, CTRL_MODE)]
    touched = {call[2] for call in fake.calls if call[0] in ("write", "save")}
    assert touched.isdisjoint({int(reg) for reg in mc.SCALING_REGISTERS})


def test_an_unwritten_scaling_cell_is_reported_as_such(bank: Any, caplog: pytest.LogCaptureFixture) -> None:
    """An erased Flash cell reads 0xFFFFFFFF, which decodes to NaN. It is a mismatch, not a bus error."""
    registers = healthy_base()
    registers[2][int(DMRegAddr.PMAX)] = float("nan")
    bank(registers)

    with caplog.at_level("ERROR"):
        verify_base()

    assert "never written" in caplog.text


def test_every_motor_has_its_scaling_read_including_the_rail(bank: Any) -> None:
    motor_list = [*BASE_MOTOR_LIST, [9, "DM8009"]]
    fake = bank(healthy_base(motor_list))

    verify_base(motor_list)

    for reg in mc.SCALING_REGISTERS:
        read_ids = {call[1] for call in fake.calls if call[0] == "read" and call[2] == int(reg)}
        assert read_ids == {1, 2, 3, 4, 5, 6, 7, 8, 9}, f"{REG_BY_ADDR[int(reg)].name} was not read everywhere"


# --------------------------------------------------------------------------------------------------
# The controller's side of the wiring
# --------------------------------------------------------------------------------------------------


def test_disabling_the_checks_says_what_it_gives_up(caplog: pytest.LogCaptureFixture) -> None:
    """--no-verify-motor-config now silences both checks, so the warning has to name both costs."""
    from i2rt.flow_base import flow_base_controller

    with caplog.at_level("WARNING"):
        flow_base_controller._warn_checks_disabled(CHANNEL)
    assert "DISABLED" in caplog.text
    assert "speed mode" in caplog.text, "the CTRL_MODE cost"
    assert "PMAX/VMAX/TMAX" in caplog.text, "the scaling cost"
    assert "gear ratio" in caplog.text, "the motor-type cost, which this flag now also skips"


def _base_chain_kwargs(monkeypatch: pytest.MonkeyPatch, verify_motor_config: bool) -> dict[str, Any]:
    """The kwargs ``_initialize_motor_chain`` hands ``DMChainCanInterface``, without opening anything.

    Unbound, with ``None`` for ``self``: the method reads only its own parameters. Asserting at the call
    site is the point -- the chain runs the checks itself now, so the flag reaching it *is* the wiring.
    """
    from i2rt.flow_base import flow_base_controller

    seen: dict[str, Any] = {}

    class _Stop(Exception):
        pass

    def _capture(*_args: Any, **kwargs: Any) -> None:
        seen.update(kwargs)
        raise _Stop

    monkeypatch.setattr(flow_base_controller, "DMChainCanInterface", _capture)
    with pytest.raises(_Stop):
        flow_base_controller.VehicleMotorController._initialize_motor_chain(
            None, CHANNEL, [0.0] * 4, [1] * 4, verify_motor_config=verify_motor_config
        )
    return seen


def test_disabling_the_check_actually_reaches_the_chain_as_both_flags_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """--no-verify-motor-config has to *silence* both checks, not just warn about them.

    The test above pins the warning's wording; this pins the suppression. The pre-rename suite guarded it
    by monkeypatching the check to raise ("the check must not run, let alone touch the bus, when it is
    disabled"), which stopped applying when the controller stopped calling the check itself.
    """
    seen = _base_chain_kwargs(monkeypatch, verify_motor_config=False)
    assert seen["check_motor_types"] is False
    assert seen["check_motor_config"] is False


def test_the_base_opts_into_both_checks_and_names_its_loop_critical_motors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One flag drives both, and STEER_MOTOR_IDS is the half the check cannot infer -- the wiring this
    PR rewired, and the reason a drive or rail motor's mis-scaled feedback does not refuse a launch."""
    from i2rt.flow_base import flow_base_controller

    seen = _base_chain_kwargs(monkeypatch, verify_motor_config=True)
    assert seen["check_motor_types"] is True
    assert seen["check_motor_config"] is True
    assert seen["loop_critical_motor_ids"] == flow_base_controller.STEER_MOTOR_IDS
    assert seen["control_mode"] == BASE_CONTROL_MODE, "VEL is what makes TMAX non-blocking on the base"


def test_the_base_marks_only_its_steering_motors_loop_critical() -> None:
    """The constant this file's ``verify_base`` mirrors. A drive or rail motor's mis-scaled feedback
    moves only odometry, so it must not be able to abort a launch."""
    from i2rt.flow_base import flow_base_controller

    assert flow_base_controller.STEER_MOTOR_IDS == STEER_MOTOR_IDS
    drive_and_rail = {motor_id for motor_id, _ in BASE_MOTOR_LIST} - set(STEER_MOTOR_IDS)
    assert drive_and_rail == {2, 4, 6, 8}, "the drive motors; the rail (9) is added per-test"


# --------------------------------------------------------------------------------------------------
# A MIT chain: the generalization the Flow Base cannot exercise
# --------------------------------------------------------------------------------------------------

ARM_MOTOR_LIST = [[1, "DM4340"], [2, "DM4310"]]
"""Two joints of a YAM. Enough to pin the MIT policy without restating a whole arm config."""


def healthy_arm() -> dict[int, dict[int, Scalar]]:
    """``healthy_base`` with MIT as the mode every motor holds."""
    registers = healthy_base(ARM_MOTOR_LIST)
    for motor in registers.values():
        motor[CTRL_MODE] = mc.expected_ctrl_mode("MIT")
    return registers


def verify_arm(motor_list: list[list[Any]] = ARM_MOTOR_LIST) -> None:
    """Run the check the way an arm's chain would: MIT, and every motor loop-critical."""
    mc.verify_motor_config(CHANNEL, motor_list, "MIT")


def test_a_healthy_mit_chain_writes_nothing(bank: Any) -> None:
    """The base's fixture would fail this: its motors hold speed mode, which is wrong for MIT."""
    fake = bank(healthy_arm())
    verify_arm()
    assert fake.writes() == []
    assert fake.saves() == []


def test_a_mit_chain_is_repaired_to_mit_not_to_speed(bank: Any) -> None:
    """The bug the hardcoded CTRL_MODE_SPEED would have been: speed mode written to every arm motor.

    A wrong value here is persisted to Flash on a *healthy* chain, so this is the one assertion in this
    file that guards against the check breaking hardware rather than failing to catch it.
    """
    registers = healthy_arm()
    registers[2][CTRL_MODE] = 3  # speed mode on an arm: wrong, and what the old code would have wanted
    fake = bank(registers)

    verify_arm()

    assert fake.writes() == [("write", 2, CTRL_MODE)]
    assert fake.saves() == [("save", 2, CTRL_MODE)]
    assert registers[2][CTRL_MODE] == 1, "MIT, not the 3 a Flow-Base-shaped check would have written"


def test_tmax_blocks_on_a_mit_chain(bank: Any, caplog: pytest.LogCaptureFixture) -> None:
    """The same register that is telemetry-only on the base encodes commanded torque here.

    ``set_control``'s MIT branch packs torque against TORQUE_MIN/MAX, so a firmware TMAX that disagrees
    rescales every torque the arm applies -- gravity compensation included, which is the whole reason an
    arm holds still.
    """
    registers = healthy_arm()
    registers[1][int(DMRegAddr.TMAX)] = as_stored(DMRegAddr.TMAX, 10.0)  # a DM4310's, on a DM4340
    bank(registers)

    with caplog.at_level("ERROR"), pytest.raises(RuntimeError, match="mis-scaled"):
        verify_arm()

    assert "encodes commanded torque through TMAX" in caplog.text
    assert "gravity compensation included" in caplog.text
    assert "NOT starting" in caplog.text


def test_omitting_the_loop_ids_makes_every_motor_able_to_block(bank: Any) -> None:
    """The strict default. On the base motor 2 is a drive motor and only warns; with no ids passed, the
    same mismatch on the same id blocks, because nothing has said it is outside the loop."""
    registers = healthy_arm()
    registers[2][int(DMRegAddr.VMAX)] = as_stored(DMRegAddr.VMAX, 45.0)
    bank(registers)

    with pytest.raises(RuntimeError, match="mis-scaled"):
        verify_arm()


def test_an_empty_loop_id_set_blocks_on_nothing(bank: Any, caplog: pytest.LogCaptureFixture) -> None:
    """``()`` is not the same as ``None``: it says no motor's feedback is acted on, so nothing blocks.

    Worth pinning because the natural implementation of "default to all" -- ``or set(motor_ids)`` -- would
    silently turn this case into the strict one.
    """
    registers = healthy_arm()
    registers[1][int(DMRegAddr.PMAX)] = as_stored(DMRegAddr.PMAX, 3.14159)  # a DM4310's, on a DM4340
    bank(registers)

    with caplog.at_level("ERROR"):
        mc.verify_motor_config(CHANNEL, ARM_MOTOR_LIST, "MIT", ())  # must not raise

    assert "not one the caller marked as inside its control loop" in caplog.text


def test_a_passive_row_is_skipped_rather_than_sent_to_get_motor_constants(bank: Any) -> None:
    """``no_gripper`` and ``yam_teaching_handle`` are spelled ``motor_type: ""``, and neither is a motor.

    Reachable through ``_get_gripper_only_robot``, which rejects NO_GRIPPER but not the teaching handle,
    so ``ArmType.NO_ARM`` + that handle builds a chain whose one row is ``[7, ""]``. Without the skip in
    ``_motor_rows`` this check would call ``get_motor_constants("")`` and raise "Motor type '' not
    recognized" -- a table-gap message about a row that is not a motor at all.
    """
    fake = bank(healthy_arm())
    mc.verify_motor_config(CHANNEL, [[7, ""]], "MIT")
    assert fake.calls == []
    # Every other test asserts the bus was closed; here it was never opened, so it was never closed.
    assert not fake.bus.closed, "nothing to check means no socket is opened"


def test_a_passive_row_beside_a_real_motor_only_skips_itself(bank: Any) -> None:
    fake = bank(healthy_arm())
    mc.verify_motor_config(CHANNEL, [[1, "DM4340"], [7, ""]], "MIT")
    read_from = [motor_id for kind, motor_id, addr in fake.calls if kind == "read" and addr == CTRL_MODE]
    assert read_from == [1]


def test_every_shipped_config_motor_type_has_motor_constants() -> None:
    """The scaling half of ``test_every_shipped_config_motor_type_has_a_gear_ratio`` next door.

    Now that arms run this check, a motor type in a shipped config with no ``MotorConstants`` entry
    would raise ``ValueError`` from ``scaling_for`` on a real launch. Failing in CI is the point: the
    gear-ratio table and the constants table are separate, and a type can be in one and not the other.
    """
    declared: dict[str, str] = {}
    for path in sorted(CONFIG_DIR.glob("*.yml")):
        raw = yaml.safe_load(path.read_text())
        for _can_id, motor_type in raw.get("motor_list", []):
            declared[motor_type] = path.name
        if raw.get("motor_type"):
            declared[raw["motor_type"]] = path.name

    assert declared, "no motor types found -- the config glob is wrong, not the table"
    missing = {}
    for name, where in declared.items():
        try:
            MotorType.get_motor_constants(name)
        except ValueError:
            missing[name] = where
    assert not missing, f"add these to get_motor_constants in i2rt/motor_drivers/utils.py: {missing}"


def test_the_arm_scaling_expectations_are_the_ones_a_survey_will_be_held_to() -> None:
    """Pins what an arm's four motor types must hold, so a constants edit cannot silently move the bar.

    DM4340's VMAX of 10 is the value worth pinning: it is the only arm entry below the 30 a DM motor is
    otherwise expected to hold, which makes a stock-30 joint 1-3 motor the likeliest reason a previously
    working arm is now refused.
    """
    P, V, T = DMRegAddr.PMAX, DMRegAddr.VMAX, DMRegAddr.TMAX
    assert mc.scaling_for("DM4310") == {P: 12.5, V: 30.0, T: 10.0}
    assert mc.scaling_for("DM4340") == {P: 12.5, V: 10.0, T: 28.0}
    assert mc.scaling_for("DM6248") == {P: 12.5, V: 20.0, T: 120.0}
    assert mc.scaling_for("DM3507") == {P: 12.5, V: 50.0, T: 5.0}


def test_a_stock_vmax_on_an_arms_shoulder_motor_refuses_to_start(bank: Any, caplog: pytest.LogCaptureFixture) -> None:
    """The concrete failure enabling this on arms buys, and the one it costs.

    A DM4340 whose Flash still holds the stock VMAX of 30 mis-scales every velocity that joint reports by
    3x -- and on a MIT arm with no loop-critical subset named, that refuses the launch rather than being
    logged. Before this was enabled the arm started and the mis-scaled dq went unnoticed.
    """
    registers = healthy_arm()
    registers[1][int(DMRegAddr.VMAX)] = as_stored(DMRegAddr.VMAX, 30.0)
    bank(registers)

    with caplog.at_level("ERROR"), pytest.raises(RuntimeError, match="mis-scaled"):
        verify_arm()

    assert "VMAX" in caplog.text
    assert "NOT starting" in caplog.text
