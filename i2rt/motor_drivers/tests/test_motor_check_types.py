"""Tests for ``motor_check.verify_motor_types`` -- the ``Gr`` half.

``verify_motor_config``, the other half of that module, has its own file next to this one.

Everything here is hardware-free. As in ``test_motor_check_config.py`` next door, the
bus-driving routine runs against a fake *register file* -- a dict per motor, one level above the wire
-- rather than a mocked ``can`` bus, so the tests pin this module's decisions and not python-can's.

``FakeMotors`` has no ``write``, and the ``bank`` fixture patches only ``read_register``, so a write
would go to a real bus. ``test_this_check_never_writes`` closes that off properly by patching both
writers to raise: ``motor_check`` does hold them, since ``verify_motor_config`` repairs ``CTRL_MODE``.

The last two sections cover ``run_startup_checks`` -- which checks may run, in what order, and whether
they may run at all -- and the ``dm_driver`` CLI that declares a chain's types by hand.
"""

from __future__ import annotations

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
from i2rt.motor_drivers.dm_driver import ControlMode, DMSingleMotorCanInterface
from i2rt.motor_drivers.utils import MotorType
from i2rt.robots.utils import _CONFIG_DIR, ArmType, _load_arm_config

CHANNEL = "can_test"
GR = int(DMRegAddr.GR)
# The arm and gripper configs, taken from the loader's own constant so the glob below cannot drift
# from what _load_arm_config reads. Not derived from mc.__file__: that module lives in
# i2rt/motor_drivers/ now, which has no config dir.
CONFIG_DIR = Path(_CONFIG_DIR)

# The chain a `yam_ultra` (v1) arm with a linear_4310 gripper is built as: see yam_ultra_v1.yml plus
# the 0x07 gripper motor get_yam_robot appends.
YAM_ULTRA_V1_CHAIN = [
    [1, "DM4340"],
    [2, "DM4340"],
    [3, "DM4340"],
    [4, "DM4310"],
    [5, "DM4310"],
    [6, "DM4310"],
    [7, "DM4310"],
]


def as_stored(value: float) -> Scalar:
    """What a motor really answers for ``Gr``: the value after a float32 round trip.

    Storing the float64 instead would let every passing test pass without exercising the tolerance.
    """
    spec = REG_BY_ADDR[GR]
    return _data_to_value(spec, bytearray(4) + _value_to_bytes(spec, value))  # value lives in data[4:8]


def matching_chain(motor_list: list[list[Any]] = YAM_ULTRA_V1_CHAIN) -> dict[int, dict[int, Scalar]]:
    """Every motor answering the gear ratio its declared type demands, plus the logged registers.

    The ratio comes from ``MotorType.get_gear_ratio`` rather than being written down again here; the
    numbers it produces are pinned separately by ``test_gear_ratios_follow_the_part_number``.
    """
    return {
        int(motor_id): {
            GR: as_stored(MotorType.get_gear_ratio(str(motor_type))),
            int(DMRegAddr.HW_VER): 1,
            int(DMRegAddr.SW_VER): 925970741,
        }
        for motor_id, motor_type in motor_list
    }


class FakeBus:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeMotors:
    """A dict-backed register file per motor id, with per-motor failure injection."""

    def __init__(
        self,
        registers: dict[int, dict[int, Scalar]],
        silent: tuple[int, ...] = (),
        unreadable: tuple[tuple[int, int], ...] = (),
        transient: dict[int, int] | None = None,
    ) -> None:
        self.registers = registers
        self.silent = set(silent)  # ids that never answer any register
        self.unreadable = set(unreadable)  # (id, addr) pairs that never answer
        self.transient = dict(transient or {})  # id -> how many leading reads still fail
        self.calls: list[tuple[int, int]] = []
        self.bus = FakeBus()

    def read(self, iface: Any, motor_id: int, reg: DMRegAddr) -> Scalar:
        addr = int(reg)
        self.calls.append((motor_id, addr))
        if motor_id in self.silent or (motor_id, addr) in self.unreadable:
            raise RuntimeError(f"Failed CAN exchange for motor {motor_id}")
        if self.transient.get(motor_id, 0) > 0:
            self.transient[motor_id] -= 1
            raise RuntimeError(f"Failed CAN exchange for motor {motor_id}")
        return self.registers[motor_id][addr]

    def reads_of(self, addr: int) -> list[int]:
        return [motor_id for motor_id, called_addr in self.calls if called_addr == addr]


@pytest.fixture
def bank(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Install a fake register file and make the retry pause free."""
    created: list[dict[str, Any]] = []

    def _install(
        registers: dict[int, dict[int, Scalar]],
        silent: tuple[int, ...] = (),
        unreadable: tuple[tuple[int, int], ...] = (),
        transient: dict[int, int] | None = None,
    ) -> FakeMotors:
        fake = FakeMotors(registers, silent=silent, unreadable=unreadable, transient=transient)

        def _open(**kwargs: Any) -> FakeBus:
            created.append(kwargs)
            return fake.bus

        monkeypatch.setattr(mc, "RawCanInterface", _open)
        monkeypatch.setattr(mc, "read_register", fake.read)
        monkeypatch.setattr(mc, "RETRY_SLEEP_S", 0.0)  # the retry policy, not its wall-clock cost
        fake.opened = created  # type: ignore[attr-defined]
        return fake

    return _install


# --------------------------------------------------------------------------------------------------
# The expectation itself
# --------------------------------------------------------------------------------------------------


def test_gear_ratios_follow_the_part_number() -> None:
    """A DM part number ends in its gear ratio; a DM8009 reading Gr 9 is what establishes that.

    Literals, not derived: this is the test that fails if a table entry is ever mistyped. The Flow Base
    types are pinned alongside the base's own check, in
    ``test_the_gear_ratios_the_flow_base_runs_on`` below.
    """
    assert MotorType.get_gear_ratio(MotorType.DM3507) == 7.0
    assert MotorType.get_gear_ratio(MotorType.DM4310) == 10.0
    assert MotorType.get_gear_ratio(MotorType.DM4340) == 40.0
    assert MotorType.get_gear_ratio(MotorType.DM6248) == 48.0
    assert MotorType.get_gear_ratio(MotorType.DM8009) == 9.0


def test_an_unrecorded_motor_type_raises_rather_than_being_skipped() -> None:
    """A gap in the table is a code gap, so it must not degrade into "cannot check this one".

    DMH6215MIT is a real ``MotorType`` that nothing in this repo builds, which is exactly the shape of
    the mistake: adding a type to a config without reading its Gr off a motor first.
    """
    with pytest.raises(ValueError, match="No gear ratio recorded"):
        MotorType.get_gear_ratio(MotorType.DMH6215MIT)
    with pytest.raises(ValueError, match=r"i2rt/motor_drivers/utils\.py"):
        MotorType.get_gear_ratio("DM_NOT_A_MOTOR")


def test_known_gear_ratios_hands_out_a_copy() -> None:
    """The reverse lookup iterates this mapping every time a mismatch is described."""
    MotorType.known_gear_ratios()["DM4310"] = 999.0
    assert MotorType.get_gear_ratio(MotorType.DM4310) == 10.0


def test_every_shipped_config_motor_type_has_a_gear_ratio() -> None:
    """Adding a motor type to a config without its gear ratio must fail in CI, not on hardware.

    Covers both shapes: an arm config's ``motor_list`` rows and a gripper config's ``motor_type``.
    The empty string is skipped -- no_gripper and yam_teaching_handle contribute no motor at all, so
    their type never reaches the check.
    """
    declared: dict[str, str] = {}
    for path in sorted(CONFIG_DIR.glob("*.yml")):
        raw = yaml.safe_load(path.read_text())
        for _can_id, motor_type in raw.get("motor_list", []):
            declared[motor_type] = path.name
        if raw.get("motor_type"):
            declared[raw["motor_type"]] = path.name

    assert declared, "no motor types found -- the config glob is wrong, not the table"
    missing = {name: where for name, where in declared.items() if name not in MotorType.known_gear_ratios()}
    assert not missing, f"add these to _GEAR_RATIO in i2rt/motor_drivers/utils.py: {missing}"


def test_the_yam_ultra_revisions_still_differ_at_joint_4() -> None:
    """The premise of this whole module. If v1 and v2 ever agree again, say so out loud."""
    v1 = _load_arm_config(ArmType.YAM_ULTRA).motor_list
    v2 = _load_arm_config(ArmType.YAM_ULTRA_2).motor_list
    assert v1[3][1] == MotorType.DM4310
    assert v2[3][1] == MotorType.DM4340
    assert MotorType.get_gear_ratio(v1[3][1]) != MotorType.get_gear_ratio(v2[3][1]), (
        "Gr cannot tell the two revisions apart, so this check cannot catch the mix-up"
    )


def test_a_mismatch_names_the_motor_that_is_actually_installed() -> None:
    """ "Gr 40" means nothing on its own; "i.e. a DM4340" names the part in the arm."""
    described = mc._describe_mismatch(4, MotorType.DM4310, 40.0, 10.0)
    assert "motor 4" in described
    assert "DM4340" in described, "the reverse lookup is what makes the message actionable"
    assert "DM4310" in described


def test_a_mismatch_on_an_unrecognised_ratio_still_reports() -> None:
    """A Gr matching no known type is still a mismatch; it just cannot name a replacement."""
    described = mc._describe_mismatch(2, MotorType.DM4340, 3.0, 40.0)
    assert "i.e. a" not in described
    assert "Gr 3" in described and "DM4340" in described


def test_the_retry_budget_is_five_attempts_a_tenth_of_a_second_apart() -> None:
    """Pinned with literals, deliberately.

    Every other retry test compares against ``MAX_READ_ATTEMPTS``, which is self-referential: without
    a literal somewhere the whole retry policy could be dialled down to a single attempt and the suite
    would stay green. This is the test that fails when that happens.
    """
    assert mc.MAX_READ_ATTEMPTS == 5
    assert mc.RETRY_SLEEP_S == 0.1


class WroteSomething(Exception):
    """Raised if ``verify_motor_types`` reaches for a writer. Deliberately not an ``AssertionError``.

    ``BUS_ERRORS`` includes ``AssertionError`` (see ``dm_motor_registers``), so an assert here would be
    caught by the code under test and logged as a bus failure -- and the test would pass for the wrong
    reason. Same reason ``test_motor_check_config.py`` defines ``ProtocolViolation``.
    """


def test_this_check_never_writes(bank: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The type check identifies a part; the repair for a mismatch is to swap it, never to write to it.

    ``motor_check`` does hold ``write_register`` and ``save_register_to_flash`` -- ``verify_motor_config``
    repairs ``CTRL_MODE`` with them -- so this has to be asserted against the running check rather than
    against the module's imports or the fake's shape.
    """

    def _forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise WroteSomething("verify_motor_types wrote to a motor")

    bank(matching_chain())
    monkeypatch.setattr(mc, "write_register", _forbidden)
    monkeypatch.setattr(mc, "save_register_to_flash", _forbidden)

    mc.verify_motor_types(CHANNEL, YAM_ULTRA_V1_CHAIN)  # passes, and must get there without writing


# --------------------------------------------------------------------------------------------------
# Against the fake register file
# --------------------------------------------------------------------------------------------------


def test_a_matching_chain_passes_and_closes_the_bus(bank: Any) -> None:
    fake = bank(matching_chain())
    mc.verify_motor_types(CHANNEL, YAM_ULTRA_V1_CHAIN)
    assert fake.bus.closed


def test_every_motor_including_the_gripper_is_checked(bank: Any) -> None:
    """The gripper motor at 0x07 is in the assembled chain, so it is checked like any other."""
    fake = bank(matching_chain())
    mc.verify_motor_types(CHANNEL, YAM_ULTRA_V1_CHAIN)
    assert fake.reads_of(GR) == [1, 2, 3, 4, 5, 6, 7]


def test_the_yam_ultra_revision_mixup_refuses_to_start(bank: Any) -> None:
    """The regression this module exists for: a v2 arm launched as --arm yam_ultra.

    Joint 4 is physically a DM4340 while yam_ultra_v1.yml declares a DM4310, so dm_driver would
    encode its torque against TORQUE_MAX 10 against a motor decoding with 28 -- 2.8x, silently.
    """
    registers = matching_chain()
    registers[4][GR] = as_stored(40.0)  # the v2 hardware answering into a v1 config
    bank(registers)

    with pytest.raises(RuntimeError) as excinfo:
        mc.verify_motor_types(CHANNEL, YAM_ULTRA_V1_CHAIN)

    message = str(excinfo.value)
    assert "motor 4" in message
    assert "DM4340" in message and "DM4310" in message
    assert "yam_ultra_2" in message, "the message should name the variant the operator probably wants"
    assert "NOT started" in message


def test_every_wrong_motor_is_named_in_one_message(bank: Any) -> None:
    """Reads have all succeeded by then, so walking the rest of the chain is nearly free."""
    registers = matching_chain()
    registers[2][GR] = as_stored(10.0)
    registers[5][GR] = as_stored(40.0)
    fake = bank(registers)

    with pytest.raises(RuntimeError) as excinfo:
        mc.verify_motor_types(CHANNEL, YAM_ULTRA_V1_CHAIN)

    message = str(excinfo.value)
    assert "motor 2" in message and "motor 5" in message
    assert fake.reads_of(GR) == [1, 2, 3, 4, 5, 6, 7], "a mismatch must not stop the walk"
    assert fake.bus.closed


def test_a_silent_motor_is_retried_then_raises(bank: Any) -> None:
    fake = bank(matching_chain(), silent=(3,))

    with pytest.raises(RuntimeError, match="did not give a readable Gr"):
        mc.verify_motor_types(CHANNEL, YAM_ULTRA_V1_CHAIN)

    assert fake.reads_of(GR).count(3) == mc.MAX_READ_ATTEMPTS
    assert 4 not in fake.reads_of(GR), "an unreadable motor bails instead of charging the rest ~3.8 s each"
    assert fake.bus.closed


def test_a_nan_gear_ratio_is_retried_then_raises(bank: Any) -> None:
    """An unwritten Flash cell reads 0xFFFFFFFF, i.e. NaN -- a reading, but not a gear ratio."""
    registers = matching_chain()
    registers[1][GR] = float("nan")
    fake = bank(registers)

    with pytest.raises(RuntimeError, match="did not give a readable Gr"):
        mc.verify_motor_types(CHANNEL, YAM_ULTRA_V1_CHAIN)

    assert fake.reads_of(GR).count(1) == mc.MAX_READ_ATTEMPTS


def test_a_transient_failure_is_absorbed_by_the_retry(bank: Any) -> None:
    """The whole point of retrying: a burst of bus noise must not refuse to start a healthy arm.

    Four leading failures as a literal, not ``MAX_READ_ATTEMPTS - 1``, so shrinking the budget turns
    this from "absorbed" into a refusal and the test says so.
    """
    fake = bank(matching_chain(), transient={2: 4})
    mc.verify_motor_types(CHANNEL, YAM_ULTRA_V1_CHAIN)
    assert fake.reads_of(GR).count(2) == 5


def test_the_retry_pauses_between_attempts(bank: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """One fewer sleep than attempts: the last failure raises rather than waiting to retry."""
    bank(matching_chain(), silent=(1,))
    monkeypatch.setattr(mc, "RETRY_SLEEP_S", 0.1)
    slept: list[float] = []
    monkeypatch.setattr(mc.time, "sleep", slept.append)

    with pytest.raises(RuntimeError):
        mc.verify_motor_types(CHANNEL, YAM_ULTRA_V1_CHAIN)

    assert slept == [0.1] * 4  # literal, for the reason in test_the_retry_budget_is_five_attempts...


def test_a_logged_register_that_will_not_read_does_not_block(bank: Any, caplog: pytest.LogCaptureFixture) -> None:
    """hw_ver/sw_ver are compared against nothing, so failing to read one proves nothing."""
    fake = bank(matching_chain(), unreadable=((3, int(DMRegAddr.SW_VER)),))
    with caplog.at_level("WARNING", logger=mc.logger.name):
        mc.verify_motor_types(CHANNEL, YAM_ULTRA_V1_CHAIN)
    assert "sw_ver" in caplog.text and "logged only" in caplog.text
    assert fake.reads_of(GR) == [1, 2, 3, 4, 5, 6, 7]


def test_the_logged_registers_are_read_after_gr(bank: Any) -> None:
    """So a motor that answers nothing bails on Gr instead of being charged for three dead reads."""
    fake = bank(matching_chain(), silent=(1,))
    with pytest.raises(RuntimeError):
        mc.verify_motor_types(CHANNEL, YAM_ULTRA_V1_CHAIN)
    assert fake.reads_of(int(DMRegAddr.SW_VER)) == []


def test_every_read_is_logged(bank: Any, caplog: pytest.LogCaptureFixture) -> None:
    """The startup log is the only instrument on a bus nobody can attach a scope to.

    Pinned so a later edit cannot collapse the trail into a bare verdict and leave the next operator
    debugging a refusal with no idea which read answered what.
    """
    bank(matching_chain())
    with caplog.at_level("INFO", logger=mc.logger.name):
        mc.verify_motor_types(CHANNEL, YAM_ULTRA_V1_CHAIN)

    for motor_id, motor_type in YAM_ULTRA_V1_CHAIN:
        for name in ("Gr", "hw_ver", "sw_ver"):
            assert f"{CHANNEL} motor {motor_id} {motor_type}: {name} (@" in caplog.text
    assert "motor type check passed" in caplog.text


def test_a_passive_device_row_is_not_treated_as_a_missing_gear_ratio(bank: Any) -> None:
    """no_gripper and yam_teaching_handle are spelled ``motor_type: ""``, and neither is a motor.

    Without the skip, a gripper-only robot built with a teaching handle -- the one path that can put
    such a row here -- would be told to add "" to the gear-ratio table, which is the wrong file
    entirely.
    """
    fake = bank(matching_chain())
    mc.verify_motor_types(CHANNEL, [[7, ""]])
    assert fake.opened == [], "nothing to check means no socket is opened"
    assert fake.calls == []


def test_a_passive_row_beside_real_motors_only_skips_itself(bank: Any) -> None:
    fake = bank(matching_chain())
    mc.verify_motor_types(CHANNEL, [[1, "DM4340"], [7, ""]])
    assert fake.reads_of(GR) == [1]


def test_an_unrecorded_type_fails_before_the_bus_is_opened(bank: Any) -> None:
    """A code gap must not cost a socket, a bus scan, or an enabled motor to discover."""
    fake = bank(matching_chain())
    with pytest.raises(ValueError, match="No gear ratio recorded"):
        mc.verify_motor_types(CHANNEL, [[1, MotorType.DMH6215MIT]])
    assert fake.opened == [], "the check resolves every expectation before touching the bus"


def test_a_bus_that_will_not_open_is_reported_as_such(monkeypatch: pytest.MonkeyPatch) -> None:
    def _explode(**kwargs: Any) -> None:
        raise OSError("No such device")

    monkeypatch.setattr(mc, "RawCanInterface", _explode)
    with pytest.raises(RuntimeError, match="ip link show"):
        mc.verify_motor_types(CHANNEL, YAM_ULTRA_V1_CHAIN)


def test_the_gear_ratios_the_flow_base_runs_on() -> None:
    """Pinned with literals, because the base's own fixtures derive from these.

    All three were read off a station on 2026-08-18 (``can_flowbase``, motors 1/3/7 steering, 2/4/6/8
    drive, 9 rail); the rail's 9 also reproduces the 2026-08-14 reading on a different base.
    """
    assert MotorType.get_gear_ratio("DM4310V") == 10.0
    assert MotorType.get_gear_ratio("DM_FLOW_WHEEL") == 10.0
    assert MotorType.get_gear_ratio("DM8009") == 9.0


def test_gr_cannot_tell_a_steering_motor_from_a_drive_motor() -> None:
    """A limit of this check, pinned so nobody reads more into a pass than it means.

    Both Flow Base motor types are 10:1, so a caster whose steering and drive motors are swapped passes.
    Catching that is the caster steering check's job, not this one's.
    """
    assert MotorType.get_gear_ratio("DM4310V") == MotorType.get_gear_ratio("DM_FLOW_WHEEL")


# --------------------------------------------------------------------------------------------------
# The call site: DMChainCanInterface.__init__
# --------------------------------------------------------------------------------------------------

CHAIN = [[1, "DM4340"]]
"""A minimal valid chain -- one motor, so the constructor's asserts pass and nothing else is exercised."""


class _SocketOpened(Exception):
    """Raised by the fake DMSingleMotorCanInterface, to stop __init__ the instant it takes the bus.

    Everything the constructor does after that point (motor_on, the reader thread) needs a real bus,
    so the tests below assert on *whether* this is reached rather than mocking the whole chain.
    """


@pytest.fixture
def chain_harness(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, Any]]:
    """Record calls to ``verify_motor_types`` and make opening the socket raise ``_SocketOpened``.

    Patched in ``motor_check``, not ``dm_driver``: the constructor reaches both verifiers through
    ``run_startup_checks``, so ``dm_driver`` no longer holds a name for either.
    """
    from i2rt.motor_drivers import dm_driver

    calls: list[tuple[str, Any]] = []

    def _record(channel: str, motor_list: Any) -> None:
        calls.append((channel, motor_list))

    def _socket(*args: Any, **kwargs: Any) -> None:
        raise _SocketOpened

    monkeypatch.setattr(mc, "verify_motor_types", _record)
    monkeypatch.setattr(dm_driver, "DMSingleMotorCanInterface", _socket)
    return calls


def _build(**kwargs: Any) -> None:
    from i2rt.motor_drivers.dm_driver import DMChainCanInterface

    DMChainCanInterface(CHAIN, [0.0], [1], channel=kwargs.pop("channel", CHANNEL), **kwargs)


def test_the_chain_does_not_check_by_default(chain_harness: list[tuple[str, Any]]) -> None:
    """Both checks are opt-in. The check reads registers and the config half writes Flash, so a chain
    built by code that has not asked for either gets neither."""
    with pytest.raises(_SocketOpened):
        _build()
    assert chain_harness == []


def test_check_motor_types_true_runs_it_on_the_chains_own_motor_list(chain_harness: list[tuple[str, Any]]) -> None:
    """What get_robot and flow_base_controller opt into."""
    with pytest.raises(_SocketOpened):
        _build(check_motor_types=True)
    assert chain_harness == [(CHANNEL, CHAIN)], "the chain's own motor_list must be what gets checked"


def test_the_check_runs_before_the_socket_is_opened(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ordering is the entire constraint: register reads need a bus nothing else is driving.

    If a future edit moves the call below the socket, this fails with ``_SocketOpened`` instead.
    """
    from i2rt.motor_drivers import dm_driver

    def _refuse(channel: str, motor_list: Any) -> None:
        raise RuntimeError("motor type check failed")

    def _socket(*args: Any, **kwargs: Any) -> None:
        raise _SocketOpened

    monkeypatch.setattr(mc, "verify_motor_types", _refuse)
    monkeypatch.setattr(dm_driver, "DMSingleMotorCanInterface", _socket)
    with pytest.raises(RuntimeError, match="motor type check failed"):
        _build(check_motor_types=True)


def test_check_motor_types_false_skips_it(chain_harness: list[tuple[str, Any]]) -> None:
    """The escape hatch for a chain whose declared types are a guess -- see dm_driver's CLI."""
    with pytest.raises(_SocketOpened):
        _build(check_motor_types=False)
    assert chain_harness == []


def test_the_type_check_runs_before_the_config_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Order is load-bearing: on the wrong part the scaling registers hold that part's own scale, so a
    config check that ran first would advise a PMAX rewrite and write CTRL_MODE to Flash on a motor
    whose repair is to be swapped."""
    from i2rt.motor_drivers import dm_driver

    order: list[str] = []
    monkeypatch.setattr(mc, "verify_motor_types", lambda *a, **k: order.append("types"))
    monkeypatch.setattr(mc, "verify_motor_config", lambda *a, **k: order.append("config"))
    monkeypatch.setattr(dm_driver, "DMSingleMotorCanInterface", lambda *a, **k: (_ for _ in ()).throw(_SocketOpened()))
    with pytest.raises(_SocketOpened):
        _build(check_motor_types=True, check_motor_config=True)
    assert order == ["types", "config"]


def test_the_config_check_gets_the_chains_control_mode_and_loop_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """It cannot derive either: the mode decides whether TMAX blocks, the ids which motors can block.

    Also pins the shape of the whole call -- all four of those forwarded positionally, and ``repair``
    as a keyword, since the chain path is the one that may write Flash.
    """
    from i2rt.motor_drivers import dm_driver

    seen: list[tuple[Any, ...]] = []
    monkeypatch.setattr(mc, "verify_motor_types", lambda *a, **k: None)  # the type check is now mandatory
    monkeypatch.setattr(mc, "verify_motor_config", lambda *args, **kwargs: seen.append((args, kwargs)))
    monkeypatch.setattr(dm_driver, "DMSingleMotorCanInterface", lambda *a, **k: (_ for _ in ()).throw(_SocketOpened()))
    with pytest.raises(_SocketOpened):
        _build(check_motor_types=True, check_motor_config=True, control_mode="VEL", loop_critical_motor_ids=(1,))
    assert seen == [((CHANNEL, CHAIN, "VEL", (1,)), {"repair": True})]


def test_the_config_check_cannot_be_asked_for_without_the_type_check() -> None:
    """The ordering is not advice. On the wrong part the scaling registers hold that part's own scale, so
    an unscreened config check reports three consequent mismatches, advises rewriting PMAX, and writes
    CTRL_MODE to Flash on a motor whose repair is to be swapped out. Refused, not merely documented."""
    with pytest.raises(ValueError, match="needs check_motor_types"):
        _build(check_motor_config=True)


def test_a_non_socketcan_channel_refuses_rather_than_starting_unverified(
    chain_harness: list[tuple[str, Any]],
) -> None:
    """Both checks only speak socketcan, so the PCAN branch cannot run them -- and a check that was asked
    for and cannot run must not degrade to a warning. This PR exists to make an unverified chain loud; its
    own inability to run is the last thing that should be quiet.

    That ``_SocketOpened`` is *not* what escapes is the assertion: the refusal precedes the socket.
    """
    with pytest.raises(ValueError, match="not a socketcan channel"):
        _build(channel="PCAN_USBBUS1", check_motor_types=True)
    assert chain_harness == []


def test_an_unchecked_chain_still_builds_on_any_channel(chain_harness: list[tuple[str, Any]]) -> None:
    """The channel guard must fire only on a check that was *asked for*.

    ``DMChainCanInterface``'s channel default is PCAN_USBBUS1 and two callers build chains with no check
    flags at all (examples/single_motor_position_pd_control, scripts/run_gello_with_passive_encoder), so a
    guard placed above the "neither flag" early return would refuse every one of them.
    """
    with pytest.raises(_SocketOpened):
        _build(channel="PCAN_USBBUS1")
    assert chain_harness == []


@pytest.mark.parametrize(
    ("arm_type", "gripper_type"),
    [
        (ArmType.YAM, "linear_4310"),
        (ArmType.YAM_ULTRA_2, "no_gripper"),
        (ArmType.NO_ARM, "linear_4310"),
    ],
)
def test_sim_robots_never_run_either_check(
    monkeypatch: pytest.MonkeyPatch, arm_type: ArmType, gripper_type: str
) -> None:
    """Guards against a future edit building a real chain on get_robot's ``sim`` path.

    The whole test suite is sim-only, so a check that ran in sim would try to open a CAN socket in CI.
    get_robot passes ``check_motor_types=True`` *and* ``check_motor_config=True`` on both real paths, so
    this is a live risk rather than a theoretical one -- and the config check is the worse of the two to
    let slip, since it writes Flash. Patched in ``motor_check``, which is where ``run_startup_checks``
    reaches both from.
    """
    from i2rt.robots import get_robot
    from i2rt.robots.utils import GripperType

    def _explode(which: str) -> Any:
        # **_kwargs so a new keyword on either verifier (verify_motor_config's `repair`) still reaches
        # the AssertionError, rather than a TypeError that passes this test for the wrong reason.
        def _boom(channel: str, *_args: Any, **_kwargs: Any) -> None:
            raise AssertionError(f"sim robot tried to check motor {which} on {channel}")

        return _boom

    monkeypatch.setattr(mc, "verify_motor_types", _explode("types"))
    monkeypatch.setattr(mc, "verify_motor_config", _explode("config"))
    get_robot.get_yam_robot(arm_type=arm_type, gripper_type=GripperType.from_string_name(gripper_type), sim=True)


@pytest.mark.parametrize("gripper_type", ["linear_4310", "no_gripper"])
def test_an_arm_chain_opts_into_both_checks(monkeypatch: pytest.MonkeyPatch, gripper_type: str) -> None:
    """get_robot's real arm path must ask for both, in MIT, with no loop-critical subset named.

    Those three together are what make an arm the strictest caller of the config check: every one of
    PMAX/VMAX/TMAX on every one of its motors can refuse the launch. Asserted at the call site because
    it is a policy decision, not a default -- both flags default to False on the constructor.
    """
    from i2rt.motor_drivers import dm_driver
    from i2rt.robots import get_robot
    from i2rt.robots.utils import GripperType

    seen: dict[str, Any] = {}

    class _Stop(Exception):
        pass

    def _capture(*_args: Any, **kwargs: Any) -> None:
        seen.update(kwargs)
        raise _Stop

    monkeypatch.setattr(get_robot, "DMChainCanInterface", _capture)
    with pytest.raises(_Stop):
        get_robot.get_yam_robot(
            arm_type=ArmType.YAM_ULTRA_2,
            gripper_type=GripperType.from_string_name(gripper_type),
            sim=False,
        )

    assert seen["check_motor_types"] is True
    assert seen["check_motor_config"] is True
    assert "loop_critical_motor_ids" not in seen, "an arm names no subset: every motor is in its loop"
    assert "control_mode" not in seen, f"the chain default is MIT: {dm_driver.ControlMode.MIT}"


def test_a_gripper_only_chain_opts_into_both_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ArmType.NO_ARM path builds its own chain and so has to opt in separately."""
    from i2rt.robots import get_robot
    from i2rt.robots.utils import GripperType

    seen: dict[str, Any] = {}

    class _Stop(Exception):
        pass

    def _capture(*_args: Any, **kwargs: Any) -> None:
        seen.update(kwargs)
        raise _Stop

    monkeypatch.setattr(get_robot, "DMChainCanInterface", _capture)
    with pytest.raises(_Stop):
        get_robot.get_yam_robot(arm_type=ArmType.NO_ARM, gripper_type=GripperType.LINEAR_4310, sim=False)

    assert seen["check_motor_types"] is True
    assert seen["check_motor_config"] is True


# --------------------------------------------------------------------------------------------------
# run_startup_checks: which checks may run, in what order, and whether they may run at all
# --------------------------------------------------------------------------------------------------


def _startup(channel: str = CHANNEL, **kwargs: Any) -> None:
    """``run_startup_checks`` with the two things every call needs, so each test names only its subject."""
    kwargs.setdefault("check_motor_types", False)
    kwargs.setdefault("check_motor_config", False)
    kwargs.setdefault("control_mode", "MIT")
    mc.run_startup_checks(channel, YAM_ULTRA_V1_CHAIN, **kwargs)


def test_the_type_check_runs_first_and_the_config_check_second(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enforced here rather than left to each caller: this is the one place both flags are seen together."""
    order: list[str] = []
    monkeypatch.setattr(mc, "verify_motor_types", lambda *a, **k: order.append("types"))
    monkeypatch.setattr(mc, "verify_motor_config", lambda *a, **k: order.append("config"))

    _startup(check_motor_types=True, check_motor_config=True)
    assert order == ["types", "config"]


def test_asking_for_config_without_types_raises_before_the_bus_is_opened(bank: Any) -> None:
    """A ValueError, not a RuntimeError: decidable from the arguments, and repaired by changing them."""
    fake = bank(matching_chain())
    with pytest.raises(ValueError, match="needs check_motor_types"):
        _startup(check_motor_config=True)
    assert fake.opened == [], "nothing may be read on a call that was refused"


def test_a_check_that_cannot_run_on_this_channel_raises(bank: Any) -> None:
    """Refusing beats warning: the alternative starts the robot with every type and scaling unverified."""
    fake = bank(matching_chain())
    with pytest.raises(ValueError, match="not a socketcan channel"):
        _startup(channel="PCAN_USBBUS1", check_motor_types=True)
    assert fake.opened == []


def test_neither_flag_is_a_no_op_on_any_channel(bank: Any) -> None:
    """The early return precedes the channel guard, which is what keeps an unchecked chain buildable
    anywhere -- including on the PCAN_USBBUS1 default that two callers in the repo rely on."""
    fake = bank(matching_chain())
    _startup(channel="PCAN_USBBUS1")
    _startup()
    assert fake.opened == []


# --------------------------------------------------------------------------------------------------
# The dm_driver CLI: what a chain gets declared as, and the read-only survey
# --------------------------------------------------------------------------------------------------


def test_the_cli_default_types_are_the_arm_they_claim_to_be() -> None:
    """``_CLI_DEFAULT_MOTOR_TYPES`` is a hand copy of yam_v1.yml plus a 4310 gripper, and nothing pinned it.

    If that config ever changes, the CLI's default silently becomes a chain the type check refuses -- the
    exact failure mode it replaced the old single ``--motor-type DM4310`` default to eliminate.
    """
    from i2rt.motor_drivers.dm_driver import _CLI_DEFAULT_MOTOR_TYPES
    from i2rt.robots.utils import GripperType

    arm = {int(motor_id): str(motor_type) for motor_id, motor_type in _load_arm_config(ArmType.YAM).motor_list}
    assert {i: _CLI_DEFAULT_MOTOR_TYPES[i] for i in arm} == arm
    assert _CLI_DEFAULT_MOTOR_TYPES[0x07] == GripperType.LINEAR_4310.get_motor_type(ArmType.YAM)
    assert set(_CLI_DEFAULT_MOTOR_TYPES) == set(arm) | {0x07}, "no id may be defaulted from nowhere"


def test_the_cli_maps_one_type_per_id_in_order() -> None:
    """The whole point of the list form: a real chain is mixed, so one type cannot describe one."""
    from i2rt.motor_drivers.dm_driver import _cli_motor_list

    # A big_yam: DM6248 at joints 1-2. Impossible to survey before this took a list.
    assert _cli_motor_list((1, 2, 3), ("DM6248", "DM6248", "DM4340")) == [
        [1, "DM6248"],
        [2, "DM6248"],
        [3, "DM4340"],
    ]


def test_the_cli_broadcasts_a_single_type_and_defaults_per_id() -> None:
    from i2rt.motor_drivers.dm_driver import _cli_motor_list

    assert _cli_motor_list((1, 2), ("DM4340",)) == [[1, "DM4340"], [2, "DM4340"]]
    assert _cli_motor_list((1, 4), None) == [[1, "DM4340"], [4, "DM4310"]]


@pytest.mark.parametrize("types", [("DM4340", "DM4310"), ()])
def test_the_cli_refuses_a_type_list_that_does_not_match_the_ids(types: tuple[str, ...]) -> None:
    """A dropped or extra id is the common mistake, so the message names both lists, not just the counts."""
    from i2rt.motor_drivers.dm_driver import _cli_motor_list

    with pytest.raises(SystemExit, match="one type per id"):
        _cli_motor_list((1, 2, 3), types)


def test_the_cli_will_not_guess_a_control_mode_for_the_check_that_writes_flash() -> None:
    """MIT and VEL are both wrong on the other machine, and the config check saves this value to Flash."""
    from i2rt.motor_drivers.dm_driver import _cli_control_mode

    with pytest.raises(SystemExit, match="needs --control-mode"):
        _cli_control_mode(None, check_motor_config=True)
    assert _cli_control_mode(None, check_motor_config=False) == "MIT"
    assert _cli_control_mode("VEL", check_motor_config=True) == "VEL"


def test_survey_only_checks_the_chain_without_building_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """The read-only survey the rollout plan needs. Without it the tool goes on to enable every motor."""
    from i2rt.motor_drivers import dm_driver

    seen: list[tuple[Any, ...]] = []
    monkeypatch.setattr(dm_driver, "run_startup_checks", lambda *args, **kwargs: seen.append((args, kwargs)))
    monkeypatch.setattr(dm_driver, "DMChainCanInterface", lambda *a, **k: (_ for _ in ()).throw(_SocketOpened()))

    dm_driver.main(channel=CHANNEL, motor_id=(1,), motor_type=("DM4340",), check_motor_types=True, survey_only=True)

    assert seen == [
        (
            (CHANNEL, [[1, "DM4340"]]),
            {"check_motor_types": True, "check_motor_config": False, "control_mode": "MIT", "repair": False},
        )
    ]


def test_only_the_survey_path_declines_to_repair(monkeypatch: pytest.MonkeyPatch) -> None:
    """The two call sites differ on exactly one argument, and that argument is the whole fix.

    ``--survey-only`` used to skip building the chain while still Flash-saving ``CTRL_MODE``, so the arm
    survey line pointed at a Flow Base rewrote all eight motors out of speed mode -- from a command
    documented as read-only. Asserted as a pair because neither value means anything on its own.
    """
    from i2rt.motor_drivers import dm_driver

    # One patch covers both call sites: the constructor and main() reach the same module-level name.
    # DMChainCanInterface is deliberately *not* patched -- the second half needs the real constructor,
    # and the survey returns before it anyway (pinned next door by test_survey_only_checks_the_chain...).
    seen: list[bool] = []
    monkeypatch.setattr(dm_driver, "run_startup_checks", lambda *_a, **kwargs: seen.append(kwargs["repair"]))
    monkeypatch.setattr(dm_driver, "DMSingleMotorCanInterface", lambda *a, **k: (_ for _ in ()).throw(_SocketOpened()))

    dm_driver.main(channel=CHANNEL, motor_id=(1,), motor_type=("DM4340",), check_motor_types=True, survey_only=True)
    with pytest.raises(_SocketOpened):
        _build(check_motor_types=True, check_motor_config=True)

    assert seen == [False, True], "the survey must not write; the path about to run the chain still repairs"


def test_survey_only_without_a_check_is_refused() -> None:
    """It would open nothing, read nothing and exit 0 -- the silent no-op these guards exist to remove."""
    from i2rt.motor_drivers import dm_driver

    with pytest.raises(SystemExit, match="would read nothing"):
        dm_driver.main(channel=CHANNEL, motor_id=(1,), motor_type=("DM4340",), survey_only=True)


# --------------------------------------------------------------------------------------------------
# The control modes the driver can actually command
# --------------------------------------------------------------------------------------------------


class _FrameBuilt(Exception):
    """Raised once ``set_control`` has encoded a frame and is about to put it on the bus."""


class _EncodeOnly:
    """Just enough of ``DMSingleMotorCanInterface`` for ``set_control``'s encoding half, and no bus.

    ``set_control`` reads only ``control_mode`` and (via ``_get_frame_id``) ``cmd_idoffset`` off ``self``
    before it branches, so the branch can be exercised unbound. Borrowing the real ``_get_frame_id``
    rather than reimplementing it keeps this stub honest if the frame id ever moves.
    """

    _get_frame_id = DMSingleMotorCanInterface._get_frame_id

    def __init__(self, control_mode: str) -> None:
        self.control_mode = control_mode
        self.cmd_idoffset = ControlMode.get_id_offset(control_mode)

    def _send_message_get_response(self, *_args: Any, **_kwargs: Any) -> None:
        raise _FrameBuilt


def _encode(control_mode: str) -> None:
    DMSingleMotorCanInterface.set_control(_EncodeOnly(control_mode), 1, MotorType.DM4310, 0, 0, 0, 0, 0)


def test_a_mode_set_control_cannot_encode_raises_instead_of_sending_zeros() -> None:
    """``POS_VEL`` had a frame id but no encoder, so it fell through and transmitted ``bytearray(8)``.

    A valid frame commanding nothing, on every motor, every cycle, with no error anywhere -- and
    ``CTRL_MODE_BY_CONTROL_MODE`` maps the mode, so ``--control-mode POS_VEL --check-motor-config``
    would have Flash-saved every motor into a mode this driver cannot command, across power cycles.
    """
    with pytest.raises(ValueError, match="cannot encode a command in POS_VEL"):
        _encode(ControlMode.POS_VEL)


def test_the_cli_offers_only_the_modes_set_control_can_encode() -> None:
    """The CLI's choices and the encoder's branches are one fact written in two places; tie them.

    Each offered mode must reach ``_send_message_get_response`` -- i.e. get past the encoding branch --
    rather than raising the unencodable-mode ``ValueError``.
    """
    import typing

    from i2rt.motor_drivers.dm_driver import main

    literal, _none = typing.get_args(typing.get_type_hints(main)["control_mode"])
    offered = typing.get_args(literal)
    assert offered, "control_mode must stay a Literal, or this guard silently checks nothing"

    for mode in offered:
        with pytest.raises(_FrameBuilt):
            _encode(mode)
