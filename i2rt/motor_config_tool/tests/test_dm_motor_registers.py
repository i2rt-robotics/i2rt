"""Unit tests for the DM register table, wire encoding, and CLI wiring.

Almost everything here is pure — no CAN bus and no mocks of one. The bus-facing functions
(`read_register` / `write_register` / `save_register_to_flash`) are covered where they reject input
*before* touching the bus, which is exactly the behaviour worth pinning: a read-only write must never
reach the hardware. The one exception is `_FakeCan`, which exists to pin which *replies* `_tx_rx`
accepts — a decision that has no other observable surface.
"""

from __future__ import annotations

import math

import can
import pytest
import tyro

from i2rt.motor_config_tool import utils
from i2rt.motor_config_tool.dm_motor_registers import (
    _DM_TABLE,
    _GUARD_REASON,
    _NAME_W,
    _SUBCOMMANDS,
    CAN_BROADCAST,
    CMD_SAVE,
    CMD_WRITE,
    CTRL_MODE_ID_TO_NAME,
    GUARDED_REGISTERS,
    REG_BY_ADDR,
    REG_BY_NAME,
    REG_MEANING,
    DMRegAddr,
    RegSpec,
    _check_tables,
    _confirm,
    _data_to_value,
    _format_register_table,
    _resolve_cli,
    _resolve_spec,
    _value_to_bytes,
    format_value,
    save_register_to_flash,
    value_fault,
    write_register,
)

RO_REGISTERS = [name for _, name, _, acc in _DM_TABLE if acc == "ro"]
FLOAT_SPEC = REG_BY_NAME["MAX_SPD"]
UINT_SPEC = REG_BY_NAME["MST_ID"]


class _FakeCan:
    """Minimal ``RawCanInterface`` stand-in: a scripted reply queue and a record of what was sent.

    ``_tx_rx`` calls ``try_receive_message`` twice per attempt -- once before the send to drain the bus,
    once after to collect the reply -- so only post-send calls consume from the queue. A queued ``None``
    models a receive timeout.
    """

    def __init__(self, replies: list[bytearray | None]) -> None:
        self.replies = list(replies)
        self.sent: list[list[int]] = []
        self.bus = self
        self._awaiting_reply = False

    def send(self, frame: can.Message) -> None:
        self.sent.append(list(frame.data))
        self._awaiting_reply = True

    def try_receive_message(self, motor_id: int) -> can.Message | None:
        if not self._awaiting_reply:
            return None  # the pre-send drain
        self._awaiting_reply = False
        data = self.replies.pop(0) if self.replies else None
        return None if data is None else can.Message(arbitration_id=CAN_BROADCAST, data=data, is_extended_id=False)


# --------------------------------------------------------------------------------------------------
# Table integrity
# --------------------------------------------------------------------------------------------------


def test_table_has_no_duplicate_names_or_addresses() -> None:
    assert len(REG_BY_NAME) == len(_DM_TABLE)
    assert len(REG_BY_ADDR) == len(_DM_TABLE)


def test_every_register_is_documented() -> None:
    _check_tables()
    assert set(REG_MEANING) == {addr for addr, _, _, _ in _DM_TABLE}


def test_check_tables_rejects_a_missing_meaning(monkeypatch: pytest.MonkeyPatch) -> None:
    trimmed = {k: v for k, v in REG_MEANING.items() if k != DMRegAddr.SW_VER}
    monkeypatch.setattr("i2rt.motor_config_tool.dm_motor_registers.REG_MEANING", trimmed)
    with pytest.raises(ValueError, match="sw_ver"):
        _check_tables()


def test_check_tables_rejects_an_extra_meaning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("i2rt.motor_config_tool.dm_motor_registers.REG_MEANING", {**REG_MEANING, 999: "nope"})
    with pytest.raises(ValueError, match="999"):
        _check_tables()


def test_register_kinds_and_access_are_known() -> None:
    for _, name, kind, acc in _DM_TABLE:
        assert kind in ("float", "uint32"), name
        assert acc in ("rw", "ro"), name


def test_dm_reg_addr_enum_matches_table() -> None:
    assert {int(member) for member in DMRegAddr} == set(REG_BY_ADDR)


def test_guarded_registers_exist_and_are_writable() -> None:
    for name in GUARDED_REGISTERS:
        assert name in REG_BY_NAME, name
        assert REG_BY_NAME[name].rw, f"{name} is read-only, so the prompt would never fire"
        assert name in _GUARD_REASON, name


# --------------------------------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("reg", ["sw_ver", 14, "14", DMRegAddr.SW_VER])
def test_resolve_spec_accepts_name_address_digit_string_and_enum(reg: str | int) -> None:
    assert _resolve_spec(reg) is REG_BY_NAME["sw_ver"]


@pytest.mark.parametrize("reg", ["nope", 999, "999"])
def test_resolve_spec_rejects_unknown(reg: str | int) -> None:
    with pytest.raises(KeyError):
        _resolve_spec(reg)


def test_resolve_cli_reports_a_typo_as_a_usage_error() -> None:
    # The CLI turns the library's KeyError into a clean exit; a traceback for a mistyped name reads
    # like a crash, and this happens before the bus is opened.
    with pytest.raises(SystemExit, match="list-registers"):
        _resolve_cli("sw_verr")


def test_resolve_cli_resolves_the_same_specs_as_the_library() -> None:
    assert _resolve_cli("sw_ver") is _resolve_spec("sw_ver")
    assert _resolve_cli("14") is _resolve_spec(14)


# --------------------------------------------------------------------------------------------------
# Wire contract. The 4-byte prefix encodes the fact that a register value lives at data[4:8].
# --------------------------------------------------------------------------------------------------


def test_float_register_roundtrips() -> None:
    encoded = bytearray(4) + _value_to_bytes(FLOAT_SPEC, 12.5)
    assert _data_to_value(FLOAT_SPEC, encoded) == 12.5


def test_uint32_register_roundtrips() -> None:
    encoded = bytearray(4) + _value_to_bytes(UINT_SPEC, 21)
    assert _data_to_value(UINT_SPEC, encoded) == 21


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_value_is_refused(value: float) -> None:
    # The load-bearing safety property: nan/inf pack to a legal float32 and would sail through a
    # read-back verify, so a protection threshold could be "written", "verified", saved, and never fire.
    with pytest.raises(ValueError, match="only finite values"):
        _value_to_bytes(REG_BY_NAME["OC_Value"], value)


@pytest.mark.parametrize("value", [5.0, -1, 2**32])
def test_uint32_register_rejects_float_and_out_of_range(value: int | float) -> None:
    with pytest.raises(TypeError, match="expects uint32"):
        _value_to_bytes(UINT_SPEC, value)


def test_value_fault_flags_nan_only() -> None:
    assert "MAX_SPD" in (value_fault(FLOAT_SPEC, math.nan) or "")
    assert value_fault(FLOAT_SPEC, 1.0) is None
    assert value_fault(UINT_SPEC, 0xFFFFFFFF) is None


@pytest.mark.parametrize("name", RO_REGISTERS)
def test_write_register_refuses_read_only_before_touching_the_bus(name: str) -> None:
    # can_if is None on purpose: if the guard ever stopped firing, this would raise AttributeError
    # instead, so the test cannot pass for the wrong reason.
    with pytest.raises(ValueError, match="read-only"):
        write_register(None, 1, name, 0)  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------
# Which replies _tx_rx accepts
# --------------------------------------------------------------------------------------------------


def test_a_short_frame_on_the_write_path_is_dropped_not_decoded() -> None:
    """The echo check is skipped for writes; the *length* check must not be skipped with it.

    ``_tx_rx`` returns the first frame on the bus and filters on nothing, so a stray short frame -- a
    late 0xAA reply, another motor's feedback -- lands on the write path too. Decoding it reaches
    ``struct.unpack("<f", data[4:8])`` and raises ``struct.error``, which is not in ``BUS_ERRORS``, so it
    escapes ``motor_check._repair``'s handler as a raw traceback instead of its tailored "the
    write failed ... nothing was saved to Flash" message.
    """
    spec = REG_BY_ADDR[int(DMRegAddr.PMAX)]
    stray = bytearray([1, 0, 0, 0])  # four bytes: nothing to unpack a float32 out of
    echo = bytearray([1, 0, CMD_WRITE, spec.addr]) + _value_to_bytes(spec, math.pi)
    can_if = _FakeCan([stray, echo])

    assert write_register(can_if, 1, DMRegAddr.PMAX, math.pi) == pytest.approx(math.pi, abs=1e-6)
    assert len(can_if.sent) == 2, "the stray frame must cost a retry, not a decode"


def test_a_save_still_accepts_its_short_reply() -> None:
    """The exemption that makes the length check conditional: 0xAA answers with fewer than 8 bytes.

    ``save_register_to_flash`` returns the raw message and never decodes it, which is why this is safe.
    """
    spec = REG_BY_ADDR[int(DMRegAddr.PMAX)]
    can_if = _FakeCan([bytearray([1, 0, CMD_SAVE, spec.addr])])

    message = save_register_to_flash(can_if, 1, DMRegAddr.PMAX)
    assert len(message.data) == 4
    assert len(can_if.sent) == 1, "a legitimate short save reply must be accepted first time"


def test_a_read_still_requires_this_transactions_own_echo() -> None:
    """The pre-existing behaviour the split must not weaken: a foreign 8-byte frame is still dropped."""
    spec = REG_BY_ADDR[int(DMRegAddr.PMAX)]
    foreign = bytearray([2, 0, 0x33, spec.addr]) + _value_to_bytes(spec, 12.5)  # another motor's id
    mine = bytearray([1, 0, 0x33, spec.addr]) + _value_to_bytes(spec, math.pi)
    can_if = _FakeCan([foreign, mine])

    from i2rt.motor_config_tool.dm_motor_registers import read_register

    assert read_register(can_if, 1, DMRegAddr.PMAX) == pytest.approx(math.pi, abs=1e-6)


def test_format_register_table_lists_every_register() -> None:
    lines = _format_register_table().splitlines()
    assert len(lines) == len(_DM_TABLE) + 2  # header + rule
    for line in lines:
        assert line == line.rstrip(), f"trailing whitespace: {line!r}"
        assert len(line) <= 119


def test_format_register_table_with_values_adds_a_column() -> None:
    lines = _format_register_table({DMRegAddr.SW_VER: "123"}).splitlines()
    sw_ver_row = next(line for line in lines if " sw_ver " in line)
    assert sw_ver_row.endswith("123")
    max_spd_row = next(line for line in lines if " MAX_SPD " in line)
    assert max_spd_row.endswith("?")


def test_format_value_shortens_floats_and_names_ctrl_mode() -> None:
    assert format_value(FLOAT_SPEC, 12.100000381469727) == "12.1"
    assert "MIT" in format_value(REG_BY_NAME["CTRL_MODE"], 1)


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [("ESC_ID", 1, "1 (0x01)"), ("ESC_ID", 8, "8 (0x08)"), ("MST_ID", 17, "17 (0x11)"), ("MST_ID", 24, "24 (0x18)")],
)
def test_can_ids_are_shown_in_hex_as_well_as_decimal(name: str, value: int, expected: str) -> None:
    """Everything that talks about these on the wire uses hex; ``--motor-id`` takes decimal.

    Showing both also makes the driver's ``MST_ID = ESC_ID + 16`` rule readable straight off the dump:
    +16 is +0x10, so the master id is the drive id with the high nibble bumped by one.
    """
    assert format_value(REG_BY_NAME[name], value) == expected


def test_only_the_can_ids_get_the_hex_treatment() -> None:
    # A uint32 that is not an identifier reads as a plain number -- TIMEOUT in ms, can_br as an index.
    assert format_value(REG_BY_NAME["TIMEOUT"], 8000) == "8000"
    assert format_value(REG_BY_NAME["can_br"], 4) == "4"


# --------------------------------------------------------------------------------------------------
# Anti-drift between the canonical table and utils.py's legacy shim
# --------------------------------------------------------------------------------------------------


def test_legacy_register_map_agrees_with_the_table() -> None:
    assert set(utils._LEGACY_REGISTER_ALIASES) == set(utils.register_addr_map)
    for legacy, canonical in utils._LEGACY_REGISTER_ALIASES.items():
        addr, convert_func = utils.register_addr_map[legacy]
        spec = REG_BY_NAME[canonical]
        assert addr == spec.addr, legacy
        assert (convert_func is utils.bytes_to_float32) == spec.is_float, legacy


def test_ctrl_mode_names_match_utils() -> None:
    for mode_id, name in utils.register_info_map["control_mode"].items():
        assert CTRL_MODE_ID_TO_NAME[mode_id] == name


# --------------------------------------------------------------------------------------------------
# The confirmation gate. CLI-only: the library functions above never prompt.
# --------------------------------------------------------------------------------------------------


def _prompt(spec: RegSpec) -> None:
    _confirm("write", spec, motor_id=1, channel="can0", detail="current value: 20  ->  requested: 21")


def test_confirm_accepts_yes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _: "Y ")
    _prompt(REG_BY_NAME["MST_ID"])


@pytest.mark.parametrize("answer", ["", "n", "no", "yes please", "yolo"])
def test_confirm_aborts_on_anything_else(monkeypatch: pytest.MonkeyPatch, answer: str) -> None:
    monkeypatch.setattr("builtins.input", lambda _: answer)
    with pytest.raises(SystemExit):
        _prompt(REG_BY_NAME["MST_ID"])


def test_confirm_treats_closed_stdin_as_abort(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(_: str) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", _raise)
    with pytest.raises(SystemExit, match="MST_ID"):
        _prompt(REG_BY_NAME["MST_ID"])


def test_confirm_does_not_prompt_for_an_unguarded_register(monkeypatch: pytest.MonkeyPatch) -> None:
    # Keeps the routine `write MAX_SPD` / `save MAX_SPD` bench loop non-interactive.
    def _fail(_: str) -> str:
        raise AssertionError("prompted for an unguarded register")

    monkeypatch.setattr("builtins.input", _fail)
    _prompt(REG_BY_NAME["MAX_SPD"])


# --------------------------------------------------------------------------------------------------
# CLI wiring. Drives the real parser, so a signature tyro cannot construct fails here, not at a bench.
# --------------------------------------------------------------------------------------------------


def test_list_registers_needs_no_bus(capsys: pytest.CaptureFixture[str]) -> None:
    tyro.extras.subcommand_cli_from_dict(_SUBCOMMANDS, args=["list-registers"])
    out = capsys.readouterr().out
    for _, name, _, _ in _DM_TABLE:
        assert name in out, name


def test_cli_help_lists_every_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        tyro.extras.subcommand_cli_from_dict(_SUBCOMMANDS, args=["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    for subcommand in _SUBCOMMANDS:
        assert subcommand in out, subcommand


def test_name_column_fits_every_register() -> None:
    assert _NAME_W == max(len(name) for _, name, _, _ in _DM_TABLE)
