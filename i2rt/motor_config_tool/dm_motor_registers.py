"""Read/write DM motor configuration registers over CAN.

Same protocol as `utils.py`: broadcast arbitration ID 0x7FF; read 0x33, write 0x55,
save-to-Flash 0xAA; registers are indexed by decimal address; the payload is a little-endian
float or uint32 (same as `get_special_message_response`, with the return value in data[4:8]).

Register addresses follow the manual's decimal column (where the manual's 8/9/10 disagree with
the hex column due to a typo, the decimal column is authoritative).

This module owns the authoritative register table; `utils.py`'s `get_special_message_response` /
`write_special_message` / `save_to_memory` are a thin compatibility shim over it. See
`dm_motor_registers.md` for the register reference and the operating procedures.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import IntEnum
from typing import Annotated, Callable, TypeAlias

import can
import tyro

from i2rt.motor_config_tool.utils import (
    RawCanInterface,
    bytes_to_float32,
    bytes_to_uint32,
    float32_to_bytes,
    uint32_to_bytes,
)

CAN_BROADCAST = 0x7FF
CMD_READ = 0x33
CMD_WRITE = 0x55
CMD_SAVE = 0xAA

Scalar: TypeAlias = int | float

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RegSpec:
    """One register's spec."""

    addr: int
    name: str
    is_float: bool
    rw: bool


# Full table compiled from the manual / figures; addresses outside the manual's sections are omitted.
# fmt: off
_DM_TABLE: list[tuple[int, str, str, str]] = [
    (0, "UV_Value", "float", "rw"),
    (1, "KT_Value", "float", "rw"),
    (2, "OT_Value", "float", "rw"),
    (3, "OC_Value", "float", "rw"),
    (4, "ACC", "float", "rw"),
    (5, "DEC", "float", "rw"),
    (6, "MAX_SPD", "float", "rw"),
    (7, "MST_ID", "uint32", "rw"),
    (8, "ESC_ID", "uint32", "rw"),
    (9, "TIMEOUT", "uint32", "rw"),
    (10, "CTRL_MODE", "uint32", "rw"),
    (11, "Damp", "float", "ro"),
    (12, "Inertia", "float", "ro"),
    (13, "hw_ver", "uint32", "ro"),
    (14, "sw_ver", "uint32", "ro"),
    (15, "SN", "uint32", "ro"),
    (16, "NPP", "uint32", "ro"),
    (17, "Rs", "float", "ro"),
    (18, "Ls", "float", "ro"),
    (19, "Flux", "float", "ro"),
    (20, "Gr", "float", "ro"),
    (21, "PMAX", "float", "rw"),
    (22, "VMAX", "float", "rw"),
    (23, "TMAX", "float", "rw"),
    (24, "I_BW", "float", "rw"),
    (25, "KP_ASR", "float", "rw"),
    (26, "KI_ASR", "float", "rw"),
    (27, "KP_APR", "float", "rw"),
    (28, "KI_APR", "float", "rw"),
    (29, "OV_Value", "float", "rw"),
    (30, "GREF", "float", "rw"),
    (31, "Deta", "float", "rw"),
    (32, "V_BW", "float", "rw"),
    (33, "IQ_c1", "float", "rw"),
    (34, "VL_c1", "float", "rw"),
    (35, "can_br", "uint32", "rw"),
    (36, "sub_ver", "uint32", "ro"),
    (50, "u_off", "float", "ro"),
    (51, "v_off", "float", "ro"),
    (52, "k1", "float", "ro"),
    (53, "k2", "float", "ro"),
    (54, "m_off", "float", "ro"),
    (55, "dir", "float", "ro"),
    (80, "p_m", "float", "ro"),
    (81, "xout", "float", "ro"),
]
# fmt: on

# What each register is for, keyed by address. Summarized from the DM manual; `dm_motor_registers.md`'s
# table mirrors this, and `_check_tables` below makes "every register is documented" structural rather
# than a habit.
#
# Kept as a separate map rather than a fifth column of _DM_TABLE so that the tuple shape every existing
# caller unpacks stays as it is, and so the `list-registers` table keeps its current width.
#
# Registers described as *calibration* or *advanced tuning* are factory-set: they are shown so an
# operator can read and compare them, not because they are meant to be edited.
REG_MEANING: dict[int, str] = {
    0: "Under-voltage protection threshold (V)",
    1: "Torque constant Kt (N·m/A)",
    2: "Over-temperature protection threshold (°C)",
    3: "Over-current protection threshold (A)",
    4: "Acceleration limit (position/speed modes)",
    5: "Deceleration limit (position/speed modes)",
    6: "Maximum speed limit",
    7: "Master/feedback CAN ID — the ID the motor sends feedback frames with; this driver expects ESC_ID + 16",
    8: "The motor's own CAN (drive) ID",
    9: "CAN loss-of-comms timeout (ms); the motor disables itself if idle past it (0 = off, no failsafe)",
    10: "Control mode: 1 MIT, 2 pos-speed, 3 speed, 4 torque-pos",
    11: "Damping coefficient",
    12: "Rotor inertia",
    13: "Hardware version",
    14: "Firmware (software) version",
    15: "Serial number",
    16: "Number of motor pole pairs",
    17: "Phase (stator) resistance (Ω)",
    18: "Phase (stator) inductance (H)",
    19: "Rotor flux linkage (Wb)",
    20: "Gear reduction ratio — the part number's last two digits: 7 on a DM3507, 9 on a DM8009, "
    "10 on a DM4310, 40 on a DM4340, 48 on a DM6248. Read-only, so it is the one register that "
    "identifies a motor; verify_motor_types in i2rt/motor_drivers/motor_check.py compares it",
    21: "Max position; the ±range used to encode position in MIT mode (rad). Must match POSITION_MAX in "
    "MotorType.get_motor_constants (i2rt/motor_drivers/utils.py) or every command and reading is rescaled",
    22: "Max velocity; the range used to encode velocity in MIT mode (rad/s). Must match VELOCITY_MAX in "
    "MotorType.get_motor_constants (i2rt/motor_drivers/utils.py) or every command and reading is rescaled",
    23: "Max torque; the range used to encode torque in MIT mode (N·m). Must match TORQUE_MAX in "
    "MotorType.get_motor_constants (i2rt/motor_drivers/utils.py) or every command and reading is rescaled",
    24: "Current-loop bandwidth (Hz)",
    25: "Speed-loop (ASR) proportional gain",
    26: "Speed-loop (ASR) integral gain",
    27: "Position-loop (APR) proportional gain",
    28: "Position-loop (APR) integral gain",
    29: "Over-voltage protection threshold (V)",
    30: "Gear efficiency — gearbox torque de-rating; default 1.0",
    31: "Advanced/vendor tuning constant",
    32: "Velocity-loop filter bandwidth (Hz); advanced tuning",
    33: "q-axis current filter coefficient; advanced tuning",
    34: "Velocity filter coefficient; advanced tuning",
    35: "CAN bus baud-rate setting",
    36: "Firmware sub-version",
    50: "U-phase current-sensor offset (calibration)",
    51: "V-phase current-sensor offset (calibration)",
    52: "Encoder linearization coefficient (calibration)",
    53: "Encoder linearization coefficient (calibration)",
    54: "Encoder mechanical zero offset (calibration)",
    55: "Motor rotation-direction sign",
    80: "Rotor (motor-side) mechanical position — runtime readout",
    81: "Output-shaft position after the gearbox — runtime readout",
}

# Registers where a wrong value costs more than a re-write: the two identity registers (a bad
# ESC_ID/MST_ID makes the motor unreachable or invisible until it is hunted down), the ones that change
# how the motor is driven, and the three MIT scaling constants, which silently rescale every
# position/velocity/torque this library sends. Consulted by the CLI only -- the library functions below
# never prompt, so `set_timeout.py` can still write TIMEOUT as a plain call.
GUARDED_REGISTERS: frozenset[str] = frozenset(
    {"ESC_ID", "MST_ID", "CTRL_MODE", "can_br", "TIMEOUT", "PMAX", "VMAX", "TMAX"}
)

_GUARD_REASON: dict[str, str] = {
    "ESC_ID": "changes the motor's own CAN id; it stops answering on the old one immediately",
    "MST_ID": "changes the id the motor sends feedback with; this library expects ESC_ID + 16",
    "CTRL_MODE": "changes how the motor interprets every command frame",
    "can_br": "changes the CAN baud rate; this tool only speaks 1 Mbit/s, so the motor goes silent",
    "TIMEOUT": "the loss-of-comms failsafe; 0 disables it and a stalled control loop then holds torque",
    "PMAX": "rescales every MIT position command and reading; must match i2rt/motor_drivers/utils.py",
    "VMAX": "rescales every MIT velocity command and reading; must match i2rt/motor_drivers/utils.py",
    "TMAX": "rescales every MIT torque command and reading; must match i2rt/motor_drivers/utils.py",
}


def _build_maps() -> tuple[dict[str, RegSpec], dict[int, RegSpec]]:
    by_name: dict[str, RegSpec] = {}
    by_addr: dict[int, RegSpec] = {}
    for addr, name, kind, acc in _DM_TABLE:
        spec = RegSpec(addr=addr, name=name, is_float=kind == "float", rw=acc == "rw")
        if name in by_name:
            raise ValueError(f"duplicate register name {name}")
        if addr in by_addr:
            raise ValueError(f"duplicate register addr {addr}")
        by_name[name] = spec
        by_addr[addr] = spec
    return by_name, by_addr


REG_BY_NAME, REG_BY_ADDR = _build_maps()


def _check_tables() -> None:
    """Fail at import if the side tables and the register table have drifted apart.

    Two invariants, both cheap to break by hand and expensive to notice later: every register carries a
    meaning, and every guarded register is one that actually exists and can actually be written. A typo
    in GUARDED_REGISTERS would otherwise leave the confirmation prompt silently never firing.
    """
    missing = [f"{name} (@{addr})" for addr, name, _, _ in _DM_TABLE if not REG_MEANING.get(addr)]
    if missing:
        raise ValueError(f"REG_MEANING is missing {', '.join(missing)}")
    unknown = sorted(set(REG_MEANING) - set(REG_BY_ADDR))
    if unknown:
        raise ValueError(f"REG_MEANING documents addresses that are not registers: {unknown}")
    for name in sorted(GUARDED_REGISTERS):
        if name not in REG_BY_NAME:
            raise ValueError(f"GUARDED_REGISTERS names {name}, which is not a register")
        if not REG_BY_NAME[name].rw:
            raise ValueError(f"GUARDED_REGISTERS names {name}, which is read-only and can never be written")
        if name not in _GUARD_REASON:
            raise ValueError(f"GUARDED_REGISTERS names {name}, which has no _GUARD_REASON entry")


_check_tables()


class DMRegAddr(IntEnum):
    """Decimal-address enum (matches the manual's table; convenient for code references)."""

    UV_VALUE = 0
    KT_VALUE = 1
    OT_VALUE = 2
    OC_VALUE = 3
    ACC = 4
    DEC = 5
    MAX_SPD = 6
    MST_ID = 7
    ESC_ID = 8
    TIMEOUT = 9
    CTRL_MODE = 10
    DAMP = 11
    INERTIA = 12
    HW_VER = 13
    SW_VER = 14
    SN = 15
    NPP = 16
    RS = 17
    LS = 18
    FLUX = 19
    GR = 20
    PMAX = 21
    VMAX = 22
    TMAX = 23
    I_BW = 24
    KP_ASR = 25
    KI_ASR = 26
    KP_APR = 27
    KI_APR = 28
    OV_VALUE = 29
    GREF = 30
    DETA = 31
    V_BW = 32
    IQ_C1 = 33
    VL_C1 = 34
    CAN_BR = 35
    SUB_VER = 36
    U_OFF = 50
    V_OFF = 51
    K1 = 52
    K2 = 53
    M_OFF = 54
    DIR = 55
    P_M = 80
    XOUT = 81


def _resolve_spec(reg: str | int | DMRegAddr) -> RegSpec:
    """Look a register up by name, decimal address, decimal-address string, or enum member."""
    if isinstance(reg, str) and reg.isdigit():
        # No register name is all digits, so a digit string is unambiguously an address. Accepting it
        # here is what lets the CLI pass its argument through untouched.
        reg = int(reg)
    if isinstance(reg, int):  # DMRegAddr is an IntEnum, so this covers it too
        if int(reg) not in REG_BY_ADDR:
            raise KeyError(f"unknown register address {int(reg)}")
        return REG_BY_ADDR[int(reg)]
    if reg not in REG_BY_NAME:
        raise KeyError(f"unknown register name {reg!r}")
    return REG_BY_NAME[reg]


def _value_to_bytes(spec: RegSpec, value: Scalar) -> bytearray:
    if spec.is_float:
        if not isinstance(value, (int, float)):
            raise TypeError(f"{spec.name} expects int/float")
        if not math.isfinite(float(value)):
            # NaN/inf are ordinary Python floats, so nothing upstream rejects them, and they pack to a
            # perfectly legal float32. The read-back verify would not catch it either: isclose(inf, inf)
            # is True, so the write "succeeds" and the value is then saved to Flash. That matters most
            # on the protection thresholds (OC/OT/UV/OV), where every firmware comparison against NaN is
            # false -- i.e. the trip that register exists to arm may simply never fire.
            raise ValueError(f"{spec.name} cannot be set to {value}: only finite values can be written")
        return float32_to_bytes(float(value))
    if not isinstance(value, int) or value < 0 or value > 0xFFFFFFFF:
        raise TypeError(f"{spec.name} expects uint32")
    return uint32_to_bytes(value)


def _data_to_value(spec: RegSpec, data: bytearray) -> Scalar:
    if spec.is_float:
        return bytes_to_float32(data)
    return bytes_to_uint32(data)


def value_fault(spec: RegSpec, value: Scalar) -> str | None:
    """Why this register's answer is not a usable number, or ``None`` when it is fine.

    Every 4-byte pattern is a valid float32, so a float register that never held a real value still
    decodes to *something*: an erased Flash cell reads 0xFFFFFFFF, which is NaN. Older DM firmware
    answers that for much of the table (see ``dm_motor_registers.md``), so this is a routine reading,
    not a corrupt bus -- the motor replied, and its reply echoed the request.

    Deliberately a predicate and not an exception in ``_data_to_value``: that decode also runs inside
    the read-before-write a confirmation prompt does, so raising there would make a register holding
    garbage impossible to *repair* -- and 21 of the 24 writable registers are floats. The caller decides
    what a faulty value means, which differs per path: a read reports it, a write ignores the old value
    and proceeds, and a verify treats it as "did not match".
    """
    if not spec.is_float or math.isfinite(float(value)):
        return None
    return (
        f"{spec.name} answered {value}, which is not a number: an unwritten Flash cell reads back as "
        f"0xFFFFFFFF, which decodes to NaN. Reported as unread rather than as a value."
    )


def _tx_rx(
    can_if: RawCanInterface,
    motor_id: int,
    payload: list[int],
    *,
    max_retry: int = 20,
) -> can.Message:
    """Send one register frame and return the motor's reply.

    Reads are checked against the request's echo (``data[0]`` = motor id, ``data[3]`` = register
    address) because ``RawCanInterface`` returns the first frame on the bus and filters on nothing: a
    late reply from the previous transaction, or another motor's feedback frame, otherwise decodes as
    this register's value — silently, and in a bulk read that shifts every later register by one. The
    *echo* half is deliberately skipped for writes: an ``ESC_ID`` write is answered by the motor's *new*
    id (it adopts it as the write lands), so rejecting that reply would turn a successful renumber into a
    spurious failure.

    The **length** half applies to every command, and is checked separately for that reason. It is what
    keeps a short frame away from ``struct.unpack("<f", data[4:8])``, which raises ``struct.error`` — not
    a member of ``BUS_ERRORS``, so it would escape every caller's handler as a raw traceback rather than
    the tailored message that caller has for a failed exchange. Only ``CMD_SAVE`` legitimately answers
    short, and ``save_register_to_flash`` never decodes its reply.

    This is why the module does not route through ``CanInterface._send_message_get_response`` with
    ``expected_id=0x7FF``: every register reply from every motor shares that one arbitration id, so
    filtering on it can distinguish neither motor from motor nor register from register. The payload
    echo check above does both.
    """
    for attempt in range(max_retry):
        try:
            can_if.try_receive_message(motor_id)
            frame = can.Message(arbitration_id=CAN_BROADCAST, data=bytearray(payload), is_extended_id=False)
            can_if.bus.send(frame)
            # try_receive_message returns None on timeout, which is just another attempt to burn.
            message = can_if.try_receive_message(motor_id)
            if message is not None:
                data = message.data
                # Length and echo are separate conditions: every command needs a decodable frame, but
                # only a read needs the frame to be *this* transaction's.
                # The length test stays inside echoes_request too: it is what stops data[0] on a
                # zero-length frame, and IndexError is no more a BUS_ERROR than struct.error is.
                well_formed = len(data) == 8 or payload[2] == CMD_SAVE
                echoes_request = len(data) == 8 and data[0] == payload[0] and data[3] == payload[3]
                if well_formed and (echoes_request or payload[2] != CMD_READ):
                    return message
                log.debug("dm reg attempt %s: dropped foreign frame %s", attempt, list(data))
        except can.CanError as e:
            log.debug("dm reg tx attempt %s failed: %s", attempt, e)
            can_if.try_receive_message(motor_id)
        time.sleep(0.002)
    raise RuntimeError(f"Failed CAN exchange for motor {motor_id}")


def read_register(can_if: RawCanInterface, motor_id: int, reg: str | int | DMRegAddr) -> Scalar:
    """Read one register's current value from RAM."""
    spec = _resolve_spec(reg)
    data = [motor_id, 0x00, CMD_READ, spec.addr, 0x00, 0x00, 0x00, 0x00]
    message = _tx_rx(can_if, motor_id, data)
    return _data_to_value(spec, bytearray(message.data))


def write_register(can_if: RawCanInterface, motor_id: int, reg: str | int | DMRegAddr, value: Scalar) -> Scalar:
    """Write one register in RAM and return the motor's echoed value.

    RAM only — follow with ``save_register_to_flash`` to persist it. Never prompts; the CLI's
    confirmation gate for the guarded registers lives in ``_confirm``, so library callers such as
    ``utils.write_special_message`` are not blocked on a console.
    """
    spec = _resolve_spec(reg)
    if not spec.rw:
        raise ValueError(f"register {spec.name} (addr {spec.addr}) is read-only")
    vb = _value_to_bytes(spec, value)
    data = [motor_id, 0x00, CMD_WRITE, spec.addr, vb[0], vb[1], vb[2], vb[3]]
    message = _tx_rx(can_if, motor_id, data)
    return _data_to_value(spec, bytearray(message.data))


def save_register_to_flash(can_if: RawCanInterface, motor_id: int, reg: str | int | DMRegAddr) -> can.Message:
    """Persist one register's current value to Flash.

    Returns the raw reply rather than a decoded value because a 0xAA reply is a *short* frame (fewer
    than 8 bytes), unlike the 8-byte echo a read or write returns — there is no data[4:8] to decode.

    Only ever save a register you have just successfully written; see ``dm_motor_registers.md``.
    Read-only registers are refused outright, which blocks the one case observed to make a save revert
    *other* unsaved registers to their stored values.
    """
    spec = _resolve_spec(reg)
    if not spec.rw:
        raise ValueError(f"cannot save read-only register {spec.name}")
    data = [motor_id, 0x00, CMD_SAVE, spec.addr, 0x00, 0x00, 0x00, 0x00]
    return _tx_rx(can_if, motor_id, data)


CTRL_MODE_ID_TO_NAME: dict[int, str] = {
    0: "unknown_0",
    1: "MIT",
    2: "pos_speed",
    3: "speed",
    4: "torque_pos",
}

_NAME_W = max(len(name) for _, name, _, _ in _DM_TABLE)


ID_REGISTERS: frozenset[int] = frozenset({int(DMRegAddr.MST_ID), int(DMRegAddr.ESC_ID)})
"""CAN identifiers, shown in hex alongside the decimal. Everything that talks about these on the wire --
the DM host tool, ``candump``, this driver's own ``0x200 + id`` arbitration -- uses hex, while the CLI's
``--motor-id`` takes decimal, so a dump that gives only one of the two forces the reader to convert."""


def format_value(spec: RegSpec, value: Scalar) -> str:
    """One register value as a display string.

    Floats go through ``.6g`` because a float32 widened to a Python float reads as e.g.
    ``12.100000381469727``, and 17 digits of round-trip noise makes a whole-table dump unreadable.
    """
    if spec.is_float:
        return f"{value:.6g}"
    if spec.addr == DMRegAddr.CTRL_MODE:
        return f"{value} ({CTRL_MODE_ID_TO_NAME.get(int(value), 'unknown')})"
    if spec.addr in ID_REGISTERS:
        return f"{value} (0x{int(value):02X})"
    return str(value)


def _format_register_table(values: dict[int, str] | None = None) -> str:
    """Aligned list of every known register, built from _DM_TABLE.

    ``values`` maps register address to an already-formatted value (see ``format_value``); when given,
    a ``value`` column is appended. One function serves both the ``list-registers`` listing and the
    ``read-all`` dump so the two cannot drift apart in column order or naming. The per-line ``rstrip``
    is what lets them share one format string: without ``values`` the padded ``access`` column becomes
    the last one again and the output is byte-identical to the value-less table.
    """
    value_header, value_rule = ("value", "-" * 5) if values else ("", "")
    header = f"  {'addr':>4}  {'name':<{_NAME_W}}  {'type':<6}  {'access':<6}  {value_header}"
    rule = f"  {'-' * 4}  {'-' * _NAME_W}  {'-' * 6}  {'-' * 6}  {value_rule}"
    rows = [
        f"  {addr:>4}  {name:<{_NAME_W}}  {kind:<6}  {acc:<6}  {values.get(addr, '?') if values else ''}"
        for addr, name, kind, acc in _DM_TABLE
    ]
    return "\n".join(line.rstrip() for line in [header, rule, *rows])


# Transport failures worth retrying or reporting per register; a usage error (bad value, read-only
# register) is not in here, so it surfaces immediately instead of being disguised as a comms problem.
# Public because it is part of the contract for callers outside this module that drive the register
# functions themselves -- see i2rt/motor_drivers/motor_check.py.
BUS_ERRORS = (RuntimeError, can.CanError, OSError, AssertionError)


@contextmanager
def _open_bus(channel: str) -> Iterator[RawCanInterface]:
    """Open ``channel`` as a SocketCAN ``RawCanInterface`` and shut it down on the way out."""
    iface = RawCanInterface(channel=channel, bustype="socketcan", name="dm_motor_registers")
    try:
        yield iface
    finally:
        iface.close()


def _current_value_line(iface: RawCanInterface, motor_id: int, spec: RegSpec, new: Scalar | None = None) -> str:
    """The register's value right now, for the confirmation prompt.

    A read failure is reported rather than raised: the prompt exists to show the operator what they are
    about to overwrite, and a register that will not read is still one they may need to repair.
    """
    try:
        current = format_value(spec, read_register(iface, motor_id, spec.addr))
    except BUS_ERRORS as e:
        current = f"<unreadable: {e}>"
    return f"current value: {current}" + (f"  ->  requested: {new}" if new is not None else "")


def _resolve_cli(reg: str) -> RegSpec:
    """Resolve a register argument, reporting a typo as a usage error rather than a traceback.

    A mistyped name is the commonest operator error, and it happens before the bus is opened; the
    library keeps raising KeyError for callers that want to handle it.
    """
    try:
        return _resolve_spec(reg)
    except KeyError as e:
        raise SystemExit(f"{e.args[0]}; run list-registers to see them all") from None


def _confirm(action: str, spec: RegSpec, motor_id: int, channel: str, detail: str) -> None:
    """Ask the operator to confirm a guarded write/save; abort unless they answer yes.

    CLI layer only. ``write_register`` and ``save_register_to_flash`` never prompt, so library callers --
    ``utils.write_special_message``, which ``set_timeout.py`` uses to write TIMEOUT -- are not blocked on
    a console. Reads are never gated.
    """
    if spec.name not in GUARDED_REGISTERS:
        return
    print(f"\n{spec.name} (@{spec.addr}) {_GUARD_REASON[spec.name]}.")
    print(f"  motor {motor_id} on {channel}")
    print(f"  {detail}")
    try:
        answer = input(f"{action} {spec.name}? [y/N] ").strip().lower()
    except EOFError:
        # Nothing on stdin and no terminal: fail closed and say why, rather than exiting on an unhandled
        # traceback that reads like a crash mid-transaction. Pipe an answer in to script this:
        #   echo y | python i2rt/motor_config_tool/dm_motor_registers.py write ESC_ID --value 5 ...
        raise SystemExit(f"aborted: {spec.name} needs confirmation and stdin is empty") from None
    if answer not in ("y", "yes"):
        raise SystemExit("aborted")


def read(reg: Annotated[str, tyro.conf.Positional], motor_id: int, channel: str = "can0") -> None:
    """Read one register and print its value.

    Args:
        reg: Register name (e.g. sw_ver) or decimal address (e.g. 14); see list-registers.
        motor_id: CAN id of the motor.
        channel: SocketCAN channel the motor chain is on.
    """
    spec = _resolve_cli(reg)
    with _open_bus(channel) as iface:
        value = read_register(iface, motor_id, spec.addr)
    fault = value_fault(spec, value)
    if fault:
        log.warning("%s", fault)
    print(f"{spec.name} (@{spec.addr}): {value}")


def read_all(motor_id: int, channel: str = "can0") -> None:
    """Read every register and print one table. A register that fails shows ERR and the dump continues.

    Args:
        motor_id: CAN id of the motor.
        channel: SocketCAN channel the motor chain is on.
    """
    values: dict[int, str] = {}
    with _open_bus(channel) as iface:
        for addr, name, _, _ in _DM_TABLE:
            try:
                values[addr] = format_value(REG_BY_ADDR[addr], read_register(iface, motor_id, addr))
            except BUS_ERRORS as e:
                # A failure on the very first register means the motor is not answering at all, and
                # every dead register costs ~0.65 s of retries — bail instead of spending ~28 s to
                # print a table of nothing but ERR. Later failures are that one register's problem.
                if not values:
                    raise RuntimeError(
                        f"motor {motor_id} did not answer on {channel}; check the id and power, and "
                        "stop anything else using the bus"
                    ) from e
                log.warning("read %s (@%d) failed: %s", name, addr, e)
                values[addr] = "ERR"
    # Table on stdout, per-register warnings on stderr, so `read-all > dump.txt` stays clean.
    print(f"motor {motor_id} on {channel}:")
    print(_format_register_table(values))


def write(
    reg: Annotated[str, tyro.conf.Positional],
    value: int | float,
    motor_id: int,
    channel: str = "can0",
) -> None:
    """Write a register in RAM and print the read-back. Follow with save to persist it.

    Args:
        reg: Register name (e.g. MAX_SPD) or decimal address (e.g. 6); see list-registers.
        value: New value. Read as an integer when typed without a "." or an exponent, otherwise as a
            float; uint32 registers only accept the integer form.
        motor_id: CAN id of the motor.
        channel: SocketCAN channel the motor chain is on.
    """
    spec = _resolve_cli(reg)
    if not spec.rw:
        raise SystemExit(f"register {spec.name} (addr {spec.addr}) is read-only")
    with _open_bus(channel) as iface:
        _confirm("write", spec, motor_id, channel, _current_value_line(iface, motor_id, spec, value))
        echo = write_register(iface, motor_id, spec.addr, value)
    print(f"write ok, readback {spec.name}: {echo}")


def save(reg: Annotated[str, tyro.conf.Positional], motor_id: int, channel: str = "can0") -> None:
    """Persist a register's current value to Flash. Only save a register you have just written.

    Args:
        reg: Register name (e.g. MAX_SPD) or decimal address (e.g. 6); see list-registers.
        motor_id: CAN id of the motor.
        channel: SocketCAN channel the motor chain is on.
    """
    spec = _resolve_cli(reg)
    if not spec.rw:
        raise SystemExit(f"cannot save read-only register {spec.name}")
    with _open_bus(channel) as iface:
        detail = f"{_current_value_line(iface, motor_id, spec)}  (save commits this to Flash permanently)"
        _confirm("save", spec, motor_id, channel, detail)
        resp = save_register_to_flash(iface, motor_id, spec.addr)
    print(f"save response arbitration_id=0x{resp.arbitration_id:x} data={list(resp.data)}")


def list_registers() -> None:
    """Print every register this tool knows about, with meanings. Needs no CAN bus."""
    print("registers (pass the name or the decimal address):\n")
    print(_format_register_table())
    print("\nmeanings:\n")
    for addr, name, _, _ in _DM_TABLE:
        print(f"  {addr:>4}  {name:<{_NAME_W}}  {REG_MEANING[addr]}")
    print(f"\nwriting or saving these asks for confirmation: {', '.join(sorted(GUARDED_REGISTERS))}")


# Hoisted to module level so the tests can drive the real CLI with `args=[...]`, which type-checks every
# signature above without touching hardware.
_SUBCOMMANDS: dict[str, Callable[..., None]] = {
    "read": read,
    "read-all": read_all,
    "write": write,
    "save": save,
    "list-registers": list_registers,
}


# Unlike the other scripts in this directory, this module is imported by `utils.py`, so the CLI must not
# run on import.
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.extras.subcommand_cli_from_dict(_SUBCOMMANDS)
