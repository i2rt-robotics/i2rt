"""The two startup register checks a motor chain can run before it takes the bus.

One function each, both opt-in on ``DMChainCanInterface``:

* :func:`verify_motor_types` reads ``Gr`` and refuses a chain that does not hold the motor types its
  ``motor_list`` declares. Arms and the Flow Base both run it (``check_motor_types=True``).
* :func:`verify_motor_config` checks ``CTRL_MODE`` -- repairing it -- and compares the ``PMAX``/
  ``VMAX``/``TMAX`` that decide how commands and feedback are scaled. Arms and the Flow Base both run
  it too (``check_motor_config=True``), but under different severity: see ``loop_registers`` and
  ``loop_critical_motor_ids`` below, because an arm is the strictest possible caller of it.

**Neither is called directly.** :func:`run_startup_checks` is the entry point, and it exists because two
rules about *which* checks may run, and in *what order*, have to hold for every caller rather than for
whichever one remembered them:

* **The type check runs first when running both**, and asking for the config check alone is refused
  rather than merely discouraged. A ``Gr`` mismatch means the wrong *part* is fitted, and on the wrong
  part the scaling registers hold that part's own scale; checking type first is what stops
  ``verify_motor_config`` from reporting three consequent mismatches, advising a ``PMAX`` rewrite, and
  writing ``CTRL_MODE`` to Flash on a motor whose repair is to be swapped.
* **A check that was asked for and cannot run refuses**, rather than warning and letting the robot start
  on a chain nobody verified. Both read registers over socketcan and nothing else.

Three constraints shape both, and are why neither is a method on a running chain:

* **They can only run before the chain's socket is open.** ``DMChainCanInterface.__init__`` opens the
  socket, enables every motor and (for some callers) starts its background reader before it returns,
  and ``close()`` neither joins that thread nor allows a restart. Register access needs an idle bus --
  one process, one thread, see ``i2rt/motor_config_tool/dm_motor_registers.md``. So on the path that
  builds a chain, the call sits at the top of that constructor, after its asserts and *before* the branch
  that constructs ``DMSingleMotorCanInterface``: everything above that line is pure Python, which is what
  makes the bus idle there, and there is no later window. The other call site builds no chain at all --
  ``dm_driver.py --survey-only`` runs the checks with ``repair=False`` and returns, which is the only way
  to read a chain's registers without energising it *or* writing to it. Sim robots reach neither --
  :mod:`i2rt.robots.get_robot` returns a
  ``SimRobot`` before building a chain at all -- so the sim-only test suite never touches CAN.
* **A read is retried before it is believed.** ``read_register`` already retries 20x inside ``_tx_rx``
  (~0.65 s). ``verify_motor_types`` retries *that* up to ``MAX_READ_ATTEMPTS`` times, because a refusal
  to start is expensive enough to be worth ~3.8 s of certainty on a chain that will not answer. A NaN
  answer is retried on the same footing as a bus failure: an unwritten Flash cell reads 0xFFFFFFFF,
  which decodes to NaN, and either way the reading is not a gear ratio.
* **Nothing is written unless every motor answered.** A bus we could not read reliably is not a bus to
  commit Flash writes on, and this is also what makes contention safe: a busy bus fails the reads, so
  the checks can never write into another process's traffic. The same rule governs the write phase: the
  first repair that does not stick stops the loop, and the motors behind it are reported as not
  attempted rather than committed over a bus that just dropped one write. And a caller that asked not to
  write at all (``repair=False``, which is what ``--survey-only`` passes) gets the mismatch reported with
  the command that fixes it, so "survey" never means "reconfigured whatever it was pointed at".

Every read is logged as it happens -- successes at INFO, each failed attempt at WARNING -- so an
operator debugging a refusal can see which reads answered and what each said, not only the verdict.

Check 1: motor type (``Gr``)
============================

Whoever builds a chain names a motor type per CAN id and nothing verifies it. ``_motor_on`` proves only
that *a* motor answers at each id; the type string is then trusted for the rest of the session, and
``dm_driver.set_control`` *encodes* every MIT frame with
``MotorType.get_motor_constants(<declared type>)`` while the firmware decodes with its own registers.
A wrong declaration is therefore not a startup error but a constant scale factor on every command,
self-consistent in both directions, with nothing raised anywhere.

``yam_ultra_v2.yml`` is where this bites: it differs from ``yam_ultra_v1.yml`` in exactly one place,
joint 4's ``DM4310`` -> ``DM4340``. Run a v2 arm as ``--arm yam_ultra`` and joint 4's torque is
encoded against ``TORQUE_MAX`` 10 while the motor decodes against 28, so gravity compensation arrives
2.8x too large; run a v1 arm as ``--arm yam_ultra_2`` and it arrives at 0.36x and the arm sags.
Velocity is 3x off either way.

``Gr`` (address 20, gear reduction ratio) is the register that settles it, because it is **read-only**
and describes the physical gearbox. ``PMAX``/``VMAX``/``TMAX`` cannot: they are writable, so they can
be wrong on a correct motor *and* right on a wrong one -- which is why check 2 can never stand in for
this one, and why this one writes nothing. The repair for a mismatch is to correct the types passed in
or the wiring, never the motor.

``hw_ver`` and ``sw_ver`` are read alongside and logged, never compared -- a firmware version is worth
having in a startup log for fleet triage, and there is no single right value to check it against. A
failure to read either is a warning, never a refusal: nothing depends on them.

Check 2: control mode and feedback scaling
==========================================

A chain commands every motor in one ``ControlMode``, and the motor only answers that mode's command
frame if its ``CTRL_MODE`` register agrees -- 1 MIT, 2 pos-speed, 3 speed. One that does not fails late
and misleadingly: ``_motor_on`` enables it over the raw-id frame, which is answered in any mode, so the
chain builds cleanly, and it is the first mode-specific command that goes unanswered. The reader thread
then raises, sets ``running = False``, and the operator is told "Motor interface is not running ...
Please check the E stop or the motor connection" -- which points at the wrong component entirely. So
``verify_motor_config`` takes the chain's control mode, reads ``CTRL_MODE`` from every motor before the
chain is opened, and repairs any that disagrees: written, verified with an independent read, then saved
to Flash. It is the only register either check ever writes.

It also reads ``PMAX``/``VMAX``/``TMAX`` and compares them against ``MotorType.get_motor_constants``,
but **never writes them**. Those three are the firmware half of the MIT feedback scale whose other half
is hard-coded in ``dm_driver``, and a register that disagrees raises no error anywhere -- it silently
rescales every reading. Nothing is written because, unlike ``CTRL_MODE``, these are not one correct
number the software gets to choose: they describe the physical motor, and writing a guess to Flash is
how a mis-scaled machine becomes a permanently mis-scaled one.

Every scaling mismatch is reported; which ones refuse to start is a two-part question, and only one
part is knowable from here:

* **Which registers reach a command or the control loop** follows from the control mode, so
  ``loop_registers`` derives it. ``PMAX`` and ``VMAX`` always scale ``MotorInfo.pos``/``.vel``, in every
  mode. ``TMAX`` is different: on a ``MIT`` chain ``set_control`` *encodes* commanded torque through it,
  so a wrong one mis-scales gravity compensation itself, but a ``VEL`` frame carries a raw float32 rad/s
  and nothing else, so there ``TMAX`` only decodes ``MotorInfo.eff`` -- a telemetry number -- and can
  never block.
* **Which motors are inside the caller's control loop** does not follow from anything here, so it is the
  ``loop_critical_motor_ids`` argument. Omit it and every motor counts, which is the strict policy an
  arm wants: it commands all six in MIT mode. The Flow Base passes its four steering motors, because
  only those are in the swerve loop -- ``PMAX`` there is the wheel angle every row of ``C``/``C_p``/
  ``C_pinv`` is rebuilt from each cycle, and ``VMAX`` is the ``dq`` the odometry integrates, the rate the
  caster-flip brake trips on and the measured half of the caster runaway detector. The same two
  registers on its drive or rail motors mis-scale only translational odometry, so those are logged and
  the base starts. See ``i2rt/flow_base/README.md`` for that reasoning in full.

Reading them at startup and nowhere else is the point. A firmware register cannot change while a robot
is running -- it changes when someone reflashes a motor or reconfigures it from the host tool -- so this
is the only moment worth spending bus time on, and reading the register is exact where inferring a
mismatch from motion needs thresholds.

``CTRL_MODE`` is a uint32 register, which is why its comparison is a plain ``int`` equality with no
tolerance. The scaling registers are float32 and are compared with a relative tolerance instead -- see
``_REL_TOL`` for why an exact compare would report every correctly configured motor as broken.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence
from typing import Any

from i2rt.motor_config_tool.dm_motor_registers import (
    BUS_ERRORS,
    REG_BY_ADDR,
    DMRegAddr,
    RegSpec,
    Scalar,
    format_value,
    read_register,
    save_register_to_flash,
    value_fault,
    write_register,
)
from i2rt.motor_config_tool.utils import RawCanInterface
from i2rt.motor_drivers.utils import MotorType

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------------------
# Shared by both checks
# --------------------------------------------------------------------------------------------------

# float32 carries ~1.2e-7 of relative precision, so a float64 expectation never reads back bit-identical:
# PMAX 3.1415926 comes back as 3.141592502593994. An exact compare would therefore report every correctly
# configured motor as mis-scaled, forever. 1e-6 clears that noise by ~16x and still rejects anything worth
# reporting: a hand-typed 3.1416 is 2.4e-6 off and fails. Every expected value is non-zero, so a relative
# tolerance alone is sufficient. Gear ratios are small integers and so exactly representable, but the same
# tolerance costs nothing there and means a motor reporting 39.999996 is not called a mismatch.
_REL_TOL = 1e-6


def matches(spec: RegSpec, actual: Scalar, expected: Scalar) -> bool:
    """Whether a register reading is the expected value, allowing for the float32 round trip.

    NaN needs no special case -- ``isclose(nan, x)`` is False -- so a Flash cell that was never written
    counts as a mismatch and is reported. ``value_fault`` is consulted only to word the log line.
    """
    if not spec.is_float:
        return int(actual) == int(expected)  # uint32: exact, no tolerance
    return math.isclose(float(actual), float(expected), rel_tol=_REL_TOL)


def describe(channel: str, motor_id: int, reg: DMRegAddr, actual: Scalar, expected: Scalar) -> str:
    """The one-line "this is wrong" clause every mismatch message below is built from."""
    spec = REG_BY_ADDR[int(reg)]
    note = " (that Flash cell was never written, which is why it reads as NaN)" if value_fault(spec, actual) else ""
    return (
        f"{channel} motor {motor_id} {spec.name} (@{spec.addr}): {format_value(spec, actual)}, "
        f"expected {format_value(spec, expected)}{note}"
    )


def _motor_rows(motor_list: Sequence[Sequence[Any]]) -> list[tuple[int, str]]:
    """Coerce a chain's ``motor_list`` to ``(can_id, motor_type)`` pairs, dropping the passive rows.

    Coerced rather than trusted because callers hand over both the tuple form an arm config holds and
    the mutable list forms ``get_robot`` and ``flow_base_controller`` build so they can append the
    gripper or rail motor.

    ``no_gripper`` and ``yam_teaching_handle`` are spelled ``motor_type: ""`` -- they are passive, and
    the assembled arm chain leaves them out entirely. A gripper-only robot built with a teaching handle
    is the one path that can still put such a row here. Both checks drop it, because nothing about ""
    is a motor: ``get_gear_ratio("")`` and ``get_motor_constants("")`` would each raise as though a
    table were missing an entry, sending the operator to the wrong file.
    """
    return [(int(motor_id), str(motor_type)) for motor_id, motor_type in motor_list if str(motor_type)]


def _nothing_to_check(channel: str, motor_list: Sequence[Sequence[Any]]) -> None:
    """Log the "every row was passive" case. Neither check opens a bus for it."""
    logger.info("nothing to check on %s: %d row(s), none of them a motor", channel, len(motor_list))


def _open_bus(channel: str, name: str, what: str) -> RawCanInterface:
    """Open the raw interface both checks read through, or say why the channel is unusable."""
    try:
        return RawCanInterface(channel=channel, bustype="socketcan", name=name)
    except BUS_ERRORS as e:
        raise RuntimeError(
            f"could not open {channel} to check the motor {what}: {e}. Check the interface is up "
            f"(ip link show {channel})."
        ) from e


def run_startup_checks(
    channel: str,
    motor_list: Sequence[Sequence[Any]],
    *,
    check_motor_types: bool,
    check_motor_config: bool,
    control_mode: str,
    loop_critical_motor_ids: Sequence[int] | None = None,
    repair: bool = True,
) -> None:
    """Run whichever startup checks a caller asked for, in the only order that is safe.

    The single entry point for both, so the two rules below hold for every caller rather than for
    whichever one remembered them. ``DMChainCanInterface.__init__`` calls this at the top of the
    constructor; ``dm_driver.py --survey-only`` calls it *instead* of building a chain at all, which is
    the only way to read a chain's registers without energising it.

    Both flags are keyword-only and have no default. Two adjacent positional bools are a swap hazard
    whose failure mode is asymmetric: read as "config only" a swap now raises, but read as "types only"
    it starts the robot having checked less than the caller asked for, and nothing downstream can tell.

    ``repair`` is what separates the two call sites: the constructor is about to run the chain, so it
    fixes a wrong ``CTRL_MODE``; ``--survey-only`` reports it and writes nothing, which is what makes
    the word "survey" true. It is the only argument here that decides whether anything is written.

    Raises:
        ValueError: if ``check_motor_config`` is asked for without ``check_motor_types``, or if either
            is asked for on a channel that cannot carry it. Both are decidable from the arguments alone
            and are repaired by changing them -- the same category as an unrecorded gear ratio. A
            ``RuntimeError`` from here down always means the bus or a motor refused.
    """
    if not (check_motor_types or check_motor_config):
        return  # before the channel guard: an unchecked chain may be built on any channel, as ever
    if check_motor_config and not check_motor_types:
        raise ValueError(
            "check_motor_config=True needs check_motor_types=True. The config check compares the "
            "registers that scale a motor's commands and feedback, and on the wrong part those hold "
            "that part's own scale, so running it unscreened would report three consequent mismatches, "
            "advise rewriting PMAX, and write CTRL_MODE to Flash on a motor whose repair is to be "
            "swapped out. Ask for both, or for neither."
        )
    if "can" not in channel:
        raise ValueError(
            f"cannot check the motors on {channel!r}: both checks read registers over socketcan, and "
            "this is not a socketcan channel. Starting anyway would trust every motor's declared type "
            "and scaling unverified, which is the failure these checks exist to make loud -- so a check "
            "that was asked for and cannot run refuses instead of warning. Pass a socketcan channel "
            "(e.g. 'can0'), or ask for neither check."
        )
    # check_motor_types is necessarily True here: neither-flag returned above, and config-without-types
    # was just refused. Positional, because the call signature is asserted that way by the tests.
    verify_motor_types(channel, motor_list)
    if check_motor_config:
        verify_motor_config(channel, motor_list, control_mode, loop_critical_motor_ids, repair=repair)


# --------------------------------------------------------------------------------------------------
# Check 1: motor type (Gr) -- see the module docstring
# --------------------------------------------------------------------------------------------------

MAX_READ_ATTEMPTS = 5
"""How many times ``Gr`` is read before the chain is declared unreadable.

Each attempt is itself a ``read_register`` call, which retries 20x internally, so this is an *outer*
loop over ~0.65 s units -- ~3.8 s in total for a motor that never answers. That is the cost of not
refusing to start an otherwise healthy chain because of one burst of bus noise.
"""

RETRY_SLEEP_S = 0.1
"""Pause between outer attempts, to let a transient (a stale frame, a contending process finishing)
clear rather than immediately re-reading into it."""

LOGGED_REGISTERS: tuple[DMRegAddr, ...] = (DMRegAddr.HW_VER, DMRegAddr.SW_VER)
"""Read and logged on every motor, compared against nothing. Ordered as they are numbered."""

_GR_SPEC = REG_BY_ADDR[int(DMRegAddr.GR)]


def _log_read(channel: str, motor_id: int, motor_type: str, reg: DMRegAddr, value: Scalar) -> None:
    """Record one successful read. ``format_value`` keeps a float32 from printing 17 digits."""
    spec = REG_BY_ADDR[int(reg)]
    logger.info(
        "%s motor %d %s: %s (@%d) = %s",
        channel,
        motor_id,
        motor_type,
        spec.name,
        spec.addr,
        format_value(spec, value),
    )


def _read_gear_ratio(iface: RawCanInterface, channel: str, motor_id: int, motor_type: str) -> float:
    """Read one motor's ``Gr``, retrying a bus failure or a non-finite answer.

    Raises:
        RuntimeError: if it still has not produced a usable number after ``MAX_READ_ATTEMPTS``. The
            message covers both causes because at this point they are indistinguishable from here:
            the motor is unreachable, or it is reachable and its Flash never held a gear ratio.
    """
    reason = ""
    for attempt in range(1, MAX_READ_ATTEMPTS + 1):
        try:
            value = read_register(iface, motor_id, DMRegAddr.GR)
        except BUS_ERRORS as e:
            reason = str(e)
        else:
            # A decode never rejects a value, so NaN arrives here as an ordinary float; value_fault is
            # the predicate that says it is not one. Retried rather than raised on, per the caller's
            # policy that any unusable reading gets MAX_READ_ATTEMPTS chances.
            fault = value_fault(_GR_SPEC, value)
            if fault is None:
                _log_read(channel, motor_id, motor_type, DMRegAddr.GR, value)
                return float(value)
            reason = fault
        if attempt < MAX_READ_ATTEMPTS:
            logger.warning(
                "%s motor %d %s (@%d) attempt %d/%d failed: %s -- retrying in %.2f s",
                channel,
                motor_id,
                _GR_SPEC.name,
                _GR_SPEC.addr,
                attempt,
                MAX_READ_ATTEMPTS,
                reason,
                RETRY_SLEEP_S,
            )
            time.sleep(RETRY_SLEEP_S)
    raise RuntimeError(
        f"motor type check failed on {channel}: motor {motor_id} ({motor_type}) did not give a "
        f"readable Gr in {MAX_READ_ATTEMPTS} attempts ({reason}), so its type could not be "
        f"confirmed and the robot was NOT started. Check that it is powered and the E stop is "
        f"released; that its CAN id really is {motor_id} (dm_motor_registers.py read ESC_ID "
        f"--motor-id {motor_id} --channel {channel}); and that nothing else is using {channel}, "
        f"since a running robot, ping_motors.py or candump makes every register read fail."
    )


def _read_logged_registers(iface: RawCanInterface, channel: str, motor_id: int, motor_type: str) -> None:
    """Read and log ``LOGGED_REGISTERS``. A failure is reported and otherwise ignored.

    Deliberately unable to stop a launch: these are not compared against anything, so not reading
    one says nothing about whether the chain matches its config. Read after ``Gr`` so a motor that
    is not answering at all has already bailed out and is not charged ~0.65 s per register here.
    """
    for reg in LOGGED_REGISTERS:
        spec = REG_BY_ADDR[int(reg)]
        try:
            value = read_register(iface, motor_id, reg)
        except BUS_ERRORS as e:
            logger.warning(
                "%s motor %d %s: %s (@%d) could not be read (%s); it is logged only, so the check continues",
                channel,
                motor_id,
                motor_type,
                spec.name,
                spec.addr,
                e,
            )
            continue
        _log_read(channel, motor_id, motor_type, reg, value)


def _describe_mismatch(motor_id: int, motor_type: str, actual: float, expected: float) -> str:
    """One "this is the wrong motor" clause, naming the type the reading actually corresponds to.

    The reverse lookup is what makes the message actionable: "Gr 40" means nothing to most readers,
    "Gr 40, i.e. a DM4340" names the part in the arm.
    """
    installed = sorted(
        name for name, ratio in MotorType.known_gear_ratios().items() if matches(_GR_SPEC, actual, ratio)
    )
    looks_like = f", i.e. a {' or '.join(installed)}" if installed else ""
    return (
        f"motor {motor_id} reported Gr {format_value(_GR_SPEC, actual)}{looks_like}, but the config "
        f"declares {motor_type} (Gr {expected:g})"
    )


def verify_motor_types(channel: str, motor_list: Sequence[Sequence[Any]]) -> None:
    """Check that every motor on ``channel`` is the type ``motor_list`` declares.

    ``motor_list`` rows are ``(can_id, motor_type)``; ``_motor_rows`` coerces them and drops the
    passive ones. ``run_startup_checks`` calls this on the ``motor_list`` the chain was handed, before
    the socket opens; see the module docstring for why that is the only valid moment.

    Nothing is ever written. Raises ``RuntimeError`` if any motor's ``Gr`` could not be read, or if
    any motor reports a gear ratio its declared type does not have; in either case the robot must not
    start on a chain nobody can vouch for. Raises ``ValueError`` before touching the bus if a
    declared type has no recorded gear ratio, which is a gap in the code rather than in the hardware.
    """
    motors = _motor_rows(motor_list)
    if not motors:
        _nothing_to_check(channel, motor_list)
        return
    # Resolved up front, so an unrecorded motor type fails as a plain ValueError without opening a
    # socket, enabling anything, or spending bus time first.
    expected = {motor_type: MotorType.get_gear_ratio(motor_type) for _, motor_type in motors}
    started = time.monotonic()
    logger.info(
        "checking motor types on %s: %s",
        channel,
        ", ".join(f"{motor_id} {motor_type}" for motor_id, motor_type in motors),
    )
    iface = _open_bus(channel, "motor_type_check", "types")

    mismatched: list[str] = []
    try:
        for motor_id, motor_type in motors:
            actual = _read_gear_ratio(iface, channel, motor_id, motor_type)
            _read_logged_registers(iface, channel, motor_id, motor_type)
            want = expected[motor_type]
            if not matches(_GR_SPEC, actual, want):
                clause = _describe_mismatch(motor_id, motor_type, actual, want)
                logger.error("%s %s", channel, clause)
                # Collected rather than raised on: every read has already succeeded, so walking the
                # rest of the chain is cheap and one message can name every wrong motor.
                mismatched.append(clause)
    finally:
        iface.close()

    if mismatched:
        raise RuntimeError(
            f"motor type check failed on {channel}: "
            + "; ".join(mismatched)
            + ". This chain was built with motor types that are not the ones plugged in. On an arm "
            "that is usually the wrong arm or gripper variant -- a yam_ultra_2 arm launched as "
            "--arm yam_ultra is the likely case, since those two configs differ only at joint 4 "
            "(DM4340 vs DM4310); on the dm_driver CLI it is usually a --motor-type that does not "
            "describe the chain. Either way it can also be a motor replaced with a different model. "
            "The declared type is what dm_driver encodes every position, velocity and torque command "
            "with, so the robot was NOT started. "
            f"Nothing was written. Confirm by hand with: python "
            f"i2rt/motor_config_tool/dm_motor_registers.py read Gr --motor-id <id> --channel {channel}"
        )
    logger.info(
        "motor type check passed on %s in %.2f s (%d motor(s))",
        channel,
        time.monotonic() - started,
        len(motors),
    )


# --------------------------------------------------------------------------------------------------
# Check 2: control mode and feedback scaling -- see the module docstring
# --------------------------------------------------------------------------------------------------

CTRL_MODE_BY_CONTROL_MODE: dict[str, int] = {"MIT": 1, "POS_VEL": 2, "VEL": 3}
"""The ``CTRL_MODE`` a motor must hold to answer each of the driver's control modes.

Keyed by the *string* rather than by ``ControlMode`` on purpose: ``ControlMode`` lives in ``dm_driver``,
which imports this module, so naming the class here would be a cycle. Its members are plain strings
("MIT", "POS_VEL", "VEL"), so the keys are those values and nothing needs importing. The numbers are the
firmware's, from register 10's own meaning: 1 MIT, 2 pos-speed, 3 speed, 4 torque-pos.

Being *in* this map is not a claim that the driver can command the mode: ``POS_VEL`` is here because the
firmware numbers it, but ``DMSingleMotorCanInterface.set_control`` encodes MIT and VEL only and refuses
anything else. Do not read this map as a list of modes to offer a caller -- ``dm_driver.py``'s
``--control-mode`` deliberately offers the two that encode, since a mode this check could Flash-save but
the driver could not command would strand a chain across power cycles.

A mode absent from this map raises rather than being skipped -- see ``expected_ctrl_mode``. Torque-pos
(4) is deliberately not here: the driver has no control mode that sends it.
"""

CTRL_MODE_SPEED = CTRL_MODE_BY_CONTROL_MODE["VEL"]
"""Speed mode, the one every Flow Base chain runs in. Kept named because the base's docs cite it."""


def expected_ctrl_mode(control_mode: str) -> int:
    """The ``CTRL_MODE`` register value a chain in ``control_mode`` needs every motor to hold.

    Raises:
        ValueError: on a control mode with no recorded register value. Raised rather than defaulted
            because the fallback would be to write *some* mode to Flash on every motor, and guessing
            which is exactly the failure this check exists to prevent.
    """
    try:
        return CTRL_MODE_BY_CONTROL_MODE[control_mode]
    except KeyError:
        raise ValueError(
            f"no CTRL_MODE recorded for control mode {control_mode!r}. Add it to "
            f"CTRL_MODE_BY_CONTROL_MODE in i2rt/motor_drivers/motor_check.py, taking the value from "
            f"register 10's meaning (1 MIT, 2 pos-speed, 3 speed, 4 torque-pos). "
            f"Recorded: {sorted(CTRL_MODE_BY_CONTROL_MODE)}"
        ) from None


IDENTITY_REGISTERS: tuple[DMRegAddr, ...] = (DMRegAddr.SW_VER,)
"""Read first, both for the log and as the reachability probe. Read-only, so never repaired.

``sw_ver`` is log-only: there is no single right firmware version to hold it to. ``Gr`` used to be read
here and compared against ``MotorType.get_gear_ratio``; that is ``verify_motor_types``' job now, and
running this check without it means the scaling mismatches below are not screened for "the wrong part is
fitted" first -- which is why the constructor runs the two in that order.
"""

SCALING_REGISTERS: tuple[DMRegAddr, ...] = (DMRegAddr.PMAX, DMRegAddr.VMAX, DMRegAddr.TMAX)
"""The firmware half of the MIT feedback scale. Read and compared on every motor, written on none."""

_CTRL_MODE_SPEC = REG_BY_ADDR[int(DMRegAddr.CTRL_MODE)]


def loop_registers(control_mode: str) -> tuple[DMRegAddr, ...]:
    """Which of ``SCALING_REGISTERS`` reach a command or the control loop in ``control_mode``.

    A mismatch on one of these, on a motor the caller called loop-critical, blocks the launch; anything
    else is reported and the robot starts. This half of the severity question is derivable, unlike which
    motors are loop-critical -- see ``_check_scaling``.

    ``PMAX`` and ``VMAX`` always qualify: they scale ``MotorInfo.pos`` and ``.vel``, which every mode
    reports and the chain unwraps position across a ``POSITION_MAX - POSITION_MIN`` window.

    ``TMAX`` qualifies only in ``MIT``, and that difference is the whole reason this is a function. In
    ``MIT`` the driver's ``set_control`` *encodes* commanded torque through it, so a wrong one mis-scales
    every torque the robot applies -- gravity compensation included. A ``VEL`` frame carries a raw
    float32 rad/s and nothing else, so there the only thing ``TMAX`` touches is ``MotorInfo.eff``, which
    on the Flow Base has exactly two readers -- ``get_wheel_states`` and the linear rail's state dict --
    both of which only publish it outward. A wrong ``TMAX`` is still wrong and is still reported on every
    motor in every mode; in ``VEL`` it mis-scales a telemetry number, which is not a reason to refuse to
    move.
    """
    if control_mode == "MIT":
        return SCALING_REGISTERS
    return (DMRegAddr.PMAX, DMRegAddr.VMAX)


def scaling_for(motor_type: str) -> dict[DMRegAddr, float]:
    """The PMAX/VMAX/TMAX a motor of this type must hold, taken from the driver's own constants.

    Derived rather than written down a second time: these registers are the firmware half of a scale
    whose other half is hard-coded in ``dm_driver``, and a copy here is exactly how the two would drift
    apart. Both Flow Base motor types -- ``DM4310V`` steering and ``DM_FLOW_WHEEL`` drive -- resolve to
    pi / 30 / 10. Note ``DMH6215MIT`` is a *different* entry at 12.5 / 45 / 10 and is not either of them.

    The four types the shipped arm configs use are 12.5 in ``PMAX`` and differ in the other two:
    ``DM4310`` 30 / 10, ``DM4340`` 10 / 28, ``DM6248`` 20 / 120, ``DM3507`` 50 / 5. ``DM4340``'s
    ``VMAX`` of 10 is the one worth knowing before a survey: it is the only arm entry below the 30 a
    DM motor is otherwise expected to hold, so a joint 1-3 motor left at a stock 30 is the mismatch an
    arm is most likely to be refused for.
    """
    constants = MotorType.get_motor_constants(motor_type)
    return {
        DMRegAddr.PMAX: float(constants.POSITION_MAX),
        DMRegAddr.VMAX: float(constants.VELOCITY_MAX),
        DMRegAddr.TMAX: float(constants.TORQUE_MAX),
    }


def _read_motor(iface: RawCanInterface, channel: str, motor_id: int) -> dict[DMRegAddr, Scalar] | None:
    """Read one motor's identity, control mode and scaling, or None if any of it could not be read.

    Bails on the first failure. A register that does not answer costs ~0.65 s of retries inside
    ``_tx_rx``, so continuing would spend seconds to learn what the first read already established --
    and the whole run is about to be abandoned anyway.

    Five registers per motor, roughly 0.4 s for a healthy eight-motor bus. The three scaling registers
    are most of that; they are read on the same open bus pass because there is no second one.
    """
    values: dict[DMRegAddr, Scalar] = {}
    for reg in (*IDENTITY_REGISTERS, DMRegAddr.CTRL_MODE, *SCALING_REGISTERS):
        spec = REG_BY_ADDR[int(reg)]
        try:
            values[reg] = read_register(iface, motor_id, reg)
        except BUS_ERRORS as e:
            if not values:
                logger.error(
                    "%s motor %d did not answer a read of %s (%s) -- skipping its remaining registers, and "
                    "writing nothing on any motor. Check that it is powered and the E stop is released; that "
                    "its CAN id really is %d (dm_motor_registers.py read ESC_ID --motor-id %d --channel %s); "
                    "and that nothing else is using %s, since a running base-controller, ping_motors.py or "
                    "candump makes every register read fail.",
                    channel,
                    motor_id,
                    spec.name,
                    e,
                    motor_id,
                    motor_id,
                    channel,
                    channel,
                )
            else:
                logger.error(
                    "%s motor %d: reading %s failed (%s) -- its configuration is unknown, so nothing is being "
                    "written on any motor.",
                    channel,
                    motor_id,
                    spec.name,
                    e,
                )
            return None
    return values


def _repair(iface: RawCanInterface, channel: str, motor_id: int, actual: Scalar, expected: int) -> bool:
    """Write the chain's control mode to one motor, verify it independently, then persist it.

    Returns True if it stuck. ``CTRL_MODE`` is the only register either check ever writes; see
    ``_check_scaling`` for why the scaling registers are reported instead.
    """
    faulty = describe(channel, motor_id, DMRegAddr.CTRL_MODE, actual, expected)
    try:
        write_register(iface, motor_id, DMRegAddr.CTRL_MODE, expected)
        # An independent read, never write_register's return value: _tx_rx deliberately does not
        # echo-check writes, so that reply proves nothing about what the motor actually stored.
        readback = read_register(iface, motor_id, DMRegAddr.CTRL_MODE)
    except BUS_ERRORS as e:
        logger.error("%s -- the write failed (%s), so nothing was saved to Flash.", faulty, e)
        return False
    if not matches(_CTRL_MODE_SPEC, readback, expected):
        logger.error(
            "%s -- wrote it, but it read back as %s. The motor did not take the value, so nothing was saved to Flash.",
            faulty,
            format_value(_CTRL_MODE_SPEC, readback),
        )
        return False
    try:
        save_register_to_flash(iface, motor_id, DMRegAddr.CTRL_MODE)
    except BUS_ERRORS as e:
        logger.error(
            "%s -- corrected in RAM, but saving it to Flash failed (%s), so it will revert on the next power cycle.",
            faulty,
            e,
        )
        return False
    # A 0x55 write changes the value the motor is running on now, but nothing establishes whether DM
    # firmware re-latches the control mode without a reboot, and the read-back only proves RAM took the
    # number. Say so, rather than let it resurface as the misleading failure this check exists to prevent.
    logger.warning(
        '%s -- written and saved to Flash. If the robot now fails with "Motor interface is not running", '
        "power-cycle motor %d: the mode change may need a reboot to take effect, and the E stop is not the "
        "problem.",
        faulty,
        motor_id,
    )
    return True


def _report_unrepaired(channel: str, wrong: Sequence[tuple[int, Scalar]], expected: int, control_mode: str) -> None:
    """Say what each wrong ``CTRL_MODE`` is and how to fix it, on a survey that may not write.

    The read-only counterpart to ``_repair``: same ``describe`` line, but ending in the by-hand command
    rather than a write, exactly as ``_check_scaling`` does for the three registers no caller ever
    repairs. A survey that silently declined to fix what it found would be the quiet failure this module
    exists to remove, so every motor gets its own line and the caller raises afterwards.
    """
    for motor_id, actual in wrong:
        logger.error(
            "%s -- this is a survey, so nothing was written. Fix it with the robot stopped: "
            "dm_motor_registers.py write CTRL_MODE --value %d --motor-id %d --channel %s, then the same "
            "with save. Or drop --survey-only and let the check repair it on the way up.",
            describe(channel, motor_id, DMRegAddr.CTRL_MODE, actual, expected),
            expected,
            motor_id,
            channel,
        )
    logger.error(
        "%s: %d motor(s) are not in %s (CTRL_MODE %d). Nothing was written on any of them.",
        channel,
        len(wrong),
        control_mode,
        expected,
    )


def _consequence(reg: DMRegAddr, control_mode: str, loop_critical: bool) -> str:
    """What this particular mismatch actually costs -- the clause the ERROR line ends with.

    Which answer applies depends on the register at least as much as on the motor, and conflating the
    two is how a ``TMAX`` typo on a Flow Base steering motor used to abort a launch over an angle it does
    not scale. Both halves of the question are spelled out in the module docstring; this renders them.
    """
    if reg not in loop_registers(control_mode):  # TMAX outside MIT
        return (
            f"a {control_mode} command frame carries no torque, so nothing encodes through TMAX here -- "
            "it scales only the torque this chain reports, as MotorInfo.eff. Starting; only the reported "
            "torque is wrong."
        )
    if not loop_critical:
        return (
            "this motor is not one the caller marked as inside its control loop, so this rescales the "
            "motor's own readings and anything integrated from them, but not what the robot steers by. "
            "Starting anyway -- fix it before trusting any number derived from this motor."
        )
    if reg == DMRegAddr.PMAX:
        return (
            "this is the scale every position this motor reports decodes through, and the chain unwraps "
            "that position across a POSITION_MAX - POSITION_MIN window on top of it, so the control loop "
            "would be solving for a place this joint is not. NOT starting."
        )
    if reg == DMRegAddr.VMAX:
        return (
            "this is the scale this motor's velocity feedback decodes through -- the dq anything "
            "integrating or rate-limiting reads. The reported position itself stays right; everything "
            "judging how fast it is moving does not. NOT starting."
        )
    return (
        f"a {control_mode} frame encodes commanded torque through TMAX, so every torque this chain "
        "applies to this motor is off by that ratio -- gravity compensation included. NOT starting."
    )


def _check_scaling(
    channel: str,
    motor_id: int,
    motor_type: str,
    values: dict[DMRegAddr, Scalar],
    control_mode: str,
    loop_critical: bool,
) -> bool:
    """Report any PMAX/VMAX/TMAX that disagrees with the driver's constants. True if it must stop the launch.

    Read-only on purpose -- see the module docstring. Every mismatch is logged; only one on a register
    that reaches a command or the loop in ``control_mode`` (``loop_registers``) *and* on a motor the
    caller called loop-critical stops the launch.

    A mismatch here can also be the *consequence* of the wrong part being fitted, in which case those
    registers hold that part's own scale and the advice below is wrong. ``verify_motor_types`` is what
    rules that out, which is why the constructor runs it first.
    """
    expected = scaling_for(motor_type)
    bad = [reg for reg in SCALING_REGISTERS if not matches(REG_BY_ADDR[int(reg)], values[reg], expected[reg])]
    if not bad:
        return False
    in_loop = loop_registers(control_mode)
    blocking = loop_critical and any(reg in in_loop for reg in bad)
    for reg in bad:
        logger.error(
            "%s -- %s",
            describe(channel, motor_id, reg, values[reg], expected[reg]),
            _consequence(reg, control_mode, loop_critical),
        )
    logger.error(
        "%s motor %d (%s): this check never writes these three registers, so fix it by hand with the robot "
        "stopped -- dm_motor_registers.py write <REG> --value <expected above> --motor-id %d --channel %s, "
        "then the same with save. If the values look like another motor's entirely, the part is probably "
        "not the one declared: run the motor type check (check_motor_types=True), which reads Gr and says "
        "so directly.",
        channel,
        motor_id,
        motor_type,
        motor_id,
        channel,
    )
    return blocking


def verify_motor_config(
    channel: str,
    motor_list: Sequence[Sequence[object]],
    control_mode: str,
    loop_critical_motor_ids: Sequence[int] | None = None,
    *,
    repair: bool = True,
) -> None:
    """Check every motor's control mode and feedback scaling on ``channel``.

    A ``CTRL_MODE`` that is not the one ``control_mode`` needs is repaired and persisted unless
    ``repair`` is False; a wrong ``PMAX``/``VMAX``/``TMAX`` is only ever reported.
    ``run_startup_checks`` calls this on the ``motor_list`` and ``control_mode`` the chain was handed,
    before the socket opens; see the module docstring for why that is the only valid moment, and why it
    refuses to call this without ``verify_motor_types``.

    An arm is the strictest caller this has: ``MIT`` puts ``TMAX`` in ``loop_registers`` and it passes
    no ``loop_critical_motor_ids``, so every one of the three registers on every one of its motors can
    refuse the launch. That is the intended reading -- an arm commands all six joints plus its gripper
    in MIT and steers by all of their feedback -- and it means an arm whose Flash was never surveyed
    finds out here rather than in a mis-scaled torque.

    Args:
        channel: SocketCAN channel the chain is on.
        motor_list: ``(can_id, motor_type)`` rows, the same ones the chain is built from. Passed
            through ``_motor_rows``, so a passive ``""`` row is dropped rather than being sent to
            ``get_motor_constants``.
        control_mode: The chain's ``ControlMode`` value, as a string. Decides both the ``CTRL_MODE`` every
            motor must hold and which scaling registers can block -- see ``loop_registers``.
        loop_critical_motor_ids: The motors whose feedback the caller's control loop acts on. Omit for
            *every* motor, the strict reading, which is what an arm wants. The Flow Base passes its four
            steering motors: a drive or rail motor's mis-scaled feedback moves only its odometry.
        repair: Whether a wrong ``CTRL_MODE`` may be written and saved to Flash. True on the path that is
            about to run the chain, since a motor in the wrong mode cannot be commanded at all. False on
            ``dm_driver.py --survey-only``, whose whole promise is that it reads: there the mismatch is
            reported with the command that fixes it, and the robot is still refused. Nothing else this
            function touches is writable by it either way -- see ``_check_scaling``.

    Raises:
        RuntimeError: if any motor could not be read, any repair did not stick, or any loop-critical
            motor disagrees on a register that reaches a command or the loop. In each case the robot must
            not start on a configuration nobody can vouch for. Every other scaling mismatch is logged and
            the robot starts. With ``repair=False`` a wrong ``CTRL_MODE`` raises too, rather than being
            reported and passed over: a survey whose exit code cannot distinguish a good chain from a
            broken one is one nobody can script a fleet sweep against.
        ValueError: before the bus is touched, if ``control_mode`` has no recorded ``CTRL_MODE`` -- a gap
            in ``CTRL_MODE_BY_CONTROL_MODE`` rather than a fault in the hardware.
    """
    motors = _motor_rows(motor_list)
    if not motors:
        _nothing_to_check(channel, motor_list)
        return
    # Resolved up front so an unrecorded control mode fails as a plain ValueError, without opening a
    # socket, enabling a motor or spending bus time to discover a code gap.
    want_mode = expected_ctrl_mode(control_mode)
    loop_critical = (
        set(motor_id for motor_id, _ in motors) if loop_critical_motor_ids is None else set(loop_critical_motor_ids)
    )
    started = time.monotonic()
    logger.info(
        "checking control mode (%s, CTRL_MODE %d) and scaling on %s: %s",
        control_mode,
        want_mode,
        channel,
        ", ".join(f"{motor_id} {motor_type}" for motor_id, motor_type in motors),
    )
    iface = _open_bus(channel, "motor_config_check", "configuration")

    try:
        wrong: list[tuple[int, Scalar]] = []
        mis_scaled: list[int] = []
        unreadable: list[int] = []
        failed: list[int] = []
        not_attempted: list[int] = []
        for motor_id, motor_type in motors:
            values = _read_motor(iface, channel, motor_id)
            if values is None:
                unreadable.append(motor_id)
                continue
            logger.info(
                "%s motor %d %s: %s",
                channel,
                motor_id,
                motor_type,
                " ".join(
                    f"{REG_BY_ADDR[int(reg)].name}={format_value(REG_BY_ADDR[int(reg)], value)}"
                    for reg, value in values.items()
                ),
            )
            actual = values[DMRegAddr.CTRL_MODE]
            if not matches(_CTRL_MODE_SPEC, actual, want_mode):
                wrong.append((motor_id, actual))
            if _check_scaling(channel, motor_id, motor_type, values, control_mode, motor_id in loop_critical):
                mis_scaled.append(motor_id)

        if unreadable:
            raise RuntimeError(
                f"motor configuration check failed on {channel}: motor(s) {unreadable} did not answer, so "
                "nothing was written on any motor and the robot was NOT started. See the ERROR lines above."
            )
        if mis_scaled:
            # Before any repair: a bus whose scaling we do not believe is not one to commit Flash writes on.
            raise RuntimeError(
                f"motor configuration check failed on {channel}: motor(s) {mis_scaled} are mis-scaled -- "
                "they do not hold the PMAX/VMAX/TMAX the driver encodes and decodes them with, on a "
                f"register that reaches this chain's {control_mode} commands or its control loop, so the "
                "robot was NOT started. "
                "Nothing was written. See the ERROR lines above for which register on which motor, and "
                "what each one costs. Mismatches outside that set are reported the same way and never "
                "block."
            )

        if not repair:
            if wrong:
                _report_unrepaired(channel, wrong, want_mode, control_mode)
        else:
            # Stops at the first repair that does not stick, rather than committing the rest over a bus
            # that just dropped one: that is the same rule the unreadable/mis_scaled guards above enforce
            # for the read phase, and save_register_to_flash is the call most likely to false-ack on a
            # contended bus (_tx_rx accepts any received frame for CMD_SAVE).
            for position, (motor_id, actual) in enumerate(wrong):
                if not _repair(iface, channel, motor_id, actual, want_mode):
                    failed.append(motor_id)
                    not_attempted = [pending_id for pending_id, _ in wrong[position + 1 :]]
                    break
    finally:
        iface.close()

    if wrong and not repair:
        raise RuntimeError(
            f"motor configuration survey on {channel}: motor(s) {[motor_id for motor_id, _ in wrong]} are "
            f"not in {control_mode} (CTRL_MODE {want_mode}), so this chain would not run as declared. "
            "Nothing was written -- this was a survey. See the ERROR lines above for the command that "
            "fixes each one, or drop --survey-only to let the check repair them on the way up."
        )

    if failed:
        remaining = (
            f" Motor(s) {not_attempted} also needed the same repair and were NOT attempted: a bus that "
            "just dropped one Flash write is not a bus to commit the others on. Fix the motor above and "
            "run again."
            if not_attempted
            else ""
        )
        raise RuntimeError(
            f"motor configuration check failed on {channel}: motor(s) {failed} would not take the new control "
            f"mode, so the robot was NOT started. See the ERROR lines above for what to do about each.{remaining}"
        )
    # A survey only reaches this line with nothing to repair, so it must not report a count of writes it
    # was never allowed to make -- "0 motor(s) repaired and saved to Flash" reads as a write path that ran.
    logger.info(
        "motor configuration %s passed on %s in %.2f s%s",
        "check" if repair else "survey",
        channel,
        time.monotonic() - started,
        f" ({len(wrong)} motor(s) repaired and saved to Flash)" if repair else " (nothing was written)",
    )
