"""Startup check of the Flow Base motors' control mode and feedback scaling.

``DMChainCanInterface`` is built with ``ControlMode.VEL`` for every motor on the chain, so every motor
must hold ``CTRL_MODE`` = 3 (speed). One that does not fails late and misleadingly: ``_motor_on`` enables
it over the raw-id frame, which is answered in any mode, so the chain builds cleanly, and it is the first
speed-mode command (arbitration ``0x200 + id``) that goes unanswered. The reader thread then raises, sets
``running = False``, and the operator is told "Motor interface is not running ... Please check the E stop
or the motor connection" -- which points at the wrong component entirely.

This module reads ``CTRL_MODE`` from every motor before the chain is opened and repairs any that is
wrong: written, verified with an independent read, then saved to Flash.

It also reads ``PMAX``/``VMAX``/``TMAX`` from every motor and compares them against
``MotorType.get_motor_constants``, but **never writes them**. Those three are the firmware half of the
MIT feedback scale whose other half is hard-coded in ``dm_driver``, and a register that disagrees raises
no error anywhere -- it silently rescales every reading. Every mismatch is reported; which ones refuse to
start depends on the register as much as on the motor. On a *steering* motor, ``PMAX`` is the wheel angle
the swerve kinematics are rebuilt from every cycle, so the base would believe its wheels point somewhere
they do not and its odometry would agree, and ``VMAX`` is the velocity that odometry integrates, that the
caster-flip brake trips on and that the caster fault check judges: those two are inside the control loop,
and a mismatch there refuses to start -- see ``LOOP_SCALING_REGISTERS``. Everything else is an output. The
same two registers on a drive or rail motor mis-scale the translational odometry, and ``TMAX`` anywhere
mis-scales only the torque the base reports -- ``MotorInfo.eff``, which ``get_wheel_states`` and the rail
state publish and nothing in the loop reads -- so those are logged and the base starts. Nothing is written
because, unlike ``CTRL_MODE``, these are not one correct number the software gets to choose: they describe
the physical motor, and writing a guess to Flash is how a mis-scaled base becomes a permanently mis-scaled
one.

Reading them at startup and nowhere else is the point. A firmware register cannot change while the base
is driving -- it changes when someone reflashes a motor or reconfigures it from the host tool -- so this
is the only moment worth spending bus time on, and reading the register is exact where inferring a
mismatch from motion needs thresholds.

Three constraints shape everything here:

* **It can only run before ``DMChainCanInterface`` exists.** That constructor opens the socket, enables
  every motor and starts its background reader before it returns, and ``close()`` neither joins that
  thread nor allows a restart. Register access needs an idle bus -- one process, one thread, see
  ``i2rt/motor_config_tool/dm_motor_registers.md`` -- so there is no later window and no way to make one.
  The call site is inside ``_initialize_motor_chain``, which only runs when the caller passed a channel
  *name*; handing the controller a live chain therefore skips this check by construction.
* **A save (0xAA) is only ever issued for a register that was just written and then independently read
  back.** ``dm_motor_registers.md`` records a save on a not-just-written register reverting ten unrelated
  registers to their stored values. ``write_register``'s reply does not count as that read-back:
  ``_tx_rx`` deliberately skips the echo check on writes, because an ``ESC_ID`` write is answered by the
  motor's *new* id.
* **Nothing is written unless every motor answered.** A bus we could not read reliably is not a bus to
  commit Flash writes on, and this is also what makes contention safe: a busy bus fails the reads, so the
  check can never write into another process's traffic.

``CTRL_MODE`` is a uint32 register, which is why its comparison is a plain ``int`` equality with no
tolerance. The scaling registers are float32 and are compared with a relative tolerance instead -- see
``_REL_TOL`` for why an exact compare would report every correctly configured base as broken.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence

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

CTRL_MODE_SPEED = 3
"""DM control mode 3, "speed" -- what a ``ControlMode.VEL`` command frame requires."""

IDENTITY_REGISTERS: tuple[DMRegAddr, ...] = (DMRegAddr.SW_VER, DMRegAddr.GR)
"""Read first, both for the log and as the reachability probe. Read-only, so never repaired."""

SCALING_REGISTERS: tuple[DMRegAddr, ...] = (DMRegAddr.PMAX, DMRegAddr.VMAX, DMRegAddr.TMAX)
"""The firmware half of the MIT feedback scale. Read and compared on every motor, written on none."""

LOOP_SCALING_REGISTERS: tuple[DMRegAddr, ...] = (DMRegAddr.PMAX, DMRegAddr.VMAX)
"""The two of the three the base actually steers and navigates on. Only these can block a launch.

``PMAX`` scales ``MotorInfo.pos``, and on a steering motor that angle is what every row of ``C``,
``C_p`` and ``C_pinv`` is rebuilt from each cycle in ``Vehicle.update_state`` -- so a wrong one has the
swerve kinematics solving for a heading the wheel does not have, twice over, because the chain also
unwraps that position across a ``POSITION_MAX - POSITION_MIN`` window. ``VMAX`` scales
``MotorInfo.vel``, which becomes ``dq``: the odometry integrates it through ``C_pinv @ dq``, the
caster-flip brake trips on it, and it is the measured half of both the rate and the runaway detector in
``caster_steering_check``. Each of the two is read by something that acts on it.

``TMAX`` is not, which is the whole reason this tuple exists apart from ``SCALING_REGISTERS``. It
scales ``MotorInfo.eff``, and ``eff`` has exactly two readers on this base -- ``get_wheel_states`` and
the linear rail's state dict -- both of which only publish it outward. Nothing in the control loop, the
kinematics, the odometry or the caster fault check reads a torque at all. The one path that would use
``TMAX`` to *encode* is the MIT branch of the motor driver's ``set_control``, and both Flow Base chains
are built with ``ControlMode.VEL``, whose frame carries a raw float32 rad/s and nothing else, so that
branch is unreachable here. A wrong ``TMAX`` is still wrong, is still reported on every motor and is
still never written -- but it mis-scales a number on a telemetry endpoint, which is not a reason to
refuse to move the base.
"""

STEER_MOTOR_IDS: tuple[int, ...] = (1, 3, 5, 7)
"""The caster steering motors, in CAN-id order. The even ids are the drive motors, and the linear rail,
when fitted, is id 9. Only these can block the launch on a scaling mismatch, and only on a
``LOOP_SCALING_REGISTERS`` one -- see ``_check_scaling``."""

# float32 carries ~1.2e-7 of relative precision, so a float64 expectation never reads back bit-identical:
# PMAX 3.1415926 comes back as 3.141592502593994. An exact compare would therefore report every correctly
# configured base as mis-scaled, forever. 1e-6 clears that noise by ~16x and still rejects anything worth
# reporting: a hand-typed 3.1416 is 2.4e-6 off and fails. Every expected value is non-zero, so a relative
# tolerance alone is sufficient.
_REL_TOL = 1e-6

_SPEC = REG_BY_ADDR[int(DMRegAddr.CTRL_MODE)]


def is_speed_mode(actual: Scalar) -> bool:
    """Whether a ``CTRL_MODE`` reading is speed mode. Exact -- register 10 is a uint32."""
    return int(actual) == CTRL_MODE_SPEED


def scaling_for(motor_type: str) -> dict[DMRegAddr, float]:
    """The PMAX/VMAX/TMAX a motor of this type must hold, taken from the driver's own constants.

    Derived rather than written down a second time: these registers are the firmware half of a scale
    whose other half is hard-coded in ``dm_driver``, and a copy here is exactly how the two would drift
    apart. Both Flow Base motor types -- ``DM4310V`` steering and ``DM_FLOW_WHEEL`` drive -- resolve to
    pi / 30 / 10. Note ``DMH6215MIT`` is a *different* entry at 12.5 / 45 / 10 and is not either of them.
    """
    constants = MotorType.get_motor_constants(motor_type)
    return {
        DMRegAddr.PMAX: float(constants.POSITION_MAX),
        DMRegAddr.VMAX: float(constants.VELOCITY_MAX),
        DMRegAddr.TMAX: float(constants.TORQUE_MAX),
    }


def matches(spec: RegSpec, actual: Scalar, expected: Scalar) -> bool:
    """Whether a register reading is the expected value, allowing for the float32 round trip.

    NaN needs no special case -- ``isclose(nan, x)`` is False -- so a Flash cell that was never written
    counts as a mismatch and is reported. ``value_fault`` is consulted only to word the log line.
    """
    if not spec.is_float:
        return int(actual) == int(expected)  # uint32: exact, no tolerance
    return math.isclose(float(actual), float(expected), rel_tol=_REL_TOL)


def describe(channel: str, motor_id: int, reg: DMRegAddr, actual: Scalar, expected: Scalar) -> str:
    """The one-line "this is wrong" clause every log message below is built from."""
    spec = REG_BY_ADDR[int(reg)]
    note = " (that Flash cell was never written, which is why it reads as NaN)" if value_fault(spec, actual) else ""
    return (
        f"{channel} motor {motor_id} {spec.name} (@{spec.addr}): {format_value(spec, actual)}, "
        f"expected {format_value(spec, expected)}{note}"
    )


def _read_motor(iface: RawCanInterface, channel: str, motor_id: int) -> dict[DMRegAddr, Scalar] | None:
    """Read one motor's identity, control mode and scaling, or None if any of it could not be read.

    Bails on the first failure. A register that does not answer costs ~0.65 s of retries inside
    ``_tx_rx``, so continuing would spend seconds to learn what the first read already established --
    and the whole run is about to be abandoned anyway.

    Six registers per motor, which is roughly 0.5 s for a healthy eight-motor bus. The three scaling
    registers doubled that; they are read on the same open bus pass because there is no second one.
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


def _repair(iface: RawCanInterface, channel: str, motor_id: int, actual: Scalar) -> bool:
    """Write speed mode to one motor, verify it independently, then persist it. True if it stuck.

    ``CTRL_MODE`` is the only register this module ever writes; see ``_check_scaling`` for why the
    scaling registers are reported instead.
    """
    faulty = describe(channel, motor_id, DMRegAddr.CTRL_MODE, actual, CTRL_MODE_SPEED)
    try:
        write_register(iface, motor_id, DMRegAddr.CTRL_MODE, CTRL_MODE_SPEED)
        # An independent read, never write_register's return value: _tx_rx deliberately does not
        # echo-check writes, so that reply proves nothing about what the motor actually stored.
        readback = read_register(iface, motor_id, DMRegAddr.CTRL_MODE)
    except BUS_ERRORS as e:
        logger.error("%s -- the write failed (%s), so nothing was saved to Flash.", faulty, e)
        return False
    if not is_speed_mode(readback):
        logger.error(
            "%s -- wrote it, but it read back as %s. The motor did not take the value, so nothing was saved to Flash.",
            faulty,
            format_value(_SPEC, readback),
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
        '%s -- written and saved to Flash. If the base now fails with "Motor interface is not running", '
        "power-cycle motor %d: the mode change may need a reboot to take effect, and the E stop is not the "
        "problem.",
        faulty,
        motor_id,
    )
    return True


def _consequence(reg: DMRegAddr, steering: bool) -> str:
    """What this particular mismatch actually costs -- the clause the ERROR line ends with.

    Which of the answers applies depends on the register at least as much as on the motor, and
    conflating the two is how a ``TMAX`` typo on a steering motor used to abort a launch over an angle
    it does not scale. Only a ``LOOP_SCALING_REGISTERS`` one on a steering motor is inside the loop;
    ``TMAX`` is inside nothing, on any motor.
    """
    if reg not in LOOP_SCALING_REGISTERS:  # TMAX, wherever it is
        return (
            "this is the scale the reported torque decodes through, and reported torque is where it "
            "stops: get_wheel_states and the linear rail state publish it as MotorInfo.eff, and nothing "
            "in the control loop, the swerve kinematics, the odometry or the caster fault check reads a "
            "torque at all. The base is starting; only the torque it reports is wrong."
        )
    if not steering:
        return (
            "this rescales the motor's own readings and the base's translational odometry, but not its "
            "steering, so the base is starting anyway. Fix it before trusting any distance it reports."
        )
    if reg == DMRegAddr.PMAX:
        return (
            "this is the scale the swerve kinematics read the wheel angle through, so the base would "
            "believe its wheels point somewhere they do not and its odometry would agree. NOT starting."
        )
    return (
        "this is the scale this motor's velocity feedback decodes through -- the dq the odometry "
        "integrates, the rate the caster-flip brake trips on, and the measured half of the rate and "
        "runaway detectors. The reported wheel angle itself stays right; everything judging how fast it "
        "is moving does not. NOT starting."
    )


def _check_scaling(channel: str, motor_id: int, motor_type: str, values: dict[DMRegAddr, Scalar]) -> bool:
    """Report any PMAX/VMAX/TMAX that disagrees with the driver's constants. True if it must stop the launch.

    Read-only on purpose -- see the module docstring. Every mismatch is logged; the severity follows the
    blast radius of the register that is wrong, not of the motor it happens to sit on. Only a steering
    motor's ``PMAX`` or ``VMAX`` is inside the control loop, so only that combination stops the launch.
    The same two registers on a drive or rail motor, and ``TMAX`` on any motor, mis-scale something the
    base reports rather than something it steers by.
    """
    expected = scaling_for(motor_type)
    bad = [reg for reg in SCALING_REGISTERS if not matches(REG_BY_ADDR[int(reg)], values[reg], expected[reg])]
    if not bad:
        return False
    steering = motor_id in STEER_MOTOR_IDS
    blocking = steering and any(reg in LOOP_SCALING_REGISTERS for reg in bad)
    for reg in bad:
        logger.error(
            "%s -- %s",
            describe(channel, motor_id, reg, values[reg], expected[reg]),
            _consequence(reg, steering),
        )
    logger.error(
        "%s motor %d (%s): this check never writes these three registers, so fix it by hand with the base "
        "stopped -- dm_motor_registers.py write <REG> --value <expected above> --motor-id %d --channel %s, "
        "then the same with save. Both Flow Base motor types want pi / 30 / 10; if a motor reads "
        "12.5 / 45 / 10 it has been configured as DMH6215MIT, which is a different motor.",
        channel,
        motor_id,
        motor_type,
        motor_id,
        channel,
    )
    return blocking


def verify_base_motor_registers(channel: str, motor_list: Sequence[Sequence[object]]) -> None:
    """Check every motor's control mode and feedback scaling on ``channel``.

    A wrong ``CTRL_MODE`` is repaired and persisted; a wrong ``PMAX``/``VMAX``/``TMAX`` is only reported.
    Call this before ``DMChainCanInterface`` is constructed, with the same ``motor_list``; see the module
    docstring for why there is no other valid moment. Raises ``RuntimeError`` if any motor could not be
    read, any repair did not stick, or any *steering* motor disagrees on ``PMAX`` or ``VMAX`` -- in each
    case the base must not start on a configuration nobody can vouch for. Every other scaling mismatch is
    logged and the base starts: the same two registers on a drive or rail motor, and ``TMAX`` on any
    motor, which scales only the torque the base reports.
    """
    motors = [(int(motor_id), str(motor_type)) for motor_id, motor_type in motor_list]
    started = time.monotonic()
    logger.info(
        "checking motor control mode and scaling on %s: %s",
        channel,
        ", ".join(f"{motor_id} {motor_type}" for motor_id, motor_type in motors),
    )
    try:
        iface = RawCanInterface(channel=channel, bustype="socketcan", name="flow_base_motor_config_check")
    except BUS_ERRORS as e:
        raise RuntimeError(
            f"could not open {channel} to check the motor configuration: {e}. Check the interface is up "
            f"(ip link show {channel})."
        ) from e

    try:
        wrong: list[tuple[int, Scalar]] = []
        mis_scaled: list[int] = []
        unreadable: list[int] = []
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
            if not is_speed_mode(actual):
                wrong.append((motor_id, actual))
            if _check_scaling(channel, motor_id, motor_type, values):
                mis_scaled.append(motor_id)

        if unreadable:
            raise RuntimeError(
                f"motor configuration check failed on {channel}: motor(s) {unreadable} did not answer, so "
                "nothing was written on any motor and the base was NOT started. See the ERROR lines above."
            )
        if mis_scaled:
            # Before any repair: a bus whose scaling we do not believe is not one to commit Flash writes on.
            raise RuntimeError(
                f"motor configuration check failed on {channel}: steering motor(s) {mis_scaled} do not hold "
                "the PMAX or VMAX the driver decodes their position and velocity feedback with, so the "
                "angles and rates the control loop steers and navigates on are mis-scaled and the base was "
                "NOT started. Nothing was written. See the ERROR lines above for which register on which "
                "motor. A wrong TMAX is reported the same way but never blocks: it scales only the torque "
                "the base reports."
            )

        failed = [motor_id for motor_id, actual in wrong if not _repair(iface, channel, motor_id, actual)]
    finally:
        iface.close()

    if failed:
        raise RuntimeError(
            f"motor configuration check failed on {channel}: motor(s) {failed} would not take the new control "
            "mode, so the base was NOT started. See the ERROR lines above for what to do about each."
        )
    logger.info(
        "motor configuration check passed on %s in %.2f s (%d motor(s) repaired and saved to Flash)",
        channel,
        time.monotonic() - started,
        len(wrong),
    )
