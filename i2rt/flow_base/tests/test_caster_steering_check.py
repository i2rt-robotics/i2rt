"""Tests for the Flow Base runtime steering-motor check.

Everything here is hardware-free. The detectors are driven by a closed-loop simulation of the
controller's *own* steer law integrated at the control rate, rather than by hand-picked arrays: the
whole design rests on the claim that a healthy caster converges to ``phi_eq`` fast enough that the
thresholds have margin, and only a simulation can pin that claim.

``_steer_row`` is transcribed from ``flow_base_controller.py`` (the ``C_steer`` rows in
``update_state``). It is deliberately a copy rather than an import: these tests exist to detect the
module drifting away from the control law, so if someone edits ``h_x``, ``h_y``, ``b_x`` or the row
expressions, ``test_equilibrium_zeroes_the_controllers_steer_row`` must fail loudly. **Edit the two
together.**

The clock is injected everywhere. The monitor runs on ``time.monotonic`` in production; here it runs
on a counter so a two-second hold-off costs no wall time and a clock step can be simulated exactly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from i2rt.flow_base import caster_steering_check as csc
from i2rt.flow_base.flow_base_controller import b_x, h_x, h_y

DT = 1.0 / 200.0
MOTOR_TAU = 0.020
"""Motor speed-loop time constant used by the simulation. Not a property of the code under test."""

RUNAWAY_SIM_RATE = 15.0
"""rad/s reported by a caster in ``runaway`` mode: comfortably over ``RUNAWAY_RATE`` and inside the
motor's own 30 rad/s limit, so it is a rate the hardware could really produce."""

TRANSPORT_LAG_CYCLES = 3
"""Command-to-feedback delay, ~15 ms. Also not a property of the code under test."""

TWISTS = [
    np.array([0.5, 0.0, 0.0]),
    np.array([0.0, 0.4, 0.0]),
    np.array([0.0, 0.0, 1.2]),
    np.array([0.3, 0.2, 0.8]),
    np.array([-0.25, 0.1, -0.6]),
    np.array([0.05, -0.05, 0.3]),
]


def _steer_row(phi: np.ndarray, twist: np.ndarray) -> np.ndarray:
    """The controller's own ``C_steer @ twist``. Transcribed from flow_base_controller.update_state."""
    s, c = np.sin(phi), np.cos(phi)
    return (s / b_x) * twist[0] + (-c / b_x) * twist[1] + ((-h_x * c - h_y * s) / b_x - 1.0) * twist[2]


class Clock:
    """Injected monotonic clock. Advanced explicitly by the simulation."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _monitor(clock: Clock) -> csc.CasterSteeringMonitor:
    return csc.CasterSteeringMonitor(h_x, h_y, b_x, DT, clock=clock)


def _longest_continuous_run(cycle_times: list[float]) -> float:
    """Longest unbroken span, in seconds, covered by ``cycle_times`` (one timestamp per cycle)."""
    longest = run = 0.0
    previous = None
    for t in cycle_times:
        run = DT if previous is None or t - previous > 1.5 * DT else run + DT
        longest = max(longest, run)
        previous = t
    return longest


def simulate(
    twist_of_t,  # noqa: ANN001 - Callable[[float], np.ndarray]
    duration: float = 8.0,
    stalled: np.ndarray | None = None,
    runaway: np.ndarray | None = None,
    phi0_offset: np.ndarray | None = None,
    brake_log: list[float] | None = None,
) -> csc.CasterFault | None:
    """Run the controller's steer law against the monitor and return the first fault, if any.

    The casters start at the equilibrium for the initial twist (plus ``phi0_offset``) and each
    steering motor tracks its commanded rate through a transport delay and a first-order speed loop.
    ``stalled`` casters do not move at all; ``runaway`` ones ignore the command outright and spin at
    ``RUNAWAY_SIM_RATE``.

    The controller's caster-flip brake is wired in at its real position in the cycle -- ``hold_off`` on
    the measured steer rate first, ``update`` on that same sample second, exactly as
    ``flow_base_controller.control_loop`` does it. Every test below therefore runs against the ordering
    that ships, which is the ordering that once made the runaway backstop unreachable. Only the
    ``hold_off`` half of the brake is modelled and not its rewrite of the OTG target: leaving the
    commanded twist slewing straight through the hold-off is the conservative choice.
    """
    clock = Clock()
    monitor = _monitor(clock)
    stalled = np.zeros(h_x.size, dtype=bool) if stalled is None else stalled
    runaway = np.zeros(h_x.size, dtype=bool) if runaway is None else runaway
    _, phi = csc.equilibrium_steering(twist_of_t(0.0), h_x, h_y, b_x)
    phi = np.nan_to_num(phi) + (np.zeros(h_x.size) if phi0_offset is None else phi0_offset)
    measured = np.where(runaway, RUNAWAY_SIM_RATE, 0.0)
    pipeline = [np.zeros(h_x.size)] * TRANSPORT_LAG_CYCLES

    for step in range(int(duration / DT)):
        twist = twist_of_t(step * DT)
        command = np.clip(_steer_row(phi, twist), -csc.STEER_VEL_MAX, csc.STEER_VEL_MAX)
        if np.max(np.abs(measured)) > csc.RUNAWAY_RATE:
            monitor.hold_off("caster-flip brake")
            if brake_log is not None:
                brake_log.append(clock.t)
        fault = monitor.update(twist, phi, command, measured)
        if fault is not None:
            return fault
        pipeline.append(command)
        measured = measured + (pipeline.pop(0) - measured) * (DT / MOTOR_TAU)
        measured = np.where(stalled, 0.0, measured)
        measured = np.where(runaway, RUNAWAY_SIM_RATE, measured)
        phi = phi + measured * DT
        clock.t += DT
    return None


def busy_drive(t: float) -> np.ndarray:
    """A continuously curving drive -- the commanded direction never stops moving."""
    return np.array([0.3 * math.cos(0.8 * t), 0.3 * math.sin(0.8 * t), 0.2 * math.sin(0.5 * t)])


def flip_at(speed: float, when: float = 3.0):  # noqa: ANN201
    """Drive forward at ``speed``, then strafe: a 90 degree change of commanded direction.

    Bounded at 1.0 m/s because this helper models one translational axis at a time, and 1.0 m/s is the
    per-axis limit passed by ``__main__``. The closed-loop plant model stops converging above roughly
    1.4 m/s of caster-point speed: the steer row carries a ``1 / b_x`` = 50x gain, so sustained clipping
    at ``STEER_VEL_MAX`` plus ``TRANSPORT_LAG_CYCLES`` of delay becomes a limit cycle. At 1.5 m/s this
    single-axis flip is still oscillating at 60 s of simulated time, against two bursts settled by
    t = 3.2 s at 1.0. Each burst is only ~0.085 s, far under ``RUNAWAY_HOLD_S``, so ``simulate`` would
    return None while the model shakes itself apart.

    The guard is not a ceiling on the full commandable twist. ``__main__`` clips vx, vy and w
    independently, so simultaneous maxima (1.0, 1.0, pi) produce a 2.303 m/s caster-point speed. That
    lies outside this model's convergent range and is documented separately below rather than being
    smuggled through a helper whose scalar ``speed`` only describes a pure single-axis flip. Neither
    result is a hardware measurement.
    """
    if not 0.0 < speed <= 1.0:
        raise ValueError(f"flip_at models a single-axis flip only for 0 < speed <= 1.0 m/s, got {speed}")
    return lambda t: np.array([speed, 0.0, 0.0]) if t < when else np.array([0.0, speed, 0.0])


# --- the kinematic claim the whole design rests on ------------------------------------------------


@pytest.mark.parametrize("twist", TWISTS, ids=lambda tw: np.array2string(tw, precision=2))
def test_equilibrium_zeroes_the_controllers_steer_row(twist: np.ndarray) -> None:
    # If this fails, either the module or flow_base_controller's C_steer rows changed without the
    # other. They must be edited together -- see the module docstring.
    _, phi_eq = csc.equilibrium_steering(twist, h_x, h_y, b_x)
    assert np.all(np.isfinite(phi_eq))
    np.testing.assert_allclose(_steer_row(phi_eq, twist), 0.0, atol=1e-12)


def test_pure_translation_points_every_wheel_along_the_twist() -> None:
    # With no rotation the trail correction vanishes and every caster faces the direction of travel.
    twist = np.array([0.3, 0.4, 0.0])
    _, phi_eq = csc.equilibrium_steering(twist, h_x, h_y, b_x)
    np.testing.assert_allclose(phi_eq, math.atan2(0.4, 0.3), atol=1e-12)


@pytest.mark.parametrize("w", [1.0, -1.0])
def test_rotation_offsets_equilibrium_by_the_trail_term(w: float) -> None:
    # Pins the SIGN of the asin term. A flipped sign still satisfies "close to atan2" and would
    # otherwise only show up as a small systematic bias on every turn.
    twist = np.array([0.5, 0.0, w])
    speed, phi_eq = csc.equilibrium_steering(twist, h_x, h_y, b_x)
    u_x, u_y = twist[0] - w * h_y, twist[1] + w * h_x
    expected = np.arctan2(u_y, u_x) + np.arcsin(b_x * w / speed)
    np.testing.assert_allclose(phi_eq, expected, atol=1e-12)
    assert np.sign(b_x * w) == np.sign(np.mean(csc.wrap_to_pi(phi_eq - np.arctan2(u_y, u_x))))


def test_no_equilibrium_exists_when_the_centre_of_rotation_sits_on_a_caster() -> None:
    # Spinning about a point 10 mm from caster 0's steering axis: its own |u| is tiny while the other
    # three are moving normally, so |b_x*w|/|u_0| exceeds 1 and no equilibrium angle exists for it.
    # It must come back NaN, gated out per caster -- and NaN must never satisfy a trip comparison.
    twist = np.array([0.21, 0.20, 1.0])
    speed, phi_eq = csc.equilibrium_steering(twist, h_x, h_y, b_x)
    assert np.all(np.isfinite(speed))
    assert math.isnan(phi_eq[0]), "caster 0 has no equilibrium here"
    assert np.all(np.isfinite(phi_eq[1:])), "the other three are unaffected"
    assert not bool(np.abs(phi_eq[0]) > csc.HEADING_TRIP), "NaN must not read as a violation"


def test_a_zero_twist_has_no_defined_heading() -> None:
    _, phi_eq = csc.equilibrium_steering(np.zeros(3), h_x, h_y, b_x)
    assert np.all(np.isnan(phi_eq))


@pytest.mark.parametrize("offset", [0.0, math.pi, -math.pi, 2 * math.pi, 5 * math.pi])
def test_backwards_and_multi_turn_angles_are_not_errors(offset: float) -> None:
    # A caster rolling backwards produces identical base motion, and steer_pos is an unwrapped
    # multi-turn accumulator, so both must fold to zero error.
    np.testing.assert_allclose(csc.wrap_to_half_pi(np.array([offset])), 0.0, atol=1e-12)


def test_ninety_degrees_is_the_largest_reportable_heading_error() -> None:
    # Guards against anyone raising HEADING_TRIP past the fold range, where it could never fire.
    folded = csc.wrap_to_half_pi(np.linspace(-10.0, 10.0, 4001))
    assert np.max(np.abs(folded)) <= math.pi / 2 + 1e-12
    assert csc.HEADING_TRIP < math.pi / 2


def test_execution_error_clips_the_command_to_the_motor_limit() -> None:
    # A 90 degree flip at 1 m/s asks for 50 rad/s; the firmware delivers 30. Without the clip that
    # reads as a 40% tracking failure on every fast direction change.
    command = np.array([50.0, -50.0, 5.0, 0.0])
    measured = np.array([csc.STEER_VEL_MAX, -csc.STEER_VEL_MAX, 5.0, 0.0])
    clipped, error = csc.execution_errors(command, measured)
    np.testing.assert_allclose(clipped, [csc.STEER_VEL_MAX, -csc.STEER_VEL_MAX, 5.0, 0.0])
    np.testing.assert_allclose(error, 0.0, atol=1e-12)


# --- false positives: a healthy base must survive every legitimate manoeuvre ----------------------


@pytest.mark.parametrize("speed", [0.05, 0.2, 0.5, 1.0])
def test_a_healthy_ninety_degree_flip_never_trips(speed: float) -> None:
    # The worst legitimate transient there is: the commanded direction jumps 90 degrees while the
    # caster is still pointing the old way.
    assert simulate(flip_at(speed)) is None


def test_low_speed_perpendicular_acceleration_never_trips() -> None:
    # At 0.1 m/s a healthy caster legitimately lags ~92 degrees behind a slewing phi_eq. This is the
    # test that fails on any design using a bare settle time instead of the lag budget.
    assert simulate(lambda t: np.array([0.1, min(0.8 * max(t - 3.0, 0.0), 0.5), 0.0])) is None


def test_repeated_reversals_never_trip() -> None:
    assert simulate(lambda t: np.array([0.5 * (1 if int(t / 0.8) % 2 == 0 else -1), 0.0, 0.0])) is None


def test_a_long_continuously_curving_drive_never_trips() -> None:
    # Twenty seconds where the commanded direction never stops moving -- the case most likely to
    # accumulate a spurious run in any of the detectors.
    assert simulate(busy_drive, duration=20.0) is None


@pytest.mark.parametrize("twist", [np.array([0.0, 0.0, 1.2]), np.array([0.02, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])])
def test_spins_and_near_standstill_never_trip(twist: np.ndarray) -> None:
    assert simulate(lambda t: twist) is None


def test_a_startup_misalignment_converges_instead_of_tripping() -> None:
    # The casters are wherever they were left; the loop pulls them in. That is not a fault.
    assert simulate(lambda t: np.array([0.3, 0.1, 0.4]), phi0_offset=np.radians([30.0, 0.0, 0.0, 0.0])) is None


def test_a_full_turn_jump_in_the_accumulator_is_ignored() -> None:
    # A stalled control loop can make _update_absolute_positions jump a whole turn. C uses only sin
    # and cos, so it is physically meaningless and must stay so here.
    assert simulate(lambda t: np.array([0.3, 0.1, 0.4]), phi0_offset=np.array([2 * math.pi, 0.0, 0.0, 0.0])) is None


def test_a_backwards_clock_step_does_not_trip() -> None:
    # These run on lab machines whose clocks get stepped. A negative dt must be discarded, not
    # treated as an instantly-satisfied persistence window.
    clock = Clock()
    monitor = _monitor(clock)
    twist = np.array([0.3, 0.0, 0.0])
    _, phi = csc.equilibrium_steering(twist, h_x, h_y, b_x)
    for _ in range(1000):
        clock.t += DT
        monitor.update(twist, phi, np.zeros(4), np.zeros(4))
    clock.t -= 1.0
    assert monitor.update(twist, phi, np.zeros(4), np.zeros(4)) is None


@pytest.mark.parametrize("reason", ["caster-flip brake", "command-stream timeout"])
def test_a_holdoff_clears_the_command_referenced_runs_and_spares_the_backstop(reason: str) -> None:
    # Both callers rewrite the OTG target mid-manoeuvre, so the rate and heading rows lose the very
    # reference they are judged against. The runaway row is judged against nothing a caller can disturb.
    clock = Clock()
    monitor = _monitor(clock)
    clock.t += csc.STARTUP_HOLDOFF_S + 1.0
    twist = np.array([0.3, 0.0, 0.0])
    _, phi_eq = csc.equilibrium_steering(twist, h_x, h_y, b_x)
    measured = np.array([0.0, RUNAWAY_SIM_RATE, 0.0, 0.0])
    for _ in range(20):
        clock.t += DT
        monitor.update(twist, phi_eq, np.full(4, 6.0), measured)
    assert np.isfinite(monitor._bad_since[csc._RATE]).any(), "a rate run should be accumulating"
    assert np.isfinite(monitor._bad_since[csc._RUNAWAY]).any(), "a runaway run should be accumulating"

    monitor.hold_off(reason)
    assert not np.isfinite(monitor._bad_since[csc._RATE]).any()
    assert not np.isfinite(monitor._bad_since[csc._HEADING]).any()
    assert np.isfinite(monitor._bad_since[csc._RUNAWAY]).any(), "the backstop must survive a hold-off"


# --- the observability limit: these MUST NOT be "fixed" -------------------------------------------


def test_a_lost_steering_zero_is_not_detected() -> None:
    """Pins an observability limit, not a bug -- do not "fix" this test.

    ``motor_offset`` and ``motor_direction`` are applied symmetrically to command and feedback, and
    the controller rebuilds ``C`` from the *reported* angle every tick, so the reported angle
    converges to ``phi_eq`` no matter where the wheel physically points. Nothing in the software
    frame is inconsistent. Catching this needs an external reference; see README section 5.
    """
    assert simulate(busy_drive, duration=20.0, phi0_offset=np.radians([45.0, 0.0, 0.0, 0.0])) is None


def test_a_reversed_steering_direction_is_not_detected() -> None:
    """Also an observability limit -- see :func:`test_a_lost_steering_zero_is_not_detected`.

    A sign error in ``STEERING_DIRECTION`` negates the outgoing command and the incoming feedback
    alike, so the reported angle still integrates the commanded rate correctly. There is no runaway.
    """
    clock = Clock()
    monitor = _monitor(clock)
    twist = np.array([0.3, 0.1, 0.4])
    _, phi = csc.equilibrium_steering(twist, h_x, h_y, b_x)
    phi = np.nan_to_num(phi)
    measured = np.zeros(4)
    for _ in range(int(10.0 / DT)):
        command = np.clip(_steer_row(phi, twist), -csc.STEER_VEL_MAX, csc.STEER_VEL_MAX)
        # Caster 2's direction constant is wrong: the physical wheel turns the other way, but the
        # driver mirrors the reading back, so the reported rate matches the command exactly.
        assert monitor.update(twist, phi, command, measured) is None
        measured = measured + (command - measured) * (DT / MOTOR_TAU)
        phi = phi + measured * DT
        clock.t += DT


def test_a_mis_scaled_pmax_is_not_detected_at_runtime() -> None:
    """Also an observability limit, and this one used to be a detector -- do not "fix" this test.

    A firmware ``PMAX`` of 12.5 rad against ``get_motor_constants``' pi makes every reported steering
    angle 0.251x the truth. It hides for the same reason as the two limits above: the controller
    rebuilds ``C`` from the *reported* angle, so the loop stays autonomous in reported coordinates, the
    reported angle still converges to ``phi_eq``, and no detector has anything to see. A
    position-versus-velocity window that could see it was removed deliberately -- a firmware register
    cannot change while the base is driving, so ``motor_drivers/motor_check.py`` reads it directly at startup
    instead, which is exact and needs no thresholds.
    """
    clock = Clock()
    monitor = _monitor(clock)
    pos_scale = 2 * math.pi / 25.0
    _, phi = csc.equilibrium_steering(busy_drive(0.0), h_x, h_y, b_x)
    phi = np.nan_to_num(phi)
    measured = np.zeros(h_x.size)
    pipeline = [np.zeros(h_x.size)] * TRANSPORT_LAG_CYCLES
    for step in range(int(20.0 / DT)):
        twist = busy_drive(step * DT)
        # Built from the *reported* angle, because that is what the controller does -- it builds C from
        # self.q. Driving it from the true angle instead manufactures a heading error the real system
        # never sees, and the heading detector then trips systemically on a fault that is not physical.
        command = np.clip(_steer_row(phi * pos_scale, twist), -csc.STEER_VEL_MAX, csc.STEER_VEL_MAX)
        assert monitor.update(twist, phi * pos_scale, command, measured) is None
        pipeline.append(command)
        measured = measured + (pipeline.pop(0) - measured) * (DT / MOTOR_TAU)
        phi = phi + measured * DT
        clock.t += DT


# --- true positives -------------------------------------------------------------------------------


def test_a_stalled_steering_motor_trips_the_rate_detector() -> None:
    fault = simulate(flip_at(0.2), stalled=np.array([False, False, True, False]))
    assert fault is not None
    assert fault.kind == "STEER_RATE_NOT_EXECUTED"
    assert fault.caster == 2


def test_the_report_names_the_caster_and_its_can_id() -> None:
    fault = simulate(flip_at(0.2), stalled=np.array([False, False, True, False]))
    assert fault is not None
    report = fault.render()
    assert "caster 2" in report
    assert f"CAN id {csc.steering_can_id(2)}" in report
    # The operator must not be sent chasing a calibration fault this check cannot see.
    assert "cannot see a wrong steering zero" in report


def test_a_caster_jammed_off_equilibrium_trips() -> None:
    fault = simulate(
        lambda t: np.array([0.2, 0.0, 0.0]),
        stalled=np.array([True, False, False, False]),
        phi0_offset=np.radians([75.0, 0.0, 0.0, 0.0]),
    )
    assert fault is not None
    assert fault.caster == 0


def test_a_stalled_motor_already_at_the_right_angle_is_not_reported() -> None:
    # Correct and deliberate: the commanded rate is ~0, so there is nothing to track, and the caster
    # is pointing exactly where it should. It becomes detectable the moment the command changes.
    assert simulate(lambda t: np.array([0.3, 0.0, 0.0]), stalled=np.array([True, False, False, False])) is None


# --- the backstop, and the brake that shares its threshold ----------------------------------------


def test_a_sustained_runaway_trips_even_though_the_brake_holds_off_every_cycle() -> None:
    """The regression for the coupling that made this detector unreachable.

    A caster spinning at ``RUNAWAY_SIM_RATE`` is over the controller's caster-flip-brake threshold on
    every cycle, and that threshold *is* ``RUNAWAY_RATE`` -- so ``hold_off`` runs immediately before
    every ``update``, on the same sample. A hold-off that cleared the runaway row would make this
    detector's trip condition imply its own suppression, and the one failure mode that needs no command
    reference at all could never be reported.
    """
    fault = simulate(lambda t: np.array([0.3, 0.0, 0.0]), runaway=np.array([False, True, False, False]))
    assert fault is not None, "a caster spinning at 15 rad/s must be reported"
    assert fault.kind == "SUSTAINED_STEER_RATE"
    assert fault.caster == 1
    # Rendered from a snapshot that, on this path, is only ever taken inside a hold-off.
    assert "+15.00" in fault.table
    assert "hold-off ACTIVE (caster-flip brake" in fault.table
    assert f"CAN id {csc.steering_can_id(1)}" in fault.render()


def test_the_backstop_needs_a_full_second_of_continuous_violation() -> None:
    # The margin between a runaway and a legitimate flip is duration, not threshold, so pin the duration.
    clock = Clock()
    monitor = _monitor(clock)
    clock.t += csc.STARTUP_HOLDOFF_S + 1.0
    twist = np.array([0.3, 0.0, 0.0])
    _, phi = csc.equilibrium_steering(twist, h_x, h_y, b_x)
    phi = np.nan_to_num(phi)
    measured = np.array([0.0, RUNAWAY_SIM_RATE, 0.0, 0.0])
    started, tripped_at = clock.t, None
    for _ in range(int(3.0 / DT)):
        clock.t += DT
        monitor.hold_off("caster-flip brake")  # exactly what the controller does, every cycle
        if monitor.update(twist, phi, np.zeros(4), measured) is not None:
            tripped_at = clock.t - started
            break
        phi = phi + measured * DT
    assert tripped_at is not None
    assert csc.RUNAWAY_HOLD_S <= tripped_at <= csc.RUNAWAY_HOLD_S + 3 * DT


def test_a_gap_in_the_sample_stream_does_not_let_a_stale_runaway_run_trip() -> None:
    # A dt outside the accepted window means the intervening interval was never observed, so a run that
    # began before it is not evidence of anything continuous across it. These are lab machines whose
    # control loops get preempted; without this, one over-threshold cycle after a 100 s stall reads as
    # 100 s of sustained runaway.
    clock = Clock()
    monitor = _monitor(clock)
    clock.t += csc.STARTUP_HOLDOFF_S + 1.0
    twist = np.array([0.3, 0.0, 0.0])
    _, phi = csc.equilibrium_steering(twist, h_x, h_y, b_x)
    phi = np.nan_to_num(phi)
    measured = np.array([0.0, 0.0, RUNAWAY_SIM_RATE, 0.0])
    for _ in range(4):
        clock.t += DT
        monitor.hold_off("caster-flip brake")
        assert monitor.update(twist, phi, np.zeros(4), measured) is None
        phi = phi + measured * DT

    clock.t += 100.0  # the control loop blocked, or the clock was stepped
    monitor.hold_off("caster-flip brake")
    assert monitor.update(twist, phi, np.zeros(4), measured) is None, "the stalled sample is discarded"
    clock.t += DT
    monitor.hold_off("caster-flip brake")
    assert monitor.update(twist, phi, np.zeros(4), measured) is None, (
        "one over-threshold cycle after a gap is not a second of continuous violation"
    )


def test_the_full_commandable_twist_is_not_the_flip_helper_ceiling() -> None:
    """Pin the command-envelope distinction without presenting this plant model as hardware evidence.

    The CLI clips vx, vy and w independently, so reversing all three saturated axes is commandable and
    reaches 2.303 m/s at one caster. It is outside this test plant's convergent range: brake activity
    persists to the end of the run. The 0.125 s assertion records only the longest continuous burst in
    that known limit cycle and its margin to the one-second backstop; a green result does not say the
    real base is stable there.
    """
    full_twist = np.array([1.0, 1.0, math.pi])
    speed, _ = csc.equilibrium_steering(full_twist, h_x, h_y, b_x)
    assert np.max(speed) == pytest.approx(2.30279015)

    brake: list[float] = []
    fault = simulate(lambda t: full_twist if t < 10.0 else -full_twist, duration=20.0, brake_log=brake)
    assert fault is None
    assert brake[-1] > 19.9, "the model must remain visibly non-convergent, not masquerade as a safe flip"
    longest = _longest_continuous_run(brake)
    assert longest == pytest.approx(0.125, abs=2 * DT)
    assert longest < 0.2 * csc.RUNAWAY_HOLD_S


@pytest.mark.parametrize(("speed", "expected_longest"), [(0.45, 0.080), (0.5, 0.075), (1.0, 0.080)])
def test_a_legitimate_flip_fires_the_brake_without_tripping_the_backstop(
    speed: float, expected_longest: float
) -> None:
    # The brake and the backstop share a threshold, so a fast flip really does fire the brake -- that is
    # the coupling. What separates them is duration: measured stays above RUNAWAY_RATE for 0.080 s at
    # 0.45 m/s, 0.075 s at 0.5 and 0.080 s at 1.0 (twice the base's default max_vel), against a 1.0 s
    # hold. 0.45 is here because it is the worst case in this helper's pure single-axis 90-degree-flip
    # range, not a low-speed curiosity: a 0.01-step sweep puts the maximum at 0.080 s, first reached at
    # 0.44 m/s and tying the 1.0 m/s endpoint. The independently clipped x+y+w command envelope is wider
    # and is pinned separately above; it must not be conflated with this model's convergent flip range.
    brake: list[float] = []
    assert simulate(flip_at(speed), brake_log=brake) is None
    assert brake, "this manoeuvre must actually fire the brake, or the test proves nothing"
    longest = _longest_continuous_run(brake)
    assert longest == pytest.approx(expected_longest, abs=2 * DT)
    assert longest < 0.2 * csc.RUNAWAY_HOLD_S


@pytest.mark.parametrize("speed", [0.05, 0.2, 0.25])
def test_a_slow_flip_never_even_reaches_the_brake_threshold(speed: float) -> None:
    # Peak measured steer rate is 2.47 rad/s at 0.05 m/s and 9.52 at 0.2, both well under 12.56. 0.25 is
    # the knee: it peaks at 11.78 and the brake first fires just above it, at about 0.27 m/s. It belongs
    # on this test rather than the one above, which requires the brake to fire.
    brake: list[float] = []
    assert simulate(flip_at(speed), brake_log=brake) is None
    assert not brake


def test_a_runaway_at_startup_is_caught_inside_the_startup_holdoff() -> None:
    # Measured velocity needs no filter warm-up and no command reference, so it is meaningful from cycle
    # 1 -- and a motor that comes up spinning is what the startup window can least afford to ignore
    # (2 s at 12.56 rad/s is four turns of a steering axis). The guard is the 1.0 s of continuous
    # violation, not the hold-off.
    clock = Clock()
    monitor = _monitor(clock)  # __init__ applies STARTUP_HOLDOFF_S
    twist = np.array([0.3, 0.0, 0.0])
    _, phi = csc.equilibrium_steering(twist, h_x, h_y, b_x)
    phi = np.nan_to_num(phi)
    measured = np.array([RUNAWAY_SIM_RATE, 0.0, 0.0, 0.0])
    fault = None
    for _ in range(int(1.5 / DT)):
        clock.t += DT
        fault = fault or monitor.update(twist, phi, np.zeros(4), measured)
        phi = phi + measured * DT
    assert clock.t < csc.STARTUP_HOLDOFF_S, "the whole run must sit inside the startup hold-off"
    assert fault is not None and fault.kind == "SUSTAINED_STEER_RATE"
    assert fault.caster == 0


def test_the_startup_holdoff_still_silences_the_command_referenced_detectors() -> None:
    # The exemption is narrow. A caster tracking nothing would trip the rate row in RATE_HOLD_S, and
    # during startup it must not: those filters have not settled and the OTG has not started.
    clock = Clock()
    monitor = _monitor(clock)
    twist = np.array([0.3, 0.0, 0.0])
    _, phi_eq = csc.equilibrium_steering(twist, h_x, h_y, b_x)
    for _ in range(int(1.5 / DT)):
        clock.t += DT
        assert monitor.update(twist, phi_eq, np.full(4, 6.0), np.zeros(4)) is None
    assert clock.t < csc.STARTUP_HOLDOFF_S


def test_a_parked_base_does_not_decay_the_armed_fraction() -> None:
    """``ARMED_WARN_FRACTION`` is scoped to "while the base is moving", so parking must not erode it.

    Integrating the EWMA on every cycle instead meant a couple of minutes parked took it under the
    threshold, and the first non-held-off cycle once the operator started driving then announced that
    the check was not protecting them -- at exactly the moment it started to.
    """
    clock = Clock()
    monitor = _monitor(clock)
    clock.t += csc.STARTUP_HOLDOFF_S + 1.0
    twist = np.array([0.3, 0.0, 0.0])
    _, phi = csc.equilibrium_steering(twist, h_x, h_y, b_x)
    phi = np.nan_to_num(phi)
    for _ in range(200):  # drive first, so the fraction is a measurement and not its initial value
        clock.t += DT
        monitor.update(twist, phi, np.zeros(4), np.zeros(4))

    before = monitor._armed_frac.copy()
    for _ in range(2000):  # 10 s parked: zero twist, so every caster's |u| is under U_MIN
        clock.t += DT
        monitor.update(np.zeros(3), phi, np.zeros(4), np.zeros(4))
    np.testing.assert_array_equal(monitor._armed_frac, before)


def test_a_fault_latches_and_does_not_drift() -> None:
    # The report is logged once and then read again after the base has stopped; it must not change
    # underneath the operator while the ramp runs.
    clock = Clock()
    monitor = _monitor(clock)
    clock.t += csc.STARTUP_HOLDOFF_S + 1.0
    twist = np.array([0.3, 0.0, 0.0])
    _, phi_eq = csc.equilibrium_steering(twist, h_x, h_y, b_x)
    first = None
    for _ in range(int(2.0 / DT)):
        clock.t += DT
        fault = monitor.update(twist, phi_eq, np.full(4, 6.0), np.zeros(4))
        if fault is not None and first is None:
            first = fault
    assert first is not None
    for _ in range(50):
        clock.t += DT
        assert monitor.update(twist, phi_eq, np.zeros(4), np.zeros(4)) is first


def test_history_csv_has_a_header_and_one_row_per_cycle() -> None:
    clock = Clock()
    monitor = _monitor(clock)
    twist = np.array([0.3, 0.0, 0.0])
    _, phi_eq = csc.equilibrium_steering(twist, h_x, h_y, b_x)
    for _ in range(25):
        clock.t += DT
        monitor.update(twist, phi_eq, np.zeros(4), np.zeros(4))
    lines = monitor.history_csv().strip().splitlines()
    assert lines[0].startswith("t,vx,vy,w,")
    assert len(lines) == 26
