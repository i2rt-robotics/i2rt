"""Runtime check that the Flow Base's four steering motors are executing their commands.

The base is steered entirely open-loop in position. ``flow_base_controller`` commands a steer *rate*
(``dq_d = C @ dx_d_local``) and the wheels only reach the right heading because the caster law drives
that rate to zero as the angle converges; nothing anywhere compares a measured steering angle against
a model prediction. So a steering motor that stalls, jams, saturates or quietly stops accepting
commands raises no error at all: the base veers, and the odometry -- derived from the same feedback --
agrees with itself. The only existing guard is ``Vehicle.running()``, which just proxies
``motor_interface.running`` and therefore fires only once the CAN chain thread has already died.

This module closes that gap for the steering motors (CAN ids 1/3/5/7, chain indices 0/2/4/6). It is
pure: no CAN access, no locks, no I/O. ``CasterSteeringMonitor.update`` is called once per control
cycle from the control thread and returns a ``CasterFault`` when one is confirmed.

**Why only the steering motors.** Two reasons, and the second is why this should not be "completed"
later by pointing the rate detector at ``dq_d[1::2]`` as well. First, the asymmetry is real: in
``Vehicle.update_state`` every row of ``C``, ``C_p``, ``CpT_Cqinv`` and ``C_pinv`` is a function of
``sin``/``cos`` of the *steering* angles alone, and drive feedback enters at exactly one place, the
``C_pinv @ dq`` that produces odometry. A wrong steering reading therefore corrupts **command
generation**, inside the loop, where it compounds; a wrong drive reading corrupts **odometry output**,
where it does not. Second, a drive-side rate check is not viable at all: how fast a wheel actually turns
depends on the floor and the load, so a stalled motor is indistinguishable from one pushing a heavy cart,
climbing a threshold, or pressed against a wall. That is precisely the physical-disturbance family of
false positives the design below suppresses by construction, and adding it back on the drive side would
give it a way in.

**What this can and cannot detect.** In ``ControlMode.VEL`` the steering loop, expressed in *reported*
coordinates, is autonomous::

    d(phi_rep)/dt = vel_sim = vel_real*dir = (cmd_sim*dir)*dir = C_steer(phi_rep) @ xi_cmd

``motor_offset`` never enters (VEL mode packs only ``vel``), ``motor_direction`` squares to 1, and the
physical world never feeds back. The fixed point is ``phi_eq(xi_cmd)`` *in reported coordinates* and it
is globally attracting, so the reported angle converges to the expected one no matter where the wheel
is actually pointing. **A lost firmware steering zero, a wrong STEERING_OFFSET/STEERING_DIRECTION, and
belt slip downstream of the encoder are therefore structurally invisible here** -- they are errors in
the map between the software frame and the world, applied symmetrically to command and feedback. They
stay the job of the manual commissioning procedure in the Flow Base README section 5. Do not "fix" the
tests that assert this; they pin an observability limit, not a bug.

**A mis-scaled firmware register is invisible here too, and is checked elsewhere.** The VEL command is a
raw IEEE float in physical rad/s with no ``VELOCITY_MAX`` scaling, while position and velocity feedback
are unpacked with ``POSITION_MIN/MAX`` and ``VELOCITY_MIN/MAX`` from ``MotorType.get_motor_constants``,
so a firmware register that disagrees rescales every reading: with ``PMAX`` at 12.5 rad instead of pi,
every reported steering angle is 0.251x the truth and feeds ``C`` directly. It hides for the same reason
as the paragraph above, though -- the controller rebuilds ``C`` from the *reported* angle, so the loop
stays autonomous in reported coordinates, the reported angle still converges to ``phi_eq``, and no
detector here has anything to see. :func:`i2rt.motor_drivers.motor_check.verify_motor_config` reads those three registers
from every motor at startup and compares them against the same constants. That is the right place for it:
a firmware register cannot change while the base is driving, so reading it once is both exact and
sufficient, where inferring a mismatch from motion needs thresholds and a moving base to work with.

**Why the reference twist is the commanded one.** Every detector here is judged against
``AXIS_SIGN * dx_d_local``, never against measured odometry. Two reasons, and the second is the
strongest robustness property in the design: measured odometry is ``C_pinv @ dq``, computed from the
same reported angles, so a faulty caster would pull the least-squares fit toward itself and dilute its
own residual; and ``phi_eq(xi_cmd)`` is the fixed point of the *commanded* law, whose convergence
constant ``tau = |b_x|/|u_i|`` is a property of that law alone. It holds with the wheels off the
ground, with the base jammed against a wall, while someone shoves it, and on a slippery floor. The
entire physical-disturbance family of false positives -- blocked base, wheel slip, carpet, thresholds,
bench testing -- is suppressed by construction rather than by tuning. Anyone "improving" this by
switching to measured motion reintroduces all of it.

**Why headings fold modulo pi.** ``phi_eq + pi`` is an equilibrium only when ``w = 0``; for ``w != 0``
the steer command there is exactly ``-2w``, and it is always the *unstable* branch, which is physical
caster trail -- a backwards wheel flutters and flips. Folding to ``[-pi/2, pi/2)`` costs nothing real,
removes the largest source of legitimate transient error (a commanded reversal), and makes a full-turn
jump in the unwrapped position accumulator a non-event, which is correct because ``C`` uses only sin
and cos of the angle.

Every timer, window and filter here runs on ``time.monotonic``. The controller's own loop uses
``time.time()``; these machines are lab Pis and x86 boxes whose clocks get stepped, and a negative or
enormous ``dt`` would satisfy any persistence window instantly.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# --- Gates -------------------------------------------------------------------------------------
U_MIN = 0.05
"""m/s. Below this a caster's steering-axis velocity has no well-defined direction and ``tau`` is
unbounded. Applied **per caster**, never to the base twist: spinning about one caster leaves that
caster's own ``|u_i|`` near zero while the other three are moving fast."""

MAX_SIN_ARG = 0.9
"""``|b_x*w| / |u_i|`` above 1 means no equilibrium heading exists (the instantaneous centre of
rotation is within the 20 mm caster trail of a steering axis). 0.9 keeps a margin off that cliff."""

DPHI_CMD_MAX = 25.0
"""rad/s. The commanded steer rate peaks at ``|u|/|b_x|`` = 50 rad/s at 1 m/s, well past the motors'
30 rad/s limit -- and ``ControlMode.VEL`` does no software clipping, so the motor silently delivers
less than asked. Disarming above this is mandatory: without it every fast direction change trips."""

LAG_BUDGET = 0.15
"""rad (8.6 deg). A healthy caster chasing a slewing ``phi_eq`` sits at a steady-state lag of
``tau_i * d(phi_eq)/dt``, which is ``0.016/|u|^2`` at 0.8 m/s^2 -- 23 deg at 0.2 m/s and 92 deg at
0.1 m/s. That is entirely legitimate, so the heading detector disarms while the predicted lag exceeds
this. It is strictly better than a bare speed gate: it keeps the detector armed through a slow steady
curve (small slew rate, small lag) and through a straight-line acceleration ramp (``phi_eq`` is exactly
constant there for ``w = 0``), and it disarms during ramp-up when ``|u|`` is small and ``tau`` large --
which a speed-adaptive hold alone does not, because that hold is evaluated at the current, faster
``|u|`` and collapses to its floor."""

LAG_FILTER_S = 0.10
"""EWMA time constant for the ``phi_eq`` slew rate feeding the lag budget."""

# --- Detector 1: steer-rate execution ----------------------------------------------------------
STEER_VEL_MAX = 30.0
"""rad/s, ``MotorType.get_motor_constants("DM4310V").VELOCITY_MAX``. The expectation is clipped to it
because the firmware clamps there while the raw-float command does not."""

RATE_ABS = 1.0
"""rad/s, 68 velocity LSBs (the 12-bit feedback quantises to 60/4096 = 0.0146 rad/s)."""

RATE_REL = 0.5
"""The motor is delivering less than half, or more than 1.5x, of what was asked."""

RATE_HOLD_S = 0.5
"""The longest *continuous* violation a healthy caster produces during a hard 90 deg command step is
0.070 s across 0.1-1.0 m/s, so this is roughly a 7x margin."""

RATE_FILTER_S = 0.15
"""Applied to command and measurement alike, so the shared filter dominates the 1-2 cycle feedback lag
instead of the lag showing up as error."""

# --- Detector 2: heading convergence -----------------------------------------------------------
HEADING_TRIP = math.radians(30.0)
"""With the lag budget capping legitimate lag at 8.6 deg, the modelled legitimate ceiling is ~9 deg
once compliance and backlash are allowed for. This is >3x that and a third of the mod-pi range."""

HEADING_WARN = math.radians(15.0)
"""Logged, never tripped on, so degradation is visible in the field before anything stops."""

SETTLE_K = 6.0
"""Hold for this many time constants: ``exp(-6)`` leaves 0.25% of any initial error."""

MIN_HOLD_S = 0.5
"""Floor for the adaptive hold. At 1 m/s ``tau`` is 20 ms, shorter than the motor's own speed-loop
bandwidth, so without a floor the detector would be racing servo dynamics."""

MAX_HOLD_S = 3.0
"""Ceiling, bounding ``tau`` growth as ``|u|`` approaches ``U_MIN`` (0.4 s there -> 2.4 s hold)."""

# --- Detector 3: runaway backstop ---------------------------------------------------------------
RUNAWAY_RATE = 12.56
"""rad/s, the constant the controller already trusts for its caster-flip brake -- ``flow_base_controller``
imports *this* symbol for that branch, so there is exactly one number. No legitimate steering rate is
sustained above it: at equilibrium the rate is zero, and the flipped branch drifts at only ``-2w``,
bounded by 2*pi. This detector needs no command reference at all, which is its point -- it survives a bug
in the command plumbing that would blind the other two. Note the coupling that follows from sharing the
constant: the brake fires on this same threshold and holds the monitor off every cycle it does, which is
why the runaway row is exempt from hold-offs -- see :meth:`CasterSteeringMonitor.hold_off`."""

RUNAWAY_HOLD_S = 1.0

# --- Hold-offs ----------------------------------------------------------------------------------
STARTUP_HOLDOFF_S = 2.0
DISTURB_HOLDOFF_S = 0.3
"""Applied when the caller reports a discontinuity in the OTG state -- the caster-flip brake, the
command-stream timeout, or a control-loop step-time overrun."""

DT_MIN_RATIO, DT_MAX_RATIO = 0.5, 5.0
"""Accepted range of measured ``dt`` as a multiple of the nominal control period. Outside it the
sample is discarded and a hold-off applied: the state is stale and ``dt`` is untrustworthy."""

SYSTEMIC_CASTERS = 3
"""At or above this many casters failing the same detector at once, report one systemic fault rather than
N independent caster faults. Four mechanically independent casters rarely fail together, so a shared
cause is far likelier -- an edit to the kinematic constants ``h_x``/``h_y``/``b_x`` or to ``AXIS_SIGN``, a
bug in the command path, one open power or E-stop leg feeding the whole chain, or every motor left out of
``CTRL_MODE`` 3. Calling that "caster 0 is broken" sends the operator to the wrong place entirely."""

ARMED_FILTER_S = 60.0
ARMED_WARN_FRACTION = 0.2
"""If a caster is armed less than this fraction of the time while the base is moving, the check is
effectively inactive and says so. An unmonitored safety check that silently never runs is worse than
no check, because it is believed."""

DETECTORS = ("rate", "heading", "runaway")
_RATE, _HEADING, _RUNAWAY = range(len(DETECTORS))
"""Row indices into the ``holds``/``bad``/``_bad_since`` stacks. The stack order built in :meth:`update`
**must** match ``DETECTORS``: ``_first_fault`` names the detector by row index alone."""

_COMMAND_REFERENCED = [_RATE, _HEADING]
"""The rows judged against the *commanded* twist, and so the only ones :meth:`hold_off` suspends. A
caller reporting a discontinuity is telling us the command reference just moved; that says nothing at all
about the measured steering rate.

A list rather than a tuple on purpose: NumPy reads a length-2 *tuple* as a two-dimensional scalar index,
so ``_bad_since[(_RATE, _HEADING)]`` would address one element instead of two rows."""

_CAUSES = {
    "rate": (
        "steering motor {cid} is not executing its commanded velocity: it is mechanically jammed or "
        "stalled, its power/E-stop leg is open, or its CTRL_MODE is not 3 (speed)"
    ),
    "heading": (
        "caster {caster} is not reaching the heading the kinematics demand: its steering motor is "
        "jammed, saturating, or being back-driven by ground load"
    ),
    "runaway": "steering motor {cid} has been spinning faster than any legitimate steering rate",
}

_SYSTEMIC_CAUSE = (
    "{n} of 4 casters failed the '{detector}' check at once. Four mechanically independent casters "
    "rarely fail together, so look for what they share before suspecting {n} separate faults: an edit "
    "to the kinematic constants h_x/h_y/b_x or to AXIS_SIGN, a bug in the command path, one open "
    "power/E-stop leg feeding the whole chain, or motors left out of CTRL_MODE 3. The register "
    "read-back is the cheapest first look:\n"
    "  python i2rt/motor_config_tool/dm_motor_registers.py read-all --motor-id 1 --channel <channel>"
)

_CANNOT_DETECT = (
    "NOTE: this check cannot see a wrong steering zero, a wrong STEERING_DIRECTION/STEERING_OFFSET, a "
    "coupling slipping downstream of the encoder, or a firmware PMAX/VMAX register that disagrees with "
    "MotorType.get_motor_constants -- the first three are applied symmetrically to command and feedback, "
    "and a mis-scaled register leaves the loop self-consistent in reported coordinates. In every case the "
    "software frame stays internally consistent while the wheel is somewhere else. The registers are "
    "checked at startup by motor_drivers/motor_check.py; for the rest, if the wheels are visibly "
    "misaligned but "
    "the table above looks clean, that is the fault, and section 5 of i2rt/flow_base/README.md is how to "
    "find it."
)


def steering_can_id(caster: int) -> int:
    """CAN id of caster ``caster``'s steering motor. Chain order is (steer, drive) per caster."""
    return 2 * caster + 1


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    """Fold into [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def wrap_to_half_pi(angle: np.ndarray) -> np.ndarray:
    """Fold into [-pi/2, pi/2) -- compares the steering *line*, not the ray. See the module docstring."""
    return (angle + math.pi / 2.0) % math.pi - math.pi / 2.0


def equilibrium_steering(
    twist: np.ndarray, h_x: np.ndarray, h_y: np.ndarray, b_x: float
) -> tuple[np.ndarray, np.ndarray]:
    """Steering-axis speed and equilibrium heading per caster, for a body twist ``(vx, vy, w)``.

    ``phi_eq`` is the angle at which the controller's own steer row evaluates to exactly zero, so it
    is the heading that twist is asking each caster to hold. It is NaN where no equilibrium exists.

    NaN is deliberate and load-bearing in one direction only: ``abs(nan) > threshold`` is False, so a
    degenerate sample can never *cause* a trip. It must never be relied on to gate anything, which is
    why callers test ``isfinite`` explicitly.
    """
    vx, vy, w = float(twist[0]), float(twist[1]), float(twist[2])
    u_x = vx - w * h_y
    u_y = vy + w * h_x
    speed = np.hypot(u_x, u_y)
    with np.errstate(divide="ignore", invalid="ignore"):
        sin_arg = b_x * w / speed
    exists = np.isfinite(sin_arg) & (np.abs(sin_arg) <= MAX_SIN_ARG)
    phi_eq = np.where(exists, np.arctan2(u_y, u_x) + np.arcsin(np.clip(sin_arg, -1.0, 1.0)), np.nan)
    return speed, phi_eq


def heading_errors(
    twist: np.ndarray, steer_pos: np.ndarray, h_x: np.ndarray, h_y: np.ndarray, b_x: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(speed, phi_eq, error)`` with the error folded to [-pi/2, pi/2).

    ``steer_pos`` is the driver's unwrapped multi-turn accumulator; the fold handles that for free.
    """
    speed, phi_eq = equilibrium_steering(twist, h_x, h_y, b_x)
    return speed, phi_eq, wrap_to_half_pi(steer_pos - phi_eq)


def execution_errors(steer_vel_cmd: np.ndarray, steer_vel_meas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(command clipped to the motor limit, |clipped command - measured|)``.

    The clip is not cosmetic: a 90 deg flip at 1 m/s asks for 50 rad/s, the firmware delivers 30, and
    without clipping that reads as a 40% tracking failure on every fast direction change.
    """
    clipped = np.clip(steer_vel_cmd, -STEER_VEL_MAX, STEER_VEL_MAX)
    return clipped, np.abs(clipped - steer_vel_meas)


@dataclass(frozen=True)
class CasterFault:
    """A confirmed fault. ``caster`` is None for a systemic one, which implicates no single caster."""

    detector: str
    caster: int | None
    held_for: float
    cause: str
    table: str

    @property
    def kind(self) -> str:
        name = {
            "rate": "STEER_RATE_NOT_EXECUTED",
            "heading": "HEADING_NOT_CONVERGED",
            "runaway": "SUSTAINED_STEER_RATE",
        }[self.detector]
        # Prefix rather than substitute. Systemic is a property of how many casters tripped, not of
        # which detector tripped, and a single substituted name reported every systemic fault as the
        # same thing whatever had actually failed.
        return f"SYSTEMIC_{name}" if self.caster is None else name

    def render(self) -> str:
        if self.caster is None:
            head = f"CASTER STEERING FAULT - {self.kind}"
        else:
            head = (
                f"CASTER STEERING FAULT - caster {self.caster} "
                f"(steering motor CAN id {steering_can_id(self.caster)}): {self.kind}"
            )
        return (
            f"{head}\n"
            f"Held for {self.held_for:.2f} s. The base is being ramped to a stop and the controller "
            f"will exit. This is NOT an E-stop or a motor error.\n\n"
            f"{self.cause}\n\n{self.table}\n{_CANNOT_DETECT}"
        )


class CasterSteeringMonitor:
    """Per-cycle steering-motor checks with continuous-violation persistence.

    Call :meth:`update` once per control cycle from the control thread, and :meth:`hold_off` from any
    branch that deliberately disturbs the OTG state. All state is per caster; a single conforming
    sample clears that caster's run timer for that detector. That is a deliberate false-positive-first
    bias -- it means an intermittent fault whose error keeps crossing zero is caught late or not at
    all, which is the trade we want on a moving base.
    """

    def __init__(
        self,
        h_x: np.ndarray,
        h_y: np.ndarray,
        b_x: float,
        control_period: float,
        clock: Callable[[], float] = time.monotonic,
        history_s: float = 2.0,
    ) -> None:
        self.h_x = np.asarray(h_x, dtype=float)
        self.h_y = np.asarray(h_y, dtype=float)
        self.b_x = float(b_x)
        self.num_casters = self.h_x.size
        self.control_period = float(control_period)
        self._clock = clock

        n = self.num_casters
        self._bad_since = np.full((len(DETECTORS), n), np.inf)
        self._lag = np.zeros(n)
        self._lp_cmd = np.zeros(n)
        self._lp_meas = np.zeros(n)
        self._armed_frac = np.ones(n)
        self._prev_phi_eq = np.full(n, np.nan)
        self._last_now: float | None = None
        self._holdoff_until = 0.0
        self._holdoff_reason = ""
        self._warned_at = np.full(n, -np.inf)
        self._armed_warned_at = -np.inf
        self._latched: CasterFault | None = None

        self._snapshot: dict[str, np.ndarray] = {}
        self._snapshot_holdoff: tuple[str, float] | None = None
        self._hist = np.zeros((max(1, round(history_s / self.control_period)), 4 + 7 * n))
        self._hist_n = 0
        self._hist_i = 0

        self.hold_off("startup", STARTUP_HOLDOFF_S)

    # -- control -------------------------------------------------------------------------------
    def hold_off(self, reason: str, duration: float = DISTURB_HOLDOFF_S) -> None:
        """Suspend the *command-referenced* detectors for ``duration``, clearing their run timers.

        For branches that legitimately break the assumption those detectors rest on: the caster-flip
        brake and the command-stream timeout both rewrite the OTG target mid-manoeuvre, so the twist
        the rate and heading rows are judged against is no longer the one the casters are chasing.

        The runaway row is deliberately **not** suspended, and that exemption is load-bearing rather
        than tidy-minded. The controller's caster-flip brake trips at exactly ``RUNAWAY_RATE`` on
        exactly ``dq[::2]``, and calls this immediately before handing the monitor the same sample --
        so if a hold-off cleared the runaway row, this detector's trip condition would strictly imply
        its own suppression and it could never fire at all. Worse, a caster genuinely running away
        holds the brake on every cycle, which would leave the rate and heading rows suspended too and
        the whole check silently blind for as long as the fault lasted. What separates a runaway from a
        legitimate flip is ``RUNAWAY_HOLD_S``, not the threshold: the worst simulated single-axis flip
        transient stays above ``RUNAWAY_RATE`` for 0.08 s, and a full command-envelope reversal reaches
        0.12 s in a deliberately non-convergent plant-model test. Both are an order of magnitude under
        the hold, and neither is a hardware measurement.

        A sample the monitor cannot trust *at all* is a different thing and clears every row; see
        :meth:`_hold_off_stale_sample`.
        """
        self._holdoff_until = self._clock() + duration
        self._holdoff_reason = reason
        # Redundant with the np.where in update() for any hold-off followed by a cycle that reaches it,
        # but this is public API: a caller may hold off and then stop calling update() entirely, which
        # is exactly what happens once a fault latches.
        self._bad_since[_COMMAND_REFERENCED] = np.inf

    def _hold_off_stale_sample(self, reason: str) -> None:
        """Hold off *everything*, runaway included, after a gap in the sample stream.

        :meth:`hold_off` spares the runaway row because a disturbed command is no reason to disbelieve
        the measurement. A gap is the opposite case: ``dt`` fell outside the accepted window, so the
        interval between the previous sample and this one was never observed, and a run that began
        before the gap is not evidence of anything *continuous* across it. Left alone, a control loop
        that blocks for 100 s and then reads one over-threshold sample would read as 100 s of sustained
        runaway and trip on that single cycle.
        """
        self.hold_off(reason)
        self._bad_since[:] = np.inf

    # -- main entry point ----------------------------------------------------------------------
    def update(
        self,
        twist: np.ndarray,
        steer_pos: np.ndarray,
        steer_vel_cmd: np.ndarray,
        steer_vel_meas: np.ndarray,
    ) -> CasterFault | None:
        """One cycle. Returns a fault the first time any detector's run exceeds its hold.

        ``twist`` must be the *commanded* body twist actually multiplied into ``C`` this cycle --
        ``AXIS_SIGN * dx_d_local``, taken verbatim. See the module docstring for why measured odometry
        is the wrong reference, and why ``AXIS_SIGN`` must be carried rather than assumed identity.
        """
        if self._latched is not None:
            return self._latched  # latched: the report must not drift while the base ramps down

        now = self._clock()
        dt = self.control_period if self._last_now is None else now - self._last_now
        self._last_now = now
        if not (DT_MIN_RATIO * self.control_period <= dt <= DT_MAX_RATIO * self.control_period):
            # Covers a stepped wall clock, a stalled loop, and the first cycle after a hold-off.
            self._hold_off_stale_sample("irregular dt")
            return None

        steer_pos = np.asarray(steer_pos, dtype=float)
        steer_vel_meas = np.asarray(steer_vel_meas, dtype=float)
        speed, phi_eq, herr = heading_errors(twist, steer_pos, self.h_x, self.h_y, self.b_x)
        # Only the clipped command is needed here; the trip compares the low-passed pair below.
        cmd, _instantaneous_rate_err = execution_errors(np.asarray(steer_vel_cmd, dtype=float), steer_vel_meas)
        self._update_lag(dt, phi_eq, speed)
        alpha = dt / RATE_FILTER_S
        self._lp_cmd += (cmd - self._lp_cmd) * alpha
        self._lp_meas += (steer_vel_meas - self._lp_meas) * alpha

        unsaturated = np.abs(cmd) <= DPHI_CMD_MAX
        armed_heading = (speed >= U_MIN) & np.isfinite(phi_eq) & unsaturated & (self._lag <= LAG_BUDGET)
        # Integrated per caster and only while that caster is actually moving, which is what
        # ARMED_WARN_FRACTION says it measures. Integrating unconditionally let a parked base decay the
        # fraction to nothing on the 60 s time constant, so the first cycle after the operator started
        # driving reported the check as disarmed at exactly the moment it became armed. Gating it here
        # is also the only gate: _warn_coverage runs on every cycle, hold-off included, so a base that
        # lives in a hold-off must not read as fully covered.
        moving = speed >= U_MIN
        self._armed_frac = np.where(
            moving,
            self._armed_frac + (armed_heading.astype(float) - self._armed_frac) * (dt / ARMED_FILTER_S),
            self._armed_frac,
        )

        tau = np.where(speed > 0.0, abs(self.b_x) / np.maximum(speed, 1e-9), MAX_HOLD_S)
        # Stack order must match DETECTORS -- _first_fault names the detector by row index alone.
        holds = np.stack(
            [
                np.full(self.num_casters, RATE_HOLD_S),  # _RATE
                np.clip(SETTLE_K * tau, MIN_HOLD_S, MAX_HOLD_S),  # _HEADING
                np.full(self.num_casters, RUNAWAY_HOLD_S),  # _RUNAWAY
            ]
        )
        lp_err = np.abs(self._lp_cmd - self._lp_meas)
        bad = np.stack(
            [
                unsaturated & (lp_err > np.maximum(RATE_ABS, RATE_REL * np.abs(self._lp_cmd))),  # _RATE
                armed_heading & (np.abs(herr) > HEADING_TRIP),  # _HEADING
                np.abs(steer_vel_meas) > RUNAWAY_RATE,  # _RUNAWAY -- ungated on purpose: no command ref
            ]
        )

        held_off = now < self._holdoff_until
        if held_off:
            # Suspend only the rows with a command reference to be disturbed; see hold_off(). Falling
            # through rather than returning is the whole point: the caster-flip brake holds off on every
            # cycle a caster is above RUNAWAY_RATE, so a real runaway lives permanently inside a
            # hold-off and has to be able to accumulate, snapshot and latch from in here.
            bad[_COMMAND_REFERENCED] = False

        # np.where already writes inf wherever bad is False, so the rows blanked above need no reset.
        self._bad_since = np.where(bad, np.where(np.isfinite(self._bad_since), self._bad_since, now), np.inf)
        # Built unconditionally. _first_fault -> _render_table reads this, and a runaway confirmed inside
        # a hold-off would otherwise render a snapshot that is stale or -- since the brake can hold off
        # on literally every cycle -- one that has never been built at all, raising KeyError on the
        # control thread.
        self._snapshot = {
            "steer_pos": steer_pos,
            "speed": speed,
            "phi_eq": phi_eq,
            "heading_error": herr,
            "cmd": cmd,
            "measured": steer_vel_meas,
            "rate_error": lp_err,
            "lag": self._lag.copy(),
            "armed": armed_heading,
            "tau": tau,
            "hold": holds,
            # Longest current run across all detectors, per caster -- one number the operator can scan.
            "bad_for": np.max(np.where(np.isfinite(self._bad_since), now - self._bad_since, 0.0), axis=0),
        }
        # Which rows were live when this was captured. Without it the first runaway report an operator
        # sees shows a 15 rad/s rate error next to a rate row that did not fire, and no way to tell why.
        self._snapshot_holdoff = (self._holdoff_reason, self._holdoff_until - now) if held_off else None
        self._record(now, twist, speed, phi_eq, herr, cmd, steer_vel_meas, armed_heading)
        if not held_off:
            self._warn_heading(now, herr, armed_heading)
        self._warn_coverage(now)

        held = np.where(np.isfinite(self._bad_since), now - self._bad_since, -np.inf)
        self._latched = self._first_fault(held >= holds, held, twist)
        return self._latched

    # -- detectors -----------------------------------------------------------------------------
    def _update_lag(self, dt: float, phi_eq: np.ndarray, speed: np.ndarray) -> None:
        """Track the legitimate steady-state lag ``tau_i * d(phi_eq)/dt``. See ``LAG_BUDGET``."""
        prev = self._prev_phi_eq
        usable = np.isfinite(phi_eq) & np.isfinite(prev) & (speed > 0.0)
        slew = np.where(usable, np.abs(wrap_to_pi(phi_eq - prev)) / dt, 0.0)
        tau = np.where(speed > 0.0, abs(self.b_x) / np.maximum(speed, 1e-9), 0.0)
        # A caster whose phi_eq is undefined this cycle keeps its previous lag rather than decaying
        # toward "safe": we would rather stay disarmed than arm on a gap in the reference.
        target = np.where(usable, tau * slew, self._lag)
        self._lag += (target - self._lag) * (dt / LAG_FILTER_S)
        self._prev_phi_eq = np.where(np.isfinite(phi_eq), phi_eq, prev)

    # -- reporting -----------------------------------------------------------------------------
    def _first_fault(self, tripped: np.ndarray, held: np.ndarray, twist: np.ndarray) -> CasterFault | None:
        if not tripped.any():
            return None
        row = int(np.argmax(tripped.any(axis=1)))
        detector = DETECTORS[row]
        casters = np.flatnonzero(tripped[row])
        table = self._render_table(twist)
        if casters.size >= SYSTEMIC_CASTERS:
            return CasterFault(
                detector=detector,
                caster=None,
                held_for=float(held[row, casters].min()),
                cause=_SYSTEMIC_CAUSE.format(n=int(casters.size), detector=detector),
                table=table,
            )
        caster = int(casters[int(np.argmax(held[row, casters]))])
        return CasterFault(
            detector=detector,
            caster=caster,
            held_for=float(held[row, caster]),
            cause=_CAUSES[detector].format(caster=caster, cid=steering_can_id(caster)),
            table=table,
        )

    def _render_table(self, twist: np.ndarray) -> str:
        s = self._snapshot
        rows = [
            ("steering CAN id", [f"{steering_can_id(i):d}" for i in range(self.num_casters)]),
            ("phi measured  deg", [f"{math.degrees(v):.1f}" for v in wrap_to_pi(s["steer_pos"])]),
            ("phi expected  deg", [f"{math.degrees(v):.1f}" for v in s["phi_eq"]]),
            ("heading err   deg", [f"{math.degrees(v):+.1f}" for v in s["heading_error"]]),
            ("cmd rate    rad/s", [f"{v:+.2f}" for v in s["cmd"]]),
            ("measured    rad/s", [f"{v:+.2f}" for v in s["measured"]]),
            ("rate err    rad/s", [f"{v:.2f}" for v in s["rate_error"]]),
            ("|u|           m/s", [f"{v:.3f}" for v in s["speed"]]),
            ("legit lag     deg", [f"{math.degrees(v):.1f}" for v in s["lag"]]),
            ("armed", ["yes" if v else "no" for v in s["armed"]]),
            ("bad for         s", [f"{v:.2f}" if v > 0 else "--" for v in s["bad_for"]]),
        ]
        header = "".join(f"{'caster ' + str(i):>12}" for i in range(self.num_casters))
        lines = [f"{'':<20}{header}"]
        lines += [f"{name:<20}" + "".join(f"{v:>12}" for v in vals) for name, vals in rows]
        lines.append("")
        lines.append(f"  reference twist (AXIS_SIGN * dx_d_local) = [{twist[0]:+.3f} {twist[1]:+.3f} {twist[2]:+.3f}]")
        lines.append(
            "  thresholds: heading "
            f"{math.degrees(HEADING_TRIP):.0f} deg, rate max({RATE_ABS:.1f} rad/s, "
            f"{RATE_REL:.0%} of command), runaway {RUNAWAY_RATE:.2f} rad/s"
        )
        if self._snapshot_holdoff is not None:
            reason, remaining = self._snapshot_holdoff
            lines.append(
                f"  hold-off ACTIVE ({reason}, {remaining:.2f} s left) when this was captured: the rate "
                f"and heading rows were suspended, only the runaway backstop was live."
            )
        return "\n".join(lines)

    def _warn_heading(self, now: float, herr: np.ndarray, armed: np.ndarray) -> None:
        """Sub-trip heading degradation, so it is visible in the field before anything stops.

        Not called during a hold-off, for the same reason the heading *trip* is suspended there: it is
        judged against a command reference the caller has just told us moved discontinuously.
        """
        for i in np.flatnonzero(armed & (np.abs(herr) > HEADING_WARN) & (np.abs(herr) <= HEADING_TRIP)):
            if now - self._warned_at[i] >= 5.0:
                self._warned_at[i] = now
                logger.warning(
                    "caster %d (steering motor %d) heading error %.1f deg, still under the %.0f deg trip",
                    i,
                    steering_can_id(int(i)),
                    math.degrees(herr[i]),
                    math.degrees(HEADING_TRIP),
                )

    def _warn_coverage(self, now: float) -> None:
        """The case where the gates have silenced the check entirely.

        Called on every cycle, hold-off included, and that is deliberate: sitting in a permanent
        hold-off is one of the exact ways this check stops running, and an alarm suppressed by the very
        condition it reports is no alarm at all. Rate-limited to one line per ``ARMED_FILTER_S``.
        """
        quiet = np.flatnonzero(self._armed_frac < ARMED_WARN_FRACTION)
        if quiet.size and now - self._armed_warned_at >= ARMED_FILTER_S:
            self._armed_warned_at = now
            logger.warning(
                "caster steering heading check has been disarmed most of the time on caster(s) %s "
                "(armed %s of the last %.0f s). It is not currently protecting them; the rate and "
                "runaway checks still are.",
                quiet.tolist(),
                ", ".join(f"{self._armed_frac[i]:.0%}" for i in quiet),
                ARMED_FILTER_S,
            )

    # -- history -------------------------------------------------------------------------------
    def _record(
        self,
        now: float,
        twist: np.ndarray,
        speed: np.ndarray,
        phi_eq: np.ndarray,
        herr: np.ndarray,
        cmd: np.ndarray,
        meas: np.ndarray,
        armed: np.ndarray,
    ) -> None:
        row = np.concatenate(
            ([now, twist[0], twist[1], twist[2]], speed, phi_eq, herr, cmd, meas, self._lag, armed.astype(float))
        )
        self._hist[self._hist_i] = row
        self._hist_i = (self._hist_i + 1) % self._hist.shape[0]
        self._hist_n = min(self._hist_n + 1, self._hist.shape[0])

    def history_csv(self) -> str:
        """The last couple of seconds, oldest first, as CSV.

        Written by the caller after the base has stopped -- never from the control thread. Without it
        the commonest field report, "it tripped and I don't know why", is unanswerable.
        """
        n = self.num_casters
        cols = ["t", "vx", "vy", "w"]
        for name in ("u", "phi_eq", "heading_err", "cmd", "measured", "lag", "armed"):
            cols += [f"{name}{i}" for i in range(n)]
        order = np.arange(self._hist_i - self._hist_n, self._hist_i) % self._hist.shape[0]
        rows = "\n".join(",".join(f"{v:.6g}" for v in self._hist[k]) for k in order)
        return ",".join(cols) + "\n" + rows + "\n"
