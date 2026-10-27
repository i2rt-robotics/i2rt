"""Read a ``--stage sweep`` trace and report one joint's gravity-model error and its friction.

Entirely offline: it touches no hardware and can be re-run on a saved CSV as often as the analysis
needs changing. That matters, because the operator's time standing next to a powered arm is the
scarce thing, not CPU.

What the sweep gives, and why both directions are needed
--------------------------------------------------------
The arm floats under gravity compensation while a slow bias creeps one joint through its travel,
first one way and then the other. At a quasi-static creep the bias has settled at whatever the
model got wrong, plus friction -- and friction reverses with direction while the model error does
not::

    b' = b - kd*q_dot                        the bias the joint actually felt, less motor damping
    b'_up   = (tau_g_true - tau_model) + tau_f
    b'_down = (tau_g_true - tau_model) - tau_f

    e(q)  = (b'_up + b'_down)/2              model error: what gravity_comp_factor must cancel
    tau_f = (b'_up - b'_down)/2              breakaway_friction

One direction cannot separate them. On this arm a single-direction fit has already returned a
*negative* Coulomb friction, which is not a small error but a sign that the two quantities were
never identifiable from that data.

Reading the result
------------------
``e`` plotted against ``tau_model`` is the whole diagnosis:

* a straight line through the origin  -> a pure **scale** error, and ``factor = 1 + slope``
* a loop, or a line that misses the origin -> a **phase** error: the model's centre of mass points
  somewhere the arm's does not. A scalar factor cannot correct that, and this tool will refuse to
  emit one rather than hand over a number that will not hold at another pose.

A phase error that shows on *every* joint is the base mounting azimuth; one that shows only on the
distal joints is the gripper's COM. That is why sweeping all seven is worth its cost.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import tyro

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from i2rt.robots.arm7_osc import ARM_DOF, MOTOR_KD_FLOAT, arm7_xml
from i2rt.utils.mujoco_utils import MuJoCoKDL

SETTLED_BAND = (0.15, 3.0)
"""Speed band, as a multiple of the run's own median creep speed, for a sample to count as
quasi-static.

Wide on purpose. Creeping a joint through Coulomb friction is stick-slip, not smooth: the joint
sticks, the bias builds, it lets go and overshoots, and the speed oscillates by an order of
magnitude around its mean. A narrow band around the commanded speed keeps only the few moments
that happen to cross it, and the two directions then select disjoint stretches of travel and
cannot be paired at all -- which is exactly what the first hardware sweep did.

What the physics actually needs is only that the joint is moving slowly in a known direction: at
that speed the inertial term is negligible, so the bias is the static balance whatever the precise
speed. The lower bound excludes the stationary samples, the upper bound excludes slips."""

MIN_SAMPLES = 20
"""Below this many paired samples the fit is not worth reporting. A short sweep usually means the
joint hit a stop or the bias ceiling before it got moving."""

PHASE_SPAN_DEG = 90.0
"""How much of the gravity curve a single sweep must cover before it can rule a phase error in or
out on its own.

A joint's gravity torque is ``A*sin(q - q0)``. A mass error scales that curve; a centre-of-mass
*direction* error shifts its phase. Over a narrow window sine and cosine are nearly collinear, so
the two are not separable from one joint's sweep however the fit is written -- an earlier version
tried a three-term regression over a 57 degree window and split the signal arbitrarily, reporting
a cleanly injected 9% scale error as 387% quadrature. Below this span the factor is still the
best estimate available and is reported as such, with the caveat stated rather than hidden.

The reliable phase test is across joints, not within one: the same shape on all seven is the base
mounting azimuth, on the distal ones only it is the gripper's centre of mass.
"""

VELOCITY_WINDOW_S = 0.3
"""Smoothing window, seconds, for differentiating position into the velocity the quasi-static
test uses. Long enough to average over the encoder's sparse ticks at creep speed, short enough
that the creep's own start and end are not smeared into the middle of the sweep."""

BREAKAWAY_DEG_S = 2.0
"""Speed, degrees per second, above which the joint counts as having broken free. Well above the
creep the sweep asks for, so a slip is not confused with the crawl it was trying to produce."""

BREAKAWAY_LOOKBACK_S = 0.2
"""How far back from a release to look for the bias that caused it. The bias is still climbing
right up to the moment the joint lets go, so the peak in this window is the breakaway value."""

MIN_BREAKAWAY_EVENTS = 6
"""Fewer releases than this per direction and the medians are not worth quoting."""

CABLE_CORRELATION = 0.5
"""Correlation between friction and joint angle above which the "friction" is not friction.

Coulomb friction does not know what angle the joint is at. A term that climbs monotonically
through the sweep is something else being wound up -- on this arm joint7's rose fifteenfold from
one end of its travel to the other, which is the gripper harness tightening as the wrist rolls.
Worth separating: a cable pull is not proportional to gravity, so no gravity_comp_factor reaches
it, while re-routing the loom removes it outright."""

MIN_R2 = 0.3
"""Below this the fit explains almost nothing and the run should be repeated rather than read.
R^2 here measures scatter in a noisy creep, not whether the shape is right, so it gates only
'unusable' and never the scale-versus-phase question."""


def _load(path: str) -> Tuple[np.ndarray, list, list]:
    raw = Path(path).read_text().strip().split("\n")
    header = raw[0].split(",")
    rows = [r.split(",") for r in raw[1:]]
    labels = [r[-1] for r in rows]
    data = np.array([[float(x) for x in r[:-1]] for r in rows])
    return data, labels, header


def _paired(q: np.ndarray, b: np.ndarray, up: np.ndarray, down: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Bin both directions onto a shared angle grid, over their overlap only.

    The two passes never land on the same angles, and averaging them at different angles would
    fold the gravity curve's own slope into the "friction". The overlap is what both directions
    actually visited.

    Binned rather than interpolated: the logged angle is not monotonic within a pass -- the joint
    jitters at creep speed -- and ``np.interp`` on a non-monotonic x silently returns nonsense. An
    earlier version of this tool interpolated, and reported a clean injected scale error as a
    phase error because of it.
    """
    lo = max(q[up].min(), q[down].min())
    hi = min(q[up].max(), q[down].max())
    if not hi > lo:
        return np.array([]), np.array([])
    # Coarsen until the two directions actually share bins. A pass that slips rather than creeps
    # leaves its samples in clusters, so a fine grid has plenty of bins holding one direction and
    # none holding both -- joint3's two passes overlapped across 63 degrees and still paired
    # nothing at 40 bins. Starting fine and widening keeps the resolution where the data supports
    # it, instead of fixing a grid that suits neither regime.
    for n_bins in (40, 20, 12, 8, 5):
        edges = np.linspace(lo, hi, n_bins + 1)
        centres, cols = [], []
        for k in range(n_bins):
            vals = []
            for sel in (up, down):
                inside = sel & (q >= edges[k]) & (q < edges[k + 1])
                if not inside.any():
                    vals = []
                    break
                vals.append(float(np.median(b[inside])))
            if vals:
                centres.append(0.5 * (edges[k] + edges[k + 1]))
                cols.append(vals)
        if len(centres) >= 5:
            return np.array(centres), np.array(cols).T
    return np.array([]), np.array([])


def _breakaway(t: np.ndarray, q: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bias at each moment the joint broke free of stiction, by angle and by direction.

    The fallback for a joint that will not creep. Above some gear ratio the stiction is large
    enough that the integral regulator cannot hold a steady crawl: the joint sticks, the bias
    climbs, it lets go and slips several tens of degrees, and the regulator backs off. Joint2 at
    48:1 does exactly this -- 1.37 N.m of stiction against a 2 N.m bias ceiling -- and pairing
    quasi-static samples finds almost nothing, because there are almost none.

    But the measurement is still there. The bias *at the instant of release* is what it took to
    break free, which is the same quantity the creep would have settled at::

        b_breakaway = (tau_g_true - tau_model) +/- tau_stiction

    So the pairing is identical; only the sampling changes, from "while moving slowly" to "just
    before it moved at all".
    """
    speed = np.gradient(np.degrees(q), t)  # unsmoothed: the slip is the signal, not noise
    moving = np.abs(speed) > BREAKAWAY_DEG_S
    events = [i for i in range(1, len(moving)) if moving[i] and not moving[i - 1]]
    angles, biases, dirs = [], [], []
    back = max(round(BREAKAWAY_LOOKBACK_S / max(float(np.median(np.diff(t))), 1e-6)), 2)
    for i in events:
        pre = b[max(0, i - back) : i]
        if pre.size == 0:
            continue
        angles.append(q[i])
        biases.append(float(pre[np.argmax(np.abs(pre))]))
        dirs.append(float(np.sign(speed[i])))
    return np.array(angles), np.array(biases), np.array(dirs)


def _report_breakaway(
    t: np.ndarray,
    q: np.ndarray,
    b: np.ndarray,
    kdl: MuJoCoKDL,
    q_all: np.ndarray,
    j: int,
    applied: float,
) -> None:
    """Print the breakaway result, pairing the two directions by angle rather than globally.

    Pairing globally would fold the gravity curve's own slope into the "stiction": the two
    directions release at different angles, and joint2's gravity changes by 15 N.m across its
    sweep. Binning first keeps each pair at a comparable angle.
    """
    angles, biases, dirs = _breakaway(t, q, b)
    up, down = dirs > 0, dirs < 0
    print(f"  {up.sum()} releases pushing +, {down.sum()} pushing -")
    if up.sum() < MIN_BREAKAWAY_EVENTS or down.sum() < MIN_BREAKAWAY_EVENTS:
        print("  Too few to quote. Lengthen --seconds so the joint sticks and slips more times,")
        print("  or raise --sweep-bias-limit if it never broke free at all.")
        return

    lo, hi = max(angles[up].min(), angles[down].min()), min(angles[up].max(), angles[down].max())
    edges = np.linspace(lo, hi, 7)
    rows = []
    for k in range(len(edges) - 1):
        a = up & (angles >= edges[k]) & (angles <= edges[k + 1])
        c = down & (angles >= edges[k]) & (angles <= edges[k + 1])
        if not (a.any() and c.any()):
            continue
        mid = 0.5 * (edges[k] + edges[k + 1])
        pose = np.median(q_all, axis=0).copy()
        pose[j] = mid
        g = float(kdl.compute_inverse_dynamics(pose, np.zeros(ARM_DOF), np.zeros(ARM_DOF))[j])
        bu, bd = float(np.median(biases[a])), float(np.median(biases[c]))
        rows.append((mid, g, (bu + bd) / 2.0, (bu - bd) / 2.0))
    if len(rows) < 2:
        print("  The two directions never released at comparable angles, so they cannot be paired.")
        print(
            f"  Stiction from the ungrouped medians only: "
            f"{(np.median(biases[up]) - np.median(biases[down])) / 2:+.3f} N.m"
        )
        return

    arr = np.array(rows)
    stiction = float(np.median(arr[:, 3]))
    slope, intercept = (float(c) for c in np.polyfit(arr[:, 1], arr[:, 2], 1))
    print(f"\n  {'q(deg)':>9}{'tau_model':>11}{'e':>9}{'stiction':>10}")
    for mid, g, e, f in arr:
        print(f"  {np.degrees(mid):9.1f}{g:11.2f}{e:9.3f}{f:10.3f}")
    print(f"\n  static friction  {stiction:+.3f} N.m   (this is the quantity to put in breakaway_friction)")
    print(f"  e = {slope:+.4f}*tau {intercept:+.3f}")
    print(f"  -> gravity_comp_factor[{j}] = {applied + slope:.3f}")
    print("     From breakaway rather than creep, so treat it as the coarser of the two: the")
    print("     releases are spread over angle and the pairing is by bin, not by matched pose.")
    # The *range* of the gravity torque, not its largest value: a factor scales the curve, so it
    # can only ever explain the part that varies. Joint5's gravity sits between -0.73 and -0.58
    # N.m, a range of 0.15 -- against 0.29 N.m of stiction there is nothing a factor can reach,
    # even though 0.73 on its own looks like plenty of signal. The creep path measures it this way
    # too; they disagreed, and this one was wrong.
    if abs(slope) * float(arr[:, 1].max() - arr[:, 1].min()) < abs(stiction):
        print(f"     But gravity only varies by {arr[:, 1].max() - arr[:, 1].min():.2f} N.m across this")
        print(
            f"     sweep, so the factor reaches at most {abs(slope) * (arr[:, 1].max() - arr[:, 1].min()):.2f} N.m of it, "
            f"under the {abs(stiction):.2f} N.m"
        )
        print("     of stiction. No value of the factor changes how this joint behaves: leave it")
        print("     at 1.0 and put the friction in breakaway_friction instead.")


def _report_friction_shape(grid: np.ndarray, friction: np.ndarray) -> None:
    """Say whether the friction is friction, or something that varies with angle."""
    print(f"\n  breakaway_friction  {float(np.median(friction)):+.3f} N.m   (median over the sweep)")
    if grid.size < 4:
        return
    corr = float(np.corrcoef(grid, friction)[0, 1])
    if abs(corr) < CABLE_CORRELATION:
        return
    slope, intercept = (float(c) for c in np.polyfit(np.degrees(grid), friction, 1))
    lo = slope * float(np.degrees(grid[0])) + intercept
    hi = slope * float(np.degrees(grid[-1])) + intercept
    print(
        f"  but it tracks angle, correlation {corr:+.2f}: {lo:.3f} N.m at "
        f"{np.degrees(grid[0]):+.0f} deg rising to {hi:.3f} at {np.degrees(grid[-1]):+.0f}."
    )
    print("  Coulomb friction does not depend on angle. Something is being wound up as this joint")
    print("  turns, and a cable loom is the usual answer. It is not proportional to gravity, so no")
    print("  gravity_comp_factor reaches it; re-routing the loom does.")


def main(
    csv: str,
    /,
    joint: int = 4,
    arm: str = "right",
    applied_factor: float = 1.0,
) -> None:
    """Analyse ``csv``, a trace written by ``arm7_bringup.py --stage sweep --joint <joint>``.

    ``applied_factor`` is the ``gravity_comp_factor`` entry the controller was *running* for this
    joint during the sweep -- whatever ``--gravity-factor`` said, or the config's value. It is not
    cosmetic: the bias measures what gravity compensation failed to cancel, so the answer is the
    factor already in force plus the slope, not the slope alone. Reading a sweep taken at 0.5 as
    though it were taken at 1.0 turns a true 0.62 into a reported 1.12 -- an arm that needs *less*
    gravity into one that appears to need more, and the sign of that mistake is the difference
    between a stable factor and a runaway.
    """
    data, _labels, header = _load(csv)
    j = joint - 1
    t, q, dq = data[:, 0], data[:, 1 + j], data[:, 1 + ARM_DOF + j]
    q_all = data[:, 1 : 1 + ARM_DOF]

    if "b" in header:
        b = data[:, header.index("b")]
    else:
        # Older traces logged only the commanded torque. The bias is that minus what gravity
        # compensation alone would have sent, which the model can reproduce from the logged pose.
        kdl = MuJoCoKDL(arm7_xml(arm))
        zero = np.zeros(ARM_DOF)
        tau = data[:, 1 + 2 * ARM_DOF + j]
        b = np.array([tau[i] - kdl.compute_inverse_dynamics(q_all[i], zero, zero)[j] for i in range(len(t))])

    kdl = MuJoCoKDL(arm7_xml(arm))
    zero = np.zeros(ARM_DOF)
    # The motor applies the bias less its own damping, and the damping it applied was computed
    # from the velocity the motor itself reported -- so that term keeps the reported signal,
    # whatever its quality.
    b_eff = b - MOTOR_KD_FLOAT[j] * dq

    # Selecting the quasi-static samples is a different question, and the reported velocity cannot
    # answer it. At a 0.03 rad/s creep a differenced rotor encoder ticks rarely: on the first
    # hardware sweep, 1989 of 2723 samples read exactly -0.002 rad/s with occasional large jumps
    # between them. Its *mean* was right (+0.031 against a true +0.033) and its median was not,
    # so any per-sample test against it selects almost nothing. Position is clean over the same
    # window, so the selection differentiates that instead.
    span = max(round(VELOCITY_WINDOW_S / max(float(np.median(np.diff(t))), 1e-6)), 3)
    kernel = np.ones(span) / span
    v = np.gradient(np.convolve(q, kernel, mode="same"), t)
    edge = span  # the convolution's ends are not smoothed; drop them rather than fit to them
    v[:edge] = 0.0
    v[-edge:] = 0.0

    speed = np.median(np.abs(v[np.abs(v) > 1e-3])) if np.any(np.abs(v) > 1e-3) else 0.0
    lo, hi = (f * max(speed, 1e-6) for f in SETTLED_BAND)
    moving = (np.abs(v) > lo) & (np.abs(v) < hi)
    up = moving & (v > 0)
    down = moving & (v < 0)
    print(
        f"{csv}\n  joint {joint}, {len(t)} samples, creep speed {speed:.3f} rad/s "
        f"(from position; the reported velocity medians {np.median(dq):+.3f})"
    )
    print(f"  quasi-static: {up.sum()} up, {down.sum()} down")
    if up.sum() < MIN_SAMPLES or down.sum() < MIN_SAMPLES:
        print("\n  Too little of the sweep was quasi-static to pair. Falling back to breakaway.")
        _report_breakaway(t, q, b, kdl, q_all, j, applied_factor)
        return

    grid, pair = _paired(q, b_eff, up, down)
    if grid.size == 0:
        print("\n  The two directions covered no common travel. Falling back to breakaway.")
        _report_breakaway(t, q, b, kdl, q_all, j, applied_factor)
        return
    # Gravity at the grid angles, evaluated from the model rather than interpolated from the log.
    # The other joints are held at what they actually were, which the sweep keeps still.
    others = np.median(q_all, axis=0)
    g_grid = np.empty_like(grid)
    for i, angle in enumerate(grid):
        pose = others.copy()
        pose[j] = angle
        g_grid[i] = kdl.compute_inverse_dynamics(pose, zero, zero)[j]
    e = (pair[0] + pair[1]) / 2.0
    friction = (pair[0] - pair[1]) / 2.0

    slope, intercept = (float(c) for c in np.polyfit(g_grid, e, 1))
    residual = e - (slope * g_grid + intercept)
    denom = float(np.sum((e - e.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / denom if denom > 0 else 0.0
    span_deg = float(np.degrees(grid[-1] - grid[0]))

    print(
        f"\n  e = {slope:+.4f}*tau {intercept:+.3f}    R2 {r2:.3f}   over {span_deg:.0f} deg of travel"
        f"\n  gravity_comp_factor in force during the sweep: {applied_factor:.3f}"
    )
    # A creep fit can pass the sample count and still be meaningless. Stick-slip supplies plenty
    # of samples -- they are just all taken during slips -- and the giveaway is an offset that does
    # not vanish where gravity does, or a fit that explains nothing. Both mean the joint never
    # crept, so the breakaway reading is the one that applies.
    if r2 < MIN_R2:
        print(f"  -> the fit explains almost nothing (R2 {r2:.3f}); the joint did not creep.")
        _report_breakaway(t, q, b, kdl, q_all, j, applied_factor)
        return
    if abs(intercept) >= 0.1:
        print(f"  -> an offset of {intercept:+.3f} N.m that does not vanish where gravity does.")
        print("     For a joint that slipped rather than crept, that offset is the slip, not a")
        print("     spring or a torque-constant bias. Reading it as breakaway instead:")
        _report_breakaway(t, q, b, kdl, q_all, j, applied_factor)
        return
    # What the factor can explain, against what would hide it. A joint whose whole gravity swing
    # is smaller than its own friction cannot report a scale error however clean the arithmetic
    # looks: joint7 fitted a slope over a 0.07 N.m signal buried in 0.12 N.m of friction, and the
    # answer then tracked whatever --applied-factor was passed in, which is the signature of a fit
    # carrying no information of its own.
    signal = abs(slope) * float(g_grid.max() - g_grid.min())
    friction_level = abs(float(np.median(friction)))
    if signal < friction_level:
        print(f"  -> no usable factor. The model error across this whole sweep is {signal:.3f} N.m,")
        print(f"     under the {friction_level:.3f} N.m of friction on this joint, so the fit is")
        print("     describing friction and noise. Leave the factor at 1.0, record the friction,")
        print("     and sweep this joint where gravity is larger if the factor is really needed.")
        _report_friction_shape(grid, friction)
        return

    factor = applied_factor + slope
    print(f"  -> gravity_comp_factor[{j}] = {factor:.3f}")
    if factor > 1.0:
        print("     Above 1.0 is formally unstable: net torque is (factor-1)*g(q) and dg/dq > 0")
        print("     about a hanging pose, so there is no restoring term. Friction masks it until")
        print("     something disturbs the arm. Prefer the largest value below 1.0 that still")
        print("     holds, and never tune up through 1.0.")
    if span_deg < PHASE_SPAN_DEG:
        print(f"     Caveat: {span_deg:.0f} deg of travel is too narrow to tell a mass error from a")
        print("     centre-of-mass direction error -- over a short arc the two look alike. Sweep")
        print("     the other joints and compare: one shape on all seven is the base mounting")
        print("     azimuth, on the distal ones only it is the gripper.")

    print("\n  per-angle detail")
    print(f"    {'q(deg)':>8}{'tau_model':>11}{'e':>9}{'friction':>10}")
    for i in range(0, len(grid), max(1, len(grid) // 12)):
        print(f"    {np.degrees(grid[i]):8.1f}{g_grid[i]:11.2f}{e[i]:9.3f}{friction[i]:10.3f}")


if __name__ == "__main__":
    tyro.cli(main)
