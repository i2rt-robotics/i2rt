"""Staged hardware bring-up for the 7-DOF OSC arm.

Runs one stage per invocation against a real motor chain, prints what it measured and stops.
Nothing advances on its own.

    B="python examples/arm7_osc/arm7_bringup.py --transport ethercat --loop-hz 500"

    $B --stage check        # motor types and registers, read-only; CAN transport only
    $B --stage pose         # current joint angles, nothing energised
    $B --stage mirror       # per-joint direction and travel against the model
    $B --stage probe        # breakaway torque per joint
    $B --stage survey       # per-motor feedback rate, skew, poll rate
    $B --stage jitter       # bare-loop timing, no control law
    $B --stage float        # gravity compensation only
    $B --stage sweep        # friction and gravity model, one joint at a time
    $B --stage null         # nullspace arc
    $B --stage task         # task-space stiffness
    $B --stage track        # joint sinusoid, tool-error RMS
    $B --stage watchdog     # command timeout and resume

Powered stages prompt before energising anything. ``--max-torque`` defaults to 3 N.m, below what
the shoulder needs to hold the arm up, so a first run sags rather than fights.

The velocity guard, torque and slew clamps, command timeout and missing-frame watchdog are all
live. See ``examples/arm7_osc/README.md``.
"""

from __future__ import annotations

import logging
import signal
import sys
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import tyro
import yaml

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from i2rt.motor_drivers.dm_driver import ControlMode, DMChainCanInterface, MotorChain, MotorType, ReceiveMode
from i2rt.motor_drivers.motor_check import run_startup_checks
from i2rt.robots.arm7_osc import (
    ARM_DOF,
    KD_NULL,
    KD_TASK_JOINT,
    KP_NULL,
    KP_TASK_JOINT,
    TORQUE_LIMIT_NM,
    Arm7OSC,
    ArmState,
    LoopTimer,
    arm7_xml,
)
from i2rt.robots.arm7_osc import (
    # `ControlMode` is taken: dm_driver's is MIT/VEL, this one is kinematic/full.
    ControlMode as OSCMode,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("arm7_bringup")

CONFIG_DIR = Path(_REPO_ROOT) / "i2rt" / "robots" / "config"

# Poses for the float-drift check, each loading the shoulder differently.
SWEEP_KI = 3.0
"""Integral gain of the creep regulator, N.m per (rad/s) per second. Slack on purpose."""

SWEEP_DT = 0.01
"""Update period of the creep regulator and of the breakaway ramp, seconds."""

BREAKAWAY_RAMP_S = 8.0
"""Seconds for the breakaway bias to ramp from zero to the limit. Slow enough that the reading is
a static one, and that a joint about to let go does so at a well-defined bias rather than being
thrown by a step."""

BREAKAWAY_RELEASE_DEG = 1.5
"""Degrees of movement that count as having let go. Above the encoder noise, below the slip a
high-stiction joint takes once it does move."""

BREAKAWAY_SETTLE_S = 1.5
"""Pause after releasing the bias, so the next cycle starts from rest and not from a coast."""

BREAKAWAY_TIMEOUT_S = 20.0
"""Give up on one load if the joint never moves: the bias ceiling was too low for it."""
"""Update period of the creep regulator, seconds."""

GOTO_GAIN_SCALE = 0.25
"""Fraction of the shipped task gains used while driving to ``--start-pose``.

The move wants to be slow and to arrive; it does not want to be stiff. At full stiffness joint6 --
the lightest and least-damped joint on the arm, 0.0087 kg.m^2 against 0.04 N.m of friction --
oscillated hard enough to trip the 4 rad/s velocity guard partway through, leaving the arm faulted
before the stage it was setting up had even started."""

LINK_NAMES = ("link1", "link2", "link3", "link4", "link5", "link6", "gripper")
"""Bodies labelled in the mirror view. The naming is ``jointN`` drives ``linkN``, so ``link6`` is
the short section between joint6 and joint7 -- 46 mm on this arm, and not something anyone will
identify from a description. ``gripper`` is joint7's child and carries the end effector."""

STILL_SPEED = 0.02
"""rad/s below which every joint counts as released. Above the float noise floor (0.007
rad/s measured limp and still), below anything a hand does."""

MIN_HOLD_S = 1.5
"""Seconds a pose must be still before it is reported. Passing through is not a measurement."""


TRACE_HZ = 50.0
"""Default rate the trace samples at. 50 Hz resolves anything up to 25 Hz, which is not enough once
a stage rings: joint6 has been seen at an apparent 19.5 Hz, close enough to that limit to be an
alias of something faster. Raise it with --trace-hz, up to the rate feedback actually arrives at --
200 Hz is a good default when hunting a ring."""


TRACK_AMPLITUDE_DEG = np.array([4.0, 8.0, 8.0, 10.0, 8.0, 8.0, 8.0])
"""Per-joint amplitude of the tracking sinusoid, in degrees.

Sized so that every joint contributes and none dominates: joint1 is smallest because it swings the
whole arm through the largest arc for a given angle, joint4 largest because the elbow is what moves
the tool most per degree without rotating the base. Together they trace a few tens of centimetres
of tool path, which is enough for a position RMS to mean something and small enough to stay clear
of the limits from a mid-workspace start."""

TRACK_PERIOD_S = 8.0
"""Period of the sinusoid. The spec asks for 5 s or slower, and slower is the point: this measures
tracking, not bandwidth, so the trajectory should be well inside what the arm can follow. Any error
left at 8 s is stiction and gain, not inertia."""


NULLSPACE_POSE = "0,45,20,-60,0,30,0"
"""Default start pose for --stage null and --stage track, in degrees.

The same commissioning pose the other stages use, and picked for the same reason: cond(J) is 17
here against 47968 at the hanging pose, and the nullspace is a clean single dimension rather than
the two that fall out of a singular configuration. An arc planned at a singular pose is not a
self-motion, it is the singularity."""

NULLSPACE_ARC_DEG = 25.0
"""Joint-space arclength to travel along the nullspace, each way from the start pose.

Big enough that a kinematic error shows as millimetres of tool motion, small enough to stay well
inside the joint ranges and away from the wrist singularity. At the commissioning pose this is
about 17 deg on joint3, 15 on joint5, 8 on joint6 -- an elbow swivel, which is the self-motion a
7-DOF arm has.
"""

NULLSPACE_STEP_DEG = 0.25
"""Continuation step. The nullspace direction rotates as the arm moves, so the arc is integrated
rather than extrapolated; each step is followed by a task-space correction, and a quarter of a
degree keeps that correction in the linear regime."""

ARC_COMMAND_HZ = 100.0
"""Rate the nullspace arc is commanded at, interpolating between plan points rather than stepping.
The plan's own resolution (``NULLSPACE_STEP_DEG``) is a path resolution, not a command rate: issuing
one point per 150 ms makes the command a zero-order-hold staircase and the arm chases its steps.
Stepping it instead puts the whole velocity spectrum on the waypoint rate and its harmonics."""

NULLSPACE_TOOL_TOL_MM = 0.05
"""How far the *planned* arc is allowed to let the tool drift in the model. This is a check on the
planner, not on the arm -- the correction step should hold it near machine precision, and anything
above this means the continuation is not converging and the arc must not be played."""

NULLSPACE_ABORT_DEG = 8.0
"""Joint tracking lag that stops playback. Past this the arm is not on the planned path, so the
tool displacement being measured is not a measurement of anything. Set above the worst stiction
deadband at the shipped gains (joint1, 1.91 N.m over 40 N.m/rad, is 2.7 deg) with room for the
transient at a direction reversal."""

NULLSPACE_ABORT_MM = 120.0
"""Tool deviation, measured from the arm's own encoders, that stops playback. Loose on purpose --
the tracking-lag abort above is the one that catches a run gone wrong, and this is a backstop for
the case where every joint tracks and the tool still leaves, which would mean the planned path was
not a self-motion at all."""


@dataclass
class Args:
    """Staged hardware bring-up for the 7-DOF OSC arm."""

    stage: str = "check"
    """check | pose | mirror | probe | survey | jitter | float | sweep | null | task | watchdog.
    That is also the order to run them in: each is riskier than the one before."""
    channel: str = "can0"
    """CAN channel."""
    arm: str = "right"
    """right or left. Selects both the model and the config YAML."""
    seconds: float = 20.0
    """How long the measuring stages run."""
    max_torque: float = 3.0
    """Per-joint torque clamp, N.m, applied on top of TORQUE_LIMIT_NM. Deliberately below what the
    shoulder needs to hold itself (14.7 N.m), so a first run sags instead of fighting. Joints 1 and
    2 carry DM6248s rated 120 N.m -- a part's rating is not a safe bound for a 6 kg arm, which is
    why both clamps exist."""
    mode: str = "kinematic"
    """kinematic or full. Start on kinematic: it is immune to inertia error in the model."""
    gain_scale: float = 0.25
    """Task-gain multiplier for the `task` stage. Start low and step up, recording where it
    destabilises -- that number is a deliverable, not a failure."""
    port: int = 8080
    """Port the `mirror` stage serves its browser view on."""
    host_hint: str = "localhost"
    """Hostname to print in the `mirror` stage's URL. Set it to this machine's address when you
    will open the page from elsewhere."""
    joint: int = 7
    """Which joint the `probe` stage pushes, 1-based. Start at 7 (the lightest) and work inward."""
    probe_torque: float = 0.3
    """Torque the `probe` stage applies, N.m. Refused above a quarter of that joint's rating."""
    probe_seconds: float = 1.5
    """Longest `probe` applies it. It stops sooner on any of the guards below."""
    probe_travel: float = 0.30
    """Stop the probe once the joint has moved this far, rad. A sign reads from a few degrees; a
    joint with nothing to stop it will otherwise keep accelerating for the whole window."""
    probe_max_speed: float = 2.0
    """Stop the probe if the joint exceeds this, rad/s."""
    start_pose: str = ""
    """Drive the arm to this pose under power before the stage begins, as 7 comma-separated
    degrees, e.g. "0,45,20,-60,0,30,0". Empty starts from wherever the arm already is.

    A hanging arm cannot be held in a raised posture by hand -- ``--stage pose`` is limp, so the
    operator would be carrying the whole 6.5 kg -- and several stages are meaningless at the hang:
    joint3 and joint5 see no gravity there at all, and the Jacobian loses rank, which inflates the
    nullspace from one dimension to two. This is how the arm gets to a usable posture."""
    start_seconds: float = 10.0
    """How long the ``--start-pose`` move takes. Slow on purpose: the move is a position command,
    and a fast one asks for whatever velocity the gains imply, which is what the velocity guard
    exists to catch."""
    sweep_speed: float = 0.05
    """rad/s the swept joint is driven at. Slow enough that inertia and Coriolis are negligible
    against gravity, which is what makes the reading a static one."""
    sweep_span: float = 60.0
    """Degrees either side of the starting angle to sweep. Kept well inside the joint's travel --
    the measurement needs the joint moving freely, and a joint resting on its stop reads the stop,
    not gravity."""
    sweep_mode: str = "creep"
    """``creep`` walks the joint through its range at a steady crawl; ``breakaway`` instead loads
    it until it lets go, releases, and repeats.

    Creep is the better measurement where it works: the bias settles at the static balance and
    every sample counts. It stops working once stiction is large. Joint2 at 48:1 and joint3 at 40:1
    never crawl at all -- they stick for tens of seconds while the bias climbs, slip twenty degrees,
    and stick again -- so every sample is taken mid-slip and the "creep speed" measured is the slip.
    Breakaway measures the same quantity from the other side: the bias at the instant of release is
    what it took to overcome the gravity error plus stiction, and alternating directions at one
    angle gives the analysis its pair without the joint ever moving smoothly."""
    breakaway_cycles: int = 6
    """Load-and-release cycles per direction at each angle. Each yields one reading and the
    analysis takes their median, so this trades run time against scatter."""
    breakaway_step: float = 12.0
    """Degrees to walk between clusters of cycles, so the readings span the travel rather than
    describing a single angle very precisely."""
    sweep_from: float = float("nan")
    sweep_to: float = float("nan")
    """Absolute joint angles in degrees to sweep between, instead of +/- ``--sweep-span`` about
    wherever the joint starts.

    The model knows the joint's own travel and nothing about what the arm is bolted to. This arm
    hangs from a pole that joint2 will strike well inside its modelled range, and no joint limit
    will stop it. Give both when the mounting, a bench or a fixture is closer than the joint stop;
    the sweep then stays inside them and never has to be interrupted, which also means both
    directions cover the same ground and can actually be paired."""
    sweep_bias_limit: float = 2.0
    """N.m ceiling on the creep bias, on top of Arm7OSC's own MAX_TORQUE_BIAS_NM and the torque
    clamp. Enough to break any joint away from rest; small enough that a runaway creeps."""
    trace: str = ""
    """Where --stage float writes its CSV of q / q_dot / tau at 50 Hz. Empty picks a timestamped
    path under /tmp. Read it back to see what the arm did when compensation switched on, which is
    a transient no terminal can show."""
    arc_deg: float = NULLSPACE_ARC_DEG
    """Degrees of joint-space arclength --stage null travels along the nullspace, each way.

    Raising it makes a kinematic error easier to see -- tool displacement grows with how far the
    arm reconfigures -- and costs joint-range headroom. The planner stops 2 degrees short of any
    limit and reports where it stopped, so an over-large value degrades rather than failing."""
    arc_deadband: float = 0.5
    """Degrees of stiction deadband --stage null sizes its stiffness for: ``kp = friction /
    deadband``, per joint, from the config's measured ``coulomb_friction``.

    This is the tracking error the arc is allowed to carry, and through the Jacobian it is most of
    the tool error the run will report -- joint1 alone moves the tool 11 mm per degree. Lower it
    for a tighter measurement, at the cost of stiffness the damping cap may not be able to match."""
    transport: str = "can"
    """Which motor transport to open: ``can`` for the USB-CAN bench, ``ethercat`` for the bridge.

    ``ethercat`` requires ``encosbridge`` to be running already, with the same ``ROS_DOMAIN_ID``
    and ``ROS_LOCALHOST_ONLY``, and its config carrying unit directions and zero offsets so that
    the calibration is applied once, here, from the same YAML simulation reads.

    Three things behave differently on it and each is called out where it happens: the startup
    register checks do not exist, the motors are enabled by the bridge rather than by this script,
    and the bridge re-applies the last command it received for ever -- it has no timeout of its
    own, so releasing the arm on exit is entirely this script's responsibility."""
    receive_mode: str = "p16"
    """Which CAN id a motor answers on: its own (``same``) or its own plus 16 (``p16``).

    A Damiao motor holds ``ID`` (register 0x08), the id it listens on, and ``MASTER_ID`` (0x07),
    the id it answers on. i2rt flashes them 16 apart; the EtherCAT-to-CAN board requires them
    equal, since it matches a reply to the slot it sent on. Pass ``same`` for motors flashed that
    way. Ignored on ``--transport ethercat``, where the bridge owns the framing."""
    track_amplitude: float = 1.0
    """Multiplier on ``TRACK_AMPLITUDE_DEG``. The planned path is checked against joint limits and
    conditioning before anything is energised."""
    track_period: float = TRACK_PERIOD_S
    track_cycles: float = 3.0
    """Cycles of the sinusoid. The RMS is reported both including and excluding the first."""
    trace_hz: float = TRACE_HZ
    """Rate the CSV is written at, Hz. Raise it above the default when a stage rings near its
    25 Hz Nyquist limit."""
    loop_hz: float = 100.0
    """Feedback rate the damping cap is computed against, Hz. **Pass the value ``--stage survey``
    measures**; the default is a pessimistic placeholder.

    The cap is ``0.5 * friction / (resolution * rate)``, so a value below the real rate lets
    through more host damping than the joint can carry. On the EtherCAT bridge the measured rate
    is 500 Hz, the DDS publish rate, not the 428.5 Hz at which the CAN side refreshes a motor."""
    arc_seconds: float = 60.0
    """Wall time for the whole four-leg arc, seconds. Slow enough that inertial and Coriolis
    torques stay negligible against the kinematic measurement."""
    gravity_factor: str = ""
    """Override gravity_comp_factor for this run, as 7 comma-separated numbers, e.g.
    "1,1.1,1,1.2,1,1,1". Empty keeps the config's values. Tuning without editing the YAML each
    time is the point; write the value back once it settles."""
    cart_zeta: float = 1.0
    """Cartesian damping ratio in ``--mode full``: ``Kd6 = 2 * zeta * sqrt(Kp6)``.

    1.0 is critical damping, which is what this shipped with and is the conservative choice -- no
    overshoot, but the return is an exponential crawl. That matters here because the crawl has to
    finish inside a Coulomb friction band, which is why raising ``Kp6`` alone stops helping.
    Below 1.0 the return is faster and may overshoot; above it the arm has been seen to break into
    a limit cycle. ``full`` mode has no automatic gain cap, so change this in steps and watch."""

    motor_kd: float = -1.0
    """Override the motor-side kd used while floating, for every joint. Negative keeps the
    per-joint defaults in MOTOR_KD_FLOAT. Pass 0 to float on the gravity term alone.

    A perfectly gravity-compensated arm is neutrally stable, so some damping keeps it from
    wandering off any disturbance -- in simulation 60 s of float drifts 380 mm without it and
    6.7 mm with. The defaults are small enough that the ripple they make from this arm's measured
    velocity noise (0.0073 rad/s at rest) is about 2 mN.m, well under the 59 mN.m a DM6248 can even
    express. Setting 0 is still a reasonable first run if you would rather see the bare term."""
    probe_dir: str = "both"
    """both | pos | neg. Use one direction when the other would drive the arm into something --
    the reverse of joint 2 reaches the mounting post on this arm."""
    yes: bool = False
    """Skip the confirmation prompt. Only for a stage you have already run by hand."""


def _paths(arm: str) -> Tuple[str, Path]:
    if arm == "right":
        return arm7_xml("right"), CONFIG_DIR / "yam_7dof_right_v1.yml"
    if arm == "left":
        return arm7_xml("left"), CONFIG_DIR / "yam_7dof_left_v1.yml"
    raise SystemExit(f"--arm must be 'right' or 'left', got {arm!r}")


def _load_config(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"no config at {path}")
    cfg = yaml.safe_load(path.read_text())
    motors = [(int(mid, 16) if isinstance(mid, str) else int(mid), str(mtype)) for mid, mtype in cfg["motor_list"]]
    if len(motors) != ARM_DOF:
        raise SystemExit(f"{path} declares {len(motors)} motors, expected {ARM_DOF}")
    cfg["motor_list"] = motors
    return cfg


def _confirm(args: Args, what: str) -> None:
    if args.yes:
        return
    print(f"\n  ABOUT TO: {what}")
    print("  e-stop in reach? arm clear? second person present?")
    if input("  type 'go' to continue: ").strip() != "go":
        raise SystemExit("aborted")


def _build_chain(args: Args, cfg: dict, check: bool) -> MotorChain:
    """Open the motor chain, **energising every motor**.

    ``can`` opens a USB-CAN adapter on ``--channel``. ``ethercat`` connects to the EtherCAT-to-CAN
    bridge over ROS 2 topics; the bridge must already be running and enables the motors itself.

    ``check`` runs the startup register checks, which need an idle bus and so happen inside the CAN
    constructor before the socket opens. The bridge offers no register access, so on that transport
    they are skipped and said to be skipped.
    """
    if args.transport == "ethercat":
        from i2rt.motor_drivers.ethercat_chain import EtherCATMotorChain

        if check:
            print("  register checks skipped: the bridge exposes no register access.")
            print("  Verify motor types and CTRL_MODE over CAN before trusting a run here.")
        return EtherCATMotorChain(
            cfg["motor_list"],
            motor_offset=np.zeros(ARM_DOF),
            # The bridge's own motor_directions must be all ones for this: applying the flip twice
            # cancels it, applying it nowhere leaves joint1 running backwards.
            motor_direction=np.array(cfg["directions"], dtype=float),
            node_name="arm7_bringup",
        )
    return DMChainCanInterface(
        cfg["motor_list"],
        np.zeros(ARM_DOF),
        np.array(cfg["directions"], dtype=float),
        args.channel,
        motor_chain_name="arm7_bringup",
        receive_mode=ReceiveMode[args.receive_mode],
        start_thread=True,
        use_buffered_reader=False,
        check_motor_types=check,
        check_motor_config=check,
    )


# ---------------------------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------------------------


def stage_check(args: Args, cfg: dict, xml: str) -> None:
    """Read every motor's registers and refuse a chain that is not what the config claims.

    Strictly read-only: runs the checks with ``repair=False`` rather than building a chain, so no
    ``CTRL_MODE`` is written to Flash and no motor is energised. A mismatch is reported with the
    command that fixes it. CAN transport only; the bridge exposes no registers.
    """
    if args.transport == "ethercat":
        raise SystemExit(
            "  --stage check cannot run over the bridge. It reads motor registers (Gr, CTRL_MODE,\n"
            "  PMAX/VMAX/TMAX) and the bridge exposes motor state as plain arrays with no register\n"
            "  access at all, so there is nothing here to read and nothing to compare.\n"
            "\n"
            "  What substitutes, and what does not:\n"
            "    ecat_scan <nic>          every motor answers, with position and temperature.\n"
            "                             Read-only, energises nothing. This catches a motor off\n"
            "                             the bus -- the defect that voided three runs on\n"
            "                             and it is what --stage check would have caught here.\n"
            "    --transport can + this   the only way to verify declared motor *types* and\n"
            "                             CTRL_MODE. Until that has been done once on this arm,\n"
            "                             every torque number is scaled by an unverified constant.\n"
            "\n"
            "  Run the scan, then go straight to --stage jitter."
        )

    print("declared:")
    for i, (mid, mtype) in enumerate(cfg["motor_list"]):
        const = MotorType.get_motor_constants(mtype)
        print(
            f"  joint{i + 1}  id 0x{mid:02x}  {mtype:8s}  ratio {MotorType.get_gear_ratio(mtype):4.0f}  "
            f"torque +/-{const.TORQUE_MAX:5.1f} N.m"
        )

    print("\nreading registers (read-only: nothing written, no motor energised) ...")
    run_startup_checks(
        args.channel,
        [[mid, mtype] for mid, mtype in cfg["motor_list"]],
        check_motor_types=True,
        check_motor_config=True,
        control_mode=ControlMode.MIT,
        repair=False,
    )
    print("\nOK -- every motor holds the type and scaling this config declares.")
    print("\nNote Gr maps many-to-one, so this confirms each motor's class, not which of a")
    print("same-ratio pair is fitted. A same-ratio swap is not detectable this way.")
    print("If CTRL_MODE was reported wrong, fix it with the dm_motor_registers.py command shown")
    print("above rather than letting a later stage write it silently.")


def _check_motors_alive(chain: MotorChain) -> None:
    """Refuse to run if any motor is not answering. Called before every powered stage.

    Checks two things the transport offers for each joint: the health byte the chain maps into
    ``MotorInfo.error_code``, and whether a timestamp has arrived at all.
    """
    infos = chain.read_states()
    if infos is None or len(infos) < ARM_DOF:
        raise SystemExit(f"the chain returned {0 if infos is None else len(infos)} motors, need {ARM_DOF}")
    dead = []
    for i in range(ARM_DOF):
        code = str(infos[i].error_code)
        if code not in ("0x1", "1"):
            dead.append(f"joint{i + 1} reports {code}")
        elif infos[i].timestamp <= 0.0:
            dead.append(f"joint{i + 1} has never been heard from")
    if dead:
        raise SystemExit(
            "  " + "; ".join(dead) + "\n"
            "  Every joint has to be on the bus before anything is driven: one missing motor does\n"
            "  not stop a run, it makes its results quietly wrong.\n"
            "  On the bridge, bit 0x08 is a one-way safe-stop latch that only a restart clears;\n"
            "  0x01/0x02/0x04 mean the motor is not answering at all -- check power and the CAN\n"
            "  connector, then `ecat_setid --ids <n>` with the bridge stopped."
        )


def _safe_teardown(arm: Optional[Arm7OSC], chain: MotorChain, trace: Optional["_Trace"] = None) -> None:
    """Release the arm, then close the trace and the chain.

    SIGINT is masked for the few milliseconds the release takes, so a second Ctrl-C cannot skip it.
    The release runs first because the EtherCAT bridge re-applies the last command it received
    indefinitely, with no timeout of its own.
    """
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        try:
            chain.set_commands(np.zeros(len(chain)), get_state=False)
        except Exception as exc:
            logging.error(f"first release failed: {exc!r}")
        if arm is not None:
            try:
                arm.close()
            except Exception as exc:
                logging.error(f"arm.close failed: {exc!r}")
        _shutdown_chain(chain)
    finally:
        signal.signal(signal.SIGINT, previous)
    if trace is not None:
        trace.close()


def _shutdown_chain(chain: MotorChain) -> None:
    """Command zero torque, let the chain's send thread notice, then close it.

    ``DMChainCanInterface.close()`` clears ``running`` and closes the bus in the same breath while
    its send thread may be mid-transmission; the pause here lets the thread see the flag first.
    The motors are not disabled -- they hold the last command, zero torque, until the firmware's
    8 s CAN timeout drops them.
    """
    try:
        chain.set_commands(np.zeros(len(chain)), get_state=False)
    except Exception:
        pass
    chain.running = False
    time.sleep(0.3)
    try:
        chain.close()
    except Exception:
        pass


def _read_joint_state(chain: DMChainCanInterface) -> Tuple[np.ndarray, np.ndarray]:
    """Position and velocity of the arm joints, in model coordinates."""
    states = chain.read_states()
    pos = np.array([st.pos for st in states[:ARM_DOF]])
    vel = np.array([st.vel for st in states[:ARM_DOF]])
    return pos, vel


def _joint_limits(xml: str) -> np.ndarray:
    """(ARM_DOF, 2) joint ranges from the model."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(xml)
    return model.jnt_range[:ARM_DOF].copy()


def stage_pose(args: Args, cfg: dict, xml: str) -> None:
    """First look at the arm's actual joint angles, with zero torque commanded.

    This is where the encoder zeros get checked, and it has to happen before any torque: gravity
    compensation computes ``g(q)`` from these numbers, so a zero that is off by a revolution or a
    sign produces a confident, wrong torque at every joint.

    Nothing is commanded but zero, which in MIT mode means **the arm is limp and will fall**.
    """
    _confirm(
        args,
        "ENERGISE all 7 motors at ZERO torque. The arm goes limp and will fall unless it is "
        "resting on something or held. Support it first.",
    )
    chain = _build_chain(args, cfg, check=True)
    try:
        limits = _joint_limits(xml)
        pos, vel = _read_joint_state(chain)
        print("\n  joint      q (rad)      q (deg)    model range (rad)      within?")
        ok = True
        for i in range(ARM_DOF):
            lo, hi = limits[i]
            inside = lo <= pos[i] <= hi
            ok &= bool(inside)
            flag = "yes" if inside else "NO -- OUT OF RANGE"
            print(f"   {i + 1}     {pos[i]:+9.4f}   {np.degrees(pos[i]):+8.2f}   [{lo:+7.3f}, {hi:+7.3f}]   {flag}")

        print(f"\n  |velocity| max {np.abs(vel).max():.4f} rad/s (should be ~0 while limp and still)")
        if ok:
            print("\n  All joints read inside the model's ranges.")
            print("  Now check them against the arm you can see: a zero that is wrong by a")
            print("  constant offset still reads 'in range' and would poison gravity compensation.")
            print("  If a joint reads far from what the pose looks like, re-zero it with")
            print("  i2rt/motor_config_tool/set_zero.py before going any further.")
        else:
            print("\n  STOP. A joint reading outside the model's range means the zero is wrong,")
            print("  the model does not match this arm, or a motor's direction is inverted.")
            print("  Gravity compensation from these numbers would be wrong at every joint.")
    finally:
        _shutdown_chain(chain)


def stage_mirror(args: Args, cfg: dict, xml: str) -> None:
    """Drive a browser view of the model from the real arm's joint readings, at zero torque.

    Move the arm by hand and watch the model: a wrong ``directions`` entry shows as that joint
    moving the opposite way, a wrong encoder zero as the model sitting in a different pose.

    Zero torque throughout, so **the arm is limp and will fall unless supported**.
    """
    _confirm(
        args,
        "ENERGISE all 7 motors at ZERO torque so you can move the arm by hand. It goes limp.",
    )
    import mujoco
    import viser

    chain = _build_chain(args, cfg, check=True)
    try:
        model = mujoco.MjModel.from_xml_path(xml)
        data = mujoco.MjData(model)
        limits = _joint_limits(xml)

        mesh_geoms = [g for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
        server = viser.ViserServer(port=args.port)
        print(f"\n  open http://{args.host_hint}:{args.port} in a browser")
        print(f"  (or forward it: ssh -L {args.port}:localhost:{args.port} <this machine>)\n")

        handles = {}
        for gid in mesh_geoms:
            mid = model.geom_dataid[gid]
            v0, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            f0, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
            handles[gid] = server.scene.add_mesh_simple(
                f"/arm/geom_{gid}",
                model.mesh_vert[v0 : v0 + vn].copy(),
                model.mesh_face[f0 : f0 + fn].copy(),
                color=tuple(int(c * 255) for c in model.geom_rgba[gid][:3]),
            )

        # A world reference, so the camera can be pinned to physical reality. Without one the view
        # is ambiguous in a way that matters: a rotation axis is a pseudovector, so looking from the
        # mirrored side reverses exactly those joints whose axes lie in the mirror plane and leaves
        # the rest alone -- which is indistinguishable from a subset of inverted signs.
        server.scene.add_frame("/world", axes_length=0.25, axes_radius=0.008)
        # Label each arrow at its own tip. Viser draws axes red-green-blue for x-y-z, but a colour
        # convention is one more thing to remember correctly under time pressure, and remembering it
        # wrong is the same failure this reference exists to prevent.
        for axis, tip, text in (
            (0, (0.30, 0.0, 0.0), "+X  sideways, along the plate"),
            (1, (0.0, 0.30, 0.0), "+Y  OUT of the mounting plate"),
            (2, (0.0, 0.0, 0.30), "+Z  UP"),
        ):
            server.scene.add_label(f"/world/axis_{axis}", text, position=tip)

        with server.gui.add_folder("Pin the camera first"):
            server.gui.add_text(
                "step 1",
                initial_value="orbit until +Z points up on screen the way up is up in the room",
                disabled=True,
            )
            server.gui.add_text(
                "step 2",
                initial_value="then until +Y points out of the plate the way it really does",
                disabled=True,
            )
            server.gui.add_text(
                "step 3",
                initial_value="bend the arm into a clearly asymmetric pose; it should match on screen",
                disabled=True,
            )

        # A marker and a label at each joint's own anchor. Adjacent wrist joints move by similar
        # small amounts and are genuinely hard to tell apart by eye; naming them in the scene
        # removes the guess, and a reading that named the wrong one of a pair is the likeliest
        # explanation for two viewings disagreeing about which joints looked reversed.
        joint_dots = []
        joint_labels = []
        for j in range(ARM_DOF):
            joint_dots.append(server.scene.add_icosphere(f"/joints/dot_{j}", radius=0.018, color=(120, 120, 120)))
            joint_labels.append(server.scene.add_label(f"/joints/label_{j}", f"j{j + 1}"))

        # The links too, not only the joints. Instructions that say "hold link6" are unusable
        # without them: the naming is jointN drives linkN, so link6 is the 46 mm section between
        # joint6 and joint7, which nobody is going to pick out of a photograph. Labelled at each
        # body's own origin, which is where that link starts.
        link_labels = []
        for name in LINK_NAMES:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            link_labels.append((bid, server.scene.add_label(f"/links/{name}", name)))

        with server.gui.add_folder("Real arm"):
            gui_moving = server.gui.add_text("MOVING NOW", initial_value="-", disabled=True)
            gui_q = server.gui.add_text("q (deg)", initial_value="-", disabled=True)
            gui_lim = server.gui.add_text("near a limit?", initial_value="-", disabled=True)
            server.gui.add_text(
                "what to look for",
                initial_value="move one joint: the model must follow the same way, not mirrored",
                disabled=True,
            )
            server.gui.add_text(
                "telling joints apart",
                initial_value="wiggle a joint and read MOVING NOW -- do not identify it by eye",
                disabled=True,
            )

        print("  FIRST pin the camera using the world triad, then bend the arm into an obviously")
        print("  asymmetric pose and check the model shows the same shape. Only then judge joints:")
        print("  from a mirrored viewpoint a subset of joints looks reversed even when all are right.\n")
        print("  Each joint is marked and labelled j1..j7 in the scene, and the one you are turning")
        print("  lights up red -- read MOVING NOW rather than identifying a joint by eye. Adjacent")
        print("  wrist joints move by similar small amounts and are easy to confuse.\n")
        print("  Move the arm by hand. The model should follow exactly.")
        print("  A joint that moves the OPPOSITE way in the browser has its `directions` sign wrong.")
        print("  A model whose whole pose differs from the arm has a zero wrong.")
        print("  Ctrl-C when done.\n")

        prev = None
        while True:
            pos, vel = _read_joint_state(chain)
            data.qpos[:ARM_DOF] = pos
            mujoco.mj_kinematics(model, data)
            for gid in mesh_geoms:
                handles[gid].position = data.geom_xpos[gid].copy()
                quat = np.zeros(4)
                mujoco.mju_mat2Quat(quat, data.geom_xmat[gid])
                handles[gid].wxyz = quat

            # Put each marker on its own joint's anchor, and light up whichever is turning.
            rate = np.abs(vel) if prev is None else np.abs(pos - prev) * 30.0
            prev = pos.copy()
            lead = int(np.argmax(rate))
            active = rate[lead] > 0.05
            for j in range(ARM_DOF):
                anchor = data.xanchor[j].copy()
                joint_dots[j].position = anchor
                joint_labels[j].position = anchor + np.array([0.0, 0.0, 0.045])
                joint_dots[j].color = (255, 80, 80) if (active and j == lead) else (120, 120, 120)
            for bid, label in link_labels:
                # Offset sideways rather than up, so a link label never sits on top of the joint
                # label that shares roughly the same place.
                label.position = data.xpos[bid] + np.array([0.0, 0.05, 0.0])
            gui_moving.value = f"j{lead + 1}   {np.degrees(rate[lead]):.0f} deg/s" if active else "- (still)"

            gui_q.value = "  ".join(f"j{i + 1} {np.degrees(pos[i]):+.1f}" for i in range(ARM_DOF))
            near = [f"j{i + 1}" for i in range(ARM_DOF) if min(pos[i] - limits[i, 0], limits[i, 1] - pos[i]) < 0.15]
            gui_lim.value = ", ".join(near) if near else "no"
            time.sleep(1.0 / 30.0)
    except KeyboardInterrupt:
        print("\n  stopped")
    finally:
        _shutdown_chain(chain)


def stage_probe(args: Args, cfg: dict, xml: str) -> None:
    """Push a joint both ways at a fixed torque and report how far it moved and what it took to
    break away.

    Checks that the joint moves at all at a sane torque and moves opposite ways for opposite signs.
    It does **not** test ``directions``: the driver multiplies commanded torque and reported
    position by the same factor, so "+torque gives +q" is invariant under it. Use ``--stage
    mirror`` for that.
    """
    j = int(args.joint) - 1
    if not 0 <= j < ARM_DOF:
        raise SystemExit(f"--joint must be 1..{ARM_DOF}")
    tau = abs(float(args.probe_torque))
    rated = MotorType.get_motor_constants(cfg["motor_list"][j][1]).TORQUE_MAX
    if tau > 0.25 * rated:
        raise SystemExit(f"--probe-torque {tau} is over a quarter of joint {j + 1}'s {rated} N.m rating")

    signs = {"both": (+1.0, -1.0), "pos": (+1.0,), "neg": (-1.0,)}.get(args.probe_dir)
    if signs is None:
        raise SystemExit("--probe-dir must be both, pos or neg")

    limits = _joint_limits(xml)
    lo, hi = limits[j]
    _confirm(
        args,
        f"apply {tau:.2f} N.m to JOINT {j + 1} ({args.probe_dir}). Every other joint stays at zero "
        f"torque, i.e. limp. Be ready to hold that joint.",
    )
    chain = _build_chain(args, cfg, check=True)
    try:
        results = {}
        for sign in signs:
            cmd = np.zeros(ARM_DOF)
            time.sleep(0.5)
            q0, _ = _read_joint_state(chain)
            cmd[j] = sign * tau
            t0 = time.monotonic()
            peak = 0.0
            early = None
            stopped = ""
            while time.monotonic() - t0 < args.probe_seconds:
                chain.set_commands(cmd)
                q_now, v = _read_joint_state(chain)
                if abs(v[j]) > abs(peak):
                    peak = float(v[j])  # keep the sign: which way it was actually going
                # The direction it first breaks free in. Net displacement over the whole window
                # cannot tell a joint that drove one way from one that drove that way, reached a
                # stop and rebounded further.
                if early is None and abs(q_now[j] - q0[j]) > 0.01:
                    early = float(q_now[j] - q0[j])

                # Cut the torque before the joint runs anywhere. A fixed-duration push assumes the
                # joint meets resistance, and a joint that does not simply keeps accelerating for
                # the whole window -- one here travelled 212 degrees, past even the model's own
                # range, with nothing to stop it. A sign only needs a few degrees to read.
                travel = abs(q_now[j] - q0[j])
                if travel > args.probe_travel:
                    stopped = f"travel cap {np.degrees(args.probe_travel):.0f} deg"
                    break
                margin = 0.10
                if q_now[j] > hi - margin or q_now[j] < lo + margin:
                    stopped = f"within {np.degrees(margin):.0f} deg of the model's joint limit"
                    break
                if abs(v[j]) > args.probe_max_speed:
                    stopped = f"speed cap {args.probe_max_speed:.1f} rad/s"
                    break
                time.sleep(0.004)
            chain.set_commands(np.zeros(ARM_DOF))
            time.sleep(0.4)
            q1, _ = _read_joint_state(chain)
            delta = float(q1[j] - q0[j])
            results[sign] = (delta, peak, early)
            e = f"{early * 1e3:+6.1f}" if early is not None else "   n/a"
            if stopped:
                print(f"    cut short: {stopped}")
            print(
                f"  commanded {sign * tau:+.2f} N.m -> joint {j + 1}: first moved {e} mrad, "
                f"net {delta * 1e3:+8.1f} mrad, peak speed {peak:+.3f} rad/s"
            )
            if sign > 0:
                input("  return the joint roughly to where it started, then press Enter ...")

        if len(signs) == 1:
            _, _, e = results[signs[0]]
            print("\n  " + "-" * 62)
            if e is None:
                print(f"  Joint {j + 1} did not break free at {tau:.2f} N.m.")
            else:
                print(f"  Joint {j + 1} first moved {e * 1e3:+.1f} mrad under {signs[0] * tau:+.2f} N.m.")
                print("  One direction alone cannot separate the command from gravity or a stop;")
                print("  run the other way too where there is room.")
            return
        d_pos, d_neg = results[+1.0][0], results[-1.0][0]
        e_pos, e_neg = results[+1.0][2], results[-1.0][2]
        if e_pos is not None and e_neg is not None:
            same = np.sign(e_pos) == np.sign(e_neg)
            print(f"\n  first motion: {e_pos * 1e3:+.1f} mrad for +torque, {e_neg * 1e3:+.1f} for -torque")
            if same:
                print("  SAME sign both ways -- something other than the command is driving this joint")
            else:
                rel = "+torque -> +q" if e_pos > 0 else "+torque -> -q"
                print(f"  torque-to-motion relation: {rel}")
        print("\n  " + "-" * 62)
        if abs(d_pos) < 5e-3 and abs(d_neg) < 5e-3:
            print(f"  Joint {j + 1} barely moved either way -- below stiction at {tau:.2f} N.m, or held")
            print("  by gravity or an end stop. Raise --probe-torque and repeat.")
        elif np.sign(d_pos) == np.sign(d_neg):
            print(
                f"  Joint {j + 1} moved the SAME way for both signs ({d_pos * 1e3:+.0f} and {d_neg * 1e3:+.0f} mrad)."
            )
            print("  Gravity or an end stop is dominating, not the command. Move this joint to a more")
            print("  balanced pose or raise --probe-torque, then repeat.")
        else:
            print(
                f"  Joint {j + 1} responds to torque, and reverses with it "
                f"({d_pos * 1e3:+.0f} / {d_neg * 1e3:+.0f} mrad). Drive and feedback are consistent."
            )
            print(f"  Broke free at {tau:.2f} N.m, so stiction on this joint is under that.")
            print("  This says nothing about `directions` -- use --stage mirror for that.")
    finally:
        _shutdown_chain(chain)


def stage_survey(args: Args, cfg: dict, xml: str) -> None:
    """Measure §4.1's three numbers. P0 lists these as mandatory, and everything downstream uses them.

    Nothing is commanded: torque stays at zero throughout, so this is safe to run on a supported
    arm without an e-stop event.
    """
    chain = _build_chain(args, cfg, check=True)
    try:
        print(f"\nreading for {args.seconds:.0f} s at zero torque ...")
        stamps: List[np.ndarray] = []
        wall: List[float] = []
        zero = np.zeros(ARM_DOF)
        t0 = time.monotonic()
        while time.monotonic() - t0 < args.seconds:
            infos = chain.set_commands(zero)
            if infos:
                stamps.append(np.array([i.timestamp for i in infos[:ARM_DOF]]))
                wall.append(time.time())
            # Required. `set_commands` returns the receive thread's cached state without blocking,
            # so with no sleep this is a pure-Python busy loop holding the GIL, and the receive
            # thread cannot run until the interpreter's 5 ms switch interval expires. The
            # seven CAN round trips take 2.6 ms of that cycle; the other 6.1 ms was this loop.
            # Polling at 2 kHz catches every update up to 1 kHz and leaves the GIL alone.
            time.sleep(0.0005)

        arr = np.array(stamps)
        loop_hz = len(wall) / (wall[-1] - wall[0])

        # Not a control rate. `set_commands` returns whatever the receive thread last stored
        # rather than waiting for a reply, so this loop spins freely -- tens of kHz, or
        # 535 reads of the same sample. The rate that governs everything is the per-motor one
        # below. `comm_freq` reads 0 here because its recorder is only fed by the send thread,
        # which this read-only path does not drive.
        print(f"\n1. poll rate            {loop_hz:8.1f} Hz  ({len(wall)} reads, not a control rate)")
        print(
            f"   GIL switch interval  {sys.getswitchinterval() * 1e3:8.1f} ms  (a busy poll here depresses the rate below)"
        )
        print("\n2. per-motor feedback rate -- the number the watchdog and the gain derating key off")
        rates = []
        for j in range(ARM_DOF):
            changes = int(np.count_nonzero(np.diff(arr[:, j]) > 0))
            rate = changes / (wall[-1] - wall[0])
            rates.append(rate)
            ratio = loop_hz / rate if rate > 0 else float("inf")
            print(f"   joint{j + 1}  {rate:8.1f} Hz   = loop / {ratio:4.2f}")
        if loop_hz > 1.5 * min(rates):
            print(f"\n   -> usable task gain follows {min(rates):.0f} Hz, NOT the {loop_hz:.0f} Hz poll above.")
        else:
            print(f"\n   -> poll and feedback are matched at {min(rates):.0f} Hz; every read is a fresh sample.")

        print("\n3. inter-group skew -- how far apart, in time, one frame's readings were taken")
        spread = arr.max(axis=1) - arr.min(axis=1)
        print(
            f"   mean {spread.mean() * 1e3:6.2f} ms   p99 {np.percentile(spread, 99) * 1e3:6.2f} ms   "
            f"max {spread.max() * 1e3:6.2f} ms"
        )
        groups: dict = {}
        for j in range(ARM_DOF):
            key = round(float(np.median(arr[:, j] - arr.min(axis=1))) * 1e3, 1)
            groups.setdefault(key, []).append(j + 1)
        print(f"   apparent grouping by lag: {dict(sorted(groups.items()))}")

        print("\n4. is sending batched too? -- not measurable from feedback alone.")
        print("   Ask the EtherCAT-to-CAN owner directly. If it is, one computed torque vector is")
        print("   applied in pieces, and the task-space force realised differs from the designed one.")
        print("\nRecord all four in the acceptance report; P0 lists the first three as mandatory.")
        # The damping cap in --stage null is J/dt, and the dt that matters is how often a
        # *fresh* velocity arrives for the joint that gets one least often -- a damping term
        # recomputed from a stale reading is what diverges. So the slowest motor sets it, not the
        # loop.
        print(f"\n-> pass --loop-hz {min(rates):.0f} to --stage null.")
        print("   That is the slowest per-motor feedback rate, which is what the damping cap has to")
        print("   respect; assuming a higher one is what let joint6 diverge at 8.77 rad/s on")
        print("   The default is a deliberately pessimistic 100 Hz.")
    finally:
        _shutdown_chain(chain)


def stage_jitter(args: Args, cfg: dict, xml: str) -> None:
    """Bare-loop timing with no control law: the baseline every later frequency claim rests on.

    **This stage energises the arm and applies no gravity compensation**, since anything holding
    the arm up would be a control law. Torque is zero throughout; ``--motor-kd`` sets the motor-side
    damping, which closes inside the motor and does not affect the timing measured here. The arm
    sinks, so park it resting or supported.
    """
    motor_kd = args.motor_kd if args.motor_kd > 0 else JITTER_MOTOR_KD
    _confirm(
        args,
        f"energise all 7 motors with NO gravity compensation, motor kd {motor_kd:.1f}. "
        "The arm will sink. It must be resting or supported before you continue",
    )
    chain = _build_chain(args, cfg, check=True)
    timer = LoopTimer()
    try:
        print(f"\nspinning for {args.seconds:.0f} s at zero torque, motor kd {motor_kd:.1f} ...")
        zero = np.zeros(ARM_DOF)
        kd_vec = np.full(ARM_DOF, motor_kd)
        dt = 1.0 / 500.0
        next_tick = time.monotonic()
        t0 = time.monotonic()
        while time.monotonic() - t0 < args.seconds:
            chain.set_commands(zero, kd=kd_vec)
            timer.tick()
            next_tick += dt
            sleep = next_tick - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_tick = time.monotonic()
        st = timer.stats()
        print(
            f"\n  mean {st['mean_hz']:7.1f} Hz   p50 {st['p50_ms']:.3f}   p99 {st['p99_ms']:.3f}   "
            f"max {st['max_ms']:.3f} ms"
        )
        ok = st["p99_ms"] <= 3.0
        print(f"  P0 gate, p99 <= 3 ms: {'PASS' if ok else 'FAIL -- fix this before writing control law'}")
    finally:
        _shutdown_chain(chain)


class _Trace:
    """Background sampler writing joint state to CSV so a powered run can be read back after it.

    The event that matters in gravity compensation is the first second after it switches on, and
    that is over before anyone can read a terminal -- so it has to go to a file or it is lost. The
    sampler runs off ``get_observations``, which copies under the state lock, so it neither blocks
    the control loop nor sees a torn read of it.
    """

    def __init__(self, arm: Arm7OSC, path: str, hz: float = TRACE_HZ) -> None:
        self.path = path
        self.label = "start"
        self.bias_joint: Optional[int] = None
        self._arm = arm
        self._dt = 1.0 / hz
        self._stop = threading.Event()
        self._fh = open(path, "w")
        cols = ["t"]
        for tag in ("q", "dq", "tau"):
            cols += [f"{tag}{i + 1}" for i in range(ARM_DOF)]
        # How stale each motor's reading was when sampled. --stage survey measures the feedback
        # rate on an otherwise idle process; a stage adds a 500 Hz control thread, this sampler
        # and a display loop, all contending for the GIL, so the rate that actually governs
        # stability during the run is not the one the survey printed. These columns are that rate,
        # measured where it matters. They are also what the watchdog keys off.
        cols += [f"age{i + 1}" for i in range(ARM_DOF)]
        # The sweep's bias, logged separately from `tau` because the analysis needs it on its own:
        # `tau` is gravity plus bias, and recovering the bias by subtracting a recomputed gravity
        # would make the answer depend on the very model under test.
        cols += ["b"]
        self._fh.write(",".join(cols) + ",label\n")
        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def mark(self, label: str) -> None:
        """Name what the operator is doing, so the CSV can be read without guessing at timings."""
        self.label = label.replace(",", ";")

    def _run(self) -> None:
        while not self._stop.is_set():
            obs = self._arm.get_observations()
            bias = self._arm.torque_bias
            row = [
                time.monotonic() - self._t0,
                *obs["joint_pos"],
                *obs["joint_vel"],
                *obs["joint_eff"],
                *obs["feedback_age"],
                bias[self.bias_joint] if self.bias_joint is not None else 0.0,
            ]
            self._fh.write(",".join(f"{v:.5f}" for v in row) + f",{self.label}\n")
            time.sleep(self._dt)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._fh.flush()
        self._fh.close()
        print(f"\ntrace written: {self.path}")


def _gravity_factor(args: Args, cfg: dict) -> np.ndarray:
    """Per-joint gravity multiplier: the command line if given, else the config."""
    if not args.gravity_factor:
        return np.array(cfg["gravity_comp_factor"], dtype=float)
    vals = [float(x) for x in args.gravity_factor.replace(" ", "").split(",") if x]
    if len(vals) != ARM_DOF:
        raise SystemExit(f"--gravity-factor needs {ARM_DOF} numbers, got {len(vals)}")
    # Net torque on a gravity-compensated joint is (factor - 1) * g(q). With dg/dq > 0 about a
    # hanging pose, a factor above 1.0 is positive feedback: the joint accelerates away from the
    # minimum until gravity reverses. Stiction masks it until something moves the arm.
    over = [i for i, v in enumerate(vals) if v > 1.0]
    if over:
        names = ", ".join(f"joint{i + 1}={vals[i]:g}" for i in over)
        print(f"\n  !! gravity_comp_factor ABOVE 1.0 on {names}")
        print("     This is unstable, not merely strong: net torque is (factor-1)*g(q) and grows")
        print("     with deflection. It will sit still until you touch the arm, then run to the")
        print("     nearest place gravity reverses -- for a shoulder joint that is straight up.")
        print("     Tune toward 1.0 from BELOW. If the arm still sags at 1.0 the fault is the")
        print("     model, the torque clamp or friction, and more factor cannot fix any of them.\n")
    return np.array(vals)


def _make_arm(
    args: Args, cfg: dict, xml: str, auto_start: bool = True, **kw: object
) -> Tuple[DMChainCanInterface, Arm7OSC]:
    chain = _build_chain(args, cfg, check=True)
    if args.motor_kd >= 0.0:
        kw = dict(kw)
        kw["motor_kd_float"] = np.full(ARM_DOF, args.motor_kd)
        kw["motor_kd_active"] = np.full(ARM_DOF, args.motor_kd)
    arm = Arm7OSC(
        chain,
        xml,
        ee_site="grasp_site",
        mode=args.mode,
        motor_list=cfg["motor_list"],
        torque_limit=TORQUE_LIMIT_NM,
        auto_start=auto_start,
        gravity_comp_factor=_gravity_factor(args, cfg),
        **kw,  # type: ignore[arg-type]
    )
    print(f"motor-side kd while floating: {np.round(arm.motor_kd_float, 3).tolist()}")
    # Clamp below the application bound until the operator raises it deliberately.
    arm._torque_limit = np.minimum(arm._torque_limit, args.max_torque)
    print(f"torque clamp in force: {np.round(arm._torque_limit, 2).tolist()} N.m")
    print(f"gravity_comp_factor:   {np.round(_gravity_factor(args, cfg), 3).tolist()}")
    _check_motors_alive(chain)
    return chain, arm


def stage_float(args: Args, cfg: dict, xml: str) -> None:
    """Gravity compensation only. Move the arm by hand; it should stay where you leave it."""
    _confirm(args, f"apply gravity-compensation torque, clamped to {args.max_torque:.1f} N.m per joint")
    chain, arm = _make_arm(args, cfg, xml)
    trace = _Trace(arm, args.trace or f"/tmp/arm7_float_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    print(f"trace -> {trace.path}")
    try:
        arm.float_arm()
        print("\nGravity compensation is on. Move the arm by hand, let go, repeat. Ctrl-C to stop.")
        _watch_holds(arm, trace)
    finally:
        _safe_teardown(arm, chain, trace)


def _watch_holds(arm: Arm7OSC, trace: _Trace) -> None:
    """Live joint readout, with one summary line per hold."""
    still_since: Optional[float] = None
    q0 = ee0 = None
    reported = False
    try:
        while True:
            obs = arm.get_observations()
            q, speed = obs["joint_pos"], float(np.abs(obs["joint_vel"]).max())
            now = time.monotonic()
            if speed < STILL_SPEED:
                if still_since is None:
                    still_since, q0, ee0, reported = now, q.copy(), arm.get_ee_pose()[:3, 3].copy(), False
                held = now - still_since
                if held >= MIN_HOLD_S and not reported:
                    trace.mark("holding")
                    reported = True
                if reported:
                    dq = q - q0
                    worst = int(np.argmax(np.abs(dq)))
                    drift = float(np.linalg.norm(arm.get_ee_pose()[:3, 3] - ee0) * 1e3)
                    print(
                        f"\r  holding {held:5.1f} s   tool {drift:6.1f} mm   "
                        f"worst j{worst + 1} {dq[worst] * 1e3:+7.1f} mrad      ",
                        end="",
                        flush=True,
                    )
            else:
                if reported:
                    print()  # keep the finished hold's last line
                elif still_since is not None:
                    print("\r" + " " * 78 + "\r", end="")
                still_since, reported = None, False
                trace.mark("moving")
                print(
                    "\r  moving   " + " ".join(f"{np.degrees(v):+6.1f}" for v in q) + " deg     ",
                    end="",
                    flush=True,
                )
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n\nstopped.")


def _goto_start_pose(args: Args, arm: Arm7OSC, xml: str) -> None:
    """Drive the arm to ``--start-pose`` under position control, then leave it there.

    The task gains have to be live for this, which they are not in every stage -- ``null`` zeroes
    them precisely so the nullspace can be seen alone. So the gains are restored for the move and
    put back afterwards, and the caller re-takes its target from where the arm actually arrived
    rather than from where it was asked to go.

    Moves through ``Arm7OSC.move_joints``, which walks the *command* rather than stepping it, so
    the arm is never asked for more velocity than the gains can deliver.
    """
    if not args.start_pose:
        return
    vals = [float(x) for x in args.start_pose.replace(" ", "").split(",") if x]
    if len(vals) != ARM_DOF:
        raise SystemExit(f"--start-pose needs {ARM_DOF} numbers in degrees, got {len(vals)}")
    goal = np.radians(np.array(vals))
    limits = _joint_limits(xml)
    if np.any(goal < limits[:, 0]) or np.any(goal > limits[:, 1]):
        raise SystemExit(f"--start-pose is outside the model's joint ranges: {np.round(vals, 1).tolist()}")

    # This is a joint-space move, so drive it with the joint-space law whatever the stage is
    # about to run. In `full` the control law reads `kp_cartesian`, which the lines below do not
    # touch -- so a `--mode full` goto would run on the stage's own Cartesian stiffness, which is
    # far too soft for a 45 degree move and arrives nowhere near the target. Under `kinematic` the
    # same move lands a few degrees short, which is the stiction dead zone at a quarter stiffness.
    saved_mode = arm.mode
    arm.mode = OSCMode.KINEMATIC
    saved = (arm.kp_task.copy(), arm.kd_task.copy(), arm.kp_cartesian.copy(), arm.kd_cartesian.copy())
    # Soft gains, not the shipped ones. This move only has to be slow and to arrive; it does not
    # have to be stiff, and stiffness is what bit here -- joint6 has 0.0087 kg.m^2 of reflected
    # inertia and 0.04 N.m of friction, the least of either on the arm, so driving it at the
    # shipped kp=15 set it oscillating at 4 rad/s and tripped the velocity guard before the arm
    # had arrived. A quarter of the stiffness moves the same 45 degrees just as well.
    arm.kp_task = KP_TASK_JOINT * GOTO_GAIN_SCALE
    arm.kd_task = KD_TASK_JOINT * np.sqrt(GOTO_GAIN_SCALE)
    here = arm.get_joint_pos()
    print(f"\nmoving to {np.round(vals, 1).tolist()} deg over {args.start_seconds:.0f} s")
    print(f"  from {np.round(np.degrees(here), 1).tolist()}")
    print("  hands off, watch it; Ctrl-C stops the move")
    try:
        arm.enable_arm()
        arm.move_joints(goal, duration=args.start_seconds)
        # Keep commanding while it settles. COMMAND_TIMEOUT_S is 0.3 s and a silent pause trips it,
        # which drops the arm to DAMPED -- and DAMPED needs an explicit resume(), so the stage
        # would then begin in a state that ignores everything it does.
        settle = time.monotonic() + 1.0
        while time.monotonic() < settle:
            arm.command_joint_pos(goal)
            time.sleep(0.02)
    finally:
        arm.kp_task, arm.kd_task, arm.kp_cartesian, arm.kd_cartesian = saved
        arm.mode = saved_mode
    if arm.fault is not None:
        # Report it here rather than letting the caller's enable_arm() raise: the fault is from
        # the move, and the stage that follows has nothing to do with it.
        raise SystemExit(
            f"the move to --start-pose faulted: {arm.fault}\n"
            f"  reached {np.round(np.degrees(arm.get_joint_pos()), 1).tolist()} deg\n"
            "  lower GOTO_GAIN_SCALE or lengthen --start-seconds, then try again"
        )
    reached = arm.get_joint_pos()
    err = np.degrees(np.abs(reached - goal)).max()
    print(f"  arrived at {np.round(np.degrees(reached), 1).tolist()} deg   worst error {err:.1f} deg")
    if err > 10.0:
        print("  that is a long way short. Most likely the torque clamp is below what the pose")
        print("  needs -- raise --max-torque a step -- or a joint is against a stop.")


SWEEP_CARRIERS: dict = {
    # Every pose here must lie inside the current model's joint ranges; re-check on a model change.
    # Joint -> the pose the rest of the arm must be put in before that joint's own gravity torque
    # is large enough to measure. joint3 and joint5 are roll axes: at the hanging pose their own
    # gravity torque is identically 0.00 N.m, so a sweep there measures nothing at all. Degrees,
    # searched over j2/j4/j6 for the largest span of g_i as joint i moves.
    1: ((0, 0, 0, 0, 0, 0, 0), 29.6),
    2: ((0, 0, 0, 0, 0, 0, 0), 20.1),
    3: ((0, 90, 0, 90, 0, 30, 0), 6.6),
    4: ((0, 0, 0, 0, 0, 0, 0), 10.5),
    5: ((0, 90, 0, 90, 0, -45, 0), 1.0),
    6: ((0, 0, 0, 0, 0, 0, 0), 1.3),
    7: ((0, 0, 0, 0, 0, 0, 0), 1.4),
}


def stage_sweep(args: Args, cfg: dict, xml: str) -> None:
    """Identify one joint's gravity-model error and its Coulomb friction, by creeping it both ways.

    Gravity compensation stays on and a slow bias is added to the swept joint only, so the arm is
    supported throughout and no position control is engaged. At a quasi-static creep the bias
    settles at the model error plus friction, and friction reverses with direction::

        b'      = b - kd*q_dot                    what the joint received, less damping
        e(q)   = (b'_up + b'_down)/2             the gravity model error
        tau_f  = (b'_up - b'_down)/2             coulomb_friction

    Writes a CSV; read it with ``examples/arm7_osc/analyze_sweep.py``.
    """
    j = int(args.joint)
    if not 1 <= j <= ARM_DOF:
        raise SystemExit(f"--joint must be 1..{ARM_DOF}, got {j}")
    carrier, span_nm = SWEEP_CARRIERS[j]
    print(f"\nSweeping joint {j}. Gravity varies by about {span_nm:.1f} N.m over its travel here.")
    print("Set the rest of the arm, by hand, to roughly:")
    print("   " + "  ".join(f"j{i + 1} {carrier[i]:+d}" for i in range(ARM_DOF)) + "   degrees")
    if span_nm < 2.0:
        print(f"\n   NOTE: only {span_nm:.1f} N.m of signal. Transmission friction on this joint is")
        print("   likely the same order, so expect a poor separation and treat the result as a")
        print("   bound rather than a value. gravity.md says not to chase the wrist joints.")

    _confirm(args, f"creep joint {j} both ways under gravity compensation, bias <= {args.sweep_bias_limit:.1f} N.m")
    chain, arm = _make_arm(args, cfg, xml)
    # The carrier is this joint's own business, so the stage drives to it rather than asking for
    # it: a hanging arm cannot be held in a raised posture by hand, and a sweep taken at the wrong
    # carrier measures the wrong thing -- joint3 and joint5 see no gravity at all at the hang.
    # An explicit --start-pose still wins, for the case where the table or a fixture is in the way.
    if not args.start_pose:
        args = replace(args, start_pose=",".join(str(v) for v in carrier))
    trace = _Trace(arm, args.trace or f"/tmp/arm7_sweep_j{j}_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    trace.bias_joint = j - 1
    print(f"trace -> {trace.path}")
    try:
        _goto_start_pose(args, arm, xml)
        arm.float_arm()
        time.sleep(0.5)
        q0 = arm.get_joint_pos()[j - 1]
        if not (np.isnan(args.sweep_from) or np.isnan(args.sweep_to)):
            lo, hi = sorted((np.radians(args.sweep_from), np.radians(args.sweep_to)))
            # Start from inside the band, so the first pass does not have to travel to reach it.
            if not lo <= q0 <= hi:
                print(f"  joint {j} is at {np.degrees(q0):+.1f} deg, outside the requested band;")
                print("  it will creep into the band before the measured pass begins")
        else:
            lo, hi = q0 - np.radians(args.sweep_span), q0 + np.radians(args.sweep_span)
        limits = _joint_limits(xml)
        lo = max(lo, limits[j - 1, 0] + np.radians(5.0))
        hi = min(hi, limits[j - 1, 1] - np.radians(5.0))
        print(f"start {np.degrees(q0):+.1f} deg, sweeping {np.degrees(lo):+.1f} .. {np.degrees(hi):+.1f} deg")
        print("Hands off from here. Ctrl-C stops and zeroes the bias.\n")

        bias = np.zeros(ARM_DOF)
        if args.sweep_mode == "breakaway":
            _run_breakaway(args, arm, trace, j, lo, hi)
            print(f"\nNow: python examples/arm7_osc/analyze_sweep.py {trace.path} --joint {j}")
            return
        if args.sweep_mode != "creep":
            raise SystemExit(f"--sweep-mode must be 'creep' or 'breakaway', got {args.sweep_mode!r}")
        # Up to the top of the band, then down to the bottom, so both passes cover the same
        # ground -- the analysis pairs them by angle and can only use the overlap.
        for direction, target in ((+1.0, hi), (-1.0, lo)):
            trace.mark(f"j{j} creep {'up' if direction > 0 else 'down'}")
            t0 = time.monotonic()
            while time.monotonic() - t0 < args.seconds:
                obs = arm.get_observations()
                q, dq = obs["joint_pos"][j - 1], obs["joint_vel"][j - 1]
                if (direction > 0 and q >= target) or (direction < 0 and q <= target):
                    break
                # Integral speed regulation, deliberately slack: the bias must follow the torque
                # the joint needs, not chase a velocity setpoint. A stiff loop here would mask the
                # very quantity being measured.
                bias[j - 1] += SWEEP_KI * (args.sweep_speed * direction - dq) * SWEEP_DT
                bias[j - 1] = float(np.clip(bias[j - 1], -args.sweep_bias_limit, args.sweep_bias_limit))
                arm.set_torque_bias(bias)
                print(
                    f"\r  {'up ' if direction > 0 else 'down'}  q {np.degrees(q):+7.1f} deg   "
                    f"dq {dq:+5.2f} rad/s   bias {bias[j - 1]:+6.3f} N.m   ",
                    end="",
                    flush=True,
                )
                time.sleep(SWEEP_DT)
            bias[j - 1] = 0.0
            arm.set_torque_bias(bias)
            trace.mark(f"j{j} settle")
            print("\n  direction done, bias released; letting it settle\n")
            time.sleep(1.5)
        print(f"Now: python examples/arm7_osc/analyze_sweep.py {trace.path} --joint {j}")
    except KeyboardInterrupt:
        print("\n\nstopped.")
    finally:
        arm.set_torque_bias(np.zeros(ARM_DOF))
        _safe_teardown(arm, chain, trace)


def _run_breakaway(args: Args, arm: Arm7OSC, trace: "_Trace", j: int, lo: float, hi: float) -> None:
    """Load the joint until it lets go, release, repeat -- alternating direction, walking the range.

    One release is one reading, so this loop's job is to produce many of them at comparable angles.
    Alternating direction each cycle keeps the joint oscillating about one place rather than
    marching off, which is what makes the two directions pairable at all; the cluster then steps
    along the travel so the readings are not all from one pose.

    The bias ramps from zero every cycle rather than being held, so each reading is an independent
    breakaway and a joint that crept slightly last time does not carry its old bias into the next.
    """
    bias = np.zeros(ARM_DOF)
    released = np.radians(BREAKAWAY_RELEASE_DEG)
    print(f"  {args.breakaway_cycles} cycles per direction, stepping {args.breakaway_step:.0f} deg between clusters")
    while True:
        here = arm.get_joint_pos()[j - 1]
        if not lo <= here <= hi:
            print(f"\n  joint {j} left the band at {np.degrees(here):+.1f} deg; stopping")
            return
        for cycle in range(args.breakaway_cycles):
            for direction in (+1.0, -1.0):
                trace.mark(f"j{j} load {'+' if direction > 0 else '-'}")
                start = arm.get_joint_pos()[j - 1]
                bias[j - 1] = 0.0
                t0 = time.monotonic()
                while time.monotonic() - t0 < BREAKAWAY_TIMEOUT_S:
                    bias[j - 1] += direction * args.sweep_bias_limit * SWEEP_DT / BREAKAWAY_RAMP_S
                    bias[j - 1] = float(np.clip(bias[j - 1], -args.sweep_bias_limit, args.sweep_bias_limit))
                    arm.set_torque_bias(bias)
                    moved = arm.get_joint_pos()[j - 1] - start
                    print(
                        f"\r  cycle {cycle + 1}/{args.breakaway_cycles} {'+' if direction > 0 else '-'}  "
                        f"q {np.degrees(arm.get_joint_pos()[j - 1]):+7.1f}   bias {bias[j - 1]:+6.3f} N.m   "
                        f"moved {np.degrees(moved):+5.1f} deg  ",
                        end="",
                        flush=True,
                    )
                    if abs(moved) > released:
                        break
                    time.sleep(SWEEP_DT)
                bias[j - 1] = 0.0
                arm.set_torque_bias(bias)
                trace.mark(f"j{j} settle")
                time.sleep(BREAKAWAY_SETTLE_S)
        target = min(here + np.radians(args.breakaway_step), hi)
        if target <= here + np.radians(1.0):
            print("\n  reached the top of the band")
            return
        trace.mark(f"j{j} step")
        t0 = time.monotonic()
        while arm.get_joint_pos()[j - 1] < target and time.monotonic() - t0 < BREAKAWAY_TIMEOUT_S:
            bias[j - 1] = min(bias[j - 1] + args.sweep_bias_limit * SWEEP_DT / BREAKAWAY_RAMP_S, args.sweep_bias_limit)
            arm.set_torque_bias(bias)
            time.sleep(SWEEP_DT)
        bias[j - 1] = 0.0
        arm.set_torque_bias(bias)
        time.sleep(BREAKAWAY_SETTLE_S)
        print()


ENCODER_RESOLUTION_RAD = 25.0 / 65536.0
"""One count of the MIT position field: 16 bits over the +/-12.5 rad every DM type shares."""

KD_NOISE_MARGIN = 0.5
"""How much of a joint's stiction the quantisation-driven torque noise may reach."""


JITTER_MOTOR_KD = 1.0
"""Motor-side damping held during --stage jitter, which carries no control law and therefore no
gravity compensation. Enough that the arm sinks rather than drops; not enough to hold it up. It
closes inside the motor, so it does not appear in the loop timing this stage measures."""

KP_QUANTISATION_MARGIN = 0.3
"""How much of a joint's stiction a single encoder count is allowed to command."""


def _cap_task_stiffness(kp: np.ndarray, friction: np.ndarray) -> np.ndarray:
    """Cap each joint's stiffness at ``KP_QUANTISATION_MARGIN * friction / encoder_resolution``.

    One encoder count of error commands ``kp * resolution`` of torque. Above this cap that reaches
    the joint's breakaway friction, so a single count is enough to move it and the joint hunts.
    Returns the capped vector and prints each joint it bound on.
    """
    cap = KP_QUANTISATION_MARGIN * friction / ENCODER_RESOLUTION_RAD
    capped = np.minimum(kp, cap)
    if np.any(capped < kp - 1e-9):
        print("\n  task stiffness capped where one encoder count would command the stiction:")
        for i in range(ARM_DOF):
            if capped[i] < kp[i] - 1e-9:
                print(
                    f"    j{i + 1}  kp {kp[i]:.1f} -> {capped[i]:.1f}   "
                    f"(stiction {friction[i]:.2f} N.m; one count at {kp[i]:.0f} commands "
                    f"{kp[i] * ENCODER_RESOLUTION_RAD:.3f} N.m)"
                )
    return capped


def _cap_host_damping(kd: np.ndarray, friction: np.ndarray, loop_hz: float) -> np.ndarray:
    """Cap host-side damping at ``KD_NOISE_MARGIN * friction / (encoder_resolution * loop_hz)``.

    Velocity is differentiated from a quantised position, so one count between samples reads as
    ``resolution * loop_hz`` of apparent speed. Multiplied by the damping gain that becomes a
    torque; above this cap it exceeds the joint's breakaway friction and sustains a limit cycle.
    Returns the capped vector and prints each joint it bound on. Damping removed here can be put
    on the motor instead with ``--motor-kd``.
    """
    noise = ENCODER_RESOLUTION_RAD * loop_hz
    cap = KD_NOISE_MARGIN * friction / noise
    capped = np.minimum(kd, cap)
    if np.any(capped < kd - 1e-9):
        print(f"\n  host damping capped against {noise:.3f} rad/s of quantisation noise:")
        for i in range(ARM_DOF):
            if capped[i] < kd[i] - 1e-9:
                print(
                    f"    j{i + 1}  kd {kd[i]:.2f} -> {capped[i]:.2f}   "
                    f"(stiction {friction[i]:.2f} N.m, cap {cap[i]:.2f})"
                )
        print("    raise --motor-kd to put that damping where it has no lag.")
    return capped


def _check_modes(
    xml: str,
    q: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
    kpn: float,
    kdn: float,
    loop_hz: float,
    motor_kd: float = 0.0,
    *,
    mode: str = "kinematic",
    kp_cart: Optional[np.ndarray] = None,
    kd_cart: Optional[np.ndarray] = None,
    friction: Optional[np.ndarray] = None,
) -> None:
    """Refuse a gain set whose closed-loop modes are too fast for the feedback rate.

    With ``kp_task`` on one projector and ``kp_null`` on the other, the mass matrix couples them,
    so the modes are the eigenvalues of

    ::

        M qdd + (N_t Kd + N_n kd_n) qd + (N_t Kp + N_n kp_n) q = 0

    which is what this computes. A per-joint ``kd dt / J`` bound is not a safe stand-in: it
    disagrees in both directions, refusing gain sets the modes clear and passing ones they do not.

    ``mode`` selects which closed loop. In ``full`` the task layer is a Cartesian impedance, so
    ``kp``/``kd`` are not what the arm is driven with -- pass ``kp_cart``/``kd_cart`` and this maps
    them to their joint-space equivalent ``J^T Mx Kp6 J`` first.

    **The thresholds are calibrated for ``kinematic`` and are advisory in ``full``**, where nothing
    here binds at any gain scale tried while simulation still found a ceiling. Inertia shaping is
    why: ``J^T Mx Kp6 J`` puts the equivalent joint stiffness far under the friction cap. Raise
    ``--gain-scale`` there by observation, not by trusting a pass.

    Two thresholds: ``w dt >= 1`` refuses, six samples per cycle being the edge of representable,
    and ``w dt >= 0.5`` warns, about twelve samples per cycle being where a sampled loop stops
    behaving like the continuous one it was designed as.
    """
    import mujoco

    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    data.qpos[:ARM_DOF] = q
    mujoco.mj_forward(model, data)
    full = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, full, data.qM)
    mass = full[:ARM_DOF, :ARM_DOF]
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
    mujoco.mj_jacSite(model, data, jacp, jacr, site)
    jac = np.vstack([jacp[:, :ARM_DOF], jacr[:, :ARM_DOF]])
    if mode == "kinematic":
        n_task = np.linalg.pinv(jac) @ jac
        n_null = np.eye(ARM_DOF) - n_task
        stiff = n_task @ np.diag(kp) + n_null * kpn
        damp = n_task @ np.diag(kd) + n_null * kdn + np.eye(ARM_DOF) * motor_kd
    else:
        if kp_cart is None or kd_cart is None:
            raise SystemExit("  --mode full needs kp_cart/kd_cart to check its real modes")
        print(
            "\n  NOTE: these thresholds are calibrated for --mode kinematic on this arm and are\n"
            "  advisory for --mode full, where nothing here binds at any gain scale tried while\n"
            "  simulation still found a ceiling. Start at --gain-scale 0.25 and raise by ear."
        )
        # tau_task = J^T Mx (Kp6 e_x - Kd6 J qd), and e_x ~ J e_q for the small errors a mode
        # analysis is about, so the joint-space stiffness the modes see is J^T Mx Kp6 J.
        minv_jt = np.linalg.solve(mass, jac.T)
        mass_x = np.linalg.pinv(jac @ minv_jt, rcond=1e-4)
        jbar = minv_jt @ mass_x
        # Dynamically consistent, hence not symmetric. Eigenvalues do not care.
        n_null = np.eye(ARM_DOF) - jac.T @ jbar.T
        jt_mx = jac.T @ mass_x
        stiff = jt_mx @ np.diag(kp_cart) @ jac + n_null * kpn
        damp = jt_mx @ np.diag(kd_cart) @ jac + n_null * kdn + np.eye(ARM_DOF) * motor_kd

        # The kinematic path caps per-joint stiffness against friction. A Cartesian gain has no
        # per-joint knob to cap, so the same criterion has to refuse instead: above it, one
        # encoder count of stiffness exceeds what friction can absorb and the joint hunts.
        if friction is not None:
            equiv = np.abs(np.diag(stiff))
            cap = KP_QUANTISATION_MARGIN * np.asarray(friction, dtype=float) / ENCODER_RESOLUTION_RAD
            over = equiv > cap
            if over.any():
                room = float(np.min(cap[over] / equiv[over]))
                names = ", ".join(f"j{i + 1} ({equiv[i]:.0f} vs {cap[i]:.0f})" for i in np.flatnonzero(over))
                raise SystemExit(
                    f"  equivalent joint stiffness is past the quantisation cap on {names}.\n"
                    f"  Cartesian gains have no per-joint cap to apply, so this refuses rather than\n"
                    f"  silently reshaping them. Scale --gain-scale by about {room:.2f} and re-run."
                )
    state = np.block(
        [
            [np.zeros((ARM_DOF, ARM_DOF)), np.eye(ARM_DOF)],
            [-np.linalg.solve(mass, stiff), -np.linalg.solve(mass, damp)],
        ]
    )
    eig_all = np.linalg.eigvals(state)

    # Divergence is checked over *every* pole, including the real ones. `full` at a low gain scale
    # is entirely overdamped, so a real positive pole there is not a corner case, and the
    # oscillatory test below would not see it.
    unstable = eig_all[eig_all.real > 1e-9]
    if unstable.size:
        worst = unstable[np.argmax(unstable.real)]
        raise SystemExit(
            f"  a closed-loop pole is in the right half plane (Re {worst.real:+.1f} rad/s).\n"
            "  This diverges at any sample rate; it is the gain *shape*, not the bus."
        )

    # Oscillatory modes only: `w*dt` counts samples per cycle and a real pole has no cycle. The
    # fast real poles here come from the motor-side kd, which closes inside the motor and is
    # effectively continuous to this loop.
    eig = eig_all[eig_all.imag > 1e-6]
    if eig.size == 0:
        print("\n  closed-loop modes at %.0f Hz feedback: all real (overdamped), none to sample." % loop_hz)
        return
    freq = np.abs(eig) / (2 * np.pi)
    zeta = -eig.real / np.abs(eig)
    wdt = np.abs(eig) / loop_hz

    print(f"\n  closed-loop modes at {loop_hz:.0f} Hz feedback:")
    print(f"    {'f (Hz)':>9}{'zeta':>8}{'w*dt':>8}")
    for i in np.argsort(-freq):
        print(f"    {freq[i]:>9.1f}{zeta[i]:>8.2f}{wdt[i]:>8.2f}")
    if wdt.max() >= 1.0 or zeta.min() <= 0.15:
        raise SystemExit(
            f"  fastest mode is {freq[np.argmax(wdt)]:.1f} Hz (w*dt {wdt.max():.2f}, min zeta "
            f"{zeta.min():.2f}).\n"
            + (
                "  The mode is too fast for the feedback rate: lower --gain-scale, or use a bus\n"
                "  that feeds back faster.\n"
                if wdt.max() >= 1.0
                else "  The mode is slow enough; what is short is damping. Raise --motor-kd, which\n"
                "  damps from the rotor encoder inside the motor's own loop with none of this\n"
                "  transport's lag -- a faster bus would not help, because the host-side cap is\n"
                "  quantisation noise and that grows with the rate.\n"
            )
        )
    if wdt.max() >= 0.5:
        print(
            f"    WARNING: fastest mode {freq[np.argmax(wdt)]:.1f} Hz is w*dt {wdt.max():.2f}, under\n"
            f"    six to twelve samples per cycle at {loop_hz:.0f} Hz. Expect ringing the rigid model\n"
            "    does not predict. Back off a step if it buzzes."
        )


def stage_task(args: Args, cfg: dict, xml: str) -> None:
    """Task-space stiffness with a soft nullspace: push the tool and it springs back, push the
    elbow and it gives way.

    Every trace sample is labelled ``tool`` when the disturbance shows in the tool pose and
    ``posture`` when the nullspace absorbs it; the split is read from the CSV afterwards. Raise
    ``--gain-scale`` to find where it rings. The gain caps and the modal check run first and refuse
    anything the feedback rate cannot carry.
    """
    args = replace(args, start_pose=args.start_pose or NULLSPACE_POSE)
    q_start = np.radians([float(x) for x in args.start_pose.replace(" ", "").split(",") if x])
    if args.mode == "kinematic":
        friction = np.asarray(cfg["coulomb_friction"], dtype=float)
        kp = _cap_task_stiffness(KP_TASK_JOINT * args.gain_scale, friction)
        kd = _cap_host_damping(
            KD_TASK_JOINT * np.sqrt(args.gain_scale),
            friction,
            args.loop_hz,
        )
        kw: dict = dict(kp_task=kp, kd_task=kd)
    else:
        # Kp6 has units of 1/s^2, not N/m: Mx carries the mass, so the task bandwidth is sqrt(Kp6)
        # and critical damping is 2*sqrt(Kp6). `--cart-zeta` scales that.
        kp6 = 40.0 * args.gain_scale
        kd6 = 2.0 * args.cart_zeta * np.sqrt(kp6)
        kp = KP_TASK_JOINT * args.gain_scale
        kd = np.full(ARM_DOF, kd6)
        kw = dict(kp_cartesian=np.full(6, kp6), kd_cartesian=np.full(6, kd6))
        print(
            f"\n  full mode: Kp6 {kp6:.0f} (task bandwidth {np.sqrt(kp6) / (2 * np.pi):.2f} Hz), "
            f"Kd6 {kd6:.1f} (zeta {args.cart_zeta:.2f})"
        )
    _check_modes(
        xml,
        q_start,
        kp,
        kd,
        KP_NULL,
        KD_NULL,
        args.loop_hz,
        max(args.motor_kd, 0.0),
        mode=args.mode,
        kp_cart=kw.get("kp_cartesian"),
        kd_cart=kw.get("kd_cartesian"),
        friction=np.asarray(cfg["coulomb_friction"], dtype=float),
    )

    _confirm(args, f"enable OSC with task gains at {args.gain_scale:.2f}x the shipped value")
    chain, arm = _make_arm(args, cfg, xml, **kw)
    trace = _Trace(arm, args.trace or f"/tmp/arm7_task_{time.strftime('%Y%m%d_%H%M%S')}.csv", hz=args.trace_hz)
    print(f"trace -> {trace.path}")
    try:
        # Not optional. At the hanging pose cond(J) is 47968 and the nullspace has two dimensions
        # instead of one, so "task space" is not well defined there and a stiffness put on it goes
        # somewhere unintended. At the commissioning pose cond(J) is 7 to 17.
        _goto_start_pose(args, arm, xml)
        arm.enable_arm()
        target = arm.get_joint_pos().copy()
        print(f"\nACTIVE at {args.gain_scale:.2f}x.")
        print("  1. push the TOOL sideways, a centimetre or two, and let go -- it should spring back")
        print("  2. push the ELBOW instead -- it should give way, and the tool should stay put")
        print("Stop and step back if you hear buzzing or the velocity column jumps.\n")
        peak = 0.0
        t0 = time.monotonic()
        while time.monotonic() - t0 < args.seconds:
            arm.command_joint_pos(target)
            obs = arm.get_observations()
            q = obs["joint_pos"]
            speed = float(np.abs(obs["joint_vel"]).max())
            peak = max(peak, speed)
            tool_err = float(np.linalg.norm(obs["ee_pos_err"]) * 1e3)
            # Split the joint displacement: what the tool sees, and what the posture absorbed.
            # This is the whole point of the stage, so it goes in the label rather than being
            # left for the reader to recompute.
            jac = arm._site_jacobian(q)
            n_null = np.eye(ARM_DOF) - np.linalg.pinv(jac) @ jac
            err = q - target
            posture = float(np.degrees(np.linalg.norm(n_null @ err)))
            if speed < STILL_SPEED and tool_err < 2.0 and posture < 1.0:
                trace.mark("settled")
            else:
                trace.mark("tool" if tool_err > 2.0 else "posture")
            print(
                f"\r  {arm.state.value:8s}  tool err {tool_err:6.2f} mm   posture {posture:5.2f} deg"
                f"   |dq| {speed:5.3f} peak {peak:5.3f} rad/s  ",
                end="",
                flush=True,
            )
            if arm.state is not ArmState.ACTIVE:
                print(f"\n  STOPPED: {arm.fault}")
                print(f"  -> {args.gain_scale:.2f}x is past the limit. Record it and back off.")
                break
            time.sleep(0.02)
        else:
            print(f"\n  held at {args.gain_scale:.2f}x, peak |dq| {peak:.3f} rad/s. Step up and repeat.")
    except KeyboardInterrupt:
        print()
    finally:
        _safe_teardown(arm, chain, trace)


TRACK_DT = 0.01
"""Interval between commanded points, 100 Hz. Well above the 8 s period being traced and well
below the feedback rate, so the reference is smooth in both directions."""


def _track_plan(
    xml: str, q0: np.ndarray, amp_deg: np.ndarray, period_s: float, cycles: float, rate_hz: float
) -> Tuple[np.ndarray, np.ndarray]:
    """A joint-space sinusoid about ``q0``, and the tool path it traces.

    Each joint gets its own phase, spread evenly, so the arm does not simply swing back and forth
    along one direction -- a trajectory where every joint moves in step exercises one line through
    the workspace and would let a badly conditioned pose pass unnoticed.

    Returns the joint path and the commanded tool positions, so the caller can check both the joint
    limits and the conditioning before anything is powered.
    """
    import mujoco

    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")

    n = max(2, round(float(cycles) * float(period_s) * float(rate_hz)))
    t = np.arange(n) / rate_hz
    phase = np.linspace(0.0, np.pi, ARM_DOF, endpoint=False)
    # 1 - cos starts at zero displacement AND zero velocity, so the arm eases in rather than being
    # asked for a step in speed at t = 0.
    shape = 0.5 * (1.0 - np.cos(2 * np.pi * t[:, None] / period_s + phase[None, :]))
    shape -= shape[0]
    path = q0[None, :] + np.radians(amp_deg)[None, :] * shape

    tool = np.empty((n, 3))
    for i, q in enumerate(path):
        data.qpos[:ARM_DOF] = q
        mujoco.mj_forward(model, data)
        tool[i] = data.site_xpos[site]
    return path, tool


def _validate_track_path(xml: str, path: np.ndarray, tool: np.ndarray, when: str) -> None:
    """Refuse a sinusoid that reaches a joint stop or a near-singular pose.

    Called twice, and both times matter. Once offline on ``--start-pose``, and again on the pose
    the arm actually reached -- the goto runs at a quarter stiffness and stops inside the stiction
    dead zone, several degrees short. Validating only the requested pose certifies a
    configuration the arm is never in, which is the same defect the nullspace arc had before it
    replanned from where it arrived.
    """
    import mujoco

    limits = _joint_limits(xml)
    margin = np.radians(2.0)
    if np.any(path < limits[:, 0] + margin) or np.any(path > limits[:, 1] - margin):
        bad = int(np.argmax(np.maximum(limits[:, 0] - path.min(axis=0), path.max(axis=0) - limits[:, 1])))
        raise SystemExit(
            f"  [{when}] the sinusoid takes joint{bad + 1} within 2 deg of its limit. Lower "
            "--track-amplitude, or start from a pose further from the stops."
        )

    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
    worst_cond = 0.0
    for i in range(0, len(path), 10):
        data.qpos[:ARM_DOF] = path[i]
        mujoco.mj_forward(model, data)
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jp, jr, site)
        sv = np.linalg.svd(np.vstack([jp[:, :ARM_DOF], jr[:, :ARM_DOF]]))[1]
        worst_cond = max(worst_cond, sv[0] / sv[5])
    span = tool.max(axis=0) - tool.min(axis=0)
    speed = np.abs(np.diff(tool, axis=0)).max() / TRACK_DT
    print(f"  [{when}] tool path spans {np.round(span * 1e3, 0).tolist()} mm, peak speed {speed * 1e3:.0f} mm/s")
    print(f"  [{when}] worst cond(J) along the path: {worst_cond:.1f}")
    if worst_cond > 60:
        raise SystemExit(
            f"  [{when}] cond(J) reaches {worst_cond:.0f} on this path, where the task projector puts\n"
            "  nearly all of joint6 into the task subspace and it sustains a limit cycle. Pick a\n"
            "  different --start-pose or a smaller --track-amplitude."
        )


def stage_track(args: Args, cfg: dict, xml: str) -> None:
    """A3: follow a slow joint-space sinusoid and report the tool's position RMS error.

    Each joint gets its own phase, so the tool traces a path rather than a line. The plan is checked
    against the joint limits and against ``cond(J)`` along its whole length before anything is
    powered, and again after the goto, on the path actually driven.

    **The reported error is tracking error, not absolute accuracy**: commanded and measured tool
    positions are both computed through the same model from joint angles, so a kinematic error
    cancels.

    Nobody touches the arm.
    """
    import mujoco

    args = replace(args, mode="kinematic", start_pose=args.start_pose or NULLSPACE_POSE)
    q0 = np.radians(np.array([float(x) for x in args.start_pose.replace(" ", "").split(",") if x]))
    if q0.size != ARM_DOF:
        raise SystemExit(f"--start-pose needs {ARM_DOF} numbers in degrees, got {q0.size}")
    amp = TRACK_AMPLITUDE_DEG * args.track_amplitude

    print(
        f"\nplanning a {args.track_period:.0f} s sinusoid, {args.track_cycles:.0f} cycles "
        f"(model only, nothing is powered yet)"
    )
    print(f"  amplitude per joint (deg): {np.round(amp, 1).tolist()}")
    path, tool = _track_plan(xml, q0, amp, args.track_period, args.track_cycles, 1.0 / TRACK_DT)

    _validate_track_path(xml, path, tool, "planned")

    kp = _cap_task_stiffness(KP_TASK_JOINT * args.gain_scale, np.asarray(cfg["coulomb_friction"], dtype=float))
    kd = _cap_host_damping(
        KD_TASK_JOINT * np.sqrt(args.gain_scale), np.asarray(cfg["coulomb_friction"], dtype=float), args.loop_hz
    )
    # Both layers on the same 7-vector: this replays a planned joint path, whose error has
    # components in both subspaces, and matching them makes `n_task + n_null = I` a plain
    # joint-space PD.
    _check_modes(xml, q0, kp, kd, kp, kd, args.loop_hz, max(args.motor_kd, 0.0))

    _confirm(args, f"drive a {args.track_period:.0f} s sinusoid for {args.track_cycles:.0f} cycles, hands off")
    chain, arm = _make_arm(args, cfg, xml, kp_task=kp, kd_task=kd, kp_null=kp, kd_null=kd)
    trace = _Trace(arm, args.trace or f"/tmp/arm7_track_{time.strftime('%Y%m%d_%H%M%S')}.csv", hz=args.trace_hz)
    print(f"trace -> {trace.path}")
    try:
        _goto_start_pose(args, arm, xml)
        arm.enable_arm()
        # Replan about where the arm actually arrived, for the same reason --stage null does:
        # the move runs at a quarter stiffness and always stops a few degrees short, and a
        # trajectory that starts where the arm is not begins with a step.
        here = arm.get_joint_pos().copy()
        path, tool_cmd = _track_plan(xml, here, amp, args.track_period, args.track_cycles, 1.0 / TRACK_DT)
        print(f"\nreplanned from {np.round(np.degrees(here), 1).tolist()} deg")
        # Re-check, because this is the path that will actually be driven.
        _validate_track_path(xml, path, tool_cmd, "replanned")

        # Own model/data for the in-loop tool-error readout. Not shared with the validator: that
        # one is a gate that runs and returns, this one is stepped every tick.
        import mujoco

        model = mujoco.MjModel.from_xml_path(xml)
        data = mujoco.MjData(model)
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
        print(f"playing {len(path)} points over {len(path) * TRACK_DT:.0f} s. Hands off the arm.\n")

        errs, lags = [], []
        cycle_len = max(1, len(path) // max(int(args.track_cycles), 1))
        for i, q_cmd in enumerate(path):
            arm.command_joint_pos(q_cmd)
            # Label by cycle, so the trace can be split without inferring the timing afterwards.
            # Omitting this left a whole run tagged "start" and unsegmentable.
            trace.mark(f"cycle{i // cycle_len + 1}")
            if arm.fault is not None:
                print(f"\n\nfaulted during the sweep: {arm.fault}")
                break
            meas = arm.get_joint_pos()
            data.qpos[:ARM_DOF] = meas
            mujoco.mj_forward(model, data)
            err_mm = float(np.linalg.norm(data.site_xpos[site] - tool_cmd[i]) * 1e3)
            lag = float(np.degrees(np.abs(meas - q_cmd)).max())
            errs.append(err_mm)
            lags.append(lag)
            print(
                f"\r  {100 * i / len(path):3.0f}%   tool error {err_mm:6.2f} mm   lag {lag:4.1f} deg   ",
                end="",
                flush=True,
            )
            time.sleep(TRACK_DT)

        if errs:
            e = np.array(errs)
            # The first cycle carries the transient of starting from a standing stiction band, and
            # the metric is meant to describe steady tracking, so it is reported both ways.
            settled = e[len(e) // int(max(args.track_cycles, 1)) :] if args.track_cycles > 1 else e
            print(
                f"\n\n  tool position error: RMS {np.sqrt((e**2).mean()):.2f} mm   "
                f"p95 {np.percentile(e, 95):.2f}   max {e.max():.2f}"
            )
            print(f"  after the first cycle: RMS {np.sqrt((settled**2).mean()):.2f} mm")
            print(f"  worst joint lag {max(lags):.2f} deg")
            print(f"\n  A3 asks for 8 mm RMS: {'MET' if np.sqrt((settled**2).mean()) <= 8.0 else 'NOT MET'}")
            print("  This is tracking error -- both positions come from the same model, so a")
            print("  kinematic error cancels. A2/A6 want a dial indicator for the rest.")
    except KeyboardInterrupt:
        print()
    finally:
        _safe_teardown(arm, chain, trace)


def stage_watchdog(args: Args, cfg: dict, xml: str) -> None:
    """A4: stop feeding commands and confirm the arm stops safely and needs an explicit resume.

    Do this by hand before every long run. It is the cheapest check in the list and the one whose
    absence is most expensive.
    """
    _confirm(args, "enable OSC, then deliberately stop commanding to trip the timeout")
    chain, arm = _make_arm(args, cfg, xml)
    try:
        arm.enable_arm()
        target = arm.get_joint_pos().copy()
        for _ in range(150):
            arm.command_joint_pos(target)
            time.sleep(0.02)
        print("\nholding. Now stopping the command stream ...")
        t0 = time.monotonic()
        while arm.state is ArmState.ACTIVE and time.monotonic() - t0 < 5.0:
            time.sleep(0.005)
        took = time.monotonic() - t0
        print(f"  stopped after {took * 1e3:.0f} ms -> state {arm.state.value}")
        print(f"  A4 (<= T_wd + 20 ms): {'PASS' if took <= 0.32 else 'FAIL'}")
        print("  the arm should now be floating on gravity, not limp and not stiff.")
        input("  confirm it is holding itself up, then press Enter to test resume ...")
        arm.clear_fault()
        arm.resume()
        print(f"  after resume: {arm.state.value}")
    finally:
        arm.close()
        _shutdown_chain(chain)


def _pose_error(model: "object", data: "object", site_id: int, p0: np.ndarray, R0: np.ndarray) -> np.ndarray:
    """6-vector tool error at the current ``data.qpos``: position, then rotation as a rotvec."""
    import mujoco

    dp = data.site_xpos[site_id] - p0
    Rerr = data.site_xmat[site_id].reshape(3, 3) @ R0.T
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, Rerr.flatten())
    rv = np.zeros(3)
    mujoco.mju_quat2Vel(rv, q, 1.0)
    return np.concatenate([dp, rv])


def _self_motion_arc(
    xml: str,
    q0: np.ndarray,
    sign: int,
    arc_deg: float = NULLSPACE_ARC_DEG,
    step_deg: float = NULLSPACE_STEP_DEG,
) -> Tuple[np.ndarray, str]:
    """Joint path from ``q0`` along the kinematic nullspace, holding the tool pose fixed.

    ``n(q)`` is the last right-singular vector of ``J(q)`` and turns as the arm reconfigures, so
    this is a continuation rather than a straight line: each step takes ``step_deg`` along the
    current ``n``, then a damped least-squares correction pulls the pose back onto the starting
    tool pose.

    Returns the path (including ``q0``) and a one-line note on what ended it.
    """
    import mujoco

    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
    limits = model.jnt_range[:ARM_DOF].copy()

    data.qpos[:ARM_DOF] = q0
    mujoco.mj_forward(model, data)
    p0 = data.site_xpos[site_id].copy()
    R0 = data.site_xmat[site_id].copy().reshape(3, 3)

    def jac(q: np.ndarray) -> np.ndarray:
        data.qpos[:ARM_DOF] = q
        mujoco.mj_forward(model, data)
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jp, jr, site_id)
        return np.vstack([jp[:, :ARM_DOF], jr[:, :ARM_DOF]])

    step = np.radians(step_deg)
    steps = max(1, round(float(np.radians(arc_deg)) / float(step)))
    path = [q0.copy()]
    q = q0.copy()
    prev_n: Optional[np.ndarray] = None
    note = f"{arc_deg:.0f} deg of arc"
    for _ in range(steps):
        J = jac(q)
        n = np.linalg.svd(J)[2][-1]
        # The SVD fixes the direction only up to sign; flipping it mid-arc would double the path
        # back on itself, so each step is aligned with the one before.
        if prev_n is not None and n @ prev_n < 0:
            n = -n
        prev_n = n
        q = q + sign * step * n
        # Pull back onto the constraint. Damped so a step taken near the wrist singularity cannot
        # ask for an unbounded correction.
        for _ in range(4):
            J = jac(q)
            e = _pose_error(model, data, site_id, p0, R0)
            if np.linalg.norm(e[:3]) < 1e-6 and np.linalg.norm(e[3:]) < 1e-6:
                break
            q = q - J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(6), e)
        if np.any(q < limits[:, 0] + np.radians(2.0)) or np.any(q > limits[:, 1] - np.radians(2.0)):
            bad = int(np.argmax(np.maximum(limits[:, 0] - q, q - limits[:, 1])))
            note = f"stopped 2 deg short of joint{bad + 1}'s limit after {len(path) * step_deg:.0f} deg"
            break
        if np.linalg.svd(jac(q))[1][-1] < 0.05:
            note = f"stopped at a singularity after {len(path) * step_deg:.0f} deg"
            break
        path.append(q.copy())
    return np.array(path), note


def _arc_report(xml: str, path: np.ndarray, label: str) -> float:
    """Print what the planned arc does, and return its worst tool drift in the model, in mm."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")

    def tool(q: np.ndarray) -> np.ndarray:
        data.qpos[:ARM_DOF] = q
        mujoco.mj_forward(model, data)
        return data.site_xpos[site_id].copy()

    p0 = tool(path[0])
    drift = max(float(np.linalg.norm(tool(q) - p0)) for q in path) * 1e3
    swing = np.degrees(path[-1] - path[0])
    print(f"  {label}: " + " ".join(f"j{i + 1}{swing[i]:+.0f}" for i in range(ARM_DOF)) + " deg")
    print(f"    {len(path)} points, planned tool drift {drift:.4f} mm")
    return drift


def _arc_gains(
    xml: str, q0: np.ndarray, cfg: dict, deadband_deg: float, loop_hz: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-joint PD gains for playing the arc: stiffness from stiction, damping from inertia.

    ``kp = friction / deadband`` is the position lead at which each joint breaks away. The arc is
    slow enough that inertia and Coriolis are negligible, so tracking error is stiction divided by
    stiffness.

    ``kd = 2 sqrt(kp J)``, critical, then capped at ``J / dt`` -- half the discrete bound
    ``kd dt / J < 2``, the other half kept back for the feedback delay. The printed table says
    which joints the cap bound on; a capped joint is underdamped rather than unstable.

    A single multiplier on ``KP_TASK_JOINT`` cannot serve this arm: its stiction spans a factor of
    48, so one scale leaves the stiffest joint inside its own deadband or the softest one hunting.
    """
    import mujoco

    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    data.qpos[:ARM_DOF] = q0
    mujoco.mj_forward(model, data)
    full = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, full, data.qM)
    inertia = np.diag(full)[:ARM_DOF]

    friction = np.asarray(cfg.get("coulomb_friction", np.zeros(ARM_DOF)), dtype=float)
    kp = friction / np.radians(deadband_deg)
    # A floor, for a joint whose friction was measured as near zero, and a ceiling well under
    # anything the arm has been driven at.
    kp = np.clip(kp, KP_TASK_JOINT * 0.25, KP_TASK_JOINT * 6.0)
    kd_crit = 2.0 * np.sqrt(kp * inertia)
    kd_cap = inertia * loop_hz
    kd = np.minimum(kd_crit, kd_cap)

    print(f"\n  arc gains, from {deadband_deg:.2f} deg of allowed stiction deadband at {loop_hz:.0f} Hz:")
    print(f"    {'':<6}{'friction':>9}{'inertia':>9}{'kp':>8}{'kd':>7}{'kd crit':>9}{'zeta':>7}{'deadband':>10}")
    for i in range(ARM_DOF):
        zeta = kd[i] / kd_crit[i] if kd_crit[i] > 0 else 1.0
        flag = "  capped" if kd[i] < kd_crit[i] - 1e-9 else ""
        print(
            f"    j{i + 1:<5}{friction[i]:>9.2f}{inertia[i]:>9.4f}{kp[i]:>8.1f}{kd[i]:>7.2f}"
            f"{kd_crit[i]:>9.2f}{zeta:>7.2f}{np.degrees(friction[i] / kp[i]):>9.2f}deg{flag}"
        )
    return kp, kd


def stage_null(args: Args, cfg: dict, xml: str) -> None:
    """Drive the arm along its own nullspace under position control and measure the tool motion.

    The path is planned to hold the tool fixed while every joint moves, so the deliverable is one
    number: the tool displacement computed from the encoders along that path. Link lengths, joint
    offsets and encoder zeros all have to agree for it to stay small.

    Nobody touches the arm during this stage.
    """
    here_pose = args.start_pose or NULLSPACE_POSE
    args = replace(args, mode="kinematic")  # the arc is played as joint positions
    args = replace(args, start_pose=here_pose)
    q0 = np.radians(np.array([float(x) for x in here_pose.replace(" ", "").split(",") if x]))
    if q0.size != ARM_DOF:
        raise SystemExit(f"--start-pose needs {ARM_DOF} numbers in degrees, got {q0.size}")

    print("\nplanning the self-motion arc (model only, nothing is powered yet)")
    fwd, fwd_note = _self_motion_arc(xml, q0, +1, args.arc_deg)
    back, back_note = _self_motion_arc(xml, q0, -1, args.arc_deg)
    d1 = _arc_report(xml, fwd, "forward ")
    print(f"    {fwd_note}")
    d2 = _arc_report(xml, back, "backward")
    print(f"    {back_note}")
    if max(d1, d2) > NULLSPACE_TOOL_TOL_MM:
        raise SystemExit(
            f"the planned arc drifts {max(d1, d2):.3f} mm in the model, over the {NULLSPACE_TOOL_TOL_MM} mm "
            "tolerance -- the continuation is not converging, so playing it would measure the planner"
        )
    if min(len(fwd), len(back)) < 20:
        raise SystemExit(
            f"only {min(len(fwd), len(back))} points of arc are available from this pose "
            f"({fwd_note}; {back_note}) -- pick a --start-pose further from the limits"
        )

    print("\n  the arc above is a feasibility check from --start-pose; the one that gets played is")
    print("  replanned from wherever the arm actually arrives, which is not the same pose.")

    kp_arc, kd_arc = _arc_gains(xml, q0, cfg, args.arc_deadband, args.loop_hz)

    _confirm(args, f"drive the arm along {args.arc_deg:.0f} deg of nullspace arc, both ways, hands off")
    chain, arm = _make_arm(
        args,
        cfg,
        xml,
        kp_task=kp_arc,
        kd_task=kd_arc,
        kp_cartesian=np.zeros(6),
        kd_cartesian=np.zeros(6),
        # Both layers on the same 7-vector. The arc lies entirely in the nullspace by
        # construction, so `n_task @ err` is zero and `kp_task` alone drives nothing -- and with
        # `n_task + n_null = I` the two projections collapse to a plain joint-space PD, which is
        # what replaying a planned path wants.
        kp_null=kp_arc,
        kd_null=kd_arc,
    )
    trace = _Trace(arm, args.trace or f"/tmp/arm7_null_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    print(f"trace -> {trace.path}")
    try:
        _goto_start_pose(args, arm, xml)
        arm.enable_arm()

        # Replan from where the arm *is*, not from where it was sent. The move runs at a quarter
        # stiffness because joint6 oscillates at the shipped gains, and a quarter of 40 N.m/rad
        # leaves joint1 a 10.9 deg stiction deadband -- so it arrives some degrees short, every
        # time, and that is not a fault to be fixed. It is fatal only if the arc is
        # then played from the requested pose, which leaves the first commanded point
        # 170 mm away in task space, the four heavy joints locked in stiction for the whole run,
        # and the three light ones tracking a path that is not a self-motion of the pose the arm
        # was actually in. The tool left by 60 mm and the abort stopped it at 20 percent.
        #
        # An arc is a self-motion of the configuration it was planned at. Planning here costs
        # about a second of pure model work and makes the first commanded point equal the measured
        # one, so nothing has to be dragged anywhere before the measurement starts.
        q_here = arm.get_joint_pos().copy()
        print(f"\nreplanning the arc from the pose actually reached: {np.round(np.degrees(q_here), 1).tolist()}")
        fwd, fwd_note = _self_motion_arc(xml, q_here, +1, args.arc_deg)
        back, back_note = _self_motion_arc(xml, q_here, -1, args.arc_deg)
        d1 = _arc_report(xml, fwd, "forward ")
        print(f"    {fwd_note}")
        d2 = _arc_report(xml, back, "backward")
        print(f"    {back_note}")
        if max(d1, d2) > NULLSPACE_TOOL_TOL_MM or min(len(fwd), len(back)) < 20:
            raise SystemExit(
                f"the arc replanned from the reached pose is not usable "
                f"(drift {max(d1, d2):.3f} mm, {min(len(fwd), len(back))} points) -- "
                "the arm did not arrive near enough to --start-pose; raise --max-torque or --start-seconds"
            )
        # Inertia is pose-dependent, so the damping cap is too; recompute both at the pose the
        # arm is actually in rather than at the one it was sent to.
        arm.kp_task, arm.kd_task = _arc_gains(xml, q_here, cfg, args.arc_deadband, args.loop_hz)
        arm.kp_null, arm.kd_null = arm.kp_task, arm.kd_task

        # One continuous path: start -> forward end -> back through start -> backward end -> start.
        plan = np.vstack([fwd, fwd[::-1][1:], back[1:], back[::-1][1:]])
        plan_path = trace.path.replace(".csv", "_plan.csv")
        np.savetxt(plan_path, plan, delimiter=",", header=",".join(f"q{i + 1}" for i in range(ARM_DOF)))
        print(f"  commanded path -> {plan_path}")

        p0 = arm.get_ee_pose()[:3, 3].copy()
        dt = 1.0 / ARC_COMMAND_HZ
        print(f"\nplaying {len(plan)} points over {args.arc_seconds:.0f} s. Hands off the arm.")
        print("Watch the gripper against a fixed mark -- how far it physically wanders is the")
        print("kinematic answer, and it is the one thing the encoders cannot give. Ctrl-C stops it.\n")
        worst = worst_lag = 0.0
        # Interpolated, not stepped. Commanding one plan point per `arc_seconds/len(plan)` holds
        # each for ~150 ms and then jumps, which is a 6.6 Hz staircase -- it puts the whole
        # velocity spectrum on the waypoint rate, shows up as the tool buzzing, and inflates the
        # tracking number this stage exists to measure. The plan is a path, so play it as one.
        plan_arr = np.asarray(plan, dtype=float)
        t_start = time.monotonic()
        next_tick = t_start
        i = 0
        while True:
            elapsed = time.monotonic() - t_start
            if elapsed >= args.arc_seconds:
                break
            s = elapsed / args.arc_seconds * (len(plan_arr) - 1)
            i = min(int(s), len(plan_arr) - 2)
            frac = s - i
            q = plan_arr[i] * (1.0 - frac) + plan_arr[i + 1] * frac
            arm.command_joint_pos(q)
            # The plan index goes in the label so the commanded pose is recoverable offline from
            # the sidecar; the trace's own columns are measured state only.
            trace.mark(f"arc:{i}")
            if arm.fault is not None:
                print(f"\n\nfaulted during the arc: {arm.fault}")
                break
            meas = arm.get_joint_pos()
            lag = float(np.degrees(np.abs(meas - q)).max())
            dev = float(np.linalg.norm(arm.get_ee_pose()[:3, 3] - p0) * 1e3)
            worst, worst_lag = max(worst, dev), max(worst_lag, lag)
            if lag > NULLSPACE_ABORT_DEG:
                print(
                    f"\n\njoint tracking is {lag:.1f} deg behind, past the {NULLSPACE_ABORT_DEG:.0f} deg abort"
                    " -- the arm is not following the path, so nothing downstream of here means anything"
                )
                break
            if dev > NULLSPACE_ABORT_MM:
                print(f"\n\ntool moved {dev:.0f} mm, past the {NULLSPACE_ABORT_MM:.0f} mm abort -- stopping")
                break
            print(
                f"\r  {100 * elapsed / args.arc_seconds:3.0f}%   tool {dev:6.1f} mm "
                f"(worst {worst:6.1f})   lag {lag:4.1f} deg   j3 {np.degrees(meas[2]):+6.1f} deg   ",
                end="",
                flush=True,
            )
            next_tick += dt
            sleep = next_tick - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_tick = time.monotonic()
        print(f"\n\nworst joint tracking lag: {worst_lag:.1f} deg")
        print(f"worst tool displacement from the encoders: {worst:.1f} mm")
        print("  This second number is *tracking* error carried into task space, not a kinematic")
        print("  check -- it is computed through the same model it would be testing. It is still")
        print("  the quantity that bounds what OSC can hold, and on this arm it is set by stiction.")
        print("  The kinematic check is what you saw: how far the gripper physically moved.")
    except KeyboardInterrupt:
        print()
    finally:
        _safe_teardown(arm, chain, trace)


STAGES = {
    "check": stage_check,
    "pose": stage_pose,
    "mirror": stage_mirror,
    "probe": stage_probe,
    "survey": stage_survey,
    "jitter": stage_jitter,
    "float": stage_float,
    "sweep": stage_sweep,
    "null": stage_null,
    "task": stage_task,
    "track": stage_track,
    "watchdog": stage_watchdog,
}


def main() -> None:
    args = tyro.cli(Args)
    if args.stage not in STAGES:
        raise SystemExit(f"--stage must be one of {', '.join(STAGES)}")
    xml, cfg_path = _paths(args.arm)
    if not Path(xml).is_file():
        raise SystemExit(f"model not found at {xml}")
    cfg = _load_config(cfg_path)
    print(f"arm={args.arm}  channel={args.channel}  mode={args.mode}  torque clamp {args.max_torque:.1f} N.m")
    print(f"config {cfg_path}")
    STAGES[args.stage](args, cfg, xml)


if __name__ == "__main__":
    main()
