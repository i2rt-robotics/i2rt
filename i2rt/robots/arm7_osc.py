"""Operational-space control for a 7-DOF arm, with nullspace regularization.

Two modes share one skeleton:

``kinematic``
    ``Mx = I`` and the orthogonal projector ``N_kin = I - J^+ J``. The task layer is a joint-space
    PD confined to the task subspace by ``(I - N_kin)``. Never forms the mass matrix.

``full``
    ``Mx = (J M^-1 J^T)^-1`` and the dynamically consistent projector ``I - J^T Jbar^T``. The task
    layer is a Cartesian impedance. See :meth:`Arm7OSC._torque_full`.

Gravity compensation is added in both modes and is not ramped.

The class owns a control thread::

    arm = Arm7OSC(motor_chain=chain, xml_path=..., ee_site="grasp_site")
    arm.enable_arm()
    arm.command_joint_pos(q_cmd)  # upstream, at whatever rate it likes
    obs = arm.get_observations()
    arm.close()

Hardware is reached only through the ``MotorChain`` protocol, so the same code drives a CAN chain,
the EtherCAT-to-CAN bridge and :class:`~i2rt.robots.sim_motor_chain.SimMotorChain`.
"""

from __future__ import annotations

import enum
import functools
import gc
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mujoco
import numpy as np

from i2rt.motor_drivers.dm_driver import MotorChain
from i2rt.motor_drivers.utils import MotorType
from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml
from i2rt.utils.mujoco_utils import MuJoCoKDL

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=None)
def arm7_xml(side: str = "right", gripper: GripperType = GripperType.LINEAR_4310) -> str:
    """Path to the composed arm-plus-gripper MJCF for the 7-DOF arm."""
    arm = ArmType.YAM_7DOF_RIGHT if side == "right" else ArmType.YAM_7DOF_LEFT
    return combine_arm_and_gripper_xml(arm, gripper)


def _null_gain(value: object, name: str) -> np.ndarray | float:
    """A nullspace gain: a float, or a 7-vector for trajectory playback. Anything else is a bug."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    if arr.shape != (ARM_DOF,):
        raise ValueError(f"{name} must be a scalar or {ARM_DOF} values, got shape {arr.shape}")
    return arr


# --- Tunables. Centralized here on purpose: safety limits scattered through a torque controller
# --- are how a limit gets missed. Every value carries the reason it has the value it does.

ARM_DOF = 7

CONTROL_HZ = 500.0
"""Control-loop rate, Hz."""

# Task layer, `kinematic` mode: joint-space stiffness and damping, Nm/rad and Nm.s/rad, applied
# through the task projector. Tapered toward the wrist, which carries two orders of magnitude less
# reflected inertia than the shoulder. Scale the whole profile with one factor rather than
# reshaping it; `arm7_bringup.py --gain-scale` does that.
KP_TASK_JOINT = np.array([40.0, 40.0, 40.0, 30.0, 15.0, 15.0, 15.0])
# Joints 3, 5, 6 and 7 carry kd = 2*sqrt(Kp*J), critical damping for their minimum reflected
# inertia over the workspace. Joints 1, 2 and 4 carry measured values instead.
KD_TASK_JOINT = np.array([4.0, 4.0, 0.70, 3.0, 0.30, 0.72, 0.73])

# Nullspace layer, applied through the nullspace projector. Scalar, so the stiffness is isotropic.
# KP_NULL = 0 leaves damping only.
KP_NULL = 4.0
KD_NULL = 0.25

# Cartesian gains for `full` mode, over [x, y, z, rx, ry, rz]. Acceleration gains: Mx carries the
# mass, so Kp6 has units of omega^2 and the task bandwidth is sqrt(Kp6) rad/s. Kd6 here is
# 2*sqrt(Kp6), critical damping; `arm7_bringup.py --cart-zeta` scales it.
KP_TASK_CARTESIAN = np.array([40.0, 40.0, 40.0, 40.0, 40.0, 40.0])
KD_TASK_CARTESIAN = np.array([12.6, 12.6, 12.6, 12.6, 12.6, 12.6])

WRIST_INERTIA_AUGMENTATION = 0.02
"""Added to the last three diagonal entries of ``M`` in ``full`` mode, kg.m^2. Set per arm."""

SINGULARITY_RCOND = 1e-2
"""Cutoff for the damped pseudoinverse of ``J``."""

MX_DET_FLOOR = 1e-8
"""``det(J M^-1 J^T)`` below which ``full`` mode uses the damped pseudoinverse of ``Mx``."""

TORQUE_RAMP_S = 1.5
"""Seconds over which control torque ramps from zero on enable. Gravity is not ramped."""

MAX_TORQUE_RATE = 200.0
"""Nm/s. Slew bound applied to the commanded torque in every state."""

MAX_JOINT_VEL = 4.0
"""rad/s. Any joint above this puts the arm in ``DAMPED``."""

MAX_TORQUE_BIAS_NM = 3.0
"""N.m. Ceiling on :meth:`Arm7OSC.set_torque_bias`."""

COMMAND_TIMEOUT_S = 0.3
"""Seconds. ``joint_cmd`` older than this puts the arm in ``DAMPED``."""

FEEDBACK_TIMEOUT_CYCLES = 6.0
"""Missing-frame timeout, in per-motor feedback cycles. The controller measures each motor's own
refresh interval and multiplies by this, so the criterion does not depend on the polling
arrangement."""

NEVER_REFRESHED_S = 0.5
"""Seconds. A motor that has never refreshed counts as absent after this. Used in place of the
interval-based check, which has no measured interval to scale against."""

FEEDBACK_TIMEOUT_FLOOR_S = 0.02
"""Seconds. Lower bound on the missing-frame timeout, whatever the measured rate implies."""

VELOCITY_CUTOFF_HZ = 40.0
"""Low-pass cutoff for the velocity observer, Hz."""

MAX_FEEDBACK_AGE_S = 0.05
"""Seconds. Readings older than this are not extrapolated to the common instant."""

TORQUE_LIMIT_NM = np.array([45.0, 45.0, 20.0, 20.0, 10.0, 8.0, 8.0])
"""Per-joint torque clamp, N.m. About 3x this arm's peak gravity torque, not the motors' ratings."""

MOTOR_KD_FLOAT = np.array([0.1, 0.1, 0.1, 0.3, 0.05, 0.05, 0.05])
"""Per-joint MIT-mode kd sent to the motors while floating."""

MOTOR_KD_ACTIVE = np.array([0.05, 0.05, 0.05, 0.1, 0.02, 0.02, 0.02])
"""Per-joint MIT-mode kd sent to the motors while OSC is driving."""

DEFAULT_MOTOR_LIST: Tuple[Tuple[int, str], ...] = (
    (0x01, MotorType.DM6248),
    (0x02, MotorType.DM6248),
    (0x03, MotorType.DM4340),
    (0x04, MotorType.DM4340),
    (0x05, MotorType.DM4340),
    (0x06, MotorType.DM4310),
    (0x07, MotorType.DM4310),
)
"""``(can_id, motor_type)`` per joint, used when the caller supplies no list. Read from the right
arm's Gr registers; see ``config/yam_7dof_right_v1.yml``. The driver encodes every MIT command with
the declared type's constants, so re-read this after any motor is swapped."""


class ArmState(enum.Enum):
    """What the control loop is doing."""

    DISABLED = "disabled"
    """Zero torque, no gravity compensation. Entered only through :meth:`go_limp`."""
    FLOATING = "floating"
    """Gravity compensation plus motor-side damping. Power-on state."""
    ACTIVE = "active"
    """Running the OSC law against ``joint_cmd``."""
    DAMPED = "damped"
    """Gravity compensation only. Entered on a fault or a command timeout; left through
    :meth:`resume`, not by a new command arriving."""


class ControlMode(enum.Enum):
    KINEMATIC = "kinematic"
    FULL = "full"


def rotation_error(r_des: np.ndarray, r_cur: np.ndarray) -> np.ndarray:
    """Orientation error ``R_des @ R_cur.T`` as a rotation vector (axis * angle, radians).

    Written out rather than taken from scipy for two reasons: the control loop must eventually be
    allocation-free, and the +/-pi branch needs to be handled deliberately. At exactly pi the axis
    is ambiguous in sign but the rotation is not, and the naive ``sin`` formula loses all precision
    there because the off-diagonal terms it reads go to zero. Near pi this reads the axis off
    ``R + I`` instead, whose columns are all parallel to the axis.
    """
    r = r_des @ r_cur.T
    cos_theta = np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))

    if theta < 1e-8:
        # Small angle: the skew part is the rotation vector to first order.
        return np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) * 0.5

    if theta < np.pi - 1e-4:
        axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
        return axis * (theta / (2.0 * np.sin(theta)))

    # Near pi: take the axis from the largest column of R + I, which is parallel to it.
    rp = r + np.eye(3)
    axis = rp[:, int(np.argmax(np.linalg.norm(rp, axis=0)))]
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.zeros(3)
    axis = axis / norm
    # R + I fixes the axis line but not its sign; pick the sign from the skew part when it is
    # readable, so the result stays continuous as theta sweeps through pi.
    skew = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    if np.dot(skew, axis) < 0:
        axis = -axis
    return axis * theta


class VelocityObserver:
    """First-order low-pass on the velocity, plus per-joint extrapolation to a common instant.

    Round-robin polling gives each joint in a feedback frame a different timestamp; ``update``
    extrapolates each one forward to the newest, using the filtered velocity.
    """

    def __init__(self, n_joints: int, cutoff_hz: float, control_dt: float, align: bool = True) -> None:
        self.n = n_joints
        self.control_dt = control_dt
        self.align = align
        self.set_cutoff(cutoff_hz)
        self._dq = np.zeros(n_joints)
        self._started = False

    def set_cutoff(self, cutoff_hz: float) -> None:
        """Set the low-pass cutoff. ``0`` or ``None`` passes the raw signal through."""
        self.cutoff_hz = float(cutoff_hz) if cutoff_hz else 0.0
        if self.cutoff_hz <= 0.0:
            self._alpha = 1.0
            return
        tau = 1.0 / (2.0 * np.pi * self.cutoff_hz)
        self._alpha = self.control_dt / (tau + self.control_dt)

    def reset(self) -> None:
        self._dq[:] = 0.0
        self._started = False

    def update(
        self,
        pos: np.ndarray,
        vel_raw: np.ndarray,
        stamps: np.ndarray,
        now: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(q_aligned, dq_filtered)`` for one feedback frame.

        Args:
            pos: per-joint position as reported.
            vel_raw: per-joint velocity as reported (already a difference, on hardware).
            stamps: per-joint time of that reading. Under round-robin polling these differ.
            now: the instant to align to, on the same clock as ``stamps``.
        """
        if not self._started:
            self._dq[:] = vel_raw
            self._started = True
        else:
            self._dq += self._alpha * (vel_raw - self._dq)

        if not self.align:
            return np.asarray(pos, dtype=float).copy(), self._dq.copy()

        # Extrapolate each joint forward over its own age. A stale reading is corrected by how
        # stale it actually is, which is why the timestamps have to be per motor and honest.
        age = np.clip(now - np.asarray(stamps, dtype=float), 0.0, MAX_FEEDBACK_AGE_S)
        return np.asarray(pos, dtype=float) + self._dq * age, self._dq.copy()


class LoopTimer:
    """Loop-period statistics on a monotonic clock, including tail latency.

    ``RateRecorder`` reports a mean, which cannot answer the question that matters for a control
    loop: a loop averaging 500 Hz while stalling 12 ms once a minute is not a 500 Hz loop. Keeps a
    bounded ring of periods so nothing is allocated as it runs.
    """

    def __init__(self, capacity: int = 200_000) -> None:
        self._periods = np.zeros(capacity)
        self._n = 0
        self._wrapped = False
        self._last: Optional[float] = None

    def tick(self) -> None:
        now = time.monotonic()
        if self._last is not None:
            if self._n == self._periods.size:
                self._n = 0
                self._wrapped = True
            self._periods[self._n] = now - self._last
            self._n += 1
        self._last = now

    def reset(self) -> None:
        self._n = 0
        self._wrapped = False
        self._last = None

    def stats(self) -> Dict[str, float]:
        n = self._periods.size if self._wrapped else self._n
        if n == 0:
            return {}
        p = self._periods[:n] * 1e3  # ms
        return {
            "count": float(n),
            "mean_hz": 1e3 / float(p.mean()),
            "p50_ms": float(np.percentile(p, 50)),
            "p99_ms": float(np.percentile(p, 99)),
            "max_ms": float(p.max()),
        }


class Arm7OSC:
    """Operational-space controller for a 7-DOF arm, running its own control thread.

    Args:
        motor_chain: the arm's motors. Injected, as ``MotorChainRobot`` does, so that a real CAN
            chain and a simulated one are interchangeable. May be longer than ``ARM_DOF`` -- a
            trailing gripper motor is commanded to zero torque and otherwise ignored.
        xml_path: MJCF used for gravity, Jacobians and forward kinematics. This is the
            controller's *model*, deliberately a separate object from any model the plant uses, so
            that model error is representable.
        ee_site: site to control. Follows the repo's convention of ``grasp_site`` for the tool
            centre and ``tcp_site`` for the flange.
        mode: see the module docstring.
        control_hz: control-loop rate.
        joint_limits: ``(ARM_DOF, 2)`` of lower/upper bounds. Defaults to the model's own.
        motor_list: ``(can_id, motor_type)`` per arm joint, used for the default torque limits.
        torque_limit: per-joint torque clamp, N.m. Defaults to each motor's rating.
        gravity_comp_factor: per-joint multiplier on the gravity term, the same calibration knob
            the YAM arms expose for unmodelled load.
        kp_task, kd_task: task-layer joint-space gains, defaulting to ``KP_TASK_JOINT`` /
            ``KD_TASK_JOINT``. Per joint, and only meaningful in ``kinematic`` mode.
        kp_null, kd_null: nullspace gains, defaulting to ``KP_NULL`` / ``KD_NULL``. Scalar, or a
            7-vector; setting them equal to ``kp_task``/``kd_task`` collapses the two projections
            into a plain joint-space PD.
        kp_cartesian, kd_cartesian: task gains for ``full`` mode, 6-vectors over
            ``[x, y, z, rx, ry, rz]``. Acceleration gains, not stiffnesses -- see
            ``KP_TASK_CARTESIAN``. Ignored in ``kinematic`` mode.
        wrist_inertia_augmentation: added to the last three diagonal entries of ``M`` in ``full``
            mode. See ``WRIST_INERTIA_AUGMENTATION``.
        velocity_cutoff_hz: velocity observer cutoff; ``0`` disables filtering. Couples to the
            usable task gain -- see ``VELOCITY_CUTOFF_HZ``.
        align_feedback: extrapolate each joint from its own timestamp to a common instant.
        freeze_gc: call ``gc.freeze()`` when the control thread starts. Process-wide.
        motor_kd_active, motor_kd_float: per-joint MIT-mode kd sent to the motors while OSC is
            driving and while floating.
        max_torque_rate: torque slew bound, Nm/s, applied in every state.
        auto_start: start the control thread immediately, in ``FLOATING``.
    """

    def __init__(
        self,
        motor_chain: MotorChain,
        xml_path: str,
        ee_site: str = "grasp_site",
        mode: str | ControlMode = ControlMode.KINEMATIC,
        control_hz: float = CONTROL_HZ,
        joint_limits: Optional[np.ndarray] = None,
        motor_list: Optional[Sequence[Tuple[int, str]]] = None,
        torque_limit: Optional[np.ndarray] = None,
        gravity_comp_factor: Optional[np.ndarray] = None,
        coulomb_friction: Optional[np.ndarray] = None,
        kp_task: Optional[np.ndarray] = None,
        kd_task: Optional[np.ndarray] = None,
        kp_null: Optional[float] = None,
        kd_null: Optional[float] = None,
        kp_cartesian: Optional[np.ndarray] = None,
        kd_cartesian: Optional[np.ndarray] = None,
        wrist_inertia_augmentation: Optional[float] = None,
        velocity_cutoff_hz: Optional[float] = None,
        align_feedback: bool = True,
        freeze_gc: bool = True,
        max_torque_rate: float = MAX_TORQUE_RATE,
        motor_kd_active: Optional[np.ndarray] = None,
        motor_kd_float: Optional[np.ndarray] = None,
        auto_start: bool = True,
    ) -> None:
        self.motor_chain = motor_chain
        self.xml_path = xml_path
        self.mode = ControlMode(mode) if not isinstance(mode, ControlMode) else mode
        self._control_dt = 1.0 / control_hz

        self._model = mujoco.MjModel.from_xml_path(xml_path)
        self._data = mujoco.MjData(self._model)
        self._cmd_data = mujoco.MjData(self._model)  # separate scratch for FK(joint_cmd)
        if self._model.nv < ARM_DOF:
            raise ValueError(f"{xml_path} has nv={self._model.nv}, need at least {ARM_DOF}")

        self._site_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, ee_site)
        if self._site_id == -1:
            raise ValueError(f"site {ee_site!r} not found in {xml_path}")
        self.ee_site = ee_site

        # Gravity gets its own MuJoCoKDL: that class keeps one MjData internally and is therefore
        # not safe to share with anything else on another thread.
        self._kdl = MuJoCoKDL(xml_path)
        # `mj_inverse` returns `qfrc_inverse`, which carries the joint-limit constraint force as
        # well as gravity. Limits are disabled on this model so that `_gravity` returns gravity
        # alone, including for a pose outside the modelled range.
        self._kdl.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_LIMIT
        # Stored for callers to read. No friction feedforward is applied.
        self._coulomb_friction = (
            np.zeros(ARM_DOF) if coulomb_friction is None else np.asarray(coulomb_friction, dtype=float)
        )
        self._gravity_comp_factor = (
            np.ones(ARM_DOF) if gravity_comp_factor is None else np.asarray(gravity_comp_factor, dtype=float)
        )

        self._joint_limits = (
            self._model.jnt_range[:ARM_DOF].copy() if joint_limits is None else np.asarray(joint_limits, dtype=float)
        )

        # Gains live on the instance so a tuning sweep can vary them without mutating module
        # state; the module constants remain the shipped defaults.
        self.kp_task = np.array(KP_TASK_JOINT if kp_task is None else kp_task, dtype=float)
        self.kd_task = np.array(KD_TASK_JOINT if kd_task is None else kd_task, dtype=float)
        # Scalar or 7-vector; both multiply `err` elementwise below. See the argument docs.
        self.kp_null = _null_gain(KP_NULL if kp_null is None else kp_null, "kp_null")
        self.kd_null = _null_gain(KD_NULL if kd_null is None else kd_null, "kd_null")
        self.kp_cartesian = np.array(KP_TASK_CARTESIAN if kp_cartesian is None else kp_cartesian, dtype=float)
        self.kd_cartesian = np.array(KD_TASK_CARTESIAN if kd_cartesian is None else kd_cartesian, dtype=float)
        self.wrist_inertia_augmentation = float(
            WRIST_INERTIA_AUGMENTATION if wrist_inertia_augmentation is None else wrist_inertia_augmentation
        )
        self.max_torque_rate = float(max_torque_rate)
        self.freeze_gc = bool(freeze_gc)
        self.motor_kd_active = np.array(MOTOR_KD_ACTIVE if motor_kd_active is None else motor_kd_active, dtype=float)
        self.motor_kd_float = np.array(MOTOR_KD_FLOAT if motor_kd_float is None else motor_kd_float, dtype=float)

        self._motor_list = list(motor_list) if motor_list is not None else list(DEFAULT_MOTOR_LIST)
        constants = [MotorType.get_motor_constants(t) for _, t in self._motor_list[:ARM_DOF]]
        rated = np.array([c.TORQUE_MAX for c in constants])
        if torque_limit is None:
            self._torque_limit = rated
            logger.info(
                "arm7_osc: torque clamped at the motors' ratings %s N.m. That is what the parts can "
                "deliver, not what this arm needs -- pass torque_limit to bound it to the application.",
                np.round(rated, 1).tolist(),
            )
        else:
            self._torque_limit = np.minimum(np.asarray(torque_limit, dtype=float), rated)

        self._n_motors = len(motor_chain)
        if self._n_motors < ARM_DOF:
            raise ValueError(f"motor chain has {self._n_motors} motors, need at least {ARM_DOF}")

        # Shared state. _lock covers the command; _state_lock covers the observation.
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._joint_cmd: Optional[np.ndarray] = None
        self._cmd_time = 0.0
        # Power-on state is gravity-comp float, not limp: see ArmState.DISABLED.
        self._state = ArmState.FLOATING
        self._enable_time = time.monotonic()
        self._fault: Optional[str] = None

        self._q = np.zeros(ARM_DOF)
        self._dq = np.zeros(ARM_DOF)
        self._tau = np.zeros(ARM_DOF)
        self._torque_bias = np.zeros(ARM_DOF)
        self._prev_tau = np.zeros(ARM_DOF)
        self._feedback_age = np.zeros(ARM_DOF)
        self._ee_pos_err = np.zeros(3)
        self._ee_rot_err = np.zeros(3)

        # Preallocated loop scratch. Full allocation-free operation is a later step; these are the
        # ones that would otherwise allocate on every single tick.
        self._jacp = np.zeros((3, self._model.nv))
        self._jacr = np.zeros((3, self._model.nv))
        self._jac = np.zeros((6, self._model.nv))
        # Separate scratch for the public introspection helpers. Those are called from other
        # threads -- a viewer, a tuning script, a test -- and sharing the loop's buffers with them
        # means a caller can silently corrupt the Jacobian the control law is midway through using.
        self._probe_data = mujoco.MjData(self._model)
        self._probe_jacp = np.zeros((3, self._model.nv))
        self._probe_jacr = np.zeros((3, self._model.nv))
        self._probe_lock = threading.Lock()
        self._mass_full = np.zeros((self._model.nv, self._model.nv))
        self._mass_pipeline = self._probe_mass_pipeline()
        self._raw_pos = np.zeros(ARM_DOF)
        self._raw_vel = np.zeros(ARM_DOF)
        self._raw_time = np.zeros(ARM_DOF)
        # Per-motor refresh interval, measured rather than assumed: the missing-frame criterion has
        # to be keyed to the rate each motor is actually heard at, which polling sets, not to the
        # loop rate. Seeded on the first few frames and then tracked slowly.
        self._refresh_interval = np.zeros(ARM_DOF)
        self._last_refresh_time = np.zeros(ARM_DOF)
        self._feedback_seen = False
        self.observer = VelocityObserver(
            ARM_DOF,
            VELOCITY_CUTOFF_HZ if velocity_cutoff_hz is None else velocity_cutoff_hz,
            self._control_dt,
            align=align_feedback,
        )
        self._cmd_vec = np.zeros(self._n_motors)
        self._zeros = np.zeros(self._n_motors)
        self._kd_vec = np.zeros(self._n_motors)

        self.timer = LoopTimer()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        if auto_start:
            self.start()

    # ---- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Start the control thread. It begins in ``FLOATING``: gravity compensation only."""
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._control_loop, name="arm7_osc", daemon=True)
        self._thread.start()

    def enable_arm(self) -> None:
        """Enable OSC, holding the current position. Torque ramps in over ``TORQUE_RAMP_S``."""
        with self._lock:
            if self._fault is not None:
                raise RuntimeError(f"cannot enable while faulted: {self._fault}. Call clear_fault() first.")
            with self._state_lock:
                q_now = self._q.copy()
            self._joint_cmd = q_now
            self._cmd_time = time.monotonic()
            self._enable_time = time.monotonic()
            self._state = ArmState.ACTIVE
        logger.info("arm7_osc: ACTIVE, holding q=%s", np.round(q_now, 4))

    def float_arm(self) -> None:
        """Gravity compensation and motor damping only, so the arm can be moved by hand."""
        with self._lock:
            self._state = ArmState.FLOATING
        logger.info("arm7_osc: FLOATING (gravity compensation only)")

    def disable_arm(self) -> None:
        """Float the arm. Kept as the name the spec uses; see :meth:`float_arm`."""
        self.float_arm()

    def go_limp(self) -> None:
        """Drop to zero torque. The arm will fall -- support it first."""
        with self._lock:
            self._state = ArmState.DISABLED
        logger.warning("arm7_osc: DISABLED (zero torque) -- the arm is unsupported")

    def resume(self) -> None:
        """Leave ``DAMPED`` and hold the current position again.

        Recovery is explicit by design. A stalled upstream process that restarts and resumes
        publishing must not cause the arm to jump to a stale target on its own.
        """
        with self._lock:
            if self._state is not ArmState.DAMPED:
                return
            if self._fault is not None:
                raise RuntimeError(f"cannot resume while faulted: {self._fault}. Call clear_fault() first.")
            with self._state_lock:
                self._joint_cmd = self._q.copy()
            self._cmd_time = time.monotonic()
            self._enable_time = time.monotonic()
            self._state = ArmState.ACTIVE
        logger.info("arm7_osc: resumed, ACTIVE")

    def clear_fault(self) -> None:
        with self._lock:
            self._fault = None

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        try:
            self.motor_chain.set_commands(np.zeros(self._n_motors))
        except Exception as e:  # a chain already torn down should not mask the shutdown
            logger.warning("arm7_osc: zeroing torque on close failed: %s", e)

    def __enter__(self) -> "Arm7OSC":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- commands and observations -------------------------------------------

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        """Set the target. Feeds the task layer via FK and the nullspace layer directly.

        Safe to call at any rate from any thread; the control loop runs at its own rate off the
        latest value. Out-of-range targets are clamped rather than refused, because refusing a
        target mid-trajectory leaves the arm holding a stale one.
        """
        q = np.asarray(joint_pos, dtype=float)[:ARM_DOF]
        if q.shape != (ARM_DOF,):
            raise ValueError(f"expected {ARM_DOF} joint positions, got {np.shape(joint_pos)}")
        clamped = np.clip(q, self._joint_limits[:, 0], self._joint_limits[:, 1])
        if not np.allclose(clamped, q):
            logger.warning("arm7_osc: joint_cmd clamped to limits (requested %s)", np.round(q, 4))
        with self._lock:
            self._joint_cmd = clamped
            self._cmd_time = time.monotonic()

    def move_joints(self, target: np.ndarray, duration: float = 2.0, rate_hz: float = 100.0) -> None:
        """Walk the command from where the arm is now to ``target``, then leave it there.

        Blocks for ``duration``. Commanding a distant pose in one step asks the controller for
        whatever velocity its gains imply, which on a stiff setting trips the ``MAX_JOINT_VEL``
        safe stop -- a correctly working guard reacting to a badly formed request. Interpolating
        keeps the request inside what the arm can do, and mirrors ``MotorChainRobot.move_joints``.

        Only meaningful while ACTIVE; in FLOATING the command is stored but nothing tracks it.
        """
        target = np.clip(np.asarray(target, dtype=float)[:ARM_DOF], self._joint_limits[:, 0], self._joint_limits[:, 1])
        with self._lock:
            start = self._joint_cmd.copy() if self._joint_cmd is not None else self._q.copy()
        steps = max(1, int(duration * rate_hz))
        for i in range(1, steps + 1):
            self.command_joint_pos(start + (target - start) * (i / steps))
            time.sleep(duration / steps)
        self.command_joint_pos(target)

    def get_observations(self) -> Dict[str, np.ndarray]:
        """Joint state plus task-space error, using the repo's observation key names."""
        with self._state_lock:
            return {
                "joint_pos": self._q.copy(),
                "joint_vel": self._dq.copy(),
                "joint_eff": self._tau.copy(),
                "ee_pos_err": self._ee_pos_err.copy(),
                "ee_rot_err": self._ee_rot_err.copy(),
                "feedback_age": self._feedback_age.copy(),
            }

    def set_torque_bias(self, bias: np.ndarray) -> None:
        """Add a constant per-joint torque on top of gravity, while ``FLOATING`` only.

        For identifying the gravity model against the arm itself. Clamped to
        ``MAX_TORQUE_BIAS_NM``; the loop's own slew and magnitude limits still apply on top, so
        this cannot produce a step or exceed the configured torque ceiling. Ignored in every other
        state, so a forgotten bias cannot leak into a driving controller.
        """
        b = np.clip(np.asarray(bias, dtype=float)[:ARM_DOF], -MAX_TORQUE_BIAS_NM, MAX_TORQUE_BIAS_NM)
        with self._state_lock:
            self._torque_bias[:] = b

    @property
    def torque_bias(self) -> np.ndarray:
        """The bias currently added to gravity while ``FLOATING``. A copy; safe to keep."""
        with self._state_lock:
            return self._torque_bias.copy()

    def get_joint_pos(self) -> np.ndarray:
        with self._state_lock:
            return self._q.copy()

    def num_dofs(self) -> int:
        return ARM_DOF

    @property
    def state(self) -> ArmState:
        with self._lock:
            return self._state

    @property
    def fault(self) -> Optional[str]:
        with self._lock:
            return self._fault

    def get_ee_pose(self, q: Optional[np.ndarray] = None) -> np.ndarray:
        """4x4 world pose of the controlled site, at ``q`` or at the current measured state.

        Thread-safe: uses its own scratch, not the control loop's.
        """
        if q is None:
            with self._state_lock:
                q = self._q.copy()
        with self._probe_lock:
            pos, rot = self._forward_kinematics(self._probe_data, q)
            pose = np.eye(4)
            pose[:3, :3] = rot
            pose[:3, 3] = pos
        return pose

    # ---- control loop --------------------------------------------------------

    def _control_loop(self) -> None:
        try:
            import os

            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(os.sched_get_priority_max(os.SCHED_FIFO)))
        except (PermissionError, OSError, AttributeError):
            logger.info(
                "arm7_osc: running without real-time priority; loop jitter will be worse. "
                "Grant it via /etc/security/limits.d/99-realtime.conf."
            )

        # Move startup state out of the generational collector, so a collection during the run
        # walks only the loop's own short-lived objects.
        if self.freeze_gc:
            gc.collect()
            gc.freeze()

        next_tick = time.monotonic()
        while not self._stop.is_set():
            self._step()
            self.timer.tick()
            next_tick += self._control_dt
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # Overran the period: give up the lost time rather than compounding the debt.
                next_tick = time.monotonic()

    def _step(self) -> None:
        infos = self.motor_chain.set_commands(self._cmd_vec, kd=self._kd_vec, get_state=True)
        if infos is not None:
            now = time.time()
            for i in range(ARM_DOF):
                self._raw_pos[i] = infos[i].pos
                self._raw_vel[i] = infos[i].vel
                self._raw_time[i] = infos[i].timestamp
            # Everything downstream -- J, M, FK, and both damping terms -- runs on the observer's
            # output, never the raw feedback: the raw signal is both noisy and skewed across
            # polling groups, and the observer is what fixes both.
            self._track_refresh_intervals(now)
            q_obs, dq_obs = self.observer.update(self._raw_pos, self._raw_vel, self._raw_time, now)
            with self._state_lock:
                self._q[:] = q_obs
                self._dq[:] = dq_obs
                self._feedback_age[:] = now - self._raw_time
                q = self._q.copy()
                dq = self._dq.copy()
        else:
            with self._state_lock:
                q, dq = self._q.copy(), self._dq.copy()

        with self._lock:
            state = self._state
            joint_cmd = None if self._joint_cmd is None else self._joint_cmd.copy()
            cmd_age = time.monotonic() - self._cmd_time
            ramp = 1.0 if TORQUE_RAMP_S <= 0 else min(1.0, (time.monotonic() - self._enable_time) / TORQUE_RAMP_S)

            if state is ArmState.ACTIVE:
                stale = self._stale_feedback(time.time())
                if stale is not None:
                    self._fault = stale
                    self._state = state = ArmState.DAMPED
                    logger.error("arm7_osc: %s -- entering DAMPED", stale)
                elif np.abs(dq).max() > MAX_JOINT_VEL:
                    self._fault = f"joint velocity {np.abs(dq).max():.2f} rad/s over limit {MAX_JOINT_VEL}"
                    self._state = state = ArmState.DAMPED
                    logger.error("arm7_osc: %s -- entering DAMPED", self._fault)
                elif cmd_age > COMMAND_TIMEOUT_S:
                    self._state = state = ArmState.DAMPED
                    logger.warning(
                        "arm7_osc: no command for %.3f s (limit %.3f) -- entering DAMPED; call resume() to re-engage",
                        cmd_age,
                        COMMAND_TIMEOUT_S,
                    )

        tau = self._compute_torque(state, q, dq, joint_cmd, ramp)

        # Bound the slew, then publish. Applies in every state, so entering or leaving a state
        # cannot produce a torque step.
        max_step = self.max_torque_rate * self._control_dt
        tau = np.clip(tau, self._prev_tau - max_step, self._prev_tau + max_step)
        tau = np.clip(tau, -self._torque_limit, self._torque_limit)
        self._prev_tau = tau

        self._cmd_vec[:] = 0.0
        self._cmd_vec[:ARM_DOF] = tau
        self._kd_vec[:] = 0.0
        self._kd_vec[:ARM_DOF] = self.motor_kd_active if state is ArmState.ACTIVE else self.motor_kd_float
        with self._state_lock:
            self._tau[:] = tau

    def _track_refresh_intervals(self, now: float) -> None:
        """Learn how often each motor actually refreshes, from its own timestamps.

        A motor that was not polled this round repeats its previous timestamp, so a change in the
        timestamp is exactly one refresh. The interval between changes is the per-motor feedback
        period, which is what the missing-frame criterion must be set against.
        """
        if not self._feedback_seen:
            self._last_refresh_time[:] = self._raw_time
            self._feedback_seen = True
            return
        for i in range(ARM_DOF):
            if self._raw_time[i] > self._last_refresh_time[i]:
                interval = self._raw_time[i] - self._last_refresh_time[i]
                self._last_refresh_time[i] = self._raw_time[i]
                # Slow first-order tracking: one long interval should not move the estimate much,
                # or a hiccup would relax the very timeout meant to catch it.
                if self._refresh_interval[i] == 0.0:
                    self._refresh_interval[i] = interval
                else:
                    self._refresh_interval[i] += 0.02 * (interval - self._refresh_interval[i])

    def _stale_feedback(self, now: float) -> Optional[str]:
        """Name the first motor whose feedback has gone quiet, or ``None`` if all are current.

        Two ways for a motor to be quiet. One that has been answering and stops is caught by
        comparing its age against the interval it was refreshing at. One that was **never**
        answering has no interval to compare against, so it needs its own check -- otherwise the
        loop skips it silently and a whole run is measured on an arm with one joint fewer than it
        believes.
        """
        if not self._feedback_seen:
            return None
        for i in range(ARM_DOF):
            interval = self._refresh_interval[i]
            if interval <= 0.0:
                # Never refreshed. Judge it against how long the others have been running rather
                # than against its own absent interval.
                age = now - self._last_refresh_time[i]
                if age > NEVER_REFRESHED_S:
                    return (
                        f"joint {i + 1} has never refreshed: its feedback is {age * 1e3:.0f} ms old "
                        f"and it has not moved since start. It is not on the bus."
                    )
                continue
            limit = max(FEEDBACK_TIMEOUT_CYCLES * interval, FEEDBACK_TIMEOUT_FLOOR_S)
            age = now - self._last_refresh_time[i]
            if age > limit:
                return (
                    f"joint {i + 1} feedback {age * 1e3:.0f} ms old, over {limit * 1e3:.0f} ms "
                    f"({FEEDBACK_TIMEOUT_CYCLES:.0f} x its measured {interval * 1e3:.1f} ms refresh)"
                )
        return None

    @property
    def feedback_intervals(self) -> np.ndarray:
        """Measured per-motor feedback period, seconds. The rate the watchdog is keyed to."""
        return self._refresh_interval.copy()

    def _compute_torque(
        self,
        state: ArmState,
        q: np.ndarray,
        dq: np.ndarray,
        joint_cmd: Optional[np.ndarray],
        ramp: float,
    ) -> np.ndarray:
        if state is ArmState.DISABLED:
            return np.zeros(ARM_DOF)

        # Only the control torque ramps; gravity is applied in full from the first tick.
        gravity = self._gravity(q)
        if state is ArmState.FLOATING:
            # The torque bias applies in FLOATING only.
            with self._state_lock:
                return gravity + self._torque_bias
        if state is ArmState.DAMPED or joint_cmd is None:
            # Gravity only. Damping comes from the motor-side kd, which _step has already chosen.
            return gravity

        if self.mode is ControlMode.KINEMATIC:
            tau_ctrl = self._torque_kinematic(q, dq, joint_cmd)
        else:
            tau_ctrl = self._torque_full(q, dq, joint_cmd)
        return tau_ctrl * ramp + gravity

    def _torque_kinematic(self, q: np.ndarray, dq: np.ndarray, joint_cmd: np.ndarray) -> np.ndarray:
        """Joint PD split between the task subspace and its orthogonal complement.

        ``N_kin = I - J^+ J`` projects onto the nullspace and ``I - N_kin = J^+ J`` onto the task
        subspace, so one joint-space PD toward ``joint_cmd`` serves both layers while each keeps
        its own gains. No mass matrix appears, which is what makes this mode immune to inertia
        error in the model.
        """
        jac = self._site_jacobian(q)
        jac_pinv = np.linalg.pinv(jac, rcond=SINGULARITY_RCOND)
        n_null = np.eye(ARM_DOF) - jac_pinv @ jac
        n_task = np.eye(ARM_DOF) - n_null

        err = joint_cmd - q
        tau_task = n_task @ (self.kp_task * err - self.kd_task * dq)
        # Scalar gains keep N_kin's symmetry, so nullspace stiffness is isotropic.
        tau_null = n_null @ (self.kp_null * err - self.kd_null * dq)

        self._record_task_error(q, joint_cmd)
        return tau_task + tau_null

    def _torque_full(self, q: np.ndarray, dq: np.ndarray, joint_cmd: np.ndarray) -> np.ndarray:
        """Cartesian impedance with inertia shaping and a dynamically consistent nullspace.

        ::

            Mx     = (J M^-1 J^T)^-1                     operational-space inertia
            F      = Kp6 * err - Kd6 * (J dq)            a wrench, in task space
            tau_task = J^T Mx F
            Jbar   = M^-1 J^T Mx                         dynamically consistent inverse
            tau_null = (I - J^T Jbar^T) (-Kp_n (q - q_cmd) - Kd_n dq)

        ``Kp6`` is an acceleration gain, ``omega^2``, not a stiffness in N/m: ``Mx`` carries the
        mass. Both terms read ``M``, so this mode depends on the inertia model.
        """
        jac, mass = self._jacobian_and_mass(q)

        # Wrist reflected inertia is small and its model noisy; inverting Mx amplifies both. See
        # WRIST_INERTIA_AUGMENTATION for why the reference's value cannot be carried over as-is.
        if self.wrist_inertia_augmentation:
            mass = mass.copy()
            idx = np.arange(ARM_DOF - 3, ARM_DOF)
            mass[idx, idx] += self.wrist_inertia_augmentation

        mass_inv = np.linalg.inv(mass)
        mx_inv = jac @ mass_inv @ jac.T

        # Near a singularity J loses rank, mx_inv collapses with it, and a plain inverse turns a
        # small task error into an enormous wrench. Fall back to the damped pseudoinverse exactly
        # there, and nowhere else -- pinv everywhere would blunt the controller at every pose.
        if abs(np.linalg.det(mx_inv)) < MX_DET_FLOOR:
            mx = np.linalg.pinv(mx_inv, rcond=SINGULARITY_RCOND)
        else:
            mx = np.linalg.inv(mx_inv)

        pos_des, rot_des = self._forward_kinematics(self._cmd_data, joint_cmd)
        pos_cur, rot_cur = self._forward_kinematics(self._data, q)
        err = np.empty(6)
        err[:3] = pos_des - pos_cur
        err[3:] = rotation_error(rot_des, rot_cur)
        with self._state_lock:
            self._ee_pos_err[:] = err[:3]
            self._ee_rot_err[:] = err[3:]

        wrench = self.kp_cartesian * err - self.kd_cartesian * (jac @ dq)
        tau_task = jac.T @ mx @ wrench

        jbar = mass_inv @ jac.T @ mx
        n_null = np.eye(ARM_DOF) - jac.T @ jbar.T
        tau_null = n_null @ (self.kp_null * (joint_cmd - q) - self.kd_null * dq)

        return tau_task + tau_null

    # ---- kinematics and dynamics helpers -------------------------------------

    def _forward_kinematics(self, data: mujoco.MjData, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Site position and rotation at ``q``, using the cheapest sufficient MuJoCo pass."""
        data.qpos[:ARM_DOF] = q
        mujoco.mj_kinematics(self._model, data)
        return data.site_xpos[self._site_id], data.site_xmat[self._site_id].reshape(3, 3)

    def _site_jacobian(self, q: np.ndarray) -> np.ndarray:
        """6 x ARM_DOF site Jacobian at ``q``.

        ``mj_jacSite`` needs body frames and the centre-of-mass offsets it differentiates about,
        which is ``mj_kinematics`` followed by ``mj_comPos`` -- markedly cheaper than
        ``mj_forward``, which would also solve constraints the controller does not care about.
        """
        self._data.qpos[:ARM_DOF] = q
        mujoco.mj_kinematics(self._model, self._data)
        mujoco.mj_comPos(self._model, self._data)
        mujoco.mj_jacSite(self._model, self._data, self._jacp, self._jacr, self._site_id)
        self._jac[:3] = self._jacp
        self._jac[3:] = self._jacr
        return self._jac[:, :ARM_DOF]

    def _jacobian_and_mass(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Site Jacobian and joint-space mass matrix at ``q``, from one MuJoCo pass.

        ``mj_crb`` alone leaves ``qM`` at zero in MuJoCo 3.8 -- ``mj_makeM`` is what assembles it,
        and the omission is silent: the mass matrix reads as all zeros, so ``Mx`` becomes garbage
        rather than raising. :meth:`_probe_mass_pipeline` pins the working sequence at startup
        rather than trusting a version check.
        """
        self._data.qpos[:ARM_DOF] = q
        for step in self._mass_pipeline:
            step(self._model, self._data)
        mujoco.mj_jacSite(self._model, self._data, self._jacp, self._jacr, self._site_id)
        self._jac[:3] = self._jacp
        self._jac[3:] = self._jacr
        mujoco.mj_fullM(self._model, self._mass_full, self._data.qM)
        return self._jac[:, :ARM_DOF], self._mass_full[:ARM_DOF, :ARM_DOF]

    def _probe_mass_pipeline(self) -> List[Any]:
        """Pick the cheapest MuJoCo sequence that actually fills ``qM``, by trying it.

        The minimal chain gained a step in MuJoCo 3.8 (``mj_makeM``), and getting it wrong yields a
        zero mass matrix with no error. Probing once beats pinning a version, and beats paying for
        ``mj_forward`` -- which also solves constraints this controller never looks at -- on every
        one of 500 ticks a second.
        """
        minimal: List[Any] = [mujoco.mj_kinematics, mujoco.mj_comPos, mujoco.mj_crb]
        if hasattr(mujoco, "mj_makeM"):
            minimal.append(mujoco.mj_makeM)

        probe = mujoco.MjData(self._model)
        probe.qpos[:ARM_DOF] = 0.1  # a pose with nonzero inertia coupling
        for step in minimal:
            step(self._model, probe)
        mass = np.zeros((self._model.nv, self._model.nv))
        mujoco.mj_fullM(self._model, mass, probe.qM)
        if np.abs(np.diag(mass)[:ARM_DOF]).min() > 0.0:
            return minimal

        logger.warning("arm7_osc: minimal mass-matrix pipeline yielded zeros; falling back to mj_forward")
        return [mujoco.mj_forward]

    def _gravity(self, q: np.ndarray) -> np.ndarray:
        """``g(q)`` on the arm joints.

        Inverse dynamics at rest. Note this already includes the weight of everything distal --
        wrist, gripper body and fingers -- reflected onto the arm joints, because it is computed on
        the whole model.
        """
        g = self._kdl.compute_inverse_dynamics(q, np.zeros(ARM_DOF), np.zeros(ARM_DOF))
        return g[:ARM_DOF] * self._gravity_comp_factor

    def _record_task_error(self, q: np.ndarray, joint_cmd: np.ndarray) -> None:
        """Task-space error, for monitoring. Not used by the ``kinematic`` control law."""
        pos_des, rot_des = self._forward_kinematics(self._cmd_data, joint_cmd)
        pos_cur, rot_cur = self._forward_kinematics(self._data, q)
        with self._state_lock:
            self._ee_pos_err[:] = pos_des - pos_cur
            self._ee_rot_err[:] = rotation_error(rot_des, rot_cur)

    # ---- introspection --------------------------------------------------------

    def site_jacobian(self, q: np.ndarray) -> np.ndarray:
        """6 x ARM_DOF site Jacobian at ``q``. Thread-safe; allocates, so not for the loop."""
        with self._probe_lock:
            self._probe_data.qpos[:ARM_DOF] = q
            mujoco.mj_kinematics(self._model, self._probe_data)
            mujoco.mj_comPos(self._model, self._probe_data)
            mujoco.mj_jacSite(self._model, self._probe_data, self._probe_jacp, self._probe_jacr, self._site_id)
            return np.vstack([self._probe_jacp, self._probe_jacr])[:, :ARM_DOF]

    def dynamic_nullspace_projector(self, q: np.ndarray) -> np.ndarray:
        """``N_dyn = I - J^T Jbar^T`` at ``q``, the projector ``full`` mode uses.

        Unlike :meth:`nullspace_projector` this is oblique, not orthogonal: it projects along the
        mass metric, which is what makes nullspace torque produce no end-effector acceleration.
        Thread-safe; allocates, so not for the loop.
        """
        with self._probe_lock:
            jac, mass = self._probe_jacobian_and_mass(q)
            mass_inv = np.linalg.inv(mass)
            mx_inv = jac @ mass_inv @ jac.T
            if abs(np.linalg.det(mx_inv)) < MX_DET_FLOOR:
                mx = np.linalg.pinv(mx_inv, rcond=SINGULARITY_RCOND)
            else:
                mx = np.linalg.inv(mx_inv)
            jbar = mass_inv @ jac.T @ mx
            return np.eye(ARM_DOF) - jac.T @ jbar.T

    def _probe_jacobian_and_mass(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """:meth:`_jacobian_and_mass` on the introspection scratch, safe to call off-thread."""
        self._probe_data.qpos[:ARM_DOF] = q
        for step in self._mass_pipeline:
            step(self._model, self._probe_data)
        mujoco.mj_jacSite(self._model, self._probe_data, self._probe_jacp, self._probe_jacr, self._site_id)
        jac = np.vstack([self._probe_jacp, self._probe_jacr])[:, :ARM_DOF]
        mass = np.zeros((self._model.nv, self._model.nv))
        mujoco.mj_fullM(self._model, mass, self._probe_data.qM)
        return jac, mass[:ARM_DOF, :ARM_DOF]

    def nullspace_projector(self, q: np.ndarray) -> np.ndarray:
        """``N_kin = I - J^+ J`` at ``q``. Exposed for tests and for tuning diagnostics.

        Thread-safe, unlike the loop's internal path, so a viewer or a tuning script can call it
        while the arm is running.
        """
        jac = self.site_jacobian(q)
        return np.eye(ARM_DOF) - np.linalg.pinv(jac, rcond=SINGULARITY_RCOND) @ jac

    def loop_stats(self) -> Dict[str, float]:
        """Loop-period statistics, including p99 and max. See :class:`LoopTimer`."""
        return self.timer.stats()

    def __repr__(self) -> str:
        return f"Arm7OSC(mode={self.mode.value}, site={self.ee_site!r}, state={self.state.value})"


def joint_names(model_path: str) -> List[str]:
    """Arm joint names from a model, in index order. Convenience for logging and recording."""
    model = mujoco.MjModel.from_xml_path(model_path)
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(min(ARM_DOF, model.njnt))]
