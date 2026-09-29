"""A MuJoCo-backed :class:`~i2rt.motor_drivers.dm_driver.MotorChain` for torque control.

Implements the ``MotorChain`` protocol against a MuJoCo model, taking MIT-mode
``(pos, vel, kp, kd, torque)`` commands and returning :class:`MotorInfo` feedback, so a controller
written for the CAN chain runs against it unmodified. The controller keeps its own model for
gravity and Jacobians; this class is the plant.

Three hardware behaviours are reproduced, each **off by default**:

* **Encoder quantization** (``encoder_bits``). Position is quantized and velocity is differentiated
  from the quantized signal, as on motors with no separate velocity sensor.
* **Round-robin polling** (``poll_groups``). Only the motors in this call's group get new readings;
  the rest keep their previous values and timestamps.
* **Dropped frames** (``drop_frames``). Listed motors keep their last value and timestamp.

Sim time advances by exactly ``control_dt`` per ``set_commands`` call, not by wall-clock elapsed
time.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

import mujoco
import numpy as np

from i2rt.motor_drivers.dm_driver import MotorChain
from i2rt.motor_drivers.utils import MotorInfo, MotorType, float_to_uint, uint_to_float

logger = logging.getLogger(__name__)

# MIT-mode command fields are packed into 12 bits on the wire (position gets 16). Torque is the one
# that matters for a torque controller: over a DM4340's +/-28 Nm that is ~13.7 mNm of resolution.
_TORQUE_CMD_BITS = 12

# Simulated motor temperatures (degrees C), matching SimRobot's idle values so that consumers
# rendering a feedback table see the same thing in either sim path.
_SIM_TEMP_MOS = 35.0
_SIM_TEMP_ROTOR = 40.0

# error_code that DMChainCanInterface reports for a healthy motor.
_OK_ERROR_CODE = "0x1"


class SimMotorChain(MotorChain):
    """A MuJoCo plant behind the ``MotorChain`` command/feedback interface.

    Args:
        xml_path: MJCF for the plant. Loaded into this object's own ``MjModel``/``MjData``.
        motor_list: ``(can_id, motor_type)`` per motor, ordered as the command vector is. Used for
            the reported ids and for the per-type torque saturation and quantization ranges.
        dof_indices: MuJoCo velocity-address per motor. Defaults to ``0..len(motor_list)-1``. Pass
            it when the model carries joints no motor drives.
        control_dt: sim time advanced per ``set_commands`` call.
        timestep: physics timestep, overriding the model's own. ``None`` keeps the model's.
        gravity: world gravity, for an explicit zero-g plant in tests.
        encoder_bits: rotor-encoder resolution. ``None`` reports MuJoCo's exact state.
        gear_ratio: per-motor reduction, used with ``encoder_bits`` to get joint-side resolution.
            Defaults to each motor type's real ratio.
        poll_groups: motor indices refreshed per call, cycled round-robin. ``None`` refreshes all.
        quantize_torque: round commanded torque to the wire's 12 bits.
        drop_frames: motor indices whose feedback is frozen. Mutable at runtime via
            :meth:`set_dropped_frames`.
        initial_qpos: starting ``qpos``. Defaults to the model's zero pose.
        enable_contacts: leave MuJoCo's contact dynamics on. Off by default: these models weld
            ``base`` to the world, which exempts the ``base``/``link1`` pair from MuJoCo's
            parent-child contact filter, and their hulls touch at every reachable pose. Use
            :meth:`apply_body_force` for a deliberate push instead.
        mechanism_held_dofs: DOFs on which ``qfrc_bias`` is supplied by the plant, standing in for
            a non-backdrivable mechanism carrying its own weight. Pass ``[7, 8]`` for a
            ``yam_7dof`` gripper, whose equality-coupled fingers otherwise leak their weight into
            the arm.
    """

    def __init__(
        self,
        xml_path: str,
        motor_list: Sequence[Tuple[int, str]],
        dof_indices: Optional[Sequence[int]] = None,
        control_dt: float = 1.0 / 500.0,
        timestep: Optional[float] = 5e-4,
        gravity: Optional[np.ndarray] = None,
        encoder_bits: Optional[int] = None,
        gear_ratio: Optional[Sequence[float]] = None,
        poll_groups: Optional[Sequence[Sequence[int]]] = None,
        quantize_torque: bool = False,
        drop_frames: Optional[Sequence[int]] = None,
        initial_qpos: Optional[np.ndarray] = None,
        enable_contacts: bool = False,
        mechanism_held_dofs: Optional[Sequence[int]] = None,
    ) -> None:
        self.motor_list = list(motor_list)
        n = len(self.motor_list)

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        if timestep is not None:
            self.model.opt.timestep = timestep
        if gravity is not None:
            self.model.opt.gravity = np.asarray(gravity, dtype=float)
        if not enable_contacts:
            # Matches MuJoCoKDL, so the plant and the controller's gravity model agree. Joint
            # limits stay on: those are real hard stops, not a modelling artefact.
            self.model.geom_contype[:] = 0
            self.model.geom_conaffinity[:] = 0

        self._dof_indices = np.arange(n) if dof_indices is None else np.asarray(dof_indices, dtype=int)
        if self._dof_indices.shape != (n,):
            raise ValueError(f"dof_indices must have one entry per motor ({n}), got {self._dof_indices.shape}")
        if self._dof_indices.max(initial=-1) >= self.model.nv:
            raise ValueError(f"dof_indices {self._dof_indices} exceed model nv={self.model.nv}")

        self._control_dt = control_dt
        self._n_substeps = max(1, round(control_dt / self.model.opt.timestep))
        self._held_dofs = (
            np.asarray(mechanism_held_dofs, dtype=int) if mechanism_held_dofs is not None else np.empty(0, dtype=int)
        )

        self._constants = [MotorType.get_motor_constants(t) for _, t in self.motor_list]
        self._encoder_bits = encoder_bits
        if gear_ratio is None:
            self._gear_ratio = np.array([MotorType.get_gear_ratio(t) for _, t in self.motor_list], dtype=float)
        else:
            self._gear_ratio = np.asarray(gear_ratio, dtype=float)
        self._quantize_torque = quantize_torque

        # Round-robin polling. One group per call; None means "the whole chain every call".
        self._poll_groups = [np.asarray(g, dtype=int) for g in poll_groups] if poll_groups is not None else None
        self._poll_cursor = 0
        self._dropped = set(int(i) for i in drop_frames) if drop_frames else set()

        if initial_qpos is not None:
            nq = min(len(initial_qpos), self.model.nq)
            self.data.qpos[:nq] = np.asarray(initial_qpos, dtype=float)[:nq]
        mujoco.mj_forward(self.model, self.data)

        # Latched feedback. Hardware feedback is whatever last arrived, not the plant's true state,
        # so the two are kept apart: these are only refreshed for the motors polled on a call.
        self._fb_pos = self._read_encoder(self.data.qpos[self._dof_indices])
        self._fb_vel = np.zeros(n)
        self._fb_eff = np.zeros(n)
        self._fb_time = np.full(n, time.time())
        self._prev_pos = self._fb_pos.copy()
        self._prev_sim_time = np.zeros(n)
        self._last_torques = np.zeros(n)

        # Reentrant on purpose. The lock is public (see the `lock` property) so a viewer can
        # synchronise with the control loop, and every public accessor here takes it too -- with a
        # plain Lock, a caller holding it and then calling get_site_pose() would deadlock, which
        # makes the exposed lock a trap. DMChainCanInterface uses an RLock for the same reason.
        self._lock = threading.RLock()
        self._sim_time = 0.0
        self._call_count = 0
        self._wall_start = time.time()

        # Attributes DMChainCanInterface exposes that callers read directly.
        self.motor_offset = np.zeros(n)
        self.motor_direction = np.ones(n)
        self.running = True

    # ---- MotorChain protocol -------------------------------------------------

    def __len__(self) -> int:
        return len(self.motor_list)

    def __repr__(self) -> str:
        return f"SimMotorChain(n_motors={len(self.motor_list)}, control_dt={self._control_dt})"

    def set_commands(
        self,
        torques: np.ndarray,
        pos: Optional[np.ndarray] = None,
        vel: Optional[np.ndarray] = None,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
        get_state: bool = True,
    ) -> Optional[List[MotorInfo]]:
        """Apply one MIT-mode command set, advance the plant by ``control_dt``, return feedback.

        The MIT law is evaluated on the plant's true state, which is what the motor's own onboard
        loop does -- it closes on its rotor encoder, not on whatever the host last received.
        """
        with self._lock:
            tau = self._mit_torque(torques, pos, vel, kp, kd)
            self._last_torques = tau

            self.data.qfrc_applied[:] = 0.0
            self.data.qfrc_applied[self._dof_indices] = tau
            for _ in range(self._n_substeps):
                if self._held_dofs.size:
                    # Recomputed per substep: qfrc_bias is configuration dependent, and holding a
                    # mechanism means holding it at wherever it now is.
                    self.data.qfrc_applied[self._held_dofs] = self.data.qfrc_bias[self._held_dofs]
                mujoco.mj_step(self.model, self.data)

            self._sim_time += self._control_dt
            self._call_count += 1
            self._refresh_feedback()

        return self.read_states(torques=torques) if get_state else None

    def read_states(self, torques: Optional[np.ndarray] = None) -> List[MotorInfo]:
        """Latest feedback per motor, mirroring ``DMChainCanInterface.read_states``.

        ``timestamp`` is per motor and is the age of *that* motor's reading, so under
        ``poll_groups`` the returned list is not a snapshot and says so.
        """
        with self._lock:
            return [
                MotorInfo(
                    id=self.motor_list[i][0],
                    error_code=_OK_ERROR_CODE,
                    target_torque=float(torques[i]) if torques is not None else 0.0,
                    pos=float(self._fb_pos[i]),
                    vel=float(self._fb_vel[i]),
                    eff=float(self._fb_eff[i]),
                    temp_mos=_SIM_TEMP_MOS,
                    temp_rotor=_SIM_TEMP_ROTOR,
                    timestamp=float(self._fb_time[i]),
                )
                for i in range(len(self.motor_list))
            ]

    def close(self) -> None:
        self.running = False

    @property
    def comm_freq(self) -> float:
        """Mean calls per second of wall time, matching what the CAN chain reports."""
        elapsed = time.time() - self._wall_start
        return self._call_count / elapsed if elapsed > 0 else 0.0

    # ---- sim-only introspection ----------------------------------------------

    @property
    def lock(self) -> "threading.RLock":
        """The lock guarding the plant's ``MjData``. Physics runs inside :meth:`set_commands` on
        the caller's thread, so anything reading ``self.data`` from another thread holds this.
        Reentrant, so it is safe to hold while calling this class's own accessors."""
        return self._lock

    @property
    def sim_time(self) -> float:
        """Sim time elapsed, i.e. ``control_dt`` per ``set_commands`` call."""
        return self._sim_time

    @property
    def effective_feedback_divisor(self) -> int:
        """Loop calls between successive refreshes of any one motor.

        This -- not the loop rate -- is what a per-motor feedback rate has to be divided by, and
        what a missing-frame criterion has to be set against.
        """
        return len(self._poll_groups) if self._poll_groups is not None else 1

    def get_true_state(self) -> Dict[str, np.ndarray]:
        """The plant's exact state, bypassing encoder and polling effects. For assertions only."""
        with self._lock:
            return {
                "pos": self.data.qpos[self._dof_indices].copy(),
                "vel": self.data.qvel[self._dof_indices].copy(),
                "torque": self._last_torques.copy(),
            }

    def get_site_pose(self, site_name: str) -> np.ndarray:
        """4x4 world pose of ``site_name`` in the plant. For assertions only."""
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if sid == -1:
            raise ValueError(f"site {site_name!r} not found in the plant model")
        with self._lock:
            pose = np.eye(4)
            pose[:3, :3] = self.data.site_xmat[sid].reshape(3, 3)
            pose[:3, 3] = self.data.site_xpos[sid]
            return pose

    def apply_body_force(self, body_name: str, force: np.ndarray) -> None:
        """Hold a world-frame force on a body, as a stand-in for a hand pushing the arm.

        Persists until called again with a zero force. ``xfrc_applied`` survives ``mj_step``, so
        this is the sim equivalent of leaning on the elbow.
        """
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid == -1:
            raise ValueError(f"body {body_name!r} not found in the plant model")
        with self._lock:
            self.data.xfrc_applied[bid, :3] = np.asarray(force, dtype=float)

    def place(self, qpos: np.ndarray) -> None:
        """Teleport the plant to ``qpos``, zeroing velocity, acceleration and any applied force."""
        q = np.asarray(qpos, dtype=float)
        with self._lock:
            self.data.qpos[: len(q)] = q
            self.data.qvel[:] = 0.0
            self.data.qacc[:] = 0.0
            self.data.qfrc_applied[:] = 0.0
            self.data.xfrc_applied[:] = 0.0
            mujoco.mj_forward(self.model, self.data)
            self._refresh_feedback()

    def gravity_torque(self) -> np.ndarray:
        """``qfrc_bias`` evaluated at the current pose with velocity zeroed, i.e. the plant's own
        gravitational torque. Velocity is restored before returning."""
        with self._lock:
            vel = self.data.qvel.copy()
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)
            bias = self.data.qfrc_bias.copy()
            self.data.qvel[:] = vel
            mujoco.mj_forward(self.model, self.data)
        return bias

    def set_dropped_frames(self, indices: Sequence[int]) -> None:
        """Freeze feedback for ``indices``, to exercise a communication watchdog."""
        with self._lock:
            self._dropped = set(int(i) for i in indices)

    # ---- internals ------------------------------------------------------------

    def _mit_torque(
        self,
        torques: np.ndarray,
        pos: Optional[np.ndarray],
        vel: Optional[np.ndarray],
        kp: Optional[np.ndarray],
        kd: Optional[np.ndarray],
    ) -> np.ndarray:
        """``tau_ff + kp*(pos - q) + kd*(vel - dq)``, saturated to each motor's torque range."""
        q = self.data.qpos[self._dof_indices]
        dq = self.data.qvel[self._dof_indices]

        tau = np.asarray(torques, dtype=float).copy()
        if self._quantize_torque:
            for i, c in enumerate(self._constants):
                raw = float_to_uint(tau[i], c.TORQUE_MIN, c.TORQUE_MAX, _TORQUE_CMD_BITS)
                tau[i] = uint_to_float(raw, c.TORQUE_MIN, c.TORQUE_MAX, _TORQUE_CMD_BITS)
        if kp is not None and pos is not None:
            tau += np.asarray(kp, dtype=float) * (np.asarray(pos, dtype=float) - q)
        if kd is not None:
            target_vel = np.asarray(vel, dtype=float) if vel is not None else 0.0
            tau += np.asarray(kd, dtype=float) * (target_vel - dq)

        lo = np.array([c.TORQUE_MIN for c in self._constants])
        hi = np.array([c.TORQUE_MAX for c in self._constants])
        return np.clip(tau, lo, hi)

    def _read_encoder(self, q: np.ndarray) -> np.ndarray:
        """Quantize joint position to the rotor encoder's joint-side resolution."""
        if self._encoder_bits is None:
            return np.asarray(q, dtype=float).copy()
        counts_per_joint_rev = (1 << self._encoder_bits) * self._gear_ratio
        step = 2.0 * np.pi / counts_per_joint_rev
        return np.round(np.asarray(q, dtype=float) / step) * step

    def _refresh_feedback(self) -> None:
        """Latch new readings for the motors polled this call.

        Velocity is differenced from *quantized* position over the interval since that motor was
        last read, so the quantization noise and the group's own sample interval both land on it --
        which is how the hardware's velocity signal actually goes bad.
        """
        if self._poll_groups is None:
            polled = np.arange(len(self.motor_list))
        else:
            polled = self._poll_groups[self._poll_cursor]
            self._poll_cursor = (self._poll_cursor + 1) % len(self._poll_groups)

        polled = np.array([i for i in polled if i not in self._dropped], dtype=int)
        if polled.size == 0:
            return

        q_meas = self._read_encoder(self.data.qpos[self._dof_indices])[polled]

        if self._encoder_bits is None:
            self._fb_vel[polled] = self.data.qvel[self._dof_indices][polled]
        else:
            dt = self._sim_time - self._prev_sim_time[polled]
            ok = dt > 0
            self._fb_vel[polled] = np.where(ok, (q_meas - self._prev_pos[polled]) / np.where(ok, dt, 1.0), 0.0)

        self._prev_pos[polled] = q_meas
        self._prev_sim_time[polled] = self._sim_time
        self._fb_pos[polled] = q_meas
        self._fb_eff[polled] = self._last_torques[polled]
        self._fb_time[polled] = time.time()
