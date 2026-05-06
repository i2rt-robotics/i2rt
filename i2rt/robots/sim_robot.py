"""Simulated robot that implements the Robot protocol using MuJoCo.

No real hardware (CAN bus, motors) is required. Joint positions are tracked
in-memory and MuJoCo forward kinematics keeps the model consistent.
Gravity-compensation torques and temperatures are simulated via MuJoCo
inverse dynamics so that the feedback matches what MotorChainRobot provides.
"""

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mujoco
import numpy as np

from i2rt.robots.robot import Robot

# Simulated motor temperatures (°C) — typical idle values for real hardware.
_SIM_TEMP_MOS = 35.0
_SIM_TEMP_ROTOR = 40.0


@dataclass
class _SimJointState:
    """Lightweight mirror of MotorChainRobot's JointStates for duck-typing."""

    names: List[str]
    pos: np.ndarray
    vel: np.ndarray
    eff: np.ndarray
    temp_mos: np.ndarray
    temp_rotor: np.ndarray
    timestamp: float


class SimRobot(Robot):
    """A simulated robot backed by a MuJoCo model.

    Implements the same Robot protocol as MotorChainRobot so it can be used
    as a drop-in replacement for testing, visualization, and offline work.
    """

    def __init__(
        self,
        xml_path: str,
        n_dofs: int,
        joint_limits: Optional[np.ndarray] = None,
        gripper_index: Optional[int] = None,
        gripper_limits: Optional[np.ndarray] = None,
        initial_qpos: Optional[np.ndarray] = None,
        gravity_comp_factor: Optional[np.ndarray] = None,
        control_freq: float = 200.0,
    ) -> None:
        self.xml_path = xml_path
        self._n_dofs = n_dofs
        self._joint_limits = joint_limits
        self._gripper_index = gripper_index
        self._gripper_limits = gripper_limits

        self._model = mujoco.MjModel.from_xml_path(xml_path)
        self._data = mujoco.MjData(self._model)

        # Build mapping from [0, 1] command space to actual MuJoCo joint range
        # for the gripper joint.  This is needed because the command convention
        # is 0 = closed, 1 = open, but the underlying MuJoCo joint range may
        # differ (e.g. [-0.048, 0] for flexible grippers).
        self._gripper_qpos_range = None
        if gripper_index is not None and gripper_index < self._model.njnt:
            jnt_lo, jnt_hi = self._model.jnt_range[gripper_index]
            if jnt_lo != 0.0 or jnt_hi != 1.0:
                self._gripper_qpos_range = (float(jnt_lo), float(jnt_hi))

        self._lock = threading.Lock()
        self._qpos = np.zeros(n_dofs) if initial_qpos is None else np.array(initial_qpos, dtype=float)
        self._qvel = np.zeros(n_dofs)

        # Push the initial state into MuJoCo so FK is consistent.
        n = min(n_dofs, self._model.nq)
        mj_qpos = self._cmd_to_mj_qpos(self._qpos)
        self._data.qpos[:n] = mj_qpos[:n]
        mujoco.mj_forward(self._model, self._data)

        # Scratch MjData for inverse dynamics (reused every call).
        self._inv_data = mujoco.MjData(self._model)
        self._last_motor_torques: Optional[np.ndarray] = None
        self._joint_state: Optional[_SimJointState] = None
        self._update_joint_state()

        self._gravity_comp_factor = gravity_comp_factor if gravity_comp_factor is not None else np.ones(n_dofs)
        self._control_freq = control_freq
        self._physics_active = False
        self._grav_comp_enabled = True
        self._stop_event = threading.Event()
        self._physics_thread: Optional[threading.Thread] = None

    def _cmd_to_mj_qpos(self, cmd: np.ndarray) -> np.ndarray:
        """Map command-space positions to MuJoCo qpos.

        For the gripper joint, [0, 1] is mapped to the actual MuJoCo joint range.
        All other joints pass through unchanged.
        """
        mj = cmd.copy()
        if self._gripper_qpos_range is not None and self._gripper_index is not None:
            lo, hi = self._gripper_qpos_range
            mj[self._gripper_index] = lo + cmd[self._gripper_index] * (hi - lo)
        return mj

    def _mj_to_cmd_qpos(self, mj: np.ndarray) -> np.ndarray:
        """Map MuJoCo qpos back to command-space positions.

        Reverse of ``_cmd_to_mj_qpos``.  For the gripper joint the physical
        range is mapped back to [0, 1].
        """
        cmd = mj.copy()
        if self._gripper_qpos_range is not None and self._gripper_index is not None:
            lo, hi = self._gripper_qpos_range
            if hi > lo:
                cmd[self._gripper_index] = (mj[self._gripper_index] - lo) / (hi - lo)
        return cmd

    # ---- simulated feedback --------------------------------------------------

    def _compute_gravity_torques(self) -> np.ndarray:
        """Compute gravity-compensation torques via MuJoCo inverse dynamics.

        Returns an array of length ``n_dofs``.  If the MuJoCo model has fewer
        joints (e.g. the gripper joint lives outside the model), the extra
        entries are zero-padded.
        """
        nq = min(self._n_dofs, self._model.nq)
        mj_qpos = self._cmd_to_mj_qpos(self._qpos)
        self._inv_data.qpos[:nq] = mj_qpos[:nq]
        self._inv_data.qvel[:] = 0.0
        self._inv_data.qacc[:] = 0.0
        mujoco.mj_inverse(self._model, self._inv_data)
        torques = np.zeros(self._n_dofs)
        torques[:nq] = self._inv_data.qfrc_inverse[:nq]
        # Zero out gripper torque (matching MotorChainRobot behaviour).
        if self._gripper_index is not None:
            torques[self._gripper_index] = 0.0
        return torques

    def _update_joint_state(self) -> None:
        """Recompute _joint_state and _last_motor_torques from current state."""
        torques = self._compute_gravity_torques()
        self._last_motor_torques = torques.copy()
        n = self._n_dofs
        power = np.abs(torques * self._qvel)
        self._joint_state = _SimJointState(
            names=[str(i) for i in range(n)],
            pos=self._qpos.copy(),
            vel=self._qvel.copy(),
            eff=torques,
            temp_mos=_SIM_TEMP_MOS + power * 0.5 + np.random.uniform(-0.5, 0.5, n),
            temp_rotor=_SIM_TEMP_ROTOR + power * 0.8 + np.random.uniform(-0.5, 0.5, n),
            timestamp=time.time(),
        )

    # ---- physics simulation --------------------------------------------------

    def start_server(self) -> None:
        """Start the gravity-compensation physics loop.

        Once running, :meth:`_physics_loop` continuously applies gravity
        compensation torques and steps MuJoCo physics.  Call
        :meth:`enable_gravity_comp` / :meth:`disable_gravity_comp` to
        toggle the physics on and off (e.g. when switching between VIS and
        CONTROL modes in the viewer).
        """
        if self._physics_active:
            return
        self._physics_active = True
        self._stop_event.clear()
        self._physics_thread = threading.Thread(
            target=self._physics_loop,
            daemon=True,
            name="sim_physics",
        )
        self._physics_thread.start()

    def _physics_loop(self) -> None:
        control_dt = 1.0 / self._control_freq
        n_substeps = max(1, round(control_dt / self._model.opt.timestep))
        nq = min(self._n_dofs, self._model.nq)
        nq_arm = self._gripper_index if self._gripper_index is not None else nq

        next_time = time.time() + control_dt
        while not self._stop_event.is_set():
            with self._lock:
                if self._grav_comp_enabled:
                    grav = self._compute_gravity_torques()
                    grav[:nq_arm] *= self._gravity_comp_factor[:nq_arm]

                    self._data.qfrc_applied[:] = 0.0
                    self._data.qfrc_applied[:nq_arm] = grav[:nq_arm]

                    for _ in range(n_substeps):
                        mujoco.mj_step(self._model, self._data)

                    raw = self._data.qpos[:nq].copy()
                    self._qpos[:nq] = self._mj_to_cmd_qpos(raw)[:nq]
                    self._qvel[:nq] = self._data.qvel[:nq]

                self._update_joint_state()
            sleep_time = next_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            next_time += control_dt

    def enable_gravity_comp(self) -> None:
        """Resume gravity-compensation physics from the current pose."""
        with self._lock:
            self._grav_comp_enabled = True

    def disable_gravity_comp(self) -> None:
        """Pause gravity-compensation physics (CONTROL mode uses teleport)."""
        with self._lock:
            self._grav_comp_enabled = False

    # ---- Robot protocol ------------------------------------------------------

    def num_dofs(self) -> int:
        return self._n_dofs

    def get_joint_pos(self) -> np.ndarray:
        with self._lock:
            return self._qpos.copy()

    def get_joint_state(self) -> Dict[str, np.ndarray]:
        with self._lock:
            return {"pos": self._qpos.copy(), "vel": self._qvel.copy()}

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        """Command the sim robot to a target joint position.

        When physics is active (sim mode), this disables gravity comp and
        teleports ``data.qpos`` directly.  Gravity comp stays off for the
        duration of control mode and is re-enabled when the caller (e.g.
        ``MujocoControlInterface._on_key``) switches back to VIS mode via
        ``enable_gravity_comp()``.
        """
        pos = np.array(joint_pos, dtype=float)
        if self._joint_limits is not None:
            arm_end = self._gripper_index if self._gripper_index is not None else len(pos)
            pos[:arm_end] = np.clip(
                pos[:arm_end],
                self._joint_limits[:, 0],
                self._joint_limits[:, 1],
            )
        if self._gripper_index is not None and self._gripper_limits is not None:
            pos[self._gripper_index] = np.clip(
                pos[self._gripper_index],
                min(self._gripper_limits),
                max(self._gripper_limits),
            )
        with self._lock:
            if self._physics_active:
                self._grav_comp_enabled = False
            self._qpos = pos
            n = min(len(pos), self._model.nq)
            mj_qpos = self._cmd_to_mj_qpos(pos)
            self._data.qpos[:n] = mj_qpos[:n]
            self._data.qvel[:] = 0.0
            mujoco.mj_forward(self._model, self._data)
            self._update_joint_state()

    def command_target_vel(self, joint_vel: np.ndarray) -> None:
        with self._lock:
            self._qvel = np.array(joint_vel, dtype=float)
            self._update_joint_state()

    def command_joint_state(self, joint_state: Dict[str, np.ndarray]) -> None:
        self.command_joint_pos(joint_state["pos"])
        if "vel" in joint_state:
            self.command_target_vel(joint_state["vel"])

    def get_observations(self) -> Dict[str, np.ndarray]:
        with self._lock:
            self._update_joint_state()
            eff = self._joint_state.eff
            if self._gripper_index is None:
                return {
                    "joint_pos": self._qpos.copy(),
                    "joint_vel": self._qvel.copy(),
                    "joint_eff": eff.copy(),
                }
            else:
                gi = self._gripper_index
                return {
                    "joint_pos": self._qpos[:gi].copy(),
                    "joint_vel": self._qvel[:gi].copy(),
                    "joint_eff": eff[:gi].copy(),
                    "gripper_pos": np.array([self._qpos[gi]]),
                    "gripper_vel": np.array([self._qvel[gi]]),
                    "gripper_eff": np.array([eff[gi]]),
                }

    def get_motor_torques(self) -> Optional[np.ndarray]:
        """Return the last computed gravity compensation torques."""
        return self._last_motor_torques

    def get_robot_info(self) -> Dict[str, Any]:
        return {
            "joint_limits": self._joint_limits,
            "gripper_limits": self._gripper_limits,
            "gripper_index": self._gripper_index,
            "sim": True,
            "gravity_comp_factor": self._gravity_comp_factor,
        }

    def close(self) -> None:
        if self._physics_active:
            self._stop_event.set()
            if self._physics_thread is not None:
                self._physics_thread.join(timeout=2.0)
            self._physics_active = False
