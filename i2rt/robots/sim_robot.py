"""Simulated robot that implements the Robot protocol using MuJoCo.

No real hardware (CAN bus, motors) is required. Joint positions are tracked
in-memory and MuJoCo forward kinematics keeps the model consistent.
"""

import threading
from typing import Any, Dict, Optional

import mujoco
import numpy as np

from i2rt.robots.robot import Robot


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
    ) -> None:
        self.xml_path = xml_path
        self._n_dofs = n_dofs
        self._joint_limits = joint_limits
        self._gripper_index = gripper_index
        self._gripper_limits = gripper_limits

        self._model = mujoco.MjModel.from_xml_path(xml_path)
        self._data = mujoco.MjData(self._model)

        self._lock = threading.Lock()
        self._qpos = np.zeros(n_dofs) if initial_qpos is None else np.array(initial_qpos, dtype=float)
        self._qvel = np.zeros(n_dofs)

        # Push the initial state into MuJoCo so FK is consistent.
        n = min(n_dofs, self._model.nq)
        self._data.qpos[:n] = self._qpos[:n]
        mujoco.mj_forward(self._model, self._data)

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
            self._qpos = pos
            n = min(len(pos), self._model.nq)
            self._data.qpos[:n] = pos[:n]
            mujoco.mj_forward(self._model, self._data)

    def command_target_vel(self, joint_vel: np.ndarray) -> None:
        with self._lock:
            self._qvel = np.array(joint_vel, dtype=float)

    def command_joint_state(self, joint_state: Dict[str, np.ndarray]) -> None:
        self.command_joint_pos(joint_state["pos"])
        if "vel" in joint_state:
            self.command_target_vel(joint_state["vel"])

    def get_observations(self) -> Dict[str, np.ndarray]:
        with self._lock:
            if self._gripper_index is None:
                return {
                    "joint_pos": self._qpos.copy(),
                    "joint_vel": self._qvel.copy(),
                    "joint_eff": np.zeros(self._n_dofs),
                }
            else:
                return {
                    "joint_pos": self._qpos[: self._gripper_index].copy(),
                    "gripper_pos": np.array([self._qpos[self._gripper_index]]),
                    "joint_vel": self._qvel.copy(),
                    "joint_eff": np.zeros(self._n_dofs),
                }

    def get_robot_info(self) -> Dict[str, Any]:
        return {
            "joint_limits": self._joint_limits,
            "gripper_limits": self._gripper_limits,
            "gripper_index": self._gripper_index,
            "sim": True,
        }

    def close(self) -> None:
        pass
