import os

import mujoco
import numpy as np


class MuJoCoKDL:
    """A simple class for computing inverse dynamics using MuJoCo."""

    def __init__(self, path: str):
        self.model = mujoco.MjModel.from_xml_path(os.path.expanduser(path))
        self.data = mujoco.MjData(self.model)
        self.set_gravity(np.array([0, 0, -9.81]))

        # Disable all collisions
        self.model.geom_contype[:] = 0
        self.model.geom_conaffinity[:] = 0
        # Disable all joint limit
        self.model.jnt_limited[:] = 0

    @property
    def joint_limits(self) -> np.ndarray:
        return self.model.jnt_range

    def compute_inverse_dynamics(self, q: np.ndarray, qdot: np.ndarray, qdotdot: np.ndarray) -> np.ndarray:
        assert len(q) == len(qdot) == len(qdotdot)
        length = len(q)
        self.data.qpos[:length] = q
        self.data.qvel[:length] = qdot
        self.data.qacc[:length] = qdotdot
        mujoco.mj_inverse(self.model, self.data)
        return self.data.qfrc_inverse[:length]

    def set_gravity(self, gravity: np.ndarray) -> None:
        """Sets the gravity vector for the robot.

        Args:
            gravity (np.ndarray): The gravity vector as a NumPy array.

        """
        assert gravity.shape == (3,)
        self.model.opt.gravity = gravity

    def set_body_mass(self, body_name: str, mass: float) -> None:
        """Sets the mass of a body in the model.

        Args:
            body_name (str): The name of the body.
            mass (float): The new mass value in kg.
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"Body '{body_name}' not found in model")
        self.model.body_mass[body_id] = mass
