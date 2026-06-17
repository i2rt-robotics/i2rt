"""Robot state helpers (transport-agnostic; no ROS).

Extract full pos/vel/eff vectors (length ``num_dofs``, trailing gripper element
when present) and turn an external position command into a full-length target.
Replaces the pos/vel/eff packing that used to live in ``ros_conversions``.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np


def _full_vector(arm: Any, grip: Any, has_gripper: bool, n: int) -> np.ndarray:
    arm = np.asarray(arm, dtype=float).reshape(-1)
    if not has_gripper:
        return arm if arm.size == n else np.zeros(n)
    g = np.asarray(grip, dtype=float).reshape(-1)
    g = g[:1] if g.size else np.zeros(1)
    out = np.concatenate([arm, g])
    return out if out.size == n else np.zeros(n)


def full_state(robot: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(pos, vel, eff)`` full vectors (length ``num_dofs``) for a robot."""
    n = int(robot.num_dofs())
    obs = robot.get_observations()
    has_grip = "gripper_pos" in obs

    pos = np.asarray(robot.get_joint_pos(), dtype=float).reshape(-1)
    if pos.size != n:
        pos = _full_vector(obs.get("joint_pos", []), obs.get("gripper_pos", [0.0]), has_grip, n)
    vel = _full_vector(obs.get("joint_vel", []), obs.get("gripper_vel", [0.0]), has_grip, n)
    eff = _full_vector(obs.get("joint_eff", []), obs.get("gripper_eff", [0.0]), has_grip, n)
    return pos, vel, eff


def ee_pose(robot: Any) -> Optional[np.ndarray]:
    """Best-effort end-effector pose for the robot, or None if it doesn't expose one.

    Looks for a cartesian field in ``get_observations()`` or a forward-kinematics
    method on the robot. Returns a flat float vector (e.g. ``[x, y, z, qw, qx, qy, qz]``)
    so it can be recorded as ``observation.eef`` and used for EEF-space control.
    """
    try:
        obs = robot.get_observations()
        for k in ("ee_pose", "eef_pose", "ee_pos", "cartesian", "tcp_pose"):
            if k in obs and obs[k] is not None:
                return np.asarray(obs[k], dtype=float).reshape(-1)
    except Exception:
        pass
    for m in ("get_ee_pose", "get_eef_pose", "get_tcp_pose", "forward_kinematics"):
        fn = getattr(robot, m, None)
        if callable(fn):
            try:
                return np.asarray(fn(), dtype=float).reshape(-1)
            except Exception:
                pass
    return None


def to_full_target(position: np.ndarray, robot: Any) -> np.ndarray:
    """Turn an external position command into a full-length (``num_dofs``) target.

    Accepts the full vector or an arm-only vector (``num_dofs - 1``); in the latter
    case the robot's current gripper position is kept. Raises ``ValueError`` on a
    mismatch.
    """
    n = int(robot.num_dofs())
    target = np.asarray(position, dtype=float).reshape(-1)
    if target.size == n:
        return target
    if target.size == n - 1:
        full = np.asarray(robot.get_joint_pos(), dtype=float).reshape(-1).copy()
        full[: n - 1] = target
        return full
    raise ValueError(f"command has {target.size} positions, expected {n} or {n - 1}")
