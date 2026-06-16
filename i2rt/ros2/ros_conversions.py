"""Conversions between I2RT robot state and standard ROS 2 messages.

Everything here uses only pre-built standard messages so the package needs no
build step:

* joint state / command  -> ``sensor_msgs/JointState``
* teaching-handle buttons -> ``sensor_msgs/Joy``

The joint vector layout is ``[joint_1 .. joint_N, gripper]`` where the trailing
``gripper`` entry is present only when the robot exposes a gripper. The gripper
value is the normalized ``[0, 1]`` position used everywhere in i2rt
(0 = closed, 1 = open).
"""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
from sensor_msgs.msg import JointState, Joy


def joint_names(robot: Any) -> Tuple[List[str], bool]:
    """Return ``(names, has_gripper)`` for the robot's full joint vector."""
    n = int(robot.num_dofs())
    has_gripper = "gripper_pos" in robot.get_observations()
    n_arm = n - 1 if has_gripper else n
    names = [f"joint_{i + 1}" for i in range(n_arm)]
    if has_gripper:
        names.append("gripper")
    return names, has_gripper


def _full_vector(arm: np.ndarray, grip: Any, has_gripper: bool, n: int) -> np.ndarray:
    """Concatenate an arm vector with the (optional) gripper scalar."""
    arm = np.asarray(arm, dtype=float).reshape(-1)
    if not has_gripper:
        return arm if arm.size == n else np.zeros(n)
    g = np.asarray(grip, dtype=float).reshape(-1)
    g = g[:1] if g.size else np.zeros(1)
    out = np.concatenate([arm, g])
    return out if out.size == n else np.zeros(n)


def robot_to_joint_state(robot: Any, names: List[str], has_gripper: bool, stamp: Any) -> JointState:
    """Build a ``JointState`` snapshot of the robot.

    ``stamp`` is a ``builtin_interfaces/Time`` (e.g. ``node.get_clock().now().to_msg()``).
    """
    n = len(names)
    obs = robot.get_observations()

    pos = np.asarray(robot.get_joint_pos(), dtype=float).reshape(-1)
    if pos.size != n:
        pos = _full_vector(obs.get("joint_pos", []), obs.get("gripper_pos", [0.0]), has_gripper, n)

    vel = _full_vector(obs.get("joint_vel", []), obs.get("gripper_vel", [0.0]), has_gripper, n)
    eff = _full_vector(obs.get("joint_eff", []), obs.get("gripper_eff", [0.0]), has_gripper, n)

    msg = JointState()
    msg.header.stamp = stamp
    msg.name = list(names)
    msg.position = pos.tolist()
    msg.velocity = vel.tolist()
    msg.effort = eff.tolist()
    return msg


def joint_state_to_target(msg: JointState, robot: Any) -> np.ndarray:
    """Extract a full-length joint-position target from a command ``JointState``.

    Accepts either the full vector (``num_dofs``) or an arm-only vector
    (``num_dofs - 1``); in the latter case the current gripper position is kept.
    Returns an array of length ``num_dofs``. Raises ``ValueError`` on mismatch.
    """
    n = int(robot.num_dofs())
    target = np.asarray(msg.position, dtype=float).reshape(-1)
    if target.size == n:
        return target
    if target.size == n - 1:
        full = np.asarray(robot.get_joint_pos(), dtype=float).reshape(-1).copy()
        full[: n - 1] = target
        return full
    raise ValueError(f"command has {target.size} positions, expected {n} or {n - 1}")


def buttons_to_joy(io_inputs: Any, gripper_pos: float, stamp: Any) -> Joy:
    """Pack teaching-handle button/trigger state into a ``Joy`` message.

    * ``buttons`` = the discrete IO inputs (0/1), as reported by the passive encoder.
    * ``axes``    = ``[gripper_pos]`` (the normalized trigger/gripper reading).
    """
    msg = Joy()
    msg.header.stamp = stamp
    msg.buttons = [int(bool(b)) for b in (io_inputs or [])]
    msg.axes = [float(gripper_pos)]
    return msg
