"""The gripper force limit: what ``get_yam_robot`` builds with, and changing it on a live robot.

Once the fingers are blocked, ``GripperForceLimiter`` holds them at the torque its force-torque map gives
for ``max_force``. ``MotorChainRobot.set_gripper_force_limit`` has to move that map and the robot's own
``_limit_gripper_force`` (reported as ``limit_gripper_effort``) together.
"""

import inspect
import threading
from typing import Any, cast

import numpy as np
import pytest

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.motor_chain_robot import MotorChainRobot
from i2rt.robots.utils import ArmType, GripperForceLimiter, GripperType

GRIPPER_INDEX = 6


def _bare_robot(limit_gripper_force: float = 50.0) -> Any:
    """A MotorChainRobot with only the gripper-limit attributes.

    Built with ``__new__`` rather than a real chain, following the idiom in
    ``test_motor_chain_recording.py``: the constructor opens a CAN socket.
    """
    robot = cast(Any, MotorChainRobot.__new__(MotorChainRobot))
    robot.motor_chain = [None] * (GRIPPER_INDEX + 1)
    robot._arm_type = ArmType.YAM
    robot._gripper_type = GripperType.LINEAR_4310
    robot._gripper_index = GRIPPER_INDEX
    robot._gripper_force_limiter = GripperForceLimiter(
        max_force=limit_gripper_force, gripper_type=GripperType.LINEAR_4310, arm_type=ArmType.YAM, kp=20.0
    )
    robot._limit_gripper_force = limit_gripper_force
    robot._state_lock = threading.Lock()
    return robot


def _held_torque(robot: Any) -> float:
    return robot._gripper_force_limiter.gripper_force_torque_map(current_angle=0.0)


def test_get_yam_robot_keeps_50_newtons_by_default() -> None:
    assert inspect.signature(get_yam_robot).parameters["limit_gripper_force"].default == 50.0


def test_set_gripper_force_limit_moves_the_held_torque_and_the_reported_limit() -> None:
    robot = _bare_robot(50.0)
    torque_at_50 = _held_torque(robot)

    assert robot.set_gripper_force_limit(12.0) == 50.0

    assert robot._gripper_force_limiter.max_force == 12.0
    assert robot._limit_gripper_force == 12.0
    # linear_4310's map is linear in force, so the held torque scales with it.
    assert _held_torque(robot) == pytest.approx(torque_at_50 * 12.0 / 50.0)
    assert robot.set_gripper_force_limit(30.0) == 12.0


@pytest.mark.parametrize("force", [0.0, -5.0, float("nan"), float("inf")])
def test_set_gripper_force_limit_rejects_a_bad_force(force: float) -> None:
    robot = _bare_robot(50.0)
    torque_at_50 = _held_torque(robot)

    with pytest.raises(ValueError, match="finite and positive"):
        robot.set_gripper_force_limit(force)

    assert robot._limit_gripper_force == 50.0
    assert _held_torque(robot) == torque_at_50


def test_set_gripper_force_limit_refuses_a_robot_built_without_a_limit() -> None:
    robot = _bare_robot(-1.0)

    with pytest.raises(RuntimeError, match="no gripper force limit"):
        robot.set_gripper_force_limit(12.0)

    assert robot._limit_gripper_force == -1.0


def test_set_gripper_force_limit_refuses_a_robot_without_a_gripper() -> None:
    robot = _bare_robot()
    robot._gripper_index = None
    robot._gripper_force_limiter = None
    robot._limit_gripper_force = -1.0

    with pytest.raises(RuntimeError, match="no gripper force limit"):
        robot.set_gripper_force_limit(12.0)


@pytest.mark.parametrize("gripper_type", [GripperType.LINEAR_4310, GripperType.CRANK_4310])
def test_limiter_set_max_force_matches_a_limiter_built_with_that_force(gripper_type: GripperType) -> None:
    changed = GripperForceLimiter(max_force=50.0, gripper_type=gripper_type, arm_type=ArmType.YAM, kp=20.0)
    changed.set_max_force(12.0)
    built = GripperForceLimiter(max_force=12.0, gripper_type=gripper_type, arm_type=ArmType.YAM, kp=20.0)

    for angle in np.linspace(0.0, 1.0, 5):
        assert changed.gripper_force_torque_map(current_angle=angle) == built.gripper_force_torque_map(
            current_angle=angle
        )
