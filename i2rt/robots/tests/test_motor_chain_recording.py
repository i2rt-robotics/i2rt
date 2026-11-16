from types import SimpleNamespace
from typing import Any, cast

from i2rt.motor_drivers.dm_driver import MotorInfo
from i2rt.robots.motor_chain_robot import MotorChainRobot


def test_motor_feedback_joint_names_include_gripper() -> None:
    robot = cast(Any, MotorChainRobot.__new__(MotorChainRobot))
    robot._gripper_index = 6
    robot.remapper = SimpleNamespace(
        to_command_joint_pos_space=lambda values: values,
        to_command_joint_vel_space=lambda values: values,
    )
    motor_state = [MotorInfo(id=index + 1, error_code=1) for index in range(7)]

    joint_state = robot._motor_state_to_joint_state(motor_state)

    assert joint_state.names == ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
