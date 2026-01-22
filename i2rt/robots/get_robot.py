import logging
import os
import time
from functools import partial
from typing import Any, Callable, Optional

import numpy as np

from i2rt.motor_drivers.dm_driver import (
    CanInterface,
    DMChainCanInterface,
    EncoderChain,
    PassiveEncoderReader,
    ReceiveMode,
)
from i2rt.robots.motor_chain_robot import MotorChainRobot
from i2rt.robots.utils import I2RT_ROOT, GripperType


def get_encoder_chain(can_interface: CanInterface) -> EncoderChain:
    passive_encoder_reader = PassiveEncoderReader(can_interface)
    return EncoderChain([0x50E], passive_encoder_reader)


def get_big_yam_robot(
    channel: str = "can0",
    gripper_type: GripperType = GripperType.LINEAR_4310,
    zero_gravity_mode: bool = True,
    joint_state_saver_factory: Optional[Callable[[str], Any]] = None,
    set_realtime_and_pin_callback: Optional[Callable[[int], None]] = None,
) -> MotorChainRobot:
    xml_path = os.path.join(I2RT_ROOT, "robot_models/big_yam/big_yam.xml")
    motor_list = [
        [0x01, "DM6248"],
        [0x02, "DM6248"],
        [0x03, "DM4340"],
        [0x04, "DM4340"],
        [0x05, "DM4310"],
        [0x06, "DM4310"],
        [0x07, "DM4310"],
    ]
    motor_offsets = [0, 0, 0, 0, 0, 0, 0]
    motor_directions = [1, -1, 1, 1, 1, 1, 1]
    kp = np.array([80, 80, 80, 40, 40, 10, 10])
    kd = np.array([5, 5, 5, 3, 3, 3, 0.5])

    motor_chain = DMChainCanInterface(
        motor_list,
        motor_offsets,
        motor_directions,
        channel,
        motor_chain_name="yam_real",
        receive_mode=ReceiveMode.p16,
    )
    motor_states = motor_chain.read_states()
    logging.info(f"YAM initial motor_states: {motor_states}")
    get_robot = partial(
        MotorChainRobot,
        motor_chain=motor_chain,
        xml_path=xml_path,
        use_gravity_comp=True,
        gravity_comp_factor=1.0,
        kp=kp,
        kd=kd,
        zero_gravity_mode=zero_gravity_mode,
        joint_state_saver_factory=joint_state_saver_factory,
        set_realtime_and_pin_callback=set_realtime_and_pin_callback,
        gripper_index=6,
        gripper_limits=gripper_type.get_gripper_limits(),
        gripper_type=gripper_type,
        limit_gripper_force=50.0,
    )
    return get_robot()


def get_yam_robot(
    channel: str = "can0",
    gripper_type: GripperType = GripperType.CRANK_4310,
    zero_gravity_mode: bool = True,
    joint_state_saver_factory: Optional[Callable[[str], Any]] = None,
    set_realtime_and_pin_callback: Optional[Callable[[int], None]] = None,
) -> MotorChainRobot:
    with_gripper = True
    with_teaching_handle = False
    if gripper_type == GripperType.YAM_TEACHING_HANDLE:
        with_gripper = False
        with_teaching_handle = True
    if gripper_type == GripperType.NO_GRIPPER:
        with_gripper = False
        with_teaching_handle = False

    model_path = gripper_type.get_xml_path()
    motor_list = [
        [0x01, "DM4340"],
        [0x02, "DM4340"],
        [0x03, "DM4340"],
        [0x04, "DM4310"],
        [0x05, "DM4310"],
        [0x06, "DM4310"],
    ]
    motor_offsets = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    motor_directions = [1, 1, 1, 1, 1, 1]
    joint_limits = np.array([[-2.617, 3.13], [0, 3.65], [0.0, 3.13], [-1.57, 1.57], [-1.57, 1.57], [-2.09, 2.09]])
    joint_limits[:, 0] += -0.15  # add some buffer to the joint limits
    joint_limits[:, 1] += 0.15

    kp = np.array([80, 80, 80, 40, 10, 10])
    kd = np.array([5, 5, 5, 1.5, 1.5, 1.5])
    if with_gripper:
        motor_type = gripper_type.get_motor_type()
        gripper_kp, gripper_kd = gripper_type.get_motor_kp_kd()
        assert motor_type != ""
        logging.info(
            f"adding gripper motor with type: {motor_type}, gripper_kp: {gripper_kp}, gripper_kd: {gripper_kd}"
        )
        motor_list.append([0x07, motor_type])
        motor_offsets.append(0.0)
        motor_directions.append(1)
        kp = np.concatenate([kp, np.array([gripper_kp])])
        kd = np.concatenate([kd, np.array([gripper_kd])])

    motor_chain = DMChainCanInterface(
        motor_list,
        motor_offsets,
        motor_directions,
        channel,
        motor_chain_name="yam_real",
        receive_mode=ReceiveMode.p16,
        get_same_bus_device_driver=get_encoder_chain if with_teaching_handle else None,
        use_buffered_reader=False,
    )
    motor_states = motor_chain.read_states()
    logging.info(f"YAM initial motor_states: {motor_states}")
    get_robot = partial(
        MotorChainRobot,
        motor_chain=motor_chain,
        xml_path=model_path,
        use_gravity_comp=True,
        gravity_comp_factor=1.3,
        joint_limits=joint_limits,
        kp=kp,
        kd=kd,
        zero_gravity_mode=zero_gravity_mode,
        joint_state_saver_factory=joint_state_saver_factory,
        set_realtime_and_pin_callback=set_realtime_and_pin_callback,
    )

    if with_gripper:
        return get_robot(
            gripper_index=6,
            gripper_limits=gripper_type.get_gripper_limits(),
            gripper_type=gripper_type,
            limit_gripper_force=50.0,
        )
    else:
        return get_robot()


if __name__ == "__main__":
    robot = get_yam_robot(channel="can0", gripper_type=GripperType.LINEAR_4310)
    while True:
        time.sleep(1)
        print(robot.get_observations())
