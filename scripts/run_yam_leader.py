import logging
import os
import time
from typing import Literal

import tyro

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import GripperType

log_level = os.environ.get("LOGLEVEL", "INFO")
logging.basicConfig(level=log_level)


def main(
    channel: str = "can0",
    gripper: Literal["yam_teaching_handle_v1", "yam_teaching_handle_v2"] = "yam_teaching_handle_v1",
) -> None:
    """Print a YAM leader arm's joint positions and its teaching handle's encoder state."""
    robot = get_yam_robot(
        channel=channel,
        gripper_type=GripperType.from_string_name(gripper),
    )
    motor_chain = robot.motor_chain

    while True:
        obs = robot.get_observations()
        encoder_obs = motor_chain.get_same_bus_device_states()
        qpos = obs["joint_pos"]
        print(qpos)
        print(encoder_obs)
        time.sleep(0.01)


if __name__ == "__main__":
    tyro.cli(main)
