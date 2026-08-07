"""Viser control interface for i2rt robots.

Opens a browser-based 3-D viewer.  The robot stays in read-only mode until
the user confirms visual alignment and clicks "Enable".  Three control modes
are then available: mirror (VIS), IK drag, and per-joint sliders.

Usage:
    python examples/control_with_viser/control_with_viser.py --sim
    python examples/control_with_viser/control_with_viser.py --arm big_yam --gripper linear_4310 --sim
    python examples/control_with_viser/control_with_viser.py --arm yam_ultra_2 --sim
    python examples/control_with_viser/control_with_viser.py --channel can0
    python examples/control_with_viser/control_with_viser.py --channel can0 --record
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from typing import Optional

import tyro

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import ArmType, GripperType
from i2rt.utils.viser_control_interface import ViserControlInterface


def main(
    arm: str = "yam",
    gripper: str = "linear_4310",
    channel: str = "can0",
    sim: bool = False,
    dt: float = 0.02,
    port: int = 8080,
    site: Optional[str] = None,
    friction: bool = False,
    record: bool = False,
) -> None:
    """Viser control interface for i2rt robots.

    Args:
        arm: arm variant (yam, yam_pro, yam_ultra, yam_ultra_2, big_yam).
        gripper: gripper variant.
        channel: CAN channel.
        sim: Use SimRobot.
        dt: Loop timestep (s).
        port: Viser server port.
        site: EE site name (auto-detected if omitted).
        friction: Enable Coulomb friction compensation in gravity comp (real hardware only).
        record: Record motor feedback and computed required torques to a ROS 2 CDR MCAP file.
    """
    arm_type = ArmType.from_string_name(arm)
    gripper_type = GripperType.from_string_name(gripper)

    if site is None:
        site = "tcp_site" if gripper_type == GripperType.YAM_TEACHING_HANDLE else "grasp_site"
    if record and sim:
        raise SystemExit("--record requires real hardware motor feedback; remove --sim")

    robot = get_yam_robot(
        channel=channel,
        arm_type=arm_type,
        gripper_type=gripper_type,
        sim=sim,
        use_coulomb_friction=friction,
    )

    try:
        if record:
            print(f"Recording motor feedback to {robot.start_mcap_recording()}")
        iface = ViserControlInterface.from_robot(robot, ee_site=site, dt=dt, port=port)
        iface.run()
    finally:
        robot.close()


if __name__ == "__main__":
    tyro.cli(main)
