"""A reusable ROS 2 node that wraps a single I2RT YAM arm.

Responsibilities (component ①, "ROS2 wrapper"):

* publish ``<ns>/joint_states`` (``sensor_msgs/JointState``) at a fixed rate
* subscribe ``<ns>/command``     (``sensor_msgs/JointState``) -> ``command_joint_pos``
* (optional) publish ``<ns>/buttons`` (``sensor_msgs/Joy``) when the arm carries a
  teaching handle (passive encoder on the same CAN bus)

The node is intentionally thin: the real-time 250 Hz control loop lives inside
``MotorChainRobot`` already, so here we only sample state and forward targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import rclpy
from builtin_interfaces.msg import Time
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState, Joy

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import ArmType, GripperType
from i2rt.ros2 import ros_conversions as conv


@dataclass
class ArmConfig:
    """Configuration for one wrapped arm."""

    name: str  # ROS namespace / side label, e.g. "left", "right", "follower_left"
    channel: str = "can0"  # CAN channel (ignored when sim=True)
    arm_type: str = "yam"  # ArmType value
    gripper_type: str = "linear_4310"  # GripperType value
    sim: bool = False
    publish_rate_hz: float = 100.0
    zero_gravity_mode: bool = False  # followers start under PD control, not floppy

    def build_robot(self) -> Any:
        # match the follower's gravity-comp model used in teleop/DAgger (wrist payload)
        from i2rt.ros2.control_config import resolve_follower_ee

        ee_mass, ee_inertia = resolve_follower_ee(ArmType(self.arm_type), GripperType(self.gripper_type))
        return get_yam_robot(
            channel=self.channel,
            arm_type=ArmType(self.arm_type),
            gripper_type=GripperType(self.gripper_type),
            zero_gravity_mode=self.zero_gravity_mode,
            sim=self.sim,
            ee_mass=ee_mass,
            ee_inertia=ee_inertia,
        )


class YamArmNode(Node):
    """Wrap one YAM arm (robot already constructed) as a ROS 2 node.

    Topics are created relative to ``cfg.name`` so a bimanual setup yields, e.g.,
    ``/left/joint_states`` and ``/right/command``.
    """

    def __init__(self, cfg: ArmConfig, robot: Any = None, publish_command: bool = True):
        super().__init__(f"yam_{cfg.name}", namespace=cfg.name)
        self.cfg = cfg
        self.robot = robot if robot is not None else cfg.build_robot()
        # enforce the same global follow gain used in teleop/DAgger so replay matches
        from i2rt.ros2.control_config import apply_follower_gains

        apply_follower_gains(self.robot)
        self._names, self._has_gripper = conv.joint_names(self.robot)

        qos = 10
        self.state_pub = self.create_publisher(JointState, "joint_states", qos)

        # Teaching-handle buttons, if this arm has a passive encoder on its bus.
        self._buttons_pub = None
        if self._has_teaching_handle():
            self._buttons_pub = self.create_publisher(Joy, "buttons", qos)

        # Command subscription (followers consume this; leaders may ignore it).
        if publish_command:
            self.create_subscription(JointState, "command", self._on_command, qos)

        period = 1.0 / max(cfg.publish_rate_hz, 1.0)
        self.create_timer(period, self._publish_state)
        self.get_logger().info(
            f"YamArmNode '{cfg.name}' up: dofs={self.robot.num_dofs()} "
            f"gripper={self._has_gripper} handle={self._buttons_pub is not None} sim={cfg.sim}"
        )

    # ------------------------------------------------------------------ helpers
    def _has_teaching_handle(self) -> bool:
        mc = getattr(self.robot, "motor_chain", None)
        return (
            mc is not None
            and getattr(mc, "same_bus_device_driver", None) is not None
            and callable(getattr(mc, "get_same_bus_device_states", None))
        )

    def _stamp(self) -> Time:
        return self.get_clock().now().to_msg()

    # ------------------------------------------------------------------ ros cb
    def _publish_state(self) -> None:
        try:
            self.state_pub.publish(conv.robot_to_joint_state(self.robot, self._names, self._has_gripper, self._stamp()))
            if self._buttons_pub is not None:
                states = self.robot.motor_chain.get_same_bus_device_states()
                if states:
                    enc = states[0]
                    self._buttons_pub.publish(conv.buttons_to_joy(enc.io_inputs, enc.position, self._stamp()))
        except Exception as e:  # never let a transient read kill the timer
            self.get_logger().warn(f"state publish failed: {e}")

    def _on_command(self, msg: JointState) -> None:
        try:
            target = conv.joint_state_to_target(msg, self.robot)
        except ValueError as e:
            self.get_logger().warn(f"ignoring command: {e}")
            return
        self.robot.command_joint_pos(target)

    # ------------------------------------------------------------------ cleanup
    def destroy_node(self) -> bool:
        try:
            if hasattr(self.robot, "close"):
                self.robot.close()
        except Exception:
            pass
        return super().destroy_node()


def main(argv: Optional[List[str]] = None) -> None:
    """Run a single wrapped arm (handy for one-arm testing)."""
    import argparse

    p = argparse.ArgumentParser(description="ROS 2 wrapper for a single YAM arm")
    p.add_argument("--name", default="arm")
    p.add_argument("--channel", default="can0")
    p.add_argument("--arm", default="yam")
    p.add_argument("--gripper", default="linear_4310")
    p.add_argument("--sim", action="store_true")
    p.add_argument("--rate", type=float, default=100.0)
    args = p.parse_args(argv)

    rclpy.init()
    node = YamArmNode(
        ArmConfig(
            name=args.name,
            channel=args.channel,
            arm_type=args.arm,
            gripper_type=args.gripper,
            sim=args.sim,
            publish_rate_hz=args.rate,
        )
    )
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
