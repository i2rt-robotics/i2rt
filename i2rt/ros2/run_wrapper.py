"""Component ① — ROS 2 wrapper runner (bimanual by default).

Spins one ``YamArmNode`` per follower arm inside a single process using a
``MultiThreadedExecutor``. Each arm publishes its state on
``/<name>/joint_states`` and accepts targets on ``/<name>/command``.

Examples::

    # Two simulated followers (no hardware needed):
    python -m i2rt.ros2.run_wrapper --sim

    # Real bimanual hardware:
    python -m i2rt.ros2.run_wrapper \
        --arm left:can_follower_l --arm right:can_follower_r --gripper linear_4310

    # Single arm:
    python -m i2rt.ros2.run_wrapper --arm follower:can0
"""

from __future__ import annotations

import argparse
from typing import List, Optional

import rclpy
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor

from i2rt.ros2.yam_arm_node import ArmConfig, YamArmNode


def _parse_arm_specs(specs: List[str], default_channel: str) -> List[tuple]:
    """Parse ``name:channel`` specs into ``(name, channel)`` tuples."""
    out = []
    for s in specs:
        if ":" in s:
            name, channel = s.split(":", 1)
        else:
            name, channel = s, default_channel
        out.append((name, channel))
    return out


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="ROS 2 wrapper for one or more YAM follower arms")
    p.add_argument(
        "--arm",
        action="append",
        dest="arms",
        metavar="NAME:CHANNEL",
        help="Arm spec 'name:can_channel' (repeatable). Default: bimanual left/right.",
    )
    p.add_argument("--arm-type", default="yam")
    p.add_argument("--gripper", default="linear_4310")
    p.add_argument("--rate", type=float, default=100.0)
    p.add_argument("--sim", action="store_true", help="Use SimRobot (no CAN hardware)")
    args = p.parse_args(argv)

    specs = args.arms or ["left:can_follower_l", "right:can_follower_r"]
    arms = _parse_arm_specs(specs, default_channel="can0")

    rclpy.init()
    executor = MultiThreadedExecutor()
    nodes = []
    for name, channel in arms:
        cfg = ArmConfig(
            name=name,
            channel=channel,
            arm_type=args.arm_type,
            gripper_type=args.gripper,
            sim=args.sim,
            publish_rate_hz=args.rate,
            zero_gravity_mode=False,
        )
        node = YamArmNode(cfg)
        nodes.append(node)
        executor.add_node(node)

    print(f"[run_wrapper] {len(nodes)} arm(s): {', '.join(n.cfg.name for n in nodes)} (sim={args.sim})")
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        for node in nodes:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
