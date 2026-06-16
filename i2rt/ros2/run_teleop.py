"""Component ② — bimanual leader-follower teleoperation with ROS 2 publishing.

Activates both leader+follower pairs at once, runs the bilateral teleop loop
in-process, and publishes the relevant streams so a workstation can record
demonstrations.

Per side ``<s>`` in {left, right}:

* pub  ``<s>/leader/joint_states``   ``sensor_msgs/JointState``
* pub  ``<s>/follower/joint_states`` ``sensor_msgs/JointState``
* pub  ``<s>/buttons``               ``sensor_msgs/Joy``  (teaching-handle buttons)
* pub  ``<s>/sync``                  ``std_msgs/Bool``    (sync engaged?)
* sub  ``<s>/sync_cmd``              ``std_msgs/Bool``    (external sync override; for sim/no-handle)

Sync toggles on the handle's top button (edge-triggered), exactly like
``examples/minimum_gello``. While synced, the follower tracks the leader and the
leader is back-driven (``--bilateral-kp``) so the human feels contact forces.

Safety: the follower target is rate-limited (:class:`~i2rt.ros2.safety.TargetSmoother`),
so engaging sync from a divergent pose ramps smoothly instead of snapping — the
safety equivalent of ``minimum_gello``'s ``slow_move``. A dropped/invalid leader
reading holds position instead of commanding garbage.

Examples::

    python -m i2rt.ros2.run_teleop --sim
    python -m i2rt.ros2.run_teleop --bilateral-kp 0.2          # real bimanual hardware
"""

from __future__ import annotations

import argparse
from typing import Callable, List, Optional

import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState, Joy
from std_msgs.msg import Bool

from i2rt.ros2 import ros_conversions as conv
from i2rt.ros2.safety import TargetSmoother, is_finite_vector, max_step_from_speed
from i2rt.ros2.teleop_common import (
    ArmPair,
    LatchingToggle,
    build_bimanual,
    build_follower_target,
    default_bimanual_specs,
    read_handle,
)


class TeleopNode(Node):
    def __init__(self, sim: bool, bilateral_kp: float, rate_hz: float, max_joint_speed: float):
        super().__init__("yam_teleop")
        self.bilateral_kp = bilateral_kp
        self.pairs = build_bimanual(default_bimanual_specs(sim), sim=sim)
        max_step = max_step_from_speed(max_joint_speed, rate_hz)

        self._names = {}
        self._toggle = {}
        self._smooth = {}
        self._pub = {}
        for side, pair in self.pairs.items():
            ln, lg = conv.joint_names(pair.leader)
            fn, fg = conv.joint_names(pair.follower)
            self._names[side] = (ln, lg, fn, fg)
            self._toggle[side] = LatchingToggle(initial=False)
            self._smooth[side] = TargetSmoother(pair.follower.get_joint_pos(), max_step)
            self._pub[side] = {
                "leader": self.create_publisher(JointState, f"{side}/leader/joint_states", 10),
                "follower": self.create_publisher(JointState, f"{side}/follower/joint_states", 10),
                "buttons": self.create_publisher(Joy, f"{side}/buttons", 10),
                "sync": self.create_publisher(Bool, f"{side}/sync", 10),
            }
            self.create_subscription(Bool, f"{side}/sync_cmd", self._make_sync_cmd(side), 10)

        self.create_timer(1.0 / max(rate_hz, 1.0), self._loop)
        self.get_logger().info(
            f"TeleopNode up: sides={list(self.pairs)} bilateral_kp={bilateral_kp} "
            f"max_joint_speed={max_joint_speed} rad/s sim={sim}"
        )

    def _make_sync_cmd(self, side: str) -> Callable[[Bool], None]:
        def cb(msg: Bool) -> None:
            self._toggle[side].state = bool(msg.data)  # external override (level)

        return cb

    def _stamp(self) -> Time:
        return self.get_clock().now().to_msg()

    def _loop(self) -> None:
        for side, pair in self.pairs.items():
            ln, lg, fn, fg = self._names[side]
            n = pair.follower.num_dofs()
            smoother = self._smooth[side]
            try:
                arm, grip, buttons = read_handle(pair.leader)
                pressed = bool(buttons[0]) if buttons else False
                synced = self._toggle[side].update(pressed)

                desired = build_follower_target(pair.follower, arm, grip)
                if synced and is_finite_vector(desired, n):
                    target = smoother.step(desired)  # rate-limited follower tracking
                    pair.follower.command_joint_pos(target)
                    self._drive_leader(pair, np.asarray(pair.follower.get_joint_pos())[: pair.leader.num_dofs()])
                else:
                    smoother.reset(pair.follower.get_joint_pos())  # hold when not synced

                stamp = self._stamp()
                self._pub[side]["leader"].publish(conv.robot_to_joint_state(pair.leader, ln, lg, stamp))
                self._pub[side]["follower"].publish(conv.robot_to_joint_state(pair.follower, fn, fg, stamp))
                self._pub[side]["buttons"].publish(conv.buttons_to_joy(buttons, grip if grip is not None else 0.0, stamp))
                self._pub[side]["sync"].publish(Bool(data=bool(synced)))
            except Exception as e:
                self.get_logger().warn(f"[{side}] teleop step failed: {e}")

    def _drive_leader(self, pair: ArmPair, target_q: np.ndarray) -> None:
        """Back-drive the leader toward ``target_q`` so the human feels follower contact forces."""
        leader = pair.leader
        if self.bilateral_kp <= 0.0 or not hasattr(leader, "update_kp_kd") or pair.base_kp is None:
            return
        try:
            m = leader.num_dofs()
            leader.update_kp_kd(pair.base_kp[:m] * self.bilateral_kp, np.zeros(m))
            leader.command_joint_pos(np.asarray(target_q, dtype=float)[:m])
        except Exception:
            pass

    def destroy_node(self) -> bool:
        for pair in self.pairs.values():
            for r in (pair.leader, pair.follower):
                try:
                    r.close()
                except Exception:
                    pass
        return super().destroy_node()


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Bimanual leader-follower teleop with ROS 2 publishing")
    p.add_argument("--sim", action="store_true")
    p.add_argument("--bilateral-kp", type=float, default=0.0, help="leader back-drive stiffness (0.1-0.2 typical)")
    p.add_argument("--rate", type=float, default=120.0)
    p.add_argument("--max-joint-speed", type=float, default=3.0, help="rad/s cap on follower target ramp (safety)")
    args = p.parse_args(argv)

    rclpy.init()
    node = TeleopNode(
        sim=args.sim, bilateral_kp=args.bilateral_kp, rate_hz=args.rate, max_joint_speed=args.max_joint_speed
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
