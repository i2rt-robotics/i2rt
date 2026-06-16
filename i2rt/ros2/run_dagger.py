"""Component ③ — HG-DAgger interactive takeover (bimanual, single gate).

Normal operation: a policy (on the workstation) publishes joint-position actions
for each follower. This node applies them to the followers and **back-drives the
leaders** so a human resting on the handles feels what the policy is doing.

Intervention: when the human presses the gate button (either leader's top button,
or an external ``/dagger/intervention_cmd`` Bool), the node switches **both** arms
to human control — each follower tracks its leader — and streams the human action
so the workstation can aggregate ``(observation, human_action)`` pairs. Releasing
the button hands control back to the policy.

Safety: every follower target passes through a per-joint rate limiter
(:class:`~i2rt.ros2.safety.TargetSmoother`), so switching between policy and human,
a stale→fresh policy action, or a dropped reading can never snap the arm at unsafe
speed. Gravity compensation is always active on every arm (it is added inside
``MotorChainRobot``'s control loop), so the leader is freely back-drivable during a
takeover. If a leader reading is missing/invalid while intervening, that arm holds
position instead of commanding garbage.

Topics (per side ``<s>`` in {left, right}):

* sub ``<s>/policy_action``        ``sensor_msgs/JointState`` (from the policy)
* pub ``<s>/follower/joint_states`` ``sensor_msgs/JointState``
* pub ``<s>/leader/joint_states``   ``sensor_msgs/JointState``
* pub ``<s>/applied_action``        ``sensor_msgs/JointState`` (what was actually sent)
* pub ``<s>/human_action``          ``sensor_msgs/JointState`` (leader q; valid while intervening)
* pub ``<s>/buttons``               ``sensor_msgs/Joy``

Global:

* pub ``/dagger/intervention``      ``std_msgs/Bool`` (gate state; both arms)
* sub ``/dagger/intervention_cmd``  ``std_msgs/Bool`` (external gate; for sim/no-handle)

Examples::

    python -m i2rt.ros2.run_dagger --sim
    python -m i2rt.ros2.run_dagger --bilateral-kp 0.15        # real bimanual hardware
"""

from __future__ import annotations

import argparse
from typing import Callable, Dict, List, Optional

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
    build_bimanual,
    build_follower_target,
    default_bimanual_specs,
    read_handle,
)


class DaggerNode(Node):
    def __init__(self, sim: bool, bilateral_kp: float, rate_hz: float, max_joint_speed: float):
        super().__init__("yam_dagger")
        self.bilateral_kp = bilateral_kp
        self.pairs = build_bimanual(default_bimanual_specs(sim), sim=sim)
        max_step = max_step_from_speed(max_joint_speed, rate_hz)

        self._intervening = False
        self._ext_gate: Optional[bool] = None  # external override; None = use handle button
        self._policy_action: Dict[str, Optional[np.ndarray]] = {s: None for s in self.pairs}
        self._names = {}
        self._smooth = {}
        self._pub = {}

        for side, pair in self.pairs.items():
            ln, lg = conv.joint_names(pair.leader)
            fn, fg = conv.joint_names(pair.follower)
            self._names[side] = (ln, lg, fn, fg)
            self._smooth[side] = TargetSmoother(pair.follower.get_joint_pos(), max_step)
            self._pub[side] = {
                "follower": self.create_publisher(JointState, f"{side}/follower/joint_states", 10),
                "leader": self.create_publisher(JointState, f"{side}/leader/joint_states", 10),
                "applied": self.create_publisher(JointState, f"{side}/applied_action", 10),
                "human": self.create_publisher(JointState, f"{side}/human_action", 10),
                "buttons": self.create_publisher(Joy, f"{side}/buttons", 10),
            }
            self.create_subscription(JointState, f"{side}/policy_action", self._make_policy_cb(side), 10)

        self.intervention_pub = self.create_publisher(Bool, "/dagger/intervention", 10)
        self.create_subscription(Bool, "/dagger/intervention_cmd", self._on_gate_cmd, 10)

        self.create_timer(1.0 / max(rate_hz, 1.0), self._loop)
        self.get_logger().info(
            f"DaggerNode up: sides={list(self.pairs)} bilateral_kp={bilateral_kp} "
            f"max_joint_speed={max_joint_speed} rad/s sim={sim}"
        )

    # ---------------------------------------------------------------- callbacks
    def _make_policy_cb(self, side: str) -> Callable[[JointState], None]:
        def cb(msg: JointState) -> None:
            try:
                self._policy_action[side] = conv.joint_state_to_target(msg, self.pairs[side].follower)
            except ValueError as e:
                self.get_logger().warn(f"[{side}] bad policy_action: {e}")

        return cb

    def _on_gate_cmd(self, msg: Bool) -> None:
        self._ext_gate = bool(msg.data)

    def _stamp(self) -> Time:
        return self.get_clock().now().to_msg()

    def _gate_pressed(self, buttons_by_side: Dict[str, list]) -> bool:
        """Single gate for both arms: external override wins, else OR of handle buttons."""
        if self._ext_gate is not None:
            return self._ext_gate
        return any(bool(b[0]) for b in buttons_by_side.values() if b)

    # --------------------------------------------------------------------- loop
    def _loop(self) -> None:
        # 1) read all leaders first so the gate sees every handle this tick
        arm_q, grip_cmd, buttons, valid = {}, {}, {}, {}
        for side, pair in self.pairs.items():
            try:
                a, g, b = read_handle(pair.leader)
            except Exception as e:
                self.get_logger().warn(f"[{side}] handle read failed: {e}")
                a, g, b = np.zeros(pair.leader.num_dofs()), None, []
            arm_q[side], grip_cmd[side], buttons[side] = a, g, b
            valid[side] = is_finite_vector(a, pair.leader.num_dofs())

        self._intervening = self._gate_pressed(buttons)

        # 2) act + publish per side
        for side, pair in self.pairs.items():
            ln, lg, fn, fg = self._names[side]
            n = pair.follower.num_dofs()
            smoother = self._smooth[side]
            applied = None
            human = None
            try:
                desired = None
                if self._intervening:
                    # human takes over: follower tracks the leader (only if the read is valid)
                    if valid[side]:
                        human = build_follower_target(pair.follower, arm_q[side], grip_cmd[side])
                        desired = human
                        # bilateral feel: back-drive leader toward the follower's current pose
                        self._drive_leader(pair, np.asarray(pair.follower.get_joint_pos())[: pair.leader.num_dofs()])
                else:
                    # policy drives follower; leader mirrors the policy action so the human feels
                    # what the policy intends and is positioned for a smooth takeover
                    act = self._policy_action[side]
                    if is_finite_vector(act, n):
                        desired = act[:n]
                        self._drive_leader(pair, act[: pair.leader.num_dofs()])

                if desired is not None:
                    target = smoother.step(desired)  # rate-limited → never snaps
                    pair.follower.command_joint_pos(target)
                    applied = target
                else:
                    # nothing valid to do this tick: hold position safely
                    smoother.reset(pair.follower.get_joint_pos())

                self._publish(side, pair, ln, lg, fn, fg, buttons[side], grip_cmd[side], applied, human)
            except Exception as e:
                self.get_logger().warn(f"[{side}] dagger step failed: {e}")

        self.intervention_pub.publish(Bool(data=bool(self._intervening)))

    def _drive_leader(self, pair: ArmPair, target_q: np.ndarray) -> None:
        """Back-drive the leader toward ``target_q`` at low stiffness (gravity comp stays on)."""
        leader = pair.leader
        if self.bilateral_kp <= 0.0 or not hasattr(leader, "update_kp_kd") or pair.base_kp is None:
            return
        try:
            m = leader.num_dofs()
            leader.update_kp_kd(pair.base_kp[:m] * self.bilateral_kp, np.zeros(m))
            leader.command_joint_pos(np.asarray(target_q, dtype=float)[:m])
        except Exception:
            pass

    def _publish(
        self,
        side: str,
        pair: ArmPair,
        ln: List[str],
        lg: bool,
        fn: List[str],
        fg: bool,
        buttons: List[int],
        grip_cmd: Optional[float],
        applied: Optional[np.ndarray],
        human: Optional[np.ndarray],
    ) -> None:
        stamp = self._stamp()
        self._pub[side]["follower"].publish(conv.robot_to_joint_state(pair.follower, fn, fg, stamp))
        self._pub[side]["leader"].publish(conv.robot_to_joint_state(pair.leader, ln, lg, stamp))
        self._pub[side]["buttons"].publish(conv.buttons_to_joy(buttons, grip_cmd if grip_cmd is not None else 0.0, stamp))
        if human is not None:
            hmsg = JointState()
            hmsg.header.stamp = stamp
            hmsg.name = fn
            hmsg.position = np.asarray(human, dtype=float).tolist()
            self._pub[side]["human"].publish(hmsg)
        if applied is not None:
            amsg = JointState()
            amsg.header.stamp = stamp
            amsg.name = fn
            amsg.position = np.asarray(applied, dtype=float).tolist()
            self._pub[side]["applied"].publish(amsg)

    def destroy_node(self) -> bool:
        for pair in self.pairs.values():
            for r in (pair.leader, pair.follower):
                try:
                    r.close()
                except Exception:
                    pass
        return super().destroy_node()


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="HG-DAgger interactive takeover (bimanual, single gate)")
    p.add_argument("--sim", action="store_true")
    p.add_argument("--bilateral-kp", type=float, default=0.0, help="leader back-drive stiffness")
    p.add_argument("--rate", type=float, default=120.0)
    p.add_argument("--max-joint-speed", type=float, default=3.0, help="rad/s cap on follower target ramp (safety)")
    args = p.parse_args(argv)

    rclpy.init()
    node = DaggerNode(
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
