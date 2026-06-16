"""Component ② — bimanual leader-follower teleop with an auto home/engage gate.

Both leader+follower pairs are activated together and driven by a single global
state machine (no per-arm buttons):

* **HOMING** — robot *and* leaders ramp smoothly to the home pose.
* **IDLE**   — sitting at home, leaders free; waiting for the human to lift them.
* **ENGAGED** — once **both** leaders are lifted past ``--engage-thr`` from home,
  teleop turns on and the followers track the leaders (rate-limited, no jump).
  When **both** leaders are brought back within ``--release-thr`` of home (held
  for ``--dwell`` s), the robot *and* leaders smoothly return home.

The exact rate-limited target sent to each follower is published on
``<s>/applied_action`` so an external node can log it and reproduce the episode
precisely.

Topics per side ``<s>`` in {left, right}:

* pub  ``<s>/leader/joint_states``   ``sensor_msgs/JointState``
* pub  ``<s>/follower/joint_states`` ``sensor_msgs/JointState``
* pub  ``<s>/applied_action``        ``sensor_msgs/JointState`` (smoothed command sent to the robot)
* pub  ``<s>/buttons``               ``sensor_msgs/Joy``

Global:

* pub  ``/teleop/state``   ``std_msgs/String`` (HOMING / IDLE / ENGAGED)
* pub  ``/teleop/active``  ``std_msgs/Bool``   (True iff ENGAGED)
* sub  ``/teleop/sim_engage`` ``std_msgs/Bool`` (debug override; forces ENGAGED in --sim)

Examples::

    python -m i2rt.ros2.run_teleop --sim
    python -m i2rt.ros2.run_teleop --bilateral-kp 0.2 --home 0,0,0,0,0,0
"""

from __future__ import annotations

import argparse
from typing import List, Optional

import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState, Joy
from std_msgs.msg import Bool, String

from i2rt.ros2 import ros_conversions as conv
from i2rt.ros2.safety import TargetSmoother, is_finite_vector, max_step_from_speed
from i2rt.ros2.teleop_common import (
    ArmPair,
    TeleopStateMachine,
    arm_distance,
    build_bimanual,
    build_follower_target,
    default_bimanual_specs,
    read_handle,
)

_HOME_TOL = 0.05  # rad; homing considered done when the ramp is within this of home


class TeleopNode(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("yam_teleop")
        self.bilateral_kp = args.bilateral_kp
        self.home_kp = args.home_kp
        self.pairs = build_bimanual(default_bimanual_specs(args.sim), sim=args.sim)
        max_step = max_step_from_speed(args.max_joint_speed, args.rate)

        # home pose: arm joints (+ optional gripper), shared by both arms
        first = next(iter(self.pairs.values())).follower
        n = int(first.num_dofs())
        self._has_grip = "gripper_pos" in first.get_observations()
        n_arm = n - 1 if self._has_grip else n
        self.home_arm, self.home_grip = self._parse_home(args.home, n_arm)
        self.home_full = np.concatenate([self.home_arm, [self.home_grip]]) if self._has_grip else self.home_arm.copy()

        self.sm = TeleopStateMachine(args.engage_thr, args.release_thr, args.dwell)
        self._sim_engage = False

        self._names, self._fsmooth, self._lsmooth, self._pub = {}, {}, {}, {}
        for side, pair in self.pairs.items():
            ln, lg = conv.joint_names(pair.leader)
            fn, fg = conv.joint_names(pair.follower)
            self._names[side] = (ln, lg, fn, fg)
            self._fsmooth[side] = TargetSmoother(pair.follower.get_joint_pos(), max_step)
            self._lsmooth[side] = TargetSmoother(np.asarray(pair.leader.get_joint_pos())[:n_arm], max_step)
            self._pub[side] = {
                "leader": self.create_publisher(JointState, f"{side}/leader/joint_states", 10),
                "follower": self.create_publisher(JointState, f"{side}/follower/joint_states", 10),
                "applied": self.create_publisher(JointState, f"{side}/applied_action", 10),
                "buttons": self.create_publisher(Joy, f"{side}/buttons", 10),
            }
        self.state_pub = self.create_publisher(String, "/teleop/state", 10)
        self.active_pub = self.create_publisher(Bool, "/teleop/active", 10)
        self.create_subscription(Bool, "/teleop/sim_engage", self._on_sim_engage, 10)

        self.create_timer(1.0 / max(args.rate, 1.0), self._loop)
        self.get_logger().info(
            f"TeleopNode up: sides={list(self.pairs)} home_arm={np.round(self.home_arm, 2).tolist()} "
            f"engage>{args.engage_thr} release<{args.release_thr} bilateral_kp={args.bilateral_kp} sim={args.sim}"
        )

    @staticmethod
    def _parse_home(home_str: str, n_arm: int) -> "tuple[np.ndarray, float]":
        if not home_str:
            return np.zeros(n_arm), 0.0
        vals = [float(x) for x in home_str.split(",") if x.strip() != ""]
        if len(vals) == n_arm:
            return np.asarray(vals), 0.0
        if len(vals) == n_arm + 1:
            return np.asarray(vals[:n_arm]), float(vals[n_arm])
        raise ValueError(f"--home expects {n_arm} or {n_arm + 1} values, got {len(vals)}")

    def _on_sim_engage(self, msg: Bool) -> None:
        self._sim_engage = bool(msg.data)

    def _stamp(self) -> Time:
        return self.get_clock().now().to_msg()

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _homing_done(self) -> bool:
        for side in self.pairs:
            if np.linalg.norm(self._fsmooth[side].cur - self.home_full) > _HOME_TOL:
                return False
            if np.linalg.norm(self._lsmooth[side].cur - self.home_arm) > _HOME_TOL:
                return False
        return True

    def _loop(self) -> None:
        # 1) read all leaders + distances from home
        arm_q, grip, buttons, dists = {}, {}, {}, []
        for side, pair in self.pairs.items():
            try:
                a, g, b = read_handle(pair.leader)
            except Exception as e:
                self.get_logger().warn(f"[{side}] handle read failed: {e}")
                a, g, b = np.asarray(pair.leader.get_joint_pos(), dtype=float), None, []
            arm_q[side], grip[side], buttons[side] = a, g, b
            dists.append(arm_distance(a, self.home_arm))

        # 2) global state machine (debug override forces ENGAGED in sim)
        state = self.sm.update(dists, self._homing_done(), self._now())
        if self._sim_engage:
            state = TeleopStateMachine.ENGAGED

        # 3) act + publish per arm
        for side, pair in self.pairs.items():
            ln, lg, fn, fg = self._names[side]
            n = pair.follower.num_dofs()
            fsm, lsm = self._fsmooth[side], self._lsmooth[side]
            try:
                if state == TeleopStateMachine.ENGAGED:
                    desired = build_follower_target(pair.follower, arm_q[side], grip[side])
                    if not is_finite_vector(desired, n):
                        desired = fsm.cur
                    applied = fsm.step(desired)
                    if self.bilateral_kp > 0.0:
                        self._drive_leader(pair, np.asarray(pair.follower.get_joint_pos())[: pair.leader.num_dofs()])
                    else:
                        self._free_leader(pair)
                    lsm.reset(np.asarray(pair.leader.get_joint_pos())[: self.home_arm.size])
                elif state == TeleopStateMachine.HOMING:
                    applied = fsm.step(self.home_full)
                    self._home_leader(pair, lsm.step(self.home_arm))
                else:  # IDLE
                    applied = fsm.step(self.home_full)
                    self._free_leader(pair)
                    lsm.reset(np.asarray(pair.leader.get_joint_pos())[: self.home_arm.size])

                pair.follower.command_joint_pos(applied)
                self._publish(side, pair, ln, lg, fn, fg, buttons[side], grip[side], applied)
            except Exception as e:
                self.get_logger().warn(f"[{side}] teleop step failed: {e}")

        self.state_pub.publish(String(data=state))
        self.active_pub.publish(Bool(data=state == TeleopStateMachine.ENGAGED))

    # ------------------------------------------------------------------ leader modes
    def _home_leader(self, pair: ArmPair, target_arm: np.ndarray) -> None:
        """Drive the leader to the home pose with gentle PD (used only while HOMING)."""
        leader = pair.leader
        if not hasattr(leader, "update_kp_kd") or pair.base_kp is None:
            return
        try:
            m = leader.num_dofs()
            kd = pair.base_kd[:m] if pair.base_kd is not None else np.full(m, 0.5)
            leader.update_kp_kd(pair.base_kp[:m] * self.home_kp, kd)
            leader.command_joint_pos(np.asarray(target_arm, dtype=float)[:m])
        except Exception:
            pass

    def _free_leader(self, pair: ArmPair) -> None:
        """Release the leader to gravity-comp idle so the human can move it freely."""
        leader = pair.leader
        if hasattr(leader, "enter_gravity_comp_idle"):
            try:
                leader.enter_gravity_comp_idle()
            except Exception:
                pass

    def _drive_leader(self, pair: ArmPair, target_q: np.ndarray) -> None:
        """Back-drive the leader toward ``target_q`` so the human feels follower forces."""
        leader = pair.leader
        if self.bilateral_kp <= 0.0 or not hasattr(leader, "update_kp_kd") or pair.base_kp is None:
            return
        try:
            m = leader.num_dofs()
            leader.update_kp_kd(pair.base_kp[:m] * self.bilateral_kp, np.zeros(m))
            leader.command_joint_pos(np.asarray(target_q, dtype=float)[:m])
        except Exception:
            pass

    def _publish(self, side, pair, ln, lg, fn, fg, buttons, grip, applied) -> None:  # noqa: ANN001
        stamp = self._stamp()
        self._pub[side]["leader"].publish(conv.robot_to_joint_state(pair.leader, ln, lg, stamp))
        self._pub[side]["follower"].publish(conv.robot_to_joint_state(pair.follower, fn, fg, stamp))
        self._pub[side]["buttons"].publish(conv.buttons_to_joy(buttons, grip if grip is not None else 0.0, stamp))
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
    p = argparse.ArgumentParser(description="Bimanual teleop with auto home/engage gate")
    p.add_argument("--sim", action="store_true")
    p.add_argument("--home", default="", help="comma-separated home arm joints (default: zeros)")
    p.add_argument("--engage-thr", type=float, default=0.6, help="rad; both leaders past this from home → ENGAGE")
    p.add_argument("--release-thr", type=float, default=0.3, help="rad; both leaders within this of home → home")
    p.add_argument("--dwell", type=float, default=0.5, help="s; release must hold this long before homing")
    p.add_argument("--home-kp", type=float, default=0.3, help="leader stiffness scale while homing")
    p.add_argument("--bilateral-kp", type=float, default=0.0, help="leader back-drive stiffness while engaged")
    p.add_argument("--rate", type=float, default=120.0)
    p.add_argument("--max-joint-speed", type=float, default=1.5, help="rad/s cap on follower target ramp (smoothness/safety)")
    args = p.parse_args(argv)

    rclpy.init()
    node = TeleopNode(args)
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
