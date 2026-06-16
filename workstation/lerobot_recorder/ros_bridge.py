"""Remote ROS 2 bridge: subscribe to the YAM teleop streams, expose latest snapshot.

Subscribes (over the network — same ``ROS_DOMAIN_ID`` as the robot machine) to:

* ``/<arm>/follower/joint_states`` — robot state (position/velocity/effort)
* ``/<arm>/applied_action``        — the smoothed command actually sent (the action)
* ``/teleop/state``                — HOMING / IDLE / ENGAGED (episode gate signal)

``get_snapshot()`` returns the latest fused observation/action vectors plus the
teleop state. ``mock=True`` synthesizes a teleop cycle + fake joints so the whole
pipeline runs with no robot/ROS.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, Optional

import numpy as np

from workstation.lerobot_recorder.config import ACTION_DIM, ARM_DOF, ARMS, STATE_DIM, RecorderConfig


class RosBridge:
    def __init__(self, cfg: RecorderConfig) -> None:
        self.cfg = cfg
        self._lock = threading.Lock()
        self._teleop_state = "IDLE"
        self._fstate: Dict[str, Optional[np.ndarray]] = {a: None for a in ARMS}  # [pos,vel,eff] = 21
        self._action: Dict[str, Optional[np.ndarray]] = {a: None for a in ARMS}  # applied = 7
        self._stamp = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._node = None

    # ------------------------------------------------------------------ public
    def start(self) -> None:
        if self.cfg.mock:
            self._thread = threading.Thread(target=self._mock_loop, daemon=True)
            self._thread.start()
        else:
            self._start_ros()

    def stop(self) -> None:
        self._stop.set()
        if self._node is not None:
            try:
                import rclpy

                self._node.destroy_node()
                if rclpy.ok():
                    rclpy.shutdown()
            except Exception:
                pass

    def get_snapshot(self) -> dict:
        """Return {teleop_state, state(42,)|None, action(14,)|None, stamp}."""
        with self._lock:
            state = self._build(self._fstate, per_arm=ARM_DOF * 3, total=STATE_DIM)
            action = self._build(self._action, per_arm=ARM_DOF, total=ACTION_DIM)
            return {"teleop_state": self._teleop_state, "state": state, "action": action, "stamp": self._stamp}

    @staticmethod
    def _build(parts: Dict[str, Optional[np.ndarray]], per_arm: int, total: int) -> Optional[np.ndarray]:
        if any(parts[a] is None or parts[a].size != per_arm for a in ARMS):
            return None
        return np.concatenate([parts[a] for a in ARMS]).astype(np.float32)

    # ------------------------------------------------------------------ ROS 2
    def _start_ros(self) -> None:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        from std_msgs.msg import String

        if not rclpy.ok():
            rclpy.init()
        node = Node("lerobot_recorder_sub")
        self._node = node
        t = self.cfg.topics
        for arm in ARMS:
            node.create_subscription(JointState, t.follower_state[arm], self._make_state_cb(arm), 10)
            node.create_subscription(JointState, t.applied_action[arm], self._make_action_cb(arm), 10)
        node.create_subscription(String, t.teleop_state, self._on_teleop_state, 10)

        def spin() -> None:
            while not self._stop.is_set() and rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.1)

        self._thread = threading.Thread(target=spin, daemon=True)
        self._thread.start()

    def _make_state_cb(self, arm: str) -> Callable[[Any], None]:
        def cb(msg: Any) -> None:  # sensor_msgs/JointState
            vec = np.concatenate(
                [
                    np.asarray(msg.position, dtype=np.float32),
                    np.asarray(msg.velocity, dtype=np.float32),
                    np.asarray(msg.effort, dtype=np.float32),
                ]
            )
            with self._lock:
                self._fstate[arm] = vec
                self._stamp = time.time()

        return cb

    def _make_action_cb(self, arm: str) -> Callable[[Any], None]:
        def cb(msg: Any) -> None:  # sensor_msgs/JointState
            with self._lock:
                self._action[arm] = np.asarray(msg.position, dtype=np.float32)

        return cb

    def _on_teleop_state(self, msg: Any) -> None:  # std_msgs/String
        with self._lock:
            self._teleop_state = str(msg.data)

    # ------------------------------------------------------------------ mock
    def _mock_loop(self) -> None:
        """Cycle IDLE→ENGAGED→HOMING→IDLE with fake joint data for testing."""
        cycle = [("IDLE", 1.0), ("ENGAGED", 3.0), ("HOMING", 1.5)]
        i = 0
        t0 = time.time()
        seg_start = t0
        while not self._stop.is_set():
            name, dur = cycle[i]
            now = time.time()
            phase = now - t0
            pos = 0.3 * np.sin(phase + np.arange(ARM_DOF))
            vel = 0.3 * np.cos(phase + np.arange(ARM_DOF))
            eff = 0.1 * np.ones(ARM_DOF)
            with self._lock:
                self._teleop_state = name
                for arm in ARMS:
                    self._fstate[arm] = np.concatenate([pos, vel, eff]).astype(np.float32)
                    self._action[arm] = pos.astype(np.float32)
                self._stamp = now
            if now - seg_start >= dur:
                i = (i + 1) % len(cycle)
                seg_start = now
            time.sleep(0.01)
