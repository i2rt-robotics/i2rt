"""Replay a recorded episode onto the robot over ROS 2.

Publishes each frame's ``action`` (14-d = both arms x 7) to the YAM wrapper's
``/<arm>/command`` topics, so the robot follows the dataset. Before playing it
**ramps from the robot's current pose to the first frame** (over ``ramp_s``) to
avoid a jump — so the robot side must be running the wrapper
(``scripts/yam wrapper``) and publishing ``/<arm>/follower/joint_states``.

If ``send_to_robot`` is False it just advances frames for preview (no commands).
``mock=True`` skips ROS entirely.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Dict, Optional

import numpy as np

from workstation.lerobot_recorder.config import ARM_DOF, ARMS, RecorderConfig
from workstation.lerobot_recorder.dataset_reader import DatasetReader


class ReplayController:
    def __init__(self, reader: DatasetReader, cfg: RecorderConfig, on_frame: Optional[Callable[[int], None]] = None) -> None:
        self.reader = reader
        self.cfg = cfg
        self.on_frame = on_frame
        self.speed = 1.0
        self._mock = cfg.mock
        self._lock = threading.Lock()
        self._current: Dict[str, Optional[np.ndarray]] = {a: None for a in ARMS}
        self._node = None
        self._pubs: Dict[str, object] = {}
        self._spin_stop = threading.Event()
        self._play_thread: Optional[threading.Thread] = None
        self._pause = threading.Event()
        self._stop = threading.Event()
        self._frame = 0
        self._playing = False

    # ------------------------------------------------------------------ ROS
    def connect(self) -> None:
        if self._mock:
            return
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState

        if not rclpy.ok():
            rclpy.init()
        node = Node("yam_replay")
        self._node = node
        t = self.cfg.topics
        for arm in ARMS:
            self._pubs[arm] = node.create_publisher(JointState, t.applied_action[arm].replace("/applied_action", "/command"), 10)
            node.create_subscription(JointState, t.follower_state[arm], self._make_state_cb(arm), 10)

        def spin() -> None:
            while not self._spin_stop.is_set() and rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.1)

        threading.Thread(target=spin, daemon=True).start()

    def _make_state_cb(self, arm: str) -> Callable[[object], None]:
        def cb(msg) -> None:  # noqa: ANN001  sensor_msgs/JointState
            with self._lock:
                self._current[arm] = np.asarray(msg.position, dtype=np.float32)
        return cb

    def _publish(self, action: np.ndarray) -> None:
        if self._mock:
            return
        from sensor_msgs.msg import JointState

        for i, arm in enumerate(ARMS):
            seg = np.asarray(action[i * ARM_DOF : (i + 1) * ARM_DOF], dtype=float)
            msg = JointState()
            msg.header.stamp = self._node.get_clock().now().to_msg()
            msg.name = [f"joint_{j + 1}" for j in range(ARM_DOF - 1)] + ["gripper"]
            msg.position = seg.tolist()
            self._pubs[arm].publish(msg)

    # ------------------------------------------------------------------ playback
    def play_episode(self, episode: int, send_to_robot: bool, ramp_s: float = 2.0) -> None:
        self.stop()
        self._stop.clear()
        self._pause.clear()
        self._play_thread = threading.Thread(target=self._run, args=(episode, send_to_robot, ramp_s), daemon=True)
        self._play_thread.start()

    def _run(self, episode: int, send_to_robot: bool, ramp_s: float) -> None:
        self._playing = True
        n = self.reader.episode_length(episode)
        if n == 0:
            self._playing = False
            return
        first = self.reader.get_action(episode, 0)

        if send_to_robot and not self._mock:
            self._ramp_to(first, ramp_s)

        for f in range(n):
            if self._stop.is_set():
                break
            while self._pause.is_set() and not self._stop.is_set():
                time.sleep(0.02)
            self._frame = f
            action = self.reader.get_action(episode, f)
            if send_to_robot:
                self._publish(action)
            if self.on_frame is not None:
                self.on_frame(f)
            time.sleep(1.0 / max(self.reader.fps * self.speed, 1.0))
        self._playing = False

    def _ramp_to(self, target: np.ndarray, ramp_s: float) -> None:
        with self._lock:
            cur = np.concatenate([self._current[a] for a in ARMS]) if all(self._current[a] is not None for a in ARMS) else None
        if cur is None or cur.size != target.size:
            self._publish(target)  # no known current pose; send once
            return
        steps = max(int(ramp_s * self.reader.fps), 1)
        for s in range(1, steps + 1):
            if self._stop.is_set():
                return
            self._publish(cur + (target - cur) * (s / steps))
            time.sleep(1.0 / self.reader.fps)

    def pause(self) -> None:
        self._pause.set()

    def resume(self) -> None:
        self._pause.clear()

    def stop(self) -> None:
        self._stop.set()
        if self._play_thread is not None:
            self._play_thread.join(timeout=1.0)
        self._playing = False

    def set_speed(self, speed: float) -> None:
        self.speed = max(0.1, float(speed))

    @property
    def frame(self) -> int:
        return self._frame

    @property
    def playing(self) -> bool:
        return self._playing

    def shutdown(self) -> None:
        self.stop()
        self._spin_stop.set()
        if self._node is not None:
            try:
                import rclpy

                self._node.destroy_node()
                if rclpy.ok():
                    rclpy.shutdown()
            except Exception:
                pass
