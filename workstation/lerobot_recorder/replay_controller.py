"""Replay a recorded episode onto the robot over portal (no ROS).

Sends each frame's ``action`` (14-d = both arms x 7) to the robot server via
``RobotClient.command({"left": ..., "right": ...})``, so the robot follows the
dataset. Before playing it **ramps from the robot's current pose to the first
frame** (over ``ramp_s``) to avoid a jump — so the robot side must be running
``i2rt.serving.run_robot_server wrapper``.

If ``send_to_robot`` is False it just advances frames for preview (no commands).
``mock=True`` skips the robot link entirely.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Dict, Optional

import numpy as np

from workstation.lerobot_recorder.config import ARM_DOF, ARMS, RecorderConfig
from workstation.lerobot_recorder.dataset_reader import DatasetReader


class ReplayController:
    def __init__(
        self, reader: DatasetReader, cfg: RecorderConfig, on_frame: Optional[Callable[[int], None]] = None
    ) -> None:
        self.reader = reader
        self.cfg = cfg
        self.on_frame = on_frame
        self.speed = 1.0
        self._mock = cfg.mock
        self._client = None
        self._play_thread: Optional[threading.Thread] = None
        self._pause = threading.Event()
        self._stop = threading.Event()
        self._frame = 0
        self._playing = False

    # ------------------------------------------------------------------ robot link
    def connect(self) -> None:
        """Connect to the robot in the background so the GUI never blocks.

        Preview (Send-to-robot off) works regardless; if the robot server is down,
        ``_client`` stays None and commands are no-ops until it comes up.
        """
        if self._mock:
            return

        def _connect() -> None:
            from i2rt.serving.robot_client import RobotClient

            try:
                self._client = RobotClient(host=self.cfg.robot_host, port=self.cfg.robot_port)
            except Exception:
                self._client = None

        threading.Thread(target=_connect, daemon=True).start()

    @property
    def connected(self) -> bool:
        return self._client is not None

    def set_estop(self, flag: bool) -> None:
        """Forward an e-stop to the robot (no-op in mock / before connect)."""
        if self._client is not None:
            try:
                self._client.set_estop(flag)
            except Exception:
                pass

    def _split(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        return {arm: np.asarray(action[i * ARM_DOF : (i + 1) * ARM_DOF], dtype=float) for i, arm in enumerate(ARMS)}

    def _publish(self, action: np.ndarray) -> None:
        if self._mock or self._client is None:
            return
        self._client.command(self._split(action))

    def _current_pose(self) -> Optional[np.ndarray]:
        """Concatenated follower pose (both arms) from the robot, or None."""
        if self._mock or self._client is None:
            return None
        try:
            obs = self._client.get_observation()
            parts = []
            for a in ARMS:
                side = obs.get(a)
                if not side or "pos" not in side:
                    return None
                parts.append(np.asarray(side["pos"], dtype=np.float32))
            return np.concatenate(parts)
        except Exception:
            return None

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
        cur = self._current_pose()
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
        self._client = None
