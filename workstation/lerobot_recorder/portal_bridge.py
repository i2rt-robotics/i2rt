"""Portal bridge: poll the YAM robot server, expose the latest fused snapshot.

Connects (plain TCP, no ROS) to the robot machine running
``i2rt.serving.run_robot_server`` and polls ``get_observation()``. ``get_snapshot()``
returns the latest fused observation/action vectors plus the teleop gate state, in
the exact shape the recorder expects:

    {teleop_state, state(42,)|None, action(14,)|None, stamp}

``mock=True`` synthesizes a teleop cycle + fake joints so the pipeline runs with no
robot.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, Optional

import numpy as np

from workstation.lerobot_recorder.config import ARM_DOF, ARMS, RecorderConfig


class PortalBridge:
    def __init__(self, cfg: RecorderConfig) -> None:
        self.cfg = cfg
        self._lock = threading.Lock()
        self._snap = {"teleop_state": "IDLE", "state": None, "action": None, "stamp": 0.0}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._client = None

    # ------------------------------------------------------------------ public
    def start(self) -> None:
        target = self._mock_loop if self.cfg.mock else self._poll_loop
        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def get_snapshot(self) -> dict:
        with self._lock:
            return dict(self._snap)

    # ------------------------------------------------------------------ poll
    def _poll_loop(self) -> None:
        # Connect inside the thread so start() never blocks the GUI if the robot
        # server isn't up yet; on a dropped connection, reconnect on the next tick.
        from i2rt.serving.robot_client import RobotClient

        period = 1.0 / max(self.cfg.fps * 2, 1)
        while not self._stop.is_set():
            try:
                if self._client is None:
                    self._client = RobotClient(host=self.cfg.robot_host, port=self.cfg.robot_port)
                obs = self._client.get_observation()
                with self._lock:
                    self._snap = self._assemble(obs)
            except Exception:
                self._client = None  # drop and retry next tick
            time.sleep(period)

    @staticmethod
    def _fuse(sides: list, fields: tuple, per_arm: int) -> Optional[np.ndarray]:
        """Concatenate ``fields`` over both arms, or None if any is missing/wrong size."""
        parts = []
        for s in sides:
            if not s or any(s.get(f) is None for f in fields):
                return None
            vec = np.concatenate([np.asarray(s[f], dtype=np.float32) for f in fields])
            if vec.size != per_arm:
                return None
            parts.append(vec)
        return np.concatenate(parts).astype(np.float32)

    @classmethod
    def _assemble(cls, obs: Dict) -> dict:
        sides = [obs.get(a) for a in ARMS]
        state = cls._fuse(sides, ("pos", "vel", "eff"), ARM_DOF * 3)
        action = cls._fuse(sides, ("applied",), ARM_DOF)
        teleop_state = obs.get("teleop_state") or ("ENGAGED" if obs.get("intervention") else "IDLE")
        return {"teleop_state": teleop_state, "state": state, "action": action, "stamp": float(obs.get("t", 0.0))}

    # ------------------------------------------------------------------ mock
    def _mock_loop(self) -> None:
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
            state = np.concatenate([np.concatenate([pos, vel, eff]) for _ in ARMS]).astype(np.float32)
            action = np.concatenate([pos for _ in ARMS]).astype(np.float32)
            with self._lock:
                self._snap = {"teleop_state": name, "state": state, "action": action, "stamp": now}
            if now - seg_start >= dur:
                i = (i + 1) % len(cycle)
                seg_start = now
            time.sleep(0.01)
