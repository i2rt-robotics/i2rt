"""Recorder orchestrator — cameras + ROS bridge + episode gate + dataset writer.

Runs a fixed-rate loop (``cfg.fps``, 60 Hz to match the cameras). Each tick it
grabs the latest camera frames and the latest ROS snapshot, lets the
:class:`EpisodeGate` decide start/record/stop from ``/teleop/state``, and writes
frames accordingly. Recording auto-starts on ENGAGED and auto-saves when homing
returns to IDLE.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from workstation.lerobot_recorder import episode_gate as eg
from workstation.lerobot_recorder.cameras import CameraManager
from workstation.lerobot_recorder.config import RecorderConfig
from workstation.lerobot_recorder.dataset_writer import DatasetWriter
from workstation.lerobot_recorder.episode_gate import EpisodeGate
from workstation.lerobot_recorder.ros_bridge import RosBridge


class Recorder:
    def __init__(self, cfg: RecorderConfig, on_status: Optional[Callable[[dict], None]] = None) -> None:
        self.cfg = cfg
        self.cameras = CameraManager(cfg)
        self.ros = RosBridge(cfg)
        self.gate = EpisodeGate()
        self.writer: Optional[DatasetWriter] = None
        self._on_status = on_status
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._status = {"running": False, "armed": False, "recording": False, "teleop": "—", "episodes": 0, "frames": 0}
        self._last_images: dict = {}

    # ------------------------------------------------------------------ control
    def start(self) -> None:
        """Open cameras + ROS + dataset and begin the record loop (gate stays disarmed)."""
        self.cameras.start()
        self.ros.start()
        self.writer = DatasetWriter(self.cfg, self.cameras.image_keys, {k: self.cameras.shape_of(k) for k in self.cameras.image_keys})
        self.writer.open()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self._set(running=True)

    def arm(self) -> None:
        """GUI 'Start collection': let episodes auto-start/stop from the teleop gate."""
        self.gate.arm()
        self._set(armed=True)

    def disarm(self) -> None:
        """GUI 'Stop collection': stop auto-recording; discard any in-progress episode."""
        if self.gate.disarm() == "abort" and self.writer is not None:
            self.writer.abort_episode()
        self._set(armed=False, recording=False)

    def shutdown(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.cameras.stop()
        self.ros.stop()
        self._set(running=False)

    def get_status(self) -> dict:
        with self._lock:
            return dict(self._status)

    def get_last_images(self) -> dict:
        with self._lock:
            return dict(self._last_images)

    # ------------------------------------------------------------------ loop
    def _loop(self) -> None:
        period = 1.0 / max(self.cfg.fps, 1.0)
        next_t = time.perf_counter()
        warned = False
        while not self._stop.is_set():
            try:
                images = self.cameras.read()  # real: paces at camera fps; mock: instant
                with self._lock:
                    self._last_images = images
                snap = self.ros.get_snapshot()
                event = self.gate.update(snap["teleop_state"])

                if event in (eg.EV_START, eg.EV_RECORD, eg.EV_STOP):
                    if snap["state"] is not None and snap["action"] is not None:
                        self.writer.add_frame(images, snap["state"], snap["action"], self.cfg.task)
                        warned = False
                    elif not warned:
                        print("[recorder] waiting for robot state/action on ROS topics…")
                        warned = True

                if event == eg.EV_STOP:
                    if self.writer.frames_in_episode > 0:
                        self.writer.save_episode()
                    else:
                        self.writer.abort_episode()

                self._set(
                    armed=self.gate.armed,
                    recording=self.gate.recording,
                    teleop=snap["teleop_state"],
                    episodes=self.writer.num_episodes,
                    frames=self.writer.frames_in_episode,
                )
            except Exception as e:  # keep the loop alive
                print(f"[recorder] step error: {e}")

            if self.cfg.mock:
                next_t += period
                time.sleep(max(0.0, next_t - time.perf_counter()))

    # ------------------------------------------------------------------ status
    def _set(self, **kw: object) -> None:
        with self._lock:
            self._status.update(kw)
        if self._on_status is not None:
            self._on_status(self.get_status())
