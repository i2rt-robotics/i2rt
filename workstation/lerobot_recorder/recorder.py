"""Recorder orchestrator — cameras + ROS bridge + episode gate + dataset writer.

Runs a fixed-rate loop (``cfg.fps``, 60 Hz to match the cameras). Each tick it
grabs the latest camera frames and the latest ROS snapshot, lets the
:class:`EpisodeGate` decide start/record/stop from ``/teleop/state``, and writes
frames accordingly. Recording auto-starts on ENGAGED and (when review is off)
auto-saves when homing returns to IDLE.

Review/delete: with ``review_before_save`` (default) each finished episode is held
unsaved in a PENDING state; the GUI plays back the buffered camera and the user
keeps (``save_episode``) or deletes (``clear_episode_buffer``) it. While pending,
new episodes do not start.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, List, Optional

import numpy as np

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
        self._status = {
            "running": False, "armed": False, "recording": False,
            "pending": False, "teleop": "—", "episodes": 0, "frames": 0,
        }
        self._last_images: dict = {}
        self._pending = False
        self._preview: List[np.ndarray] = []  # downsampled review frames for the pending episode

    # ------------------------------------------------------------------ control
    def start(self) -> None:
        """Open cameras + ROS + dataset and begin the record loop (gate stays disarmed)."""
        self.cameras.start()
        self.ros.start()
        shapes = {k: self.cameras.shape_of(k) for k in self.cameras.image_keys}
        self.writer = DatasetWriter(self.cfg, self.cameras.image_keys, shapes)
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
        """GUI 'Stop collection': stop auto-recording; discard any in-progress/pending episode."""
        if self.gate.disarm() == "abort" and self.writer is not None:
            self.writer.abort_episode()
        if self._pending:
            self._discard_pending()
        self._set(armed=False, recording=False, pending=False)

    def keep_episode(self) -> None:
        """Review decision: keep the pending episode (save it)."""
        if self._pending and self.writer is not None:
            self.writer.save_episode()
            self._pending = False
            self._preview = []
            self._set(pending=False, episodes=self.writer.num_episodes, frames=0)

    def delete_episode(self) -> None:
        """Review decision: discard the pending episode."""
        if self._pending:
            self._discard_pending()
            self._set(pending=False, frames=0)

    def shutdown(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._pending:
            self._discard_pending()
        if self.writer is not None:
            self.writer.finalize()
        self.cameras.stop()
        self.ros.stop()
        self._set(running=False)

    def get_status(self) -> dict:
        with self._lock:
            return dict(self._status)

    def get_last_images(self) -> dict:
        with self._lock:
            return dict(self._last_images)

    def get_review_frames(self) -> List[np.ndarray]:
        with self._lock:
            return list(self._preview)

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

                if self._pending:
                    # awaiting Keep/Delete: do not start/record a new episode
                    self._set(teleop=snap["teleop_state"])
                else:
                    self._step(images, snap)
                    warned = self._warn_state(snap, warned)
            except Exception as e:  # keep the loop alive
                print(f"[recorder] step error: {e}")

            if self.cfg.mock:
                next_t += period
                time.sleep(max(0.0, next_t - time.perf_counter()))

    def _step(self, images: dict, snap: dict) -> None:
        event = self.gate.update(snap["teleop_state"])

        if event in (eg.EV_START, eg.EV_RECORD, eg.EV_STOP):
            if snap["state"] is not None and snap["action"] is not None:
                self.writer.add_frame(images, snap["state"], snap["action"], self.cfg.task)
                self._buffer_preview(images)

        if event == eg.EV_STOP:
            if self.writer.frames_in_episode <= 0:
                self.writer.abort_episode()
            elif self.cfg.review_before_save:
                self._pending = True  # hold for Keep/Delete
            else:
                self.writer.save_episode()

        self._set(
            armed=self.gate.armed,
            recording=self.gate.recording,
            pending=self._pending,
            teleop=snap["teleop_state"],
            episodes=self.writer.num_episodes,
            frames=self.writer.frames_in_episode,
        )

    def _warn_state(self, snap: dict, warned: bool) -> bool:
        if self.gate.recording and (snap["state"] is None or snap["action"] is None):
            if not warned:
                print("[recorder] waiting for robot state/action on ROS topics…")
            return True
        return False

    # ------------------------------------------------------------------ review buffer
    def _buffer_preview(self, images: dict) -> None:
        key = self.cfg.review_cam if self.cfg.review_cam in images else next(iter(images), None)
        if key is None:
            return
        s = max(self.cfg.review_downscale, 1)
        small = np.ascontiguousarray(images[key][::s, ::s])
        with self._lock:
            self._preview.append(small)
            if len(self._preview) > 1200:  # cap memory (~20 s @ 60 fps downsampled)
                self._preview.pop(0)

    def _discard_pending(self) -> None:
        if self.writer is not None:
            self.writer.abort_episode()
        self._pending = False
        self._preview = []

    # ------------------------------------------------------------------ status
    def _set(self, **kw: object) -> None:
        with self._lock:
            self._status.update(kw)
        if self._on_status is not None:
            self._on_status(self.get_status())
