"""Recorder orchestrator — cameras + portal bridge + episode gate + async writer.

Runs a fixed-rate loop (``cfg.fps``, 60 Hz to match the cameras). Each tick it
grabs the latest camera frames and the latest robot snapshot (polled over portal),
lets the :class:`EpisodeGate` decide start/record/stop from the ``teleop_state``,
and **buffers** the frame locally. A finished episode is handed to the
:class:`AsyncDatasetWriter` queue, so LeRobot's per-trajectory encoding never
blocks the next collection.

Every frame carries ``observation.control_mode`` (teleop / policy / intervention),
plus whatever the robot reports (``observation.state``, ``observation.leader``,
``observation.eef`` when available) — so provenance is always in the dataset.

Review/delete: with ``review_before_save`` (default) each finished episode is held
unsaved in a PENDING state; the GUI plays back the buffered camera and the user
keeps (with a success/fail outcome) or deletes it. Two leader buttons can also
label+save automatically (see ``record_source`` / ``HOME_BUTTONS``).
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Dict, List, Optional

import numpy as np

from workstation.lerobot_recorder import episode_gate as eg
from workstation.lerobot_recorder.cameras import CameraManager
from workstation.lerobot_recorder.config import ACTION_DIM, EEF_DIM, LEADER_DIM, STATE_DIM, RecorderConfig
from workstation.lerobot_recorder.dataset_writer import AsyncDatasetWriter
from workstation.lerobot_recorder.episode_gate import EpisodeGate
from workstation.lerobot_recorder.portal_bridge import PortalBridge

# Leader handle buttons that end + label an episode (and trigger homing on the
# robot). Index into the per-side buttons list reported in the snapshot.
# "discard" ends the episode without saving it.
DISCARD_BUTTON = 0
SUCCESS_BUTTON = 1
FAIL_BUTTON = 2


class Recorder:
    def __init__(self, cfg: RecorderConfig, on_status: Optional[Callable[[dict], None]] = None) -> None:
        self.cfg = cfg
        self.cameras = CameraManager(cfg)
        self.robot = PortalBridge(cfg)
        self.gate = EpisodeGate()
        self.writer: Optional[AsyncDatasetWriter] = None
        self._on_status = on_status
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._status = {
            "running": False,
            "armed": False,
            "recording": False,
            "pending": False,
            "teleop": "—",
            "episodes": 0,
            "frames": 0,
            "queue": 0,
            "cam_ok": True,
            "robot_ok": False,
            "kept": 0,
            "success": 0,
            "fail": 0,
            "discarded": 0,
        }
        self._last_images: dict = {}
        self._pending = False
        self._episode: List[dict] = []  # buffered frames for the in-progress / pending episode
        self._preview: List[np.ndarray] = []  # downsampled review frames
        self._btn_prev: Dict[str, list] = {}
        self._btn_outcome: Optional[str] = None  # outcome chosen via a leader button this episode
        # "eval": record a continuous rollout (policy / intervention) from arm to disarm,
        # instead of gating on the teleop engage signal.
        self._eval = cfg.record_source == "eval"

    # ------------------------------------------------------------------ control
    def start(self) -> None:
        """Open cameras + robot link + dataset and begin the record loop (gate stays disarmed)."""
        self.cameras.start()
        self.robot.start()
        shapes = {k: self.cameras.shape_of(k) for k in self.cameras.image_keys}
        self.writer = AsyncDatasetWriter(self.cfg, self.cameras.image_keys, shapes)
        self.writer.open(self._sample_frame())
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self._set(running=True)

    def _sample_frame(self) -> dict:
        """The FIXED recorded schema, derived from the robot's known outputs (see config)."""
        return {
            "images": {k: np.zeros(self.cameras.shape_of(k), np.uint8) for k in self.cameras.image_keys},
            "observation.state": np.zeros(STATE_DIM, np.float32),
            "observation.leader": np.zeros(LEADER_DIM, np.float32),
            "observation.eef": np.zeros(EEF_DIM, np.float32),
            "observation.control_mode": np.zeros(1, np.float32),
            "action": np.zeros(ACTION_DIM, np.float32),
        }

    @staticmethod
    def _fit(vec: Optional[np.ndarray], dim: int) -> np.ndarray:
        """Coerce a vector to exactly ``dim`` (zeros if missing; pad/truncate otherwise)."""
        if vec is None:
            return np.zeros(dim, np.float32)
        a = np.asarray(vec, dtype=np.float32).reshape(-1)
        if a.size == dim:
            return a
        out = np.zeros(dim, np.float32)
        out[: min(dim, a.size)] = a[:dim]
        return out

    def arm(self) -> None:
        """GUI 'Start collection': begin gating (teleop/dagger) or a rollout (eval)."""
        self.gate.arm()
        if self._eval:  # eval: arm starts one continuous rollout
            self._episode, self._preview, self._btn_outcome = [], [], None
        self._set(armed=True, recording=self._eval)

    def disarm(self) -> None:
        """GUI 'Stop collection'. In eval mode this ENDS the rollout and saves it
        (or holds it for review); otherwise it just stops the gate and drops partials."""
        self.gate.disarm()
        if self._eval:
            if self._btn_outcome == "discard":
                self._discard_episode(counted=True)
            elif not self._episode:
                self._discard_episode()
            elif self.cfg.review_before_save and self._btn_outcome is None:
                self._pending = True
            else:
                self._submit(self._btn_outcome)
            self._set(armed=False, recording=False, pending=self._pending, queue=self.writer.queue_depth)
            return
        if self._pending:
            self._discard_episode()
        self._set(armed=False, recording=False, pending=False)

    def keep_episode(self, outcome: Optional[str] = None) -> None:
        """Review decision: keep the pending episode (submit it), with an optional outcome label."""
        if self._pending and self.writer is not None:
            self._submit(outcome)
            self._set(pending=False, episodes=self.writer.num_episodes, frames=0, queue=self.writer.queue_depth)

    def delete_episode(self) -> None:
        """Review decision: discard the pending episode."""
        if self._pending:
            self._discard_episode(counted=True)
            self._set(pending=False, frames=0)

    def _submit(self, outcome: Optional[str]) -> None:
        """Hand the buffered episode to the writer queue and update live stats."""
        self.writer.submit(self._episode, outcome, self.cfg.task)
        with self._lock:
            self._status["kept"] += 1
            if outcome in ("success", "fail"):
                self._status[outcome] += 1
        self._episode, self._preview, self._pending = [], [], False

    def _discard_episode(self, *, counted: bool = False) -> None:
        if counted:
            with self._lock:
                self._status["discarded"] += 1
        self._episode, self._preview, self._pending = [], [], False

    def shutdown(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.writer is not None:
            self.writer.finalize()
        self.cameras.stop()
        self.robot.stop()
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
                images = self.cameras.read()
                with self._lock:
                    self._last_images = images
                snap = self.robot.get_snapshot()
                if self._pending:
                    self._set(teleop=snap["teleop_state"])  # awaiting Keep/Delete: don't start a new episode
                else:
                    self._step(images, snap)
                    warned = self._warn_state(snap, warned)
            except Exception as e:  # keep the loop alive
                print(f"[recorder] step error: {e}")
            if self.cfg.mock:
                next_t += period
                time.sleep(max(0.0, next_t - time.perf_counter()))

    def _frame(self, images: dict, snap: dict) -> dict:
        return {
            "images": {k: np.ascontiguousarray(v) for k, v in images.items()},
            "observation.state": self._fit(snap.get("state"), STATE_DIM),
            "observation.leader": self._fit(snap.get("leader"), LEADER_DIM),
            "observation.eef": self._fit(snap.get("eef"), EEF_DIM),
            "observation.control_mode": np.array([snap.get("control_mode", 0)], dtype=np.float32),
            "action": self._fit(snap.get("action"), ACTION_DIM),
        }

    def _step(self, images: dict, snap: dict) -> None:
        self._scan_buttons(snap)

        if self._eval:  # continuous rollout while armed; no engage gate
            if self.gate.armed and self.cameras.healthy and snap["state"] is not None and snap["action"] is not None:
                self._episode.append(self._frame(images, snap))
                self._buffer_preview(images)
            self._set(
                armed=self.gate.armed,
                recording=self.gate.armed,
                pending=self._pending,
                teleop=snap["teleop_state"],
                episodes=self.writer.num_episodes,
                frames=len(self._episode),
                queue=self.writer.queue_depth,
                cam_ok=self.cameras.healthy,
                robot_ok=self.robot.connected,
            )
            return

        event = self.gate.update(snap["teleop_state"])

        if event in (eg.EV_START, eg.EV_RECORD, eg.EV_STOP):
            if event == eg.EV_START:
                self._episode, self._preview, self._btn_outcome = [], [], None
            if snap["state"] is not None and snap["action"] is not None and self.cameras.healthy:
                self._episode.append(self._frame(images, snap))
                self._buffer_preview(images)

        if event == eg.EV_STOP:
            if not self._episode:
                pass  # nothing recorded
            elif self._btn_outcome == "discard":
                self._discard_episode(counted=True)  # leader discard button: end without saving
            elif self._btn_outcome is not None:
                self._submit(self._btn_outcome)  # button auto-label+save
            elif self.cfg.review_before_save:
                self._pending = True  # hold for Keep/Delete
            else:
                self._submit(None)

        self._set(
            armed=self.gate.armed,
            recording=self.gate.recording,
            pending=self._pending,
            teleop=snap["teleop_state"],
            episodes=self.writer.num_episodes,
            frames=len(self._episode),
            queue=self.writer.queue_depth,
            cam_ok=self.cameras.healthy,
            robot_ok=self.robot.connected,
        )

    def _scan_buttons(self, snap: dict) -> None:
        """Latch a success/fail/discard outcome on a rising edge of the leader label buttons."""
        if not (self.gate.recording or (self._eval and self.gate.armed)):
            return
        for side, btns in (snap.get("buttons") or {}).items():
            prev = self._btn_prev.get(side, [])
            for idx, outcome in ((SUCCESS_BUTTON, "success"), (FAIL_BUTTON, "fail"), (DISCARD_BUTTON, "discard")):
                pressed = idx < len(btns) and bool(btns[idx])
                was = idx < len(prev) and bool(prev[idx])
                if pressed and not was:
                    self._btn_outcome = outcome
            self._btn_prev[side] = list(btns)

    def _warn_state(self, snap: dict, warned: bool) -> bool:
        if self.gate.recording and (snap["state"] is None or snap["action"] is None):
            if not warned:
                print("[recorder] waiting for robot state/action from the robot server…")
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

    # ------------------------------------------------------------------ status
    def _set(self, **kw: object) -> None:
        with self._lock:
            self._status.update(kw)
        if self._on_status is not None:
            self._on_status(self.get_status())
