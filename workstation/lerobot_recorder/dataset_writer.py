"""Async, queued ``LeRobotDataset`` (v3.0) writer for bimanual YAM episodes.

The recorder buffers each episode's frames in memory and **submits the whole
episode** to this writer. A single background worker thread then encodes/saves
episodes **one at a time** off a queue — so LeRobot's per-trajectory processing
(``save_episode``: parquet + video encode) never blocks the next collection.

Targets the official LeRobot Dataset **v3.0** API (``lerobot >= 0.4.0``):

* ``LeRobotDataset.create(...)`` / load existing for ``--resume``
* ``add_frame(frame)`` — per-frame ``task`` is a key inside the frame dict
* ``save_episode()`` / ``clear_episode_buffer()`` / ``finalize()``

Each frame is a dict of ``{feature_key: np.ndarray}`` plus ``"images"`` (a
``{cam: HxWx3}`` dict) and ``"task"``. The feature schema is built from a sample
frame so new fields (leader pose, eef, control_mode, …) flow through with no
schema edits here. ``mock=True`` skips ``lerobot`` and just counts.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from typing import Dict, List, Optional

import numpy as np

from workstation.lerobot_recorder.config import RecorderConfig


def _import_lerobot_dataset() -> type:
    try:
        from lerobot.datasets import LeRobotDataset  # lerobot >= 0.4.0 (v3.0)
    except ImportError:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset


class AsyncDatasetWriter:
    """Queued episode writer. ``submit()`` returns immediately; a worker saves."""

    def __init__(self, cfg: RecorderConfig, image_keys: List[str], image_shapes: Dict[str, tuple]) -> None:
        self.cfg = cfg
        self.image_keys = image_keys
        self.image_shapes = image_shapes
        self._mock = cfg.mock
        self._ds = None
        self._features: Optional[dict] = None
        self._root = os.path.expanduser(cfg.root)
        self._outcomes_path = os.path.join(self._root, "outcomes.jsonl")

        self._queue: "queue.Queue" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._n_episodes = 0  # saved episodes (incremented by the worker)
        self._n_submitted = 0

    # ------------------------------------------------------------------ schema
    @staticmethod
    def _vector_names(key: str, dim: int) -> List[str]:
        from workstation.lerobot_recorder.config import action_names, leader_names, state_names

        if key == "observation.state":
            return state_names()
        if key == "action":
            return action_names()
        if key == "observation.leader":
            return leader_names()
        return [f"{key.rsplit('.', 1)[-1]}.{i}" for i in range(dim)]

    def _build_features(self, sample: dict) -> dict:
        img_dtype = "video" if self.cfg.use_videos else "image"
        feats: dict = {}
        for key in self.image_keys:
            feats[f"observation.images.{key}"] = {
                "dtype": img_dtype,
                "shape": self.image_shapes[key],
                "names": ["height", "width", "channels"],
            }
        for key, val in sample.items():
            if key in ("images", "task"):
                continue
            vec = np.asarray(val, dtype=np.float32).reshape(-1)
            feats[key] = {"dtype": "float32", "shape": (vec.size,), "names": self._vector_names(key, vec.size)}
        return feats

    # ------------------------------------------------------------------ lifecycle
    def open(self, sample_frame: dict) -> None:
        """Open the dataset using ``sample_frame`` to derive the feature schema."""
        self._features = self._build_features(sample_frame)
        if not self._mock:
            LeRobotDataset = _import_lerobot_dataset()
            if self.cfg.resume and os.path.isdir(self._root):
                self._ds = LeRobotDataset(self.cfg.repo_id, root=self._root)
                self._n_episodes = int(
                    getattr(self._ds, "num_episodes", getattr(getattr(self._ds, "meta", None), "total_episodes", 0))
                )
                print(f"[dataset] resuming at {self._root} ({self._n_episodes} existing episodes)")
            else:
                self._ds = LeRobotDataset.create(
                    repo_id=self.cfg.repo_id,
                    fps=self.cfg.fps,
                    features=self._features,
                    root=self._root,
                    robot_type=self.cfg.robot_type,
                    use_videos=self.cfg.use_videos,
                )
                print(f"[dataset] created v3.0 dataset at {self._root}")
        else:
            print(f"[dataset] MOCK writer (repo_id={self.cfg.repo_id}); features={sorted(self._features)}")

        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------ submit
    def submit(self, frames: List[dict], outcome: Optional[str], task: str) -> None:
        """Enqueue a complete episode (list of frame dicts) for background saving."""
        if frames:
            self._queue.put((frames, outcome, task))
            with self._lock:
                self._n_submitted += 1

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def num_episodes(self) -> int:
        with self._lock:
            return self._n_episodes

    # ------------------------------------------------------------------ worker
    def _run(self) -> None:
        while not (self._stop.is_set() and self._queue.empty()):
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            frames, outcome, task = item
            try:
                self._save_episode(frames, outcome, task)
            except Exception as e:
                print(f"[dataset] episode save failed: {e}")
            finally:
                self._queue.task_done()

    def _save_episode(self, frames: List[dict], outcome: Optional[str], task: str) -> None:
        if not self._mock:
            for f in frames:
                frame = {f"observation.images.{k}": v for k, v in f["images"].items()}
                for key, val in f.items():
                    if key in ("images", "task"):
                        continue
                    frame[key] = np.asarray(val, dtype=np.float32)
                frame["task"] = task
                self._ds.add_frame(frame)
            self._ds.save_episode()
        with self._lock:
            episode_index = self._n_episodes
            self._n_episodes += 1
        self._record_outcome(episode_index, outcome, task, len(frames))
        print(f"[dataset] saved episode #{episode_index} ({len(frames)} frames, outcome={outcome})")

    def _record_outcome(self, episode_index: int, outcome: Optional[str], task: str, n_frames: int) -> None:
        entry = {
            "episode": episode_index,
            "outcome": outcome,
            "task": task,
            "frames": n_frames,
            "source": self.cfg.record_source,
            "t": time.time(),
        }
        try:
            os.makedirs(self._root, exist_ok=True)
            with open(self._outcomes_path, "a") as fh:
                fh.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"[dataset] could not write outcome sidecar: {e}")

    # ------------------------------------------------------------------ shutdown
    def finalize(self) -> None:
        """Drain the queue, stop the worker, then close the LeRobot dataset."""
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=600.0)
        if not self._mock and self._ds is not None and self._n_episodes > 0:
            try:
                self._ds.finalize()
                print("[dataset] finalized (parquet/metadata closed)")
            except Exception as e:
                print(f"[dataset] finalize failed: {e}")
