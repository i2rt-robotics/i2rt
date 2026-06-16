"""Thin wrapper over ``LeRobotDataset`` (v3.0) for recording bimanual YAM episodes.

Targets the official LeRobot Dataset **v3.0** API (``lerobot >= 0.4.0``):

* ``LeRobotDataset.create(repo_id, fps, features, root=..., robot_type=..., use_videos=...)``
* ``add_frame(frame)``  — the per-frame ``task`` is a **key inside the frame dict**
* ``save_episode()``    — commits the buffered episode
* ``clear_episode_buffer(delete_images=True)`` — discards an unsaved episode (used by review/delete)
* ``finalize()``        — **must** be called when done so parquet writers/footers close

``mock=True`` skips ``lerobot`` entirely and just counts frames/episodes.
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np

from workstation.lerobot_recorder.config import ACTION_DIM, STATE_DIM, RecorderConfig, action_names, state_names


def _import_lerobot_dataset() -> type:
    """Import LeRobotDataset (v3.0 layout, with a fallback for older trees)."""
    try:
        from lerobot.datasets import LeRobotDataset  # lerobot >= 0.4.0 (v3.0)
    except ImportError:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset


class DatasetWriter:
    def __init__(self, cfg: RecorderConfig, image_keys: List[str], image_shapes: Dict[str, tuple]) -> None:
        self.cfg = cfg
        self.image_keys = image_keys
        self.image_shapes = image_shapes
        self._ds = None
        self._mock = cfg.mock
        self._n_frames_episode = 0
        self._n_episodes = 0
        self._n_frames_total = 0
        self._finalized = False

    # ------------------------------------------------------------------ schema
    def _features(self) -> dict:
        img_dtype = "video" if self.cfg.use_videos else "image"
        feats: dict = {}
        for key in self.image_keys:
            feats[f"observation.images.{key}"] = {
                "dtype": img_dtype,
                "shape": self.image_shapes[key],
                "names": ["height", "width", "channels"],
            }
        feats["observation.state"] = {"dtype": "float32", "shape": (STATE_DIM,), "names": state_names()}
        feats["action"] = {"dtype": "float32", "shape": (ACTION_DIM,), "names": action_names()}
        return feats

    # ------------------------------------------------------------------ lifecycle
    def open(self) -> None:
        if self._mock:
            print(f"[dataset] MOCK writer (repo_id={self.cfg.repo_id}, fps={self.cfg.fps})")
            return
        LeRobotDataset = _import_lerobot_dataset()
        root = os.path.expanduser(self.cfg.root)
        self._ds = LeRobotDataset.create(
            repo_id=self.cfg.repo_id,
            fps=self.cfg.fps,
            features=self._features(),
            root=root,
            robot_type=self.cfg.robot_type,
            use_videos=self.cfg.use_videos,
        )
        print(f"[dataset] created v3.0 dataset at {root} (repo_id={self.cfg.repo_id})")

    def add_frame(self, images: Dict[str, np.ndarray], state: np.ndarray, action: np.ndarray, task: str) -> None:
        self._n_frames_episode += 1
        self._n_frames_total += 1
        if self._mock:
            return
        frame: dict = {f"observation.images.{k}": images[k] for k in self.image_keys}
        frame["observation.state"] = np.asarray(state, dtype=np.float32)
        frame["action"] = np.asarray(action, dtype=np.float32)
        frame["task"] = task  # v3.0: task is a key in the frame dict
        self._ds.add_frame(frame)

    def save_episode(self) -> None:
        if not self._mock:
            self._ds.save_episode()
        self._n_episodes += 1
        print(f"[dataset] saved episode #{self._n_episodes} ({self._n_frames_episode} frames)")
        self._n_frames_episode = 0

    def abort_episode(self) -> None:
        """Discard the in-progress (unsaved) episode buffer (review 'Delete' / aborts)."""
        if not self._mock and self._ds is not None and hasattr(self._ds, "clear_episode_buffer"):
            try:
                self._ds.clear_episode_buffer(delete_images=True)
            except TypeError:
                self._ds.clear_episode_buffer()
            except Exception:
                pass
        if self._n_frames_episode:
            print(f"[dataset] discarded episode ({self._n_frames_episode} frames)")
        self._n_frames_episode = 0

    def finalize(self) -> None:
        """Close parquet writers / metadata footers (required by v3.0 before reuse)."""
        if self._finalized:
            return
        self._finalized = True
        if not self._mock and self._ds is not None and self._n_episodes > 0:
            try:
                self._ds.finalize()
                print("[dataset] finalized (parquet/metadata closed)")
            except Exception as e:
                print(f"[dataset] finalize failed: {e}")

    # ------------------------------------------------------------------ stats
    @property
    def num_episodes(self) -> int:
        return self._n_episodes

    @property
    def frames_in_episode(self) -> int:
        return self._n_frames_episode

    @property
    def total_frames(self) -> int:
        return self._n_frames_total
