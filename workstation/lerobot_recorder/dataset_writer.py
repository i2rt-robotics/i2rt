"""Thin wrapper over ``LeRobotDataset`` for recording bimanual YAM episodes.

Builds the feature schema (3 camera images + 42-d state + 14-d action), then
buffers frames per episode and saves on demand. The LeRobot API has shifted
across releases, so the few version-sensitive calls are isolated here with
fallbacks; adjust them in one place if your ``lerobot`` differs.

``mock=True`` skips ``lerobot`` entirely and just counts frames/episodes.
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np

from workstation.lerobot_recorder.config import ACTION_DIM, STATE_DIM, RecorderConfig, action_names, state_names


def _import_lerobot_dataset() -> type:
    """Import LeRobotDataset across known module paths."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # newer layout
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # older layout
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
            root=root,
            robot_type=self.cfg.robot_type,
            features=self._features(),
            use_videos=self.cfg.use_videos,
        )
        print(f"[dataset] created at {root} (repo_id={self.cfg.repo_id})")

    def add_frame(self, images: Dict[str, np.ndarray], state: np.ndarray, action: np.ndarray, task: str) -> None:
        self._n_frames_episode += 1
        self._n_frames_total += 1
        if self._mock:
            return
        frame: dict = {f"observation.images.{k}": images[k] for k in self.image_keys}
        frame["observation.state"] = np.asarray(state, dtype=np.float32)
        frame["action"] = np.asarray(action, dtype=np.float32)
        try:
            self._ds.add_frame(frame, task=task)  # newer API
        except TypeError:
            frame["task"] = task  # older API expects task in the frame
            self._ds.add_frame(frame)

    def save_episode(self) -> None:
        if not self._mock:
            self._ds.save_episode()
        self._n_episodes += 1
        print(f"[dataset] saved episode #{self._n_episodes} ({self._n_frames_episode} frames)")
        self._n_frames_episode = 0

    def abort_episode(self) -> None:
        """Discard the in-progress (unsaved) episode buffer."""
        if not self._mock and self._ds is not None and hasattr(self._ds, "clear_episode_buffer"):
            try:
                self._ds.clear_episode_buffer()
            except Exception:
                pass
        if self._n_frames_episode:
            print(f"[dataset] aborted episode ({self._n_frames_episode} frames discarded)")
        self._n_frames_episode = 0

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
