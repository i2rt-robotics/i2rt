"""DummyPolicy — a no-model policy for testing the whole serving loop.

Returns an action chunk of shape ``(action_horizon, action_dim)``. The default
``mode="hold"`` repeats the incoming ``observation/state`` (when its size matches
``action_dim``) so a connected robot simply holds its pose — the safe choice for
an end-to-end smoke test. ``"zeros"`` and ``"random"`` are also available.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from ..base_policy import BasePolicy


class DummyPolicy(BasePolicy):
    def __init__(self, action_dim: int = 14, action_horizon: int = 16, mode: str = "hold", seed: int = 0) -> None:
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.mode = mode
        self._rng = np.random.default_rng(seed)

    def infer(self, obs: Dict) -> Dict:
        h, d = self.action_horizon, self.action_dim
        if self.mode == "hold":
            state = np.asarray(obs.get("observation/state", []), dtype=np.float32).reshape(-1)
            base = state[:d] if state.size >= d else np.zeros(d, dtype=np.float32)
            actions = np.tile(base, (h, 1)).astype(np.float32)
        elif self.mode == "random":
            actions = self._rng.standard_normal((h, d)).astype(np.float32) * 0.02
        else:  # zeros
            actions = np.zeros((h, d), dtype=np.float32)
        return {"actions": actions}
