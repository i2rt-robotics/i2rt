"""ActionChunkBroker — serve a predicted action chunk one step at a time.

Mirrors ``openpi_client.action_chunk_broker``. Wrap any :class:`BasePolicy` (e.g.
a :class:`WebsocketClientPolicy`); the broker calls the inner policy only every
``action_horizon`` steps and returns one action per step from the cached chunk,
so you query the (possibly remote) policy ~once per N control ticks.

The inner policy must return ``{"actions": ndarray(action_horizon, action_dim), ...}``.
Each ``infer`` returns the same dict with array fields sliced to the current step.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .base_policy import BasePolicy


class ActionChunkBroker(BasePolicy):
    def __init__(self, policy: BasePolicy, action_horizon: int) -> None:
        self._policy = policy
        self._action_horizon = int(action_horizon)
        self._cur_step = 0
        self._last_results: Dict | None = None

    def infer(self, obs: Dict) -> Dict:
        if self._last_results is None:
            self._last_results = self._policy.infer(obs)
            self._cur_step = 0

        results = {
            k: (v[self._cur_step, ...] if isinstance(v, np.ndarray) and v.ndim > 0 else v)
            for k, v in self._last_results.items()
        }

        self._cur_step += 1
        if self._cur_step >= self._action_horizon:
            self._last_results = None

        return results

    def reset(self) -> None:
        self._last_results = None
        self._cur_step = 0
        self._policy.reset()
