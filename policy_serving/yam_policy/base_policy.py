"""Policy interface — wire-compatible with openpi's ``openpi_client.BasePolicy``.

A policy maps an observation dict to an action dict. The canonical action key is
``"actions"`` holding an array of shape ``(action_horizon, action_dim)`` (an
action *chunk*); see :mod:`yam_policy.action_chunk_broker` for executing a chunk
open-loop.

Keeping this identical to openpi's contract means: a policy you serve with openpi
works against our client unchanged, and a policy written against this class can be
served by openpi's server. The only required method is :meth:`infer`.
"""

from __future__ import annotations

import abc
from typing import Dict


class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def infer(self, obs: Dict) -> Dict:
        """Infer actions from observations.

        Args:
            obs: observation dict (e.g. ``observation/state``, image keys, ``prompt``).

        Returns:
            A dict containing at least ``"actions"`` -> ndarray
            ``(action_horizon, action_dim)``.
        """

    def reset(self) -> None:  # noqa: B027  intentional optional no-op hook
        """Reset any per-episode state (chunk caches, RNN hidden state, ...)."""
