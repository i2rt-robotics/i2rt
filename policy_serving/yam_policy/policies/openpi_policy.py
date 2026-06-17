"""TEMPLATE: serve a real openpi checkpoint through this server.

Usually you'd just run openpi's own ``scripts/serve_policy.py`` — but this shows
how to host an openpi policy behind *our* server so every policy you deploy uses
one launcher (:mod:`yam_policy.serve`). Install ``openpi`` into THIS env.

    uv pip install -e /path/to/openpi   # heavy: JAX/torch, GPU

Then:
    python -m yam_policy.serve \
        --policy yam_policy.policies.openpi_policy:OpenPiPolicy \
        --config config_name=pi0_fast_droid \
        --config checkpoint_dir=/abs/path/to/checkpoint
"""

from __future__ import annotations

from typing import Dict

from ..base_policy import BasePolicy


class OpenPiPolicy(BasePolicy):
    def __init__(self, config_name: str, checkpoint_dir: str) -> None:
        # Imported lazily so this module imports fine in envs without openpi.
        from openpi.policies import policy_config
        from openpi.training import config as openpi_config

        cfg = openpi_config.get_config(config_name)
        self._policy = policy_config.create_trained_policy(cfg, checkpoint_dir)

    def infer(self, obs: Dict) -> Dict:
        # openpi policies already return {"actions": (horizon, dim), ...}.
        return self._policy.infer(obs)

    def reset(self) -> None:
        if hasattr(self._policy, "reset"):
            self._policy.reset()
