"""TEMPLATE: serve a LeRobot-trained policy (torch) through this server.

Copy this file and adapt the two clearly-marked sections to your policy:
  (1) how observations from the bridge map to your policy's input batch, and
  (2) how the policy's output maps to ``{"actions": (horizon, action_dim)}``.

Install your policy's deps (torch, lerobot, ...) into THIS env only — it never
needs the robot's Python.

    python -m yam_policy.serve \
        --policy yam_policy.policies.lerobot_policy:LeRobotPolicy \
        --config pretrained_path=outputs/train/my_act/checkpoints/last/pretrained_model \
        --config device=cuda
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from ..base_policy import BasePolicy


class LeRobotPolicy(BasePolicy):
    def __init__(self, pretrained_path: str, device: str = "cuda") -> None:
        import torch  # noqa: F401
        from lerobot.common.policies.factory import make_policy  # API name may vary by lerobot version

        self._device = device
        # NOTE: exact loader differs by lerobot version; adapt as needed.
        self._policy = make_policy(pretrained_path)  # type: ignore[call-arg]
        self._policy.to(device)
        self._policy.eval()

    def infer(self, obs: Dict) -> Dict:
        import torch

        # ---- (1) obs dict -> policy batch -------------------------------------
        # The bridge sends openpi-style keys, e.g.:
        #   obs["observation/state"]                -> (state_dim,)
        #   obs["observation/images/<cam>"]         -> (H, W, 3) uint8
        #   obs["prompt"]                           -> str (optional)
        batch = {
            "observation.state": torch.as_tensor(np.asarray(obs["observation/state"], dtype=np.float32))
            .unsqueeze(0)
            .to(self._device),
        }
        for k, v in obs.items():
            if k.startswith("observation/images/"):
                cam = k.split("/")[-1]
                img = torch.as_tensor(np.asarray(v)).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
                batch[f"observation.images.{cam}"] = img.to(self._device)

        # ---- (2) run policy -> action chunk -----------------------------------
        with torch.no_grad():
            action = self._policy.select_action(batch)  # adapt to your policy's API

        actions = action.squeeze(0).cpu().numpy().astype(np.float32)
        if actions.ndim == 1:  # single action -> a length-1 chunk
            actions = actions[None, :]
        return {"actions": actions}

    def reset(self) -> None:
        if hasattr(self._policy, "reset"):
            self._policy.reset()
