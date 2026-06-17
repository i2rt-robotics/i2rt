"""Generic launcher: serve any :class:`BasePolicy` over websocket.

    python -m yam_policy.serve --policy <module>:<Class> [--config k=v ...]

Examples::

    # zero-model smoke test (holds the robot's pose):
    python -m yam_policy.serve

    # your LeRobot policy:
    python -m yam_policy.serve \
        --policy yam_policy.policies.lerobot_policy:LeRobotPolicy \
        --config pretrained_path=/abs/path --config device=cuda

``--config k=v`` pairs become keyword args to the policy constructor; values are
parsed with ``ast.literal_eval`` when possible (so ``action_dim=14`` is an int),
otherwise kept as strings.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import logging
from typing import Any, Dict, List

from .base_policy import BasePolicy
from .websocket_server import WebsocketPolicyServer


def _parse_config(pairs: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"--config expects key=value, got {p!r}")
        k, v = p.split("=", 1)
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    return out


def load_policy(spec: str, config: Dict[str, Any]) -> BasePolicy:
    if ":" not in spec:
        raise ValueError(f"--policy must be 'module:Class', got {spec!r}")
    module_path, cls_name = spec.split(":", 1)
    cls = getattr(importlib.import_module(module_path), cls_name)
    return cls(**config)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Serve a policy over websocket (openpi-compatible).")
    p.add_argument("--policy", default="yam_policy.policies.dummy:DummyPolicy", help="module:Class")
    p.add_argument("--config", action="append", default=[], help="key=value policy kwargs (repeatable)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    config = _parse_config(args.config)
    logging.info("Loading policy %s with config %s", args.policy, config)
    policy = load_policy(args.policy, config)

    # Advertise the obs/action spec so the bridge can self-configure (action_horizon,
    # image keys/size). A policy declares these via attributes / an `obs_spec` dict.
    metadata = {"policy": args.policy, "config": {k: str(v) for k, v in config.items()}}
    if hasattr(policy, "action_horizon"):
        metadata["action_horizon"] = int(policy.action_horizon)
    if isinstance(getattr(policy, "obs_spec", None), dict):
        metadata.update(policy.obs_spec)

    server = WebsocketPolicyServer(policy, host=args.host, port=args.port, metadata=metadata)
    server.serve_forever()


if __name__ == "__main__":
    main()
