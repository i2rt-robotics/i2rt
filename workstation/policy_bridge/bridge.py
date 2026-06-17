"""Policy bridge: robot (portal) <-> policy (websocket), on the workstation.

Topology 2 (deployment):

    robot server  ──portal──▶  THIS bridge  ──websocket+msgpack──▶  policy server
       (i2rt.serving)              │ build openpi obs, infer            (yam_policy)
                       ◀──portal───┘ set_policy_action(chunk step)

Each tick the bridge:
  1. reads robot state via ``RobotClient.get_observation()`` and cameras locally,
  2. builds an openpi-style obs dict (``observation/state``, ``observation/images/*``,
     ``prompt``),
  3. queries the policy via ``ActionChunkBroker(WebsocketClientPolicy)`` — only
     hitting the (possibly remote) server every ``action_horizon`` steps,
  4. splits the action into per-arm targets and sends them with
     ``RobotClient.set_policy_action({"left": ..., "right": ...})``.

The robot must run in **dagger** mode (policy drives the followers; a human can take
over with a handle button), or **wrapper** mode. The policy env is unconstrained —
it never imports ROS or i2rt.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from yam_policy import ActionChunkBroker, WebsocketClientPolicy, image_tools

from workstation.lerobot_recorder.cameras import CameraManager
from workstation.lerobot_recorder.config import ARM_DOF, ARMS, RecorderConfig

logger = logging.getLogger(__name__)


@dataclass
class BridgeConfig:
    robot_host: str = "127.0.0.1"
    robot_port: int = 11331
    policy_host: str = "127.0.0.1"
    policy_port: int = 8000
    action_horizon: int = 16
    rate_hz: float = 30.0
    image_size: int = 224
    prompt: str = "do the task"
    # camera role -> obs key sent to the policy (align to your policy's config)
    image_keys: Dict[str, str] = field(
        default_factory=lambda: {
            "agentview": "observation/images/agentview",
            "wrist_left": "observation/images/wrist_left",
            "wrist_right": "observation/images/wrist_right",
        }
    )


class PolicyBridge:
    def __init__(self, cfg: BridgeConfig, recorder_cfg: RecorderConfig):
        from i2rt.serving.robot_client import RobotClient

        self.cfg = cfg
        self.cameras = CameraManager(recorder_cfg)
        self.robot = RobotClient(host=cfg.robot_host, port=cfg.robot_port)
        self.policy = ActionChunkBroker(
            WebsocketClientPolicy(host=cfg.policy_host, port=cfg.policy_port),
            action_horizon=cfg.action_horizon,
        )
        self._stop = False

    # ---- obs assembly -------------------------------------------------------
    def _build_obs(self, robot_obs: Dict, images: Dict[str, np.ndarray]) -> Dict:
        state_parts: List[np.ndarray] = []
        for a in ARMS:
            side = robot_obs.get(a)
            if not side or "pos" not in side:
                return {}
            state_parts.append(np.asarray(side["pos"], dtype=np.float32))
        obs = {"observation/state": np.concatenate(state_parts), "prompt": self.cfg.prompt}
        for role, key in self.cfg.image_keys.items():
            if role in images:
                img = image_tools.resize_with_pad(images[role], self.cfg.image_size, self.cfg.image_size)
                obs[key] = image_tools.convert_to_uint8(img)
        return obs

    @staticmethod
    def _split(action: np.ndarray) -> Dict[str, np.ndarray]:
        return {arm: np.asarray(action[i * ARM_DOF : (i + 1) * ARM_DOF], dtype=float) for i, arm in enumerate(ARMS)}

    # ---- run loop -----------------------------------------------------------
    def run(self) -> None:
        self.cameras.start()
        logger.info(
            "PolicyBridge up: robot=%s:%d policy=%s:%d horizon=%d rate=%.0f Hz",
            self.cfg.robot_host,
            self.cfg.robot_port,
            self.cfg.policy_host,
            self.cfg.policy_port,
            self.cfg.action_horizon,
            self.cfg.rate_hz,
        )
        period = 1.0 / max(self.cfg.rate_hz, 1.0)
        try:
            while not self._stop:
                t0 = time.monotonic()
                try:
                    robot_obs = self.robot.get_observation()
                    images = self.cameras.read()
                    obs = self._build_obs(robot_obs, images)
                    if obs:
                        action = self.policy.infer(obs)["actions"]
                        self.robot.set_policy_action(self._split(np.asarray(action, dtype=float)))
                except Exception as e:
                    logger.warning("bridge tick failed: %s", e)
                remaining = period - (time.monotonic() - t0)
                if remaining > 0:
                    time.sleep(remaining)
        except KeyboardInterrupt:
            pass
        finally:
            self.cameras.stop()

    def stop(self) -> None:
        self._stop = True
