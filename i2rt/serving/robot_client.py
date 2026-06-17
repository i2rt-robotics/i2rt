"""Portal robot client — the workstation handle to the remote YAM rig.

Mirrors the ``ClientRobot`` pattern (minimum_gello): construct with the robot
host/port and call methods as if the rig were local. Reads return the decoded
result; setters are fire-and-forget (portal dispatches them).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import portal

from i2rt.serving.robot_server import DEFAULT_PORT


class RobotClient:
    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT):
        self._client = portal.Client(f"{host}:{port}")
        self.metadata: Dict = self._client.get_metadata().result()

    def get_observation(self) -> Dict:
        return self._client.get_observation().result()

    def set_policy_action(self, data: Dict[str, np.ndarray]) -> None:
        self._client.set_policy_action(data)

    def command(self, data: Dict[str, np.ndarray]) -> None:
        self._client.command(data)

    def set_intervention(self, flag: bool) -> None:
        self._client.set_intervention(bool(flag))

    def set_sim_engage(self, flag: bool) -> None:
        self._client.set_sim_engage(bool(flag))
