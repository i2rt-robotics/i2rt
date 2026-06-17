"""Portal robot server — exposes a controller's snapshot + inputs over the network.

Runs the real-time control loop (``controller.step()``) on the robot at a fixed
rate, and serves these portal methods to the workstation:

* ``get_observation()``      -> latest snapshot dict (state/action/gate per side)
* ``get_metadata()``         -> {mode, sides, has_gripper}
* ``set_policy_action(data)``-> {side: position} (dagger / wrapper)
* ``set_intervention(flag)`` -> external gate override (dagger)
* ``command(data)``          -> {side: position} direct follower target (wrapper/replay)
* ``set_sim_engage(flag)``   -> force ENGAGED in sim (teleop)

This is the same ``portal`` pattern used by ``ServerRobot`` (minimum_gello) and the
flow-base controller — no ROS.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

import portal

if TYPE_CHECKING:
    from i2rt.serving.controllers import BaseController

logger = logging.getLogger(__name__)

DEFAULT_PORT = 11331


class RobotServer:
    def __init__(self, controller: "BaseController", port: int = DEFAULT_PORT, rate_hz: float = 120.0):
        self.controller = controller
        self.port = int(port)
        self.rate_hz = float(rate_hz)
        self._server = portal.Server(self.port)
        self._server.bind("get_observation", self.controller.snapshot)
        self._server.bind("get_metadata", self.controller.metadata)
        self._server.bind("set_policy_action", self.controller.set_policy_action)
        self._server.bind("set_intervention", self.controller.set_intervention)
        self._server.bind("command", self.controller.command)
        self._server.bind("set_sim_engage", self.controller.set_sim_engage)
        self._stop = threading.Event()

    def serve(self) -> None:
        self._server.start(block=False)
        logger.info(
            "RobotServer on portal :%d (mode=%s, rate=%.0f Hz) — Ctrl-C to stop",
            self.port,
            self.controller.mode,
            self.rate_hz,
        )
        dt = 1.0 / max(self.rate_hz, 1.0)
        try:
            while not self._stop.is_set():
                t0 = time.monotonic()
                self.controller.step()
                remaining = dt - (time.monotonic() - t0)
                if remaining > 0:
                    time.sleep(remaining)
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self) -> None:
        self._stop.set()
        try:
            self.controller.close()
        except Exception:
            pass
        try:
            self._server.close()
        except Exception:
            pass
