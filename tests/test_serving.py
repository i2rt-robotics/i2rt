"""Robot serving (portal) round-trip, e-stop, and controller-step tests (sim)."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

pytest.importorskip("portal")
pytest.importorskip("mujoco")  # sim robot

from i2rt.serving.controllers import (
    DaggerConfig,
    DaggerController,
    TeleopConfig,
    TeleopController,
    WrapperConfig,
    WrapperController,
)
from i2rt.serving.robot_client import RobotClient
from i2rt.serving.robot_server import RobotServer
from tests._util import free_port, wait_port


def test_controllers_step_sim():
    tc = TeleopController(TeleopConfig(sim=True))
    for _ in range(3):
        tc.step()
    snap = tc.snapshot()
    assert snap["teleop_state"] in ("HOMING", "IDLE", "ENGAGED")
    assert len(snap["left"]["pos"]) == 7
    tc.close()

    dc = DaggerController(DaggerConfig(sim=True))
    dc.set_policy_action({"left": np.zeros(7), "right": np.zeros(7)})
    for _ in range(3):
        dc.step()
    ds = dc.snapshot()
    assert ds["intervention"] is False
    assert len(ds["left"]["applied"]) == 7  # policy drives the follower when not intervening
    dc.close()


def test_teleop_bilateral_engage_steps():
    """Engage with bilateral_kp>0 runs through the (fixed) free-until-caught-up path."""
    tc = TeleopController(TeleopConfig(sim=True, bilateral_kp=0.2))
    tc.set_sim_engage(True)
    for _ in range(10):
        tc.step()
    snap = tc.snapshot()
    assert snap["teleop_state"] == "ENGAGED"
    assert snap["left"]["applied"] is not None
    tc.close()


def test_command_staleness_watchdog():
    """A wrapper follower holds (applied=None) once external commands go stale."""
    ctrl = WrapperController(WrapperConfig(sim=True, rate=100.0, command_timeout=0.2))
    ctrl.command({"left": np.zeros(7), "right": np.zeros(7)})
    ctrl.step()
    assert ctrl.snapshot()["left"]["applied"] is not None  # fresh command -> applied
    time.sleep(0.3)  # let the command go stale (link loss)
    ctrl.step()
    assert ctrl.snapshot()["left"]["applied"] is None  # stale -> hold
    ctrl.close()


def test_portal_roundtrip_and_estop():
    port = free_port()
    srv = RobotServer(WrapperController(WrapperConfig(sim=True, rate=100.0)), port=port, rate_hz=100.0)
    threading.Thread(target=srv.serve, daemon=True).start()
    assert wait_port(port), "robot server did not start"

    client = RobotClient(host="127.0.0.1", port=port)
    assert client.metadata["mode"] == "wrapper"

    client.command({"left": np.zeros(7), "right": np.zeros(7)})
    time.sleep(0.3)
    assert client.get_observation()["left"]["applied"] is not None

    # e-stop: commands are ignored while engaged
    client.set_estop(True)
    time.sleep(0.2)
    client.command({"left": np.ones(7), "right": np.ones(7)})
    time.sleep(0.3)
    obs = client.get_observation()
    assert bool(obs["estop"])  # portal serializes bool -> numpy scalar; check truthiness
    assert obs["left"]["applied"] is None

    client.set_estop(False)
    srv.close()
