"""Dataset replay over portal: mock dataset -> sim wrapper robot."""

from __future__ import annotations

import threading
import time

import pytest

pytest.importorskip("portal")
pytest.importorskip("mujoco")

from i2rt.serving.controllers import WrapperConfig, WrapperController
from i2rt.serving.robot_client import RobotClient
from i2rt.serving.robot_server import RobotServer
from tests._util import free_port, wait_port
from workstation.lerobot_recorder.config import RecorderConfig
from workstation.lerobot_recorder.dataset_reader import DatasetReader
from workstation.lerobot_recorder.replay_controller import ReplayController


def test_replay_streams_to_robot():
    port = free_port()
    srv = RobotServer(WrapperController(WrapperConfig(sim=True, rate=100.0)), port=port, rate_hz=100.0)
    threading.Thread(target=srv.serve, daemon=True).start()
    assert wait_port(port), "robot server did not start"

    reader = DatasetReader("mock/ds", "~/x", mock=True)
    reader.load()
    ep_len = reader.episode_length(0)

    played = []
    cfg = RecorderConfig(robot_host="127.0.0.1", robot_port=port, mock=False)
    ctrl = ReplayController(reader, cfg, on_frame=played.append)
    ctrl.connect()
    time.sleep(1.2)  # background connect
    ctrl.set_speed(8.0)
    ctrl.play_episode(0, send_to_robot=True)

    t0 = time.time()
    while ctrl.playing and time.time() - t0 < 10:
        time.sleep(0.1)

    mon = RobotClient(host="127.0.0.1", port=port)
    applied = mon.get_observation()["left"]["applied"]
    ctrl.shutdown()
    srv.close()

    assert len(played) == ep_len
    assert applied is not None
