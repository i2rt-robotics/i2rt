"""Recorder data-collection (mock) + DAgger-source assembly + outcome sidecar tests."""

from __future__ import annotations

import json
import time

import numpy as np

from workstation.lerobot_recorder.config import RecorderConfig
from workstation.lerobot_recorder.portal_bridge import PortalBridge
from workstation.lerobot_recorder.recorder import Recorder


def test_recorder_records_episode_and_outcome(tmp_path):
    cfg = RecorderConfig(repo_id="test/yam", root=str(tmp_path), fps=60, mock=True)
    rec = Recorder(cfg)
    rec.start()
    rec.arm()

    captured, seen = False, set()
    t0 = time.time()
    while time.time() - t0 < 10:
        st = rec.get_status()
        seen.add(st["teleop"])
        if st["pending"]:
            captured = True
            rec.keep_episode(outcome="success")
            break
        time.sleep(0.05)
    st = rec.get_status()
    rec.shutdown()

    assert captured, "gate never produced a pending episode"
    assert "ENGAGED" in seen and "IDLE" in seen
    assert st["episodes"] >= 1

    sidecar = tmp_path / "outcomes.jsonl"
    assert sidecar.exists()
    entry = json.loads(sidecar.read_text().splitlines()[0])
    assert entry["outcome"] == "success"
    assert entry["episode"] == 0


def test_dagger_source_assembly():
    cfg = RecorderConfig(record_source="dagger", mock=False)
    bridge = PortalBridge(cfg)
    human_l, human_r = np.arange(7, dtype=float), np.arange(7, 14, dtype=float)
    pose = {"pos": [0.0] * 7, "vel": [0.0] * 7, "eff": [0.0] * 7}

    intervening = {
        "intervention": True,
        "left": {**pose, "human": human_l.tolist()},
        "right": {**pose, "human": human_r.tolist()},
        "t": 1.0,
    }
    snap = bridge._assemble(intervening)
    assert snap["teleop_state"] == "ENGAGED"
    assert snap["action"] is not None and snap["action"].shape == (14,)
    assert np.allclose(snap["action"], np.concatenate([human_l, human_r]))

    idle = {"intervention": False, "left": pose, "right": pose, "t": 2.0}
    snap_idle = bridge._assemble(idle)
    assert snap_idle["teleop_state"] == "IDLE"
    assert snap_idle["action"] is None  # not intervening -> nothing to record
