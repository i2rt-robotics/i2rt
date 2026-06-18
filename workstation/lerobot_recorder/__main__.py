"""Entry point: launch the recorder GUI.

    python -m workstation.lerobot_recorder --repo-id user/yam_pick --task "pick the cube"
    python -m workstation.lerobot_recorder --mock      # no robot/cameras/lerobot

Camera serials and other defaults live in ``config.py`` (or pass --serials).
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from i2rt.serving.rig_config import Resolver, apply_camera_serials, load_rig
from workstation.lerobot_recorder.config import RecorderConfig, default_cameras


def build_config(argv: Optional[List[str]] = None) -> RecorderConfig:
    p = argparse.ArgumentParser(description="YAM ↔ LeRobot dataset recorder")
    p.add_argument("--config", default=None, help="rig.yaml (robot/cameras/tasks/recorder defaults)")
    p.add_argument("--repo-id", default="user/yam_bimanual")
    p.add_argument("--root", default="~/lerobot_data")
    p.add_argument("--task", default="do the task", help="active language instruction")
    p.add_argument(
        "--tasks",
        default="",
        help="';'-separated task templates for quick switching in the GUI (active task persists)",
    )
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--robot-host", default="127.0.0.1", help="YAM robot server host (run_robot_server)")
    p.add_argument("--robot-port", type=int, default=11331)
    p.add_argument(
        "--source",
        choices=["teleop", "dagger", "eval"],
        default="teleop",
        help="teleop = gate on teleop_state (action=applied); dagger = HG-DAgger interventions "
        "(action=human); eval = record a policy rollout from Start to Stop (action=executed)",
    )
    p.add_argument("--min-free-gb", type=float, default=1.0, help="refuse to save below this free disk")
    p.add_argument("--resume", action="store_true", help="append to an existing dataset at --root")
    p.add_argument("--mock", action="store_true", help="synthetic cameras + teleop (no hardware/robot/lerobot)")
    p.add_argument(
        "--serials",
        default="",
        help="comma-separated RealSense serials for wrist_left,wrist_right,agentview",
    )
    args = p.parse_args(argv)

    # Unified rig config: explicit CLI flag > rig.yaml section > built-in default.
    rig = load_rig(args.config)
    rec = Resolver(args, p, rig.get("recorder", {}))
    rob = Resolver(args, p, rig.get("robot", {}))

    cams = apply_camera_serials(default_cameras(), rig)  # rig 'cameras' by key
    if args.serials:  # explicit CLI serials win
        for cam, serial in zip(cams, [s.strip() for s in args.serials.split(",")], strict=False):
            cam.serial = serial

    tasks = [t.strip() for t in args.tasks.split(";") if t.strip()] or list(rig.get("tasks", []) or [])

    return RecorderConfig(
        repo_id=rec.get("repo_id"),
        root=rec.get("root"),
        task=rec.get("task"),
        tasks=tasks,
        fps=int(rec.get("fps")),
        cameras=cams,
        robot_host=rob.get("robot_host", key="host"),
        robot_port=int(rob.get("robot_port", key="port")),
        record_source=rec.get("source"),
        resume=args.resume,
        min_free_gb=float(rec.get("min_free_gb")),
        mock=args.mock,
    )


def main(argv: Optional[List[str]] = None) -> None:
    cfg = build_config(argv)
    from PyQt5 import QtWidgets

    from workstation.lerobot_recorder import theme
    from workstation.lerobot_recorder.gui import RecorderGUI

    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(theme.QSS)  # app-level so every window/dialog is themed
    gui = RecorderGUI(cfg)
    gui.resize(760, 900)
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
