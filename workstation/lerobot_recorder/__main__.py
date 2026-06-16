"""Entry point: launch the recorder GUI.

    python -m workstation.lerobot_recorder --repo-id user/yam_pick --task "pick the cube"
    python -m workstation.lerobot_recorder --mock      # no robot/cameras/lerobot

Camera serials and other defaults live in ``config.py`` (or pass --serials).
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from workstation.lerobot_recorder.config import RecorderConfig, default_cameras


def build_config(argv: Optional[List[str]] = None) -> RecorderConfig:
    p = argparse.ArgumentParser(description="YAM ↔ LeRobot dataset recorder")
    p.add_argument("--repo-id", default="user/yam_bimanual")
    p.add_argument("--root", default="~/lerobot_data")
    p.add_argument("--task", default="do the task", help="language instruction")
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--mock", action="store_true", help="synthetic cameras + teleop (no hardware/ROS/lerobot)")
    p.add_argument(
        "--serials",
        default="",
        help="comma-separated RealSense serials for wrist_left,wrist_right,agentview",
    )
    args = p.parse_args(argv)

    cams = default_cameras()
    if args.serials:
        serials = [s.strip() for s in args.serials.split(",")]
        for cam, serial in zip(cams, serials, strict=False):
            cam.serial = serial

    return RecorderConfig(
        repo_id=args.repo_id, root=args.root, task=args.task, fps=args.fps, cameras=cams, mock=args.mock
    )


def main(argv: Optional[List[str]] = None) -> None:
    cfg = build_config(argv)
    from PyQt5 import QtWidgets

    from workstation.lerobot_recorder.gui import RecorderGUI

    app = QtWidgets.QApplication(sys.argv)
    gui = RecorderGUI(cfg)
    gui.resize(720, 420)
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
