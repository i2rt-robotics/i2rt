"""Entry point: launch the dataset replay GUI.

    python -m workstation.lerobot_recorder.replay_main --robot-host <ROBOT_IP> --repo-id user/yam_pick
    python -m workstation.lerobot_recorder.replay_main --mock      # synthetic dataset, no robot/lerobot

With "Send to robot" enabled, the robot side must run the wrapper server:
    scripts/yam wrapper
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from workstation.lerobot_recorder.config import RecorderConfig


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="YAM ↔ LeRobot dataset replay")
    p.add_argument("--repo-id", default="user/yam_bimanual")
    p.add_argument("--root", default="~/lerobot_data")
    p.add_argument("--robot-host", default="127.0.0.1", help="YAM robot server host (run_robot_server wrapper)")
    p.add_argument("--robot-port", type=int, default=11331)
    p.add_argument("--mock", action="store_true", help="synthetic dataset (no robot/lerobot)")
    args = p.parse_args(argv)

    cfg = RecorderConfig(
        repo_id=args.repo_id, root=args.root, robot_host=args.robot_host, robot_port=args.robot_port, mock=args.mock
    )

    from PyQt5 import QtWidgets

    from workstation.lerobot_recorder.replay_gui import ReplayGUI

    app = QtWidgets.QApplication(sys.argv)
    gui = ReplayGUI(cfg)
    gui.resize(640, 560)
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
