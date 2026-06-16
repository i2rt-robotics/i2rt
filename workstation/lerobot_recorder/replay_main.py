"""Entry point: launch the dataset replay GUI.

    python -m workstation.lerobot_recorder.replay_main --repo-id user/yam_pick --root ~/lerobot_data
    python -m workstation.lerobot_recorder.replay_main --mock      # synthetic dataset, no ROS/lerobot

With "Send to robot" enabled, the robot side must run the wrapper:
    scripts/yam wrapper --arm left:can_follower_l --arm right:can_follower_r
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
    p.add_argument("--mock", action="store_true", help="synthetic dataset (no ROS/lerobot)")
    args = p.parse_args(argv)

    cfg = RecorderConfig(repo_id=args.repo_id, root=args.root, mock=args.mock)

    from PyQt5 import QtWidgets

    from workstation.lerobot_recorder.replay_gui import ReplayGUI

    app = QtWidgets.QApplication(sys.argv)
    gui = ReplayGUI(cfg)
    gui.resize(640, 560)
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
