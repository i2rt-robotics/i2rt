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
    p.add_argument("--config", default=None, help="config.yaml (robot/cameras/tasks/recorder defaults)")
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
    p.add_argument(
        "--no-review",
        action="store_true",
        help="auto-save every episode (skip the Keep/Delete review hold)",
    )
    p.add_argument(
        "--auto-arm",
        action="store_true",
        help="arm collection automatically on Start (record on the next teleop engage)",
    )
    p.add_argument("--resume", action="store_true", help="append to an existing dataset at --root")
    p.add_argument("--mock", action="store_true", help="synthetic cameras + teleop (no hardware/robot/lerobot)")
    p.add_argument(
        "--serials",
        default="",
        help="comma-separated RealSense serials for wrist_left,wrist_right,agentview",
    )
    args = p.parse_args(argv)

    # Unified rig config: explicit CLI flag > config.yaml section > built-in default.
    rig = load_rig(args.config)
    rec = Resolver(args, p, rig.get("recorder", {}))
    rob = Resolver(args, p, rig.get("robot", {}))

    cams = apply_camera_serials(default_cameras(), rig)  # rig 'cameras' by key
    if args.serials:  # explicit CLI serials win
        for cam, serial in zip(cams, [s.strip() for s in args.serials.split(",")], strict=False):
            cam.serial = serial

    tasks = [t.strip() for t in args.tasks.split(";") if t.strip()] or list(rig.get("tasks", []) or [])

    # Booleans: config.yaml recorder.* sets the baseline; the CLI flag forces it on.
    rec_section = rig.get("recorder", {}) or {}
    review_before_save = bool(rec_section.get("review_before_save", True)) and not args.no_review
    auto_arm = bool(rec_section.get("auto_arm", False)) or args.auto_arm

    cfg = RecorderConfig(
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
        review_before_save=review_before_save,
        auto_arm=auto_arm,
    )
    buttons = rec_section.get("buttons")  # leader button -> outcome (keeps the built-in default if unset)
    if buttons:
        cfg.button_map = {str(k): str(v) for k, v in buttons.items()}
    # video-encoding knobs (saving speed): config.yaml recorder.* overrides the defaults
    cfg.use_videos = bool(rec_section.get("use_videos", cfg.use_videos))
    cfg.vcodec = str(rec_section.get("vcodec", cfg.vcodec))
    cfg.encoder_threads = int(rec_section.get("encoder_threads", cfg.encoder_threads))
    cfg.batch_encoding_size = int(rec_section.get("batch_encoding_size", cfg.batch_encoding_size))
    return cfg


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
