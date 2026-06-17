"""Configuration for the LeRobot recorder (cameras, topics, dataset schema)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

# ----------------------------------------------------------------------------
# Robot / dataset dimensions
# ----------------------------------------------------------------------------
ARMS = ("left", "right")
ARM_DOF = 7  # 6 joints + gripper per arm

# observation.state = per arm [pos(7), vel(7), eff(7)] concatenated over arms.
# action            = per arm applied_action [7] concatenated over arms.
STATE_DIM = len(ARMS) * ARM_DOF * 3  # 42
ACTION_DIM = len(ARMS) * ARM_DOF  # 14


def state_names() -> List[str]:
    names: List[str] = []
    for arm in ARMS:
        for field_ in ("pos", "vel", "eff"):
            names += [f"{arm}.{field_}.{i}" for i in range(ARM_DOF)]
    return names


def action_names() -> List[str]:
    return [f"{arm}.act.{i}" for arm in ARMS for i in range(ARM_DOF)]


# ----------------------------------------------------------------------------
# Cameras (RealSense). Map each physical serial to a role/image key.
# Two D405 on the wrists + one D455 agentview.
# ----------------------------------------------------------------------------
@dataclass
class CameraSpec:
    key: str  # dataset image key, e.g. "wrist_left"
    serial: str  # RealSense serial number ("" = pick any remaining device)
    width: int = 640
    height: int = 480
    fps: int = 60  # native stream fps (>= record fps so frames don't repeat)


def default_cameras() -> List[CameraSpec]:
    # Fill in the real serials (run `python -m workstation.lerobot_recorder.cameras --list`).
    return [
        CameraSpec("wrist_left", serial="", width=640, height=480, fps=60),
        CameraSpec("wrist_right", serial="", width=640, height=480, fps=60),
        CameraSpec("agentview", serial="", width=640, height=480, fps=60),
    ]


# ----------------------------------------------------------------------------
# Top-level recorder config
# ----------------------------------------------------------------------------
@dataclass
class RecorderConfig:
    repo_id: str = "user/yam_bimanual"
    root: str = "~/lerobot_data"
    task: str = "do the task"  # language instruction (per session; editable in GUI)
    fps: int = 60  # dataset / record-loop rate (matched to the 60 fps cameras)
    robot_type: str = "yam_bimanual"
    use_videos: bool = True
    cameras: List[CameraSpec] = field(default_factory=default_cameras)
    # Robot link: the YAM robot machine running i2rt.serving.run_robot_server (portal).
    robot_host: str = "127.0.0.1"
    robot_port: int = 11331
    mock: bool = False  # synthetic cameras + teleop stream (no hardware / robot)
    review_before_save: bool = True  # hold each episode for Keep/Delete instead of auto-saving
    review_cam: str = "agentview"  # which camera to buffer (downsampled) for review playback
    review_downscale: int = 4  # spatial stride for the review preview (640x480 -> 160x120)
