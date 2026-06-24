"""Configuration for the LeRobot recorder (cameras, topics, dataset schema)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

# ----------------------------------------------------------------------------
# Robot / dataset dimensions
# ----------------------------------------------------------------------------
ARMS = ("left", "right")
ARM_DOF = 7  # 6 joints + gripper per arm (follower)
LEADER_ARM_DOF = 6  # teaching-handle leader exposes 6 arm joints (gripper is a passive trigger)

# The recorded schema is FIXED from the robot's known outputs (no runtime probe).
# MotorChainRobot.get_observations() gives joint_pos/vel/eff (+ gripper_*), so:
#   observation.state  = per arm [pos(7), vel(7), eff(7)]  -> 42
#   observation.leader = per arm leader joint positions    -> 12
#   action             = per arm applied/human action [7]  -> 14
# (The robot exposes no end-effector pose, so observation.eef is not recorded.)
STATE_DIM = len(ARMS) * ARM_DOF * 3  # 42
ACTION_DIM = len(ARMS) * ARM_DOF  # 14
LEADER_DIM = len(ARMS) * LEADER_ARM_DOF  # 12
EEF_POSE_DIM = 7  # per arm: position(3) + quaternion wxyz(4)
EEF_DIM = len(ARMS) * EEF_POSE_DIM  # 14 (zeros when the robot can't provide FK)

# Per-frame control-mode label (always written as observation.control_mode), so a
# dataset records whether each frame came from teleop, a policy, an intervention,
# or replay — useful for HG-DAgger filtering and provenance.
CONTROL_MODE = {"teleop": 0, "policy": 1, "intervention": 2, "replay": 3}


def state_names() -> List[str]:
    names: List[str] = []
    for arm in ARMS:
        for field_ in ("pos", "vel", "eff"):
            names += [f"{arm}.{field_}.{i}" for i in range(ARM_DOF)]
    return names


def action_names() -> List[str]:
    return [f"{arm}.act.{i}" for arm in ARMS for i in range(ARM_DOF)]


def leader_names() -> List[str]:
    return [f"{arm}.leader.{i}" for arm in ARMS for i in range(LEADER_ARM_DOF)]


def eef_names() -> List[str]:
    comp = ["x", "y", "z", "qw", "qx", "qy", "qz"]
    return [f"{arm}.eef.{c}" for arm in ARMS for c in comp]


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
    task: str = "do the task"  # ACTIVE language instruction (persists until changed)
    tasks: List[str] = field(default_factory=list)  # quick-switch templates shown in the GUI
    fps: int = 60  # dataset / record-loop rate (matched to the 60 fps cameras)
    robot_type: str = "yam_bimanual"
    use_videos: bool = True
    # Video encoding speed (LeRobot). The default 'libsvtav1' (AV1) is SLOW to encode;
    # 'h264' is much faster, 'auto' picks a hardware encoder (e.g. nvenc) if available.
    vcodec: str = "libsvtav1"
    encoder_threads: int = 0  # threads per video encode (0 = let LeRobot decide)
    # >1 accumulates that many episodes and encodes their videos in PARALLEL (LeRobot's
    # ProcessPoolExecutor) — the supported way to use more than one encoder "worker".
    batch_encoding_size: int = 1
    cameras: List[CameraSpec] = field(default_factory=default_cameras)
    # Robot link: the YAM robot machine running i2rt.serving.run_robot_server (portal).
    robot_host: str = "127.0.0.1"
    robot_port: int = 11331
    # What drives episode gating + the recorded action:
    #   "teleop" -> gate on teleop_state (ENGAGED..IDLE), action = applied command
    #   "dagger" -> gate on intervention (HG-DAgger), action = human (leader) action
    record_source: str = "teleop"
    resume: bool = False  # append to an existing dataset at `root` instead of creating a new one
    min_free_gb: float = 1.0  # refuse to save an episode when free disk drops below this
    mock: bool = False  # synthetic cameras + teleop stream (no hardware / robot)
    review_before_save: bool = True  # hold each episode for Keep/Delete instead of auto-saving
    auto_arm: bool = False  # arm collection automatically on Start (record on the next teleop engage)
    review_cam: str = "agentview"  # which camera to buffer (downsampled) for review playback
    # Leader-handle button -> episode outcome, keyed "<side>.<button_index>" (upper=0,
    # lower=1). Default: left lower=success, right lower=fail, both uppers=discard — so
    # all three outcomes are reachable with two buttons per arm.
    button_map: dict = field(
        default_factory=lambda: {
            "left.0": "discard",
            "left.1": "success",
            "right.0": "discard",
            "right.1": "fail",
        }
    )
    review_downscale: int = 4  # spatial stride for the review preview (640x480 -> 160x120)
