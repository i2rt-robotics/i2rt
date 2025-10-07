# lerobot/robots/i2rt/config.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from lerobot.robots import RobotConfig
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras import CameraConfig
from lerobot.teleoperators import TeleoperatorConfig

CAMERA_SETTINGS: Dict[str, int] = dict(fps=30, width=640, height=480)
HOST: str = "127.0.0.1"
# ---------------- Recording ----------------
@dataclass
class RecordingConfig:
    num_episodes: int = 20
    task_description: str = "Teleop leader - Followers"
    hf_repo_id: str = "zetanschy/i2rt_test"
    fps: int = 30
    episode_time_sec: int = 180
    reset_time_sec: int = 2
    use_videos: bool = True
    batch_encoding_size: int = 3

# ---------------- Followers (observations) ----------------
@dataclass
class I2RTFollowerEndpoint:
    name: str                  # e.g., "right", "left"
    host: str = HOST
    port: int = 1234        
    # Action/observation key prefix; if not set, code will fall back to `name`
    prefix: Optional[str] = None

@dataclass
class I2RTFollowerConfig(RobotConfig):
    followers: List[I2RTFollowerEndpoint] = field(
        default_factory=lambda: [
            I2RTFollowerEndpoint(name="right", port=1234, prefix="right"),
            I2RTFollowerEndpoint(name="left",  port=1235, prefix="left"),
        ]
    )

    cameras: Dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "teleop_left":  RealSenseCameraConfig(serial_number_or_name="148522073102", **CAMERA_SETTINGS),
            "teleop_right": RealSenseCameraConfig(serial_number_or_name="151222074096", **CAMERA_SETTINGS),
            "torso":        RealSenseCameraConfig(serial_number_or_name="148122071428", **CAMERA_SETTINGS),
            #"torso":        OpenCVCameraConfig(index_or_path='/dev/video0', **CAMERA_SETTINGS),
        }
    )

# ---------------- Leaders (actions) ----------------
@dataclass
class LeaderArmCfg:
    # Prefix used to emit action keys (e.g., "right.j0.pos")
    prefix: str                      # "right" or "left"
    host: str = HOST
    port: int = 2234                 

@dataclass
class i2rtLeaderConfig(TeleoperatorConfig):
    arms: List[LeaderArmCfg] = field(
        default_factory=lambda: [
            LeaderArmCfg(prefix="right", port=2234),
            LeaderArmCfg(prefix="left",  port=2235),
        ]
    )
    name: str = "i2rt_leaders"
