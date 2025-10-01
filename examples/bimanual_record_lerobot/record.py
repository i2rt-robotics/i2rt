#!/usr/bin/env python3
# Copyright 2025
# Record a LeRobot-format dataset from a custom portal-based robot.

from __future__ import annotations
from dataclasses import dataclass

# ---- LeRobot imports ----
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.processor import make_default_processors
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.constants import OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

from i2rt import I2RTRobot, PortalLeaderTeleop
from config import I2RTFollowerConfig, i2rtLeaderConfig, RecordingConfig

import os, shutil
from pathlib import Path

def _clear_local_lerobot_cache(repo_id: str) -> None:
    # expands to ~/.cache/huggingface/lerobot/<namespace>/<name>
    cache_root = Path(os.path.expanduser("~")) / ".cache" / "huggingface" / "lerobot"
    cache_dir = cache_root / repo_id  # repo_id like "user/name"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

# ==================== Main ====================
def main():
    recording_cfg = RecordingConfig()
    _clear_local_lerobot_cache(recording_cfg.hf_repo_id)

    # ---- Build robot (followers) ----
    robot_cfg = I2RTFollowerConfig()
    robot = I2RTRobot(robot_cfg)

    # ---- Teleop (leader) ----
    teleop_cfg = i2rtLeaderConfig()
    teleop = PortalLeaderTeleop(teleop_cfg)

    # ---- Connect endpoints FIRST ----
    print("Connecting robot...")
    robot.connect()
    print(f"Robot connected: {robot.is_connected}")
    
    print("Connecting teleop...")
    teleop.connect()
    print(f"Teleop connected: {teleop.is_connected}")

    if not robot.is_connected or not teleop.is_connected:
        raise RuntimeError("Robot or teleop is not connected!")

    # ---- Processors (default identity) ----
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # ---- Dataset features (AFTER connection) ----
    print("Raw robot features:")
    print(f"  - robot.action_features: {robot.action_features}")
    print(f"  - robot.observation_features: {robot.observation_features}")
    
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features    = hw_to_dataset_features(robot.observation_features, OBS_STR)
    dataset_features = {**action_features, **obs_features}
    
    print("Dataset features:")
    print(f"  - Action features: {action_features}")
    print(f"  - Observation features: {obs_features}")
    print(f"  - Total features: {len(dataset_features)}")

    # ---- Create dataset ----
    dataset = LeRobotDataset.create(
        repo_id=recording_cfg.hf_repo_id,
        fps=recording_cfg.fps,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=recording_cfg.use_videos,
        image_writer_threads=4,
        batch_encoding_size=1,
    )

    # ---- UI helpers ----
    listener, events = init_keyboard_listener()
    init_rerun(session_name="i2rt_record")
    
    # Test data flow
    print("Testing data flow...")
    test_action = teleop.get_action()
    test_obs = robot.get_observation()
    print(f"Action keys: {list(test_action.keys())}")
    print(f"Observation keys: {list(test_obs.keys())}")
    print(f"Action sample: {dict(list(test_action.items())[:3])}")
    print(f"Observation sample: {dict(list(test_obs.items())[:3])}")
    
    # Check robot feature definitions
    print(f"Robot action features: {robot.action_features}")
    print(f"Robot observation features: {robot.observation_features}")
    print(f"Robot motors features: {robot.motors_features}")
    print(f"Robot camera features: {robot.camera_features}")
    
    # Test if robot methods work
    print("Testing robot methods directly...")
    print(f"Robot is_connected: {robot.is_connected}")
    print(f"Robot is_calibrated: {robot.is_calibrated}")
    
    # Test observation method directly
    print("Calling robot.get_observation() directly...")
    direct_obs = robot.get_observation()
    print(f"Direct observation keys: {list(direct_obs.keys())}")
    print(f"Direct observation sample: {dict(list(direct_obs.items())[:3])}")

    print("Starting record loop...")
    recorded_episodes = 0
    try:
        while recorded_episodes < recording_cfg.num_episodes and not events["stop_recording"]:
            log_say(f"Recording episode {recorded_episodes}")

            # Main record loop
            print(f"Starting record loop for episode {recorded_episodes}...")
            record_loop(
                robot=robot,
                events=events,
                fps=recording_cfg.fps,
                dataset=dataset,
                teleop=teleop,  # single teleop (multi-teleop list is restricted to LeKiwi in upstream)
                control_time_s=recording_cfg.episode_time_sec,
                single_task=recording_cfg.task_description,
                display_data=True,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
            )
            
            # Debug: Check what's in the episode buffer
            print(f"Episode buffer info after recording:")
            print(f"  - Buffer length: {len(dataset.episode_buffer)}")
            if len(dataset.episode_buffer) > 0:
                sample_frame = dataset.episode_buffer
                print(f"  - Sample frame keys: {list(sample_frame.keys())}")
                print(f"  - Action keys: {[k for k in sample_frame.keys() if 'action' in k]}")
                print(f"  - Observation keys: {[k for k in sample_frame.keys() if 'observation' in k]}")
                print(f"  - Camera keys: {[k for k in sample_frame.keys() if any(cam in k for cam in ['teleop_left', 'teleop_right', 'torso'])]}")
                print(f"  - Motor keys: {[k for k in sample_frame.keys() if '.j' in k]}")
                
                # Check if we have the test key
                if 'aaaa' in sample_frame:
                    print(f"  - Test key 'aaaa' found: {sample_frame['aaaa']}")
                else:
                    print("  - Test key 'aaaa' NOT found!")
                    
                # Check a few sample values
                print(f"  - Sample values:")
                for key in list(sample_frame.keys())[:5]:
                    if isinstance(sample_frame[key], (int, float, str)):
                        print(f"    {key}: {sample_frame[key]}")
                    else:
                        print(f"    {key}: {type(sample_frame[key])} (shape: {getattr(sample_frame[key], 'shape', 'N/A')})")
            else:
                print("  - WARNING: Episode buffer is empty!")

            # Reset phase (not saved)
            if not events["stop_recording"] and (
                (recorded_episodes < recording_cfg.num_episodes - 1) or events["rerecord_episode"]
            ):
                log_say("Reset the environment")
                record_loop(
                    robot=robot,
                    events=events,
                    fps=recording_cfg.fps,
                    teleop=teleop,
                    control_time_s=recording_cfg.reset_time_sec,
                    single_task=recording_cfg.task_description,
                    display_data=True,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                )

            if events["rerecord_episode"]:
                log_say("Re-record episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            # Save episode
            print(f"Saving episode {recorded_episodes}...")
            
            # Debug: Check what's in the buffer before saving
            print(f"Before save - Buffer length: {len(dataset.episode_buffer)}")
            if len(dataset.episode_buffer) > 0:
                print(f"Before save - Sample keys: {list(dataset.episode_buffer.keys())}")
            
            dataset.save_episode()
            
            # Debug: Check what's in the buffer after saving
            print(f"After save - Buffer length: {len(dataset.episode_buffer)}")
            
            print(f"✓ Episode {recorded_episodes} saved successfully")
            recorded_episodes += 1

    finally:
        # ---- Clean up ----
        log_say("Stop recording")
        try:
            teleop.disconnect()
        except Exception:
            pass
        try:
            robot.disconnect()
        except Exception:
            pass
        try:
            if listener is not None:
                listener.stop()
        except Exception:
            pass

if __name__ == "__main__":
    main()
