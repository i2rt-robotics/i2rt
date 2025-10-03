#!/usr/bin/env python3
# Copyright 2025
# Record a LeRobot-format dataset from a custom portal-based robot.

from __future__ import annotations
from dataclasses import dataclass
import sys
import tty
import termios

# ---- LeRobot imports ----
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.processor import make_default_processors
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.constants import OBS_STR
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

from i2rt import I2RTRobot, PortalLeaderTeleop
from config import I2RTFollowerConfig, i2rtLeaderConfig, RecordingConfig

import os, shutil
from pathlib import Path
import threading
import time

def _clear_local_lerobot_cache(repo_id: str) -> None:
    # expands to ~/.cache/huggingface/lerobot/<namespace>/<name>
    cache_root = Path(os.path.expanduser("~")) / ".cache" / "huggingface" / "lerobot"
    cache_dir = cache_root / repo_id  # repo_id like "user/name"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

def create_events_dict() -> dict:
    """Create a simple events dictionary for recording control."""
    return {
        "stop_recording": False,
        "rerecord_episode": False,
        "exit_early": False,
        "pause": False,
        "resume": False,
        "start_requested": False
    }

def get_key():
    """Get a single key press from the user."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def get_arrow_key():
    """Get arrow key or other special key press."""
    key = get_key()
    if key == '\x1b':  # ESC sequence
        next_key = get_key()
        if next_key == '[':
            arrow = get_key()
            if arrow == 'A':
                return 'up'
            elif arrow == 'B':
                return 'down'
            elif arrow == 'C':
                return 'right'
            elif arrow == 'D':
                return 'left'
    elif key == ' ':
        return 'space'
    elif key == '\r' or key == '\n':
        return 'enter'
    elif key == 'q':
        return 'quit'
    elif key == 's':
        return 'start'
    elif key == 'r':
        return 'rerecord'
    elif key == 'h':
        return 'help'
    return key

def keyboard_monitor_thread(events):
    """Monitor keyboard input in a separate thread."""
    print("🎹 Keyboard monitor started. Press 'Q' to quit, 'R' to exit episode early, 'H' for help")
    
    while not events["stop_recording"]:
        try:
            cmd = get_arrow_key()
            
            if cmd == "quit" or cmd == "q":
                print("\n🛑 Quit requested via keyboard!")
                events["stop_recording"] = True
                break
            elif cmd == "help" or cmd == "h":
                print("\n📖 Commands: [Q]uit anytime, [R]exit episode early, [S]tart next episode, [H]elp")
            elif cmd == "rerecord" or cmd == "r":
                print("\n⏹️ Exit episode early requested via keyboard!")
                events["exit_early"] = True
            elif cmd == "start" or cmd == "s":
                print("\n▶️ Start requested via keyboard!")
                events["start_requested"] = True
            else:
                # Ignore other keys during recording
                pass
                
        except Exception as e:
            print(f"⚠️ Keyboard monitor error: {e}")
            break
    
    print("🎹 Keyboard monitor stopped")

# ==================== Main ====================
def main():
    # Use config for all settings
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
        batch_encoding_size=recording_cfg.batch_encoding_size,
    )

    # ---- UI helpers ----
    events = create_events_dict()
    init_rerun(session_name="i2rt_record")
    
    # ---- Start keyboard monitor thread ----
    keyboard_thread = threading.Thread(target=keyboard_monitor_thread, args=(events,), daemon=True)
    keyboard_thread.start()
    
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
    
    # Get and print button state
    print("Leader button state:")
    button_states = teleop.get_button_states()
    for button_name, button_state in button_states.items():
        print(f"  {button_name}: {button_state[0]}")
    
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
    print(f"Recording {recording_cfg.num_episodes} episodes of {recording_cfg.episode_time_sec}s each")
    print("🎹 Keyboard commands: [Q]uit anytime, [R]exit episode early, [S]tart next episode, [H]elp")
    
    recorded_episodes = 0
    current_episode = 0
    waiting_for_start = True
    
    try:
        while recorded_episodes < recording_cfg.num_episodes and not events["stop_recording"]:
            if waiting_for_start:
                print(f"\nReady to record episode {current_episode + 1}/{recording_cfg.num_episodes}")
                print("Press [S] to start, [Q] to quit, [H] for help:")
                # Wait for start command from keyboard thread
                while waiting_for_start and not events["stop_recording"]:
                    time.sleep(0.1)  # Small delay to prevent busy waiting
                    # Check if start was requested
                    if events.get("start_requested", False):
                        events["start_requested"] = False
                        events["rerecord_episode"] = False
                        waiting_for_start = False
                        break
            
            if events["stop_recording"]:
                break
                
            log_say(f"Recording episode {current_episode + 1}")

            # Main record loop
            print(f"Starting record loop for episode {current_episode + 1}...")
            print("💡 Press 'R' during recording to exit this episode early")
            
            # Reset exit_early flag before recording
            events["exit_early"] = False
            
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
            
            # Check if episode was exited early
            if events.get("exit_early", False):
                print("⏹️ Episode recording ended early!")
                events["exit_early"] = False
            
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
            
            dataset.save_episode()
            print(f"✓ Episode {current_episode+1} saved successfully")
            recorded_episodes += 1
            current_episode += 1
            
            # Reset for next episode
            waiting_for_start = True
            
            # Command handling after episode
            if recorded_episodes < recording_cfg.num_episodes and not events["stop_recording"]:
                print(f"\nEpisode {recorded_episodes} completed. Next: episode {recorded_episodes + 1}/{recording_cfg.num_episodes}")
                print("Press [S] for next episode, [Q] to quit, [H] for help:")
                # Wait for next command from keyboard thread
                while waiting_for_start and not events["stop_recording"]:
                    time.sleep(0.1)  # Small delay to prevent busy waiting
                    # Check if start was requested
                    if events.get("start_requested", False):
                        events["start_requested"] = False
                        waiting_for_start = False
                        break

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

if __name__ == "__main__":
    main()
