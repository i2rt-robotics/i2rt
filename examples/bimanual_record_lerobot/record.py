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

# ==================== Main ====================
def main():
    recording_cfg = RecordingConfig()

    # ---- Build robot (followers) ----
    robot_cfg = I2RTFollowerConfig()
    robot = I2RTRobot(robot_cfg)

    # ---- Teleop (leader) ----
    teleop_cfg = i2rtLeaderConfig()
    teleop = PortalLeaderTeleop(teleop_cfg)

    # ---- Processors (default identity) ----
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # ---- Dataset features ----
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features    = hw_to_dataset_features(robot.observation_features, OBS_STR)
    dataset_features = {**action_features, **obs_features}

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

    # ---- Connect endpoints ----
    robot.connect()
    teleop.connect()

    # ---- UI helpers ----
    listener, events = init_keyboard_listener()
    init_rerun(session_name="i2rt_record")

    if not robot.is_connected or not teleop.is_connected:
        raise RuntimeError("Robot or teleop is not connected!")

    print("Starting record loop...")
    recorded_episodes = 0
    try:
        while recorded_episodes < recording_cfg.num_episodes and not events["stop_recording"]:
            log_say(f"Recording episode {recorded_episodes}")

            # Main record loop
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
            dataset.save_episode()
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
