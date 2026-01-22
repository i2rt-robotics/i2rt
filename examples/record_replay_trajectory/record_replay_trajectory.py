import argparse
import curses
import os
import time
from typing import Any

import numpy as np

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.motor_chain_robot import GripperType


def main(stdscr: Any) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", type=str, default="can0")
    parser.add_argument("--gripper", type=str, default="linear_4310")
    parser.add_argument("--output", type=str, default="./example_trajectory.npy")
    parser.add_argument("--load", type=str, help="Load trajectory from file")
    args, _ = parser.parse_known_args()

    # Initialize robot
    gripper_type = GripperType.from_string_name(args.gripper)
    robot = get_yam_robot(channel=args.channel, gripper_type=gripper_type)

    # Curses setup
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(0)

    trajectory = []
    timestamps = []
    recording = False
    replaying = False
    replay_idx = 0
    target_freq = 60.0  # 30 Hz
    dt = 1.0 / target_freq

    # Load trajectory if specified
    if args.load and os.path.exists(args.load):
        try:
            data = np.load(args.load, allow_pickle=True).item()
            trajectory = data["trajectory"].tolist()
            timestamps = data["timestamps"].tolist()
            if "frequency" in data:
                target_freq = data["frequency"]
                dt = 1.0 / target_freq
        except Exception as e:
            print(f"Error loading trajectory: {e}")
            trajectory = []
            timestamps = []

    instructions = [
        "Controls:",
        "  r : Start/stop recording",
        "  p : Start replay",
        "  s : Save trajectory",
        "  l : Load trajectory from file",
        "  q : Quit",
        "",
        "Status:",
    ]

    last_record_time = time.monotonic()
    last_replay_time = time.monotonic()

    while True:
        current_time = time.monotonic()
        key = stdscr.getch()

        if key != -1:
            if key == ord("q"):
                break
            elif key == ord("r"):
                recording = not recording
                replaying = False
                if recording:
                    trajectory = []
                    timestamps = []
                    last_record_time = current_time
                stdscr.addstr(len(instructions) + 2, 0, f"Recording: {recording}         ")
            elif key == ord("p"):
                if len(trajectory) > 0:
                    replaying = True
                    recording = False
                    replay_idx = 0
                    last_replay_time = current_time
                else:
                    stdscr.addstr(len(instructions) + 2, 0, "No trajectory to replay.      ")
            elif key == ord("s"):
                if len(trajectory) > 0:
                    # Save both trajectory and timestamps
                    data = {
                        "trajectory": np.array(trajectory),
                        "timestamps": np.array(timestamps),
                        "frequency": target_freq,
                    }
                    np.save(args.output, data)
                    stdscr.addstr(len(instructions) + 2, 0, f"Saved to {args.output}        ")
                else:
                    stdscr.addstr(len(instructions) + 2, 0, "No trajectory to save.        ")
            elif key == ord("l"):
                # Load trajectory from file
                stdscr.addstr(len(instructions) + 2, 0, "Enter filename to load: ")
                stdscr.refresh()

                # Simple filename input (basic implementation)
                filename = ""
                while True:
                    key = stdscr.getch()
                    if key == ord("\n"):  # Enter key
                        break
                    elif key == ord("\x1b"):  # Escape key
                        filename = ""
                        break
                    elif key == ord("\x7f"):  # Backspace
                        if filename:
                            filename = filename[:-1]
                    elif 32 <= key <= 126:  # Printable characters
                        filename += chr(key)

                    stdscr.addstr(len(instructions) + 2, 0, f"Enter filename to load: {filename}")
                    stdscr.refresh()

                if filename and os.path.exists(filename):
                    try:
                        data = np.load(filename, allow_pickle=True).item()
                        trajectory = data["trajectory"].tolist()
                        timestamps = data["timestamps"].tolist()
                        if "frequency" in data:
                            target_freq = data["frequency"]
                            dt = 1.0 / target_freq
                        stdscr.addstr(len(instructions) + 2, 0, f"Loaded {filename} successfully    ")
                    except Exception as e:
                        stdscr.addstr(len(instructions) + 2, 0, f"Error loading {filename}      ")
                elif filename:
                    stdscr.addstr(len(instructions) + 2, 0, f"File {filename} not found      ")

        # UI
        stdscr.erase()
        for i, line in enumerate(instructions):
            stdscr.addstr(i, 0, line)
        stdscr.addstr(len(instructions), 0, f"Recording: {recording}  Replaying: {replaying}")
        stdscr.addstr(len(instructions) + 1, 0, f"Trajectory length: {len(trajectory)} samples")
        stdscr.addstr(len(instructions) + 3, 0, "Press 'q' to quit.")

        # Record trajectory at 30Hz
        if recording and (current_time - last_record_time) >= dt:
            qpos = robot.get_joint_pos()
            trajectory.append(np.copy(qpos))
            timestamps.append(current_time)
            last_record_time = current_time

        # Replay trajectory at 30Hz
        if replaying and len(trajectory) > 0:
            if replay_idx == 0:
                # slowly move the arm to fist way point
                robot.move_joints(np.array(trajectory[replay_idx]), time_interval_s=1.5)
            if replay_idx < len(trajectory) and (current_time - last_replay_time) >= dt:
                robot.command_joint_pos(trajectory[replay_idx])
                replay_idx += 1
                last_replay_time = current_time
                stdscr.addstr(len(instructions) + 4, 0, f"Replaying: {replay_idx}/{len(trajectory)}")
            elif replay_idx >= len(trajectory):
                replaying = False
                stdscr.addstr(len(instructions) + 4, 0, "Replay finished.              ")

        stdscr.refresh()
        time.sleep(0.02)  # Higher refresh rate for smoother operation


if __name__ == "__main__":
    curses.wrapper(main)
