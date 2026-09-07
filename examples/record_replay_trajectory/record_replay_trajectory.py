"""Record a hand-guided arm motion to an MCAP file and replay it.

Recording goes through the same ``RobotMcapRecorder`` that ``motor_chain_robot.py --record``
uses, so a trajectory recorded here is one more ROS 2 CDR MCAP file: it is written from the
control thread at the full control-loop rate, and it lands under ``~/.i2rt/``.

Usage:
    python examples/record_replay_trajectory/record_replay_trajectory.py --channel can0
    python examples/record_replay_trajectory/record_replay_trajectory.py --arm yam_ultra_2
    python examples/record_replay_trajectory/record_replay_trajectory.py --arm big_yam --gripper linear_4310
"""

import curses
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import tyro

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import ArmType, GripperType
from i2rt.utils.recording import read_joint_positions


@dataclass
class _CliArgs:
    """Record and replay a YAM arm trajectory."""

    channel: str = "can0"
    """CAN channel."""
    arm: str = "yam"
    """Arm variant (yam, yam_pro, yam_ultra, yam_ultra_2, big_yam, no_arm)."""
    gripper: str = "linear_4310"
    """Yam gripper type so the gravity compensation can load the correct model."""
    load: Optional[Path] = None
    """MCAP recording to load for replay at startup."""


def _load_trajectory(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read one recording's timestamps (s) and joint positions (rad) for replay."""
    recording = read_joint_positions(path)
    return np.array(recording.timestamps), np.array(recording.positions)


def main(stdscr: Any, args: _CliArgs) -> None:
    arm_type = ArmType.from_string_name(args.arm)
    gripper_type = GripperType.from_string_name(args.gripper)
    robot = get_yam_robot(channel=args.channel, arm_type=arm_type, gripper_type=gripper_type)

    # Curses setup
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(0)

    timestamps = np.empty(0)
    trajectory = np.empty((0, robot.num_dofs()))
    status = ""
    recording_path: Optional[Path] = None
    replaying = False
    replay_idx = 0
    replay_start = 0.0

    if args.load is not None:
        try:
            timestamps, trajectory = _load_trajectory(args.load)
            status = f"Loaded {len(trajectory)} samples from {args.load}"
        except (OSError, ValueError) as e:
            status = f"Error loading trajectory: {e}"

    instructions = [
        "Controls:",
        "  r : Start/stop recording",
        "  p : Start replay",
        "  l : Load trajectory from file",
        "  q : Quit",
        "",
        "Status:",
    ]

    try:
        while True:
            current_time = time.monotonic()
            key = stdscr.getch()

            if key != -1:
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    replaying = False
                    if recording_path is None:
                        recording_path = robot.start_mcap_recording()
                        status = f"Recording to {recording_path}"
                    else:
                        robot.stop_mcap_recording()
                        timestamps, trajectory = _load_trajectory(recording_path)
                        status = f"Recorded {len(trajectory)} samples to {recording_path}"
                        recording_path = None
                elif key == ord("p"):
                    if len(trajectory) > 0:
                        # slowly move the arm to first way point
                        robot.move_joints(trajectory[0], time_interval_s=1.5)
                        replaying = True
                        replay_idx = 0
                        replay_start = time.monotonic()
                    else:
                        status = "No trajectory to replay."
                elif key == ord("l"):
                    # Simple filename input (basic implementation)
                    stdscr.addstr(len(instructions) + 2, 0, "Enter filename to load: ")
                    stdscr.refresh()

                    filename = ""
                    while True:
                        key = stdscr.getch()
                        if key == ord("\n"):  # Enter key
                            break
                        elif key == ord("\x1b"):  # Escape key
                            filename = ""
                            break
                        elif key == ord("\x7f"):  # Backspace
                            filename = filename[:-1]
                        elif 32 <= key <= 126:  # Printable characters
                            filename += chr(key)

                        stdscr.addstr(len(instructions) + 2, 0, f"Enter filename to load: {filename}")
                        stdscr.refresh()

                    if filename:
                        try:
                            timestamps, trajectory = _load_trajectory(Path(filename))
                            status = f"Loaded {len(trajectory)} samples from {filename}"
                        except (OSError, ValueError) as e:
                            status = f"Error loading {filename}: {e}"

            # Replay at the rate the trajectory was recorded at, by following its own timestamps.
            # The recorder runs at the control-loop rate, well above this loop, so each iteration
            # skips ahead to the sample the elapsed replay time has reached.
            if replaying:
                elapsed = current_time - replay_start
                while replay_idx + 1 < len(trajectory) and timestamps[replay_idx + 1] - timestamps[0] <= elapsed:
                    replay_idx += 1
                robot.command_joint_pos(trajectory[replay_idx])
                if replay_idx + 1 == len(trajectory):
                    replaying = False
                    status = "Replay finished."

            # UI
            stdscr.erase()
            for i, line in enumerate(instructions):
                stdscr.addstr(i, 0, line)
            stdscr.addstr(len(instructions), 0, f"Recording: {recording_path is not None}  Replaying: {replaying}")
            stdscr.addstr(len(instructions) + 1, 0, f"Trajectory length: {len(trajectory)} samples")
            stdscr.addstr(len(instructions) + 2, 0, status)
            if replaying:
                stdscr.addstr(len(instructions) + 3, 0, f"Replaying: {replay_idx + 1}/{len(trajectory)}")
            stdscr.addstr(len(instructions) + 4, 0, "Press 'q' to quit.")
            stdscr.refresh()

            time.sleep(0.02)  # Higher refresh rate for smoother operation
    finally:
        # close() finalizes any in-progress MCAP recording.
        robot.close()


if __name__ == "__main__":
    curses.wrapper(main, tyro.cli(_CliArgs))
