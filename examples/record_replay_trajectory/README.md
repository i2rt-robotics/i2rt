# Record and Replay Trajectory Example

Record and replay robot arm movements using the I2RT Python API.

[`record_replay_trajectory.py`](record_replay_trajectory.py) drives every arm i2rt ships. It takes
its trajectory width from `robot.num_dofs()` and its joint names from the motor chain, so it does
not know or care which arm it is driving: a 6-DOF arm plus gripper records seven wide, a 7-DOF arm
plus gripper records eight.

## Quick Start

```bash
python record_replay_trajectory.py                              # yam + linear_4310 on can0
python record_replay_trajectory.py --arm yam_ultra_2            # a different arm variant
python record_replay_trajectory.py --arm big_yam --gripper linear_4310
python record_replay_trajectory.py --arm yam_7dof_left          # the 7-DOF arm's left model
```

The arm and gripper must match the hardware on the bus: `--arm` selects the motor IDs, gains and
directions the CAN chain is built with as well as the MuJoCo model gravity compensation runs on, so
a mismatch silently mis-scales torque. On startup the motor chain verifies every motor's type and
control mode against the arm's config and refuses to run on a mismatch, before anything moves.

`yam_7dof_left` and `yam_7dof_right` are the same arm mirrored, and they share a joint-coordinate
convention, so a recording **does** replay on the other one, as the mirror image of the motion that
was recorded. That is useful for symmetric bimanual work and wrong if you wanted the same path in
world space.

This example is **real-hardware only**: `move_joints` and the MCAP recording methods live on
`MotorChainRobot`, not on the `Robot` protocol, so there is no `--sim` flag.

With `gripper_limits: null` in the arm's config the gripper calibrates its travel at construction
time, so **it strokes once before the interface appears**, with the arm held on its configured PD
gains. Pass `gripper_limits_override` to `get_yam_robot` to skip it.

## What It Does

- **Record**: Move the robot arm by hand while every control-loop frame is written to an MCAP file
- **Replay**: Robot automatically reproduces your recorded motion, at the speed it was recorded
- **Load**: Replay a previously recorded MCAP file

Recording uses the same `RobotMcapRecorder` as `motor_chain_robot.py --record`, so the files this
example produces are interchangeable with every other recording in the repo.

## Controls

- `r` - Start/stop recording
- `p` - Play back recorded motion
- `l` - Load trajectory from file
- `q` - Quit

## What You'll See

```
Controls:
  r : Start/stop recording
  p : Start replay
  l : Load trajectory from file
  q : Quit

Status:
Recording: False  Replaying: False
Trajectory length: 0 samples

Press 'q' to quit.
```

## Workflow

1. Run the script
2. **Option A**: Press `r` to start recording, move arm, press `r` to stop
3. **Option B**: Press `l` to load a previously recorded MCAP file
4. Press `p` to replay the motion
5. Press `q` to quit

Stopping a recording loads it straight back in, so you can record and then replay without
touching the filesystem.

## Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--channel` | `can0` | CAN interface name |
| `--arm` | `yam` | Arm variant: `yam`, `yam_pro`, `yam_ultra`, `yam_ultra_2`, `big_yam`, `yam_7dof_left`, `yam_7dof_right`, `no_arm` |
| `--gripper` | `linear_4310` | Gripper type, so gravity compensation loads the correct model |
| `--load` | none | MCAP recording to load for replay at startup |

There is no `--output`: recordings are written to a timestamped path chosen by the recorder,
`~/.i2rt/YYYYMMDD/YYYYMMDD-HHmmss/robot.mcap`, which is printed on screen when recording starts.

## Recorded Schema

Each recording is one ROS 2 CDR MCAP file, written asynchronously from the control thread at the
full control-loop rate (a few hundred Hz) — not at the UI loop rate. Per motor feedback frame it
holds one message on each of these topics:

| Topic | Message type | Contents |
|-------|--------------|----------|
| `/joint_states` | `sensor_msgs/msg/JointState` | `name`, `position` (rad), `velocity` (rad/s), `effort` (Nm) |
| `/required_torques` | `sensor_msgs/msg/JointState` | `effort` = the torque the control loop asked for (Nm); `position`/`velocity` empty |
| `/<joint>/temperature/mos` | `sensor_msgs/msg/Temperature` | Motor MOS temperature (°C), one topic per joint |
| `/<joint>/temperature/rotor` | `sensor_msgs/msg/Temperature` | Motor rotor temperature (°C), one topic per joint |

`name` is one entry per motor in the CAN chain, with the gripper as the last entry. For the default
YAM arm plus gripper that is `joint1 … joint6, gripper`, while `--gripper no_gripper` records
`joint1 … joint6` and `--arm no_arm` records `gripper` alone; either `yam_7dof_*` arm records
`joint1 … joint7, gripper` (8 entries). Message `log_time`, and the header stamp, are the motor
feedback timestamp in nanoseconds.

Replay reads only `/joint_states` back, via `read_joint_positions` in
[`i2rt/utils/recording.py`](../../i2rt/utils/recording.py), and follows the recorded timestamps so
playback runs at the original speed regardless of the rate the file was recorded at:

```python
from pathlib import Path

from i2rt.utils.recording import read_joint_positions

recording = read_joint_positions(Path("~/.i2rt/20260907/20260907-103820/robot.mcap").expanduser())
print(recording.names)          # ('joint1', ..., 'joint6', 'gripper')
print(len(recording.positions)) # samples
```

Those `positions` rows are exactly what `command_joint_pos` accepts, gripper column included — do
not substitute `get_observations()["joint_pos"]`, which splits the gripper out into its own key and
so is one element short.

The files are ordinary MCAP, so they also open directly in Foxglove or `mcap info`.
