# Record and Replay Trajectory Example

Record and replay robot arm movements using the I2RT Python API.

## Quick Start

```bash
python record_replay_trajectory.py                              # yam + linear_4310 on can0
python record_replay_trajectory.py --arm yam_ultra_2            # a different arm variant
python record_replay_trajectory.py --arm big_yam --gripper linear_4310
```

The arm and gripper must match the hardware on the bus: `--arm` selects the motor IDs, gains and
directions the CAN chain is built with as well as the MuJoCo model gravity compensation runs on, so
a mismatch silently mis-scales torque.

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
| `--arm` | `yam` | Arm variant: `yam`, `yam_pro`, `yam_ultra`, `yam_ultra_2`, `big_yam`, `no_arm` |
| `--gripper` | `linear_4310` | Yam gripper type so the gravity compensation can load the correct model |
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

`name` is one entry per motor in the CAN chain, so it follows `--arm`/`--gripper`: for the default
arm plus gripper that is `joint1 … joint6, gripper`, with the gripper as the last entry, while
`--gripper no_gripper` records `joint1 … joint6` and `--arm no_arm` records `gripper` alone.
Message `log_time`, and the header stamp, are the motor feedback timestamp in nanoseconds.

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

The files are ordinary MCAP, so they also open directly in Foxglove or `mcap info`.
