# Record & Replay Trajectory

**Location:** `examples/record_replay_trajectory/`

Record a manipulation trajectory through teleoperation (or gravity-comp hand-guiding) and replay it exactly — useful for dataset collection and validating robot configurations.

## Hardware Required

- 1× YAM arm (follower or standalone)
- 1× CANable USB-CAN adapter
- Optional: leader arm + teaching handle for teleoperation recording

## Overview

```
Record phase:
  Arm in gravity-comp mode  ──►  Guide by hand  ──►  Save joint trajectory

Replay phase:
  Load trajectory  ──►  Command arm via PD control  ──►  Arm replays motion
```

## Video

<MediaPlaceholder
  type="video"
  description="Split screen: left side shows recording phase (operator guiding arm by hand through a pouring task), right side shows replay phase (arm autonomously repeating the motion). 60–90 seconds."
/>

## Running

```bash
python examples/record_replay_trajectory/record_replay_trajectory.py --channel can0 --gripper linear_4310
```

### Controls

| Key | Action |
|-----|--------|
| `r` | Start / stop recording |
| `p` | Play back recorded motion |
| `s` | Save trajectory to file |
| `l` | Load trajectory from file |
| `q` | Quit |

### Options

```bash
--channel can0          # CAN bus channel
--gripper linear_4310   # Gripper type (for gravity compensation)
--output file.npy       # Output filename
--load file.npy         # Load and replay a trajectory at startup
```

### Workflow

1. Run the script — arm enters gravity-comp mode
2. Press `r` to start recording, guide the arm by hand, press `r` to stop
3. Press `p` to replay the captured motion
4. Press `s` to save, or `l` to load a previously saved trajectory

## Output Format

Trajectories are saved as NumPy arrays:

```python
import numpy as np

# Shape: (T, 6) — T timesteps, 6 joints
trajectory = np.load("trajectory.npy")
print(trajectory.shape)  # (500, 6)
```

## See Also

- [YAM Arm API](/sdk/yam-arm)
- [Bimanual Teleoperation](/examples/bimanual-teleoperation)
