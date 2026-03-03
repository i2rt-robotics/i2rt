# YAM Arm Series

<div class="product-badges">
  <span class="product-badge available">✓ Python SDK</span>
  <span class="product-badge available">✓ MuJoCo Sim</span>
  <span class="product-badge available">✓ Gravity Compensation</span>
  <span class="product-badge available">✓ Teleoperation</span>
</div>

**YAM** (Your Arm Matters) is I2RT's flagship robotic arm — a 6-DOF, CAN bus–driven manipulator designed for real-world research and embodied AI data collection. The YAM family spans four tiers to match different reach, payload, and budget requirements.

<MediaPlaceholder
  type="photo"
  description="Hero photo of the full YAM arm family — YAM, YAM Pro, YAM Ultra, and BIG YAM — arranged side by side on a clean white surface."
/>

## Model Overview

| Model | Price | Notes |
|-------|-------|-------|
| **YAM** | $2,999 | Standard research arm |
| **YAM Pro** | $3,499 | Enhanced actuators |
| **YAM Ultra** | $4,299 | Highest spec standard arm |
| **BIG YAM** | $4,999 | Larger reach and payload |
| **YAM Leader** | $2,999 | Teaching handle for teleoperation |

## Specifications

| Parameter | Value |
|-----------|-------|
| Degrees of Freedom | 6 |
| Communication | CAN bus (1 Mbit/s) |
| Motor Series | DM series brushless |
| Control Modes | Joint position PD · Gravity compensation · Zero-gravity |
| Simulation | MuJoCo (MJCF + URDF provided) |
| Safety | 400 ms motor timeout (configurable) |
| Mounting | Table-top (standard) |

## Grippers

YAM supports four interchangeable end effectors:

<MediaPlaceholder
  type="photo"
  description="The three YAM gripper variants shown side by side: crank_4310, linear_3507, and yam_teaching_handle. Macro photo on a dark surface."
/>

| Gripper | Description |
|---------|-------------|
| `crank_4310` | Zero-linkage crank design — minimizes total gripper width for tight workspaces. |
| `linear_3507` | Lightweight linear gripper with DM3507 motor. Requires starting in the closed position for calibration. |
| `linear_4310` | Linear gripper with the heavier DM4310 motor. Marginally more gripping force. |
| `yam_teaching_handle` | Leader arm handle with a trigger for gripper and two programmable buttons. Used for teleoperation. |

## 3D Model

The YAM URDF and MuJoCo XML are included in the repository:

```
i2rt/robot_models/arm/yam/
├── yam.urdf
├── yam.xml
└── assets/          # STL meshes (visual + collision)
```

## Videos

<MediaPlaceholder
  type="video"
  description="YAM arm performing a pick-and-place task on a cluttered tabletop. Close-up of gripper engagement. 30–60 seconds."
/>

<MediaPlaceholder
  type="video"
  description="YAM in zero-gravity mode — operator guides the arm by hand through a full range of motion. Emphasizes backdrivability and smooth motion."
/>

## Getting Started

1. [Install the SDK](/getting-started/installation)
2. [Set up CAN bus](/getting-started/hardware-setup)
3. See [YAM Arm API](/sdk/yam-arm) for the full Python reference
4. Try the [Quick Start](/getting-started/quick-start)

```python
from i2rt.robots.motor_chain_robot import get_yam_robot
import numpy as np

# Connect to the arm (zero-gravity mode on by default)
robot = get_yam_robot(channel="can0", zero_gravity_mode=True)

# Read current joint positions
joints = robot.get_joint_pos()  # shape: (6,) radians

# Command a target configuration
robot.command_joint_pos(np.zeros(6))
```

## Where to Buy

Visit [i2rt.com](https://i2rt.com) or contact [sales@i2rt.com](mailto:sales@i2rt.com).

<style scoped>
.product-badges { display: flex; flex-wrap: wrap; gap: 8px; margin: 16px 0 24px; }
.product-badge { display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; border: 1px solid; }
.product-badge.available { color: #4CCFB0; border-color: rgba(76,207,176,0.4); background: rgba(76,207,176,0.08); }
</style>
