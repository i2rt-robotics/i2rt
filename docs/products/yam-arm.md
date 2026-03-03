# YAM Arm

<div class="product-badges">
  <span class="product-badge available">✓ Python SDK</span>
  <span class="product-badge available">✓ MuJoCo Sim</span>
  <span class="product-badge available">✓ Gravity Compensation</span>
</div>

The **YAM Arm** is the follower/manipulation arm in the YAM family — a 6-DOF, CAN bus–driven manipulator built for real-world research and embodied AI data collection. Pair it with a [YAM Leader](/products/yam-leader) for teleoperation, or run it standalone with the Python SDK.

<MediaPlaceholder
  type="photo"
  description="YAM arm standalone on a white tabletop — front and 3/4 views showing all six joints, base plate, and default linear gripper."
/>

## Models

| Model | Price | Notes |
|-------|-------|-------|
| **YAM** | $2,999 | Standard research arm |
| **YAM Pro** | $3,499 | Enhanced actuators |
| **YAM Ultra** | $4,299 | Highest spec standard arm |
| **BIG YAM** | $4,999 | Larger reach and payload |

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

| Gripper | Motor | Notes |
|---------|-------|-------|
| `crank_4310` | DM4310 | Zero-linkage crank — minimal sweep width |
| `linear_3507` | DM3507 | Lightweight linear; requires closed-position calibration |
| `linear_4310` | DM4310 | Standard linear; slightly higher force |

See [Grippers](/sdk/grippers) for calibration and model details.

## 3D Model

```
i2rt/robot_models/arm/yam/
├── yam.urdf
├── yam.xml
└── assets/   # STL meshes (visual + collision)
```

## Videos

<MediaPlaceholder
  type="video"
  description="YAM arm performing a pick-and-place task on a cluttered tabletop. Close-up of gripper engagement. 30–60 seconds."
/>

<MediaPlaceholder
  type="video"
  description="YAM in zero-gravity mode — operator guides the arm by hand through a full range of motion. Demonstrates backdrivability."
/>

## Quick Start

```python
from i2rt.robots.motor_chain_robot import get_yam_robot
import numpy as np

robot = get_yam_robot(channel="can0", gripper_type="linear_4310")

# Read joint positions (radians)
q = robot.get_joint_pos()   # shape: (6,)

# Command home position
robot.command_joint_pos(np.zeros(6))
```

## See Also

- [YAM Leader](/products/yam-leader) — teaching handle for teleoperation
- [YAM Arm API](/sdk/yam-arm)
- [Grippers](/sdk/grippers)
- [Hardware Setup](/getting-started/hardware-setup)

<style scoped>
.product-badges { display: flex; flex-wrap: wrap; gap: 8px; margin: 16px 0 24px; }
.product-badge { display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; border: 1px solid; }
.product-badge.available { color: #4CCFB0; border-color: rgba(76,207,176,0.4); background: rgba(76,207,176,0.08); }
</style>
