# YAM Demo — Zero-Gravity & Joint Control

Run your first YAM arm demo in 5 minutes.

::: tip Prerequisite
[YAM Hardware Setup](/getting-started/hardware/yam) done — arm is wired, powered, and CAN is up.
:::

## 1. Test without hardware (sim mode)

You don't even need an arm. Launch the MuJoCo viewer:

```bash
python examples/minimum_gello/minimum_gello.py --mode visualizer_local
```

A 3D window opens showing the YAM model. Use this to verify your install before plugging in hardware.

## 2. Zero-gravity mode (with hardware)

The arm enters a **gravity-compensated floating state** — push it freely, it stays where you leave it.

```bash
python i2rt/robots/get_robot.py --channel can0 --gripper linear_4310
```

Press `Ctrl+C` to exit.

## 3. Python API — read state and move

```python
from i2rt.robots.get_robot import get_yam_robot
import numpy as np

# Connect (zero-gravity ON by default)
robot = get_yam_robot(channel="can0", zero_gravity_mode=True)

# Read current observations
obs = robot.get_observations()
print("Arm joints:", obs["joint_pos"])   # (6,) radians
print("Gripper:",    obs["gripper_pos"])  # (1,) normalized

# Command home position (6 arm joints + 1 gripper)
robot.command_joint_pos(np.zeros(7))

robot.close()
```

## 4. Different arm variants

```python
from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import ArmType

robot = get_yam_robot(arm_type=ArmType.BIG_YAM, channel="can0")
```

| Arm type | Constant |
|---|---|
| Standard | `ArmType.YAM` |
| Pro | `ArmType.YAM_PRO` |
| Ultra | `ArmType.YAM_ULTRA` |
| Big | `ArmType.BIG_YAM` |
| Gripper-only | `ArmType.NO_ARM` |

## Next steps

- Full API reference, gravity comp tuning, MuJoCo / Viser controls, record & replay → [YAM product page](/products/yam)
- Bimanual teleoperation → [YAM Cell demo](/getting-started/demos/yam-cell)
