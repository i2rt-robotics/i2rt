# Quick Start

Get the YAM arm moving in 5 minutes. This guide assumes you have [installed the SDK](/getting-started/installation) and [set up the CAN bus](/getting-started/hardware-setup).

## 1. Test Without Hardware (Simulator)

You don't need an arm to try the SDK. Launch the MuJoCo visualizer:

```bash
python examples/minimum_gello/minimum_gello.py --mode visualizer_local
```

A 3D window opens with the YAM arm. You can inspect the model and verify your environment is working.

## 2. Zero-Gravity Mode

Enable zero-gravity mode to move the arm freely by hand:

```bash
python i2rt/robots/motor_chain_robot.py --channel can0 --gripper_type linear_4310
```

The arm enters a gravity-compensated floating state. You can push it to any configuration; it holds position when released.

## 3. Python API: Move to a Joint Target

```python
from i2rt.robots.motor_chain_robot import get_yam_robot
import numpy as np

# Connect (zero-gravity on by default)
robot = get_yam_robot(channel="can0", zero_gravity_mode=True)

# Read current joint positions (6 values, in radians)
q = robot.get_joint_pos()
print("Current joints:", q)

# Command a home position
robot.command_joint_pos(np.zeros(6))
```

## 4. Leader–Follower Teleoperation

You need one **follower arm** (any gripper) and one **leader arm** (with `yam_teaching_handle` gripper), each on a separate CAN channel.

**Terminal 1 — follower:**

```bash
python examples/minimum_gello/minimum_gello.py \
  --gripper linear_4310 \
  --mode follower \
  --can-channel can0 \
  --bilateral_kp 0.2
```

**Terminal 2 — leader:**

```bash
python examples/minimum_gello/minimum_gello.py \
  --gripper yam_teaching_handle \
  --mode leader \
  --can-channel can1 \
  --bilateral_kp 0.2
```

Press the **top button** on the teaching handle to enable synchronization.

## 5. Flow Base

```python
from i2rt.flow_base.flow_base_controller import Vehicle
import time

vehicle = Vehicle()
vehicle.start_control()

# Drive forward 0.1 m/s for 2 seconds
start = time.time()
while time.time() - start < 2.0:
    vehicle.set_target_velocity((0.1, 0.0, 0.0), frame="local")
```

## What's Next?

| Goal | Resource |
|------|----------|
| Bimanual teleoperation | [YAM Cell](/products/yam-cell) |
| Record and replay motions | [Record & Replay example](/examples/record-replay) |
| Full Python API reference | [YAM Arm API](/sdk/yam-arm) |
| Mobile manipulation | [Linear Bot](/products/linear-bot) |
| Troubleshooting | [Troubleshooting](/support/troubleshooting) |
