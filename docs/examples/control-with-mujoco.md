# MuJoCo Control Interface

**Location:** `examples/control_with_mujoco/`

Interactive MuJoCo viewer for i2rt robots. Visualises the robot in real time and lets you move it by dragging a target marker via inverse kinematics — no hardware required in simulation mode.

## Hardware Required

- None for simulation (`--sim` flag)
- 1× YAM / YAM Pro / YAM Ultra / big_yam arm + CANable adapter for real hardware

## Overview

The interface has two modes toggled with **SPACE**:

```
VIS mode (default):
  Robot joint states  ──►  MuJoCo viewer  (gravity comp active on real hw)

CONTROL mode (press SPACE):
  Drag target marker  ──►  IK solver  ──►  Command arm
```

## Running

### Simulation

```bash
# YAM arm + linear_4310 gripper (default)
python examples/control_with_mujoco/control_with_mujoco.py --sim

# big_yam arm
python examples/control_with_mujoco/control_with_mujoco.py --arm big_yam --gripper linear_4310 --sim

# Arm only, no gripper
python examples/control_with_mujoco/control_with_mujoco.py --arm yam --gripper no_gripper --sim
```

### Real Hardware

```bash
python examples/control_with_mujoco/control_with_mujoco.py --channel can0
python examples/control_with_mujoco/control_with_mujoco.py --arm big_yam --gripper linear_4310 --channel can0
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--arm` | `yam` | Arm type: `yam`, `yam_pro`, `yam_ultra`, `big_yam` |
| `--gripper` | `linear_4310` | Gripper: `linear_4310`, `linear_3507`, `crank_4310`, `yam_teaching_handle`, `no_gripper` |
| `--channel` | `can0` | CAN interface name (real hardware only) |
| `--sim` | off | Use simulation instead of real hardware |
| `--dt` | `0.02` | Control loop timestep in seconds |
| `--site` | `grasp_site` | MuJoCo site name used as end-effector |

## Viewer Controls

1. Press **SPACE** to enter CONTROL mode (marker turns red)
2. **Double-click** the red target sphere to select it
3. **Ctrl + right-drag** — translate the target
4. **Ctrl + left-drag** — rotate the target
5. Press **SPACE** again to return to VIS mode

## Supported Arms and Grippers

All YAM-family arm × gripper combinations are supported:

| Arm | Grippers |
|-----|---------|
| `yam` | `linear_4310`, `linear_3507`, `crank_4310`, `yam_teaching_handle`, `no_gripper` |
| `yam_pro` | same as yam |
| `yam_ultra` | same as yam |
| `big_yam` | same as yam |
