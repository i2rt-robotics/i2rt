# Viser Control Interface

**Location:** `examples/control_with_viser/`

Browser-based 3-D viewer and teleop interface for i2rt robots, powered by [viser](https://github.com/nerfstudio-project/viser). Open the viewer in any browser — no native viewer required, no display attached to the robot.

## Hardware Required

- None for simulation (`--sim` flag)
- 1× YAM / YAM Pro / YAM Ultra / big_yam arm + CANable adapter for real hardware

## Overview

The interface starts in **read-only mode**. The robot accepts no commands until the operator visually confirms alignment and clicks **Enable Robot**. Once enabled, three control modes are available:

```
VIS mode (default):
  Robot joint states  ──►  Browser viewer  (read-only mirror)

IK mode:
  Drag transform gizmo  ──►  mink IK solver  ──►  Command arm

Joint-slider mode:
  Per-joint sliders (deg)  ──►  Command arm directly
```

A gripper slider (normalised 0–1) is always available, independent of the arm mode.

## Running

### Simulation

```bash
# YAM arm + linear_4310 gripper (default)
python examples/control_with_viser/control_with_viser.py --sim

# big_yam arm with the soft-tip gripper
python examples/control_with_viser/control_with_viser.py --arm big_yam --gripper flexible_4310 --sim

# Custom port (multiple instances on one machine)
python examples/control_with_viser/control_with_viser.py --sim --port 8090
```

### Real Hardware

```bash
python examples/control_with_viser/control_with_viser.py --channel can0
python examples/control_with_viser/control_with_viser.py --arm big_yam --gripper linear_4310 --channel can0
```

Open `http://localhost:8080` (or whichever `--port` you chose) in your browser.

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--arm` | `yam` | Arm type: `yam`, `yam_pro`, `yam_ultra`, `big_yam`, `no_arm` |
| `--gripper` | `linear_4310` | Gripper: `linear_4310`, `linear_3507`, `crank_4310`, `flexible_4310`, `yam_teaching_handle`, `no_gripper` |
| `--channel` | `can0` | CAN interface name (real hardware only) |
| `--sim` | off | Use simulation instead of real hardware |
| `--dt` | `0.02` | Control loop timestep in seconds |
| `--port` | `8080` | Viser server port |
| `--site` | auto | EE site name. Auto-detects `grasp_site` (or `tcp_site` for `yam_teaching_handle`) when omitted. |

## Enabling the Robot

1. Open the viewer URL printed on launch.
2. Verify the rendered pose matches the physical robot.
3. Tick **Alignment Confirmed**.
4. Click **Enable Robot**.

The robot only starts responding to slider / IK input after step 4.

## Teaching-Handle Indicators

When running with `--gripper yam_teaching_handle`, two on-screen LEDs mirror the leader-arm buttons:

- **SYNC** — top button, latches arm-sync on/off.
- **RECORD** — second button, user-programmable (commonly bound to recording).

## MuJoCo vs Viser

| Feature | MuJoCo | Viser |
|---------|--------|-------|
| Display | Native window | Browser (any client) |
| Multi-user | Local only | Multiple browsers can connect to one server |
| IK target | Drag in 3-D scene | Drag transform gizmo |
| Per-joint sliders | Yes (CONTROL mode) | Yes (Joint-slider mode) |
| Safety gate | None | Read-only until "Enable Robot" |
| Best for | Local debugging | Remote / headless robot, browser-only setups |

## See Also

- [MuJoCo Control Interface](/examples/control-with-mujoco)
- [Minimum Gello](/examples/minimum-gello)
- [YAM Arm API](/sdk/yam-arm)
