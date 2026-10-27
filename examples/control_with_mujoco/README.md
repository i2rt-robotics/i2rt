# MuJoCo Control Interface

Interactive MuJoCo viewer for i2rt robots. Starts in gravity-comp visualisation mode
and can toggle into IK-based mocap control.

## Modes

| Mode | Description |
|------|-------------|
| **VIS** (default) | Mirrors the robot's live joint positions. Gravity-comp is active on real hardware. |
| **CONTROL** | Moves the robot by dragging a target marker in the viewer via inverse kinematics. |

Press **SPACE** in the viewer to toggle between modes.

## Usage

> **macOS:** launch with `mjpython` instead of `python` (`uv run mjpython ...`).
> MuJoCo's passive viewer only opens from the process `mjpython` started, so on
> macOS the viewer runs in the main process and only the robot gets a subprocess.

### Simulation (no hardware required)

```bash
# YAM arm with linear_4310 gripper (default)
python examples/control_with_mujoco/control_with_mujoco.py --sim

# big_yam arm with linear_4310 gripper
python examples/control_with_mujoco/control_with_mujoco.py --arm big_yam --gripper linear_4310 --sim

# Arm-only (no gripper)
python examples/control_with_mujoco/control_with_mujoco.py --arm yam --gripper no_gripper --sim

# 7-DOF arm (7 arm joints + a gripper), left side
python examples/control_with_mujoco/control_with_mujoco.py --arm yam_7dof_left --sim
```

### Real hardware

```bash
python examples/control_with_mujoco/control_with_mujoco.py --channel can0
python examples/control_with_mujoco/control_with_mujoco.py --arm big_yam --gripper linear_4310 --channel can0
# Through an EtherCAT-to-CAN board on NIC enp99s0
python examples/control_with_mujoco/control_with_mujoco.py --arm yam_7dof_right --channel ecat:enp99s0
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--arm` | `yam` | Arm type: `yam`, `yam_pro`, `yam_ultra`, `yam_ultra_2`, `big_yam`, `yam_7dof_left`, `yam_7dof_right`, `no_arm` (end effector only; cannot combine with `--gripper no_gripper`, and either teaching handle is passive there — read-only, no `--record`) |
| `--gripper` | `linear_4310` | Gripper type: `linear_4310`, `linear_3507`, `crank_4310`, `flexible_4310`, `yam_teaching_handle_v1`, `yam_teaching_handle_v2`, `no_gripper` |
| `--channel` | `can0` | CAN interface name (real hardware only). `ecat:<nic>` (e.g. `ecat:enp99s0`) drives an arm through an EtherCAT-to-CAN board instead; its motors must answer on `MASTER_ID == ID` — see [arm7_osc](../arm7_osc/README.md) |
| `--sim` | off | Use simulation instead of real hardware |
| `--dt` | `0.02` | Control loop timestep in seconds |
| `--site` | auto | MuJoCo site used as end-effector (auto-detected from gripper: `tcp_site` for either teaching handle, else `grasp_site`) |
| `--friction` | off | Enable Coulomb friction compensation in gravity comp (real hardware only) |
| `--record` | off | Record real-hardware motor feedback and computed required torques to a timestamped ROS 2 CDR MCAP file |

## Viewer Controls (CONTROL mode)

1. Press **SPACE** to enter CONTROL mode (marker turns red)
2. **Double-click** the target sphere to select it
3. **Ctrl + right-drag** — translate the target
4. **Ctrl + left-drag** — rotate the target
5. Press **SPACE** again to return to VIS mode
