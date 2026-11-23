# Minimum Gello (Teleoperation)

Minimal leader-follower teleoperation example. Supports different arm and gripper assemblies, simulation mode, and local/remote visualization.

## Quick Start

```bash
# Follower mode (default) — serves robot state over portal
python examples/minimum_gello/minimum_gello.py --can-channel can0

# Follower mode in simulation (no hardware required)
python examples/minimum_gello/minimum_gello.py --sim

# Leader mode — reads teaching handle and drives a remote follower
python examples/minimum_gello/minimum_gello.py --mode leader --can-channel can1

# Visualizer — MuJoCo viewer mirrors live robot state
python examples/minimum_gello/minimum_gello.py --mode visualizer_local

# Different arm and gripper combinations
python examples/minimum_gello/minimum_gello.py --arm big_yam --gripper linear_4310 --sim
python examples/minimum_gello/minimum_gello.py --arm yam_pro --gripper flexible_4310 --sim
```

## Leader-Follower Setup

Run the follower on one terminal (or machine):

```bash
python examples/minimum_gello/minimum_gello.py \
    --gripper linear_4310 --mode follower --can-channel can0
```

Run the leader on another:

```bash
python examples/minimum_gello/minimum_gello.py \
    --gripper yam_teaching_handle_v1 --mode leader --can-channel can1 --bilateral-kp 0.2
```

Press **button 0** on the teaching handle to synchronize. Press again to desync.

> **Note:** Leader mode requires real hardware (`--sim` is not supported).

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--arm` | `yam` | Arm type (`yam`, `yam_pro`, `yam_ultra`, `yam_ultra_2`, `big_yam`, `no_arm`) |
| `--gripper` | `yam_teaching_handle_v1` | Gripper type (`crank_4310`, `linear_3507`, `linear_4310`, `flexible_4310`, `yam_teaching_handle_v1`, `yam_teaching_handle_v2`, `no_gripper`) |
| `--mode` | `follower` | Operation mode (`follower`, `leader`, `visualizer_local`, `visualizer_remote`) |
| `--sim` | `False` | Use SimRobot instead of real hardware |
| `--can-channel` | `can0` | CAN interface name |
| `--server-host` | `localhost` | Portal server host (for leader/remote visualizer) |
| `--server-port` | `11333` | Portal server port |
| `--bilateral-kp` | `0.0` | Bilateral force feedback gain (leader mode) |
| `--ee-mass` | XML default | Override the end-effector body's mass (kg) for gravity compensation. See [Overriding Handle Weight](#overriding-handle-weight) |

## Overriding Handle Weight

Both shipped teaching-handle models carry the mass their ONShape export reports, and those
exports have real materials assigned. `yam_teaching_handle_v1` is 0.252595 kg of handle shell plus
0.0049991 kg of trigger, so 0.2576 kg hangs off the flange; `yam_teaching_handle_v2` is
0.238592 kg plus 0.00499911 kg, so 0.2436 kg. Gravity compensation already sees whichever you
select, and a stock handle needs no override.

`--ee-mass` is for handles that differ — older 3D-printed shells in particular. Weigh yours and
pass it only if it disagrees with the default for the handle you selected:

```bash
python examples/minimum_gello/minimum_gello.py --ee-mass 0.31 --can-channel can0
```

The override replaces the mass of the composed end-effector body outright: it sets the handle
shell's mass only, leaving the trigger's 0.005 kg alone, and it does not rescale the inertia
tensor, which stays at the export's values.
