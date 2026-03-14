# Minimum Gello (Teleoperation)

Minimal leader-follower teleoperation example using a YAM arm and teaching handle (gello). Supports follower mode, leader mode, and local/remote visualization.

## Quick Start

```bash
# Follower mode (default) — serves robot state over portal
python examples/minimum_gello/minimum_gello.py --can-channel can0

# Leader mode — reads teaching handle and drives a remote follower
python examples/minimum_gello/minimum_gello.py --mode leader --can-channel can1

# Visualizer — MuJoCo viewer mirrors live robot state
python examples/minimum_gello/minimum_gello.py --mode visualizer_local
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
    --gripper yam_teaching_handle --mode leader --can-channel can1 --bilateral-kp 0.2
```

Press **button 0** on the teaching handle to synchronize. Press again to desync.

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--gripper` | `yam_teaching_handle` | Gripper type (`crank_4310`, `linear_3507`, `linear_4310`, `yam_teaching_handle`, `no_gripper`) |
| `--mode` | `follower` | Operation mode (`follower`, `leader`, `visualizer_local`, `visualizer_remote`) |
| `--can-channel` | `can0` | CAN interface name |
| `--server-host` | `localhost` | Portal server host (for leader/remote visualizer) |
| `--server-port` | `11333` | Portal server port |
| `--bilateral-kp` | `0.0` | Bilateral force feedback gain (leader mode) |
| `--ee-mass` | XML default | Override link_6 mass (kg) for gravity compensation with different handles |

## Overriding Handle Weight

Older 3D-printed teaching handles may have a different mass than the default (0.258 kg). Use `--ee-mass` to compensate:

```bash
python examples/minimum_gello/minimum_gello.py --ee-mass 0.350 --can-channel can0
```
