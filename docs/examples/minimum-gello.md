# Minimum Gello (Teleoperation)

**Location:** `examples/minimum_gello/`

The minimal leader–follower teleoperation script. Supports any YAM-family arm + gripper assembly, simulation mode, and local or remote visualization. This is the foundation that [Bimanual Teleoperation](/examples/bimanual-teleoperation) builds on.

## Hardware Required

- None for simulation (`--sim` flag, follower-only).
- Single-arm teleoperation: 1× YAM follower + 1× YAM leader (with `yam_teaching_handle`) + 2× CANable adapters.
- Visualizer-only: 1× YAM arm or sim.

## Modes

| Mode | What it does |
|------|--------------|
| `follower` *(default)* | Drives the local robot from commands received over a portal server. Used as the receiving side in a leader→follower pair. |
| `leader` | Reads a local teaching handle and sends commands to a remote follower. **Requires real hardware** — `--sim` is not supported in leader mode. |
| `visualizer_local` | MuJoCo viewer mirrors the local robot's live state. No motion is commanded. |
| `visualizer_remote` | MuJoCo viewer mirrors a remote robot's state via the portal server. |

## Quick Start

```bash
# Follower (default) on real hardware
python examples/minimum_gello/minimum_gello.py --can-channel can0

# Follower in simulation — no hardware required
python examples/minimum_gello/minimum_gello.py --sim

# Live MuJoCo viewer for the local robot
python examples/minimum_gello/minimum_gello.py --mode visualizer_local

# Try a different arm + gripper combination in sim
python examples/minimum_gello/minimum_gello.py --arm big_yam --gripper linear_4310 --sim
python examples/minimum_gello/minimum_gello.py --arm yam_pro --gripper flexible_4310 --sim
```

## Leader → Follower Setup

Run the follower on one terminal (or machine):

```bash
python examples/minimum_gello/minimum_gello.py \
    --gripper linear_4310 --mode follower --can-channel can0
```

Run the leader on another (separate CAN bus, real hardware only):

```bash
python examples/minimum_gello/minimum_gello.py \
    --gripper yam_teaching_handle --mode leader --can-channel can1 --bilateral-kp 0.2
```

Press **button 0** on the teaching handle to sync the leader to the follower. Press again to desync.

::: tip Bilateral force feedback
`--bilateral-kp` (default 0.0) controls how much the follower's load is reflected back to the leader. Try 0.1–0.3 to feel object weight; values >0.3 can feel sluggish.
:::

## Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--arm` | `yam` | Arm type: `yam`, `yam_pro`, `yam_ultra`, `big_yam`, `no_arm` |
| `--gripper` | `yam_teaching_handle` | Gripper: `crank_4310`, `linear_3507`, `linear_4310`, `flexible_4310`, `yam_teaching_handle`, `no_gripper` |
| `--mode` | `follower` | Operation mode (see table above) |
| `--sim` | off | Use `SimRobot` instead of real hardware (follower / visualizer only) |
| `--can-channel` | `can0` | CAN interface name |
| `--server-host` | `localhost` | Portal server host (used by leader / remote visualizer) |
| `--server-port` | `11333` | Portal server port |
| `--bilateral-kp` | `0.0` | Bilateral force feedback gain (leader mode) |
| `--ee-mass` | model default | Override end-effector mass (kg) for gravity comp |

## Overriding Handle Weight

3-D-printed teaching handles vary in mass. The default model assumes 0.258 kg. If your handle is heavier or lighter, pass `--ee-mass` so gravity compensation matches the real hardware:

```bash
python examples/minimum_gello/minimum_gello.py --ee-mass 0.350 --can-channel can0
```

## See Also

- [Bimanual Teleoperation](/examples/bimanual-teleoperation) — extends this script to four arms.
- [YAM Arm API](/sdk/yam-arm) — programmatic equivalent of `--arm` / `--gripper`.
- [Viser Control Interface](/examples/control-with-viser) — browser-based alternative to MuJoCo visualizer mode.
