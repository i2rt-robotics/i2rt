# Bimanual Teleoperation

**Location:** `examples/bimanual_lead_follower/`

Run coordinated dual-arm teleoperation with two leader and two follower YAM arms. This is the primary example for the [YAM Cell](/products/yam-cell).

## Hardware Required

- 2× YAM follower arms (any gripper)
- 2× YAM leader arms (with `yam_teaching_handle` gripper)
- 4× CANable USB-CAN adapters
- 1× Workstation PC (Ubuntu 22.04)

## Setup

<MediaPlaceholder
  type="photo"
  description="Bimanual YAM Cell setup on a table: leader arms on the left (operator side), follower arms on the right (task side). All four arms visible, cables routed neatly."
/>

### 1. Assign persistent CAN IDs

Plug **one CANable at a time** and follow the [Persistent CAN IDs](/getting-started/hardware-setup#persistent-can-ids) guide to assign each adapter a fixed name.

Target layout:

| Arm | Interface |
|-----|-----------|
| Left follower | `can_follower_l` |
| Right follower | `can_follower_r` |
| Left leader | `can_leader_l` |
| Right leader | `can_leader_r` |

### 2. Verify all four interfaces

```bash
ip a | grep can
# Expected:
# can_follower_r  UP
# can_follower_l  UP
# can_leader_r    UP
# can_leader_l    UP
```

### 3. Activate the virtual environment

```bash
source .venv/bin/activate
```

### 4. Launch

```bash
python examples/bimanual_lead_follower/bimanual_lead_follower.py
```

## Operation

| Action | Result |
|--------|--------|
| Move either leader arm | Corresponding follower mirrors motion |
| Squeeze trigger on teaching handle | Follower gripper closes |
| **Top button (press once)** | **Enable synchronization** |
| **Top button (press again)** | **Disable synchronization** |

::: tip Start position
Before enabling sync, move the leader arms to roughly match the follower arm positions. Large position errors on first sync can cause abrupt motion.
:::

## Video

<MediaPlaceholder
  type="video"
  description="Bimanual teleoperation demo: operator moves two leader arms to pick up objects with two follower arms simultaneously. Tasks shown: pick-and-place, handoff, assembly. 2–3 minutes."
/>

<MediaPlaceholder
  type="video"
  description="Close-up: leader arm teaching handle trigger being used to operate the follower gripper while picking a small object."
/>

## Architecture

The example launches two `minimum_gello.py` instances internally — one per arm pair — sharing the same enable/disable logic through the top button.

```python
# Conceptually equivalent to:
python examples/minimum_gello/minimum_gello.py --gripper linear_4310 --mode follower --can-channel can_follower_l --bilateral_kp 0.2
python examples/minimum_gello/minimum_gello.py --gripper yam_teaching_handle --mode leader --can-channel can_leader_l --bilateral_kp 0.2
# (mirrored for right pair)
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Missing CAN interface | Check `ip a`, replug adapters one at a time |
| Arm not following | Ensure sync is enabled (top button) |
| Jittery motion | Lower `--bilateral_kp` to `0.1` |
| Motor timeout errors | Reduce loop latency; check USB-CAN adapter |

## See Also

- [YAM Cell](/products/yam-cell)
- [YAM Arm API](/sdk/yam-arm)
- [Hardware Setup](/getting-started/hardware-setup)
