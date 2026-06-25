# Bimanual Lead-Follower System Setup

This tutorial demonstrates how to set up a bimanual lead-follower system. This tutorial assumes you've followed the main README and finished setting up the environment.

## Required Hardware

- **Two YAM Leader arms**
- **Two YAM Follower arms**

## Hardware Setup

1. Install two YAM leader and two YAM follower arms safely on the tabletop
2. Connect the arms to power
3. Set up the CANable devices
4. **Important:** Do NOT plug any CANable devices into your computer yet

## Setup Different CAN IDs for Leader and Follower Arms

**⚠️ Critical Step:** Plug in **one CAN device at a time** to avoid conflicts.

1. Follow the instructions in [set-persistent-can-ids.md](../../docs/guides/set-persistent-can-ids.md) to configure each arm:
   - **Left leader arm:** `can_leader_l`
   - **Right leader arm:** `can_leader_r`
   - **Left follower arm:** `can_follower_l`
   - **Right follower arm:** `can_follower_r`

> **⚠️ This step is critical!** Our code uses these CAN IDs to match arms together.

## Verification

After setting up all CAN IDs:

1. Plug in all 4 CAN devices
2. Run `ip a` to verify the CAN devices are mapped correctly
3. You should see output similar to:

```bash
i2rt@ioheart-dev03:~/i2rt$
.........
.........
9: can_follower_r: <NOARP,UP,LOWER_UP,ECHO> mtu 16 qdisc pfifo_fast state UP group default qlen 10
    link/can
10: can_follower_l: <NOARP,UP,LOWER_UP,ECHO> mtu 16 qdisc pfifo_fast state UP group default qlen 10
    link/can
11: can_leader_r: <NOARP,UP,LOWER_UP,ECHO> mtu 16 qdisc pfifo_fast state UP group default qlen 10
    link/can
12: can_leader_l: <NOARP,UP,LOWER_UP,ECHO> mtu 16 qdisc pfifo_fast state UP group default qlen 10
    link/can
```

## Launch the System

Once all CAN devices are properly configured and connected, launch the system:
Assume you've installed the python virtual environment and run `source .venv/bin/activate`

```bash
python bimanual_lead_follower.py
```

## Operation

- **Press the top button on the teaching handle to enable the system**
- This code essentially launches two `minimum_gello.py` instances for bimanual control

## Troubleshooting

- If you don't see all four CAN interfaces, double-check that:
  - All CAN devices are properly connected
  - Each device has a unique CAN ID as configured
  - The devices are powered on and in UP state
- Use `ip link show` to check the status of individual CAN interfaces
