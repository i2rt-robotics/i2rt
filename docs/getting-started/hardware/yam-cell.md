# YAM Cell — Hardware Setup

Set up a 4-arm bimanual teleoperation cell: 2 leader arms + 2 follower arms, each on its own CAN channel.

::: tip Prerequisites
- Finish [SW Setup](/getting-started/sw-setup) first
- Two YAM follower arms (any gripper) + two YAM leader arms (`yam_teaching_handle` gripper)
- Four CANable USB-CAN adapters
:::

## Checklist

### 1. Mount the 4 arms

- [ ] Mount **both followers** to the front workbench
- [ ] Mount **both leaders** to the operator-side workbench (within arm's reach of the operator)
- [ ] Maintain mirror symmetry — left leader ↔ left follower, right leader ↔ right follower

### 2. Wire CAN + power

- [ ] Connect a separate CANable adapter to each arm (4 total)
- [ ] Plug all 4 CANable adapters into the host PC's USB ports
- [ ] Power each arm independently from its 24 V supply

### 3. Assign persistent CAN names

Without persistent names you can't tell which `can0…can3` belongs to which arm. Follow the [persistent CAN names](/getting-started/sw-setup#_4-persistent-can-names-multi-arm-only) section in SW Setup to set up:

| Arm | Interface name |
|-----|---------------|
| Left follower | `can_follower_l` |
| Right follower | `can_follower_r` |
| Left leader | `can_leader_l` |
| Right leader | `can_leader_r` |

### 4. Verify all 4 arms

```bash
ip link show | grep can_
```

All 4 named interfaces should be **UP**.

### 5. Quick floating test (one arm at a time)

```bash
python i2rt/robots/get_robot.py --channel can_follower_l --gripper linear_4310
```

Repeat for each arm to confirm each CAN channel maps to the expected arm.

## Done

You're ready to run the [bimanual teleop demo](/getting-started/demos/yam-cell).

For full specs and examples — see the [YAM Cell product page](/products/yam-cell).
