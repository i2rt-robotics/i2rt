# YAM Arm — Hardware Setup

Get a single YAM arm out of the box, wired, powered, and ready to run a [demo](/getting-started/demos/yam).

::: tip Prerequisite
Finish [SW Setup](/getting-started/sw-setup) first — Python SDK + CAN environment must be ready.
:::

## Checklist

### 1. Unbox

- [ ] Take the arm and base plate out of the foam
- [ ] Verify accessories: CANable USB-CAN adapter, power supply, gripper
- [ ] Inspect for shipping damage on the joint covers and cables

### 2. Mount on a stable surface

- [ ] Bolt the base plate to a workbench (or place on a heavy table)
- [ ] Keep at least 1 m clearance in all directions for the arm's reach
- [ ] Route the CAN + power cables away from the workspace

### 3. Wire it up

- [ ] Connect the **CAN cable** between the arm and your CANable adapter
- [ ] Plug the CANable into a USB port on the host computer
- [ ] Connect the **24 V power supply** to the arm's power input

### 4. Power on

- [ ] Flip the supply switch — joints should hum briefly as motors initialize
- [ ] Verify the CAN device shows up:
  ```bash
  ls -l /sys/class/net/can*
  ```

### 5. Verify

```bash
# Bring up CAN if not auto-enabled
sudo ip link set can0 up type can bitrate 1000000

# Quick zero-gravity test — the arm should float
python i2rt/robots/get_robot.py --channel can0 --gripper linear_4310
```

Push the arm gently — it should hold position when released.

## Done

You're ready to run the [YAM demo](/getting-started/demos/yam).

For full specs, API reference, examples, and tuning — see the [YAM product page](/products/yam).
