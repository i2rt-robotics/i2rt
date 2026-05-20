# YAM Box — Hardware Setup

YAM Box is an enclosed manipulation station built around a YAM arm.

::: tip Prerequisite
Finish [SW Setup](/getting-started/sw-setup) first.
:::

## Checklist

### 1. Unbox & inventory

- [ ] Aluminum frame panels (×4) + base plate
- [ ] YAM arm + gripper
- [ ] CANable USB-CAN adapter
- [ ] 24 V power supply
- [ ] M5 / M6 hardware bag
- [ ] Acrylic / polycarbonate side panels (if equipped)

### 2. Assemble the frame

The full assembly is documented step-by-step on the [YAM Box product page](/products/yam-box) — 9 illustrated steps from base plate to power-on.

In short:

- [ ] Build the base + corner posts
- [ ] Attach the top frame
- [ ] Mount the YAM arm on the base plate inside the frame
- [ ] Install side panels
- [ ] Route CAN + power through the cable channel
- [ ] Power on

### 3. Verify

```bash
python i2rt/robots/get_robot.py --channel can0 --gripper linear_4310
```

Arm should float in zero-gravity mode inside the enclosure.

## Done

For the full illustrated assembly guide and YAM Box demo — see the [YAM Box product page](/products/yam-box).
