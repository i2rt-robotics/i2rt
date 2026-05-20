# Motors — Hardware Setup

I2RT motors (DM and GF series) communicate over CAN bus and are used in YAM arms, Flow Base, and Linear Bot. This checklist covers using a motor **standalone** for development or testing.

::: tip Prerequisite
Finish [SW Setup](/getting-started/sw-setup) first.
:::

## Checklist

### 1. Inventory

- [ ] Motor
- [ ] Motor cable (CAN + power)
- [ ] CANable USB-CAN adapter
- [ ] 24 V or 48 V power supply (check motor label)

### 2. Wire it up

- [ ] Connect the motor's CAN cable to your CANable adapter
- [ ] Connect the motor's power input to the supply
- [ ] Plug the CANable into the host PC's USB

### 3. Bring up CAN

```bash
sudo ip link set can0 up type can bitrate 1000000
ls -l /sys/class/net/can*
```

### 4. (If needed) Set the motor ID

Each motor needs a unique ID on the bus (1–6 for YAM, custom for standalone).

```bash
python i2rt/motor_config_tool/set_id.py --channel can0 --old-id 1 --new-id 2
```

### 5. (If needed) Zero the motor offset

```bash
python i2rt/motor_config_tool/set_zero.py --channel can0 --motor_id 1
```

### 6. Quick spin test

See the [Motors demo](/getting-started/demos/motors) for a simple position command example.

## Done

For the full motor catalog (DM4310, DM4340, DM6248, GF series) and gravity-comp tuning — see the [Motors product page](/products/motors).
