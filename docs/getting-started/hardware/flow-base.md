# Flow Base — Hardware Setup

Get the Flow Base omnidirectional mobile platform unboxed, powered, and joystick-ready.

::: tip Prerequisite
SW Setup is **not required** — the Flow Base ships with a Raspberry Pi pre-configured with the SDK. You only need network access to SSH in.
:::

## Checklist

### 1. Unbox

- [ ] Flow Base chassis
- [ ] Battery pack (already installed)
- [ ] Joystick remote controller
- [ ] Ethernet cable (for wired SSH)
- [ ] Charger

### 2. Power on

- [ ] Twist the **E-stop button** counter-clockwise to release it
- [ ] Set the **CAN selector switch** to **UP** position (selects the on-board Pi as CAN master)
- [ ] Press the power button — the Pi display should light up
- [ ] Wait ~30 seconds for the Pi to finish booting

### 3. Connect to the Pi

```bash
# Wired (recommended) — static IP
ssh i2rt@172.6.2.20
# Password: root
```

::: tip Wi-Fi setup
If you prefer Wi-Fi, connect a keyboard + monitor to the Pi the first time and run `sudo raspi-config` to join your network.
:::

### 4. Verify the SDK is current

```bash
cd ~/i2rt && git pull
```

### 5. Test with the joystick

On the Pi:

```bash
python i2rt/flow_base/flow_base_controller.py
```

- [ ] Left joystick → base translates (XY)
- [ ] Right joystick X → base rotates (yaw)
- [ ] Press **Left2** to override API commands (safety)

## Done

You're ready to run the [Flow Base demo](/getting-started/demos/flow-base).

For full specs, API reference, and remote control layout — see the [Flow Base product page](/products/flow-base).
