# Troubleshooting

## CAN Bus Issues

### `ls -l /sys/class/net/can*` shows no devices

- Unplug and replug the USB-CAN adapter
- Try a different USB port
- Run `dmesg | tail -20` to check for USB errors
- Verify the adapter firmware (some CANable units need flashing)

### `RTNETLINK answers: Device or resource busy`

```bash
sh scripts/reset_all_can.sh
```

If that fails, unplug the USB-CAN adapter, wait 5 seconds, and replug.

### CAN interface is `UP` but no motor response

- Verify CAN bitrate matches: `1000000` (1 Mbit/s)
- Check that the motor is powered (indicator LED on)
- Check cable polarity (CAN-H, CAN-L, GND)
- Verify motor ID matches what the code expects

---

## YAM Arm Issues

### Motor timeout errors (arm enters damping mode)

The default 400 ms timeout is triggering — your control loop is too slow.

Options:
1. Optimize the control loop to send commands faster than 400 ms
2. Disable the timeout for testing (see [Hardware Setup](/getting-started/hardware-setup#motor-configuration))
3. Contact [sales@i2rt.com](mailto:sales@i2rt.com) for bulk firmware with timeout disabled

### Arm drifts or vibrates in zero-gravity mode

- Add a PD target to stabilize when using no timeout:
  ```python
  robot = get_yam_robot(channel="can0", zero_gravity_mode=False)
  ```
- Check for loose cable connections at each joint
- Verify the URDF/MuJoCo model matches your arm configuration

### Gripper doesn't reach full open/close (linear gripper)

The linear gripper needs to start in the closed position. Calibrate:
1. Power off the arm
2. Manually close the gripper fingers completely
3. Power on and initialize normally

---

## Flow Base Issues

### Remote is unresponsive

Toggle the remote power switch off, wait 3 seconds, power back on. The remote enters sleep mode after inactivity.

### Odometry is inaccurate

Wheel odometry is inherently imprecise. For precise localization:
- Use `reset_odometry()` at known positions
- Integrate a visual odometry sensor (RealSense T265, ZED)

### Linear rail not homing

1. Check GPIO connections between the Pi and limit switch
2. Run `get_linear_rail_state()` to inspect switch state
3. Manually verify both limit switches activate when pressed

### Base keeps stopping unexpectedly

The 0.25 s velocity timeout is triggering. Ensure `FlowBaseClient` maintains its heartbeat — verify your control loop sends commands at > 4 Hz.

### `RTNETLINK answers: Device or resource busy` (Flow Base Pi)

```bash
ssh i2rt@172.6.2.20
sh ~/i2rt/scripts/reset_all_can.sh
```

---

## Installation Issues

### `uv pip install -e .` fails with build errors

```bash
sudo apt install build-essential python3-dev linux-headers-$(uname -r)
uv pip install -e .
```

### MuJoCo viewer doesn't open

- Ensure you have a display (or set up virtual framebuffer for headless)
- Try: `export DISPLAY=:0` before running
- On SSH sessions: `export MUJOCO_GL=egl` for off-screen rendering

---

## Getting Help

| Channel | Link |
|---------|------|
| GitHub Issues | [github.com/i2rt-robotics/i2rt/issues](https://github.com/i2rt-robotics/i2rt/issues) |
| Discord | [discord.gg/i2rt](https://discord.gg/i2rt) |
| Email | [support@i2rt.com](mailto:support@i2rt.com) |
| Sales | [sales@i2rt.com](mailto:sales@i2rt.com) |
