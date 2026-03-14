# Hardware Setup

## CAN Bus Setup

All I2RT arms and the Flow Base communicate over CAN bus at **1 Mbit/s**. Each arm requires one USB-CAN adapter (CANable or compatible).

### 1. Bring up the CAN interface

```bash
# Check which CAN devices are detected
ls -l /sys/class/net/can*

# Bring up the interface at 1 Mbit/s
sudo ip link set can0 up type can bitrate 1000000
```

### 2. (Recommended) Auto-enable on boot

Install the provided udev rule so CAN interfaces come up automatically:

```bash
sudo sh devices/install_devices.sh
```

This installs a udev rule that runs `ip link set ... up` for every interface whose name starts with `can*`.

::: warning Persistent CAN IDs
If you later assign persistent names (e.g. `can_follower_l`), you may need to update the udev rule to match. See [Persistent CAN IDs](#persistent-can-ids).
:::

### 3. Reset CAN devices

If a CAN device becomes unresponsive:

```bash
sh scripts/reset_all_can.sh
```

If you see `RTNETLINK answers: Device or resource busy`, unplug and replug the USB adapter.

## Persistent CAN IDs

For multi-arm setups (e.g. YAM Cell with 4 arms), assign deterministic names to each adapter using udev rules based on the USB device's serial number or path.

::: tip CANable firmware
Visit [canable.io/updater](https://canable.io/updater/) to flash candlelight firmware if needed. YAM ships with candlelight pre-flashed.
:::

### Step 1 — Find sysfs paths

```bash
ls -l /sys/class/net/can*
# lrwxrwxrwx ... /sys/class/net/can0 -> ../../devices/.../can0
# lrwxrwxrwx ... /sys/class/net/can1 -> ../../devices/.../can1
```

### Step 2 — Read the serial number

Plug adapters **one at a time** and note the serial for each:

```bash
udevadm info -a -p /sys/class/net/can0 | grep -i serial
```

### Step 3 — Create udev rules

```bash
sudo vim /etc/udev/rules.d/90-can.rules
```

Add one line per adapter:

```
SUBSYSTEM=="net", ACTION=="add", ATTRS{serial}=="004E00275548501220373234", NAME="can_follower_l"
SUBSYSTEM=="net", ACTION=="add", ATTRS{serial}=="0031005F5548501220373234", NAME="can_follower_r"
```

::: warning Name length limit
Interface names must start with `can` and be **13 characters or fewer**.
:::

### Step 4 — Reload and verify

```bash
sudo udevadm control --reload-rules && sudo systemctl restart systemd-udevd && sudo udevadm trigger
```

Unplug and replug each adapter, then verify:

```bash
ip link show
```

You should see the named interfaces (`can_follower_l`, etc.) with state `UP`.

**Recommended naming convention for YAM Cell:**

| Arm | Interface name |
|-----|---------------|
| Left follower | `can_follower_l` |
| Right follower | `can_follower_r` |
| Left leader | `can_leader_l` |
| Right leader | `can_leader_r` |

## Motor Configuration

### Zero offsets

If the arm was disassembled or the motors replaced, you may need to re-zero them:

```bash
python i2rt/motor_config_tool/set_zero.py --channel can0 --motor_id 1
```

Run for each motor ID (1–6 for a standard YAM).

### Safety timeout

The factory default is a **400 ms safety timeout**: if no command is received within 400 ms, the motor enters damping mode. This prevents runaway behavior if the CAN connection drops.

**Disable timeout (advanced users only):**

```bash
python i2rt/motor_config_tool/set_timeout.py --channel can0
python i2rt/motor_config_tool/set_timeout.py --channel can0  # run twice
```

**Re-enable timeout:**

```bash
python i2rt/motor_config_tool/set_timeout.py --channel can0 --timeout
```

::: danger Safety note
Disabling the timeout removes a key safety mechanism. Without it, a failed gravity-compensation loop can produce uncontrolled positive-feedback torque. Always set a PD target when operating without a timeout:

```python
robot = get_yam_robot(channel="can0", zero_gravity_mode=False)
```
:::

## Flow Base Hardware

<MediaPlaceholder
  type="photo"
  description="Flow Base control panel labeled diagram: E-stop location, CAN selector switch (UP = internal Pi, DOWN = external CAN), battery connector, RJ45 port."
/>

### Initial setup

1. Install the battery and power on — the Raspberry Pi display lights up
2. Verify the **E-stop is not pressed** (twist to release)
3. Set the **CAN selector switch to UP** (uses on-board Pi)
4. SSH in once the Pi has booted:

```bash
ssh i2rt@172.6.2.20   # wired Ethernet (static IP)
# Password: root
```

### Firmware update

If the pre-installed software is outdated:

```bash
ssh i2rt@172.6.2.20
cd ~/i2rt && git pull
```

For Pi OS firmware, see the [Pi SD Card Guide](#pi-sd-card-backup--restore) below.

## Pi SD Card Backup & Restore

### 1. Find the SD card device

```bash
lsblk
```

### 2. Create image from SD card

```bash
sudo dd if=/dev/sdX of=pi_system.img bs=4M status=progress
sync
```

*(replace `/dev/sdX` with your SD card device, e.g. `/dev/sdc`)*

### 3. Flash image to a new SD card

```bash
sudo wipefs -a /dev/sdX
sudo dd if=pi_system.img of=/dev/sdX bs=4M status=progress
sync
```

### 4. Eject

```bash
eject /dev/sdX
```

### 5. Shrink image size (optional)

```bash
sudo ./pishrink.sh pi_system.img pi_system_shrunk.img
```

The latest pre-built Pi firmware image is available in the [Flow Base firmware folder](https://drive.google.com/drive/folders/1BAvdCFFR2lsmHqKH9YQ_lMbPV0TAIKik).

## Connecting YAM to Flow Base

1. Mount the YAM arm on the Flow Base top plate
2. Route the CAN cable through the internal cable management
3. Connect the arm CAN cable to the on-board CANable adapter
4. Set the CAN selector switch to **UP** (Pi mode)
5. SSH into the Pi and launch both arm and base controllers

```bash
ssh i2rt@172.6.2.20
python i2rt/flow_base/flow_base_controller.py &
python scripts/minimum_gello.py --gripper linear_4310 --mode follower --can-channel can0
```
