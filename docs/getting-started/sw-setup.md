# SW Setup

Set up the i2rt Python SDK and CAN bus environment. This is a **one-time setup** that all I2RT products share — once it's done, you can move on to the per-product [Hardware Setup](/getting-started/hardware/yam) checklists.

## Prerequisites

| Requirement | Version |
|-------------|---------|
| OS | Ubuntu 22.04 LTS (recommended) |
| Python | 3.10+ (3.11 recommended) |
| CAN adapter | CANable or compatible USB-CAN device |
| Build tools | `build-essential`, `python3-dev` |

::: tip Raspberry Pi users
The Flow Base ships with a Raspberry Pi pre-configured with the SDK. You can skip this page entirely and jump to [Flow Base Hardware Setup](/getting-started/hardware/flow-base).
:::

## 1. Install from source

### 1.1 Clone the repository

```bash
git clone https://github.com/i2rt-robotics/i2rt.git
cd i2rt
```

### 1.2 Install `uv` (fast Python package manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

### 1.3 Create a virtual environment

```bash
uv venv --python 3.11
source .venv/bin/activate
```

### 1.4 Install system build dependencies

```bash
sudo apt update
sudo apt install build-essential python3-dev linux-headers-$(uname -r)
```

### 1.5 Install the package

```bash
uv pip install -e .
```

The `-e` flag installs in editable mode so you can modify source files and run examples without reinstalling.

## 2. Verify the install

```bash
python -c "import i2rt; print('i2rt installed successfully')"
```

Or launch the MuJoCo simulator (no hardware required):

```bash
python examples/minimum_gello/minimum_gello.py --mode visualizer_local
```

This opens a 3D viewer with the YAM arm model.

## 3. CAN bus setup

All I2RT arms and the Flow Base communicate over CAN bus at **1 Mbit/s**.

### Bring up the interface

```bash
# Check which CAN devices are detected
ls -l /sys/class/net/can*

# Bring up the interface at 1 Mbit/s
sudo ip link set can0 up type can bitrate 1000000
```

### Auto-enable on boot (recommended)

```bash
sudo sh devices/install_devices.sh
```

This installs a udev rule that runs `ip link set ... up` for every `can*` interface.

### Reset a stuck CAN device

```bash
sh scripts/reset_all_can.sh
```

If you see `RTNETLINK answers: Device or resource busy`, unplug and replug the USB adapter.

## 4. Persistent CAN names (multi-arm only)

For multi-arm setups (e.g. YAM Cell with 4 arms), assign deterministic names like `can_follower_l` instead of `can0`/`can1`.

### Find sysfs paths

```bash
ls -l /sys/class/net/can*
```

### Read the serial number (plug adapters one at a time)

```bash
udevadm info -a -p /sys/class/net/can0 | grep -i serial
```

### Create udev rules

```bash
sudo vim /etc/udev/rules.d/90-can.rules
```

Add one line per adapter:

```
SUBSYSTEM=="net", ACTION=="add", ATTRS{serial}=="004E00275548501220373234", NAME="can_follower_l"
SUBSYSTEM=="net", ACTION=="add", ATTRS{serial}=="0031005F5548501220373234", NAME="can_follower_r"
```

::: warning Interface name limit
Names must start with `can` and be **13 characters or fewer**.
:::

### Reload and verify

```bash
sudo udevadm control --reload-rules && sudo systemctl restart systemd-udevd && sudo udevadm trigger
ip link show
```

**Naming convention for YAM Cell:**

| Arm | Interface name |
|-----|---------------|
| Left follower | `can_follower_l` |
| Right follower | `can_follower_r` |
| Left leader | `can_leader_l` |
| Right leader | `can_leader_r` |

## Next step

Pick the product you're setting up:

- [YAM Arm](/getting-started/hardware/yam)
- [YAM Cell](/getting-started/hardware/yam-cell)
- [YAM Box](/getting-started/hardware/yam-box)
- [Flow Base](/getting-started/hardware/flow-base)
- [Linear Bot](/getting-started/hardware/linear-bot)
- [Motors](/getting-started/hardware/motors)
