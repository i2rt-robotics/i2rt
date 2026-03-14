# Installation

This guide covers installing the `i2rt` Python package from source on Ubuntu (recommended) or any Linux system with CAN bus support.

## Prerequisites

| Requirement | Version |
|-------------|---------|
| OS | Ubuntu 22.04 LTS (recommended) |
| Python | 3.11 |
| CAN adapter | CANable or compatible USB-CAN device |
| Build tools | `build-essential`, `python3-dev` |

::: tip Using a Raspberry Pi?
The Flow Base ships with a Raspberry Pi that is pre-configured with the latest firmware. See the [Hardware Setup](/getting-started/hardware-setup) guide for SSH access and update instructions.
:::

## Install from Source

### 1. Clone the repository

```bash
git clone https://github.com/i2rt-robotics/i2rt.git
cd i2rt
```

### 2. Install `uv` (fast Python package manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

### 3. Create a virtual environment

```bash
uv venv --python 3.11
source .venv/bin/activate
```

### 4. Install system build dependencies

```bash
sudo apt update
sudo apt install build-essential python3-dev linux-headers-$(uname -r)
```

### 5. Install the package

```bash
uv pip install -e .
```

The `-e` flag installs in editable mode so you can modify source files and run examples without reinstalling.

## Verify Installation

```python
python -c "import i2rt; print('i2rt installed successfully')"
```

## Optional: MuJoCo Visualization

The simulator requires MuJoCo, which is installed automatically as a dependency. Verify it works:

```bash
python examples/minimum_gello/minimum_gello.py --mode visualizer_local
```

This opens the MuJoCo viewer with the YAM arm model. No hardware required.

## Next Steps

- [Hardware Setup](/getting-started/hardware-setup) — CAN bus configuration
- [Quick Start](/getting-started/quick-start) — move the arm in 5 minutes
