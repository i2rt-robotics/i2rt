# Motors

<div class="product-badges">
  <span class="product-badge available">✓ CAN Bus</span>
  <span class="product-badge available">✓ Dual Encoder</span>
  <span class="product-badge available">✓ Integrated Driver</span>
  <span class="product-badge available">✓ Planetary Reduction</span>
</div>

I2RT's **GF series** are all-in-one joint modules — planetary gearbox, brushless DC motor, and driver integrated into a single compact housing. Gears are vacuum-nitrided for high strength and service life. The concentrated-winding motor design delivers high power density with optimized pole-arc ratio for low cogging torque and smooth motion.

## Model Comparison

| Model | Reduction Ratio | Rated Torque | Peak Torque | Rated Speed | Weight | Size (⌀ × L) |
|-------|:-:|:-:|:-:|:-:|:-:|:-:|
| [GF43X40-16](/products/motor-gf43x40-16) | 40 | 14 N·m | 23.5 N·m | 44 RPM | 464.5 g | 57 × 62.5 mm |
| [GF43X40-10](/products/motor-gf43x40-10) | 40 | 8.9 N·m | 28 N·m | 43 RPM | 362 g | 57 × 56.5 mm |
| [GF43X10-10](/products/motor-gf43x10-10) | 10 | 2.2 N·m | 7 N·m | 172 RPM | 300 g | 57 × 45.9 mm |

All models share the same 24 V supply and CAN bus interface.

## Key Features

- **Dual encoder** — single-turn absolute output-shaft position; survives power-off without losing zero reference
- **Integrated driver** — compact single-unit assembly with no external controller required
- **CAN bus feedback** — real-time speed, position, torque, and motor temperature
- **Dual temperature protection** — hardware and software over-temperature shutdown
- **Trapezoidal motion profile** — configurable acceleration/deceleration in position mode
- **Visual debugging** — host-PC GUI for tuning and firmware upgrade

## Communication

All GF series motors communicate over **CAN bus (1 Mbit/s)**, the same bus used by YAM arm actuators. Multiple motors can share a single CAN channel using standard node addressing.

## Naming Convention

```
GF  43  X  40  -  16
│   │      │      └─ Motor variant (current/winding spec)
│   │      └─────── Reduction ratio
│   └────────────── Outer diameter class (43 mm motor stator)
└────────────────── GF joint module series
```

## Hardware Setup

Using a motor **standalone** for development or testing.

::: tip Prerequisite
Finish [SW Setup](/getting-started/sw-setup) first.
:::

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

---

## Quick Start Demo

### Read motor state

```python
from i2rt.motor_drivers.dm_driver import DMChainCanInterface, MotorType

motor_chain = DMChainCanInterface(
    motor_list=[(1, MotorType.DM4310)],
    motor_offsets=[0.0],
    motor_directions=[1],
    channel="can0",
)

state = motor_chain.read_states()
print(state)
# JointState(pos=..., vel=..., eff=..., temp_mos=..., temp_rotor=...)
```

### MIT-mode position command

```python
import numpy as np

# Command motor 1 to position 0 rad with PD gains
motor_chain.set_commands(
    kp=np.array([5.0]),
    kd=np.array([0.5]),
    pos=np.array([0.0]),
    vel=np.array([0.0]),
    torque=np.array([0.0]),
)
```

::: danger Safety timeout
By default each motor has a **400 ms safety timeout**. If no command arrives for 400 ms, the motor switches to damping mode automatically. Disable only when you have a reliable PD loop running — see the [Set safety timeout](#set-safety-timeout) section below.
:::

For a full interactive PD-control teleop example (arrow-key control, live state display), see the [Single Motor PD Control](#single-motor-pd-control) section below.

---

## Single Motor PD Control

**Location:** `examples/single_motor_position_pd_control/`

The simplest possible example — command a single DM-series motor to a target position with a PD controller. Useful for testing new hardware, debugging CAN connectivity, or learning the motor driver API.

### Running

```bash
python examples/single_motor_position_pd_control/single_motor_position_pd_control.py \
  --channel can0 --motor_id 1 --motor_type DM4340 --kd 3
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--channel` | `can0` | CAN interface name |
| `--motor_id` | `1` | Motor ID on the bus (1–7 typical) |
| `--motor_type` | `DM4340` | Motor model: `DM4310`, `DM4340`, `DM6248`, `DM3507` |
| `--kp` | `80.0` | Proportional gain |
| `--kd` | `3.0` | Derivative gain |
| `--rate_hz` | `200.0` | Control loop rate |
| `--step` | `0.01` | Step size per arrow key press (rad) |

The interactive panel shows live motor state:

```
Arrow-key PD teleop (q to quit)
Current pos : -1.64359 rad
Target  pos : -1.49720 rad
Velocity    : -0.00244 rad/s
Torque      : +11.65812 Nm
Temp rotor  : 45.0 °C   Temp MOS: 35.0 °C

Step size   : 0.01000 rad   (↑ bigger / ↓ smaller)
KP=80.00  KD=3.00

Controls: ←/→ move • r reset-to-current • SPACE hold • q quit
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `←` / `→` (or `h` / `l`) | Decrease / increase target position |
| `↑` / `↓` (or `k` / `j`) | Increase / decrease step size |
| `r` | Reset target to current position |
| `Space` | Hold current position |
| `q` / `ESC` | Quit |

### Example Code

The example wraps a single motor in a `DMChainCanInterface` and runs a PD position loop:

```python
from i2rt.motor_drivers.dm_driver import DMChainCanInterface

motor_list = [[1, "DM4340"]]            # [(motor_id, motor_type), ...]
motor_directions = [1]
chain = DMChainCanInterface(
    motor_list, [0], motor_directions, channel="can0", receive_mode="p16"
)

states = chain.read_states()             # current pos / vel / torque
chain.set_commands([{"pos": 0.0, "kp": 80.0, "kd": 3.0}])

chain.close()
```

## Motor Configuration Tools

One-time motor configuration utilities live in `i2rt/motor_config_tool/`. When `--motor_id` is omitted, all three commands operate on **motors 1–7** on the bus by default — pass an explicit `--motor_id N` to target a single motor.

### Ping motors

```bash
# Ping every motor on can0 (IDs 1–7)
python i2rt/motor_config_tool/ping_motors.py --channel can0

# Ping motor 3 only
python i2rt/motor_config_tool/ping_motors.py --channel can0 --motor_id 3
```

### Zero motor offset

```bash
# Zero every motor on can0 (run with the arm in its mechanical zero pose)
python i2rt/motor_config_tool/set_zero.py --channel can0

# Zero motor 1 only
python i2rt/motor_config_tool/set_zero.py --channel can0 --motor_id 1
```

### Set safety timeout

```bash
# Disable the 400 ms motor safety timeout (default)
python i2rt/motor_config_tool/set_timeout.py --channel can0

# Re-enable the safety timeout
python i2rt/motor_config_tool/set_timeout.py --channel can0 --timeout
```

::: warning Power cycle required
After running `set_zero.py` or `set_timeout.py`, power-cycle the motor for the new configuration to take effect.
:::

## See Also

- [YAM Arm](/products/yam) — full motor chain usage in an arm
- [Motors hardware setup](/products/motors#hardware-setup)
- [Motors demo](/products/motors#quick-start-demo)

<style scoped>
.product-badges { display: flex; flex-wrap: wrap; gap: 8px; margin: 16px 0 24px; }
.product-badge { display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; border: 1px solid; }
.product-badge.available { color: #4C6762; border-color: rgba(76,103,98,0.4); background: rgba(76,103,98,0.08); }
</style>
