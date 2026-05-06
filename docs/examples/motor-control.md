# Single Motor PD Control

**Location:** `examples/single_motor_position_pd_control/`

The simplest possible example — command a single DM-series motor to a target position with a PD controller. Useful for testing new hardware, debugging CAN connectivity, or learning the motor driver API.

## Hardware Required

- 1× DM-series motor
- 1× CANable USB-CAN adapter

## Running

```bash
python examples/single_motor_position_pd_control/single_motor_position_pd_control.py \
  --channel can0 --motor_id 1 --motor_type DM4340 --kd 3
```

Common arguments:

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

## Example Code

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

- [YAM Arm API](/sdk/yam-arm)
- [Hardware Setup](/getting-started/hardware-setup)
