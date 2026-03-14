# Single Motor PD Control

**Location:** `examples/single_motor_position_pd_control/`

The simplest possible example — command a single DM-series motor to a target position with a PD controller. Useful for testing new hardware, debugging CAN connectivity, or learning the motor driver API.

## Hardware Required

- 1× DM-series motor
- 1× CANable USB-CAN adapter

## Running

```bash
python i2rt/scripts/single_motor_pd_pos_control.py --channel can0 --motor_id 1 --kd 5
```

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

```python
from i2rt.motor_drivers.dm_motor import DmMotor

motor = DmMotor(channel="can0", motor_id=1)
motor.enable()

# Command position (radians) with PD gains
motor.set_position_pd(target=0.0, kp=10.0, kd=0.5)

import time
time.sleep(2.0)
motor.disable()
```

## Expected Output

```
Motor 1: pos=0.0042  vel=0.0  torque=0.12
Motor 1: pos=0.0021  vel=-0.01  torque=0.08
Motor 1: pos=0.0003  vel=0.0  torque=0.02
```

## See Also

- [YAM Arm API](/sdk/yam-arm)
- [Hardware Setup](/getting-started/hardware-setup)
