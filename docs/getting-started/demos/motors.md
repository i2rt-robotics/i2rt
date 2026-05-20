# Motors Demo — Basic CAN Control

Spin an I2RT motor directly via CAN, without the full robot stack.

::: tip Prerequisite
[Motors Hardware Setup](/getting-started/hardware/motors) done — motor wired, CAN bus up, motor ID set.
:::

## 1. Read motor state

```python
from i2rt.motor_drivers.dm_driver import DMChainCanInterface, MotorType

# motor_list: [(id, motor_type), ...]
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

## 2. MIT-mode position command

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

## 3. Safety: timeout

By default each motor has a **400 ms safety timeout**. If no command arrives for 400 ms, the motor switches to damping mode automatically. This protects you if the CAN connection drops.

```bash
# Disable timeout (advanced users only — must run twice)
python i2rt/motor_config_tool/set_timeout.py --channel can0
python i2rt/motor_config_tool/set_timeout.py --channel can0

# Re-enable
python i2rt/motor_config_tool/set_timeout.py --channel can0 --timeout
```

::: danger Without timeout
A failed gravity-compensation loop can produce uncontrolled torque. Always send a PD target when running without the timeout safety net.
:::

## Next steps

- Full motor catalog (DM4310, DM4340, DM6248, GF series), dimensions, characteristics → [Motors product page](/products/motors)
- See how motors are used in a full arm stack → [YAM product page](/products/yam)
