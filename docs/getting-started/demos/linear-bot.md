# Linear Bot Demo — Mobile Manipulation

Drive the Flow Base, lift the rail, and command the YAM arm — all from one Python script.

::: tip Prerequisite
[Linear Bot Hardware Setup](/getting-started/hardware/linear-bot) done — base + rail + arm all functional.
:::

## 1. Joystick (base + rail)

```bash
ssh i2rt@172.6.2.20
python i2rt/flow_base/flow_base_controller.py
```

| Input | Action |
|---|---|
| Left joystick | Base XY translation |
| Right joystick X | Base rotation |
| **Right joystick Y** | **Linear rail lift** |
| Left2 | Override API commands |

## 2. Combined Python API — drive + raise rail + arm

```python
from i2rt.flow_base.flow_base_client import FlowBaseClient
from i2rt.robots.get_robot import get_yam_robot
import numpy as np
import time

# Enable linear rail support on the client
client = FlowBaseClient(host="172.6.2.20", with_linear_rail=True)
robot = get_yam_robot(channel="can0")

# 4D velocity command: [vx, vy, theta_dot, rail_velocity]
client.set_target_velocity(np.array([0.1, 0.0, 0.0, 0.2]), frame="local")
time.sleep(2.0)

# Stop everything
client.set_target_velocity(np.zeros(4), frame="local")

# Inspect rail state
print(client.get_linear_rail_state())

client.close()
robot.close()
```

## 3. Rail-only control

```python
client.set_linear_rail_velocity(0.5)   # rad/s — raise
client.set_linear_rail_velocity(0.0)   # stop
client.set_linear_rail_velocity(-0.5)  # lower
```

::: tip Linear rail safety
The rail homes to the lower limit on init and has hardware limit switches. Commands time out after **0.25 s** of inactivity.
:::

## Next steps

- Full mobile manipulation walkthrough, telemetry, mapping → [Linear Bot product page](/products/linear-bot)
- Flow Base alone (no arm/rail) → [Flow Base demo](/getting-started/demos/flow-base)
