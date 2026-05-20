# Flow Base Demo — Joystick & API Control

Drive the Flow Base with the joystick remote, then control it programmatically.

::: tip Prerequisite
[Flow Base Hardware Setup](/getting-started/hardware/flow-base) done — base powered, SSH access working.
:::

## 1. Joystick demo (on the Pi)

```bash
ssh i2rt@172.6.2.20
python i2rt/flow_base/flow_base_controller.py
```

| Input | Action |
|---|---|
| Left joystick | XY translation |
| Right joystick X | Rotation (yaw) |
| Right joystick Y | Linear rail lift (if equipped) |
| Left1 | Reset odometry |
| Mode | Toggle local ↔ global frame |
| Left2 | Override API commands |

## 2. Python API — drive forward from a laptop

From your laptop (on the same network):

```python
from i2rt.flow_base.flow_base_client import FlowBaseClient
import numpy as np
import time

client = FlowBaseClient(host="172.6.2.20")

# Drive forward at 0.1 m/s for 2 seconds
start = time.time()
while time.time() - start < 2.0:
    client.set_target_velocity(np.array([0.1, 0.0, 0.0]), frame="local")
    time.sleep(0.05)

# Read odometry
print(client.get_odometry())
client.close()
```

## 3. Velocity command format

```python
# 3D velocity command [vx, vy, theta_dot] in m/s and rad/s
client.set_target_velocity(np.array([vx, vy, omega]), frame="local")
```

::: warning Velocity timeout
The base stops automatically if no command arrives within **0.25 seconds**. The client maintains a heartbeat thread (20 ms) while connected.
:::

## Next steps

- Full API reference, frame conventions, linear rail control → [Flow Base product page](/products/flow-base)
- Linear Bot (Flow Base + arm + rail) demo → [Linear Bot demo](/getting-started/demos/linear-bot)
