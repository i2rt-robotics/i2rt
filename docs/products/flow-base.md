<script setup>
import { withBase } from 'vitepress'
</script>

# Flow Base

<div class="product-badges">
  <span class="product-badge available">✓ Python SDK</span>
  <span class="product-badge available">✓ Omnidirectional</span>
  <span class="product-badge available">✓ Remote Control</span>
  <span class="product-badge available">✓ Raspberry Pi On-board</span>
</div>

**Flow Base** is I2RT's omnidirectional holonomic mobile platform. Designed to pair with YAM arms, it enables precise whole-body mobile manipulation for tasks that demand exact positioning and free orientation.

<div class="product-gallery hero-single">
  <figure>
    <img :src="withBase('/images/flow_base/fb-hero.webp')" alt="Flow Base MB-4310-ST" />
  </figure>
</div>

## Tagline

> *"It likes to move it move it"* — precise, omnidirectional control for tasks where positioning and stability are critical.

## Key Features

- **Holonomic drive** — simultaneous XY translation and rotation with no kinematic constraints
- **On-board Raspberry Pi** — pre-configured with i2rt SDK; SSH-accessible over Wi-Fi or Ethernet
- **CAN bus motor control** — same DM-series protocol as YAM arms
- **Remote controller** — joystick remote included for manual operation
- **API control** — full network Python API via `FlowBaseClient`
- **Odometry** — wheel odometry with reset; external sensor integration supported
- **Safety** — E-stop, velocity timeout (0.25 s), remote override

## Specifications

| Parameter | Value |
|-----------|-------|
| Drive | Holonomic (4-wheel) |
| Communication (external) | Ethernet (static IP `172.6.2.20`) / Wi-Fi |
| Communication (motors) | CAN bus |
| On-board computer | Raspberry Pi |
| Power | Internal battery |
| SSH credentials | `i2rt` / `root` |
| API port | `11323` |
| Velocity timeout | 0.25 s |

## Photos & Videos

<div class="product-gallery">
  <figure>
    <img :src="withBase('/images/flow_base/fb-1.webp')" alt="Flow Base MB-4310-ST — view 1" />
  </figure>
  <figure>
    <img :src="withBase('/images/flow_base/fb-3.webp')" alt="Flow Base MB-4310-ST — view 3" />
  </figure>
  <figure>
    <img :src="withBase('/images/flow_base/fb-4.webp')" alt="Flow Base MB-4310-ST — view 4" />
  </figure>
  <figure>
    <img :src="withBase('/images/flow_base/fb-5.webp')" alt="Flow Base MB-4310-ST — view 5" />
  </figure>
  <figure>
    <img :src="withBase('/images/flow_base/fb-6.webp')" alt="Flow Base MB-4310-ST — view 6" />
  </figure>
  <figure>
    <img :src="withBase('/images/flow_base/fb-7.webp')" alt="Flow Base MB-4310-ST — view 7" />
  </figure>
  <figure>
    <img :src="withBase('/images/flow_base/fb-controller.webp')" alt="Flow Base remote controller" />
  </figure>
</div>

<MediaPlaceholder
  type="video"
  description="Flow Base driving around a lab environment — forward, lateral, diagonal, and rotation movements. 30–60 seconds, bird's-eye and side views."
/>

<MediaPlaceholder
  type="video"
  description="YAM arm mounted on Flow Base, doing a mobile pick-and-place from a shelf. Showcases the full mobile manipulation capability."
/>

## Getting Started

1. Follow [Flow Base hardware setup](/getting-started/hardware/flow-base)
2. Run the [Flow Base demo](/getting-started/demos/flow-base)
3. See the [API Reference](#api-reference) below for full SDK details

## Remote Control Layout

| Input | Function |
|-------|----------|
| Left joystick | XY translation |
| Right joystick X | Rotation (yaw) |
| Right joystick Y | Linear rail lift (if equipped) |
| Left1 | Reset odometry |
| Mode | Toggle local ↔ global frame |
| Left2 | Override API commands (safety) |

## Coordinate Systems

The base supports two control frames toggled with the **Mode** button on the remote:

| Mode | Behaviour |
|------|-----------|
| **Local** | XY motion is relative to the base's current heading |
| **Global** | XY motion is relative to the world frame (headless mode) |

::: warning Odometry drift
Wheel odometry accumulates error, especially during aggressive movements. For precise mobile manipulation, integrate a visual odometry sensor (RealSense T265, ZED Camera). Press **Left1** to reset odometry at any time.
:::

---

## API Reference

The Flow Base SDK has two layers:

| Class | Location | Use |
|-------|----------|-----|
| `Vehicle` | `flow_base_controller.py` | Runs **on-board** the Pi — joystick demo |
| `FlowBaseClient` | `flow_base_client.py` | Runs **remotely** — network Python API |

### `FlowBaseClient`

For remote control from your development machine.

```python
from i2rt.flow_base.flow_base_client import FlowBaseClient

client = FlowBaseClient(
    host="172.6.2.20",
    with_linear_rail=False,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `host` | `str` | `"localhost"` | IP address of the Flow Base Pi |
| `with_linear_rail` | `bool` | `False` | Set `True` if the linear rail module is installed |

::: tip No port parameter
The port is hardcoded internally using `BASE_DEFAULT_PORT`. You only need to specify the host IP.
:::

### Movement Commands

#### `set_target_velocity(velocity, frame)`

```python
import numpy as np

# 3D (base only)
client.set_target_velocity(np.array([vx, vy, omega]), frame="local")

# 4D (base + linear rail)
client.set_target_velocity(np.array([vx, vy, omega, rail_vel]), frame="local")
```

| Parameter | Unit | Description |
|-----------|------|-------------|
| `vx` | m/s | Forward/backward |
| `vy` | m/s | Left/right (strafe) |
| `omega` | rad/s | Rotation (yaw rate) |
| `rail_vel` | rad/s | Linear rail speed (positive = up) |
| `frame` | — | `"local"` (relative to base) or `"global"` (world frame) |

::: warning Velocity must be a NumPy array
`set_target_velocity()` expects a `np.ndarray` with shape `(3,)` or `(4,)`. Pass `np.array([...])`, not a plain Python list.
:::

::: warning Velocity timeout
The base stops automatically if no command arrives within **0.25 seconds**. `FlowBaseClient` maintains a heartbeat automatically while connected via a background thread (20 ms interval).
:::

### Odometry

#### `get_odometry() → dict`

```python
odom = client.get_odometry()
# {'translation': array([x, y]), 'rotation': array(theta)}
```

Wheel odometry only. Errors accumulate over time — integrate visual odometry (RealSense T265, ZED) for precise localization.

#### `reset_odometry() → None`

Resets position and heading to zero.

### Linear Rail API

Only available when `with_linear_rail=True`.

#### `get_linear_rail_state() → dict`

```python
state = client.get_linear_rail_state()
# {
#   'position': float,
#   'velocity': float,
#   'upper_limit_triggered': bool,
#   'lower_limit_triggered': bool,
# }
```

#### `set_linear_rail_velocity(velocity: float) → None`

```python
client.set_linear_rail_velocity(0.5)    # raise
client.set_linear_rail_velocity(0.0)    # stop
client.set_linear_rail_velocity(-0.5)   # lower
```

#### Combined base + rail command

```python
client.set_target_velocity(np.array([vx, vy, omega, rail_vel]), frame="local")
```

::: tip Auto-homing
The rail homes to the **lower limit switch** on init. Ensure clearance below before powering on.
:::

### Cleanup

Always close the client when done to stop the background heartbeat thread:

```python
client.close()
```

### Command-line Client

Quick functional tests without writing Python:

```bash
# Read odometry
python i2rt/flow_base/flow_base_client.py --command get_odometry --host 172.6.2.20

# Reset odometry
python i2rt/flow_base/flow_base_client.py --command reset_odometry --host 172.6.2.20

# Run a short movement test (base will move)
python i2rt/flow_base/flow_base_client.py --command test_command --host 172.6.2.20

# Test linear rail (rail will move)
python i2rt/flow_base/flow_base_client.py --command test_linear_rail --host 172.6.2.20

# Monitor linear rail state
python i2rt/flow_base/flow_base_client.py --command get_linear_rail_state --host 172.6.2.20
```

### `Vehicle` (On-board Controller)

Runs directly on the Pi. Used for the joystick demo.

```python
from i2rt.flow_base.flow_base_controller import Vehicle
import time

v = Vehicle()
v.start_control()

start = time.time()
while time.time() - start < 2.0:
    v.set_target_velocity((0.15, 0.0, 0.0), frame="local")
```

### Coordinate Frames

| Frame | Description |
|-------|-------------|
| `local` | Relative to the current base orientation. Joystick forward = robot forward regardless of heading. |
| `global` | World frame from odometry zero. Similar to drone headless mode. Accumulates error. |

Switch frames at runtime via the remote **Mode** button, or programmatically by passing `frame=` to `set_target_velocity`.

---

## External CAN Control

To bypass the on-board Pi and control the base from an external computer:

1. Connect your CAN adapter to the external CAN connector
2. Set the CAN selector switch to the **DOWN** position
3. Clone the i2rt repo on your external machine and control via CAN directly

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Remote unresponsive | Toggle remote off and on to wake from sleep |
| Slow boot | Normal — screen firmware adds delay, SSH is available quickly |
| Inaccurate odometry | Expected with wheel odometry; use external visual sensor for precision |
| Linear rail not homing | Check GPIO connections and limit switches |
| Linear rail stuck at limit | Run `get_linear_rail_state()` to check switch status |

## See Also

- [Linear Bot](/products/linear-bot) — Flow Base + linear rail lift + YAM arm
- [Flow Base hardware setup](/getting-started/hardware/flow-base)
- [Flow Base demo](/getting-started/demos/flow-base)

<style scoped>
.product-badges { display: flex; flex-wrap: wrap; gap: 8px; margin: 16px 0 24px; }
.product-badge { display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; border: 1px solid; }
.product-badge.available { color: #4C6762; border-color: rgba(76,103,98,0.4); background: rgba(76,103,98,0.08); }
.product-gallery { display: flex; flex-wrap: wrap; gap: 16px; margin: 16px 0 8px; }
.product-gallery figure { flex: 1 1 220px; margin: 0; }
.product-gallery img { width: 100%; border-radius: 8px; }
.product-gallery figcaption { font-size: 0.8rem; color: var(--vp-c-text-2); text-align: center; margin-top: 6px; }
.product-gallery.hero-single figure { flex: 1 1 100%; }
</style>
