# Flow Base

<div class="product-badges">
  <span class="product-badge available">✓ Python SDK</span>
  <span class="product-badge available">✓ Omnidirectional</span>
  <span class="product-badge available">✓ Remote Control</span>
  <span class="product-badge available">✓ Raspberry Pi On-board</span>
</div>

**Flow Base** is I2RT's omnidirectional holonomic mobile platform. Designed to pair with YAM arms, it enables precise whole-body mobile manipulation for tasks that demand exact positioning and free orientation.

<MediaPlaceholder
  type="photo"
  description="Flow Base standalone — overhead 3/4 view showing the four mecanum/omni wheels, the top mounting plate, Raspberry Pi display, and E-stop button."
/>

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

<MediaPlaceholder
  type="photo"
  description="Flow Base control panel close-up: E-stop, CAN selector switch, Raspberry Pi display showing boot status."
/>

<MediaPlaceholder
  type="video"
  description="Flow Base driving around a lab environment — forward, lateral, diagonal, and rotation movements. 30–60 seconds, bird's-eye and side views."
/>

<MediaPlaceholder
  type="video"
  description="YAM arm mounted on Flow Base, doing a mobile pick-and-place from a shelf. Showcases the full mobile manipulation capability."
/>

## Remote Control Layout

| Input | Function |
|-------|----------|
| Left joystick | XY translation |
| Right joystick X | Rotation (yaw) |
| Right joystick Y | Linear rail lift (if equipped) |
| Left1 | Reset odometry |
| Mode | Toggle local ↔ global frame |
| Left2 | Override API commands (safety) |

## Getting Started

### 1. Unbox & power on

Follow the [unboxing guide](https://www.canva.com/design/DAGvHpqzf-Y). Ensure:
- Battery is connected and charged
- E-stop is **not pressed**
- CAN selector switch is in the **UP** position

### 2. SSH in

```bash
# Wired (Ethernet, static IP)
ssh i2rt@172.6.2.20

# Or find the Pi on your LAN after Wi-Fi setup
ssh i2rt@<pi-ip>
# Password: root
```

### 3. Run joystick demo

```bash
python i2rt/flow_base/flow_base_controller.py
```

### 4. Python API

```python
from i2rt.flow_base.flow_base_controller import Vehicle
import time

vehicle = Vehicle()
vehicle.start_control()

# Move forward at 0.1 m/s for 1 second
start = time.time()
while time.time() - start < 1.0:
    vehicle.set_target_velocity((0.1, 0.0, 0.0), frame="local")
```

## Coordinate Systems

The base supports two control frames toggled with the **Mode** button on the remote:

| Mode | Behaviour |
|------|-----------|
| **Local** | XY motion is relative to the base's current heading |
| **Global** | XY motion is relative to the world frame (headless mode) |

::: warning Odometry drift
Wheel odometry accumulates error, especially during aggressive movements. For precise mobile manipulation, integrate a visual odometry sensor (RealSense T265, ZED Camera). Press **Left1** to reset odometry at any time.
:::

## API Reference

```python
from i2rt.flow_base.flow_base_client import FlowBaseClient

client = FlowBaseClient(host="172.6.2.20")

# Read wheel odometry
odom = client.get_odometry()
# {'translation': array([x, y]), 'rotation': array([theta])}

# Reset odometry
client.reset_odometry()

# Command velocity [x, y, theta] in m/s and rad/s
client.set_target_velocity([0.1, 0.0, 0.0], frame="local")
```

### Linear Rail (if equipped)

```python
client = FlowBaseClient(host="172.6.2.20", with_linear_rail=True)

# Get rail state (position, velocity, limit switches)
state = client.get_linear_rail_state()

# Set rail velocity (rad/s)
client.set_linear_rail_velocity(0.5)

# Combined base + rail command [x, y, theta, rail_vel]
client.set_target_velocity([0.1, 0.0, 0.0, 0.2], frame="local")
```

::: tip Linear rail safety
The rail homes to the lower limit on init and has hardware limit switches. Set velocity to `0.0` to stop — do not control the brake directly. Commands time out after **0.25 s** of inactivity.
:::

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

- [Linear Bot](/products/linear-bot) — Flow Base + linear rail lift
- [Flow Base API](/sdk/flow-base)
- [Hardware Setup](/getting-started/hardware-setup)

<style scoped>
.product-badges { display: flex; flex-wrap: wrap; gap: 8px; margin: 16px 0 24px; }
.product-badge { display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; border: 1px solid; }
.product-badge.available { color: #4CCFB0; border-color: rgba(76,207,176,0.4); background: rgba(76,207,176,0.08); }
</style>
