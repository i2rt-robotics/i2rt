<script setup>
import { withBase } from 'vitepress'
</script>

# Linear Bot

<div class="product-badges">
  <span class="product-badge available">✓ Python SDK</span>
  <span class="product-badge available">✓ Full Mobile Manipulation</span>
  <span class="product-badge available">✓ Height Adjustment</span>
</div>

**Linear Bot** is the [Flow Base](/products/flow-base) combined with a vertical linear rail actuator. The linear rail adds a height axis to the omnidirectional base, enabling the mounted YAM arm to reach objects at varying heights — from floor-level to shelf-height — without repositioning.

<div class="product-gallery hero-single">
  <figure>
    <img :src="withBase('/images/linearbot/lb-1.webp')" alt="Linear Bot full system" />
  </figure>
</div>


## System Architecture

```
┌─────────────────────────┐
│   YAM Arm               │  ← 6-DOF manipulation
├─────────────────────────┤
│   Linear Rail           │  ← Vertical axis (Z height)
├─────────────────────────┤
│   Flow Base             │  ← XY + rotation (holonomic)
└─────────────────────────┘
```

The three subsystems are controlled together through a unified Python API, giving the robot **9 degrees of freedom** (6-DOF arm + 3-DOF mobile base including linear rail).

## Key Features

- **Vertical height axis** — linear rail extends and retracts under API or remote control
- **Limit switches** — hardware safety stops at both rail ends; auto-home on initialization
- **Brake management** — brake automatically released on init, engaged on shutdown
- **Integrated API** — 4D base velocity commands `[x, y, θ, rail_vel]`
- **Safety timeouts** — both base and rail stop after 0.25 s without a heartbeat

## Specifications

| Parameter | Value |
|-----------|-------|
| Mobile base | Flow Base (holonomic) |
| Vertical actuator | Linear rail with DM-series motor |
| Arm (typical) | YAM or YAM Pro |
| Total DOF (arm + base + rail) | 9 |
| Rail control | Velocity [rad/s] |
| Rail velocity timeout | 0.25 s |
| Rail home | Lower limit (on init) |

## Photos & Videos

<div class="product-gallery">
  <figure>
    <img :src="withBase('/images/linearbot/lb-2.webp')" alt="Linear Bot — view 2" />
  </figure>
  <figure>
    <img :src="withBase('/images/linearbot/lb-3.webp')" alt="Linear Bot — view 3" />
  </figure>
  <figure>
    <img :src="withBase('/images/linearbot/lb-arm-ultra.webp')" alt="Linear Bot with YAM Ultra arm" />
  </figure>
  <figure>
    <img :src="withBase('/images/linearbot/lb-arm.webp')" alt="Linear Bot arm close-up" />
  </figure>
  <figure>
    <img :src="withBase('/images/linearbot/lb-base-panel.webp')" alt="Linear Bot base panel" />
  </figure>
  <figure>
    <img :src="withBase('/images/linearbot/lb-cable-rail.webp')" alt="Linear Bot cable rail detail" />
  </figure>
</div>

<MediaPlaceholder
  type="video"
  description="Linear Bot navigating to a shelf, extending the rail to the correct height, and using the YAM arm to retrieve an object. Full task, 1–2 minutes."
/>

<MediaPlaceholder
  type="video"
  description="Time-lapse or sped-up footage of Linear Bot performing repeated fetch tasks in a simulated warehouse environment."
/>

## Python API

```python
from i2rt.flow_base.flow_base_client import FlowBaseClient

client = FlowBaseClient(host="172.6.2.20", with_linear_rail=True)

# Move forward + raise rail simultaneously
client.set_target_velocity([0.1, 0.0, 0.0, 0.05], frame="local")
#                           x    y    θ    rail_vel

# Get rail position and limit switch states
state = client.get_linear_rail_state()
print(state)  # {'position': ..., 'velocity': ..., 'limit_switches': ...}

# Stop rail
client.set_linear_rail_velocity(0.0)
```

## Important Notes

::: warning Auto-homing on init
The linear rail homes to the **lower limit switch** on every initialization. Ensure there is clearance below the carriage before powering on.
:::

::: tip Stopping the rail
Always set velocity to `0.0` to stop the rail. Do not try to engage the brake directly — the system manages brake state automatically.
:::

## Pricing

Starting at **$18,999**. Contact [sales@i2rt.com](mailto:sales@i2rt.com) for configuration options.

## See Also

- [Flow Base](/products/flow-base) — base-only configuration
- [Flow Base API](/sdk/flow-base)
- [YAM Arm Series](/products/yam)

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
