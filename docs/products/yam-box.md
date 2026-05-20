<script setup>
import { withBase } from 'vitepress'
</script>

# YAM Box

<div class="product-badges">
  <span class="product-badge coming-soon">🔜 Code Coming Soon</span>
  <span class="product-badge available">✓ Hardware Available</span>
</div>

**YAM Box** is an all-in-one enclosed manipulation station built around the YAM arm family. It provides a self-contained, cable-managed workspace for manipulation research and automated data collection — with two follower arms, integrated cameras, and an on-board Mini PC running Ubuntu.

::: warning Code support not yet available
The YAM Box hardware is available for purchase. Software and SDK support is under active development and will be released in an upcoming update. Sign up at [i2rt.com](https://i2rt.com) or join the [Discord](https://discord.gg/i2rt) to be notified.
:::

## Overview

<div class="dim-gallery">
  <figure>
    <img :src="withBase('/images/yam-box/step-9.webp')" alt="YAM Box assembled with two arms and center camera" />
  </figure>
</div>

## Intended Use Cases

- **Automated data generation** — Run unattended overnight collection campaigns without cable snagging.
- **Demonstration stations** — Clean, professional setup for trade shows and lab demos.
- **Safe testing environments** — Enclosed workspace limits accidental contact during development.

## What's Inside

| Component | Details |
|-----------|---------|
| YAM follower arms | 2× YAM arms mounted on left and right sides |
| Camera — top center | 1× camera on center vertical post, connects to USB Hub |
| Cameras — arm-mounted | 2× wrist cameras routed through side holes to Mini PC |
| On-board Mini PC | Pre-installed Ubuntu, connected via USB to cameras and CAN adapters |
| USB Hub | Mounted inside front panel; center camera plugs into leftmost port |
| Power distribution | XT30 (2+2) cables supply each arm from the YAM Box power rail |
| Power input | Dual XT30U-F adapters combined into parallel connector |

---

## Pre-installed Software

The YAM Box ships with **Ubuntu pre-installed** on the on-board Mini PC.

| Item | Value |
|------|-------|
| OS | Ubuntu (latest pre-installed image) |
| Default password | `root` (newer units) or `123` (early units) |

::: tip First boot
Connect a monitor and keyboard to the Mini PC's HDMI/USB ports on the rear panel to log in for the first time. Change the default password after initial setup.
:::

---

## Assembly Guide

> **Document:** Standard Operating Instructions for YAM Box Assembly V1.0 (2026/4/23)
>
> For the Bill of Materials (BOM), refer to the printed sheet included in the carton.

### Step 1 — Middle top camera

Open the front cover. Secure the camera mount to the **center mount base** using thumb screws. Plug the camera cable into the **leftmost port on the USB Hub**. Fasten the top camera with **2× M3×6 screws**.

<div class="dim-gallery">
  <figure>
    <img :src="withBase('/images/yam-box/setp-1.webp')" alt="Step 1: Mount center top camera" />
  </figure>
</div>

### Step 2 — Side arm cameras (cable routing)

Route the two arm camera cables through the **holes on each side** of the enclosure. Plug their USB ends into the USB ports on the **right side of the Mini PC**.

<div class="dim-gallery">
  <figure>
    <img :src="withBase('/images/yam-box/step-2-1.webp')" alt="Step 2: Route side camera cables to Mini PC" />
  </figure>
  <figure>
    <img :src="withBase('/images/yam-box/step-2-2.webp')" alt="Step 2: Camera cable USB connection to Mini PC" />
  </figure>
</div>

### Step 3 — Close front cover

Close and secure the front cover.

<div class="dim-gallery">
  <figure>
    <img :src="withBase('/images/yam-box/step-3-1.webp')" alt="Step 3: Close front cover" />
  </figure>
  <figure>
    <img :src="withBase('/images/yam-box/step-3-2.webp')" alt="Step 3: Front cover secured" />
  </figure>
</div>

### Step 4 — Mount follower arms

Place **4× T-nuts** on each side into the slots of the front two rows. Mount both follower arms to the left and right sides using **M5×16 screws** (4 screws per arm).

<div class="dim-gallery">
  <figure>
    <img :src="withBase('/images/yam-box/step-4.webp')" alt="Step 4: Mount follower arms with T-nuts and M5×16 screws" />
  </figure>
</div>

### Step 5 — Power cables to arms

Plug the **XT30 (2+2) cables** on both sides of the YAM Box into the corresponding power ports on each robotic arm base.

<div class="dim-gallery">
  <figure>
    <img :src="withBase('/images/yam-box/step-5-1.webp')" alt="Step 5: Connect XT30 power cables to arm bases" />
  </figure>
  <figure>
    <img :src="withBase('/images/yam-box/step-5-2.webp')" alt="Step 5: XT30 cable connected to arm base" />
  </figure>
</div>

### Step 6 — Arm camera brackets

For each arm-mounted camera:
- Secure the camera **bracket to the arm** with **2× M3×8 screws**
- Secure the **camera cable to the bracket** with **2× M3×14 screws**
- Secure the **camera to the bracket** with **2× M3×8 screws**

<div class="dim-gallery">
  <figure>
    <img :src="withBase('/images/yam-box/step-6-1.webp')" alt="Step 6: Attach camera brackets to arms" />
  </figure>
  <figure>
    <img :src="withBase('/images/yam-box/step-6-2.webp')" alt="Step 6: Camera secured to bracket" />
  </figure>
</div>

### Step 7 — Main power connection

Connect the two **XT30U-F adapter plugs** to the power parallel connector first, then plug the combined adapter into the power input port of the YAM Box.

<div class="dim-gallery">
  <figure>
    <img :src="withBase('/images/yam-box/step-7.webp')" alt="Step 7: Connect main power via XT30U-F parallel adapter" />
  </figure>
</div>

### Step 8 — Power on

Plug in the power supply and switch on. The Mini PC will boot automatically — the YAM Box is ready to use.

<div class="dim-gallery">
  <figure>
    <img :src="withBase('/images/yam-box/step-8-1.webp')" alt="Step 8: Power on the YAM Box" />
  </figure>
  <figure>
    <img :src="withBase('/images/yam-box/step-8-2.webp')" alt="Step 8: YAM Box powered and ready" />
  </figure>
</div>

---

## General Assembly Notes

::: warning Before powering on
After all wiring is complete, verify every connection is firm and there are no loose, mis-connected, or shorted cables before turning on the power.
:::

| Rule | Details |
|------|---------|
| **Fastener torque** | Tighten to spec — neither loose (fall-out risk) nor over-torqued (thread strip) |
| **Cable routing** | Route neatly, even zip-tie spacing; no tangling, sharp bends, or pinching |
| **Missing parts** | Do not force assembly with wrong or damaged hardware — contact [support@i2rt.com](mailto:support@i2rt.com) |
| **Post-assembly** | Clear all packaging and leftover hardware from the workspace before operation |

---

## Quick Start Demo

After assembly, YAM Box runs the same SDK as a standalone YAM arm. Test with:

```bash
python i2rt/robots/get_robot.py --channel can0 --gripper linear_4310
```

The arm should float in zero-gravity mode inside the enclosure. Push it through reachable space to verify nothing collides with the walls or top frame.

Then try the Python API:

```python
from i2rt.robots.get_robot import get_yam_robot
import numpy as np

robot = get_yam_robot(channel="can0")
print(robot.get_observations())
robot.command_joint_pos(np.zeros(7))
robot.close()
```

For the full SDK reference, MuJoCo / Viser visualizers, and gravity-comp tuning, see the [YAM Arm product page](/products/yam#api-reference).

---

## Pricing

Starting at **$3,500**. Contact [sales@i2rt.com](mailto:sales@i2rt.com) for custom configurations.

## Stay Updated

- [Discord](https://discord.gg/i2rt)
- [GitHub](https://github.com/i2rt-robotics/i2rt) — watch the repo for SDK releases
- Email [support@i2rt.com](mailto:support@i2rt.com)

<style scoped>
.product-badges { display: flex; flex-wrap: wrap; gap: 8px; margin: 16px 0 24px; }
.product-badge { display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; border: 1px solid; }
.product-badge.available { color: #4C6762; border-color: rgba(76,103,98,0.4); background: rgba(76,103,98,0.08); }
.product-badge.coming-soon { color: #9ca3b8; border-color: rgba(156,163,184,0.3); background: rgba(156,163,184,0.06); }
.dim-gallery { display: flex; flex-direction: column; gap: 16px; margin: 16px 0 24px; }
.dim-gallery figure { margin: 0; }
.dim-gallery img { width: 100%; border-radius: 8px; background: #fff; padding: 4px; box-sizing: border-box; }
</style>
