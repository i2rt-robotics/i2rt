<script setup>
import { withBase } from 'vitepress'
</script>

# YAM Leader Arm

<div class="product-badges">
  <span class="product-badge available">✓ Teleoperation</span>
  <span class="product-badge available">✓ Teaching Handle</span>
  <span class="product-badge available">✓ Bimanual Ready</span>
</div>

The **YAM Leader** is the operator-side arm used for teleoperation. It is kinematically identical to the [YAM Arm](/products/yam-arm) but fitted with the `yam_teaching_handle` instead of a gripper. The operator holds the handle and moves the leader arm — the follower arm mirrors the motion in real time.

<div class="product-gallery">
  <figure>
    <img :src="withBase('/images/yam-leader/YAM-ST-Leader-2.webp')"  alt="YAM Leader Arm front view" />
    
  </figure>
  <figure>
    <img :src="withBase('/images/yam-leader/YAM-ST-Leader-1.webp')"  alt="YAM Leader Arm side view" />
    
  </figure>
  <figure>
    <img :src="withBase('/images/yam-leader/YAM-ST-Leader-3.webp')"  alt="YAM Leader Arm 3/4 view" />
    
  </figure>
</div>


## Specifications

| Parameter | Value |
|-----------|-------|
| Base arm | YAM (6-DOF, CAN bus) |
| End effector | `yam_teaching_handle` |
| Price | $2,999 |
| Typical use | Bimanual teleoperation (YAM Cell) |

## Teaching Handle

<div class="product-gallery">
  <figure>
    <img :src="withBase('/images/yam-leader/YAM-ST-Leader-4.webp')"  alt="YAM teaching handle close-up" />
    
  </figure>
</div>

| Control | Function |
|---------|----------|
| **Trigger** | Open / close the follower gripper |
| **Top button** | Enable / disable arm synchronisation |
| **Second button** | User-programmable |

### Reading the trigger

```bash
python scripts/read_encoder.py --channel $CAN_CHANNEL
```

Example output:

```
[PassiveEncoderInfo(id=1294, position=0.004382, velocity=0.0, io_inputs=[0, 0])]
```

- `position` → follower gripper command (`0.0` = open, `~1.0` = closed)
- `io_inputs` → button states

### Resetting encoder zero

If the magnet has shifted or after repairs:

```bash
python devices/config_passive_encoder.py --bus $CAN_CHANNEL reset-zero-position
```

Re-run `read_encoder.py` to confirm the trigger-released reading is near `0.0`.

## Teleoperation Setup

The leader arm is launched via `minimum_gello.py` in leader mode:

```bash
python examples/minimum_gello/minimum_gello.py \
  --gripper yam_teaching_handle \
  --mode leader \
  --can-channel can_leader_l
```

For full bimanual setup see [Bimanual Teleoperation](/products/yam-cell#bimanual-teleoperation).

## Videos

<video controls style="width:100%;border-radius:8px;margin:16px 0 8px">
  <source :src="withBase('/images/yam-leader/YAM-ST-Leader-video.mp4')" type="video/mp4" />
</video>


## See Also

- [YAM Arm](/products/yam-arm) — the follower arm
- [YAM Cell](/products/yam-cell) — full bimanual teleoperation station
- [Bimanual Teleoperation](/products/yam-cell#bimanual-teleoperation)
- [Grippers](/products/yam#grippers)

<style scoped>
.product-badges { display: flex; flex-wrap: wrap; gap: 8px; margin: 16px 0 24px; }
.product-badge { display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; border: 1px solid; }
.product-badge.available { color: #4C6762; border-color: rgba(76,103,98,0.4); background: rgba(76,103,98,0.08); }
.product-gallery { display: flex; flex-wrap: wrap; gap: 16px; margin: 16px 0 8px; }
.product-gallery figure { flex: 1 1 220px; margin: 0; }
.product-gallery img { width: 100%; border-radius: 8px; }
.product-gallery figcaption { font-size: 0.8rem; color: var(--vp-c-text-2); text-align: center; margin-top: 6px; }
</style>
