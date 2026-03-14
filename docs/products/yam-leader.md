# YAM Leader Arm

<div class="product-badges">
  <span class="product-badge available">✓ Teleoperation</span>
  <span class="product-badge available">✓ Teaching Handle</span>
  <span class="product-badge available">✓ Bimanual Ready</span>
</div>

The **YAM Leader** is the operator-side arm used for teleoperation. It is kinematically identical to the [YAM Arm](/products/yam-arm) but fitted with the `yam_teaching_handle` instead of a gripper. The operator holds the handle and moves the leader arm — the follower arm mirrors the motion in real time.

<MediaPlaceholder
  type="photo"
  description="YAM Leader arm with teaching handle mounted — trigger and two top buttons clearly visible. Side-by-side with a YAM follower arm to show the matched pair."
/>

## Specifications

| Parameter | Value |
|-----------|-------|
| Base arm | YAM (6-DOF, CAN bus) |
| End effector | `yam_teaching_handle` |
| Price | $2,999 |
| Typical use | Bimanual teleoperation (YAM Cell) |

## Teaching Handle

<MediaPlaceholder
  type="photo"
  description="Close-up of yam_teaching_handle: trigger, two programmable top buttons, and cable exit."
/>

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

For full bimanual setup see [Bimanual Teleoperation](/examples/bimanual-teleoperation).

## Videos

<MediaPlaceholder
  type="video"
  description="Operator using two YAM Leader arms to control two YAM follower arms — picking and placing objects. Shows trigger use for gripper control."
/>

## See Also

- [YAM Arm](/products/yam-arm) — the follower arm
- [YAM Cell](/products/yam-cell) — full bimanual teleoperation station
- [Bimanual Teleoperation](/examples/bimanual-teleoperation)
- [Grippers](/sdk/grippers)

<style scoped>
.product-badges { display: flex; flex-wrap: wrap; gap: 8px; margin: 16px 0 24px; }
.product-badge { display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; border: 1px solid; }
.product-badge.available { color: #4CCFB0; border-color: rgba(76,207,176,0.4); background: rgba(76,207,176,0.08); }
</style>
