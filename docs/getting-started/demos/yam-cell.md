# YAM Cell Demo — Bimanual Teleoperation

Drive 2 follower arms with 2 leader arms — full bimanual leader-follower teleop with bilateral force feedback.

::: tip Prerequisite
[YAM Cell Hardware Setup](/getting-started/hardware/yam-cell) done — 4 arms with persistent CAN names (`can_follower_l`, `can_follower_r`, `can_leader_l`, `can_leader_r`).
:::

## Launch — one terminal per arm

**Terminal 1 — left follower:**
```bash
python examples/minimum_gello/minimum_gello.py \
  --gripper linear_4310 --mode follower \
  --can-channel can_follower_l --bilateral-kp 0.2
```

**Terminal 2 — right follower:**
```bash
python examples/minimum_gello/minimum_gello.py \
  --gripper linear_4310 --mode follower \
  --can-channel can_follower_r --bilateral-kp 0.2
```

**Terminal 3 — left leader:**
```bash
python examples/minimum_gello/minimum_gello.py \
  --gripper yam_teaching_handle --mode leader \
  --can-channel can_leader_l --bilateral-kp 0.2
```

**Terminal 4 — right leader:**
```bash
python examples/minimum_gello/minimum_gello.py \
  --gripper yam_teaching_handle --mode leader \
  --can-channel can_leader_r --bilateral-kp 0.2
```

## Engage

Press the **top button on each teaching handle** to enable leader-follower sync. The followers track the leaders in real time.

## What `--bilateral-kp` does

| Value | Behavior |
|---|---|
| `0.0` | Open-loop — leader feels nothing |
| `0.1` | Soft force feedback |
| `0.2` | Recommended — leader feels follower load |
| `0.3+` | Stiff — risk of oscillation |

## Next steps

- Full configuration, advanced teleop options, recording trajectories → [YAM Cell product page](/products/yam-cell)
