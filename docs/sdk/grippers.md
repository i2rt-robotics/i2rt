# Grippers

YAM supports four interchangeable end effectors. The gripper type is specified when creating the robot:

```python
robot = get_yam_robot(channel="can0", gripper_type="linear_4310")
```

## Gripper Reference

### `crank_4310`

Zero-linkage crank gripper powered by the DM4310 motor. The crank mechanism minimizes the total gripper width — ideal for reaching into tight spaces.

<MediaPlaceholder
  type="photo"
  description="crank_4310 gripper close-up: crank mechanism, finger geometry, and motor mount. Side and front views."
/>

| Property | Value |
|----------|-------|
| Motor | DM4310 |
| Mechanism | Zero-linkage crank |
| Calibration | Not required |
| Best for | Narrow workspace, minimizing sweep width |

---

### `linear_3507`

Lightweight linear gripper with a DM3507 motor. Smaller and lighter than the 4310 variant.

<MediaPlaceholder
  type="photo"
  description="linear_3507 gripper showing the linear actuator, finger tips, and compact motor housing."
/>

| Property | Value |
|----------|-------|
| Motor | DM3507 |
| Mechanism | Linear actuator |
| Calibration | **Required** — must start in fully closed position, or run calibration routine |
| Best for | Weight-sensitive setups |

::: warning Calibration required
Because the motor travels more than 2π radians over the full stroke, the `linear_3507` needs to know its start position. Either:
1. Close the gripper fully by hand before powering on, **or**
2. Run the provided calibration routine
:::

---

### `linear_4310`

Standard linear gripper with the heavier DM4310 motor. Slightly more gripping force than the 3507.

<MediaPlaceholder
  type="photo"
  description="linear_4310 gripper. Similar form factor to linear_3507 but with a larger motor body."
/>

| Property | Value |
|----------|-------|
| Motor | DM4310 |
| Mechanism | Linear actuator |
| Calibration | **Required** — same as `linear_3507` |
| Best for | General-purpose tasks, higher force |

---

### `yam_teaching_handle`

The leader arm handle — not a manipulation gripper, but a hand controller for teleoperation.

<MediaPlaceholder
  type="photo"
  description="yam_teaching_handle: the handle with trigger clearly visible, two programmable buttons on top, and cable exit."
/>

| Feature | Description |
|---------|-------------|
| Trigger | Controls follower gripper open/close |
| Top button | Enable/disable arm synchronization |
| Second button | User-programmable |

For full usage — trigger reading, encoder calibration, and teleoperation setup — see the [YAM Leader Arm](/products/yam-leader) page.

---

## Gripper Models

MuJoCo and URDF models for each gripper:

```
i2rt/robot_models/gripper/
├── crank_4310/
│   ├── crank_4310.xml
│   └── assets/link_6_collision.stl
├── linear_3507/
│   ├── linear_3507.xml
│   └── assets/  gripper.stl  tip_left.stl  tip_right.stl
├── linear_4310/
│   ├── linear_4310.xml
│   └── assets/  gripper.stl  tip_left.stl  tip_right.stl
├── yam_teaching_handle/
│   └── yam_teaching_handle.xml
└── no_gripper/
    └── no_gripper.xml
```

## Gripper Calibration (Linear)

For `linear_3507` and `linear_4310`:

1. Manually close the gripper fully (fingers touching)
2. Power on the arm
3. Initialize normally:

```python
robot = get_yam_robot(channel="can0", gripper_type="linear_4310")
```

Or, if you cannot close it manually before power-on, use the calibration script (see `i2rt/motor_config_tool/`).
