# Grippers

YAM supports six interchangeable end-effector options. The gripper type is specified when creating the robot:

```python
from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import GripperType

robot = get_yam_robot(channel="can0", gripper_type=GripperType.LINEAR_4310)
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
| Calibration | Not required — fixed limits (0.0 to -2.7 rad) |
| PD Gains | kp=20, kd=0.5 |
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
| Calibration | **Required** — auto-detected on startup |
| PD Gains | kp=10, kd=0.3 |
| Best for | Weight-sensitive setups |

::: warning Calibration required
Because the motor travels more than 2pi radians over the full stroke, the `linear_3507` needs to know its start position. On startup, the SDK automatically runs a calibration routine that moves the gripper in both directions to detect limits. Ensure the gripper can move freely during initialization.
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
| Calibration | **Required** — same auto-calibration as `linear_3507` |
| PD Gains | kp=20, kd=0.5 |
| Best for | General-purpose tasks, higher force |

---

### `flexible_4310`

Linear gripper with flexible soft fingertips. Identical drivetrain to `linear_4310` (DM4310 motor, same stroke), but the compliant tips conform to the workpiece — useful for grasping fragile or irregular objects without precise pose alignment.

<MediaPlaceholder
  type="photo"
  description="flexible_4310 gripper showing the soft compliant fingertips deforming around an irregular object."
/>

| Property | Value |
|----------|-------|
| Motor | DM4310 |
| Mechanism | Linear actuator with flexible tips |
| Calibration | **Required** — same auto-calibration as `linear_4310` |
| PD Gains | kp=20, kd=0.5 |
| Best for | Fragile or irregular objects, tolerant grasping |

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

### `no_gripper`

Arm-only configuration with no end effector. Use when the application doesn't require grasping.

```python
robot = get_yam_robot(gripper_type=GripperType.NO_GRIPPER)
# robot.num_dofs() returns 6 (arm joints only)
```

---

## Gripper Models

MuJoCo XML models for each gripper. Arm and gripper models are combined at runtime:

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
├── flexible_4310/
│   ├── flexible_4310.xml
│   ├── flexible_4310.urdf
│   └── assets/
├── yam_teaching_handle/
│   └── yam_teaching_handle.xml
└── no_gripper/
    └── no_gripper.xml
```

## Gripper Force Limiting

The SDK includes automatic gripper force limiting when the gripper is clogged (stalled against an object). This is enabled by default with `limit_gripper_force=50.0` N. The system monitors motor effort and speed to detect when the gripper has hit an object, then limits the applied torque accordingly.

## See Also

- [YAM Arm API](/sdk/yam-arm) — `get_observations()` for reading gripper state
- [Bimanual Teleoperation](/examples/bimanual-teleoperation)
