# 7-DOF Arm (right) Physical Properties

This document records the physical and kinematic properties of the **right** seven-DOF arm from [`yam_7dof_right.urdf`](yam_7dof_right.urdf). The corresponding MuJoCo model is [`yam_7dof_right.xml`](yam_7dof_right.xml).

## Scope and conventions

- The arm consists of `base` and `link1` through `link6`, connected by `joint1` through `joint7`.
- The child of `joint7` is named `gripper`, and as on every YAM arm it is the **end-effector mount frame, not a physical link**. The MJCF gives it a `mass=1e-6` placeholder inertial purely so the arm-only model compiles; `combine_arm_and_gripper_xml` overwrites that inertial with the chosen gripper's at runtime. It is therefore absent from the mass and inertia tables below and present in the frame tables, where it is kinematically real.
- It does carry **geometry**: `assets/wrist.stl`, the arm-side wrist section between `joint7` and the gripper's mounting face. Composition keeps a mount body's geoms and replaces only its inertial, so this renders and collides while contributing no mass. Its **0.103291 kg is deliberately not modelled** — the as-built ONShape export fused it into the gripper link, and the split keeps the geometry without inventing a place to put the mass.
- The URDF, like `yam.urdf`, is the **whole robot**: its terminal `gripper` link is the canonical `linear_4310` and it carries two fingers on prismatic `joint8`/`joint9`, alongside the wrist visual. None of the gripper half is emitted into the MJCF, and none of it is tabulated here — the gripper is an interchangeable module with its own model.
- Because the arm has seven joints, its gripper's motor sits on CAN `0x08` rather than the YAM grippers' `0x07`, and composition renumbers the gripper's own finger joints to `joint8`/`joint9` to follow the arm's count.
- SI units are used: metres, kilograms, radians, and kg·m².
- Home pose means all seven arm joint coordinates are zero.
- URDF RPY uses `R = Rz(yaw) Ry(pitch) Rx(roll)`.
- Joint axes are expressed in their local joint frames.
- “Link length” is the Euclidean norm of the parent-to-child joint-origin translation. The full translation vector remains the authoritative kinematic value.
- The ONShape exporter writes angles to six significant figures, so exact multiples of π appear in the URDF as e.g. `1.5708` rather than `1.57079632679`; such values are shown below as the multiple they represent.
- Frame orientations (RPY) and joint ranges within floating-point tolerance of `0` or a multiple of π are written as `0` or that multiple (e.g. `π/2`); any near-zero numeric component with magnitude below `1e-4` is likewise shown as `0` (diagonal inertia moments always keep their value). All other values keep their full precision.

### Provenance

Exported from ONShape by `onshape-to-robot` (document `a30d2ef2655863e671c75240`). Unlike this arm's first export, this one arrives in the shape the pipeline expects: rooted at `base` with `joint1` turning the arm rather than the base, with `link6` already named positionally, with every mesh placed from one clean assembly origin, and with the `base` link frame already on the mounting face. The re-rooting, the `link8` → `link6` rename and the `base`/`link1` frame swap that the first import needed are all gone.

One heading rotation is applied, `R = Rz(-90°)·Ry(90°)` — pipeline stage 3, `apply_urdf_heading.py --rpy "0 1.5707963267948966 -1.5707963267948966"`, which this export passes (its base visual, collision and inertial origins share one orientation, the precondition that the first export failed). The rotation goes onto the base link's visual, collision and inertial origins and onto the `base → link1` joint origin together, so the base frame stays the identity world root while the body turns inside it.

The result is a **plate mount**: the base's mounting face lies in the world x–z plane with its outward normal along **+Y** (the face passes through the origin), `joint1` turns about **+Y**, and the arm hangs along **−Z** at the home pose. Note this is not the z-up convention the YAM arms use: the arm mounts to a vertical surface and reaches downwards rather than standing on a table.

Two things are then applied by hand, both recorded in the URDF's `i2rt-port` header comment. `joint1`, `joint3`, `joint5` and `joint6` have their `<axis>` re-signed, each carrying its `<limit>` with it negated and swapped, so the home-pose world axes are the specified directions (see below). And `joint7`'s origin is right-multiplied by `Rz(180°)`, which leaves its axis *line* — and therefore the kinematics — untouched while keeping the terminal `gripper` block in the frame it was derived in. That block is not export data: it is the canonical `linear_4310` plus the split wrist, and it carries over from the previous model unchanged.

### The gripper mount

Every YAM's last link ends in a tapered spigot — a ø38.08 mm boss that flares to the ø57 mm body over the 10.5 mm immediately proximal of the gripper's mounting face — which seats in the `linear_4310`'s matching bore and locates the gripper. This arm's as-built export fused wrist and gripper into a single solid, erasing that interface, and the split that recovered the arm half cut at the mounting face, handing the whole 10.5 mm band to the gripper side. The shipped wrist therefore butted flat on the gripper's collar with no locating feature at all.

`assets/wrist.stl` now carries that geometry, taken from `arm/yam/v1/assets/link5.stl` and cut at the mounting face. The gripper-root frame is the `linear_4310`'s own and is shared by both arms, so it is the same geometry at the same coordinates: nothing is scaled or invented, and the gripper's seat depth and the tool centre do not move, because the boss only fills a bore that was empty. `test_wrist_carries_the_same_mount_spigot_as_a_yam` measures the seating face, the engagement depth and the spigot diameter on both arms and requires them to agree.

### Gravity compensation

`gravity_comp_factor` in [`config/yam_7dof_right_v1.yml`](../../../../robots/config/yam_7dof_right_v1.yml) is **`[1.10, 1.08, 0.59, 0.56, 0.48, 0.97, 1.23]`, shared with the left arm** (2026-10-13), as are `grav_comp_kd` and `coulomb_friction`. Before that, the first tune against this model (2026-09-22) replaced the values found on the bench against the first export, whose links 2–5 were 0.536 kg lighter in total: the arm's mass is 5.843760 kg where that model said 5.307950 kg, and the home-pose moment about `joint2` is 11.481 N·m where it was 9.572 N·m, so the old factors over-compensated.

Over a 4000-pose sweep with the `linear_4310` fitted, the model's gravity torque changed between the two exports as:

| Joint | j1 | j2 | j3 | j4 | j5 | j6 | j7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| New / old model torque | ×1.141 | ×1.144 | ×1.080 | ×1.080 | ×1.00 | ×1.00 | ×1.00 |
| Old factor | 1.20 | 1.19 | 0.75 | 0.64 | 1.00 | 0.80 | 1.80 |
| Rescaled to preserve the delivered torque | 1.051 | 1.040 | 0.695 | 0.593 | 1.00 | 0.80 | 1.80 |
| Bench-tuned against this model | 1.20 | 1.00 | 0.60 | 0.50 | 1.00 | 0.50 | 1.20 |

Joints 2–4 land near what the mass change alone predicts. Joints 5–7 are unaffected by that change — their gravity load is almost entirely the end effector, which did not change between exports — so the drops at `joint6` and `joint7`, and `joint1` holding at 1.20, are bench findings the model does not account for.

### Motor torque registers

The motor turns an MIT torque command into current through `KT_Value * GREF`, so those two Flash registers scale every torque the arm applies, and the gravity factor and friction above hold only for the values below. Both arms declare them in their config's `motor_registers` block, for joints 1–7 only:

| Joint | CAN id | Motor | `Gr` | `GREF` | `KT_Value` (N·m/A) |
| --- | ---: | --- | ---: | ---: | ---: |
| `joint1` | `0x01` | `DM6248` | 48 | 1.0 | 0.0720294 |
| `joint2` | `0x02` | `DM6248` | 48 | 1.0 | 0.0720294 |
| `joint3` | `0x03` | `DM4340` | 40 | 1.0 | 0.044631 |
| `joint4` | `0x04` | `DM4340` | 40 | 1.0 | 0.044631 |
| `joint5` | `0x05` | `DM4340` | 40 | 1.0 | 0.044631 |
| `joint6` | `0x06` | `DM4310` | 10 | 1.0 | 0.083200 |
| `joint7` | `0x07` | `DM4310` | 10 | 1.0 | 0.083200 |

The gripper on `0x08` is not declared and not checked. At startup `get_yam_robot` passes the block to the motor config check (`verify_motor_config` in [`motor_drivers/motor_check.py`](../../../../motor_drivers/motor_check.py)), which reads both registers on each of the seven motors and **refuses to start** if any differs, logging the command that fixes it. It never writes them. To set them all on one arm, with the robot stopped, run [`scripts/set_7dof_motor_registers.py`](../../../../../scripts/set_7dof_motor_registers.py): it checks the motor types first, writes and saves only the registers that differ, then runs the same startup check read-only. Or set one by hand:

```bash
python scripts/set_7dof_motor_registers.py --arm yam_7dof_right --channel can0 --dry-run  # then without --dry-run
python i2rt/motor_config_tool/dm_motor_registers.py write KT_Value --value 0.0720294 --motor-id 1 --channel can0
python i2rt/motor_config_tool/dm_motor_registers.py save KT_Value --motor-id 1 --channel can0
```

### The left arm

[`../../yam_7dof_left/v1/`](../../yam_7dof_left/v1/) holds this model reflected about the world x-z plane, generated by `mirror_arm_model.py`. It is not a separate hardware revision — the two sides are the same arm, and every reflection-invariant quantity below (masses, principal inertia moments, link lengths, joint travel magnitudes) is shared. [`../../yam_7dof_left/v1/README.md`](../../yam_7dof_left/v1/README.md) tabulates what differs.

The two sides are **true mirror images**, so they share a joint-coordinate convention: an identical joint vector puts them in mirror-symmetric poses, and their joint ranges are identical; `directions` differs on `joint3`, `joint5` and `joint6`, whose motors on the left do not reverse with their axes. What differs is the world direction each joint turns about — the same for the joints whose axis lies along the mirror normal (`±y`), opposite for those lying in the mirror plane. See the left README.

### Joint sign convention

Each joint's axis is set so that, at the home pose, it turns about a **specified world direction**:

| Joint | j1 | j2 | j3 | j4 | j5 | j6 | j7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| World axis at home | `-y` | `-x` | `-z` | `-y` | `-z` | `-x` | `-y` |

That is the requirement; the local `<axis>` values in the table further down are what satisfies it, and they are not uniform (`joint1`, `joint3`, `joint5` and `joint6` sit on `-Z` of their own frame, the rest on `+Z`). The export writes every joint on `+Z`, which after the heading gives `+y -x +z -y +z +x -y`; those four are the ones re-signed, each with its `range` negated and swapped so the physical travel does not move. The left arm's directions are the reflection of these — see [`../../yam_7dof_left/v1/README.md`](../../yam_7dof_left/v1/README.md).

A specified world direction is not generally the positive rotation of the motor behind it, so `directions` in [`config/yam_7dof_right_v1.yml`](../../../../robots/config/yam_7dof_right_v1.yml) carries a mixed sign, `[-1, +1, +1, +1, +1, +1, +1]`, where every YAM arm is a uniform `+1`. **That is the correct place for it.** The driver applies `directions` symmetrically on read and write, so the arm's physical behaviour is untouched — for a given motor position the torque sent to each motor is bit-identical to what the model without it would produce — while the model's joint limits and `SimRobot`, which both read the model and neither of which sees `directions`, stay correct in model coordinates.

The distinction matters because a sign error of the *other* kind — the model disagreeing with the hardware — has to be fixed in the model instead. Recording that in `directions` would fix gravity compensation and nothing else: the driver's polarity flip does not reach the model, so the joint **limits** would stay mirrored and sim would stay mirrored against hardware, because `SimRobot` has no notion of motor polarity. Re-signing an axis is otherwise a pure re-parameterization — link frames, masses, inertias and home-pose FK are bit-identical — provided the joint's `range` is carried with it, negated and swapped.

A sign error of either kind is invisible at the home pose, where it cannot show, and visible everywhere else.

## Summary

| Property | Value |
| --- | ---: |
| Arm DOF | 7 |
| Gripper DOF | 1 (two coupled fingers) |
| Arm bodies, `base` through `gripper` | 8 (the last is the mount frame) |
| Arm mass, `base` through `link6` | 5.843760 kg |
| Total mass, composed with `linear_4310` | 6.538987 kg |
| Sum of parent-to-child translation norms | 0.8217221650 m |

## Link mass and inertial frames

The COM position and inertial-frame RPY are relative to the corresponding link frame.

| Link | Mass (kg) | COM xyz (m) | Inertial-frame RPY (rad) |
| --- | ---: | --- | --- |
| `base` | 1.28789 | `1.95071e-05 -0.0497882 -3.13786e-05` | `0 π/2 -π/2` |
| `link1` | 1.0034 | `0 -0.00383646 -0.0549694` | `0 0 0` |
| `link2` | 0.95628 | `0.106908 8.70615e-07 0.0475462` | `0 0 0` |
| `link3` | 0.83983 | `-0.0011414 -2.4009e-05 -0.0698293` | `0 0 0` |
| `link4` | 0.66056 | `0.0654359 9.05861e-07 0.0354242` | `0 0 0` |
| `link5` | 0.6884 | `-0.00468947 -0.000452694 -0.0656789` | `0 0 0` |
| `link6` | 0.4074 | `-6.47372e-05 0.000188696 -0.038139` | `0 0 0` |

## Link inertia

Each tensor is about the link COM and expressed in the inertial frame listed above:

```text
I = [[Ixx, Ixy, Ixz],
     [Ixy, Iyy, Iyz],
     [Ixz, Iyz, Izz]]
```

| Link | Ixx | Iyy | Izz | Ixy | Ixz | Iyz |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `base` | 0.00160702 | 0.00153482 | 0.00129912 | 6.64e-10 | 1.53831e-06 | -9.41377e-07 |
| `link1` | 0.00109454 | 0.000939051 | 0.000841783 | 0 | 0 | -3.052e-05 |
| `link2` | 0.00160881 | 0.00577331 | 0.0052621 | -1.0518e-08 | -0.000241694 | 1.6171e-08 |
| `link3` | 0.00095532 | 0.00109338 | 0.00045315 | 6.858e-08 | -2.41461e-05 | 1.08025e-06 |
| `link4` | 0.00105591 | 0.00199688 | 0.00160454 | -1.368e-08 | -2.38897e-06 | -6.15e-09 |
| `link5` | 0.001363 | 0.00128827 | 0.000864955 | 4.02887e-06 | -6.7093e-05 | -1.19113e-05 |
| `link6` | 0.000183696 | 0.00017194 | 0.000168067 | 1.99675e-06 | -1.07788e-06 | 1.27128e-06 |

## Joint frames, axes, ranges, and link lengths

The origin is the joint frame relative to its parent link. At zero joint displacement, the joint frame coincides with the child link frame. The ranges are this export's CAD travel.

| Joint | Parent → child | Origin xyz (m) | Origin RPY (rad) | Axis | Range (rad) | Range (deg) | Link length (m) |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| `joint1` | `base → link1` | `0.000116187 -0.0945 0` | `0 π/2 π/2` | `0 0 -1` | `[-2.79253, 2.79253]` | `[-160.0, 160.0]` | 0.0945000714 |
| `joint2` | `link1 → link2` | `0 -0.0501 -0.062` | `-π/2 0 0` | `0 0 1` | `[-0.436332, 3.57792]` | `[-25.0, 205.0]` | 0.0797120443 |
| `joint3` | `link2 → link3` | `0.219 0 0.0501` | `π/2 0 -π/2` | `0 0 -1` | `[-1.832596, 1.832596]` | `[-105.0, 105.0]` | 0.2246575394 |
| `joint4` | `link3 → link4` | `0.03545 0 -0.0894` | `-π π/2 0` | `0 0 1` | `[-2.0944, 2.0944]` | `[-120.0, 120.0]` | 0.0961720464 |
| `joint5` | `link4 → link5` | `0.1235 0 0.03545` | `π π/2 0` | `0 0 -1` | `[-1.8326, 1.8326]` | `[-105.0, 105.0]` | 0.1284871686 |
| `joint6` | `link5 → link6` | `0 -0.0381 -0.1475` | `π/2 π/2 0` | `0 0 -1` | `[-0.959931, 0.785398]` | `[-55.0, 45.0]` | 0.1523412616 |
| `joint7` | `link6 → gripper` | `4.47332e-06 0.02565 -0.0380064` | `-π/2 0 -π` | `0 0 1` | `[-1.5708, 1.5708]` | `[-90.0, 90.0]` | 0.0458520333 |

`joint3`'s limits are **bench-measured, not CAD**: the export wrote `[-15, 195]`, but driving the joint to both hard stops puts them at `[-105, 105]` (measured `[-107.37, +105.36]`, repeatable to 0.02°, less ~1.35° of press into each stop). The travel CAD gives — 210° — is right; where CAD put zero was not, and the hardware's zero sits at mid-travel. `joint6` is narrower than the first export said and comes straight from CAD.

## Global link and joint frames at home

The base frame is the world/TF root. Each `jointN` frame coincides with its corresponding child link frame at home. The final column gives each joint's rotation axis expressed in the world frame at home; `base` is fixed and has no joint.

| Link / coincident joint frame | Global xyz (m) | Global RPY (rad) | Joint axis, world frame |
| --- | --- | --- | --- |
| `base` | `0 0 0` | `0 0 0` | — |
| `link1 / joint1` | `0.000116187 -0.0945 0` | `0 π/2 π/2` | `0 -1 0` |
| `link2 / joint2` | `0.0502163515221 -0.156499867055 0` | `-π/2 π/2 π/2` | `-1 0 0` |
| `link3 / joint3` | `0.000116351523103 -0.156500184027 -0.219` | `0 0 π/2` | `0 0 -1` |
| `link4 / joint4` | `0.000115798851776 -0.121049855645 -0.308399869784` | `-π/2 π/2 -π` | `0 -1 0` |
| `link5 / joint5` | `0.000115475426379 -0.156499855647 -0.431899869783` | `0 0 π/2` | `0 0 -1` |
| `link6 / joint6` | `0.0382153250309 -0.156499173901 -0.579399908629` | `-π π/2 -π` | `-1 0 0` |
| `gripper / joint7` | `0.000208925026336 -0.130849173901 -0.579404343197` | `π/2 -π/2 0` | `0 -1 0` |

Several home-frame orientations sit very close to the RPY pitch singularity at `±π/2`, where roll and yaw are not separately determined. Equivalent roll/yaw pairs may therefore look different while representing the same rotation; use rotation matrices or the model quaternions for numerical comparisons.

## Classical Denavit–Hartenberg parameters

Standard ("distal") DH convention: `Zᵢ₋₁` lies on joint axis `i`, and `Xᵢ` is the common normal between `Zᵢ₋₁` and `Zᵢ`. The link transform is

```text
T(i-1, i) = Rot_z(θᵢ) · Trans_z(dᵢ) · Trans_x(aᵢ) · Rot_x(αᵢ)
```

`θᵢ` is the revolute joint variable; the table lists its **home offset** (value at zero displacement), so `θᵢ = offset + qᵢ` with no per-joint sign — the table is derived against **this model's** axis directions, not the export's. Frame `{0}` has `Z₀` on joint axis 1 and `X₀` aligned with `X₁`, so `θ₁`'s home offset is zero; `{0}` is not the `base` frame itself, and the fixed transform between them is given below, as is the one from frame `{7}` to the `gripper` frame. Frame `{7}` has no joint 8 to point `Z₇` at, so it is taken coincident with `{6}` and the last row is a pure `Rot_z(θ₇)`. The full chain is `T(base, gripper) = T(base,{0}) · Πᵢ T(i-1,i) · T({7},gripper)`. Every `aᵢ` is zero: consecutive joint axes on this arm all intersect. Verified against direct URDF forward kinematics at 300 random joint configurations, agreeing to `1e-15`.

| i | θᵢ home offset (rad) | dᵢ (m) | aᵢ (m) | αᵢ (rad) |
| --- | ---: | ---: | ---: | ---: |
| 1 | `0` | `0` | `0` | `π/2` |
| 2 | `π/2` | `0` | `0` | `π/2` |
| 3 | `π/2` | `0.3084` | `0` | `π/2` |
| 4 | `0` | `0` | `0` | `-π/2` |
| 5 | `π/2` | `0.271` | `0` | `π/2` |
| 6 | `π/2` | `0` | `0` | `-π/2` |
| 7 | `0` | `0` | `0` | `0` |

`T(base, {0})`:

```text
[ -0.000000  +1.000000  +0.000003  +0.000116 ]
[ +0.000000  +0.000003  -1.000000  -0.156500 ]
[ -1.000000  -0.000000  -0.000000  -0.000000 ]
[ +0.000000  +0.000000  +0.000000  +1.000000 ]
```

`T({7}, gripper)`:

```text
[ -1.000000  +0.000000  -0.000000  +0.000000 ]
[ -0.000000  -1.000000  +0.000000  +0.000000 ]
[ +0.000000  +0.000000  +1.000000  -0.025650 ]
[ +0.000000  +0.000000  +0.000000  +1.000000 ]
```

## Product-of-exponentials screw axes

Space-frame (base-frame) screw axes for the Modern Robotics PoE formulation

```text
T(θ) = e^([S₁]θ₁) · e^([S₂]θ₂) ··· e^([S₇]θ₇) · M
```

where each `Sᵢ = (ωᵢ, vᵢ)` with `ωᵢ` the unit joint axis and `vᵢ = -ωᵢ × qᵢ` (`qᵢ` a point on the axis), all at the home configuration. The end-effector frame is the `gripper` frame; the space frame is the `base` frame.

| Joint | ωᵢ (unit) | vᵢ (m) |
| --- | --- | --- |
| `joint1` | `0 -1 0` | `0 0 -0.000115936` |
| `joint2` | `-1 0 0` | `0 0 -0.1565` |
| `joint3` | `0 0 -1` | `0.156501 0.000117156 0` |
| `joint4` | `0 -1 0` | `-0.3084 0 -0.000115354` |
| `joint5` | `0 0 -1` | `0.156501 0.000115916 0` |
| `joint6` | `-1 0 0` | `0 0.5794 -0.156499` |
| `joint7` | `0 -1 0` | `-0.579404 0 -0.000209406` |

Home configuration `M` (end-effector pose at zero joints), as a 4×4 homogeneous transform in the base frame:

```text
[ +0.000001  -1.000000  -0.000004  +0.000209 ]
[ +0.000000  +0.000004  -1.000000  -0.130849 ]
[ +1.000000  +0.000001  +0.000000  -0.579404 ]
[ +0.000000  +0.000000  +0.000000  +1.000000 ]
```

The same joint axes expressed in the **body** (end-effector) frame give the body-form PoE, with each `Bᵢ` the space axis `Sᵢ` transformed by the adjoint of `M⁻¹`:

```text
T(θ) = M · e^([B₁]θ₁) · e^([B₂]θ₂) ··· e^([B₇]θ₇)
Bᵢ   = [Ad_{M⁻¹}] Sᵢ
```

| Joint | ωᵢ (unit) | vᵢ (m) |
| --- | --- | --- |
| `joint1` | `0 0 1` | `9.32323e-05 -0.579404 0` |
| `joint2` | `0 1 0` | `-0.025651 0 0.579404` |
| `joint3` | `-1 0 0` | `0 -0.0256497 9.38031e-05` |
| `joint4` | `0 0 1` | `9.33665e-05 -0.271004 0` |
| `joint5` | `-1 0 0` | `0 -0.0256501 9.35058e-05` |
| `joint6` | `0 1 0` | `-0.02565 0 0` |
| `joint7` | `0 0 1` | `0 0 0` |
