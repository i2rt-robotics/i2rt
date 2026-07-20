# YAM Ultra Arm Physical Properties

This document records the physical and kinematic properties of the six-DOF YAM Ultra arm from [`yam_ultra.urdf`](yam_ultra.urdf). The corresponding arm-only MuJoCo model is [`yam_ultra.xml`](yam_ultra.xml).

## Scope and conventions

- The physical arm consists of `base` and `link1` through `link5`, connected by `joint1` through `joint6`.
- The URDF names the child of `joint6` `gripper`. The arm-only MJCF represents that same attachment frame as `link6`; it is not a physical arm link.
- The URDF names these joints `dof_joint1` through `dof_joint6`; this document uses the normalized `joint1` through `joint6` names from the MJCF [`yam_ultra.xml`](yam_ultra.xml).
- Gripper, finger, tip, tool, and top bodies are excluded.
- SI units are used: metres, kilograms, radians, and kg·m².
- Home pose means all six arm joint coordinates are zero.
- URDF RPY uses `R = Rz(yaw) Ry(pitch) Rx(roll)`.
- Joint axes are expressed in their local joint frames.
- “Link length” is the Euclidean norm of the parent-to-child joint-origin translation. The full translation vector remains the authoritative kinematic value.
- Frame orientations (RPY) and joint ranges within floating-point tolerance of `0` or a multiple of π are written as `0` or that multiple (e.g. `π/2`, `-2π/3`); any near-zero numeric component with magnitude below `1e-4` — COM offsets, off-diagonal inertia terms, translation, axis, and global-frame coordinates — is likewise shown as `0` (diagonal inertia moments always keep their value). Degree joint ranges are rounded to the nearest integer. All other values keep their full precision.

## Summary

| Property | Value |
| --- | ---: |
| Arm DOF | 6 |
| Physical arm bodies | 6 |
| Physical arm mass, excluding gripper and tips | 4.520578 kg |
| Sum of parent-to-child translation norms | 0.8128308576 m |

## Link mass and inertial frames

The COM position and inertial-frame RPY are relative to the corresponding link frame.

| Link | Mass (kg) | COM xyz (m) | Inertial-frame RPY (rad) |
| --- | ---: | --- | --- |
| `base` | 0.940115 | `0 0 0.0274792` | `0 π/2 π` |
| `link1` | 0.101218 | `-0.0233636 -0.00172503 -0.00727828` | `0 0 0` |
| `link2` | 1.63122 | `0.000134495 -0.0321253 0.129165` | `0 0 0` |
| `link3` | 0.982553 | `-0.0556042 0.0357986 -0.135083` | `0 0 0` |
| `link4` | 0.462165 | `-0.0548959 -0.0350242 -0.0575534` | `0 0 0` |
| `link5` | 0.403307 | `0.0375847 0 -0.00717397` | `0 0 0` |
| `link6` mount | — | — | — |

The `link6` MJCF placeholder uses mass `1e-6 kg` and diagonal inertia `1e-9 1e-9 1e-9 kg·m²` only to satisfy MuJoCo. These are simulation artifacts and are not included in the physical arm mass.

## Link inertia

Each tensor is about the link COM and expressed in the inertial frame listed above:

```text
I = [[Ixx, Ixy, Ixz],
     [Ixy, Iyy, Iyz],
     [Ixz, Iyz, Izz]]
```

| Link | Ixx | Iyy | Izz | Ixy | Ixz | Iyz |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `base` | 0.0016196 | 0.000815619 | 0.00184612 | 0 | 0 | 0 |
| `link1` | 0.000114689 | 4.76603e-05 | 0.000125996 | 0 | 0 | 0 |
| `link2` | 0.0162876 | 0.0162103 | 0.000888393 | 0 | 0 | 0 |
| `link3` | 0.00758945 | 0.00763097 | 0.000937606 | 0 | -0.00064069 | 0 |
| `link4` | 0.000473594 | 0.000764022 | 0.000604492 | 0 | -0.000238801 | 0 |
| `link5` | 0.000184284 | 0.000219791 | 0.000181053 | 0 | 0 | 0 |

## Joint frames, axes, ranges, and link lengths

The origin is the joint frame relative to its parent link. At zero joint displacement, the joint frame coincides with the child link frame.

| Joint | Parent → child | Origin xyz (m) | Origin RPY (rad) | Axis | Range (rad) | Range (deg) | Link length (m) |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| `joint1` | `base → link1` | `0 0 0.0733` | `0 π/2 π` | `-1 0 0` | `[-5π/6, π]` | `[-150, 180]` | 0.0733000000 |
| `joint2` | `link1 → link2` | `-0.0455 0.03285 -0.02` | `0 0 0` | `0 -1 0` | `[0, 7π/6]` | `[0, 210]` | 0.0595766103 |
| `joint3` | `link2 → link3` | `0 -0.0678 0.264` | `0 0 0` | `0 1 0` | `[0, π]` | `[0, 180]` | 0.2725671293 |
| `joint4` | `link3 → link4` | `-0.0600003 0.0678 -0.244999` | `0 0 0` | `0 1 0` | `[-1.69297, π/2]` | `[-97, 90]` | 0.2611922395 |
| `joint5` | `link4 → link5` | `-0.0403003 -0.0338507 -0.0703851` | `0 0 0` | `1 0 0` | `[-π/2, π/2]` | `[-90, 90]` | 0.0878865540 |
| `joint6` | `link5 → link6 mount` | `0.0404996 0 -0.0419481` | `0 0 0` | `0 0 -1` | `[-2π/3, 2π/3]` | `[-120, 120]` | 0.0583083244 |

## Global link and joint frames at home

The base frame is the world/TF root. Each `jointN` frame coincides with its corresponding `linkN` frame at home. The global RPY values below use the canonical extraction with pitch in `[-π/2, π/2]`. The final column gives each joint's rotation axis expressed in the world frame at home — the local `Axis` from the previous table carried through that frame's global orientation; `base` is fixed and has no joint.

| Link / coincident joint frame | Global xyz (m) | Global RPY (rad) | Joint axis, world frame |
| --- | --- | --- | --- |
| `base` | `0 0 0` | `0 0 0` | — |
| `link1 / joint1` | `0 0 0.0733` | `0 π/2 π` | `0 0 1` |
| `link2 / joint2` | `0.0200000000002 -0.03285 0.1188` | `0 π/2 π` | `0 1 0` |
| `link3 / joint3` | `-0.244 0.0349499999999 0.11879959157` | `0 π/2 π` | `0 -1 0` |
| `link4 / joint4` | `0.000999000000524 -0.03285 0.178799891569` | `0 π/2 π` | `0 -1 0` |
| `link5 / joint5` | `0.0713841000007 0.00100070000002 0.219100191569` | `0 π/2 π` | `0 0 -1` |
| `link6 mount / joint6` | `0.113332200001 0.00100046014203 0.178600591568` | `0 π/2 π` | `1 0 0` |

Several home-frame orientations are very close to the RPY pitch singularity at `±π/2`. Equivalent roll/yaw pairs may therefore look different while representing the same rotation; use rotation matrices or the model quaternions for numerical comparisons.

## Classical Denavit–Hartenberg parameters

Standard ("distal") DH convention: `Zᵢ₋₁` lies on joint axis `i`, and `Xᵢ` is the
common normal between `Zᵢ₋₁` and `Zᵢ`. The link transform is

```text
T(i-1, i) = Rot_z(θᵢ) · Trans_z(dᵢ) · Trans_x(aᵢ) · Rot_x(αᵢ)
```

`θᵢ` is the revolute joint variable; the table lists its **home offset** (value at
zero displacement — add the joint coordinate to it). Frame `{0}` is the `base`
frame (`Z₀` on joint axis 1) and frame `{6}` is the `link6` mount frame (the
end-effector frame `M` used below). Joints 2–4 have parallel axes, so the
`aᵢ`/`dᵢ` spanning them are not unique; origins there are placed by perpendicular
projection from the preceding frame, giving the values shown.

| i | θᵢ home offset (rad) | dᵢ (m) | aᵢ (m) | αᵢ (rad) |
| --- | ---: | ---: | ---: | ---: |
| 1 | `0` | `0.1188` | `0.02` | `-π/2` |
| 2 | `π` | `0` | `0.264` | `π` |
| 3 | `-2.901421` | `0` | `0.252239` | `0` |
| 4 | `-0.240173` | `-0.001001` | `0.070385` | `π/2` |
| 5 | `π/2` | `0.000199` | `0` | `π/2` |
| 6 | `π/2` | `0.041948` | `0` | `-π` |

## Product-of-exponentials screw axes

Space-frame (base-frame) screw axes for the Modern Robotics PoE formulation

```text
T(θ) = e^([S₁]θ₁) · e^([S₂]θ₂) ··· e^([S₆]θ₆) · M
```

where each `Sᵢ = (ωᵢ, vᵢ)` with `ωᵢ` the unit joint axis and `vᵢ = -ωᵢ × qᵢ`
(`qᵢ` a point on the axis), all at the home configuration. The end-effector frame
is the `link6` mount frame; the space frame is the `base` frame.

| Joint | ωᵢ (unit) | vᵢ (m) |
| --- | --- | --- |
| `joint1` | `0 0 1` | `0 0 0` |
| `joint2` | `0 1 0` | `-0.1188 0 0.02` |
| `joint3` | `0 -1 0` | `0.1188 0 0.244` |
| `joint4` | `0 -1 0` | `0.1788 0 -0.000999` |
| `joint5` | `0 0 -1` | `-0.001001 0.071384 0` |
| `joint6` | `1 0 0` | `0 0.178601 -0.001` |

Home configuration `M` (end-effector pose at zero joints), as a 4×4 homogeneous
transform in the base frame:

```text
[ +0.000000  +0.000000  -1.000000  +0.113332 ]
[ +0.000000  -1.000000  +0.000000  +0.001000 ]
[ -1.000000  +0.000000  +0.000000  +0.178601 ]
[ +0.000000  +0.000000  +0.000000  +1.000000 ]
```

The same joint axes expressed in the **body** (end-effector) frame give the
body-form PoE, with each `Bᵢ` the space axis `Sᵢ` transformed by the adjoint of `M⁻¹`:

```text
T(θ) = M · e^([B₁]θ₁) · e^([B₂]θ₂) ··· e^([B₆]θ₆)
Bᵢ   = [Ad_{M⁻¹}] Sᵢ
```

| Joint | ωᵢ (unit) | vᵢ (m) |
| --- | --- | --- |
| `joint1` | `-1 0 0` | `0 -0.113332 0.001` |
| `joint2` | `0 -1 0` | `0.093332 0 -0.059801` |
| `joint3` | `0 1 0` | `-0.357332 0 0.059801` |
| `joint4` | `0 1 0` | `-0.112333 0 -0.000199` |
| `joint5` | `1 0 0` | `0 0.041948 0` |
| `joint6` | `0 0 -1` | `0 0 0` |
